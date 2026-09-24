"""Fault-tolerant batch processing with an adaptive submission pool.

Submits each URL as its own job through a WorkerPool that shrinks on 429s /
high latency and ramps back up when the backend is healthy. Polls
independently, and never lets one failed reel kill the whole batch.
Tracks failures to `failed_urls.txt` so you can retry just the bad ones.

Resume semantics:
  - Re-reads `urls_file`.
  - Skips URLs whose job already finished (status == "done") AND whose output
    files exist on disk.
  - Re-submits everything else (including previously-failed ones).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Callable

from httpx import AsyncClient, Timeout

from .jobs import STATUS_DONE, STATUS_FAILED, STATUS_QUEUED, STATUS_DOWNLOADING, STATUS_SPLITTING, STATUS_TRANSCRIBING, STATUS_STITCHING
from .pool import WorkerPool

_BATCH_POLL_INTERVAL = 3.0
_MAX_UNKNOWN_POLLS = 20

# How long a submit can block waiting for a pool slot before we bail.
_SUBMIT_ACQUIRE_TIMEOUT = 60.0


def _output_exists(data: dict) -> bool:
    """True when the job is effectively completed AND its artifacts are on disk.

    "Completed" here means done or failed. We treat failed jobs as skippable too,
    because re-submitting them repeatedly on every batch restart makes the workflow
    noisy and wastes submission slots / Groq calls.
    """
    status = data.get("status")
    if status not in (STATUS_DONE, STATUS_FAILED):
        return False
    for key in ("srt_path", "txt_path", "timings_path"):
        p = data.get(key)
        if not p:
            return False
        if not Path(p).exists():
            return False
    return True


async def _submit(
    client: AsyncClient,
    pool: WorkerPool,
    url: str,
    api_key: str | None,
    lang: str | None,
    translate: bool,
) -> dict | None:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    _t0 = time.monotonic()
    try:
        async with pool.acquire(timeout=_SUBMIT_ACQUIRE_TIMEOUT) as _tok:
            resp = await client.post(
                "/transcribe",
                json={"url": url, "lang": lang, "translate": translate},
                headers=headers,
                timeout=Timeout(10.0),
            )
            _ttf_ms = (time.monotonic() - _t0) * 1000
            if resp.status_code == 429:
                pool.mark_rate_limited()
                pool.mark_slow(ttf_ms=_ttf_ms)
                return {"error": "backend rate-limited (429)"}
            if resp.status_code == 202:
                pool.mark_fast(ttf_ms=_ttf_ms)
                return resp.json()
            pool.mark_slow(ttf_ms=_ttf_ms)
            return {"error": f"submit returned {resp.status_code}"}
    except TimeoutError:
        pool.mark_slow(ttf_ms=(time.monotonic() - _t0) * 1000)
        return {"error": "submit timed out waiting for pool slot"}
    except Exception as e:
        pool.mark_slow(ttf_ms=(time.monotonic() - _t0) * 1000)
        return {"error": str(e)}


async def _poll(client: AsyncClient, job_id: str, api_key: str | None) -> dict:
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        resp = await client.get(f"/jobs/{job_id}", headers=headers, timeout=Timeout(10.0))
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"status": "unknown", "error": "poll failed"}


async def run_batch(
    urls: list[str],
    base_url: str,
    api_key: str | None,
    lang: str | None,
    translate: bool,
    urls_file: Path | None = None,
    seen_file: Path | None = None,
    failed_file: Path | None = None,
    *,
    resume: bool = True,
    submit_hook: Callable[[str, dict], None] | None = None,
    poll_hook: Callable[[str, dict], None] | None = None,
    max_submitters: int = 4,
) -> dict:
    """Run a batch of Instagram reel URLs through the backend.

    Returns a summary dict with keys: submitted, done, failed, skipped,
    failed_urls, elapsed.
    """
    project_root = Path(__file__).resolve().parent.parent
    if urls_file is None:
        urls_file = project_root / "scripts" / "urls.txt"
    if failed_file is None:
        failed_file = project_root / "data" / "failed_urls.txt"

    client = AsyncClient(base_url=base_url, timeout=Timeout(10.0))
    pool = WorkerPool(
        max_workers=max_submitters,
        min_workers=1,
        cooldown=30.0,
        ramp_interval=5.0,
        stable_window=60.0,
    )
    pool.start()
    started = time.time()

    submitted: list[tuple[str, str, str]] = []  # (url, job_id, status)
    done = 0
    failed = 0
    skipped = 0
    failed_urls: list[str] = []

    # Phase 1: submit (with optional resume skip)
    for url in urls:
        url = url.strip()
        if not url:
            continue

        # Resume: skip any URL that the backend already has a final job for.
        # We treat "done" and "failed" as final here so re-running a batch does not
        # keep re-submitting the same failed reels.
        if resume:
            existing = await _find_existing_job(client, api_key, url)
            if existing and _output_exists(existing):
                skipped += 1
                if submit_hook:
                    submit_hook(url, {"skipped": True, "reason": f"already processed ({existing.get('status')})"})
                continue

        result = await _submit(client, pool, url, api_key, lang, translate)
        if result and "job_id" in result:
            submitted.append((url, result["job_id"], result.get("status", "queued")))
            if submit_hook:
                submit_hook(url, result)
        else:
            failed += 1
            failed_urls.append(url)
            if submit_hook:
                submit_hook(url, {"error": result.get("error") if result else "submit failed"})

    # Phase 2: poll until all submitted jobs settle
    settled: int = 0
    n = len(submitted)
    unknown_polls = [0] * n
    while settled < n:
        settled = 0
        for i, (url, job_id, _status) in list(enumerate(submitted)):
            data = await _poll(client, job_id, api_key)
            status = data.get("status", "unknown")
            if status == "unknown":
                unknown_polls[i] += 1
                if unknown_polls[i] >= _MAX_UNKNOWN_POLLS:
                    status = STATUS_FAILED
                    data = {
                        "status": STATUS_FAILED,
                        "error": "job status unavailable after repeated polling attempts",
                    }
            else:
                unknown_polls[i] = 0
            submitted[i] = (url, job_id, status)
            if poll_hook:
                poll_hook(url, data)
            if data.get("status") in (STATUS_DONE, STATUS_FAILED):
                settled += 1
                if data.get("status") == STATUS_DONE:
                    done += 1
                else:
                    failed += 1
                    failed_urls.append(url)
        if settled < n:
            await asyncio.sleep(_BATCH_POLL_INTERVAL)

    elapsed = time.time() - started

    # Persist failed URLs
    if failed_urls:
        failed_file.parent.mkdir(parents=True, exist_ok=True)
        with failed_file.open("a", encoding="utf-8") as f:
            for u in failed_urls:
                f.write(u.strip() + "\n")

    await pool.close()
    await client.aclose()

    return {
        "submitted": n,
        "done": done,
        "failed": failed,
        "skipped": skipped,
        "failed_urls": failed_urls,
        "elapsed": round(elapsed, 1),
        "final_pool_limit": pool.limit,
    }


async def _find_existing_job(
    client: AsyncClient, api_key: str | None, url: str
) -> dict | None:
    """Best-effort: look for a job whose url matches `url`.

    The API doesn't expose a search endpoint, so we scan the jobs JSON files
    on disk (the job registry mirrors state there).
    """
    project_root = Path(__file__).resolve().parent.parent
    jobs_dir = project_root / "data" / "jobs"
    if not jobs_dir.exists():
        return None
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    for jf in jobs_dir.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if data.get("url") == url:
                # Also verify it's still there via the API
                jid = data.get("job_id", "")
                if jid:
                    resp = await client.get(f"/jobs/{jid}", headers=headers, timeout=Timeout(5.0))
                    if resp.status_code == 200:
                        return resp.json()
        except (json.JSONDecodeError, OSError):
            continue
    return None
