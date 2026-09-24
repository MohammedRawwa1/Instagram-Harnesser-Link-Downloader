"""Instagram → Groq → lip-synced SRT/TXT backend.

A pure async backend (no UI): POST an Instagram URL, get a job id back
immediately, and pull perfectly-synced SRT / TXT / timings when it's done
(or have the backend POST them to your callback URL).

Endpoints:
    POST /transcribe                 submit {url, job_id?, lang?, callback_url?} → 202
    GET  /jobs/{job_id}              job status + progress + artifact paths
    GET  /jobs/{job_id}/srt          download the subtitle file
    GET  /jobs/{job_id}/txt          download the transcript
    GET  /jobs/{job_id}/timings      download per-word/per-cue timings JSON
    GET  /health
    POST /shutdown                   graceful shutdown: drain running jobs, then stop
"""

import asyncio
import json
import logging
import re
import secrets
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import Settings
from .jobs import Job, JobRegistry, STATUS_QUEUED, STATUS_DONE, STATUS_FAILED
from .worker import run_job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("insta.api")

settings = Settings()
settings.ensure_dirs()
registry = JobRegistry(settings.jobs_dir())


async def _load_persisted_jobs() -> None:
    """Load persisted jobs before accepting requests after a restart."""
    for job_file in settings.jobs_dir().glob("*.json"):
        try:
            data = json.loads(job_file.read_text(encoding="utf-8"))
            await registry.create(Job.from_dict(data))
        except (OSError, json.JSONDecodeError, TypeError):
            log.warning("could not load persisted job %s", job_file, exc_info=True)

app = FastAPI(
    title="Instagram Groq Lip-Sync Backend",
    description="Async backend: Instagram URL → word-level transcription → perfectly synced SRT/TXT/timings.",
    version="1.0.0",
)


@app.post("/shutdown")
async def shutdown(x_api_key: Optional[str] = Header(default=None)) -> dict:
    """Trigger a graceful server shutdown.

    Drains in-flight transcription jobs first, then stops the server.
    Optional shared secret if the server is configured with an API key.
    """
    _require_api_key(x_api_key)
    _shutdown_event.set()
    return {"status": "shutdown_requested"}

# Track in-flight job tasks so the server can drain them on shutdown instead
# of leaving them hanging as orphaned background tasks.
_running_tasks: set[asyncio.Task] = set()
_shutdown_event = asyncio.Event()


@app.on_event("startup")
async def _on_startup() -> None:
    await _load_persisted_jobs()
    log.info("backend started; max_workers=%d", settings.max_workers)


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    log.info("shutdown requested; draining %d in-flight jobs...", len(_running_tasks))
    _shutdown_event.set()
    if _running_tasks:
        # Give jobs a short window to finish cleanly; then we just let the
        # process exit (uvicorn will cancel remaining tasks).
        try:
            await asyncio.wait_for(
                asyncio.gather(*_running_tasks, return_exceptions=True),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            log.warning("drain timed out; %d jobs still running at exit", len(_running_tasks))
    log.info("shutdown complete")


def _run_job_safely(job: Job, registry: JobRegistry, settings: Settings) -> asyncio.Task:
    """Schedule a job and track it so shutdown can drain it."""

    async def _wrap() -> None:
        try:
            await run_job(job, registry, settings)
        except asyncio.CancelledError:
            log.info("[%s] job cancelled during shutdown", job.job_id)
            await registry.update(job.job_id, status="cancelled", finished_at=time.time())
            raise
        except Exception:
            log.exception("[%s] job crashed", job.job_id)
            # worker.py already writes the failed status, but if the crash
            # happened before that, make sure the job is marked failed.
            current = await registry.get(job.job_id)
            if current and current.status == STATUS_QUEUED:
                await registry.update(
                    job.job_id, status="failed",
                    error="job crashed before starting", finished_at=time.time(),
                )

    task = asyncio.create_task(_wrap(), name=f"job-{job.job_id}")
    _running_tasks.add(task)
    task.add_done_callback(lambda t: _running_tasks.discard(t))
    return task

# Only public Instagram post/reel/tv/story links. Strict charset keeps the value
# safe to hand to yt-dlp and to use in filesystem paths. Real Instagram URLs
# usually end with a trailing slash and often carry a query string.
INSTAGRAM_URL_RE = re.compile(
    r"^https://(www\.)?instagram\.com/(reel|reels|p|tv|stories)/[A-Za-z0-9_-]+/?(\?.*)?$"
)
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LANG_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")


class TranscribeRequest(BaseModel):
    url: str = Field(..., description="Public Instagram post/reel/tv/story URL")
    job_id: Optional[str] = Field(None, description="Optional caller-chosen id (A-Za-z0-9_- only)")
    lang: Optional[str] = Field(None, description="Optional ISO-639-1 language, e.g. 'en'")
    callback_url: Optional[str] = Field(None, description="Optional https URL to POST results to")
    translate: bool = Field(False, description="Translate any language to English using Groq translations endpoint")


def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


def _validate_callback(url: str) -> Optional[str]:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or not parsed.hostname:
        raise HTTPException(status_code=422, detail="callback_url must be a plain https:// URL without credentials")
    return url


def _validate_request(payload: TranscribeRequest) -> None:
    if not INSTAGRAM_URL_RE.match(payload.url):
        raise HTTPException(
            status_code=422,
            detail="url must be a public https://instagram.com reel/post/tv/story link",
        )
    if payload.job_id and not JOB_ID_RE.match(payload.job_id):
        raise HTTPException(status_code=422, detail="job_id may only contain A-Z a-z 0-9 dash underscore (max 64)")
    if payload.lang and not LANG_RE.match(payload.lang):
        raise HTTPException(status_code=422, detail="lang must look like 'en' or 'pt-BR'")


@app.post("/transcribe", status_code=202)
async def create_transcription(payload: TranscribeRequest, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    _validate_request(payload)

    # Stop as soon as this URL already has any job, terminal or not.
    terminal = {STATUS_DONE, STATUS_FAILED}
    existing = await registry.find_by_url(payload.url)
    if existing:
        log.info("[%s] url already exists %s -> skip", payload.job_id or "auto", payload.url)
        return {
            "job_id": existing.job_id,
            "status": existing.status,
            "status_url": f"/jobs/{existing.job_id}",
            "srt_url": f"/jobs/{existing.job_id}/srt",
            "txt_url": f"/jobs/{existing.job_id}/txt",
            "timings_url": f"/jobs/{existing.job_id}/timings",
        }

    job_id = (payload.job_id or f"job_{secrets.token_hex(8)}")[:64]

    # Idempotency gate: if caller supplied an explicit job_id and we already have
    # a job with that id for the same URL, return that job. This also stops the
    # server from creating new work when a client resubmits with the same
    # explicit job_id, even if the existing job is terminal.
    existing_by_id = await registry.get(job_id)
    if existing_by_id and existing_by_id.url == payload.url:
        log.info("[%s] same url same job_id -> skip", job_id)
        return {
            "job_id": job_id,
            "status": existing_by_id.status,
            "status_url": f"/jobs/{job_id}",
            "srt_url": f"/jobs/{job_id}/srt",
            "txt_url": f"/jobs/{job_id}/txt",
            "timings_url": f"/jobs/{job_id}/timings",
        }

    # If the URL already has a terminal result for this exact job_id, stop
    # without creating a new job.
    if existing and existing.job_id == job_id and existing.status in terminal:
        log.info("[%s] same url same job_id terminal -> skip", job_id)
        return {
            "job_id": job_id,
            "status": existing.status,
            "status_url": f"/jobs/{job_id}",
            "srt_url": f"/jobs/{job_id}/srt",
            "txt_url": f"/jobs/{job_id}/txt",
            "timings_url": f"/jobs/{job_id}/timings",
        }

    # Also stop when the URL already has a non-terminal job for this exact
    # job_id. We do not want to queue a second running job under the same id.
    if existing and existing.job_id == job_id and existing.status not in terminal:
        log.info("[%s] same url same job_id running -> skip", job_id)
        return {
            "job_id": job_id,
            "status": existing.status,
            "status_url": f"/jobs/{job_id}",
            "srt_url": f"/jobs/{job_id}/srt",
            "txt_url": f"/jobs/{job_id}/txt",
            "timings_url": f"/jobs/{job_id}/timings",
        }

    job = Job(
        job_id=job_id,
        url=payload.url,
        lang=payload.lang,
        translate=payload.translate,
        callback_url=_validate_callback(payload.callback_url) if payload.callback_url else None,
    )
    await registry.create(job)

    # Fire-and-forget: the HTTP response returns immediately (async), the
    # pipeline keeps running in the background.
    _run_job_safely(job, registry, settings)
    log.info("[%s] queued %s", job_id, payload.url)

    return {
        "job_id": job_id,
        "status": STATUS_QUEUED,
        "status_url": f"/jobs/{job_id}",
        "srt_url": f"/jobs/{job_id}/srt",
        "txt_url": f"/jobs/{job_id}/txt",
        "timings_url": f"/jobs/{job_id}/timings",
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    job = await registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


def _job_file(job: Job, attr: str, mime: str, label: str) -> FileResponse:
    path = getattr(job, attr)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"{label} not ready")
    return FileResponse(path, media_type=mime, filename=Path(path).name)


@app.get("/jobs/{job_id}/srt")
async def get_srt(job_id: str, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    job = await registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_file(job, "srt_path", "application/x-subrip", "SRT")


@app.get("/jobs/{job_id}/txt")
async def get_txt(job_id: str, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    job = await registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_file(job, "txt_path", "text/plain", "TXT")


@app.get("/jobs/{job_id}/timings")
async def get_timings(job_id: str, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    job = await registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_file(job, "timings_path", "application/json", "timings")


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.groq_model}