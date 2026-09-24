"""
Instagram Reel Transcriber - Windows Native
Run: python transcribe.py
"""

import json
import asyncio
import os
import re
import subprocess
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "data" / "output"
SERVER_LOG = SCRIPT_DIR / "data" / "server.log"
SERVER_PID = SCRIPT_DIR / "data" / "server.pid"
VENV_PYTHON = SCRIPT_DIR / ".venv" / "Scripts" / "python.exe"
URLS_FILE = SCRIPT_DIR / "scripts" / "urls.txt"
BASE_URL = "http://localhost:8000"


def _api_key():
    """Shared secret the backend expects in the X-API-Key header.
    Read from:
      1. API_KEY env var (preferred, keeps secrets out of .env files you share)
      2. A local .env-style file next to this script (API_KEY=...)
      3. Interactive prompt on first use (value cached to disk, redacted in logs)
    """
    import json

    val = os.getenv("API_KEY", "").strip()
    if val:
        return val

    env_path = SCRIPT_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("API_KEY=") and not line.startswith("API_KEY=#"):
                candidate = line.split("=", 1)[1].strip()
                if candidate:
                    return candidate

    cache_path = SCRIPT_DIR / "data" / "api_key.json"
    cached = ""
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8")).get("key", "")
        except (json.JSONDecodeError, OSError):
            cached = ""
    if cached:
        return cached

    while True:
        print("  API key not configured.")
        print(f"  Set API_KEY env var, or add API_KEY=... to {env_path}, or paste it now.")
        print(f"  (This is the same value you set in the server's .env / API_KEY.)")
        key = input("  API_KEY: ").strip()
        if not key:
            print("  Empty key — aborting.")
            raise SystemExit(1)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"key": key}, indent=2), encoding="utf-8")
        return key


def _redacted(key: str) -> str:
    if len(key) <= 4:
        return key
    return key[:2] + "*" * (len(key) - 4) + key[-2:]


def banner():
    os.system("cls" if os.name == "nt" else "clear")
    print("")
    print("+------------------------------------------+")
    print("|     Instagram Reel Transcriber           |")
    print("+------------------------------------------+")
    print("")


def menu():
    print("  [1] Single link")
    print("  [2] Batch links")
    print("  [3] Full channel")
    print("  [0] Exit")
    print("")


def server_is_running():
    import urllib.request
    try:
        urllib.request.urlopen(f"{BASE_URL}/health", timeout=2)
        return True
    except Exception:
        return False


def start_server():
    if server_is_running():
        print("  Server running.")
        return True
    print("  Starting server...", end=" ", flush=True)
    cmd = [str(VENV_PYTHON), "-m", "uvicorn", "app.main:app",
           "--host", "0.0.0.0", "--port", "8000"]
    proc = subprocess.Popen(
        cmd, cwd=str(SCRIPT_DIR),
        stdout=open(SERVER_LOG, "w"), stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    SERVER_PID.write_text(str(proc.pid))
    for _ in range(30):
        time.sleep(1)
        if server_is_running():
            print("ready.")
            return True
        print(".", end="", flush=True)
    print("failed. Check data/server.log")
    return False


def server_has_api_key():
    """Probe the running server to see whether it expects an API key.
    We infer this from whether a trivial POST /transcribe with no key gets 401/403.
    """
    import urllib.request
    import urllib.error

    try:
        body = json.dumps({"url": "https://www.instagram.com/reel/INVALID_TEST_KEY/check/"}).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/transcribe", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                code = r.status
        except urllib.error.HTTPError as e:
            code = e.code
        return code in (401, 403)
    except Exception:
        return False


def api_post(ep, data, api_key: str | None = None):
    import urllib.request

    headers = {"Content-Type": "application/json"}
    key = api_key or _api_key()
    if key:
        headers["X-API-Key"] = key
        log_api("POST", ep, redacted=True)
    req = urllib.request.Request(
        f"{BASE_URL}{ep}", data=json.dumps(data).encode(),
        headers=headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def log_api(method: str, ep: str, redacted: bool = False) -> None:
    """Best-effort request log for debugging; never blocks the call."""
    try:
        key = _api_key()
        label = _redacted(key) if redacted and key else "no-key"
        print(f"    [api] {method} {ep} key={label}", flush=True)
    except Exception:
        pass


def api_get(ep, api_key: str | None = None):
    import urllib.request
    req = urllib.request.Request(f"{BASE_URL}{ep}")
    if api_key:
        req.add_header("X-API-Key", api_key)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def api_get_text(ep, api_key: str | None = None):
    import urllib.request
    req = urllib.request.Request(f"{BASE_URL}{ep}")
    if api_key:
        req.add_header("X-API-Key", api_key)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def submit(url, lang=None, translate=False, api_key: str | None = None):
    body = {"url": url}
    if lang:
        body["lang"] = lang
    if translate:
        body["translate"] = True
    return api_post("/transcribe", body, api_key=api_key)


def poll(job_id, max_wait=180, api_key: str | None = None):
    print("  Polling...", end="", flush=True)
    elapsed = 0
    last_status = None
    while elapsed < max_wait:
        time.sleep(3)
        elapsed += 3
        try:
            s = api_get(f"/jobs/{job_id}", api_key=api_key).get("status", "")
        except Exception:
            print(".", end="", flush=True)
            continue
        if s in ("done", "failed"):
            if s == "done":
                print(" done.")
            else:
                print(" FAILED.")
            return s == "done"
        if s != last_status:
            print(".", end="", flush=True)
            last_status = s
    print(" timeout.")
    return False


def save(job_id, api_key: str | None = None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sid = re.sub(r"[^a-zA-Z0-9_-]", "_", job_id)
    srt = OUTPUT_DIR / f"transcript_{sid}.srt"
    txt = OUTPUT_DIR / f"transcript_{sid}.txt"
    try:
        srt.write_text(api_get_text(f"/jobs/{job_id}/srt", api_key=api_key), encoding="utf-8")
    except Exception as e:
        print(f"  SRT error: {e}")
        return
    try:
        txt.write_text(api_get_text(f"/jobs/{job_id}/txt", api_key=api_key), encoding="utf-8")
    except Exception as e:
        print(f"  TXT error: {e}")
        return
    print(f"  SRT: {srt}")
    print(f"  TXT: {txt}")
    print()
    # Show transcript
    content = txt.read_text(encoding="utf-8")
    print("  -- Transcript --")
    print(f"  {content.strip()}")
    print()
    # Show first 5 cues
    blocks = srt.read_text(encoding="utf-8").strip().split("\n\n")
    print(f"  -- Subtitles ({len(blocks)} cues) --")
    for b in blocks[:5]:
        for line in b.strip().split("\n"):
            print(f"  {line}")
        print()
    if len(blocks) > 5:
        print(f"  ... +{len(blocks)-5} more")


def ask_lang():
    print("  Language?")
    print("  [1] Auto-detect  [2] English  [3] Arabic")
    print("  [4] Translate -> English")
    c = input("  > ").strip()
    if c == "2":
        return "en", False
    if c == "3":
        return "ar", False
    if c == "4":
        return None, True
    return None, False


def process_url(url, lang=None, translate=False, api_key: str | None = None):
    print(f"\n  Checking: {url}")
    try:
        existing = asyncio.run(_find_job_for_url(BASE_URL, api_key, url))
    except Exception as e:
        print(f"  Check failed: {e}")
        existing = None

    if existing and existing.get('status') in ('done', 'failed'):
        jid = existing.get('job_id', '')
        print(f"  Already processed [{existing.get('status')}] job_id={jid}")
        if existing.get('status') == 'done' and jid:
            save(jid, api_key=api_key)
        return

    print(f"  Submitting: {url}")
    try:
        resp = submit(url, lang=lang, translate=translate, api_key=api_key)
    except Exception as e:
        print(f"  Submit failed: {e}")
        return
    jid = resp.get("job_id", "")
    if not jid:
        print(f"  Failed: {resp}")
        return
    if poll(jid, api_key=api_key):
        save(jid, api_key=api_key)


def process_batch(urls, lang=None, translate=False, api_key: str | None = None):
    """Process batch URLs serially, one complete download/transcription at a time."""
    process_channel_sequential(urls, lang=lang, translate=translate, api_key=api_key, label="Batch")


def _batch_submit_hook(url: str, result: dict) -> None:
    jid = result.get('job_id', '')
    if jid:
        print(f"  [{jid}] submitted: {url[:50]}")
    else:
        print(f"  ERROR submitting {url[:50]}: {result.get('error')}")


def _batch_poll_hook(url: str, data: dict) -> None:
    status = data.get('status', '')
    jid = data.get('job_id', '')
    if status in ('done', 'failed') and jid:
        label = 'done' if status == 'done' else 'FAILED'
        print(f"  [{jid}] {label}: {url[:50]}")


async def _find_job_for_url(base_url: str, api_key: str | None, url: str) -> dict | None:
    from app.batch import _find_existing_job
    from httpx import AsyncClient, Timeout
    client = AsyncClient(base_url=base_url, timeout=Timeout(10.0))
    try:
        return await _find_existing_job(client, api_key, url)
    finally:
        await client.aclose()


def mode_single():
    lang, translate = ask_lang()
    url = input("\n  Paste Instagram link:\n  > ").strip()
    if not url or "instagram.com" not in url:
        print("  Not a valid Instagram link.")
        return
    key = os.getenv("API_KEY", "").strip() or _api_key()
    process_url(url, lang=lang, translate=translate, api_key=key)


def get_channel_reels(username):
    """Collect reel URLs from an Instagram channel by pasting from your browser.

    Instagram 429s / bot-blocks automated profile enumeration (web_profile_info,
    embed endpoint, _sharedData), so we go straight to the reliable path:
    open the profile in your browser and paste the reel URLs you see.
    """
    from app.channel import collect_reel_urls
    print(f"\n  Fetching reels from @{username}...")
    print(f"  Open https://www.instagram.com/{username}/ in your browser.")
    urls = collect_reel_urls(username)
    print(f"\n  Found {len(urls)} new reel(s).")
    return urls


def mode_single():
    lang, translate = ask_lang()
    url = input("\n  Paste Instagram link:\n  > ").strip()
    if not url or "instagram.com" not in url:
        print("  Not a valid Instagram link.")
        return
    key = os.getenv("API_KEY", "").strip() or _api_key()
    process_url(url, lang=lang, translate=translate, api_key=key)


def mode_batch():
    lang, translate = ask_lang()
    if not URLS_FILE.exists():
        print(f"\n  {URLS_FILE} not found.")
        print(f"  Create it with one Instagram link per line.")
        return
    urls = []
    for line in URLS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and "instagram.com" in line:
            urls.append(line)
    if not urls:
        print(f"\n  No Instagram links found in {URLS_FILE.name}")
        return
    print(f"\n  Found {len(urls)} links in {URLS_FILE.name}")
    confirm = input("  Transcribe all? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return
    key = os.getenv("API_KEY", "").strip() or _api_key()
    process_batch(urls, lang=lang, translate=translate, api_key=key)


def mode_channel():
    lang, translate = ask_lang()
    username = input("\n  Instagram username (without @):\n  > ").strip().lstrip("@")
    if not username:
        print("  No username.")
        return
    urls = get_channel_reels(username)
    if not urls:
        print("  No reels found.")
        return
    print(f"\n  Found {len(urls)} reels.")
    confirm = input("  Transcribe all sequentially (one after another)? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return
    key = os.getenv("API_KEY", "").strip() or _api_key()
    process_channel_sequential(urls, lang=lang, translate=translate, api_key=key)
    print(f"\n  All done. Run: python -m app.export_all")
    print(f"  To export all transcripts: data/exports/")


def process_channel_sequential(
    urls, lang=None, translate=False, api_key: str | None = None, label: str = "Channel"
):
    """Transcribe each reel one at a time, saving before moving to the next.

    Sequential avoids hitting Instagram/Groq rate limits in parallel and gives
    a clear per-reel progress log.
    """
    print(f"\n  Transcribing {len(urls)} reel(s) sequentially...\n")
    done = 0
    failed = 0
    for i, url in enumerate(urls, 1):
        print(f"  [{i}/{len(urls)}] {url[:60]}")
        try:
            existing = asyncio.run(_find_job_for_url(BASE_URL, api_key, url))
        except Exception as e:
            print(f"    Check failed: {e}")
            existing = None
        if existing and existing.get('status') in ('done', 'failed'):
            jid = existing.get('job_id', '')
            print(f"    Already processed [{existing.get('status')}] job_id={jid}")
            if existing.get('status') == 'done' and jid:
                save(jid, api_key=api_key)
            if existing.get('status') == 'done':
                done += 1
            else:
                failed += 1
            continue
        try:
            resp = submit(url, lang=lang, translate=translate, api_key=api_key)
            jid = resp.get("job_id", "")
            if not jid:
                print(f"    Submit failed: {resp}")
                failed += 1
                continue
            print(f"    job_id={jid} — polling...")
            if poll(jid, max_wait=600, api_key=api_key):
                save(jid, api_key=api_key)
                done += 1
            else:
                print(f"    Job did not complete in time.")
                failed += 1
        except Exception as e:
            print(f"    Error: {e}")
            failed += 1
        print()
    print(f"\n  {label} complete: {done} done, {failed} failed.")
    if failed:
        print(f"  Failed URLs: data/failed_urls.txt")


def cleanup():
    """Remove old output and job files."""
    for folder in [OUTPUT_DIR, SCRIPT_DIR / "data" / "jobs", SCRIPT_DIR / "data" / "work"]:
        if folder.exists():
            for f in folder.iterdir():
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    import shutil
                    shutil.rmtree(f)


def main():
    banner()
    if not start_server():
        return
    if server_has_api_key():
        key = os.getenv("API_KEY", "").strip() or _api_key()
        print(f"  Server requires API key ({_redacted(key)}).")
    else:
        print("  Server does not require an API key.")
    try:
        while True:
            menu()
            choice = input("  Pick (0-3): ").strip()
            if choice == "0":
                print("\n  Shutting down server...")
                _shutdown_server()
                print("\n  Bye!")
                break
            elif choice == "1":
                mode_single()
            elif choice == "2":
                mode_batch()
            elif choice == "3":
                mode_channel()
            else:
                print("  Invalid.")
            input("\n  Press Enter to continue...")
            banner()
    except KeyboardInterrupt:
        print("\n\n  Interrupted - shutting down server...")
        _shutdown_server()
        print("\n  Bye!")


def _shutdown_server() -> None:
    """Request a graceful server shutdown and then hard-kill the backend.

    We try POST /shutdown first so the backend can mark running jobs cancelled,
    but the important part here is that we reliably kill the spawned uvicorn
    process tree afterwards so the bat/terminal can exit with no orphan process.
    """
    try:
        import json as _json
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"{BASE_URL}/shutdown",
            data=_json.dumps({}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                pass
        except urllib.error.HTTPError:
            pass
        except Exception:
            pass
    except Exception:
        pass

    _kill_server_process()


def _kill_server_process() -> None:
    """Best-effort kill of the server started by start_server().

    Uses taskkill /F /T so the whole process tree (uvicorn + any child
    processes) is terminated on Windows, then removes the stale pid file.
    """
    import os
    import subprocess

    pid_path = SCRIPT_DIR / "data" / "server.pid"
    pid = None
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pid = None

    if pid:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
            else:
                import signal
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except OSError:
                    pass
        except Exception:
            pass

    try:
        pid_path.unlink(missing_ok=True)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Bye!")
