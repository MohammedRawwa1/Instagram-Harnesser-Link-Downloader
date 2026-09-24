# Instagram → Groq → lip-synced SRT/TXT backend

A pure **async backend** (no UI, no n8n): a group of `async/await` functions
running together that take any public Instagram video and return
**perfectly synced subtitles** — word-level timestamps from Groq's
whisper-large-v3, engineered into readable, lip-sync-accurate SRT cues.

Instagram videos are small, so the pipeline is built around that: every video
is **always split** into (possibly overlapping) chunks, transcribed in
parallel, then stitched back into one continuous, deduped timeline.

```
POST /transcribe {url} ──► 202 {job_id}          ← async ack, instant
        │
        ▼  (background task)
 ┌─ download.py ────── yt-dlp grabs the Instagram media
 ├─ audio.py ───────── ffmpeg → 16 kHz mono MP3
 ├─ audio.py ───────── ALWAYS split into overlapping chunks  (reel → 1 chunk)
 ├─ groq_client.py ─── transcribe chunks in parallel
 │                      timestamp_granularities=["word","segment"]
 ├─ stitch.py ───────── merge to source timeline → dedupe overlap → cue-engineer
 ├─ write artifacts ── subtitle - <id>.srt / transcript - <id>.txt / timings - <id>.json
 └─ worker.py ──────── POST results to callback_url (optional)
```

## Verified end-to-end

Tested live against a real public Instagram reel (`instagram.com/reel/DXq30xCAl6V/`,
63.4 s): submit → `202` → poll through `downloading → splitting → transcribing
1/1 → done` → artifacts written. The SRT validated as: 29 cues, avg 2.13 s on
screen (max 6.41 s), monotonic and non-overlapping timestamps, and the SRT word
stream exactly matches the TXT transcript (nothing dropped or duplicated).
Sample artifacts from that run are in `data/output/`.

## Quick start

```bash
cd instagram-groq-backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # set GROQ_API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker compose up --build
```

For the Windows menu, run `python transcribe.py` and choose **[2] Batch links**.
Put one Instagram URL per line in `scripts/urls.txt` at the project root.
Blank lines and lines without an Instagram URL are ignored.

Submit a video (the `scripts/test_insta.sh` script does this + polls):

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.instagram.com/reel/ABC123/", "callback_url": "https://your.app/hook"}'
# → {"job_id": "job_1a2b3c4d", "status": "queued", ...}

curl http://localhost:8000/jobs/job_1a2b3c4d            # poll status + progress
curl http://localhost:8000/jobs/job_1a2b3c4d/srt        # subtitle file
curl http://localhost:8000/jobs/job_1a2b3c4d/txt        # transcript
curl http://localhost:8000/jobs/job_1a2b3c4d/timings    # per-word + per-cue JSON
```

## API

| Endpoint | Body | Returns |
|---|---|---|
| `POST /transcribe` | `{url, job_id?, lang?, callback_url?}` | `202` + `{job_id, status, srt_url, txt_url, timings_url}` |
| `GET /jobs/{job_id}` | — | status, progress (`3/8`), duration, artifact paths |
| `GET /jobs/{job_id}/srt` | — | the `.srt` file |
| `GET /jobs/{job_id}/txt` | — | the `.txt` transcript |
| `GET /jobs/{job_id}/timings` | — | timings JSON (words + cues) |
| `GET /health` | — | `{status: ok}` |

Job lifecycle: `queued → downloading → splitting → transcribing (n/m chunks)
→ stitching → done` (or `failed` with an error message).

When `callback_url` is provided, the backend POSTs `{job_id, status,
duration, srt_path, txt_path, timings_path, srt_text, txt_text, error}` once
the job finishes (success or failure).

## How the lip-sync accuracy is engineered

1. **Word-level timestamps** — every chunk is transcribed with
   `timestamp_granularities=["word","segment"]`, so each word carries exact
   start/end times instead of 30-second whisper segments.
2. **Overlapping chunks** — chunks overlap by 2 s (`OVERLAP_SECONDS`), so a
   word straddling a boundary is heard whole by at least one chunk. Words are
   never cut in half, which is what causes whisper's boundary garble
   ("subject. subject.").
3. **Overlap dedupe** — identical words at overlapping times are removed when
   stitching chunks back together.
4. **Cue engineering** (`stitch.py`) — words are grouped into cues that look
   and sync like a professional captioneer's: ≤ 42 chars, ≤ 7 s on screen,
   cues break at pauses > 0.35 s, and each cue starts exactly at its first word
   and ends just after its last word (padded 0.3 s, never bleeding into the
   next cue).
5. **Artifacts** — SRT (for editors/players), TXT (reading), and a `timings`
   JSON with every word and cue timestamped (for markers / further tooling).

## Security (every step)

- **Webhook → API key** — if `API_KEY` is set, every endpoint requires
  `X-API-Key`. Leave empty only behind a firewall.
- **Strict URL allowlist** — only public `https://instagram.com/
  (reel|reels|p|tv|stories)/<id>` links pass validation; everything else is
  rejected before any network or filesystem work.
- **job_id sanitization** — `[A-Za-z0-9_-]{1,64}` only, because it becomes
  part of file paths. No traversal possible.
- **callback_url validation** — must be plain `https`, no credentials in URL.
- **No secrets in code** — the Groq key comes from `GROQ_API_KEY` (env /
  `.env` / docker secret), never from the request.
- **Fixed pipeline code** — yt-dlp/ffmpeg/Groq are called with validated
  inputs only; nothing from the request is interpolated into a shell command.
- **Bounded concurrency** — `MAX_WORKERS` semaphore caps parallel Groq calls
  (rate-limit friendly); each chunk retries with exponential backoff.

## Configuration

See `.env.example`. Key knobs: `CHUNK_SECONDS` (chunk size), `OVERLAP_SECONDS`
(overlap), `MAX_WORKERS` (parallelism), `GROQ_MODEL`, `WORK_DIR` (storage).

## Notes & troubleshooting

- **Private/age-restricted posts** need Instagram cookies for yt-dlp — add
  `cookiesfrombrowser`/`cookiefile` support in `download.py` for those.
- **Groq size limits** — a chunk of `CHUNK_SECONDS` at 64 kbps ≈ 0.5 MB/min,
  so even 15-min chunks are far below Groq's 25 MB upload limit.
- **Rate limits** — if Groq 429s appear, lower `MAX_WORKERS`.
- **`groq/` folder** — a standalone copy of the CLI transcription pipeline
  (same logic) for running outside the backend; see `groq/README.md`.

## Layout

```
app/
  main.py         FastAPI endpoints (async submit / status / file download)
  config.py       settings from environment
  jobs.py         job registry + disk mirror
  download.py     yt-dlp download (async)
  audio.py        ffmpeg compress + always-split overlapping chunks (async)
  groq_client.py  Groq word-level transcription, semaphore + retries (async)
  stitch.py       stitch → dedupe → cue engineering → SRT/TXT/timings
  worker.py       the async pipeline that ties everything together
groq/             standalone CLI transcription folder (same core logic)
tests/            unit tests for the cue engineering
scripts/          test_insta.sh (submit + poll)
```