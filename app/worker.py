"""The pipeline — a group of async/await functions running together:

    download → compress → ALWAYS split into overlapping chunks
             → transcribe chunks concurrently (word-level timestamps)
             → stitch + dedupe + cue-engineer → SRT / TXT / timings
             → optional callback POST

Each stage updates the job registry, so callers can watch progress live.
"""

import asyncio
import logging
import time
from pathlib import Path

import httpx

from . import stitch
from .audio import compress_to_mp3, split_into_chunks
from .config import Settings
from .download import download_instagram_video
from .groq_client import GroqTranscriber
from .jobs import (
    Job,
    JobRegistry,
    STATUS_DONE,
    STATUS_DOWNLOADING,
    STATUS_FAILED,
    STATUS_SPLITTING,
    STATUS_STITCHING,
    STATUS_TRANSCRIBING,
)

log = logging.getLogger("insta.worker")


async def _notify_callback(job: Job, settings: Settings) -> None:
    """Best-effort callback POST with the finished job (never raises)."""
    if not job.callback_url:
        return
    payload = {
        "job_id": job.job_id,
        "status": job.status,
        "duration": job.duration,
        "srt_path": job.srt_path,
        "txt_path": job.txt_path,
        "timings_path": job.timings_path,
        "error": job.error,
    }
    try:
        # Attach full text too — handy for downstream reposting.
        if job.status == STATUS_DONE:
            for key, attr in (("srt_text", "srt_path"), ("txt_text", "txt_path")):
                path = getattr(job, attr)
                if path and Path(path).exists():
                    payload[key] = Path(path).read_text(encoding="utf-8")
        async with httpx.AsyncClient(timeout=settings.callback_timeout) as client:
            await client.post(job.callback_url, json=payload)
    except Exception as e:  # noqa: BLE001 — callbacks must never kill the job
        log.warning("Callback to %s failed: %s", job.callback_url, e)


async def run_job(job: Job, registry: JobRegistry, settings: Settings) -> None:
    """Execute one transcription job end-to-end."""
    try:
        work_dir = settings.work_dir() / job.job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        # -- 1. Download ------------------------------------------------------
        await registry.update(job.job_id, status=STATUS_DOWNLOADING)
        source = await download_instagram_video(job.url, work_dir)
        log.info("[%s] downloaded %s", job.job_id, source.name)

        # -- 2. Compress + always split into overlapping chunks ---------------
        await registry.update(job.job_id, status=STATUS_SPLITTING)
        compressed = work_dir / "audio.mp3"
        duration = await compress_to_mp3(source, compressed)
        chunks = await split_into_chunks(
            compressed, work_dir, settings.chunk_seconds, settings.overlap_seconds
        )
        log.info("[%s] %.1f min → %d chunk(s)", job.job_id, duration / 60, len(chunks))

        # -- 3. Transcribe chunks concurrently (word-level timestamps) --------
        await registry.update(
            job.job_id, status=STATUS_TRANSCRIBING,
            progress_total=len(chunks), progress_done=0,
        )

        async def _on_progress() -> None:
            current = await registry.get(job.job_id)
            await registry.update(job.job_id, progress_done=current.progress_done + 1)

        # Use per-job translate flag, falling back to global config default
        do_translate = job.translate or settings.translate_to_en
        # Use per-job lang, falling back to global DEFAULT_LANG config
        effective_lang = job.lang or settings.default_lang or None

        async with GroqTranscriber(
            settings.groq_api_key,
            model=settings.groq_model,
            max_workers=settings.max_workers,
            timeout=settings.groq_timeout,
        ) as transcriber:
            raw_results = await transcriber.transcribe_many(
                chunks, effective_lang, translate=do_translate, on_progress=_on_progress
            )
            # transcribe_many returns exceptions for failed chunks
            # (return_exceptions=True). Fail the job if any chunk failed.
            results = []
            for r in raw_results:
                if isinstance(r, Exception):
                    raise RuntimeError(f"chunk transcription failed: {r}") from r
                results.append(r)

        # -- 4. Stitch, dedupe, cue-engineer ----------------------------------
        await registry.update(job.job_id, status=STATUS_STITCHING)
        srt, txt, timings = stitch.build_artifacts(chunks, results, duration)

        # -- 5. Write artifacts ------------------------------------------------
        out_dir = settings.output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        srt_path = out_dir / f"subtitle - {job.job_id}.srt"
        txt_path = out_dir / f"transcript - {job.job_id}.txt"
        timings_path = out_dir / f"timings - {job.job_id}.json"
        # Tiny writes — fine to do inline; keeps ordering deterministic.
        srt_path.write_text(srt, encoding="utf-8")
        txt_path.write_text(txt, encoding="utf-8")
        timings_path.write_text(timings, encoding="utf-8")

        await registry.update(
            job.job_id, status=STATUS_DONE, duration=duration,
            srt_path=str(srt_path), txt_path=str(txt_path),
            timings_path=str(timings_path), finished_at=time.time(),
        )
        log.info("[%s] done → %s, %s", job.job_id, srt_path.name, txt_path.name)

    except Exception as e:  # noqa: BLE001 — surface everything on the job
        log.exception("[%s] failed", job.job_id)
        # Include the full traceback in the error so we can debug remotely.
        import traceback as _tb
        error_detail = f"{type(e).__name__}: {e}\n" + "".join(_tb.format_exception(type(e), e, e.__traceback__))
        await registry.update(
            job.job_id, status=STATUS_FAILED, error=error_detail,
            finished_at=time.time(),
        )

    await _notify_callback(await registry.get(job.job_id), settings)