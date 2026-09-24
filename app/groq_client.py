"""Async Groq Whisper client.

Requests word-level timestamps (timestamp_granularities=["word", "segment"]) so
the whole backend works from exact per-word timing — the foundation of the
lip-sync architecture. Chunks are transcribed concurrently through an adaptive
worker pool that shrinks on 429s / high latency and ramps back up when the
coast is clear.
"""

import asyncio
import logging
import time
from pathlib import Path

from groq import AsyncGroq

from .pool import WorkerPool

log = logging.getLogger("insta.groq")

MAX_RETRIES = 3


class GroqTranscriber:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "whisper-large-v3",
        max_workers: int = 5,
        timeout: float = 300.0,
        pool: WorkerPool | None = None,
    ):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        self._key = api_key
        self._model = model
        self._timeout = timeout
        self._pool = pool or WorkerPool(
            max_workers=max_workers,
            min_workers=1,
            cooldown=30.0,
            ramp_interval=5.0,
        )
        self._client: AsyncGroq | None = None

    async def __aenter__(self) -> "GroqTranscriber":
        self._pool.start()
        self._client = AsyncGroq(api_key=self._key, timeout=self._timeout, max_retries=2)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        await self._pool.close()

    async def transcribe_chunk(self, path: Path, lang: str | None = None, translate: bool = False) -> dict:
        """Transcribe one audio chunk. Returns a Groq verbose_json-style dict
        with `text`, `segments` and `words` (word-level timestamps).

        When translate=True, uses the /audio/translations endpoint which
        translates any language to English automatically.
        """
        assert self._client is not None
        data = path.read_bytes()

        last_error = "unknown error"
        result = None
        _start = time.monotonic()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Adaptive pool: acquire a slot (with timeout so we don't hang
                # if the pool has shrunk to a standstill).
                async with await self._pool.acquire(timeout=60.0) as _tok:
                    _req_start = time.monotonic()
                    if translate:
                        import httpx as _httpx
                        async with _httpx.AsyncClient(timeout=self._timeout) as _client:
                            resp = await _client.post(
                                "https://api.groq.com/openai/v1/audio/translations",
                                headers={"Authorization": f"Bearer {self._key}"},
                                files={"file": (path.name, data, "audio/mpeg")},
                                data={
                                    "model": self._model,
                                    "response_format": "verbose_json",
                                },
                            )
                        _ttf_ms = (time.monotonic() - _req_start) * 1000
                        if resp.status_code == 429:
                            self._pool.mark_rate_limited()
                            self._pool.mark_slow(ttf_ms=_ttf_ms)
                            resp.raise_for_status()
                        resp.raise_for_status()
                        _data = resp.json()
                        class _SafeObj:
                            def __init__(self, d):
                                self._d = d
                            def __getattr__(self, name):
                                return self._d.get(name)
                        result = _SafeObj(_data)
                    else:
                        kwargs: dict = {
                            "file": (path.name, data, "audio/mpeg"),
                            "model": self._model,
                            "response_format": "verbose_json",
                            "timestamp_granularities": ["word", "segment"],
                        }
                        if lang:
                            kwargs["language"] = lang
                        result = await self._client.audio.transcriptions.create(**kwargs)
                    _ttf_ms = (time.monotonic() - _req_start) * 1000
                    if _ttf_ms < self._pool.slow_threshold_ms:
                        self._pool.mark_fast(ttf_ms=_ttf_ms)
                    else:
                        self._pool.mark_slow(ttf_ms=_ttf_ms)
                break
            except Exception as e:  # noqa: BLE001 — SDK raises typed errors; retry transient ones
                status = getattr(e, "status_code", None)
                last_error = f"{type(e).__name__}: {e}"
                if status == 429:
                    self._pool.mark_rate_limited()
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                wait = min(2 ** attempt, 20)
                log.warning("  chunk %s failed (attempt %d/%d): %s — retrying in %ds (pool limit=%d)",
                            path.name, attempt, MAX_RETRIES, last_error, wait, self._pool.limit)
                await asyncio.sleep(wait)
        if result is None:
            raise RuntimeError(f"Chunk {path.name} failed after {MAX_RETRIES} attempts: {last_error}")

        # Groq may return words/segments as dicts or objects — handle both.
        def _val(item, key: str):
            return item[key] if isinstance(item, dict) else getattr(item, key)

        words = []
        for w in _val(result, "words") or []:
            w_start, w_end, w_word = float(_val(w, "start")), float(_val(w, "end")), str(_val(w, "word")).strip()
            if not w_word or w_end <= w_start:
                continue
            words.append({"start": w_start, "end": w_end, "word": w_word})

        segments = []
        for seg in _val(result, "segments") or []:
            segments.append({
                "start": float(_val(seg, "start")),
                "end": float(_val(seg, "end")),
                "text": str(_val(seg, "text") or "").strip(),
            })

        return {
            "text": str(_val(result, "text") or "").strip(),
            "segments": segments,
            "words": words,
        }

    async def transcribe_many(
        self,
        chunks: list[tuple[Path, float]],
        lang: str | None = None,
        translate: bool = False,
        on_progress=None,
    ) -> list[dict]:
        """Transcribe all chunks concurrently through the adaptive pool.

        Each chunk acquires a slot from the pool, which shrinks on 429s and
        high latency and ramps back up when things are stable.
        """
        async def one(chunk: tuple[Path, float]) -> dict:
            path, _offset = chunk
            async with await self._pool.acquire(timeout=90.0):
                result = await self.transcribe_chunk(path, lang, translate=translate)
            if on_progress is not None:
                await on_progress()
            return result

        return list(await asyncio.gather(*(one(c) for c in chunks), return_exceptions=True))