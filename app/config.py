"""Central configuration — every knob lives here, read from environment variables
or a local .env file (loaded automatically via python-dotenv)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env in the working directory, if present


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # -- Groq -----------------------------------------------------------------
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "whisper-large-v3"))
    groq_timeout: float = field(default_factory=lambda: float(_env("GROQ_TIMEOUT", "300")))
    max_workers: int = field(default_factory=lambda: int(_env("MAX_WORKERS", "5")))

    # -- Chunking (the pipeline ALWAYS splits, even tiny reels → 1 chunk) ----
    chunk_seconds: int = field(default_factory=lambda: int(_env("CHUNK_SECONDS", "900")))   # 15 min
    overlap_seconds: float = field(default_factory=lambda: float(_env("OVERLAP_SECONDS", "2")))

    # -- Language ---------------------------------------------------------------
    # ISO-639-1 code: en, ar, fr, es, etc. Empty = auto-detect by Whisper.
    # Ignored when translate_to_en=true (translations endpoint always outputs English).
    default_lang: str = field(default_factory=lambda: _env("DEFAULT_LANG", ""))

    # -- Translation -----------------------------------------------------------
    translate_to_en: bool = field(default_factory=lambda: _env("TRANSLATE_TO_EN", "").lower() in ("1", "true", "yes"))

    # -- Server ---------------------------------------------------------------
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))
    api_key: str = field(default_factory=lambda: _env("API_KEY", ""))  # optional shared secret

    # -- Storage --------------------------------------------------------------
    # ./data works locally out of the box; docker-compose overrides to
    # /data/instagram-groq inside the container.
    base_dir: str = field(default_factory=lambda: _env("WORK_DIR", "./data"))
    callback_timeout: float = field(default_factory=lambda: float(_env("CALLBACK_TIMEOUT", "10")))

    def work_dir(self) -> Path:
        return Path(self.base_dir) / "work"

    def output_dir(self) -> Path:
        return Path(self.base_dir) / "output"

    def jobs_dir(self) -> Path:
        return Path(self.base_dir) / "jobs"

    def ensure_dirs(self) -> None:
        for d in (self.work_dir(), self.output_dir(), self.jobs_dir()):
            d.mkdir(parents=True, exist_ok=True)