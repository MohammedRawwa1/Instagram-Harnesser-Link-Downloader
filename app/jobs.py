"""Job registry: tracks every transcription job and mirrors its state to disk
so the whole workbase (API, callbacks, output files) stays in sync."""

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

STATUS_QUEUED = "queued"
STATUS_DOWNLOADING = "downloading"
STATUS_SPLITTING = "splitting"
STATUS_TRANSCRIBING = "transcribing"
STATUS_STITCHING = "stitching"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


@dataclass
class Job:
    job_id: str
    url: str
    lang: Optional[str]
    callback_url: Optional[str]
    translate: bool = False
    status: str = STATUS_QUEUED
    progress_done: int = 0
    progress_total: int = 0
    duration: float = 0.0
    srt_path: str = ""
    txt_path: str = ""
    timings_path: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        """Restore a job while ignoring derived fields from persisted JSON."""
        names = {item.name for item in fields(cls)}
        return cls(**{name: value for name, value in data.items() if name in names})

    def to_dict(self) -> dict:
        data = asdict(self)
        data["progress"] = (
            f"{self.progress_done}/{self.progress_total}"
            if self.progress_total
            else ""
        )
        return data


class JobRegistry:
    def __init__(self, jobs_dir: Path):
        self._jobs_dir = jobs_dir
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self, job: Job) -> Job:
        async with self._lock:
            self._jobs[job.job_id] = job
            # Mirror state to disk so get/find lookups across restarts see the
            # job before any in-memory-only updates happen.
            self._persist(job)
        return job

    async def get(self, job_id: str) -> Optional[Job]:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                try:
                    jf = self._jobs_dir / f"{job_id}.json"
                    if jf.exists():
                        data = json.loads(jf.read_text(encoding="utf-8"))
                        job = Job.from_dict(data)
                        self._jobs[job_id] = job
                except (json.JSONDecodeError, OSError):
                    pass
            return job

    async def update(self, job_id: str, **changes) -> Job:
        async with self._lock:
            job = self._jobs[job_id]
            for k, v in changes.items():
                setattr(job, k, v)
            self._persist(job)
            return job

    def _persist(self, job: Job) -> None:
        """Mirror state to a JSON file so restarts and external tools can read it."""
        try:
            (self._jobs_dir / f"{job.job_id}.json").write_text(
                json.dumps(job.to_dict(), indent=2), encoding="utf-8"
            )
        except OSError:
            pass  # status file is best-effort; in-memory state is authoritative

    async def find_by_url(self, url: str) -> Optional[Job]:
        """Return an existing job for `url`, if any (any status)."""
        async with self._lock:
            # Check in-memory first, then disk so newly created jobs are found
            # immediately while still supporting cross-call/restart visibility.
            for job in self._jobs.values():
                if job.url == url:
                    return job
            try:
                for jf in self._jobs_dir.glob("*.json"):
                    data = json.loads(jf.read_text(encoding="utf-8"))
                    if data.get("url") == url:
                        return Job.from_dict(data)
            except (json.JSONDecodeError, OSError):
                pass
            return None