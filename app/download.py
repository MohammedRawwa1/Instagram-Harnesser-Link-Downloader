"""Instagram video download via yt-dlp (async-friendly: runs in a worker thread)."""

import asyncio
import logging
from pathlib import Path

from yt_dlp import YoutubeDL

log = logging.getLogger("insta.download")


def _run_download(url: str, out_dir: Path) -> Path:
    opts = {
        "outtmpl": str(out_dir / "source.%(ext)s"),
        # Prefer a single mp4 file; no merging needed for most Instagram posts.
        "format": "mp4/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "no_color": True,
        "noprogress": True,
        "retries": 2,
        "socket_timeout": 30,
    }
    with YoutubeDL(opts) as ydl:
        # Raises yt_dlp.utils.DownloadError on failure (private posts, removed
        # videos, rate limiting...).
        ydl.download([url])

    files = sorted(out_dir.glob("source.*"))
    if not files:
        raise RuntimeError("yt-dlp finished but produced no media file.")
    return files[0]


async def download_instagram_video(url: str, out_dir: Path) -> Path:
    """Download the Instagram media to `out_dir` and return the local file path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        return await asyncio.to_thread(_run_download, url, out_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}") from e