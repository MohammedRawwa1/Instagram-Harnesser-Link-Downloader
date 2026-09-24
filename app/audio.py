"""Async ffmpeg helpers: compress to 16 kHz mono MP3, probe duration, and
ALWAYS split audio into overlapping chunks (the core of the lip-sync pipeline).

Uses system ffmpeg if it's on PATH (local install), otherwise falls back to the
imageio-ffmpeg bundle. Fails fast with a clear message if neither is available.
"""

import asyncio
import re
import shutil
import sys
from pathlib import Path

import imageio_ffmpeg


def _find_ffmpeg() -> str:
    """Return the ffmpeg binary path to use.

    Prefers system ffmpeg (on PATH) for a local install; falls back to the
    imageio-ffmpeg bundle. Raises RuntimeError if neither is available.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    bundle_path = imageio_ffmpeg.get_ffmpeg_exe()
    if Path(bundle_path).exists():
        return bundle_path
    raise RuntimeError(
        "ffmpeg not found. Install it system-wide (e.g. apt install ffmpeg, "
        "brew install ffmpeg, or choco install ffmpeg) or let imageio-ffmpeg "
        "bundle it (pip install imageio-ffmpeg)."
    )


FFMPEG_BIN = _find_ffmpeg()


async def _run(cmd: list[str], *, timeout: float = 600) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd[:3])} ...")
    return proc.returncode or 0, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


async def probe_duration(path: Path) -> float:
    """Read the media duration from `ffmpeg -i` output (no ffprobe binary needed)."""
    code, _out, err = await _run([FFMPEG_BIN, "-i", str(path)])
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", err)
    if not match:
        raise RuntimeError(f"Could not determine duration for {path}")
    h, m, s = match.groups()
    return float(h) * 3600 + float(m) * 60 + float(s)


async def compress_to_mp3(source: Path, out_path: Path) -> float:
    """Convert any audio/video to a compressed mono 16 kHz MP3. Returns duration."""
    cmd = [
        FFMPEG_BIN, "-y", "-i", str(source),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        str(out_path),
    ]
    code, _out, err = await _run(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg compression failed: {err[-500:]}")
    return await probe_duration(out_path)


async def split_into_chunks(
    compressed: Path,
    work_dir: Path,
    chunk_seconds: int,
    overlap_seconds: float,
) -> list[tuple[Path, float]]:
    """Split audio into overlapping chunks.

    Chunk i covers [i*C - O, (i+1)*C - O) of the source timeline (chunk 0 starts
    at 0). The overlap means a word straddling a boundary is heard whole by the
    next chunk — words are never cut in half, which is what kills whisper's
    boundary garble. Returns [(path, global_offset), ...].

    The pipeline always splits — a 30 s reel yields exactly one chunk.
    """
    duration = await probe_duration(compressed)
    chunks: list[tuple[Path, float]] = []
    idx = 0
    start = 0.0
    while start < duration - 1e-6:
        if idx == 0:
            ss, t = 0.0, min(chunk_seconds, duration)
        else:
            ss = max(0.0, start - overlap_seconds)
            t = min(chunk_seconds, duration - ss)
        out_path = work_dir / f"chunk_{idx:04d}.mp3"
        cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{ss:.3f}", "-i", str(compressed),
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
            "-t", f"{t:.3f}",
            str(out_path),
        ]
        code, _out, err = await _run(cmd)
        if code != 0:
            raise RuntimeError(f"ffmpeg chunk split failed: {err[-500:]}")
        chunks.append((out_path, ss))
        start += chunk_seconds
        idx += 1
    if not chunks:
        raise RuntimeError("No chunks produced — check the input media.")
    return chunks