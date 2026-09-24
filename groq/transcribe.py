#!/usr/bin/env python3
"""
transcribe.py — One script for all transcription: short or long, single file or a whole folder.

How it works:
    - Point it at a single file OR a folder of files (including subfolders — a
      subfolder of audios becomes a matching subfolder in transcripts/).
    - Every file is first converted to a compressed mono MP3 (16kHz) — this
      also handles video files (mp4, mkv, etc.) by extracting just the audio.
    - Long files get split into chunks and uploaded to Groq in parallel.
    - Compressed/chunk files are dropped into chunks/ (visible, so you can
      watch progress) and auto-deleted once that file's transcript is done.
    - Skips any file that already has a matching transcript (so re-running on
      a folder only processes new additions).

Folder layout expected:
    whisper/
    ├── inputs/         <- put audio/video files here (subfolders OK)
    ├── chunks/          <- compressed/split audio lands here while working, then is deleted
    ├── transcripts/    <- .txt and .srt output lands here (mirrors inputs/ subfolders)
    ├── .env            <- GROQ_API_KEY=...
    └── transcribe.py   <- this script

Output: for every file, you get:
    transcripts/transcript - <name>.txt              <- clean reading transcript, no timestamps
    transcripts/subtitle - <name>.srt   (with --srt) <- subtitle file with real per-line timestamps

SRT accuracy (--srt): subtitles are built from Groq's *word-level* timestamps
(whisper-large-v3 timestamp_granularities=["word","segment"]), then engineered
into readable, lip-sync-accurate cues:
    - Long files are chunked with a small overlap so no word is ever cut in
      half at a chunk boundary (overlapping audio is deduped when stitching).
    - Words are grouped into cues: max ~42 chars per cue, max ~7s on screen,
      cues break at natural pauses, and each cue ends just after its last word
      (padded slightly, never bleeding into the next cue).

Usage:
    python transcribe.py inputs/                      # batch: process every new file in inputs/ (incl. subfolders)
    python transcribe.py inputs/lecture3.mp3           # single file
    python transcribe.py inputs/ --workers 8           # more parallel uploads (faster, watch rate limits)
    python transcribe.py inputs/ --force               # re-transcribe even if a transcript already exists
    python transcribe.py inputs/ --srt                 # also generate timestamped, lip-sync-accurate .srt subtitles
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import hashlib

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

AUDIO_VIDEO_EXTENSIONS = {
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".webm",
    ".mp4", ".mkv", ".mov", ".avi",
}

# Files longer than this get chunked + parallelized. Below it, one direct upload is simpler and just as fast.
LONG_FILE_THRESHOLD_SECONDS = 20 * 60  # 20 minutes

# Chunks overlap by this much (seconds) so a word that straddles a chunk
# boundary is heard in full by at least one chunk — kills mid-word garble.
CHUNK_OVERLAP_SECONDS = 2.0

# Cue engineering knobs for SRT output.
CUE_MAX_CHARS = 42        # max chars per cue (≈2 lines of 21, comfortable to read)
CUE_MAX_SECONDS = 7.0     # never leave a cue on screen longer than this
CUE_BREAK_GAP = 0.35      # a pause longer than this splits into a new cue
CUE_END_PAD = 0.30        # keep the last word visible a beat longer (lip-sync friendly)


def load_env_key(env_path: Path = Path(".env")):
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("GROQ_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def get_client():
    api_key = os.environ.get("GROQ_API_KEY") or load_env_key()
    if not api_key:
        print("ERROR: No GROQ_API_KEY found. Put it in a .env file in this folder.", file=sys.stderr)
        sys.exit(1)
    try:
        from groq import Groq
    except ImportError:
        print("ERROR: groq package not installed. Run: pip install groq", file=sys.stderr)
        sys.exit(1)
    return Groq(api_key=api_key)


try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_BIN = "ffmpeg"


def get_duration_seconds(path: Path) -> float:
    import re
    result = subprocess.run([FFMPEG_BIN, "-i", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_str = result.stderr.decode("utf-8", errors="replace")
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", stderr_str)
    if match:
        h, m, s = match.groups()
        return float(h) * 3600 + float(m) * 60 + float(s)
    raise RuntimeError(f"Could not determine duration for {path}")


def compress_to_mp3(source: Path, out_path: Path):
    """Convert any audio/video file to a compressed mono 16kHz MP3 (drops video track)."""
    cmd = [
        FFMPEG_BIN, "-y", "-i", str(source),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        str(out_path),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def extract_and_chunk(compressed_source: Path, work_dir: Path, chunk_seconds: int,
                      overlap_seconds: float = 0.0):
    """Split audio into chunks of `chunk_seconds`, returning (chunk_path, global_offset) pairs.

    - overlap_seconds <= 0: exact cuts with stream copy (`-c copy -f segment`),
      chunk i covers [i*C, (i+1)*C). This keeps the plain-.txt fast path
      byte-identical to before.
    - overlap_seconds > 0: chunk i covers [i*C - O, (i+1)*C - O) (chunk 0 starts
      at 0) via re-encoded -ss/-t cuts, so a word straddling a boundary is heard
      whole by the next chunk and never gets split.
    """
    if overlap_seconds <= 0:
        pattern = work_dir / "chunk_%04d.mp3"
        cmd = [
            FFMPEG_BIN, "-y", "-i", str(compressed_source),
            "-c", "copy",
            "-f", "segment", "-segment_time", str(chunk_seconds),
            "-reset_timestamps", "1",
            str(pattern),
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        chunks = sorted(work_dir.glob("chunk_*.mp3"))
        if not chunks:
            raise RuntimeError("ffmpeg produced no chunks — check the input file is valid audio/video.")
        return [(c, i * chunk_seconds) for i, c in enumerate(chunks)]

    duration = get_duration_seconds(compressed_source)
    chunks = []
    idx = 0
    start = 0.0
    while start < duration - 1e-6:
        if idx == 0:
            ss = 0.0
            t = min(chunk_seconds, duration)
        else:
            ss = max(0.0, start - overlap_seconds)
            t = min(chunk_seconds, duration - ss)
        out_path = work_dir / f"chunk_{idx:04d}.mp3"
        cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{ss:.3f}", "-i", str(compressed_source),
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
            "-t", f"{t:.3f}",
            str(out_path),
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        chunks.append((out_path, ss))
        start += chunk_seconds
        idx += 1
    if not chunks:
        raise RuntimeError("ffmpeg produced no chunks — check the input file is valid audio/video.")
    return chunks


def transcribe_bytes(client, filename: str, data: bytes, lang, want_words: bool, retries: int = 3):
    """Returns (full_text, segments, words).

    - want_words=False -> plain text mode (fast path for .txt-only runs): no
      verbose_json overhead, segments/words are empty.
    - want_words=True  -> requests word-level timestamps (whisper-large-v3
      timestamp_granularities=["word","segment"]) for lip-sync-accurate SRT.
      If the installed groq SDK doesn't accept that parameter, degrades
      gracefully to segment-level timestamps instead of failing.
    """
    granular = want_words  # downgraded to False if the SDK rejects word granularity
    for attempt in range(1, retries + 1):
        try:
            kwargs = {
                "file": (filename, data),
                "model": "whisper-large-v3",
                "response_format": "verbose_json" if (want_words or granular) else "text",
            }
            if granular:
                kwargs["timestamp_granularities"] = ["word", "segment"]
            if lang:
                kwargs["language"] = lang
            result = client.audio.transcriptions.create(**kwargs)

            if not want_words and not granular:
                # response_format="text" returns a plain string, not an object
                text = result.strip() if isinstance(result, str) else result.text.strip()
                return text, [], []

            text = result.text.strip()

            raw_segments = getattr(result, "segments", None)
            segments = []
            if raw_segments:
                for seg in raw_segments:
                    seg_start = seg["start"] if isinstance(seg, dict) else seg.start
                    seg_end = seg["end"] if isinstance(seg, dict) else seg.end
                    seg_text = seg["text"] if isinstance(seg, dict) else seg.text
                    segments.append({"start": seg_start, "end": seg_end, "text": seg_text.strip()})

            words = []
            if granular:
                raw_words = getattr(result, "words", None)
                if raw_words:
                    for w in raw_words:
                        w_start = w["start"] if isinstance(w, dict) else w.start
                        w_end = w["end"] if isinstance(w, dict) else w.end
                        w_word = w["word"] if isinstance(w, dict) else w.word
                        word_text = str(w_word).strip()
                        if not word_text:
                            continue
                        if w_end <= w_start:  # sanitize bad timestamps
                            w_end = w_start + 0.3
                        words.append({"start": float(w_start), "end": float(w_end), "word": word_text})

            if not segments and text:
                segments = [{"start": 0.0, "end": 0.0, "text": text}]

            return text, segments, words
        except TypeError as e:
            if granular:
                # Older groq SDK without timestamp_granularities support.
                print("  groq SDK too old for word-level timestamps — falling back to segment timestamps")
                granular = False
                continue
            if attempt == retries:
                err = f"[TRANSCRIPTION FAILED after {retries} attempts: {e}]"
                return err, [{"start": 0.0, "end": 0.0, "text": err}], []
            wait = 2 ** attempt
            print(f"  {filename} failed (attempt {attempt}/{retries}): {e} — retrying in {wait}s")
            time.sleep(wait)
        except Exception as e:
            if attempt == retries:
                err = f"[TRANSCRIPTION FAILED after {retries} attempts: {e}]"
                return err, [{"start": 0.0, "end": 0.0, "text": err}], []
            wait = 2 ** attempt
            print(f"  {filename} failed (attempt {attempt}/{retries}): {e} — retrying in {wait}s")
            time.sleep(wait)


def format_srt_timestamp(seconds: float) -> str:
    ms_total = round(seconds * 1000)
    hh, rem = divmod(ms_total, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _normalize_word(text: str) -> str:
    return text.strip().lower().strip(".,!?;:…—-'\"")


def dedupe_overlapping_words(words: list) -> list:
    """Drop duplicate transcriptions caused by chunk overlap. When two chunks both
    transcribe the same audio, the same word appears twice at (nearly) the same
    time — keep only the first occurrence. Words are sorted by start time first."""
    words = sorted(words, key=lambda w: w["start"])
    out = []
    for w in words:
        if out:
            prev = out[-1]
            if _normalize_word(w["word"]) == _normalize_word(prev["word"]) and \
               w["start"] < prev["end"] + 0.05:
                continue
        out.append(w)
    return out


def words_to_cues(words: list, duration: float = None) -> list:
    """Group word-level timestamps into readable, lip-sync-accurate subtitle cues.

    Rules:
        - Max CUE_MAX_CHARS per cue (break long lines before they overflow).
        - Max CUE_MAX_SECONDS per cue.
        - A pause longer than CUE_BREAK_GAP starts a new cue.
        - Each cue starts exactly at its first word; it ends just after its last
          word (padded by CUE_END_PAD, but never bleeding into the next cue).
    Returns a list of {"start", "end", "text"}.
    """
    cues = []
    cur = []
    cur_start = None

    def make_cue():
        text = " ".join(x["word"] for x in cur).strip()
        start = cur_start
        end = cur[-1]["end"] + CUE_END_PAD
        if duration is not None:
            end = min(end, duration)
        return {"start": start, "end": end, "text": text}

    for w in words:
        wtext = w["word"].strip()
        if not wtext:
            continue
        start, end = w["start"], w["end"]
        if cur:
            gap = start - cur[-1]["end"]
            line_len = len(" ".join(x["word"] for x in cur))
            too_long = line_len + 1 + len(wtext) > CUE_MAX_CHARS
            too_long_dur = end - cur_start > CUE_MAX_SECONDS
            if too_long or too_long_dur or gap > CUE_BREAK_GAP:
                cues.append(make_cue())
                cur, cur_start = [], None
        if cur_start is None:
            cur_start = start
        cur.append(w)

    if cur:
        cues.append(make_cue())

    # Never let a cue overlap the next one, and keep a minimum display time.
    for i, cue in enumerate(cues):
        end = cue["end"]
        if i + 1 < len(cues):
            end = min(end, cues[i + 1]["start"] - 0.05)
        cue["end"] = max(end, cue["start"] + 0.1)

    return [c for c in cues if c["text"]]


def segments_to_cues(segments: list, duration: float = None) -> list:
    """Fallback when word-level timestamps are missing: split each whisper segment
    into pseudo-words with time-proportional positions, then reuse the cue builder."""
    words = []
    for seg in segments:
        seg_text = (seg.get("text") or "").strip()
        if not seg_text:
            continue
        seg_dur = seg["end"] - seg["start"]
        tokens = seg_text.split()
        per = seg_dur / len(tokens) if seg_dur > 0 else 0.35
        for j, tok in enumerate(tokens):
            words.append({
                "word": tok,
                "start": seg["start"] + j * per,
                "end": seg["start"] + (j + 1) * per,
            })
    return words_to_cues(words, duration=duration)


def cues_to_srt(cues: list) -> str:
    lines = []
    for i, cue in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{format_srt_timestamp(cue['start'])} --> {format_srt_timestamp(cue['end'])}")
        lines.append(cue["text"])
        lines.append("")  # blank line between cues
    return "\n".join(lines)


def words_to_plain_text(words: list) -> str:
    """Reconstruct a clean reading transcript from word tokens (keeps punctuation
    attached to words, drops nothing)."""
    return " ".join(w["word"] for w in words).strip()


def transcribe_one(client, source: Path, txt_path: Path, srt_path, lang, workers: int, chunk_minutes: int,
                   want_srt: bool, chunk_work_dir: Path, overlap_seconds: float = CHUNK_OVERLAP_SECONDS):
    duration = get_duration_seconds(source)
    print(f"\n{source.name} — {duration/60:.1f} min")

    # Visible working folder for this file's compressed/split audio — cleaned up when done.
    chunk_work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- Fast path: .txt only, no timestamps needed. Kept exactly as-is. ---
        if not want_srt:
            if duration <= LONG_FILE_THRESHOLD_SECONDS:
                print("  Short file — compressing and uploading...")
                compressed_path = chunk_work_dir / f"{source.stem}.mp3"
                compress_to_mp3(source, compressed_path)
                text, _, _ = transcribe_bytes(
                    client, compressed_path.name, compressed_path.read_bytes(), lang, want_words=False
                )
                txt_path.write_text(text.strip(), encoding="utf-8")
                print(f"  Done -> {txt_path.name}")
                return

            print(f"  Long file — compressing, splitting into {chunk_minutes}-min chunks, {workers} parallel workers...")
            compressed_path = chunk_work_dir / f"{source.stem}_compressed.mp3"
            compress_to_mp3(source, compressed_path)

            chunk_seconds = chunk_minutes * 60
            chunks = extract_and_chunk(compressed_path, chunk_work_dir, chunk_seconds, overlap_seconds=0.0)
            print(f"  {len(chunks)} chunks created. Uploading...")

            results = {}  # chunk name -> (text, segments, words)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for c, _ in chunks:
                    data = c.read_bytes()
                    futures[executor.submit(transcribe_bytes, client, c.name, data, lang, False)] = c.name
                done_count = 0
                for future in concurrent.futures.as_completed(futures):
                    name = futures[future]
                    results[name] = future.result()
                    done_count += 1
                    print(f"    [{done_count}/{len(chunks)}] done: {name}")

            chunk_texts = []
            for c, _ in chunks:
                text, _, _ = results.get(c.name, (f"[missing chunk {c.name}]", [], []))
                chunk_texts.append(text.strip())

            full_text = " ".join(t for t in chunk_texts if t)
            txt_path.write_text(full_text, encoding="utf-8")
            print(f"  Done -> {txt_path.name}")
            return

        # --- SRT path: word-level timestamps, overlapping chunks, engineered cues. ---
        if duration <= LONG_FILE_THRESHOLD_SECONDS:
            print("  Short file — compressing and uploading (word-level timestamps)...")
            compressed_path = chunk_work_dir / f"{source.stem}.mp3"
            compress_to_mp3(source, compressed_path)

            text, segments, words = transcribe_bytes(
                client, compressed_path.name, compressed_path.read_bytes(), lang, want_words=True
            )
            words = dedupe_overlapping_words(words)
            cues = words_to_cues(words, duration=duration) if words else segments_to_cues(segments, duration=duration)
            if not cues and text:
                cues = [{"start": 0.0, "end": duration, "text": text}]

            txt_path.write_text(words_to_plain_text(words) if words else text.strip(), encoding="utf-8")
            srt_path.write_text(cues_to_srt(cues), encoding="utf-8")
            print(f"  Done -> {txt_path.name}, {srt_path.name}")
            return

        print(f"  Long file — compressing, splitting into {chunk_minutes}-min chunks with {overlap_seconds:.0f}s overlap, "
              f"{workers} parallel workers...")
        compressed_path = chunk_work_dir / f"{source.stem}_compressed.mp3"
        compress_to_mp3(source, compressed_path)

        chunk_seconds = chunk_minutes * 60
        chunks = extract_and_chunk(compressed_path, chunk_work_dir, chunk_seconds, overlap_seconds)
        print(f"  {len(chunks)} chunks created. Uploading...")

        results = {}  # chunk name -> (text, segments, words)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for c, _ in chunks:
                data = c.read_bytes()
                futures[executor.submit(transcribe_bytes, client, c.name, data, lang, True)] = c.name
            done_count = 0
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                results[name] = future.result()
                done_count += 1
                print(f"    [{done_count}/{len(chunks)}] done: {name}")

        # Stitch every chunk into the source timeline, then dedupe overlap duplicates.
        all_words = []
        all_segments = []
        chunk_texts = []
        for c, offset in chunks:
            text, segments, words = results.get(c.name, (f"[missing chunk {c.name}]", [], []))
            if words:
                for w in words:
                    all_words.append({
                        "start": max(0.0, w["start"] + offset),
                        "end": w["end"] + offset,
                        "word": w["word"],
                    })
            if segments:
                for seg in segments:
                    all_segments.append({
                        "start": max(0.0, seg["start"] + offset),
                        "end": seg["end"] + offset,
                        "text": seg["text"],
                    })
            chunk_texts.append(text.strip())

        all_words = dedupe_overlapping_words(all_words)
        if all_words:
            cues = words_to_cues(all_words, duration=duration)
            txt = words_to_plain_text(all_words)
        else:
            cues = segments_to_cues(all_segments, duration=duration)
            txt = " ".join(t for t in chunk_texts if t)
        if not cues and txt:
            cues = [{"start": 0.0, "end": duration, "text": txt}]

        txt_path.write_text(txt, encoding="utf-8")
        srt_path.write_text(cues_to_srt(cues), encoding="utf-8")
        print(f"  Done -> {txt_path.name}, {srt_path.name}")
    finally:
        # Auto-delete this file's chunk/compressed working folder now that it's done.
        shutil.rmtree(chunk_work_dir, ignore_errors=True)


def collect_files(root: Path):
    """Recursively find audio/video files under root, returning (file_path, relative_subdir) pairs
    so subfolder structure can be mirrored into transcripts/ and chunks/."""
    results = []
    for entry in sorted(root.rglob("*")):
        if entry.is_file() and entry.suffix.lower() in AUDIO_VIDEO_EXTENSIONS:
            rel_dir = entry.parent.relative_to(root)
            results.append((entry, rel_dir))
    return results


def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Transcribe one file or a whole folder, short or long, via Groq")
    parser.add_argument("input_path", help="A single audio/video file, or a folder to batch-process")
    parser.add_argument("--transcripts-dir", default="transcripts", help="Where output goes (default: transcripts/)")
    parser.add_argument("--chunks-dir", default="chunks", help="Where working compressed/split audio goes (default: chunks/)")
    parser.add_argument("--chunk-minutes", type=int, default=15, help="Chunk length for long files (default: 15)")
    parser.add_argument("--workers", type=int, default=5, help="Parallel upload workers for long files (default: 5)")
    parser.add_argument("--lang", default=None, help="Language code, e.g. 'en' (optional — auto-detects if omitted)")
    parser.add_argument("--force", action="store_true", help="Re-transcribe even if a transcript already exists")
    parser.add_argument("--srt", action="store_true", help="Also generate a timestamped .srt subtitle file (off by default — .txt only)")
    parser.add_argument("--overlap", type=float, default=CHUNK_OVERLAP_SECONDS,
                        help=f"Overlap in seconds between chunks so words aren't cut mid-word (SRT mode, default: {CHUNK_OVERLAP_SECONDS:.0f})")
    args = parser.parse_args()

    source = Path(args.input_path)
    if not source.exists():
        matched = [
            p for p in source.parent.glob(source.name + ".*")
            if p.suffix.lower() in AUDIO_VIDEO_EXTENSIONS
        ] if source.parent.exists() else []
        if matched:
            source = matched[0]
        else:
            print(f"ERROR: not found: {source}", file=sys.stderr)
            sys.exit(1)

    transcripts_root = Path(args.transcripts_dir)
    chunks_root = Path(args.chunks_dir)
    transcripts_root.mkdir(exist_ok=True)
    chunks_root.mkdir(exist_ok=True)

    if source.is_dir():
        files = collect_files(source)
        if not files:
            print(f"No audio/video files found in {source}/ (including subfolders)")
            return
        print(f"Found {len(files)} file(s) in {source}/ (including subfolders)")
    else:
        files = [(source, Path("."))]

    client = get_client()
    processed, skipped = 0, 0

    for f, rel_dir in files:
        out_dir = transcripts_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / f"transcript - {f.stem}.txt"
        srt_path = out_dir / f"subtitle - {f.stem}.srt"
        needed = [txt_path] + ([srt_path] if args.srt else [])
        if all(p.exists() for p in needed) and not args.force:
            print(f"Skipping {f.name} — already exists (use --force to redo)")
            skipped += 1
            continue

        dir_name = hashlib.md5(str(f.absolute()).encode("utf-8")).hexdigest()
        chunk_work_dir = chunks_root / dir_name
        transcribe_one(client, f, txt_path, srt_path, args.lang, args.workers, args.chunk_minutes,
                       args.srt, chunk_work_dir, args.overlap)
        processed += 1

    print(f"\n--- Summary: {processed} transcribed, {skipped} skipped (already done) ---")


if __name__ == "__main__":
    main()