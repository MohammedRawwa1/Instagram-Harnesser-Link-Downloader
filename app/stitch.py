"""The lip-sync brain.

Turns per-chunk word timestamps into perfectly synced subtitles:

1. merge_chunk_results — maps every chunk's words/segments onto the source
   timeline using the chunk offsets produced by split_into_chunks().
2. dedupe_overlapping_words — removes the duplicate transcriptions that come
   from chunk overlap (the same word heard by two chunks).
3. words_to_cues — groups words into readable, lip-sync-accurate subtitle
   cues (max chars, max on-screen time, breaks at natural pauses, end-padding
   that never bleeds into the next cue).
4. cues_to_srt / words_to_plain_text / cues_to_timings — emit the artifacts.
"""

import json
from pathlib import Path

CUE_MAX_CHARS = 42        # max chars per cue (≈2 lines of 21)
CUE_MAX_SECONDS = 7.0     # never leave a cue on screen longer than this
CUE_BREAK_GAP = 0.35      # a pause longer than this starts a new cue
CUE_END_PAD = 0.30        # keep the last word visible a beat longer


# --------------------------------------------------------------------------
# 1. Merge chunk results onto the source timeline
# --------------------------------------------------------------------------

def merge_chunk_results(
    chunks: list[tuple[Path, float]],
    results: list[dict],
) -> tuple[list[dict], list[dict], list[str]]:
    """Combine per-chunk Groq responses into global-time words/segments.

    Returns (words, segments, chunk_texts) with all timestamps in the source
    timeline. `chunks` is [(path, global_offset)] from split_into_chunks().
    """
    all_words: list[dict] = []
    all_segments: list[dict] = []
    chunk_texts: list[str] = []

    for (path, offset), result in zip(chunks, results):
        text = (result.get("text") or "").strip()
        for w in result.get("words") or []:
            w_start, w_end, w_word = float(w["start"]), float(w["end"]), str(w["word"]).strip()
            if not w_word or w_end <= w_start:
                continue
            all_words.append({
                "start": max(0.0, w_start + offset),
                "end": w_end + offset,
                "word": w_word,
            })
        for seg in result.get("segments") or []:
            all_segments.append({
                "start": max(0.0, float(seg["start"]) + offset),
                "end": float(seg["end"]) + offset,
                "text": (seg.get("text") or "").strip(),
            })
        chunk_texts.append(text)

    return all_words, all_segments, chunk_texts


# --------------------------------------------------------------------------
# 2. Dedupe overlap double-transcriptions
# --------------------------------------------------------------------------

def _normalize_word(text: str) -> str:
    return text.strip().lower().strip(".,!?;:…—-'\"")


def dedupe_overlapping_words(words: list[dict]) -> list[dict]:
    """Drop duplicate words caused by chunk overlap: the same word transcribed
    by two chunks at (nearly) the same time is kept only once."""
    words = sorted(words, key=lambda w: w["start"])
    out: list[dict] = []
    for w in words:
        if out:
            prev = out[-1]
            if _normalize_word(w["word"]) == _normalize_word(prev["word"]) and \
               w["start"] < prev["end"] + 0.05:
                continue
        out.append(w)
    return out


# --------------------------------------------------------------------------
# 3. Cue engineering
# --------------------------------------------------------------------------

def words_to_cues(words: list[dict], duration: float | None = None) -> list[dict]:
    """Group word timestamps into readable, lip-sync-accurate cues."""
    cues: list[dict] = []
    cur: list[dict] = []
    cur_start: float | None = None

    def make_cue() -> dict:
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


def segments_to_cues(segments: list[dict], duration: float | None = None) -> list[dict]:
    """Fallback when word timestamps are missing: split each whisper segment
    into pseudo-words with time-proportional positions, then reuse the builder."""
    words: list[dict] = []
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


# --------------------------------------------------------------------------
# 4. Artifacts: SRT, TXT, timings
# --------------------------------------------------------------------------

def format_srt_timestamp(seconds: float) -> str:
    ms_total = round(seconds * 1000)
    hh, rem = divmod(ms_total, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def cues_to_srt(cues: list[dict]) -> str:
    lines: list[str] = []
    for i, cue in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{format_srt_timestamp(cue['start'])} --> {format_srt_timestamp(cue['end'])}")
        lines.append(cue["text"])
        lines.append("")
    return "\n".join(lines)


def words_to_plain_text(words: list[dict]) -> str:
    return " ".join(w["word"] for w in words).strip()


def cues_to_timings(cues: list[dict], words: list[dict], duration: float) -> str:
    """JSON artifact with every word and cue timestamped — for editors/markers."""
    payload = {
        "duration": round(duration, 3),
        "word_count": len(words),
        "cue_count": len(cues),
        "words": [{"w": w["word"], "s": round(w["start"], 3), "e": round(w["end"], 3)} for w in words],
        "cues": [{"s": round(c["start"], 3), "e": round(c["end"], 3), "t": c["text"]} for c in cues],
    }
    return json.dumps(payload, indent=2)


def build_artifacts(
    chunks: list[tuple[Path, float]],
    results: list[dict],
    duration: float,
) -> tuple[str, str, str]:
    """Full stitch: merge → dedupe → cues → (srt, txt, timings)."""
    all_words, all_segments, chunk_texts = merge_chunk_results(chunks, results)
    all_words = dedupe_overlapping_words(all_words)

    if all_words:
        cues = words_to_cues(all_words, duration=duration)
        txt = words_to_plain_text(all_words)
    else:
        cues = segments_to_cues(all_segments, duration=duration)
        txt = " ".join(t for t in chunk_texts if t)
    if not cues and txt:
        cues = [{"start": 0.0, "end": duration, "text": txt}]

    srt = cues_to_srt(cues)
    timings = cues_to_timings(cues, all_words, duration)
    return srt, txt, timings