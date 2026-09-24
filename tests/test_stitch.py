"""Unit tests for the lip-sync brain (stitch.py)."""

import json

from app import stitch


def _words(*pairs):
    """Build a word stream from (text, start, end) tuples."""
    return [{"word": t, "start": s, "end": e} for t, s, e in pairs]


def test_cues_break_at_pauses():
    words = []
    t = 0.0
    for phrase in ["Hello everyone,", "and welcome to", "the show."]:
        for w in phrase.split():
            words.append({"word": w, "start": t, "end": t + 0.4})
            t += 0.45
        t += 0.8  # pause between phrases
    words.append({"word": "Bye!", "start": t, "end": t + 0.4})

    cues = stitch.words_to_cues(words, duration=10.0)
    texts = [c["text"] for c in cues]
    assert texts == ["Hello everyone,", "and welcome to", "the show.", "Bye!"]
    for c in cues:
        assert c["end"] >= c["start"] + 0.1
        assert c["text"]


def test_dedupe_overlap_duplicates():
    dupes = [
        {"start": 5.0, "end": 5.4, "word": "subject"},
        {"start": 5.1, "end": 5.5, "word": "subject"},
    ]
    assert len(stitch.dedupe_overlapping_words(dupes)) == 1


def test_segment_fallback_splits_long_segments():
    segs = [{"start": 10.0, "end": 16.0, "text": "one two three four five six seven eight nine ten"}]
    cues = stitch.segments_to_cues(segs, duration=20.0)
    assert len(cues) >= 1
    assert all(c["text"] for c in cues)


def test_char_and_duration_caps():
    long_run = [{"word": "word", "start": i * 0.4, "end": i * 0.4 + 0.35} for i in range(40)]
    cues = stitch.words_to_cues(long_run, duration=100.0)
    assert all(len(c["text"]) <= stitch.CUE_MAX_CHARS + 8 for c in cues)
    assert all(c["end"] - c["start"] <= stitch.CUE_MAX_SECONDS + 0.5 for c in cues)


def test_no_overlapping_cues():
    long_run = [{"word": "word", "start": i * 0.4, "end": i * 0.4 + 0.35} for i in range(40)]
    cues = stitch.words_to_cues(long_run, duration=100.0)
    for a, b in zip(cues, cues[1:]):
        assert a["end"] <= b["start"] + 1e-6


def test_srt_format():
    srt = stitch.cues_to_srt([{"start": 0.0, "end": 1.234, "text": "Hi there"}])
    lines = srt.splitlines()
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:01,234"
    assert lines[2] == "Hi there"


def test_build_artifacts_offsets_chunks():
    """Two overlapping chunks must merge into one continuous timeline."""
    chunk_a = {
        "text": "Hello world.",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hello world."}],
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.4},
            {"word": "world.", "start": 0.5, "end": 1.0},
        ],
    }
    chunk_b = {
        "text": "world. Goodbye.",
        "segments": [{"start": 0.5, "end": 2.0, "text": "world. Goodbye."}],
        "words": [
            {"word": "world.", "start": 0.5, "end": 1.0},     # overlap duplicate
            {"word": "Goodbye.", "start": 1.1, "end": 2.0},
        ],
    }
    # chunk_b covers [8, 10) → offset 8.0 (chunk_seconds=10, overlap=2)
    chunks = [("chunk_0000.mp3", 0.0), ("chunk_0001.mp3", 8.0)]
    srt, txt, timings = stitch.build_artifacts(chunks, [chunk_a, chunk_b], duration=10.0)

    assert "world." in txt
    assert "Hello" in txt and "Goodbye." in txt
    parsed = json.loads(timings)
    # words sorted into source timeline
    starts = [w["s"] for w in parsed["words"]]
    assert starts == sorted(starts)
    assert any(8.0 <= w["s"] <= 9.0 for w in parsed["words"])  # chunk_b shifted
    assert srt.splitlines()[0] == "1"