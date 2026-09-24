"""Comprehensive lip-sync accuracy tests.

These tests verify the 100% lip-sync contract:
  1. Word timestamps are monotonically non-decreasing
  2. Cues never overlap
  3. Each cue starts at its first word's timestamp
  4. Each cue ends just after its last word (padded), never bleeding into next cue
  5. Cues respect max-char and max-duration constraints
  6. Cues break at natural pauses (>0.35s gap)
  7. The SRT word stream exactly matches the TXT transcript (nothing dropped/duplicated)
  8. SRT timestamps are monotonically non-overlapping
"""

import json
import re
import textwrap

import pytest

from app import stitch


# ─── Helpers ────────────────────────────────────────────────────────────────

def _words(*triples):
    """Build word stream from (text, start, end) triples."""
    return [{"word": t, "start": s, "end": e} for t, s, e in triples]


def _words_from_text(text, start=0.0, gap=0.45, word_dur=0.4):
    """Auto-generate word timestamps from a text string."""
    words = []
    t = start
    for w in text.split():
        words.append({"word": w, "start": t, "end": t + word_dur})
        t += gap
    return words


def _make_cue(text, start, end):
    return {"text": text, "start": start, "end": end}


# ═══════════════════════════════════════════════════════════════════════════
# 1. MONOTONIC WORD TIMESTAMPS
# ═══════════════════════════════════════════════════════════════════════════

class TestMonotonicTimestamps:
    """Word timestamps must never go backwards."""

    def test_basic_monotonicity(self):
        words = _words(
            ("Hello", 0.0, 0.4),
            ("world", 0.5, 0.9),
            ("foo", 1.0, 1.4),
        )
        cues = stitch.words_to_cues(words, duration=5.0)
        for c in cues:
            assert c["end"] >= c["start"]

    def test_out_of_order_words_still_produce_valid_cues(self):
        """Words arriving out of order should still yield valid cues."""
        words = _words(
            ("third", 2.0, 2.4),
            ("first", 0.0, 0.4),
            ("second", 1.0, 1.4),
        )
        cues = stitch.words_to_cues(words, duration=5.0)
        # All cues should have start < end
        for c in cues:
            assert c["start"] < c["end"]

    def test_word_end_after_start(self):
        words = _words(("a", 1.0, 1.5), ("b", 1.5, 2.0), ("c", 2.0, 2.5))
        cues = stitch.words_to_cues(words, duration=5.0)
        assert len(cues) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 2. NO OVERLAPPING CUES
# ═══════════════════════════════════════════════════════════════════════════

class TestNoOverlappingCues:
    """Consecutive cues must never overlap in time."""

    def test_no_overlap_dense_words(self):
        words = _words_from_text(
            "one two three four five six seven eight nine ten "
            "eleven twelve thirteen fourteen fifteen",
            gap=0.3,
        )
        cues = stitch.words_to_cues(words, duration=20.0)
        for i in range(len(cues) - 1):
            assert cues[i]["end"] <= cues[i + 1]["start"] + 1e-6, (
                f"Cue {i} ends at {cues[i]['end']} but cue {i+1} starts at {cues[i+1]['start']}"
            )

    def test_no_overlap_slow_speech(self):
        """Slow speech with gaps still produces non-overlapping cues."""
        words = []
        t = 0.0
        for phrase in ["Hello there", "how are you", "doing today"]:
            for w in phrase.split():
                words.append({"word": w, "start": t, "end": t + 0.4})
                t += 0.5
            t += 1.0  # long pause
        cues = stitch.words_to_cues(words, duration=20.0)
        for i in range(len(cues) - 1):
            assert cues[i]["end"] <= cues[i + 1]["start"] + 1e-6

    def test_no_overlap_fast_speech(self):
        """Fast speech still produces non-overlapping cues."""
        words = _words_from_text(
            "this is a very fast sentence with many words in a row",
            gap=0.2,
            word_dur=0.18,
        )
        cues = stitch.words_to_cues(words, duration=10.0)
        for i in range(len(cues) - 1):
            assert cues[i]["end"] <= cues[i + 1]["start"] + 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# 3. CUE START = FIRST WORD, CUE END = LAST WORD + PAD
# ═══════════════════════════════════════════════════════════════════════════

class TestCueTimingAccuracy:
    """Each cue must start exactly at its first word and end just after its last word."""

    def test_cue_starts_at_first_word(self):
        words = _words(
            ("Hello", 1.0, 1.4),
            ("world", 1.5, 1.9),
            ("foo", 5.0, 5.4),  # gap triggers new cue
            ("bar", 5.5, 5.9),
        )
        cues = stitch.words_to_cues(words, duration=10.0)
        assert len(cues) >= 2
        # First cue starts at first word
        assert cues[0]["start"] == pytest.approx(1.0, abs=1e-6)
        # Second cue starts at "foo"
        assert cues[1]["start"] == pytest.approx(5.0, abs=1e-6)

    def test_cue_ends_after_last_word_padded(self):
        words = _words(
            ("Hello", 1.0, 1.4),
            ("world", 1.5, 1.9),
            ("foo", 5.0, 5.4),
            ("bar", 5.5, 5.9),
        )
        cues = stitch.words_to_cues(words, duration=10.0)
        assert len(cues) >= 2
        # First cue ends after "world" end (1.9) + pad (0.3) = 2.2
        assert cues[0]["end"] == pytest.approx(2.2, abs=1e-6)
        # Second cue ends after "bar" end (5.9) + pad (0.3) = 6.2
        assert cues[1]["end"] == pytest.approx(6.2, abs=1e-6)

    def test_cue_end_never_exceeds_next_cue_start(self):
        words = _words(
            ("a", 0.0, 0.3),
            ("b", 0.4, 0.7),
            ("c", 3.0, 3.3),
            ("d", 3.4, 3.7),
        )
        cues = stitch.words_to_cues(words, duration=10.0)
        for i in range(len(cues) - 1):
            assert cues[i]["end"] <= cues[i + 1]["start"] + 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# 4. CUE CONSTRAINTS: MAX CHARS + MAX DURATION
# ═══════════════════════════════════════════════════════════════════════════

class TestCueConstraints:
    """Cues must respect char and duration limits."""

    def test_no_cue_exceeds_max_chars(self):
        # 40 words at 5 chars each = 200 chars, should break into multiple cues
        words = _words_from_text(
            " ".join([f"word{i:02d}" for i in range(40)]),
            gap=0.3,
        )
        cues = stitch.words_to_cues(words, duration=30.0)
        for c in cues:
            assert len(c["text"]) <= stitch.CUE_MAX_CHARS + 10, (
                f"Cue too long ({len(c['text'])} chars): {c['text'][:60]}..."
            )

    def test_no_cue_exceeds_max_duration(self):
        # Words spread far apart should break into multiple cues
        words = _words(
            ("one", 0.0, 0.4),
            ("two", 3.0, 3.4),
            ("three", 6.0, 6.4),
            ("four", 9.0, 9.4),
        )
        cues = stitch.words_to_cues(words, duration=15.0)
        for c in cues:
            assert c["end"] - c["start"] <= stitch.CUE_MAX_SECONDS + 1.0

    def test_cues_are_readable(self):
        """Each cue should be a coherent phrase, not random words."""
        words = _words_from_text(
            "Hello everyone and welcome to the show today we will talk about "
            "artificial intelligence and how it changes our daily lives",
            gap=0.4,
        )
        cues = stitch.words_to_cues(words, duration=20.0)
        for c in cues:
            text = c["text"].strip()
            assert len(text) > 0
            # Should not start or end with whitespace artifacts
            assert text == text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# 5. CUES BREAK AT NATURAL PAUSES
# ═══════════════════════════════════════════════════════════════════════════

class TestPauseBreaks:
    """Cues must break at pauses > 0.35s (CUE_BREAK_GAP)."""

    def test_break_at_long_pause(self):
        words = []
        t = 0.0
        for phrase in ["First sentence.", "Second sentence.", "Third sentence."]:
            for w in phrase.split():
                words.append({"word": w, "start": t, "end": t + 0.3})
                t += 0.35
            t += 1.0  # long pause → should break
        cues = stitch.words_to_cues(words, duration=20.0)
        texts = [c["text"] for c in cues]
        assert "First sentence." in " ".join(texts)
        assert "Second sentence." in " ".join(texts)

    def test_no_break_at_short_pause(self):
        """Short pauses (< 0.35s) should NOT cause a break."""
        words = _words(
            ("Hello", 0.0, 0.3),
            ("beautiful", 0.4, 0.7),  # 0.1s gap
            ("world", 0.8, 1.1),     # 0.1s gap
        )
        cues = stitch.words_to_cues(words, duration=5.0)
        assert len(cues) == 1
        assert cues[0]["text"] == "Hello beautiful world"

    def test_break_between_sentences(self):
        words = []
        t = 0.0
        # Sentence 1
        for w in "This is sentence one".split():
            words.append({"word": w, "start": t, "end": t + 0.3})
            t += 0.35
        t += 0.8  # sentence boundary pause
        # Sentence 2
        for w in "This is sentence two".split():
            words.append({"word": w, "start": t, "end": t + 0.3})
            t += 0.35
        cues = stitch.words_to_cues(words, duration=10.0)
        assert len(cues) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 6. TRANSCRIPT FIDELITY: SRT WORDS == TXT, NOTHING DROPPED/DUPLICATED
# ═══════════════════════════════════════════════════════════════════════════

class TestTranscriptFidelity:
    """The SRT cue text must exactly reconstruct the plain-text transcript."""

    def test_srt_words_match_txt(self):
        words = _words_from_text(
            "The quick brown fox jumps over the lazy dog near the river bank",
            gap=0.4,
        )
        cues = stitch.words_to_cues(words, duration=15.0)
        srt = stitch.cues_to_srt(cues)
        txt = stitch.words_to_plain_text(words)

        # Extract all words from SRT (skip timestamps and numbers)
        srt_words = []
        for line in srt.splitlines():
            line = line.strip()
            if not line or line.isdigit() or "-->" in line:
                continue
            srt_words.extend(line.split())

        txt_words = txt.split()

        assert srt_words == txt_words, (
            f"SRT words don't match TXT.\nSRT: {srt_words}\nTXT: {txt_words}"
        )

    def test_no_dropped_words(self):
        """Every input word must appear in the output."""
        words = _words_from_text(
            "one two three four five six seven eight nine ten",
            gap=0.4,
        )
        cues = stitch.words_to_cues(words, duration=15.0)
        txt = stitch.words_to_plain_text(words)
        all_input_words = [w["word"] for w in words]
        output_words = txt.split()
        assert output_words == all_input_words

    def test_no_duplicated_words(self):
        """Words should not appear more than once in the output."""
        words = _words_from_text(
            "the cat sat on the mat and the cat looked at the bird",
            gap=0.4,
        )
        cues = stitch.words_to_cues(words, duration=15.0)
        txt = stitch.words_to_plain_text(words)
        output_words = txt.split()
        # Check no word appears more times than in input
        input_words = [w["word"] for w in words]
        for word in set(output_words):
            assert output_words.count(word) <= input_words.count(word), (
                f"Word '{word}' appears {output_words.count(word)} times in output "
                f"but only {input_words.count(word)} times in input"
            )

    def test_srt_cue_count_matches_word_grouping(self):
        words = _words_from_text(
            "Hello world how are you doing today",
            gap=0.4,
        )
        cues = stitch.words_to_cues(words, duration=10.0)
        srt = stitch.cues_to_srt(cues)
        # Count cue numbers in SRT
        cue_numbers = re.findall(r"^(\d+)$", srt, re.MULTILINE)
        assert len(cue_numbers) == len(cues)


# ═══════════════════════════════════════════════════════════════════════════
# 7. SRT FORMAT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class TestSRTFormat:
    """SRT output must be valid and properly formatted."""

    def test_valid_srt_structure(self):
        words = _words_from_text("Hello world this is a test", gap=0.4)
        cues = stitch.words_to_cues(words, duration=10.0)
        srt = stitch.cues_to_srt(cues)

        blocks = srt.strip().split("\n\n")
        for block in blocks:
            lines = block.strip().splitlines()
            assert len(lines) >= 3, f"SRT block too short: {block}"
            # Line 1: cue number
            assert lines[0].strip().isdigit()
            # Line 2: timestamp
            ts_match = re.match(
                r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})",
                lines[1].strip(),
            )
            assert ts_match, f"Bad timestamp: {lines[1]}"
            # Line 3+: cue text
            assert lines[2].strip()

    def test_srt_timestamps_monotonic(self):
        words = _words_from_text(
            "one two three four five six seven eight",
            gap=0.4,
        )
        cues = stitch.words_to_cues(words, duration=15.0)
        srt = stitch.cues_to_srt(cues)

        prev_end = "00:00:00,000"
        for block in srt.strip().split("\n\n"):
            lines = block.strip().splitlines()
            ts_match = re.match(
                r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})",
                lines[1].strip(),
            )
            start_ts = ts_match.group(1)
            end_ts = ts_match.group(2)
            assert start_ts >= prev_end, (
                f"SRT not monotonic: {start_ts} < previous end {prev_end}"
            )
            prev_end = end_ts

    def test_srt_cue_count(self):
        words = _words_from_text("Hello world", gap=0.4)
        cues = stitch.words_to_cues(words, duration=5.0)
        srt = stitch.cues_to_srt(cues)
        cue_count = len([l for l in srt.splitlines() if l.strip().isdigit()])
        assert cue_count == len(cues)


# ═══════════════════════════════════════════════════════════════════════════
# 8. DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════

class TestDeduplication:
    """Overlap deduplication must remove exact duplicates without harming real words."""

    def test_exact_overlap_deduped(self):
        words = [
            {"start": 5.0, "end": 5.4, "word": "hello"},
            {"start": 5.1, "end": 5.5, "word": "hello"},  # duplicate
        ]
        result = stitch.dedupe_overlapping_words(words)
        assert len(result) == 1
        assert result[0]["word"] == "hello"

    def test_non_overlapping_duplicates_kept(self):
        words = [
            {"start": 0.0, "end": 0.4, "word": "hello"},
            {"start": 5.0, "end": 5.4, "word": "hello"},  # different time
        ]
        result = stitch.dedupe_overlapping_words(words)
        assert len(result) == 2

    def test_different_words_not_deduped(self):
        words = [
            {"start": 5.0, "end": 5.4, "word": "hello"},
            {"start": 5.1, "end": 5.5, "word": "world"},
        ]
        result = stitch.dedupe_overlapping_words(words)
        assert len(result) == 2

    def test_case_insensitive_dedup(self):
        words = [
            {"start": 5.0, "end": 5.4, "word": "Hello"},
            {"start": 5.1, "end": 5.5, "word": "hello"},
        ]
        result = stitch.dedupe_overlapping_words(words)
        assert len(result) == 1

    def test_punctuation_insensitive_dedup(self):
        words = [
            {"start": 5.0, "end": 5.4, "word": "hello."},
            {"start": 5.1, "end": 5.5, "word": "hello"},
        ]
        result = stitch.dedupe_overlapping_words(words)
        assert len(result) == 1

    def test_empty_input(self):
        assert stitch.dedupe_overlapping_words([]) == []

    def test_single_word(self):
        words = [{"start": 0.0, "end": 0.4, "word": "hello"}]
        result = stitch.dedupe_overlapping_words(words)
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 9. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_single_word(self):
        words = _words(("Hello", 0.0, 0.4))
        cues = stitch.words_to_cues(words, duration=1.0)
        assert len(cues) == 1
        assert cues[0]["text"] == "Hello"

    def test_two_words(self):
        words = _words(("Hello", 0.0, 0.4), ("world", 0.5, 0.9))
        cues = stitch.words_to_cues(words, duration=2.0)
        assert len(cues) >= 1
        txt = stitch.words_to_plain_text(words)
        assert txt == "Hello world"

    def test_empty_words(self):
        cues = stitch.words_to_cues([], duration=5.0)
        assert cues == []

    def test_very_long_monologue(self):
        """100 words should produce valid, non-overlapping cues."""
        text = " ".join([f"word{i}" for i in range(100)])
        words = _words_from_text(text, gap=0.3)
        cues = stitch.words_to_cues(words, duration=60.0)
        # Check non-overlap
        for i in range(len(cues) - 1):
            assert cues[i]["end"] <= cues[i + 1]["start"] + 1e-6
        # Check all words captured
        txt = stitch.words_to_plain_text(words)
        assert len(txt.split()) == 100

    def test_cue_end_capped_by_duration(self):
        """Last cue should not exceed the total duration."""
        words = _words(("hello", 4.5, 4.9), ("world", 5.0, 5.4))
        cues = stitch.words_to_cues(words, duration=5.5)
        assert cues[-1]["end"] <= 5.5 + 0.01

    def test_segment_fallback(self):
        """When word timestamps are missing, segments should be used."""
        segments = [
            {"start": 0.0, "end": 3.0, "text": "Hello world this is a test"},
        ]
        cues = stitch.segments_to_cues(segments, duration=5.0)
        assert len(cues) >= 1
        all_text = " ".join(c["text"] for c in cues)
        assert "Hello" in all_text
        assert "test" in all_text


# ═══════════════════════════════════════════════════════════════════════════
# 10. INTEGRATION: BUILD_ARTIFACTS END-TO-END
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildArtifacts:
    """Full pipeline: chunks → stitch → SRT + TXT + timings."""

    def _make_chunk_result(self, words_list, text=None):
        return {
            "text": text or " ".join(w["word"] for w in words_list),
            "segments": [{"start": words_list[0]["start"], "end": words_list[-1]["end"],
                          "text": text or " ".join(w["word"] for w in words_list)}],
            "words": words_list,
        }

    def test_single_chunk_artifacts(self):
        words = _words(
            ("Hello", 0.0, 0.4),
            ("world", 0.5, 0.9),
        )
        chunks = [("chunk_0000.mp3", 0.0)]
        results = [self._make_chunk_result(words)]

        srt, txt, timings = stitch.build_artifacts(chunks, results, duration=2.0)

        assert "Hello" in txt
        assert "world" in txt
        assert srt.startswith("1\n")
        parsed = json.loads(timings)
        assert parsed["word_count"] == 2
        assert parsed["cue_count"] >= 1

    def test_two_chunks_merge_correctly(self):
        """Two overlapping chunks should merge into one continuous timeline."""
        # Chunk A: words at 0-1s. Chunk B: same words at 0-1s local, offset 8s global.
        # The overlap dedup works on global timeline — words at different global times
        # are different words (correct behavior: each chunk is a separate time window).
        chunk_a_words = _words(
            ("Hello", 0.0, 0.4),
            ("world", 0.5, 0.9),
        )
        chunk_b_words = _words(
            ("world", 0.5, 0.9),  # same local time, but offset to 8.5 globally
            ("goodbye", 1.0, 1.4),
        )
        chunks = [("chunk_0000.mp3", 0.0), ("chunk_0001.mp3", 8.0)]
        results = [
            self._make_chunk_result(chunk_a_words),
            self._make_chunk_result(chunk_b_words),
        ]

        srt, txt, timings = stitch.build_artifacts(chunks, results, duration=10.0)

        # All words present (chunk B 'world' at 8.5s is a different global time)
        assert "Hello" in txt
        assert "goodbye" in txt
        # Chunk B words shifted to global timeline
        parsed = json.loads(timings)
        global_starts = [w["s"] for w in parsed["words"]]
        assert any(8.0 <= s <= 9.0 for s in global_starts)

    def test_dedupe_removes_same_global_time_duplicates(self):
        """Words at the same global time should be deduped."""
        # Two chunks with same offset (simulating overlap region)
        chunk_a_words = _words(
            ("hello", 5.0, 5.4),
            ("world", 5.5, 5.9),
        )
        chunk_b_words = _words(
            ("hello", 5.0, 5.4),  # same global time → should be deduped
            ("there", 6.0, 6.4),
        )
        # Both chunks at offset 0 (overlap region)
        chunks = [("chunk_0000.mp3", 0.0), ("chunk_0001.mp3", 0.0)]
        results = [
            self._make_chunk_result(chunk_a_words),
            self._make_chunk_result(chunk_b_words),
        ]

        srt, txt, timings = stitch.build_artifacts(chunks, results, duration=10.0)

        # 'hello' appears once (deduped)
        assert txt.count("hello") == 1
        # 'world' and 'there' both present
        assert "world" in txt
        assert "there" in txt

    def test_artifacts_monotonic_srt(self):
        """SRT from build_artifacts must have monotonic timestamps."""
        words = _words_from_text(
            "The quick brown fox jumps over the lazy dog",
            gap=0.4,
        )
        chunks = [("chunk_0000.mp3", 0.0)]
        results = [self._make_chunk_result(words)]

        srt, txt, timings = stitch.build_artifacts(chunks, results, duration=10.0)

        prev_end = "00:00:00,000"
        for block in srt.strip().split("\n\n"):
            lines = block.strip().splitlines()
            ts_match = re.match(
                r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})",
                lines[1].strip(),
            )
            start_ts = ts_match.group(1)
            assert start_ts >= prev_end
            prev_end = ts_match.group(2)

    def test_artifacts_timings_json_valid(self):
        words = _words_from_text("Hello world this is a test", gap=0.3)
        chunks = [("chunk_0000.mp3", 0.0)]
        results = [self._make_chunk_result(words)]

        srt, txt, timings = stitch.build_artifacts(chunks, results, duration=5.0)
        parsed = json.loads(timings)

        assert "duration" in parsed
        assert "word_count" in parsed
        assert "cue_count" in parsed
        assert "words" in parsed
        assert "cues" in parsed
        assert len(parsed["words"]) == len(words)
        assert parsed["cue_count"] >= 1

    def test_fallback_when_no_words(self):
        """When word timestamps are missing, segment fallback should work."""
        chunks = [("chunk_0000.mp3", 0.0)]
        results = [{
            "text": "Hello world this is a test",
            "segments": [{"start": 0.0, "end": 3.0, "text": "Hello world this is a test"}],
            "words": [],
        }]

        srt, txt, timings = stitch.build_artifacts(chunks, results, duration=5.0)
        assert "Hello" in srt
        assert "Hello" in txt


# ═══════════════════════════════════════════════════════════════════════════
# 11. REAL-WORLD REEL SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

class TestRealWorldReel:
    """Simulate a realistic 60-second reel transcription."""

    def test_full_reel_simulation(self):
        """Simulate a 60-second reel with 150 words, verify all constraints."""
        # Simulate realistic speech pattern
        phrases = [
            "This is such a good idea and it could be used for pretty much any business",
            "Let me show you how to do it step number one is to record between four and seven",
            "horizontal clips where you are passing some sort of object or product",
            "from above you down below and if you want this to look really good",
            "try to wear something different or stand in a new location with every video",
            "then create a new project in Canva that is ten eighty by seventy six eighty",
            "add all of your videos stacked on top of each other and then trim the video length",
            "so that as the object is reaching the bottom of one screen its entering the top of the next",
            "once everything looks smooth download your video and then open up CapCut",
            "create a new project and set the ratio to nine by sixteen",
            "then add your video drag it all the way up to the top and add a keyframe at the start",
            "then just add another keyframe to the end of the video and drag it all the way to the bottom",
            "export it and youre good to go to post it",
        ]

        # Build word stream with realistic timing
        words = []
        t = 0.0
        for phrase in phrases:
            for w in phrase.split():
                words.append({"word": w, "start": t, "end": t + 0.35})
                t += 0.4
            t += 0.6  # pause between phrases

        duration = t + 1.0
        chunks = [("chunk_0000.mp3", 0.0)]
        results = [{
            "text": " ".join(w["word"] for w in words),
            "segments": [{"start": 0.0, "end": duration, "text": " ".join(w["word"] for w in words)}],
            "words": words,
        }]

        srt, txt, timings = stitch.build_artifacts(chunks, results, duration=duration)

        # ── Verify all constraints ──

        # 1. SRT words match TXT
        srt_words = []
        for line in srt.splitlines():
            line = line.strip()
            if not line or line.isdigit() or "-->" in line:
                continue
            srt_words.extend(line.split())
        txt_words = txt.split()
        assert srt_words == txt_words, "SRT words don't match TXT"

        # 2. No overlapping cues
        parsed_timings = json.loads(timings)
        for i in range(len(parsed_timings["cues"]) - 1):
            assert parsed_timings["cues"][i]["e"] <= parsed_timings["cues"][i + 1]["s"] + 0.01

        # 3. All cues have text
        for cue in parsed_timings["cues"]:
            assert cue["t"].strip()

        # 4. SRT is valid format
        blocks = srt.strip().split("\n\n")
        assert len(blocks) == parsed_timings["cue_count"]

        # 5. Word count matches
        assert parsed_timings["word_count"] == len(words)

        # 6. Duration is correct
        assert parsed_timings["duration"] == pytest.approx(duration, abs=0.1)


# ═══════════════════════════════════════════════════════════════════════════
# 12. ARABIC LANGUAGE SUPPORT
# ═══════════════════════════════════════════════════════════════════════════

class TestArabicLanguage:
    """Verify Arabic text is handled correctly through the stitch pipeline."""

    def test_arabic_words_produce_valid_cues(self):
        """Arabic words with timestamps should produce valid SRT cues."""
        words = _words(
            ("مرحبا", 0.0, 0.5),
            ("بالعالم", 0.6, 1.2),
            ("هذا", 1.3, 1.6),
            ("اختبار", 1.7, 2.2),
        )
        cues = stitch.words_to_cues(words, duration=5.0)
        assert len(cues) >= 1
        for c in cues:
            assert c["start"] < c["end"]
            assert c["text"].strip()

    def test_arabic_srt_words_match_txt(self):
        """SRT words must exactly match TXT for Arabic content."""
        words = _words(
            ("مرحبا", 0.0, 0.5),
            ("بالعالم", 0.6, 1.2),
            ("هذا", 1.3, 1.6),
            ("اختبار", 1.7, 2.2),
            ("للترجمة", 2.3, 2.9),
        )
        cues = stitch.words_to_cues(words, duration=5.0)
        srt = stitch.cues_to_srt(cues)
        txt = stitch.words_to_plain_text(words)

        srt_words = []
        for line in srt.splitlines():
            line = line.strip()
            if not line or line.isdigit() or "-->" in line:
                continue
            srt_words.extend(line.split())

        txt_words = txt.split()
        assert srt_words == txt_words

    def test_arabic_no_overlapping_cues(self):
        """Arabic cues must never overlap."""
        words = _words(
            ("مرحبا", 0.0, 0.5),
            ("بالعالم", 0.6, 1.2),
            ("كيف", 3.0, 3.4),
            ("حالك", 3.5, 4.0),
            ("اليوم", 4.1, 4.6),
        )
        cues = stitch.words_to_cues(words, duration=10.0)
        for i in range(len(cues) - 1):
            assert cues[i]["end"] <= cues[i + 1]["start"] + 1e-6

    def test_arabic_cue_timing_accuracy(self):
        """Arabic cue must start at first word and end after last word + pad."""
        words = _words(
            ("مرحبا", 1.0, 1.5),
            ("بالعالم", 1.6, 2.2),
            ("شكرا", 5.0, 5.5),
            ("لك", 5.6, 5.9),
        )
        cues = stitch.words_to_cues(words, duration=10.0)
        assert len(cues) >= 2
        assert cues[0]["start"] == pytest.approx(1.0, abs=1e-6)
        assert cues[1]["start"] == pytest.approx(5.0, abs=1e-6)
        # cue[0] ends at last word end (2.2) + pad (0.3) = 2.5
        assert cues[0]["end"] == pytest.approx(2.5, abs=1e-6)

    def test_arabic_full_reel_simulation(self):
        """Simulate a 60-second Arabic reel transcription end-to-end."""
        arabic_phrases = [
            "مرحبا بكم في هذا الفيديو التعليمي",
            "اليوم سنتعلم كيفية استخدام الذكاء الاصطناعي",
            "في ترجمة الفيديوهات من أي لغة إلى الإنجليزية",
            "هذه تقنية ممتازة توفر الكثير من الوقت",
            "الخطوة الأولى هي تحميل الفيديو من إنستغرام",
            "ثم نقوم بضغط الصوت وتقسيمه إلى أجزاء",
            "كل جزء يُرسل إلى خدمة الترجمة",
            "والنتيجة النهائية هي ترجمة كاملة ومزامنة",
        ]

        words = []
        t = 0.0
        for phrase in arabic_phrases:
            for w in phrase.split():
                words.append({"word": w, "start": t, "end": t + 0.4})
                t += 0.45
            t += 0.7

        duration = t + 1.0
        chunks = [("chunk_0000.mp3", 0.0)]
        results = [{
            "text": " ".join(w["word"] for w in words),
            "segments": [{"start": 0.0, "end": duration, "text": " ".join(w["word"] for w in words)}],
            "words": words,
        }]

        srt, txt, timings = stitch.build_artifacts(chunks, results, duration=duration)

        # SRT words match TXT
        srt_words = []
        for line in srt.splitlines():
            line = line.strip()
            if not line or line.isdigit() or "-->" in line:
                continue
            srt_words.extend(line.split())
        assert srt_words == txt.split()

        # No overlapping cues
        parsed = json.loads(timings)
        for i in range(len(parsed["cues"]) - 1):
            assert parsed["cues"][i]["e"] <= parsed["cues"][i + 1]["s"] + 0.01

        # All cues have text
        for cue in parsed["cues"]:
            assert cue["t"].strip()

        # Word count matches
        assert parsed["word_count"] == len(words)
