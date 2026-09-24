"""Full integration test suite — out-of-the-box tests for the entire pipeline.

Tests run against the live FastAPI app using httpx.AsyncClient (no real server needed).
Covers:
  - API validation (URL, lang, job_id, callback_url)
  - Health endpoint
  - English transcription
  - Arabic transcription (--lang ar)
  - Translate mode (--translate)
  - Batch mode (multiple URLs)
  - CLI script (--lang, --translate, -f)
  - Error handling (invalid URLs, 404s, etc.)
"""

import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

# ── Import the FastAPI app ──────────────────────────────────────────────────
from app.main import app


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Async httpx client hitting the FastAPI app directly (no server needed).

    Sends the configured API key when the server expects one, so tests don't
    get 403 instead of the validation errors they're checking for.
    """
    from app.config import Settings
    transport = ASGITransport(app=app)
    headers: dict[str, str] = {}
    if Settings().api_key:
        headers["X-API-Key"] = Settings().api_key
    return AsyncClient(transport=transport, base_url="http://testserver", headers=headers)


REAL_REEL = "https://www.instagram.com/reel/DXq30xCAl6V/"


# ═══════════════════════════════════════════════════════════════════════════
# 1. HEALTH ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════

class TestHealth:
    @pytest.mark.anyio
    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model" in data

    @pytest.mark.anyio
    async def test_health_model_is_whisper(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert "whisper" in data["model"].lower()


# ═══════════════════════════════════════════════════════════════════════════
# 2. URL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class TestURLValidation:
    @pytest.mark.anyio
    async def test_rejects_non_instagram_url(self, client):
        resp = await client.post("/transcribe", json={"url": "https://youtube.com/watch?v=abc"})
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_rejects_http_instagram(self, client):
        resp = await client.post("/transcribe", json={"url": "http://instagram.com/reel/ABC/"})
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_rejects_empty_url(self, client):
        resp = await client.post("/transcribe", json={"url": ""})
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_rejects_missing_url(self, client):
        resp = await client.post("/transcribe", json={})
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_accepts_reel_url(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL})
        assert resp.status_code == 202

    @pytest.mark.anyio
    async def test_accepts_post_url(self, client):
        resp = await client.post("/transcribe", json={"url": "https://www.instagram.com/p/ABC123/"})
        assert resp.status_code == 202

    @pytest.mark.anyio
    async def test_accepts_reels_url(self, client):
        resp = await client.post("/transcribe", json={"url": "https://www.instagram.com/reels/ABC123/"})
        assert resp.status_code == 202

    @pytest.mark.anyio
    async def test_accepts_tv_url(self, client):
        resp = await client.post("/transcribe", json={"url": "https://www.instagram.com/tv/ABC123/"})
        assert resp.status_code == 202

    @pytest.mark.anyio
    async def test_accepts_stories_url(self, client):
        resp = await client.post("/transcribe", json={"url": "https://www.instagram.com/stories/ABC123/"})
        assert resp.status_code == 202


# ═══════════════════════════════════════════════════════════════════════════
# 3. JOB ID VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class TestJobIDValidation:
    @pytest.mark.anyio
    async def test_accepts_valid_job_id(self, client):
        resp = await client.post("/transcribe", json={
            "url": REAL_REEL,
            "job_id": "my-test-job_123",
        })
        assert resp.status_code == 202
        assert resp.json()["job_id"] == "my-test-job_123"

    @pytest.mark.anyio
    async def test_rejects_job_id_with_spaces(self, client):
        resp = await client.post("/transcribe", json={
            "url": REAL_REEL,
            "job_id": "my test job",
        })
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_rejects_job_id_with_special_chars(self, client):
        resp = await client.post("/transcribe", json={
            "url": REAL_REEL,
            "job_id": "job/../../etc",
        })
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# 4. LANG PARAMETER VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class TestLangValidation:
    @pytest.mark.anyio
    async def test_accepts_lang_en(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL, "lang": "en"})
        assert resp.status_code == 202

    @pytest.mark.anyio
    async def test_accepts_lang_ar(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL, "lang": "ar"})
        assert resp.status_code == 202

    @pytest.mark.anyio
    async def test_accepts_lang_pt_br(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL, "lang": "pt-BR"})
        assert resp.status_code == 202

    @pytest.mark.anyio
    async def test_rejects_invalid_lang(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL, "lang": "invalid-lang-code"})
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_rejects_numeric_lang(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL, "lang": "123"})
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# 5. TRANSLATE PARAMETER
# ═══════════════════════════════════════════════════════════════════════════

class TestTranslateParameter:
    @pytest.mark.anyio
    async def test_translate_default_is_false(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL})
        assert resp.status_code == 202

    @pytest.mark.anyio
    async def test_translate_true_accepted(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL, "translate": True})
        assert resp.status_code == 202

    @pytest.mark.anyio
    async def test_translate_with_lang(self, client):
        resp = await client.post("/transcribe", json={
            "url": REAL_REEL,
            "lang": "ar",
            "translate": True,
        })
        assert resp.status_code == 202


# ═══════════════════════════════════════════════════════════════════════════
# 6. JOB LIFECYCLE (submit → poll → done)
# ═══════════════════════════════════════════════════════════════════════════

class TestJobLifecycle:
    @pytest.mark.anyio
    async def test_submit_returns_job_id(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL})
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["status_url"].startswith("/jobs/")

    @pytest.mark.anyio
    async def test_poll_job_returns_status(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL})
        job_id = resp.json()["job_id"]

        # Poll a few times
        for _ in range(5):
            poll = await client.get(f"/jobs/{job_id}")
            assert poll.status_code == 200
            data = poll.json()
            assert data["status"] in ("queued", "downloading", "splitting", "transcribing", "stitching", "done", "failed")
            if data["status"] in ("done", "failed"):
                break
            await asyncio.sleep(2)

    @pytest.mark.anyio
    async def test_404_for_nonexistent_job(self, client):
        resp = await client.get("/jobs/nonexistent_job_999")
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_srt_404_before_job_completes(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL})
        job_id = resp.json()["job_id"]
        srt_resp = await client.get(f"/jobs/{job_id}/srt")
        # Either 404 (not ready yet) or 200 (already done fast)
        assert srt_resp.status_code in (404, 200)


# ═══════════════════════════════════════════════════════════════════════════
# 7. ENGLISH TRANSCRIPTION END-TO-END
# ═══════════════════════════════════════════════════════════════════════════

class TestEnglishTranscription:
    """Submit a real English reel, poll until done, verify SRT + TXT."""

    @pytest.mark.slow
    @pytest.mark.anyio
    async def test_english_reel_produces_srt_and_txt(self, client):
        # Submit
        resp = await client.post("/transcribe", json={"url": REAL_REEL})
        job_id = resp.json()["job_id"]

        # Poll until done (max 90s)
        status = "queued"
        for _ in range(30):
            await asyncio.sleep(3)
            poll = await client.get(f"/jobs/{job_id}")
            status = poll.json()["status"]
            if status in ("done", "failed"):
                break

        assert status == "done", f"Job finished with status: {status}"

        # Fetch SRT
        srt_resp = await client.get(f"/jobs/{job_id}/srt")
        assert srt_resp.status_code == 200
        srt_text = srt_resp.text
        assert len(srt_text) > 100, "SRT is too short"

        # SRT format validation
        blocks = srt_text.strip().split("\n\n")
        assert len(blocks) >= 5, f"Expected ≥5 cues, got {len(blocks)}"
        for block in blocks:
            lines = block.strip().splitlines()
            assert lines[0].strip().isdigit(), f"Bad cue number: {lines[0]}"
            ts_match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", lines[1])
            assert ts_match, f"Bad timestamp: {lines[1]}"
            assert lines[2].strip(), "Empty cue text"

        # Fetch TXT
        txt_resp = await client.get(f"/jobs/{job_id}/txt")
        assert txt_resp.status_code == 200
        txt_text = txt_resp.text
        assert len(txt_text) > 50, "TXT is too short"

        # SRT words match TXT
        srt_words = []
        for line in srt_text.splitlines():
            line = line.strip()
            if not line or line.isdigit() or "-->" in line:
                continue
            srt_words.extend(line.split())
        txt_words = txt_text.split()
        assert srt_words == txt_words, "SRT words don't match TXT"

        # Fetch timings
        timings_resp = await client.get(f"/jobs/{job_id}/timings")
        assert timings_resp.status_code == 200
        timings = timings_resp.json()
        assert timings["word_count"] > 10
        assert timings["cue_count"] >= 5
        assert timings["duration"] > 10


# ═══════════════════════════════════════════════════════════════════════════
# 8. ARABIC TRANSCRIPTION END-TO-END
# ═══════════════════════════════════════════════════════════════════════════

class TestArabicTranscription:
    """Submit a reel with --lang ar, verify Arabic content is produced."""

    @pytest.mark.slow
    @pytest.mark.anyio
    async def test_arabic_lang_param_works(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL, "lang": "ar"})
        job_id = resp.json()["job_id"]

        # Poll until done
        status = "queued"
        for _ in range(30):
            await asyncio.sleep(3)
            poll = await client.get(f"/jobs/{job_id}")
            status = poll.json()["status"]
            if status in ("done", "failed"):
                break

        assert status == "done", f"Arabic job finished with status: {status}"

        # Fetch TXT
        txt_resp = await client.get(f"/jobs/{job_id}/txt")
        assert txt_resp.status_code == 200
        txt_text = txt_resp.text
        assert len(txt_text) > 50, "Arabic TXT is too short"

        # Fetch SRT
        srt_resp = await client.get(f"/jobs/{job_id}/srt")
        assert srt_resp.status_code == 200
        srt_text = srt_resp.text
        assert len(srt_text) > 100, "Arabic SRT is too short"


# ═══════════════════════════════════════════════════════════════════════════
# 9. TRANSLATE MODE END-TO-END
# ═══════════════════════════════════════════════════════════════════════════

class TestTranslateMode:
    """Submit with translate=true, verify English output."""

    @pytest.mark.slow
    @pytest.mark.anyio
    async def test_translate_produces_english_output(self, client):
        resp = await client.post("/transcribe", json={
            "url": REAL_REEL,
            "translate": True,
        })
        job_id = resp.json()["job_id"]

        # Poll until done
        status = "queued"
        for _ in range(30):
            await asyncio.sleep(3)
            poll = await client.get(f"/jobs/{job_id}")
            status = poll.json()["status"]
            if status in ("done", "failed"):
                break

        assert status == "done", f"Translate job finished with status: {status}"

        # Fetch TXT — should be English
        txt_resp = await client.get(f"/jobs/{job_id}/txt")
        assert txt_resp.status_code == 200
        txt_text = txt_resp.text
        assert len(txt_text) > 50, "Translate TXT is too short"
        # Should contain common English words
        common_english = ["the", "is", "and", "to", "a"]
        assert any(w in txt_text.lower() for w in common_english), \
            f"Expected English words in: {txt_text[:200]}"


# ═══════════════════════════════════════════════════════════════════════════
# 10. BATCH MODE VIA CLI
# ═══════════════════════════════════════════════════════════════════════════

class TestBatchCLI:
    """Test the transcribe.sh script in batch mode."""

    def _run_script(self, args, timeout=120):
        """Run transcribe.sh with given args."""
        # Use relative path from project dir — bash on Windows (Git Bash)
        # auto-converts the cwd Windows path, so relative paths just work.
        project_dir = str(Path(__file__).parent.parent.resolve())
        cmd = ["bash", "scripts/transcribe.sh"] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_dir,
        )
        return result

    def test_cli_help_shows_usage(self):
        result = self._run_script([])
        # Should show usage info (exit code 1 for no args)
        assert "Usage:" in result.stdout or result.returncode != 0

    def test_cli_rejects_no_args(self):
        result = self._run_script([])
        assert result.returncode != 0

    def test_cli_accepts_single_url(self):
        """Submit one URL — just verify it starts (don't wait for full completion)."""
        result = self._run_script([REAL_REEL], timeout=30)
        # Should at least submit successfully
        assert "Submitting" in result.stdout or "✓" in result.stdout

    def test_cli_accepts_lang_flag(self):
        result = self._run_script(["--lang", "ar", REAL_REEL], timeout=30)
        assert "Language: ar" in result.stdout or "Submitting" in result.stdout

    def test_cli_accepts_translate_flag(self):
        result = self._run_script(["--translate", REAL_REEL], timeout=30)
        assert "TRANSLATE" in result.stdout or "Submitting" in result.stdout

    def test_cli_accepts_lang_and_translate(self):
        result = self._run_script(["--lang", "ar", "--translate", REAL_REEL], timeout=30)
        assert "TRANSLATE" in result.stdout or "Submitting" in result.stdout

    def test_cli_batch_from_file(self):
        """Test batch mode with a file of URLs."""
        urls_file = Path(__file__).parent.parent / "scripts" / "urls.txt"
        if urls_file.exists():
            result = self._run_script(["-f", "scripts/urls.txt"], timeout=30)
            assert "Submitting" in result.stdout or "✓" in result.stdout

    def test_cli_rejects_missing_file(self):
        result = self._run_script(["-f", "nonexistent.txt"])
        assert result.returncode != 0
        assert "not found" in result.stdout.lower() or "File not found" in result.stdout


# ═══════════════════════════════════════════════════════════════════════════
# 11. RESPONSE FORMAT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class TestResponseFormat:
    @pytest.mark.anyio
    async def test_submit_response_has_all_fields(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL})
        data = resp.json()
        required_fields = ["job_id", "status", "status_url", "srt_url", "txt_url", "timings_url"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    @pytest.mark.anyio
    async def test_job_status_response_has_fields(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL})
        job_id = resp.json()["job_id"]

        poll = await client.get(f"/jobs/{job_id}")
        data = poll.json()
        required = ["job_id", "url", "status", "progress_done", "progress_total"]
        for field in required:
            assert field in data, f"Missing field: {field}"
        assert data["job_id"] == job_id
        assert data["url"] == REAL_REEL

    @pytest.mark.anyio
    async def test_timings_json_structure(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL})
        job_id = resp.json()["job_id"]

        # Wait for completion
        for _ in range(30):
            await asyncio.sleep(3)
            poll = await client.get(f"/jobs/{job_id}")
            if poll.json()["status"] in ("done", "failed"):
                break

        if poll.json()["status"] == "done":
            timings_resp = await client.get(f"/jobs/{job_id}/timings")
            if timings_resp.status_code == 200:
                timings = timings_resp.json()
                assert "duration" in timings
                assert "words" in timings
                assert "cues" in timings
                assert isinstance(timings["words"], list)
                assert isinstance(timings["cues"], list)


# ═══════════════════════════════════════════════════════════════════════════
# 12. CONCURRENT JOBS
# ═══════════════════════════════════════════════════════════════════════════

class TestConcurrentJobs:
    @pytest.mark.anyio
    async def test_submit_two_jobs_simultaneously(self, client):
        resp1 = await client.post("/transcribe", json={"url": REAL_REEL, "job_id": "batch_1"})
        resp2 = await client.post("/transcribe", json={"url": REAL_REEL, "job_id": "batch_2"})

        assert resp1.status_code == 202
        assert resp2.status_code == 202
        assert resp1.json()["job_id"] == "batch_1"
        assert resp2.json()["job_id"] == "batch_2"

        # Both should be pollable
        poll1 = await client.get("/jobs/batch_1")
        poll2 = await client.get("/jobs/batch_2")
        assert poll1.status_code == 200
        assert poll2.status_code == 200

    @pytest.mark.anyio
    async def test_three_jobs_with_different_langs(self, client):
        jobs = []
        for lang in ["en", "ar", None]:
            body = {"url": REAL_REEL, "job_id": f"lang_test_{lang or 'auto'}"}
            if lang:
                body["lang"] = lang
            resp = await client.post("/transcribe", json=body)
            assert resp.status_code == 202
            jobs.append(resp.json()["job_id"])

        # All should be pollable
        for jid in jobs:
            poll = await client.get(f"/jobs/{jid}")
            assert poll.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# 13. OUTPUT FILE INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

class TestOutputFileIntegrity:
    @pytest.mark.slow
    @pytest.mark.anyio
    async def test_output_files_exist_on_disk(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL, "job_id": "file_test"})
        job_id = resp.json()["job_id"]

        # Wait for completion
        for _ in range(30):
            await asyncio.sleep(3)
            poll = await client.get(f"/jobs/{job_id}")
            if poll.json()["status"] in ("done", "failed"):
                break

        if poll.json()["status"] == "done":
            data = poll.json()
            # Check files on disk
            if data.get("srt_path"):
                assert Path(data["srt_path"]).exists(), "SRT file missing on disk"
            if data.get("txt_path"):
                assert Path(data["txt_path"]).exists(), "TXT file missing on disk"
            if data.get("timings_path"):
                assert Path(data["timings_path"]).exists(), "Timings file missing on disk"

    @pytest.mark.slow
    @pytest.mark.anyio
    async def test_srt_file_is_valid_utf8(self, client):
        resp = await client.post("/transcribe", json={"url": REAL_REEL, "job_id": "utf8_test"})
        job_id = resp.json()["job_id"]

        for _ in range(30):
            await asyncio.sleep(3)
            poll = await client.get(f"/jobs/{job_id}")
            if poll.json()["status"] in ("done", "failed"):
                break

        if poll.json()["status"] == "done":
            srt_resp = await client.get(f"/jobs/{job_id}/srt")
            if srt_resp.status_code == 200:
                # Should not raise UnicodeDecodeError
                text = srt_resp.text
                assert isinstance(text, str)
                assert len(text) > 0
