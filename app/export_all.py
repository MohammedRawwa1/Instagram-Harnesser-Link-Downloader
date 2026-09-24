"""Collect all finished job transcripts into a single exports folder.

After running a channel batch (mode 3), use this to grab every artifact:

    python -m app.export_all

Or as a module:

    from app.export_all import export_all_jobs
    export_all_jobs()

What it does:
  1. Scans data/jobs/*.json for jobs with status == "done".
  2. Copies each job's SRT, TXT, and timings into data/exports/<job_id>/.
  3. Optionally concatenates all TXT transcripts into data/exports/all_transcripts.txt.
  4. Prints a summary of what it found (and what's missing).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

JOBS_DIR = Path(__file__).resolve().parent.parent / "data" / "jobs"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "output"
EXPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "exports"


def export_all_jobs(*, concat_txt: bool = True, overwrite: bool = False) -> dict:
    """Export all done jobs' artifacts.

    Returns a summary dict with keys: exported, missing, total, concat_path.
    """
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for jf in sorted(JOBS_DIR.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if data.get("status") == "done":
                jobs.append((jf.stem, data))
        except (json.JSONDecodeError, OSError):
            continue

    exported = 0
    missing = 0
    summary = []

    for job_id, data in jobs:
        export_dir = EXPORTS_DIR / job_id
        if export_dir.exists() and not overwrite:
            summary.append((job_id, "skipped (already exported)"))
            continue

        export_dir.mkdir(parents=True, exist_ok=True)

        files_copied = []
        for key in ("srt_path", "txt_path", "timings_path"):
            src = Path(data.get(key, ""))
            if not src.exists():
                files_copied.append((key, None))
                continue
            dst = export_dir / src.name
            shutil.copy2(src, dst)
            files_copied.append((key, str(dst)))

        exported += 1
        missing += sum(1 for _, dst in files_copied if dst is None)
        summary.append((job_id, files_copied))

    # Concatenated transcript
    concat_path: str | None = None
    if concat_txt:
        all_txt_path = EXPORTS_DIR / "all_transcripts.txt"
        txt_files = sorted(EXPORT_DIR.glob("*/transcript - *.txt")) if False else []
        # Re-scan from exports dir
        txt_files = sorted(EXPORTS_DIR.glob("*"))  # all job dirs
        txt_paths = []
        for job_dir in txt_files:
            if job_dir.is_dir():
                for f in job_dir.glob("transcript - *.txt"):
                    txt_paths.append(f)
        if txt_paths:
            all_txt_path.parent.mkdir(parents=True, exist_ok=True)
            with all_txt_path.open("w", encoding="utf-8") as out:
                for i, p in enumerate(sorted(txt_paths)):
                    if i > 0:
                        out.write("\n\n")
                    out.write(f"# {p.parent.name}\n")
                    out.write(p.read_text(encoding="utf-8"))
                    out.write("\n")
            concat_path = str(all_txt_path)

    return {
        "exported": exported,
        "missing_artifacts": missing,
        "total_done_jobs": len(jobs),
        "summary": summary,
        "concat_txt_path": concat_path,
    }


def main() -> None:
    result = export_all_jobs()
    print(f"Exported: {result['exported']}/{result['total_done_jobs']} jobs")
    if result["missing_artifacts"]:
        print(f"Missing artifacts: {result['missing_artifacts']}")
    for job_id, files in result["summary"]:
        if isinstance(files, list):
            parts = []
            for key, dst in files:
                if dst:
                    parts.append(f"{key}: {Path(dst).name}")
                else:
                    parts.append(f"{key}: MISSING")
            print(f"  {job_id}: {', '.join(parts)}")
        else:
            print(f"  {job_id}: {files}")
    if result["concat_txt_path"]:
        print(f"\nConcatenated transcript: {result['concat_txt_path']}")
    else:
        print("\nNo TXT files to concatenate.")
    print("\nExports directory: data/exports/")


if __name__ == "__main__":
    main()
