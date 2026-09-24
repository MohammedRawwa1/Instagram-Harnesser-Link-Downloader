"""Instagram channel reel URL collector (local-first).

Instagram aggressively rate-limits automated profile enumeration (Instaloader's
web_profile_info API and yt-dlp's user extractor both hit the same 429 wall from
the same IP). The reliable path is to paste reel URLs from your browser session,
which isn't subject to the same automated-scrape limits.

Paste URLs are canonicalized (UTM/query params stripped) and deduplicated against
`seen_file` so re-runs don't re-submit the same reel. Discovered URLs are also
appended to `urls_file` (default: scripts/urls.txt) for the batch processor.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

INSTA_REEL_RE = re.compile(
    r"^https://(?:www\.)?instagram\.com/reel/([A-Za-z0-9_-]+)(?:/(?:\?)?|\?.*|/\?.*)?$"
)


def _is_reel_url(url: str) -> bool:
    return bool(INSTA_REEL_RE.match(url.strip()))


def _canonical_reel_url(url: str) -> str:
    """Return the canonical reel URL: https://www.instagram.com/reel/<shortcode>/

    Accepts pasted URLs with UTM/query params and/or a trailing slash and
    normalizes them to the canonical form used for deduplication/storage.
    """
    url = url.strip()
    m = INSTA_REEL_RE.match(url)
    if not m:
        return url
    shortcode = m.group(1)
    return f"https://www.instagram.com/reel/{shortcode}/"


def _load_seen(seen_file: Path) -> set[str]:
    if not seen_file.exists():
        return set()
    try:
        return {line.strip() for line in seen_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    except (UnicodeDecodeError, OSError):
        return set()


def _append_seen(seen_file: Path, urls: Sequence[str]) -> None:
    if not urls:
        return
    seen_file.parent.mkdir(parents=True, exist_ok=True)
    with seen_file.open("a", encoding="utf-8") as f:
        for u in urls:
            f.write(u.strip() + "\n")


def _append_urls(urls_file: Path, urls: Sequence[str]) -> None:
    if not urls:
        return
    urls_file.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if urls_file.exists():
        try:
            existing = {line.strip() for line in urls_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        except (UnicodeDecodeError, OSError):
            pass
    new = [u for u in urls if u.strip() not in existing]
    if not new:
        return
    with urls_file.open("a", encoding="utf-8") as f:
        for u in new:
            f.write(u.strip() + "\n")


def collect_reel_urls(
    username: str,
    urls_file: Path | None = None,
    seen_file: Path | None = None,
    *,
    max_reels: int = 0,
) -> list[str]:
    """Collect reel URLs for `username` by pasting from your browser.

    Parameters
    ----------
    username:
        Instagram username without leading `@`.
    urls_file:
        Where to append newly-discovered URLs (batch input file). Defaults to
        `scripts/urls.txt` next to this project root.
    seen_file:
        Where to record URLs we've already collected (dedupe across runs).
        Defaults to `data/collected_urls.txt`.
    max_reels:
        Stop after this many reels (0 = no limit). Useful for testing.
    """
    project_root = _project_root()
    if urls_file is None:
        urls_file = project_root / "scripts" / "urls.txt"
    if seen_file is None:
        seen_file = project_root / "data" / "collected_urls.txt"

    seen = _load_seen(seen_file)
    out: list[str] = []

    profile_url = f"https://www.instagram.com/{username}/"
    print(f"  Open {profile_url} in your browser and copy the reel URLs.")
    print("  Paste reel URLs below (one per line, empty line to finish):")
    count = 0
    while True:
        try:
            line = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        if _is_reel_url(line):
            canonical = _canonical_reel_url(line)
            if canonical not in seen:
                out.append(canonical)
                seen.add(canonical)
                count += 1
                if max_reels and count >= max_reels:
                    break
        else:
            print(f"    Skipped (not a reel URL): {line[:60]}")

    if out:
        _append_seen(seen_file, out)
        _append_urls(urls_file, out)
    return out


def _project_root() -> Path:
    """Best-effort project root: parent of this file's parent (app/)."""
    return Path(__file__).resolve().parent.parent


def manually_add_url(url: str, urls_file: Path | None = None) -> bool:
    """Validate and append a single reel URL to `urls_file` (idempotent)."""
    if not _is_reel_url(url):
        return False
    project_root = _project_root()
    if urls_file is None:
        urls_file = project_root / "scripts" / "urls.txt"
    canonical = _canonical_reel_url(url)
    existing: set[str] = set()
    if urls_file.exists():
        try:
            existing = {line.strip() for line in urls_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        except (UnicodeDecodeError, OSError):
            pass
    if canonical in existing:
        return True  # already there
    urls_file.parent.mkdir(parents=True, exist_ok=True)
    with urls_file.open("a", encoding="utf-8") as f:
        f.write(canonical + "\n")
    return True
