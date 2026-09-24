"""Iterate an Instagram channel's reels via pagination, without relying on the
web_profile_info endpoint that 429s so easily.

Strategy:
  1. Resolve the username to a user id via the embedd endpoint
     (/embed/ProfilePage/:id) or the public profile page.
  2. Paginate the user's media via the GraphQL media endpoint that the web app
     uses to load the profile grid. This endpoint carries its own rate limits
     and often still responds when web_profile_info is 429'd.
  3. Yield reel shortcodes as we go.

If pagination fails (429 / blocked), fall back to pasting from the browser.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Generator, Optional

import httpx

INSTA_REEL_RE = re.compile(r"^https://(www\.)?instagram\.com/reel/([A-Za-z0-9_-]+)/?$")
GRAPHQL_MEDIA_QUERY_HASH = "b30d5e618d828d30b98d1941aacb277359babe57e0840e8be176945618085051"
GRAPHQL_MEDIA_QUERY_VARS = (
    "after=|address_book_visible=|fetch_media_item_count=|include_biography|"
    "include_biography_image|include_shopping|include_tm|lad optimisation|"
    "media_type|minimize_result|fetch_funding=|should_show_friends_posts|"
    "show_pep_reel|surfaces|use_soq|ext_ref_travel_metadata|__v|__a=1"
)
# The real query vars Instagram uses — we build them properly below.

# Common user resolver: try embedd, then public page HTML.
USER_RESOLVE_TIMEOUT = 15.0

# Pagination: fetch at most this many reels to keep the demo snappy.
DEFAULT_MAX_REELS = 50

# How long to wait between paginated requests (avoid rate limits).
PAGINATE_DELAY = 1.0  # seconds between pages


def _canonical_reel_url(shortcode: str) -> str:
    return f"https://www.instagram.com/reel/{shortcode}/"


def _is_reel_post(media: dict) -> bool:
    """Return True if this media entry is a video reel."""
    # Instagram GraphQL media entries have a `media_type` field:
    #   1 = image, 2 = video, 8 = carousel (may contain video), etc.
    # A reel is typically media_type 2 (video) or an IGTV/reel variant.
    mt = media.get("media_type")
    if mt == 2:
        return True
    # Carousels sometimes contain video — check the edge_media_to_video.
    if mt == 8:
        return bool(media.get("edge_media_to_video") or media.get("is_video"))
    # Some entries expose `is_video` directly.
    return bool(media.get("is_video"))


def resolve_user_id(username: str, client: httpx.Client) -> Optional[str]:
    """Resolve a username to a numeric user id.

    Tries:
      1. /embed/ProfilePage/:username (returns JSON-like HTML with user id).
      2. /api/v1/users/web_profile_info/?username= (if not 429'd).
      3. The public profile page's initial state JSON.
    """
    # 1. Embed endpoint — lightweight, often not 429'd.
    try:
        r = client.get(f"https://www.instagram.com/embed/ProfilePage/{username}/", timeout=USER_RESOLVE_TIMEOUT)
        if r.status_code == 200:
            # The embed page contains a JSON blob with the user id.
            m = re.search(r'"user_id"\s*:\s*"(\d+)"', r.text)
            if m:
                return m.group(1)
            m = re.search(r'"id"\s*:\s*"?(\d{6,})"?', r.text)
            if m:
                return m.group(1)
    except Exception:
        pass

    # 2. web_profile_info — may 429, but worth one try.
    try:
        r = client.get(f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}", timeout=USER_RESOLVE_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            user = data.get("data", {}).get("user", {})
            if user:
                return str(user.get("id") or user.get("user_id"))
    except Exception:
        pass

    # 3. Public profile page initial state.
    try:
        r = client.get(f"https://www.instagram.com/{username}/", timeout=USER_RESOLVE_TIMEOUT)
        if r.status_code == 200:
            # Instagram embeds a JSON in a <script> tag: window._sharedData
            m = re.search(r'window\._sharedData\s*=\s*({.*?});', r.text, re.DOTALL)
            if m:
                shared = json.loads(m.group(1))
                user = _find_user_in_shared(shared, username)
                if user:
                    return str(user.get("id"))
    except Exception:
        pass

    return None


def _find_user_in_shared(shared: dict, username: str) -> Optional[dict]:
    """Walk the _sharedData structure to find a user by username."""
    # _sharedData has a "entry_data" → "ProfilePage" → [ { "graphql" → "user" } ]
    entry_data = shared.get("entry_data", {})
    profile_pages = entry_data.get("ProfilePage", [])
    for page in profile_pages:
        graphql = page.get("graphql", {})
        user = graphql.get("user", {})
        if user.get("username") == username:
            return user
    return None


def iter_channel_reels(
    username: str,
    *,
    max_reels: int = DEFAULT_MAX_REELS,
    client: Optional[httpx.Client] = None,
    delay: float = PAGINATE_DELAY,
) -> Generator[str, None, None]:
    """Yield reel shortcodes for `username` by paginating the GraphQL media endpoint.

    Yields canonical reel URLs: https://www.instagram.com/reel/{shortcode}/

    If the endpoint 429s or blocks, the generator stops early (caller should
    fall back to pasting from the browser).
    """
    if client is None:
        client = httpx.Client(timeout=15.0, follow_redirects=True)

    user_id = resolve_user_id(username, client)
    if not user_id:
        return  # can't resolve; caller should fall back

    # Build the GraphQL query variables.
    variables = {
        "id": user_id,
        "resolution": 540,
        "first": min(max_reels, 50),  # page size
        "after": None,
    }

    # The query hash for "Profile Media" (user's posts grid).
    # This is Instagram's current query hash for profile media — it may rotate;
    # if it stops working, fall back to pasting.
    query_hash = GRAPHQL_MEDIA_QUERY_HASH

    count = 0
    after = None
    while count < max_reels:
        variables["after"] = after
        variables["first"] = min(max_reels - count, 50)

        try:
            # Encode variables as JSON.
            vars_json = json.dumps(variables, separators=(",", ":"))
            url = f"https://www.instagram.com/graphql/query/?query_hash={query_hash}&variables={vars_json}"
            r = client.get(url, timeout=15.0)
            if r.status_code == 429:
                # Rate limited — stop pagination, caller falls back.
                return
            if r.status_code != 200:
                return
            data = r.json()
        except Exception:
            return

        # Parse the media edges.
        remission = data.get("data", {}).get("user", {}).get("edge_owner_to_timeline_media", {})
        edges = remission.get("edges", [])
        for edge in edges:
            if count >= max_reels:
                break
            node = edge.get("node", {})
            if _is_reel_post(node):
                shortcode = node.get("shortcode") or node.get("code")
                if shortcode:
                    yield _canonical_reel_url(shortcode)
                    count += 1

        # Pagination cursor.
        page_info = remission.get("page_info", {})
        after = page_info.get("end_cursor")
        if not page_info.get("has_next_page") or not after:
            break
        time.sleep(delay)

    return


def collect_reel_urls_paginated(
    username: str,
    urls_file: Optional[Path] = None,
    seen_file: Optional[Path] = None,
    *,
    max_reels: int = DEFAULT_MAX_REELS,
) -> list[str]:
    """Try pagination first; fall back to the paste prompt if it 429s.

    Persists discovered URLs to `urls_file` and `seen_file` (same semantics as
    the paste-based collector).
    """
    project_root = Path(__file__).resolve().parent.parent
    if urls_file is None:
        urls_file = project_root / "scripts" / "urls.txt"
    if seen_file is None:
        seen_file = project_root / "data" / "collected_urls.txt"

    seen = set()
    if seen_file.exists():
        try:
            seen = {line.strip() for line in seen_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        except (UnicodeDecodeError, OSError):
            pass

    out: list[str] = []

    # 1. Pagination-based iteration.
    print(f"  Iterating @{username} via pagination...")
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True, headers=_user_agent()) as client:
            for url in iter_channel_reels(username, max_reels=max_reels, client=client):
                if url not in seen:
                    out.append(url)
                    seen.add(url)
    except Exception as e:
        print(f"  Pagination failed for @{username}: {e}")

    if out:
        _append_seen(seen_file, out)
        _append_urls(urls_file, out)
        return out

    # 2. Fallback: paste from browser.
    print(f"  Pagination didn't yield reels for @{username}.")
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
        if INSTA_REEL_RE.match(line.strip()):
            canonical = _canonical_reel_url_from_input(line)
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


# ---- helpers ----

def _canonical_reel_url_from_input(url: str) -> str:
    url = url.strip()
    if "?" in url:
        url = url.split("?", 1)[0]
    if not url.endswith("/"):
        url += "/"
    return url


def _append_seen(seen_file: Path, urls: list[str]) -> None:
    if not urls:
        return
    seen_file.parent.mkdir(parents=True, exist_ok=True)
    with seen_file.open("a", encoding="utf-8") as f:
        for u in urls:
            f.write(u.strip() + "\n")


def _append_urls(urls_file: Path, urls: list[str]) -> None:
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


def _user_agent() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
