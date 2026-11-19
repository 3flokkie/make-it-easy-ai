#!/usr/bin/env python3
"""
GTA 6 Stream & Video Autopilot:
1. Searches YouTube for fresh GTA 6 gameplay / stream / trailer reaction videos via yt-dlp.
2. Skips already processed video IDs (tracked in output/gta6_processed.json).
3. Runs stream_clipper.py to extract vertical 9:16 highlights with karaoke subtitles.
4. Uploads resulting Shorts via uploader.py (with uploadLimitExceeded queue protection).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
PROCESSED_FILE = BASE_DIR / "output" / "gta6_processed.json"
STREAM_SHORTS_DIR = BASE_DIR / "output" / "stream_shorts"

sys.path.insert(0, str(SCRIPTS_DIR))
from uploader import upload_short  # noqa: E402


DEFAULT_QUERIES = [
    "ytsearch5:GTA 6 gameplay stream highlights",
    "ytsearch5:GTA 6 live reaction",
]


def load_processed() -> set[str]:
    if PROCESSED_FILE.exists():
        try:
            return set(json.loads(PROCESSED_FILE.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def save_processed(ids: set[str]) -> None:
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_FILE.write_text(json.dumps(sorted(ids), indent=2), encoding="utf-8")


def search_videos(queries: list[str]) -> list[dict[str, str]]:
    """Finds candidate videos using yt-dlp flat-playlist search."""
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for q in queries:
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--flat-playlist",
            "--dump-single-json",
            q,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"⚠️ Search failed for {q}: {res.stderr.strip()[:200]}")
            continue
        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError:
            continue
        for entry in data.get("entries", []):
            vid_id = entry.get("id")
            if not vid_id or vid_id in seen:
                continue
            seen.add(vid_id)
            found.append({
                "id": vid_id,
                "title": entry.get("title", "GTA 6 Highlight"),
                "url": f"https://www.youtube.com/watch?v={vid_id}",
            })
    return found


def clip_video(url: str) -> list[Path]:
    """Runs stream_clipper.py on a URL and returns newly created mp4 files."""
    before = set(STREAM_SHORTS_DIR.glob("*.mp4")) if STREAM_SHORTS_DIR.exists() else set()
    cmd = [sys.executable, str(SCRIPTS_DIR / "stream_clipper.py"), "--url", url]
    print(f"🎬 Running stream_clipper for {url}...")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"⚠️ stream_clipper exited with {res.returncode} for {url}")
        return []
    after = set(STREAM_SHORTS_DIR.glob("*.mp4")) if STREAM_SHORTS_DIR.exists() else set()
    return sorted(after - before)


def main() -> None:
    parser = argparse.ArgumentParser(description="GTA 6 Auto-Clipper & Shorts Uploader")
    parser.add_argument("--limit", type=int, default=2, help="Max source videos to process per run")
    parser.add_argument("--dry-run", action="store_true", help="Only search and print candidates")
    parser.add_argument("--no-upload", action="store_true", help="Clip without uploading to YouTube")
    args = parser.parse_args()

    processed = load_processed()
    candidates = [v for v in search_videos(DEFAULT_QUERIES) if v["id"] not in processed]
    print(f"🔍 Found {len(candidates)} new GTA 6 candidates.")

    if args.dry_run:
        for v in candidates[: args.limit]:
            print(f"  - [{v['id']}] {v['title']} ({v['url']})")
        return

    for v in candidates[: args.limit]:
        clips = clip_video(v["url"])
        processed.add(v["id"])
        save_processed(processed)

        if args.no_upload:
            print(f"✅ Created {len(clips)} clips (upload skipped).")
            continue

        for idx, clip_path in enumerate(clips, start=1):
            short_title = f"GTA 6 Insane Moment #{idx} #shorts #gta6"
            short_desc = (
                f"Clipped from: {v['title']}\n\n"
                "#shorts #gta6 #gtavi #gaming #ai"
            )
            upload_short(str(clip_path), short_title, short_desc)


if __name__ == "__main__":
    main()
