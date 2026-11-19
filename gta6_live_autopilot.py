#!/usr/bin/env python3
"""
GTA 6 Live Stream Auto-Finder + Clipper Loop.
Automatically discovers active YouTube live streams for GTA 6,
picks the top live stream, and runs live_chat_clipper.py on it.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_QUERIES = [
    "ytsearch10:GTA 6 live stream",
    "ytsearch10:GTA VI launch live",
]


def find_live_streams(queries: list[str]) -> list[dict[str, str]]:
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
            continue
        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError:
            continue
        for entry in data.get("entries", []):
            vid = entry.get("id")
            if not vid or vid in seen:
                continue
            if entry.get("live_status") != "is_live":
                continue
            seen.add(vid)
            found.append({
                "id": vid,
                "title": entry.get("title", "GTA 6 Live"),
                "url": f"https://www.youtube.com/watch?v={vid}",
            })
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-find GTA 6 live streams and run live_chat_clipper.py")
    parser.add_argument("--token", default="token_channel2.pickle")
    parser.add_argument("--max-clips", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    streams = find_live_streams(DEFAULT_QUERIES)
    print(f"🔴 Found {len(streams)} live GTA 6 streams.")
    for s in streams:
        print(f"  - [{s['id']}] {s['title']} ({s['url']})")

    if args.dry_run or not streams:
        return

    for s in streams:
        print(f"\n🚀 Attaching live_chat_clipper to: {s['title']} ({s['url']})")
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "live_chat_clipper.py"),
            "--url", s["url"],
            "--token", args.token,
            "--spike-mult", "3.0",
            "--clip-sec", "40",
            "--cooldown", "180",
            "--max-clips", str(args.max_clips),
            "--title-prefix", "GTA 6 LIVE",
        ]
        if args.no_upload:
            cmd.append("--no-upload")
        subprocess.run(cmd)
        time.sleep(5)


if __name__ == "__main__":
    main()
