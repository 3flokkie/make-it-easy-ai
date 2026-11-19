#!/usr/bin/env python3
"""
Instagram Reels Uploader via Meta Graph API.
Reads IG_ACCESS_TOKEN and IG_USER_ID from scripts/.env.
Supports uploading local .mp4 files via resumable rupload.facebook.com endpoint.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")


def _api_post(url: str, data: dict[str, str] | None = None, headers: dict[str, str] | None = None, raw_body: bytes | None = None) -> dict:
    body = raw_body if raw_body is not None else (urllib.parse.urlencode(data or {}).encode("utf-8"))
    req = urllib.request.Request(url, data=body, headers=headers or {}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"IG API HTTP {e.code}: {err_body}") from e


def _api_get(url: str) -> dict:
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"IG API HTTP {e.code}: {err_body}") from e


def upload_reel(video_path: str, caption: str) -> str:
    token = os.getenv("IG_ACCESS_TOKEN", "").strip()
    ig_user_id = os.getenv("IG_USER_ID", "").strip()
    if not token or not ig_user_id:
        raise ValueError("IG_ACCESS_TOKEN and IG_USER_ID must be set in scripts/.env")

    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    file_size = vpath.stat().st_size
    print(f"📤 [IG] Creating Reels container for {vpath.name} ({file_size // (1024 * 1024)} MB)...")

    init_url = f"https://graph.facebook.com/v21.0/{ig_user_id}/media"
    init_res = _api_post(init_url, data={
        "media_type": "REELS",
        "upload_type": "resumable",
        "caption": caption,
        "access_token": token,
    })
    creation_id = init_res["id"]
    upload_uri = init_res.get("uri") or f"https://rupload.facebook.com/ig-api-upload/v21.0/{creation_id}"

    print(f"⬆️ [IG] Uploading video bytes to container {creation_id}...")
    video_bytes = vpath.read_bytes()
    _api_post(
        upload_uri,
        headers={
            "Authorization": f"OAuth {token}",
            "offset": "0",
            "file_size": str(file_size),
            "Content-Type": "application/octet-stream",
        },
        raw_body=video_bytes,
    )

    print("⏳ [IG] Waiting for Instagram to process video...")
    for _ in range(30):
        status_url = f"https://graph.facebook.com/v21.0/{creation_id}?fields=status_code&access_token={token}"
        st = _api_get(status_url).get("status_code")
        if st == "FINISHED":
            break
        if st == "ERROR":
            raise RuntimeError(f"Instagram processing failed for container {creation_id}")
        time.sleep(4)

    pub_url = f"https://graph.facebook.com/v21.0/{ig_user_id}/media_publish"
    pub_res = _api_post(pub_url, data={
        "creation_id": creation_id,
        "access_token": token,
    })
    media_id = pub_res["id"]
    print(f"🎉 [IG] Published Reel! Media ID: {media_id}")
    return media_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload local MP4 to Instagram Reels")
    parser.add_argument("--video", required=True, help="Path to local MP4 file")
    parser.add_argument("--caption", required=True, help="English caption with hashtags")
    args = parser.parse_args()
    upload_reel(args.video, args.caption)
