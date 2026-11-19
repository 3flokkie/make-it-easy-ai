#!/usr/bin/env python3
"""
live_chat_clipper.py — real-time hype-spike clipper + Shorts auto-uploader for live streams
(GTA 6 / gaming).

How it works
  1. A background ffmpeg keeps a rolling DVR buffer of the live HLS stream (small .ts segments).
  2. Live chat is watched:
       YouTube: pytchat -> yt-dlp live_chat -> YouTube Data API polling.
       Twitch:  anonymous IRC (irc.chat.twitch.tv:6667, justinfanNNNN) — no API key needed.
  3. Rolling 30 s baseline of msgs/sec vs. the latest 5 s window. When
     window_rate >= spike_mult * baseline_rate AND window has >= min msgs -> CLIP.
  4. Last --clip-sec seconds are cut from the buffer, cropped to 1080x1920, transcribed
     with faster-whisper (word timestamps), karaoke subtitles burned in
     (libass if ffmpeg has it, otherwise Pillow PNG overlays), the real chat messages from the spike
     burst into a dark glass chat box under the "CHAT EXPLODED xN" hook for the first 5 s, then uploaded via
     uploader.upload_short(..., token_name=args.token).

Usage
  .venv/bin/python scripts/live_chat_clipper.py --url "https://www.youtube.com/watch?v=XXXX" \
      --token token_channel2.pickle --spike-mult 3.0 --clip-sec 40 --cooldown 120
  .venv/bin/python scripts/live_chat_clipper.py --url "https://www.twitch.tv/<channel>" --no-upload

Optional dependency
  pytchat (best chat source):  .venv/bin/pip install pytchat
  Without it the script falls back to yt-dlp's live_chat downloader automatically.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import queue
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR / "output" / "live_clips"
HISTORY_FILE = SCRIPT_DIR / "live_clip_history.json"

WINDOW_SEC = 5.0
BASELINE_SEC = 30.0
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
# Chat overlay: bold author names + a wide-coverage face for message bodies (Cyrillic, accents, CJK).
CHAT_AUTHOR_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
CHAT_TEXT_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
CHAT_RECENT_MAX = 30   # rolling deque of the newest chat messages kept for the overlay
HOOK_SEC = 5.0         # "CHAT EXPLODED xN" banner + chat burst duration
CHAT_OVERLAY_Y = 400   # top of the chat box PNG (just under the hook banner, far above karaoke subs)

VIRAL_COVER_HOOKS = [
    "GTA 6 WHAT?! HOW?!",
    "HOW DID HE DO THAT?!",
    "NO WAY THIS HAPPENED!",
    "WAIT FOR THE END...",
]

VIRAL_TITLES = [
    "GTA 6 WHAT?! HOW DID HE DO THAT?! 😱🔥 #shorts",
    "GTA 6 NO WAY THIS HAPPENED LIVE 🤯🔥 #shorts",
    "GTA 6 CHAT WENT INSANE AT THIS MOMENT 💀🔥 #shorts",
    "GTA 6 WAIT FOR THE ENDING 😳🔥 #shorts",
]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# ─────────────────────────── ffmpeg helpers ───────────────────────────

def find_ffmpeg() -> str:
    for cand in (shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if cand and Path(cand).exists():
            return cand
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("❌ ffmpeg not found. Install it: brew install ffmpeg")


def find_ffprobe(ffmpeg: str) -> str | None:
    sibling = Path(ffmpeg).with_name("ffprobe")
    return str(sibling) if sibling.exists() else shutil.which("ffprobe")


def ffmpeg_has_libass(ffmpeg: str) -> bool:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=15).stdout
        return bool(re.search(r"^\s*\S+\s+subtitles\s", out, re.M))
    except Exception:
        return False


def run_ffmpeg(ffmpeg: str, args: list[str], what: str) -> None:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({what}): {res.stderr.strip()[-800:]}")


@dataclass
class MediaProbe:
    duration: float
    has_video: bool
    has_audio: bool


def probe_media(ffprobe: str | None, path: Path) -> MediaProbe | None:
    """Duration + which stream types exist. None if ffprobe is unavailable or the file is unreadable."""
    if not ffprobe:
        return None
    res = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        data = json.loads(res.stdout or "{}")
        dur = float(data.get("format", {}).get("duration") or 0.0)
    except (ValueError, json.JSONDecodeError):
        return None
    kinds = {st.get("codec_type") for st in data.get("streams", [])}
    return MediaProbe(dur, "video" in kinds, "audio" in kinds)


# ─────────────────────────── stream resolving ───────────────────────────

@dataclass
class StreamInfo:
    video_id: str
    title: str
    hls_url: str
    is_live: bool
    audio_url: str | None = None  # set only when video and audio come as separate streams


# Prefer a single HLS variant that carries BOTH video and audio (YouTube live 91-96/300/301),
# capped at 1080p; fall back to separate video+audio, then whatever exists.
YTDLP_FORMAT = (
    "best[vcodec!=none][acodec!=none][height<=1080][protocol^=m3u8]"
    "/best[vcodec!=none][acodec!=none][height<=1080]"
    "/best[vcodec!=none][acodec!=none]"
    "/bv*[height<=1080]+ba/bv*+ba/b"
)


# twitch.tv first-level paths that are site pages, not channels.
_TWITCH_RESERVED = {
    "videos", "directory", "settings", "subscriptions", "inventory", "wallet", "drops", "downloads",
    "search", "p", "jobs", "turbo", "prime", "friends", "messages", "store", "broadcast", "moderator",
    "popout", "embed", "login", "signup", "logout", "user", "u", "team", "collections", "following",
}
_TWITCH_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.|m\.|go\.)?twitch\.tv/(?P<ch>[A-Za-z0-9_]{1,40})/?(?:[?#].*)?$", re.I)


def twitch_channel_from_url(url: str) -> str | None:
    """'https://www.twitch.tv/Some_Channel' -> 'some_channel'; None for non-Twitch / VOD / clip / site URLs."""
    m = _TWITCH_URL_RE.match(url.strip())
    if not m or m.group("ch").lower() in _TWITCH_RESERVED:
        return None
    return m.group("ch").lower()


def resolve_stream(url: str) -> StreamInfo:
    import yt_dlp  # type: ignore

    twitch = twitch_channel_from_url(url)
    if twitch:
        url = f"https://www.twitch.tv/{twitch}"  # canonical form: drops ?referrer= etc.
    opts = {"quiet": True, "no_warnings": True, "ignore_no_formats_error": True, "format": YTDLP_FORMAT}
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            if twitch:
                low = str(e).lower()
                why = ("is offline (not currently live)" if "not currently live" in low
                       else "does not exist" if "does not exist" in low else f"could not be resolved: {e}")
                raise RuntimeError(f"Twitch channel '{twitch}' {why}") from None
            raise
    if info.get("_type") == "playlist" and info.get("entries"):
        info = next(e for e in info["entries"] if e)

    video_url = audio_url = None
    req = info.get("requested_formats")
    if req:  # separate video + audio picked by the format selector
        # YouTube live HLS often reports acodec=None (unknown) on the audio-only format and marks it
        # only via vcodec='none' / resolution='audio only' — so detect audio as "not the video one".
        v = next((f for f in req if f.get("vcodec") not in (None, "none")), None) or req[0]
        au = next((f for f in req if f is not v and f.get("acodec") != "none"), None)
        video_url = v and v.get("url")
        audio_url = au and au.get("url")
    elif info.get("url"):
        video_url = info["url"]
        if info.get("acodec") == "none":
            log("⚠️  selected stream format has no audio track — clips will be silent")
    if not video_url:
        # last resort: any muxed format, else the master manifest
        muxed = [f for f in info.get("formats") or []
                 if f.get("url") and f.get("vcodec") not in (None, "none") and f.get("acodec") not in (None, "none")]
        video_url = muxed[-1]["url"] if muxed else info.get("manifest_url")
    if not video_url:
        raise RuntimeError("Could not resolve a stream URL via yt-dlp")
    return StreamInfo(
        # Twitch's info id is a numeric broadcast id — prefix it with the channel for readable paths.
        video_id=f"twitch_{twitch}_{info['id']}" if twitch else info["id"],
        title=info.get("title") or "",
        hls_url=video_url,
        is_live=bool(info.get("is_live")),
        audio_url=audio_url,
    )


# ─────────────────────────── rolling DVR recorder ───────────────────────────

class RollingRecorder:
    """Keeps the last `buffer_sec` seconds of the live stream as wrapped .ts segments.

    Segments keep CONTINUOUS timestamps (no -reset_timestamps), so byte-concatenating
    consecutive MPEG-TS segments yields a valid stream with the true duration.
    The newest segment is always the one ffmpeg is still writing and is never used.
    """

    def __init__(self, ffmpeg: str, url: str, buf_dir: Path, buffer_sec: int, seg_sec: int = 2,
                 ffprobe: str | None = None):
        self.ffmpeg, self.url, self.buf_dir, self.ffprobe = ffmpeg, url, buf_dir, ffprobe
        self.seg_sec = seg_sec
        self.wrap = max(10, math.ceil(buffer_sec / seg_sec))
        self.proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._probe_cache: dict[tuple[str, int, int], MediaProbe | None] = {}
        self._thread = threading.Thread(target=self._supervise, daemon=True)

    def _spawn(self) -> None:
        info = resolve_stream(self.url)
        # A restart begins a new timeline: drop old segments so they never get spliced with new ones.
        with self._lock:
            for old in self.buf_dir.glob("buf_*.ts"):
                old.unlink(missing_ok=True)
            self._probe_cache.clear()
        net = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *net, "-i", info.hls_url]
        if info.audio_url:
            cmd += [*net, "-i", info.audio_url, "-map", "0:v:0", "-map", "1:a:0"]
        else:
            cmd += ["-map", "0:v:0", "-map", "0:a:0?"]
        cmd += [
            "-c", "copy",
            "-f", "segment", "-segment_time", str(self.seg_sec), "-segment_wrap", str(self.wrap),
            "-segment_format", "mpegts",
            str(self.buf_dir / "buf_%04d.ts"),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        src = "video+audio (separate)" if info.audio_url else "muxed HLS"
        log(f"🎥 DVR buffer recording {src} ({self.wrap * self.seg_sec}s rolling) → {self.buf_dir}")

    def _supervise(self) -> None:
        backoff = 5
        while not self._stop.is_set():
            if self.proc is None or self.proc.poll() is not None:
                if self.proc is not None:
                    err = (self.proc.stderr.read() or b"").decode(errors="ignore")[-300:] if self.proc.stderr else ""
                    log(f"⚠️  recorder exited (code {self.proc.returncode}) {err.strip()} — restarting in {backoff}s")
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 60)
                try:
                    self._spawn()
                except Exception as e:  # noqa: BLE001
                    log(f"⚠️  could not (re)start recorder: {e}")
                    self.proc = None
                    self._stop.wait(backoff)
                    continue
            else:
                backoff = 5
            self._stop.wait(3)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    # ---- buffer inspection ----

    def _probe(self, p: Path) -> MediaProbe | None:
        try:
            st = p.stat()
        except FileNotFoundError:
            return None
        key = (p.name, st.st_size, st.st_mtime_ns)
        if key not in self._probe_cache:
            if self.ffprobe:
                self._probe_cache[key] = probe_media(self.ffprobe, p)
            else:  # no ffprobe: assume a nominal full segment with A+V
                self._probe_cache[key] = MediaProbe(float(self.seg_sec), True, True)
            if len(self._probe_cache) > self.wrap * 4:
                self._probe_cache.pop(next(iter(self._probe_cache)))
        return self._probe_cache[key]

    def completed_segments(self) -> list[tuple[Path, MediaProbe]]:
        """Finished segments, oldest→newest, each with a probe (the in-progress one is excluded)."""
        with self._lock:
            segs = []
            for p in self.buf_dir.glob("buf_*.ts"):
                try:
                    st = p.stat()
                except FileNotFoundError:
                    continue
                if st.st_size > 0:
                    segs.append((st.st_mtime_ns, p))
            segs.sort()
            done = [p for _, p in segs[:-1]]  # newest = still being written
            out = []
            for p in done:
                pr = self._probe(p)
                if pr and pr.duration > 0 and pr.has_video:
                    out.append((p, pr))
            return out

    def buffered(self) -> tuple[float, bool]:
        """(seconds of contiguous valid A/V at the tail of the buffer, tail segments have audio)."""
        segs = self.completed_segments()
        if not segs:
            return 0.0, False
        want_audio = segs[-1][1].has_audio
        total = 0.0
        for _p, pr in reversed(segs):
            if pr.has_audio != want_audio:
                break
            total += pr.duration
        return total, want_audio

    def snapshot(self, seconds: float, dest: Path) -> Path:
        """Concatenate the newest finished segments covering at least `seconds` into one .ts file."""
        segs = self.completed_segments()
        if not segs:
            raise RuntimeError("DVR buffer is empty — stream not recording yet")
        chosen: list[Path] = []
        total = 0.0
        for p, pr in reversed(segs):
            chosen.append(p)
            total += pr.duration
            if total >= seconds + self.seg_sec:  # one extra segment of slack for keyframe alignment
                break
        chosen.reverse()
        with open(dest, "wb") as out:
            for p in chosen:
                try:
                    out.write(p.read_bytes())
                except FileNotFoundError:  # wrapped/overwritten between listing and reading
                    continue
        return dest


# ─────────────────────────── chat sources ───────────────────────────
# Each source yields ~1 s batches of ChatMsg (ts = epoch seconds, chat-server clock when available).
# The spike detector only ever sees the timestamps; author/text feed the on-screen chat overlay.

@dataclass
class ChatMsg:
    ts: float
    author: str
    text: str
    color: str | None = None  # "#RRGGBB" when the platform provides one (Twitch)


_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F\U0000200D\U000E0000-\U000E007F]")
_SHORTCODE_RE = re.compile(r":[A-Za-z0-9_\-]{2,40}:")  # YouTube custom emoji shortcodes, e.g. :yt: :face-blue-smiling:
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


def clean_chat_text(text: str) -> str:
    """Drop colour emoji / shortcodes / control chars (flat fonts would render them as tofu boxes)."""
    text = _SHORTCODE_RE.sub(" ", text or "")
    text = _EMOJI_RE.sub(" ", text)
    text = _CTRL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def _ts_from_usec(v) -> float | None:
    try:
        return int(v) / 1_000_000
    except (TypeError, ValueError):
        return None


def chat_pytchat(video_id: str) -> Iterator[list[ChatMsg]]:
    import pytchat  # type: ignore

    chat = pytchat.create(video_id=video_id, interruptable=False)
    log("💬 chat source: pytchat")
    while chat.is_alive():
        batch: list[ChatMsg] = []
        for c in chat.get().items:
            ts = getattr(c, "timestamp", None)
            author = getattr(getattr(c, "author", None), "name", "") or ""
            batch.append(ChatMsg(ts / 1000 if ts else time.time(), author, getattr(c, "message", "") or ""))
        yield batch
        time.sleep(1.0)
    try:
        chat.raise_for_status()
    except Exception as e:  # noqa: BLE001
        log(f"pytchat ended: {e}")


def _runs_text(node) -> str:
    """YouTube {simpleText} / {runs:[{text}|{emoji}]} -> plain text (emoji runs are dropped)."""
    if not isinstance(node, dict):
        return ""
    if "simpleText" in node:
        return str(node["simpleText"])
    return "".join(str(r.get("text", "")) for r in node.get("runs", []) if isinstance(r, dict))


def _walk_messages(obj, out: list[ChatMsg]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.endswith("MessageRenderer") and isinstance(v, dict) and "timestampUsec" in v:
                ts = _ts_from_usec(v["timestampUsec"])
                out.append(ChatMsg(ts if ts else time.time(), _runs_text(v.get("authorName")),
                                   _runs_text(v.get("message"))))
            else:
                _walk_messages(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_messages(v, out)


def chat_ytdlp(url: str, work: Path) -> Iterator[list[ChatMsg]]:
    chat_dir = work / "chat"
    chat_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "yt_dlp", "--quiet", "--no-warnings", "--skip-download",
        "--write-subs", "--sub-langs", "live_chat", "-o", str(chat_dir / "chat.%(ext)s"), url,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    log("💬 chat source: yt-dlp live_chat (tip: `.venv/bin/pip install pytchat` for lower latency)")
    fh = None
    buf = ""
    try:
        while True:
            if fh is None:
                files = sorted(chat_dir.glob("*live_chat.json*"))
                if files:
                    fh = open(files[0], "r", encoding="utf-8", errors="ignore")
            batch: list[ChatMsg] = []
            if fh is not None:
                buf += fh.read()
                *lines, buf = buf.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        _walk_messages(json.loads(line), batch)
                    except json.JSONDecodeError:
                        continue
            if proc.poll() is not None and not batch:
                err = (proc.stderr.read() or b"").decode(errors="ignore").strip() if proc.stderr else ""
                log(f"yt-dlp chat downloader exited (code {proc.returncode}). {err[-300:]}")
                return
            yield batch
            time.sleep(1.0)
    finally:
        if fh:
            fh.close()
        if proc.poll() is None:
            proc.terminate()


def chat_api(video_id: str, token: str) -> Iterator[list[ChatMsg]]:
    from uploader import get_authenticated_service  # type: ignore

    yt = get_authenticated_service(token_name=token)
    items = yt.videos().list(part="liveStreamingDetails", id=video_id).execute().get("items", [])
    chat_id = items[0]["liveStreamingDetails"].get("activeLiveChatId") if items else None
    if not chat_id:
        raise RuntimeError("No activeLiveChatId (stream not live or chat disabled)")
    log("💬 chat source: YouTube Data API polling (uses quota!)")
    page = None
    while True:
        resp = yt.liveChatMessages().list(liveChatId=chat_id, part="snippet,authorDetails", pageToken=page,
                                          maxResults=2000).execute()
        page = resp.get("nextPageToken")
        batch: list[ChatMsg] = []
        for it in resp.get("items", []):
            sn = it.get("snippet", {})
            try:
                ts = datetime.fromisoformat(sn.get("publishedAt").replace("Z", "+00:00")).timestamp()
            except Exception:  # noqa: BLE001
                ts = time.time()
            batch.append(ChatMsg(ts, it.get("authorDetails", {}).get("displayName", ""),
                                 sn.get("displayMessage", "")))
        yield batch
        time.sleep(max(resp.get("pollingIntervalMillis", 5000) / 1000, 2.0))

# ---- Twitch: anonymous IRC (no OAuth app, no API key) ----

TWITCH_IRC_HOST = "irc.chat.twitch.tv"
TWITCH_IRC_PORT = 6667
TWITCH_IDLE_TIMEOUT = 360.0  # Twitch PINGs ~every 5 min; silence longer than this = dead socket
_TAG_UNESCAPE = {":": ";", "s": " ", "\\": "\\", "r": "\r", "n": "\n"}


@dataclass
class IrcMessage:
    tags: dict[str, str]
    prefix: str | None
    command: str
    params: list[str]


def _unescape_tag(v: str) -> str:
    out, i = [], 0
    while i < len(v):
        if v[i] == "\\" and i + 1 < len(v):
            out.append(_TAG_UNESCAPE.get(v[i + 1], v[i + 1]))
            i += 2
        else:
            if v[i] != "\\":  # a trailing lone backslash is dropped (IRCv3 spec)
                out.append(v[i])
            i += 1
    return "".join(out)


def parse_irc_line(line: str) -> IrcMessage | None:
    """Parse one IRC line (IRCv3 tags + prefix + command + params, trailing ':' param)."""
    line = line.rstrip("\r\n")
    if not line:
        return None
    tags: dict[str, str] = {}
    prefix = None
    if line.startswith("@"):
        raw, _, line = line[1:].partition(" ")
        for kv in raw.split(";"):
            k, _, v = kv.partition("=")
            if k:
                tags[k] = _unescape_tag(v)
        line = line.lstrip(" ")
    if line.startswith(":"):
        prefix, _, line = line[1:].partition(" ")
        line = line.lstrip(" ")
    head, sep, trailing = line.partition(" :")
    parts = head.split()
    if not parts:
        return None
    params = parts[1:] + ([trailing] if sep else [])
    return IrcMessage(tags, prefix, parts[0].upper(), params)


def twitch_message_ts(msg: IrcMessage) -> float:
    """Server-side send time (tmi-sent-ts, ms) when the tags capability is on, else local clock."""
    try:
        return int(msg.tags["tmi-sent-ts"]) / 1000
    except (KeyError, ValueError):
        return time.time()


class TwitchChatError(RuntimeError):
    pass


def _twitch_connect(channel: str, host: str, port: int) -> tuple[socket.socket, str]:
    nick = f"justinfan{random.randint(10000, 99999)}"
    sock = socket.create_connection((host, port), timeout=15)
    sock.sendall((
        "CAP REQ :twitch.tv/tags twitch.tv/commands\r\n"  # tags → tmi-sent-ts timestamps
        "PASS oauth:none\r\n"  # anonymous read-only login: any PASS + justinfanNNNN nick
        f"NICK {nick}\r\n"
        f"JOIN #{channel}\r\n"
    ).encode())
    sock.settimeout(1.0)
    return sock, nick


def twitch_chat_msg(msg: IrcMessage) -> ChatMsg:
    """PRIVMSG -> ChatMsg (display-name/color tags when the tags capability is on, else the nick prefix)."""
    author = msg.tags.get("display-name") or (msg.prefix or "").split("!", 1)[0]
    text = msg.params[-1] if len(msg.params) > 1 else ""
    if text.startswith("\x01ACTION ") and text.endswith("\x01"):  # /me message
        text = text[8:-1]
    return ChatMsg(twitch_message_ts(msg), author, text, msg.tags.get("color") or None)


def chat_twitch(channel: str, host: str = TWITCH_IRC_HOST, port: int = TWITCH_IRC_PORT,
                reconnect: bool = True) -> Iterator[list[ChatMsg]]:
    """Yields ~1 s batches of PRIVMSG ChatMsg from #channel via anonymous Twitch IRC."""
    channel = channel.lower().lstrip("#")
    target = f"#{channel}"
    backoff = 2.0
    first = True
    while True:
        try:
            sock, nick = _twitch_connect(channel, host, port)
        except OSError as e:
            if first or not reconnect:
                raise TwitchChatError(f"cannot connect to {host}:{port}: {e}") from e
            log(f"⚠️  Twitch IRC connect failed ({e}); retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        log(f"💬 chat source: Twitch IRC {host}:{port} as {nick} → {target} (anonymous)")
        first = False
        buf = b""
        batch: list[ChatMsg] = []
        last_yield = last_rx = time.monotonic()
        reason = "connection closed"
        try:
            while True:
                try:
                    data = sock.recv(65536)
                    if not data:
                        break
                    last_rx = time.monotonic()
                    buf += data
                except socket.timeout:
                    if time.monotonic() - last_rx > TWITCH_IDLE_TIMEOUT:
                        reason = "no data/PING for too long"
                        break
                *lines, buf = buf.split(b"\r\n")
                restart = False
                for raw in lines:
                    msg = parse_irc_line(raw.decode("utf-8", errors="replace"))
                    if msg is None:
                        continue
                    if msg.command == "PRIVMSG" and msg.params and msg.params[0].lower() == target:
                        batch.append(twitch_chat_msg(msg))
                    elif msg.command == "PING":
                        sock.sendall(f"PONG :{msg.params[-1] if msg.params else 'tmi.twitch.tv'}\r\n".encode())
                    elif msg.command == "366":  # end of NAMES = JOIN confirmed
                        log(f"✅ joined Twitch chat {target}")
                        backoff = 2.0
                    elif msg.command == "RECONNECT":  # server maintenance: reconnect right away
                        reason, restart = "server requested RECONNECT", True
                    elif msg.command == "NOTICE":
                        text = msg.params[-1] if msg.params else ""
                        mid = msg.tags.get("msg-id", "")
                        if "login" in text.lower() and "unsuccessful" in text.lower():
                            raise TwitchChatError(f"Twitch IRC login rejected: {text}")
                        if mid in ("msg_channel_suspended", "msg_banned", "tos_ban"):
                            raise TwitchChatError(f"cannot read {target}: {mid} ({text})")
                        log(f"ℹ️  Twitch NOTICE: {text}")
                if restart:
                    break
                if time.monotonic() - last_yield >= 1.0:
                    yield batch
                    batch = []
                    last_yield = time.monotonic()
        except OSError as e:
            reason = f"socket error: {e}"
        finally:
            try:
                sock.close()
            except OSError:
                pass
        if batch:
            yield batch
        if not reconnect:
            log(f"📴 Twitch IRC ended: {reason}")
            return
        log(f"⚠️  Twitch IRC dropped ({reason}); reconnecting in {backoff:.0f}s")
        time.sleep(backoff)
        backoff = min(backoff * 2, 60)


def open_chat(source: str, url: str, video_id: str, token: str, work: Path) -> Iterator[list[ChatMsg]]:
    channel = twitch_channel_from_url(url)
    if channel:
        if source not in ("auto", "twitch"):
            sys.exit(f"❌ --chat-source {source} is YouTube-only; Twitch URLs use the IRC reader (auto/twitch)")
        try:
            gen = chat_twitch(channel)
            first = next(gen)  # prime: connect errors surface here, before the main loop
        except TwitchChatError as e:
            sys.exit(f"❌ Twitch chat: {e}")
        return itertools.chain([first], gen)
    if source == "twitch":
        sys.exit("❌ --chat-source twitch needs a Twitch URL (https://www.twitch.tv/<channel>)")
    if source in ("auto", "pytchat"):
        if has_module("pytchat"):
            try:
                gen = chat_pytchat(video_id)
                first = next(gen)  # generators are lazy: prime it so connect errors fall back here
                return itertools.chain([first], gen)
            except Exception as e:  # noqa: BLE001
                log(f"⚠️  pytchat failed ({e}); falling back")
        else:
            msg = "pytchat not installed → `.venv/bin/pip install pytchat`"
            if source == "pytchat":
                sys.exit(f"❌ {msg}")
            log(f"ℹ️  {msg}. Using yt-dlp live chat fallback.")
    if source == "api":
        return chat_api(video_id, token)
    return chat_ytdlp(url, work)


# ─────────────────────────── spike detection ───────────────────────────

@dataclass
class SpikeDetector:
    mult: float
    min_msgs: int = 8
    window: float = WINDOW_SEC
    baseline: float = BASELINE_SEC
    stamps: deque = field(default_factory=deque)
    first_ts: float | None = None
    now: float = 0.0

    def add(self, stamps: list[float]) -> None:
        for t in stamps:
            self.stamps.append(t)
            self.now = max(self.now, t)
            if self.first_ts is None or t < self.first_ts:
                self.first_ts = t

    def evaluate(self) -> tuple[bool, int, float, float]:
        """Returns (is_spike, msgs_in_window, window_rate, baseline_rate) on the chat clock."""
        if self.first_ts is None:
            return False, 0, 0.0, 0.0
        horizon = self.now - self.window - self.baseline
        while self.stamps and self.stamps[0] < horizon:
            self.stamps.popleft()
        split = self.now - self.window
        n_win = sum(1 for t in self.stamps if t > split)
        n_base = sum(1 for t in self.stamps if horizon <= t <= split)
        win_rate = n_win / self.window
        base_rate = n_base / self.baseline
        warmed = (self.now - self.first_ts) >= self.window + self.baseline
        spike = warmed and n_win >= self.min_msgs and win_rate >= self.mult * base_rate
        return spike, n_win, win_rate, base_rate


# ─────────────────────────── transcription ───────────────────────────

_WHISPER = None
_WHISPER_LOCK = threading.Lock()


def transcribe_words(audio: Path, model_size: str, language: str | None) -> list[dict]:
    global _WHISPER
    if not has_module("faster_whisper"):
        log("⚠️  faster-whisper missing (`.venv/bin/pip install faster-whisper`) — no subtitles")
        return []
    from faster_whisper import WhisperModel  # type: ignore

    with _WHISPER_LOCK:
        if _WHISPER is None:
            log(f"🧠 loading faster-whisper '{model_size}'…")
            _WHISPER = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _info = _WHISPER.transcribe(
            str(audio), word_timestamps=True, vad_filter=True, language=language, beam_size=1
        )
        words = []
        for seg in segments:
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    words.append({"word": text, "start": float(w.start), "end": float(w.end)})
    return words


def chunk_words(words: list[dict], max_words: int = 4, max_gap: float = 0.6) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur and (len(cur) >= max_words or w["start"] - cur[-1]["end"] > max_gap or cur[-1]["word"][-1] in ".!?"):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def caption_events(words: list[dict], clip_len: float) -> list[tuple[float, float, list[str], int]]:
    """(start, end, chunk_words, active_index) — one event per spoken word (karaoke highlight)."""
    events = []
    for chunk in chunk_words(words):
        texts = [re.sub(r"\s+", " ", w["word"]).upper() for w in chunk]
        for i, w in enumerate(chunk):
            start = w["start"]
            end = chunk[i + 1]["start"] if i + 1 < len(chunk) else max(w["end"], start + 0.25)
            end = min(end, clip_len)
            if end > start:
                events.append((start, end, texts, i))
    return events


# ─────────────────────────── subtitle burn-in ───────────────────────────

def _ass_time(t: float) -> str:
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def write_ass(events, hook: str, path: Path) -> None:
    lines = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920", "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        "Style: Sub,Arial Black,92,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,7,4,2,60,60,520,1",
        "Style: Hook,Arial Black,70,&H0000FFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,6,3,8,60,60,230,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if hook:
        lines.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(HOOK_SEC)},Hook,,0,0,0,,{{\\fad(0,250)}}{hook}")
    for start, end, texts, idx in events:
        parts = [
            "{\\c&H00FFFF&\\fscx115\\fscy115}" + t + "{\\c&HFFFFFF&\\fscx100\\fscy100}" if i == idx else t
            for i, t in enumerate(texts)
        ]
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Sub,,0,0,0,,{' '.join(parts)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _font(size: int, candidates: list[str] | None = None):
    from PIL import ImageFont

    for p in candidates or FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render_caption_png(texts: list[str], active: int | None, path: Path, size: int = 88, color_all=None) -> None:
    """Bold white words with black stroke; active word yellow + slightly larger. 1080-wide transparent PNG."""
    from PIL import Image, ImageDraw

    base, big = _font(size), _font(int(size * 1.15))
    W, H, stroke = 1080, 420, 8
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    space = draw.textlength(" ", font=base)

    def width(i: int) -> float:
        return draw.textlength(texts[i], font=big if i == active else base)

    # greedy line wrap inside 960 px
    rows, row, row_w = [], [], 0.0
    for i in range(len(texts)):
        w = width(i)
        if row and row_w + space + w > 960:
            rows.append(row)
            row, row_w = [], 0.0
        row_w += (space if row else 0) + w
        row.append(i)
    if row:
        rows.append(row)

    line_h = int(size * 1.25)
    y = (H - line_h * len(rows)) // 2 + line_h // 2
    for r in rows:
        total = sum(width(i) for i in r) + space * (len(r) - 1)
        x = (W - total) / 2
        for i in r:
            f = big if i == active else base
            fill = color_all or ((255, 230, 0) if i == active else (255, 255, 255))
            draw.text((x + 4, y + 6), texts[i], font=f, fill=(0, 0, 0, 160), anchor="lm", stroke_width=stroke)
            draw.text((x, y), texts[i], font=f, fill=fill, anchor="lm", stroke_width=stroke, stroke_fill=(0, 0, 0))
            x += width(i) + space
        y += line_h
    img.save(path)


# ─────────────────────────── live chat overlay ───────────────────────────

_AUTHOR_PALETTE = [(255, 82, 82), (255, 171, 64), (255, 235, 59), (105, 240, 174), (64, 196, 255),
                   (124, 77, 255), (255, 64, 129), (0, 230, 118), (24, 255, 255), (255, 145, 0)]


def _author_rgb(m: ChatMsg) -> tuple[int, int, int]:
    """Platform colour (Twitch) brightened for readability on dark glass, else a stable hash colour."""
    if m.color and re.fullmatch(r"#[0-9A-Fa-f]{6}", m.color):
        r, g, b = (int(m.color[i:i + 2], 16) for i in (1, 3, 5))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        if lum < 110:  # dark names vanish on the dark box → lift toward white
            k = 0.55
            r, g, b = (int(c + (255 - c) * k) for c in (r, g, b))
        return r, g, b
    h = sum(ord(c) * 31 ** i for i, c in enumerate(m.author.lower()[:16]))
    return _AUTHOR_PALETTE[h % len(_AUTHOR_PALETTE)]


def pick_overlay_chat(recent: list[ChatMsg], limit: int) -> list[ChatMsg]:
    """Newest `limit` chat lines with displayable text (cleaned), oldest→newest."""
    out: list[ChatMsg] = []
    for m in reversed(recent):
        text = clean_chat_text(m.text)
        author = clean_chat_text(m.author)[:24] or "viewer"
        if text:
            out.append(ChatMsg(m.ts, author, text, m.color))
        if len(out) >= limit:
            break
    out.reverse()
    return out


def _fit(draw, text: str, font, max_w: float) -> str:
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "…", font=font) > max_w:
        text = text[:-1]
    return text.rstrip() + "…"


class ChatBoxRenderer:
    """Dark glass chat panel (1080-wide transparent PNG) with the newest message at the bottom."""

    W = 1080
    BOX_X0, BOX_X1 = 70, 1010
    HEADER_H = 66
    ROW_H = 68
    PAD = 14
    SHADOW = 26

    def __init__(self, rows: int = 5):
        self.rows = rows
        self.f_author = _font(34, CHAT_AUTHOR_FONTS)
        self.f_text = _font(34, CHAT_TEXT_FONTS)
        self.f_head = _font(28, CHAT_AUTHOR_FONTS)
        self.box_h = self.HEADER_H + self.rows * self.ROW_H + self.PAD
        self.H = self.box_h + self.SHADOW * 2
        self._base = None

    def _panel(self):
        """Shadow + glass box + header — identical for every frame, rendered once."""
        from PIL import Image, ImageDraw, ImageFilter

        if self._base is not None:
            return self._base.copy()
        img = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        x0, x1, y0 = self.BOX_X0, self.BOX_X1, self.SHADOW // 2
        y1 = y0 + self.box_h
        shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle((x0 + 6, y0 + 14, x1 + 6, y1 + 14), 30, fill=(0, 0, 0, 150))
        img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14)))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((x0, y0, x1, y1), 30, fill=(10, 10, 16, 205), outline=(255, 255, 255, 48), width=2)
        # header: red LIVE dot, title, gold accent rule
        cy = y0 + self.HEADER_H // 2 + 2
        d.ellipse((x0 + 30, cy - 9, x0 + 48, cy + 9), fill=(255, 45, 70, 255))
        d.ellipse((x0 + 24, cy - 15, x0 + 54, cy + 15), outline=(255, 45, 70, 110), width=3)
        d.text((x0 + 68, cy), "LIVE CHAT", font=self.f_head, fill=(255, 255, 255, 235), anchor="lm")
        d.line((x0 + 24, y0 + self.HEADER_H, x1 - 24, y0 + self.HEADER_H), fill=(255, 205, 60, 150), width=2)
        self._base = img
        return img.copy()

    def render(self, history: list[ChatMsg], path: Path, pop: float = 1.0) -> None:
        """history = messages shown so far (oldest→newest). pop∈(0,1]: newest row sliding/fading in."""
        from PIL import Image, ImageDraw

        img = self._panel()
        x0, x1 = self.BOX_X0, self.BOX_X1
        top = self.SHADOW // 2 + self.HEADER_H + 4
        area_h = self.rows * self.ROW_H
        layer = Image.new("RGBA", (x1 - x0, area_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        shown = history[-(self.rows + 1):]  # one extra row scrolls out the top while the new one pops in
        slide = (1.0 - pop) * self.ROW_H
        max_w = (x1 - x0) - 60
        for k, m in enumerate(reversed(shown)):  # k=0 → newest, bottom slot
            y = area_h - (k + 1) * self.ROW_H + slide
            if y + self.ROW_H <= 0 or y >= area_h:
                continue
            alpha = int(255 * pop) if k == 0 else 255
            cy = y + self.ROW_H / 2
            if k == 0:  # highlight the freshest message
                d.rounded_rectangle((12, y + 5, x1 - x0 - 12, y + self.ROW_H - 5), 16,
                                    fill=(255, 255, 255, int(26 * pop)))
                d.rounded_rectangle((12, y + 12, 18, y + self.ROW_H - 12), 3, fill=(255, 205, 60, alpha))
            name = _fit(d, m.author, self.f_author, 300)
            r, g, b = _author_rgb(m)
            d.text((30, cy), name, font=self.f_author, fill=(r, g, b, alpha), anchor="lm")
            nx = 30 + d.textlength(name + ": ", font=self.f_author)
            d.text((30 + d.textlength(name, font=self.f_author), cy), ":", font=self.f_author,
                   fill=(255, 255, 255, int(alpha * 0.7)), anchor="lm")
            body = _fit(d, m.text, self.f_text, max_w - nx + 30)
            d.text((nx, cy), body, font=self.f_text, fill=(255, 255, 255, alpha), anchor="lm")
        img.alpha_composite(layer, (x0, top))
        img.save(path)


def chat_burst_overlays(msgs: list[ChatMsg], out_dir: Path, y: int = CHAT_OVERLAY_Y,
                        duration: float = HOOK_SEC, rows: int = 5) -> list[tuple[Path, float, float, int]]:
    """PNG overlay schedule: messages pop in one by one (slide+fade) and scroll inside the box for `duration` s."""
    if not msgs:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    rend = ChatBoxRenderer(rows=rows)
    t0, t_last = 0.2, duration - 0.9  # first pop just after the hook lands; last pop leaves ~1 s to read
    step = min(0.55, (t_last - t0) / max(1, len(msgs) - 1))
    tween = min(0.1, step / 2)
    overlays: list[tuple[Path, float, float, int]] = []
    for i in range(len(msgs)):
        t = t0 + i * step
        t_next = t0 + (i + 1) * step if i + 1 < len(msgs) else duration
        hist = msgs[: i + 1]
        mid = out_dir / f"chat_{i:02d}a.png"
        full = out_dir / f"chat_{i:02d}b.png"
        rend.render(hist, mid, pop=0.45)
        rend.render(hist, full, pop=1.0)
        overlays.append((mid, t, t + tween, y))
        overlays.append((full, t + tween, t_next, y))
    return overlays


def overlay_graph(base_chain: str, overlays: list[tuple[Path, float, float, int]],
                  first_input: int = 1) -> tuple[list[str], list[str], str]:
    """-> (extra -i args, filter_complex parts, final label). Each PNG is overlaid at (0, y) only in [s, e)."""
    extra: list[str] = []
    graph = [f"[0:v]{base_chain}[v0]"]
    last = "v0"
    for i, (p, s, e, y) in enumerate(overlays, start=first_input):
        extra += ["-i", str(p)]
        # gte/lt (not between) so back-to-back frames never double-draw on the boundary frame
        graph.append(f"[{last}][{i}:v]overlay=0:{y}:enable='gte(t,{s:.3f})*lt(t,{e:.3f})'[v{i}]")
        last = f"v{i}"
    return extra, graph, last


# ─────────────────────────── clip pipeline ───────────────────────────

@dataclass
class ClipJob:
    triggered_at: float
    n_win: int
    win_rate: float
    base_rate: float
    chat: list[ChatMsg] = field(default_factory=list)  # recent chat around the spike, oldest→newest


class Clipper:
    def __init__(self, args, stream: StreamInfo, recorder: RollingRecorder, ffmpeg: str, work: Path):
        self.args, self.stream, self.recorder, self.ffmpeg = args, stream, recorder, ffmpeg
        self.ffprobe = find_ffprobe(ffmpeg)
        self.libass = ffmpeg_has_libass(ffmpeg)
        self.work = work
        self.jobs: queue.Queue[ClipJob | None] = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        log(f"📝 subtitle renderer: {'libass (ffmpeg)' if self.libass else 'Pillow overlays'}")

    def start(self) -> None:
        self.thread.start()

    def submit(self, job: ClipJob) -> None:
        self.jobs.put(job)

    def stop(self, wait: float = 600) -> None:
        self.jobs.put(None)
        self.thread.join(timeout=wait)

    def _worker(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                self.process(job)
            except Exception as e:  # noqa: BLE001
                log(f"❌ clip failed: {e}")

    def process(self, job: ClipJob) -> None:
        a = self.args
        if a.post_roll > 0:
            log(f"⏳ post-roll {a.post_roll}s so the reaction lands inside the clip…")
            time.sleep(a.post_roll)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUT_ROOT / f"{self.stream.video_id}_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        raw = self.recorder.snapshot(a.clip_sec, out_dir / "raw.ts")

        pr = probe_media(self.ffprobe, raw)
        dur = pr.duration if pr and pr.duration > 0 else float(a.clip_sec)
        has_audio = pr.has_audio if pr else True
        if pr and not pr.has_video:
            raise RuntimeError("DVR snapshot has no video stream")
        if dur < a.min_buffer:
            raise RuntimeError(f"DVR snapshot only {dur:.1f}s (< --min-buffer {a.min_buffer}s) — skipping clip")
        clip_len = min(float(a.clip_sec), dur)
        start = max(0.0, dur - clip_len)
        log(f"✂️  cutting last {clip_len:.1f}s (buffer {dur:.1f}s, audio={'yes' if has_audio else 'NO'})")

        words: list[dict] = []
        if has_audio:
            audio = out_dir / "audio.wav"
            try:
                run_ffmpeg(self.ffmpeg, ["-ss", f"{start:.3f}", "-i", str(raw), "-t", f"{clip_len:.3f}",
                                         "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", str(audio)],
                           "audio extract")
                words = transcribe_words(audio, a.whisper_model, a.language)
            except RuntimeError as e:
                log(f"⚠️  {e} — continuing without captions")
        else:
            log("⚠️  no audio in buffer — rendering with silent track, no word captions")
        (out_dir / "words.json").write_text(json.dumps(words, ensure_ascii=False, indent=1), encoding="utf-8")
        events = caption_events(words, clip_len)
        ratio = job.win_rate / job.base_rate if job.base_rate > 0 else float(job.n_win)
        hook = a.hook_text or f"CHAT EXPLODED x{ratio:.0f}"
        chat_overlays: list[tuple[Path, float, float, int]] = []
        if job.chat:
            try:
                chat_overlays = chat_burst_overlays(job.chat, out_dir / "chat", duration=min(HOOK_SEC, clip_len))
                (out_dir / "chat.json").write_text(json.dumps([m.__dict__ for m in job.chat], ensure_ascii=False,
                                                              indent=1), encoding="utf-8")
                log(f"💬 chat overlay: {len(job.chat)} messages bursting in the first {HOOK_SEC:.0f}s")
            except Exception as e:  # noqa: BLE001 — never lose a clip over the overlay
                log(f"⚠️  chat overlay failed ({e}) — rendering without it")
                chat_overlays = []

        final = out_dir / f"short_{self.stream.video_id}_{stamp}.mp4"
        crop = "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',scale=1080:1920:flags=lanczos,setsar=1"
        enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-r", "30"]
        inputs = ["-ss", f"{start:.3f}", "-i", str(raw)]
        enc = ["-t", f"{clip_len:.3f}", *enc]
        # audio source: the stream's own track, or generated silence (YouTube Shorts want an audio track)
        silent = [] if has_audio else ["-f", "lavfi", "-t", f"{clip_len:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]

        if self.libass:
            ass = out_dir / "subs.ass"
            write_ass(events, hook, ass)
            esc = str(ass).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            extra, graph, last = overlay_graph(f"setpts=PTS-STARTPTS,{crop},subtitles='{esc}'", chat_overlays)
            amap = f"{len(chat_overlays) + 1}:a:0" if silent else "0:a:0"  # silence input comes after the PNGs
            run_ffmpeg(self.ffmpeg, [*inputs, *extra, *silent, "-filter_complex", ";".join(graph),
                                     "-map", f"[{last}]", "-map", amap, *enc, str(final)], "render+subs")
        else:
            self._render_pillow(inputs, silent, crop, enc, events, hook, out_dir, final, chat_overlays)

        fp = probe_media(self.ffprobe, final)
        if fp and not (fp.has_video and fp.has_audio):
            raise RuntimeError(f"rendered clip is missing streams (video={fp.has_video}, audio={fp.has_audio})")

        log(f"✅ short ready: {final}")
        result = {"file": str(final), "source": a.url, "video_id": self.stream.video_id,
                  "triggered_at": datetime.fromtimestamp(job.triggered_at).isoformat(),
                  "window_msgs": job.n_win, "window_rate": round(job.win_rate, 2),
                  "baseline_rate": round(job.base_rate, 2), "chat_overlay_msgs": len(job.chat),
                  "uploaded_id": None}

        if not a.no_upload:
            result["uploaded_id"] = self._upload(final, words)
        append_history(result)

    @staticmethod
    def _render_hook_banner_png(headline: str, subline: str, path: Path) -> None:
        """High-CTR Shorts cover banner (red viral pill + gold chat-spike subline), 1080x290."""
        from PIL import Image, ImageDraw

        W, H = 1080, 290
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        f_head = _font(64)
        f_sub = _font(54)

        hb = draw.textbbox((0, 0), headline, font=f_head)
        hw, hh = hb[2] - hb[0], hb[3] - hb[1]
        pad_x, pad_y = 44, 22
        rx0 = max(30, (W - hw) // 2 - pad_x)
        ry0 = 14
        rx1 = min(W - 30, rx0 + hw + pad_x * 2)
        ry1 = ry0 + hh + pad_y * 2

        # Drop shadow + high-contrast red Shorts pill with white border
        draw.rounded_rectangle((rx0 + 6, ry0 + 8, rx1 + 6, ry1 + 8), radius=26, fill=(0, 0, 0, 175))
        draw.rounded_rectangle((rx0, ry0, rx1, ry1), radius=26, fill=(225, 29, 46, 245),
                               outline=(255, 255, 255, 235), width=5)
        draw.text(((W - hw) // 2, ry0 + pad_y - hb[1]), headline, font=f_head,
                  fill=(255, 255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0, 255))

        if subline:
            sb = draw.textbbox((0, 0), subline, font=f_sub)
            sw = sb[2] - sb[0]
            draw.text(((W - sw) // 2, ry1 + 22), subline, font=f_sub,
                      fill=(255, 230, 0, 255), stroke_width=7, stroke_fill=(0, 0, 0, 255))

        img.save(path)

    def _render_pillow(self, inputs, silent, crop, enc, events, hook, out_dir: Path, final: Path,
                       chat_overlays: list[tuple[Path, float, float, int]] | None = None) -> None:
        cap_dir = out_dir / "captions"
        cap_dir.mkdir(exist_ok=True)
        overlays: list[tuple[Path, float, float, int]] = []
        if hook:
            p = cap_dir / "hook.png"
            headline = random.choice(VIRAL_COVER_HOOKS)
            self._render_hook_banner_png(headline, hook, p)
            overlays.append((p, 0.0, HOOK_SEC, 95))
        overlays += chat_overlays or []
        for n, (s, e, texts, idx) in enumerate(events):
            p = cap_dir / f"cap_{n:04d}.png"
            render_caption_png(texts, idx, p)
            overlays.append((p, s, e, 1190))

        extra_inputs, graph, last = overlay_graph(f"setpts=PTS-STARTPTS,{crop}", overlays)
        (out_dir / "filter_graph.txt").write_text(";\n".join(graph), encoding="utf-8")  # debug copy
        amap = f"{len(overlays) + 1}:a:0" if silent else "0:a:0"  # silence input comes after the PNGs
        run_ffmpeg(self.ffmpeg, [*inputs, *extra_inputs, *silent, "-filter_complex", ";".join(graph),
                                 "-map", f"[{last}]", "-map", amap, *enc, str(final)], "render+pillow subs")

    def _upload(self, final: Path, words: list[dict]) -> str | None:
        a = self.args
        try:
            from uploader import upload_short  # type: ignore
        except Exception as e:  # noqa: BLE001
            log(f"❌ cannot import uploader.upload_short: {e}")
            return None
        spoken = " ".join(w["word"] for w in words[:5]).strip()
        base_title = random.choice(VIRAL_TITLES)
        if spoken and len(base_title) + len(spoken) + 4 <= 100:
            title = base_title.replace("#shorts", f"({spoken}) #shorts")
        else:
            title = base_title
        title = re.sub(r"\s+", " ", title)[:100]
        desc = (f"{a.description}\n\nClipped live from: {a.url}\n"
                f"Stream: {self.stream.title}\n#shorts #gta6 #gaming #livestream")
        log(f"⬆️  uploading as '{title}' (token: {a.token})")
        try:
            vid = upload_short(str(final), title, description=desc, token_name=a.token)
        except Exception as e:  # noqa: BLE001
            log(f"❌ upload error: {e}")
            return None
        log(f"🚀 uploaded: https://youtube.com/shorts/{vid}" if vid else "⚠️  upload returned no id (queued?)")
        return vid


def append_history(entry: dict) -> None:
    try:
        hist = json.loads(HISTORY_FILE.read_text(encoding="utf-8")) if HISTORY_FILE.exists() else []
    except json.JSONDecodeError:
        hist = []
    hist.append(entry)
    HISTORY_FILE.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────────────── main loop ───────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time live-chat hype-spike clipper → vertical karaoke Short → YouTube upload.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url", required=True, help="YouTube live URL (watch?v=, /live/, youtu.be, @channel/live) or Twitch https://www.twitch.tv/<channel>")
    p.add_argument("--token", default="token_channel2.pickle", help="OAuth token pickle passed to upload_short(token_name=)")
    p.add_argument("--spike-mult", type=float, default=3.0, help="trigger when 5s rate >= mult × 30s baseline rate")
    p.add_argument("--min-msgs", type=int, default=8, help="minimum messages inside the 5s window to trigger")
    p.add_argument("--clip-sec", type=int, default=40, help="seconds of stream to capture (ending at trigger + post-roll)")
    p.add_argument("--cooldown", type=int, default=120, help="seconds to ignore new spikes after a trigger")
    p.add_argument("--post-roll", type=float, default=4.0, help="seconds to keep recording after the spike before cutting")
    p.add_argument("--min-buffer", type=float, default=20.0,
                   help="ignore spikes until the DVR holds at least this many seconds of valid A/V segments")
    p.add_argument("--no-upload", action="store_true", help="render clips locally, skip YouTube upload")
    p.add_argument("--chat-source", choices=["auto", "pytchat", "ytdlp", "api", "twitch"], default="auto",
                   help="auto = Twitch IRC for twitch.tv URLs; for YouTube pytchat if installed, else yt-dlp "
                        "live_chat; api = Data API polling (quota)")
    p.add_argument("--whisper-model", default="small", help="faster-whisper model size (tiny/base/small/medium/large-v3)")
    p.add_argument("--language", default=None, help="force transcription language (e.g. en, ru); default auto-detect")
    p.add_argument("--title-prefix", default="GTA 6 STREAM MOMENT", help="Shorts title prefix")
    p.add_argument("--hook-text", default="", help="top banner text for first 5s (default: 'CHAT EXPLODED xN')")
    p.add_argument("--chat-overlay-msgs", type=int, default=12,
                   help="recent chat messages burst on screen under the hook in the first 5s (0 = off)")
    p.add_argument("--description", default="The chat lost it at this exact moment 😳", help="Shorts description head")
    p.add_argument("--max-clips", type=int, default=0, help="stop after N clips (0 = unlimited)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not has_module("yt_dlp"):
        sys.exit("❌ yt-dlp missing → .venv/bin/pip install yt-dlp")
    if not has_module("PIL"):
        sys.exit("❌ Pillow missing → .venv/bin/pip install pillow")
    sys.path.insert(0, str(SCRIPT_DIR))

    ffmpeg = find_ffmpeg()
    log(f"🔎 resolving {args.url}")
    try:
        stream = resolve_stream(args.url)
    except RuntimeError as e:
        sys.exit(f"❌ {e}")
    if not stream.is_live:
        log("⚠️  yt-dlp reports this video is NOT live — chat/buffer may end immediately.")
    log(f"📡 {stream.title} [{stream.video_id}]")

    work = Path(tempfile.mkdtemp(prefix=f"liveclip_{stream.video_id}_"))
    buf_dir = work / "dvr"
    buf_dir.mkdir()
    recorder = RollingRecorder(ffmpeg, args.url, buf_dir, buffer_sec=max(args.clip_sec * 3, 120),
                               ffprobe=find_ffprobe(ffmpeg))
    recorder.start()
    clipper = Clipper(args, stream, recorder, ffmpeg, work)
    clipper.start()

    detector = SpikeDetector(mult=args.spike_mult, min_msgs=args.min_msgs)
    recent_chat: deque[ChatMsg] = deque(maxlen=CHAT_RECENT_MAX)
    last_trigger = 0.0
    clips = 0
    last_status = 0.0
    last_buf_warn = 0.0
    log(f"👀 watching chat: spike ≥ {args.spike_mult}× baseline & ≥ {args.min_msgs} msgs/5s, "
        f"clip {args.clip_sec}s, cooldown {args.cooldown}s, upload={'off' if args.no_upload else 'on'}")
    try:
        for batch in open_chat(args.chat_source, args.url, stream.video_id, args.token, work):
            detector.add([m.ts for m in batch])
            recent_chat.extend(batch)
            spike, n_win, win_rate, base_rate = detector.evaluate()
            now = time.time()
            if now - last_status >= 10:
                buf_sec, buf_audio = recorder.buffered()
                log(f"   chat {win_rate:5.2f} msg/s (5s) | baseline {base_rate:5.2f} msg/s (30s) | "
                    f"DVR {buf_sec:.0f}s audio={'yes' if buf_audio else 'no'}")
                last_status = now
            if spike and now - last_trigger >= args.cooldown:
                buf_sec, buf_audio = recorder.buffered()
                if buf_sec < args.min_buffer:
                    if now - last_buf_warn >= 10:
                        log(f"⏸  spike ignored: DVR has {buf_sec:.1f}s valid (need {args.min_buffer:.0f}s), "
                            f"audio={'yes' if buf_audio else 'no'}")
                        last_buf_warn = now
                    continue
                if not buf_audio and now - last_buf_warn >= 60:
                    log("⚠️  DVR segments carry no audio — clip will use a silent track")
                    last_buf_warn = now
                last_trigger = now
                clips += 1
                log(f"🔥 HYPE SPIKE #{clips}: {n_win} msgs/5s = {win_rate:.2f}/s vs baseline {base_rate:.2f}/s")
                chat = pick_overlay_chat(list(recent_chat), args.chat_overlay_msgs) if args.chat_overlay_msgs > 0 else []
                clipper.submit(ClipJob(now, n_win, win_rate, base_rate, chat))
                if args.max_clips and clips >= args.max_clips:
                    log("🏁 max clips reached")
                    break
        else:
            log("📴 chat stream ended")
    except KeyboardInterrupt:
        log("🛑 interrupted")
    finally:
        log("finishing queued clips…")
        clipper.stop()
        recorder.stop()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
