import concurrent.futures
import functools
import hashlib
import io
import json
import math
import os
import random
import sys
import textwrap
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from google import genai
from google.genai import types
from moviepy import AudioFileClip, VideoClip, concatenate_videoclips
from moviepy.audio.io.readers import FFMPEG_AudioReader
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# Replace end-of-file audio reads with silence instead of aborting the render.
_ORIGINAL_GET_FRAME = FFMPEG_AudioReader.get_frame


def _safe_get_frame(self: FFMPEG_AudioReader, tt: Any) -> np.ndarray:
    try:
        return _ORIGINAL_GET_FRAME(self, tt)
    except (OSError, IOError):
        if isinstance(tt, np.ndarray):
            return np.zeros((len(tt), self.nchannels), dtype=np.float64)
        return np.zeros((self.nchannels,), dtype=np.float64)


FFMPEG_AudioReader.get_frame = _safe_get_frame

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

WIDTH, HEIGHT, FPS = 1080, 1920, 30
MAX_SHORT_SECONDS = 175
try:
    EPISODES_PER_SHORT = min(
        4, max(1, int(os.getenv("EPISODES_PER_SHORT", "1")))
    )
except ValueError:
    EPISODES_PER_SHORT = 1

SERIES_HISTORY_FILE = BASE_DIR / "series_history.json"
ASSETS_DIR = BASE_DIR / "assets" / "images"
AUDIO_DIR = BASE_DIR.parent / "output" / "audio"
VIDEOS_DIR = BASE_DIR.parent / "output" / "videos"
for directory in (ASSETS_DIR, AUDIO_DIR, VIDEOS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ENABLED = os.getenv("ENABLE_VOICE", "0").lower() in {"1", "true", "yes"}

if not GEMINI_KEY or (VOICE_ENABLED and not ELEVEN_KEY):
    raise ValueError("GEMINI_API_KEY обязателен; ELEVENLABS_API_KEY нужен только при ENABLE_VOICE=1")

gemini_client = genai.Client(
    api_key=GEMINI_KEY,
    http_options=types.HttpOptions(
        api_version="v1beta",
        timeout=int(os.getenv("GEMINI_TIMEOUT_MS", "60000")),
        retry_options=types.HttpRetryOptions(attempts=2),
    ),
)
eleven_client = ElevenLabs(api_key=ELEVEN_KEY) if VOICE_ENABLED else None


def check_elevenlabs_quota(min_required_chars: int = 600) -> dict[str, Any] | None:
    """Проверяет оставшуюся квоту символов в ElevenLabs перед генерацией речи."""
    if not VOICE_ENABLED or eleven_client is None:
        return None
    try:
        sub = eleven_client.user.subscription.get()
        used = sub.character_count
        limit = sub.character_limit
        remaining = max(0, limit - used)
        percent = (used / limit * 100.0) if limit else 0.0
        reset_dt = ""
        if getattr(sub, "next_character_count_reset_unix", None):
            reset_dt = datetime.fromtimestamp(sub.next_character_count_reset_unix).strftime("%Y-%m-%d %H:%M")

        print(
            f"🎙️ [ElevenLabs] Тариф: {sub.tier} | "
            f"Использовано: {used:,} / {limit:,} ({percent:.1f}%) | "
            f"Осталось: {remaining:,} симв. (сброс {reset_dt})"
        )

        if remaining < min_required_chars:
            print(
                f"⚠️ [ElevenLabs Warning] Осталось всего {remaining} символов! "
                f"Для ролика требуется ~{min_required_chars}. Голос может быть обрезан или вызвать ошибку лимита."
            )
        return {
            "tier": sub.tier,
            "character_count": used,
            "character_limit": limit,
            "remaining": remaining,
            "percent_used": round(percent, 1),
            "reset_date": reset_dt,
        }
    except Exception as e:
        print(f"ℹ️ [ElevenLabs] Не удалось получить статус подписки (продолжаем штатно): {e}")
        return None


# All supported procedural canvas operations (100% implemented in draw_canvas_op)
CANVAS_OPS = {
    "bg_forest", "bg_room", "bg_corridor", "bg_street", "bg_water", "bg_facility",
    "bg_lab_board", "bg_space", "diagram_flow", "anatomy_xray", "chart_bars", "math_board",
    "nuke_mushroom", "thumb_rule",
    "door_closeup", "window_view", "valve_diagram", "dashed_zone",
    "stickman", "actor", "entity", "hands_reach", "key_palm",
    "padlock_chain", "building_grid", "highlight_ring", "light_beam",
    "weather", "rect", "circle", "line", "polygon", "phone_ui", "radio_kit", "callout",
}

_IMAGE_CACHE: dict[str, Image.Image] = {}
_GRADIENT_CACHE: dict[tuple[tuple[int, int, int], tuple[int, int, int]], Image.Image] = {}


@functools.lru_cache(maxsize=64)
def get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def get_gradient_bg(bg_top: tuple[int, int, int], bg_bottom: tuple[int, int, int]) -> Image.Image:
    key = (bg_top, bg_bottom)
    cached = _GRADIENT_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    image = Image.new("RGB", (WIDTH, HEIGHT), bg_top)
    draw = ImageDraw.Draw(image)
    for y_band in range(0, HEIGHT, 16):
        ratio = y_band / HEIGHT
        color = tuple(int(bg_top[i] * (1 - ratio) + bg_bottom[i] * ratio) for i in range(3))
        draw.rectangle((0, y_band, WIDTH, y_band + 16), fill=color)
    _GRADIENT_CACHE[key] = image
    return image.copy()


def _rgb(val: Any, default: tuple[int, int, int] = (200, 200, 200)) -> tuple[int, int, int]:
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        try:
            return tuple(max(0, min(255, int(val[index]))) for index in range(3))
        except (TypeError, ValueError):
            return default
    return default


def _clamp_number(value: Any, low: float, high: float, default: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def get_scene_duration(scene: dict[str, Any]) -> float:
    timing = str(scene.get("timing", "0-5")).split("-")
    try:
        return max(float(timing[1]) - float(timing[0]), 0.1)
    except (IndexError, ValueError):
        return 5.0


def load_series_history() -> dict[str, list[str]]:
    if SERIES_HISTORY_FILE.exists():
        try:
            data = json.loads(SERIES_HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    key: [str(value) for value in data.get(key, [])][-40:]
                    for key in ("used_titles", "used_locations", "used_premises")
                }
        except (OSError, ValueError, TypeError):
            pass
    return {"used_titles": [], "used_locations": [], "used_premises": []}


def save_series_history(
    history: dict[str, list[str]],
    series_title: str,
    location: str,
    premise: str,
) -> None:
    updated = {
        "used_titles": (history.get("used_titles", []) + [series_title])[-40:],
        "used_locations": (history.get("used_locations", []) + [location])[-40:],
        "used_premises": (history.get("used_premises", []) + [premise])[-40:],
    }
    SERIES_HISTORY_FILE.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def sanitize_canvas(canvas: Any) -> list[dict[str, Any]]:
    if not isinstance(canvas, list):
        return []
    numeric_limits = {
        "x": (0, WIDTH), "y": (0, HEIGHT), "x1": (0, WIDTH), "y1": (0, HEIGHT),
        "x2": (0, WIDTH), "y2": (0, HEIGHT), "tx": (0, WIDTH), "ty": (0, HEIGHT),
        "w": (10, WIDTH), "h": (10, HEIGHT), "r": (8, min(WIDTH, HEIGHT) // 2),
        "dx": (-WIDTH, WIDTH), "dy": (-HEIGHT, HEIGHT),
        "scale": (0.2, 3.5), "w_stroke": (1, 40), "eyes": (1, 8), "count": (1, 30),
    }
    clean: list[dict[str, Any]] = []
    for raw in canvas[:16]:
        if not isinstance(raw, dict):
            continue
        op_name = str(raw.get("op", "")).strip()
        if op_name not in CANVAS_OPS:
            continue
        item: dict[str, Any] = {"op": op_name}
        for key, (low, high) in numeric_limits.items():
            if key in raw:
                default = low if key in {"w", "h", "r", "scale", "w_stroke"} else 0
                value = _clamp_number(raw[key], low, high, default)
                item[key] = round(value, 3) if key in {"dx", "dy", "scale"} else int(value)
        for key in ("fill", "outline", "shirt", "skin", "hair", "color", "glow"):
            if key in raw:
                item[key] = _rgb(raw[key])
        for key in ("locked", "slip", "hood", "open", "bars", "broken", "infected"):
            if key in raw:
                item[key] = bool(raw[key])
        for key in ("header", "subtext", "label", "emotion", "pose", "name_tag",
                    "kind", "style", "frequency", "direction"):
            if key in raw:
                item[key] = str(raw[key])[:80]
        if "points" in raw and isinstance(raw["points"], list):
            pts: list[tuple[int, int]] = []
            for pt in raw["points"][:12]:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    px = int(_clamp_number(pt[0], 0, WIDTH, 540))
                    py = int(_clamp_number(pt[1], 0, HEIGHT, 800))
                    pts.append((px, py))
            if len(pts) >= 3:
                item["points"] = pts
        clean.append(item)
    return clean


# ============================================================================
# UNIVERSAL PROCEDURAL VECTOR RENDERER (ALL CANVAS_OPS IMPLEMENTED)
# ============================================================================

def draw_stickman(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    scale: float = 1.0,
    shirt: tuple[int, int, int] = (42, 92, 68),
    pants: tuple[int, int, int] = (28, 32, 40),
    skin: tuple[int, int, int] = (14, 14, 18),
    infected: bool = False,
) -> None:
    """Draws a secondary character in the signature 'ЧТО, ЕСЛИ?' flat-vector silhouette + cream rim-light style."""
    rim_col = (245, 216, 160) if not infected else (235, 65, 65)
    head_col = (14, 14, 18)
    cream = (244, 239, 227)
    if infected:
        shirt, pants = (52, 32, 36), (26, 22, 26)
    r = int(42 * scale)
    bw, bh = int(40 * scale), int(125 * scale)
    by = y + r - int(4 * scale)
    lh = int(125 * scale)
    # Legs with cream rim highlight
    for lx1, lx2 in ((x - bw + int(6 * scale), x - int(4 * scale)), (x + int(4 * scale), x + bw - int(6 * scale))):
        draw.rounded_rectangle((lx1 + int(3 * scale), by + bh - int(10 * scale), lx2 + int(3 * scale), by + bh + lh),
                               radius=int(10 * scale), fill=rim_col)
        draw.rounded_rectangle((lx1, by + bh - int(10 * scale), lx2, by + bh + lh),
                               radius=int(10 * scale), fill=pants)
    # Torso with cream rim highlight
    draw.rounded_rectangle((x - bw + int(4 * scale), by + int(2 * scale), x + bw + int(4 * scale), by + bh + int(2 * scale)),
                           radius=int(18 * scale), fill=rim_col)
    draw.rounded_rectangle((x - bw, by, x + bw, by + bh), radius=int(18 * scale), fill=shirt)
    # Pitch-black head with warm golden-cream rim crescent
    draw.ellipse((x - r + int(4 * scale), y - r + int(3 * scale), x + r + int(5 * scale), y + r + int(5 * scale)), fill=rim_col)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=head_col)
    # Eyes
    eye_col = (245, 55, 55) if infected else cream
    er = int(9 * scale)
    for ex in (x - int(14 * scale), x + int(14 * scale)):
        draw.ellipse((ex - er, y - int(4 * scale) - er, ex + er, y - int(4 * scale) + er), fill=eye_col)
        draw.ellipse((ex - int(4 * scale), y - int(4 * scale) - int(4 * scale), ex + int(4 * scale), y - int(4 * scale) + int(4 * scale)), fill=head_col)
    draw.line((x - int(10 * scale), y + int(14 * scale), x + int(10 * scale), y + int(14 * scale)), fill=cream, width=max(2, int(3 * scale)))


def _draw_actor(draw: ImageDraw.ImageDraw, cmd: dict[str, Any], dx: int, dy: int) -> None:
    """Draws the main character in the exact 'ЧТО, ЕСЛИ?' flat-vector style:
    pitch-black circular head, warm golden-cream rim light crescent, curly hair / hood,
    large expressive concentric cream-white eyes, eyebrows, sweat drop, and speech bubble pill."""
    x = int(cmd.get("x", 540)) + dx
    y = int(cmd.get("y", 920)) + dy
    scale = float(cmd.get("scale", 1.0))
    shirt = _rgb(cmd.get("shirt") or cmd.get("fill"), (42, 92, 68))
    head_col = (14, 14, 18)
    rim_col = (245, 216, 160)
    cream = (244, 239, 227)
    emotion = str(cmd.get("emotion", "shock"))
    pose = str(cmd.get("pose", "stand"))

    if pose == "lie":
        # Lying character (exactly like shot_04.png coffin/bed pose)
        draw.rounded_rectangle((x - int(180 * scale), y - int(4 * scale), x + int(145 * scale), y + int(95 * scale)),
                               radius=int(24 * scale), fill=rim_col)
        draw.rounded_rectangle((x - int(180 * scale), y, x + int(140 * scale), y + int(92 * scale)),
                               radius=int(24 * scale), fill=(38, 42, 50))
        hr = int(52 * scale)
        hx, hy = x - int(205 * scale), y + int(38 * scale)
        draw.ellipse((hx - hr - int(4 * scale), hy - hr - int(4 * scale), hx + hr, hy + hr), fill=rim_col)
        draw.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), fill=head_col)
        # Closed sleeping eyes
        draw.line((hx - int(14 * scale), hy - int(14 * scale), hx - int(14 * scale), hy + int(4 * scale)), fill=cream, width=max(2, int(4 * scale)))
        draw.line((hx + int(6 * scale), hy - int(14 * scale), hx + int(6 * scale), hy + int(4 * scale)), fill=cream, width=max(2, int(4 * scale)))
        return

    body_w, body_h = int(68 * scale), int(215 * scale)
    leg_h = int(175 * scale)
    # Legs (dark silhouette with golden-cream rim-light edge)
    for left, right in (
        (x - body_w + int(12 * scale), x - int(6 * scale)),
        (x + int(6 * scale), x + body_w - int(12 * scale)),
    ):
        draw.rounded_rectangle((left + int(4 * scale), y + body_h - int(15 * scale), right + int(5 * scale), y + body_h + leg_h),
                               radius=int(12 * scale), fill=rim_col)
        draw.rounded_rectangle((left, y + body_h - int(15 * scale), right, y + body_h + leg_h),
                               radius=int(12 * scale), fill=(26, 30, 38))
        # Rounded shoe with cream sole rim
        draw.rounded_rectangle((left - int(4 * scale), y + body_h + leg_h - int(22 * scale), right + int(16 * scale), y + body_h + leg_h + int(6 * scale)),
                               radius=int(10 * scale), fill=rim_col)
        draw.rounded_rectangle((left - int(4 * scale), y + body_h + leg_h - int(18 * scale), right + int(14 * scale), y + body_h + leg_h + int(6 * scale)),
                               radius=int(10 * scale), fill=(36, 40, 48))

    # Torso (flat vector coat/shirt with cream rim light)
    draw.rounded_rectangle((x - body_w + int(5 * scale), y + int(3 * scale), x + body_w + int(6 * scale), y + body_h + int(3 * scale)),
                           radius=int(26 * scale), fill=rim_col)
    draw.rounded_rectangle((x - body_w, y, x + body_w, y + body_h),
                           radius=int(26 * scale), fill=shirt)
    # Subtle lapel / collar detail
    draw.line((x, y + int(12 * scale), x, y + int(95 * scale)), fill=(max(0, shirt[0] - 18), max(0, shirt[1] - 18), max(0, shirt[2] - 18)), width=max(2, int(4 * scale)))

    radius = int(74 * scale)
    head_y = y - radius + int(4 * scale)

    # Concentric shock/thought radar rings around head (signature 'ЧТО, ЕСЛИ?' effect)
    if emotion in ("scared", "shock", "pain", "thinking"):
        for ring_i in (1, 2):
            rr = radius + int((22 + ring_i * 18) * scale)
            draw.ellipse((x - rr, head_y - rr, x + rr, head_y + rr), outline=(120, 115, 108), width=1)

    # Curly silhouette hair with cream rim light (or crimson hood if hood=True)
    if cmd.get("hood"):
        hr = radius + int(16 * scale)
        draw.ellipse((x - hr, head_y - hr, x + hr, head_y + hr), fill=(214, 88, 84))
        for dot_ang in (-50, -25, 0, 25, 50):
            rad = math.radians(dot_ang - 90)
            dx_dot = x + int(math.cos(rad) * (radius + int(4 * scale)))
            dy_dot = head_y + int(math.sin(rad) * (radius + int(4 * scale)))
            draw.ellipse((dx_dot - int(5 * scale), dy_dot - int(5 * scale), dx_dot + int(5 * scale), dy_dot + int(5 * scale)), fill=cream)
    else:
        for puff_ang in range(195, 355, 28):
            rad = math.radians(puff_ang)
            px = x + int(math.cos(rad) * (radius - int(6 * scale)))
            py = head_y + int(math.sin(rad) * (radius - int(6 * scale)))
            pr = int(26 * scale)
            draw.ellipse((px - pr, py - pr - int(4 * scale), px + pr, py + pr), fill=rim_col)
            draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=head_col)

    # Pitch-black circular head with warm golden-cream rim crescent
    draw.ellipse((x - radius + int(5 * scale), head_y - radius + int(4 * scale),
                  x + radius + int(7 * scale), head_y + radius + int(7 * scale)), fill=rim_col)
    draw.ellipse((x - radius, head_y - radius, x + radius, head_y + radius), fill=head_col)

    # Expressive 'ЧТО, ЕСЛИ?' eyes (concentric cream circles + pupil dot + eyebrows)
    eye_outer = int(20 * scale) if emotion in ("scared", "shock", "curious", "pain") else int(15 * scale)
    eye_inner = max(4, int(eye_outer * 0.56))
    pupil_r = max(2, int(eye_outer * 0.28))
    for idx_e, eye_x in enumerate((x - int(25 * scale), x + int(25 * scale))):
        ey = head_y - int(4 * scale)
        draw.ellipse((eye_x - eye_outer, ey - eye_outer, eye_x + eye_outer, ey + eye_outer), fill=cream)
        draw.ellipse((eye_x - eye_inner, ey - eye_inner, eye_x + eye_inner, ey + eye_inner), fill=head_col)
        draw.ellipse((eye_x - pupil_r, ey - pupil_r, eye_x + pupil_r, ey + pupil_r), fill=cream)
        # Eyebrow bar
        brow_y = ey - eye_outer - int(8 * scale)
        tilt = int(6 * scale) if (idx_e == 0 and emotion in ("scared", "shock", "pain")) else int(-6 * scale if emotion in ("scared", "shock", "pain") else 0)
        draw.line((eye_x - int(14 * scale), brow_y + tilt, eye_x + int(14 * scale), brow_y - tilt),
                  fill=cream, width=max(3, int(6 * scale)))

    # Cyan sweat drop on temple when stressed/thinking
    if emotion in ("scared", "shock", "pain", "thinking"):
        sx_drop, sy_drop = x + int(46 * scale), head_y - int(26 * scale)
        draw.ellipse((sx_drop - int(5 * scale), sy_drop - int(8 * scale), sx_drop + int(5 * scale), sy_drop + int(8 * scale)),
                     fill=(130, 216, 230))

    # Mouth expression
    if emotion in ("scared", "shock"):
        mr = int(12 * scale)
        draw.ellipse((x - mr, head_y + int(24 * scale) - mr, x + mr, head_y + int(24 * scale) + mr), fill=cream)
    elif emotion == "pain":
        # Wavy mouth (like frame_01.png)
        pts_m = [
            (x - int(22 * scale), head_y + int(26 * scale)),
            (x - int(8 * scale), head_y + int(34 * scale)),
            (x + int(8 * scale), head_y + int(20 * scale)),
            (x + int(22 * scale), head_y + int(28 * scale)),
        ]
        draw.line(pts_m, fill=cream, width=max(3, int(6 * scale)))
    elif emotion in ("happy", "curious"):
        draw.arc((x - int(18 * scale), head_y + int(14 * scale), x + int(18 * scale), head_y + int(38 * scale)),
                 0, 180, fill=cream, width=max(3, int(5 * scale)))
    else:
        draw.line((x - int(15 * scale), head_y + int(28 * scale), x + int(15 * scale), head_y + int(28 * scale)),
                  fill=cream, width=max(3, int(5 * scale)))

    # Arm pose
    if pose == "cover_head":
        for left, right in (
            (x - body_w - int(22 * scale), x - body_w + int(12 * scale)),
            (x + body_w - int(12 * scale), x + body_w + int(22 * scale)),
        ):
            draw.rounded_rectangle((left, head_y - int(15 * scale), right, y + int(85 * scale)),
                                   radius=int(14 * scale), fill=shirt, outline=rim_col, width=2)
    elif pose == "point":
        draw.rounded_rectangle((x + body_w - int(12 * scale), y + int(18 * scale),
                                x + body_w + int(95 * scale), y + int(50 * scale)),
                               radius=int(14 * scale), fill=shirt, outline=rim_col, width=2)
        draw.ellipse((x + body_w + int(85 * scale), y + int(16 * scale),
                      x + body_w + int(118 * scale), y + int(48 * scale)),
                     fill=head_col, outline=rim_col, width=2)
    elif pose == "phone":
        draw.rounded_rectangle((x + body_w - int(18 * scale), head_y,
                                x + body_w + int(30 * scale), y + int(105 * scale)),
                               radius=int(14 * scale), fill=shirt, outline=rim_col, width=2)
        draw.rounded_rectangle((x + radius - int(5 * scale), head_y - int(25 * scale),
                                x + radius + int(28 * scale), head_y + int(35 * scale)),
                               radius=6, fill=(18, 20, 24), outline=rim_col, width=2)
    else:
        # Side silhouette arm
        draw.rounded_rectangle((x + body_w - int(22 * scale), y + int(20 * scale),
                                x + body_w + int(8 * scale), y + int(135 * scale)),
                               radius=int(12 * scale), fill=head_col)

    # Speech bubble pill above head (exactly like shot_05.png 'Тёма?')
    name_tag = str(cmd.get("name_tag", "")).strip()
    bubble_text = name_tag or ("ЧТО?!" if emotion == "shock" else ("КАК?" if emotion in ("thinking", "curious") else ""))
    if bubble_text:
        by_pill = head_y - radius - int(48 * scale)
        tw = max(90, int(draw.textlength(bubble_text[:18], font=get_font(26))) + 38)
        draw.rounded_rectangle((x - tw // 2, by_pill - 24, x + tw // 2, by_pill + 24),
                               radius=22, fill=cream)
        draw.polygon([(x - 10, by_pill + 22), (x + 8, by_pill + 22), (x - 16, by_pill + 38)], fill=cream)
        draw.text((x, by_pill), bubble_text[:18], font=get_font(26), fill=(18, 18, 22), anchor="mm")


def _draw_entity(draw: ImageDraw.ImageDraw, cmd: dict[str, Any], dx: int, dy: int, phase: float) -> None:
    x = int(cmd.get("x", 560)) + dx
    y = int(cmd.get("y", 720)) + dy + int(math.sin(phase * math.pi * 2) * 8)
    scale = float(cmd.get("scale", 1.3))
    body_col = _rgb(cmd.get("fill"), (12, 14, 18))
    outline_col = _rgb(cmd.get("outline"), (75, 22, 28))
    glow_col = _rgb(cmd.get("glow"), (245, 38, 38))

    bw, bh = int(95 * scale), int(270 * scale)
    foot_y = y + bh

    # Two uncanny diverging shadows cast by a single figure ("бросает 2 тени")
    sway = int(math.sin(phase * math.pi * 2) * 18)
    shadow_left = [
        (x - int(bw * 0.6), foot_y - 10),
        (x + int(bw * 0.2), foot_y - 10),
        (x - int(260 * scale) - sway, min(1470, foot_y + int(210 * scale))),
        (x - int(390 * scale) - sway, min(1470, foot_y + int(175 * scale))),
    ]
    shadow_right = [
        (x - int(bw * 0.2), foot_y - 10),
        (x + int(bw * 0.6), foot_y - 10),
        (x + int(380 * scale) + sway, min(1470, foot_y + int(185 * scale))),
        (x + int(245 * scale) + sway, min(1470, foot_y + int(215 * scale))),
    ]
    draw.polygon(shadow_left, fill=(6, 8, 12, 165), outline=(38, 16, 20, 140))
    draw.polygon(shadow_right, fill=(6, 8, 12, 165), outline=(38, 16, 20, 140))

    # Subtle dark aura behind the figure
    for aura_r in (int(135 * scale), int(102 * scale)):
        draw.ellipse((x - aura_r, y - aura_r, x + aura_r, y + aura_r), outline=(48, 18, 24), width=2)

    # Tall dark humanoid silhouette
    draw.polygon([
        (x - bw, foot_y),
        (x - int(bw * 1.12), y + int(bh * 0.22)),
        (x - int(bw * 0.52), y - int(42 * scale)),
        (x + int(bw * 0.52), y - int(42 * scale)),
        (x + int(bw * 1.12), y + int(bh * 0.22)),
        (x + bw, foot_y),
    ], fill=body_col, outline=outline_col, width=4)

    # Head silhouette
    hr = int(62 * scale)
    hy = y - hr - int(20 * scale)
    draw.ellipse((x - hr, hy - hr, x + hr, hy + hr), fill=body_col, outline=outline_col, width=4)

    # Two piercing glowing red eyes
    er = max(6, int(12 * scale))
    for ex in (x - int(22 * scale), x + int(22 * scale)):
        draw.ellipse((ex - er - 5, hy - er - 5, ex + er + 5, hy + er + 5), fill=(95, 12, 18))
        draw.ellipse((ex - er, hy - er, ex + er, hy + er), fill=glow_col)
        draw.ellipse((ex - 3, hy - 3, ex + 3, hy + 3), fill=(255, 220, 220))


def draw_canvas_op(
    draw: ImageDraw.ImageDraw,
    command: dict[str, Any],
    phase: float,
    accent: tuple[int, int, int],
) -> None:
    kind = str(command.get("op", "rect"))
    dx = float(command.get("dx", 0)) * phase
    dy = float(command.get("dy", 0)) * phase
    fill = _rgb(command.get("fill")) if "fill" in command else None
    outline = _rgb(command.get("outline")) if "outline" in command else None
    stroke_width = int(command.get("w_stroke", 4))

    if kind == "bg_forest":
        sky_col = fill or (14, 24, 22)
        draw.rectangle((0, 240, WIDTH, 1460), fill=sky_col)
        # Moon / eerie mist
        moon_x = int(command.get("x", 780)) + int(dx * 0.3)
        moon_y = int(command.get("y", 420))
        draw.ellipse((moon_x - 75, moon_y - 75, moon_x + 75, moon_y + 75), fill=(215, 225, 210))
        for idx, tx_base in enumerate(range(-60, WIDTH + 120, 145)):
            shift = int(dx * (0.8 if idx % 2 == 0 else -0.5))
            tx = tx_base + shift
            top_y = 340 + (idx % 3) * 65
            col = (20, 38, 30) if idx % 2 == 0 else (12, 24, 19)
            draw.polygon([(tx, top_y), (tx - 150, 1420), (tx + 150, 1420)], fill=col)
            draw.polygon([(tx, top_y + 160), (tx - 180, 1480), (tx + 180, 1480)], fill=col)
            draw.rectangle((tx - 18, 1280, tx + 18, 1480), fill=(28, 22, 18))

    elif kind == "bg_room":
        wall_col = fill or (34, 38, 48)
        floor_col = (20, 23, 30)
        draw.rectangle((0, 250, WIDTH, 1180), fill=wall_col)
        draw.rectangle((0, 1180, WIDTH, 1480), fill=floor_col)
        draw.line((0, 1180, WIDTH, 1180), fill=(65, 72, 86), width=6)
        # Wallpaper stripes & window frame
        for x_line in range(90, WIDTH, 180):
            draw.line((x_line, 260, x_line, 1180), fill=(28, 32, 41), width=2)
        wx, wy, ww, wh = int(command.get("x", 560)), int(command.get("y", 380)), 340, 440
        draw.rectangle((wx, wy, wx + ww, wy + wh), fill=(12, 18, 28), outline=(140, 148, 162), width=10)
        draw.line((wx + ww // 2, wy, wx + ww // 2, wy + wh), fill=(140, 148, 162), width=8)
        draw.line((wx, wy + wh // 2, wx + ww, wy + wh // 2), fill=(140, 148, 162), width=8)

    elif kind == "bg_corridor":
        base_col = fill or (22, 26, 34)
        draw.rectangle((0, 250, WIDTH, 1480), fill=base_col)
        vx, vy = int(command.get("x", 540)) + int(dx * 0.5), int(command.get("y", 720))
        # Perspective walls converging to vanishing point
        draw.polygon([(0, 250), (WIDTH, 250), (vx + 170, vy - 180), (vx - 170, vy - 180)], fill=(16, 20, 27))
        draw.polygon([(0, 1480), (WIDTH, 1480), (vx + 170, vy + 220), (vx - 170, vy + 220)], fill=(14, 17, 22))
        for corner_x, corner_y, inner_x, inner_y in (
            (0, 250, vx - 170, vy - 180),
            (WIDTH, 250, vx + 170, vy - 180),
            (0, 1480, vx - 170, vy + 220),
            (WIDTH, 1480, vx + 170, vy + 220),
        ):
            draw.line((corner_x, corner_y, inner_x, inner_y), fill=(72, 82, 98), width=5)
        draw.rectangle((vx - 170, vy - 180, vx + 170, vy + 220), fill=(8, 10, 14), outline=accent, width=4)
        # Ceiling lights
        for step in (0.25, 0.55, 0.82):
            ly = int(250 + (vy - 180 - 250) * step)
            lw = int((1 - step) * 140)
            flicker = (245, 220, 145) if math.sin(phase * 4 + step * 5) > -0.4 else (80, 75, 55)
            draw.rounded_rectangle((vx - lw, ly - 10, vx + lw, ly + 10), radius=5, fill=flicker)

    elif kind == "bg_street":
        draw.rectangle((0, 250, WIDTH, 980), fill=fill or (14, 18, 28))
        # City skyline
        buildings = ((20, 560, 150), (160, 480, 170), (320, 620, 140),
                     (450, 440, 200), (640, 530, 165), (790, 470, 155), (930, 580, 140))
        for idx, (bx, by, bw) in enumerate(buildings):
            bcol = (24, 30, 42) if idx % 2 == 0 else (18, 22, 32)
            draw.rectangle((bx, by, bx + bw, 980), fill=bcol, outline=(10, 14, 20), width=3)
            for wy in range(by + 30, 940, 60):
                for wx in range(bx + 20, bx + bw - 30, 45):
                    if (wx + wy + idx) % 4 == 0:
                        draw.rectangle((wx, wy, wx + 20, wy + 30), fill=(235, 195, 100))
        # Road perspective
        draw.polygon([(0, 1480), (WIDTH, 1480), (680, 980), (400, 980)], fill=(26, 29, 35))
        for dash_i in range(5):
            y1 = 1020 + dash_i * 85 + int(phase * 25) % 85
            y2 = min(1460, y1 + 45)
            draw.line((540, y1, 540, y2), fill=(235, 195, 75), width=8)

    elif kind == "bg_water":
        draw.rectangle((0, 250, WIDTH, 1480), fill=fill or (10, 28, 42))
        for wave_i in range(8):
            wy = 360 + wave_i * 135 + int(math.sin(phase * 2 + wave_i) * 14)
            draw.arc((-100, wy - 35, WIDTH + 100, wy + 35), 0, 180, fill=(28, 72, 98), width=5)
        # Porthole / submarine frame
        cx, cy, cr = int(command.get("x", 540)), int(command.get("y", 740)), int(command.get("r", 310))
        draw.ellipse((cx - cr, cy - cr, cx + cr, cy + cr), fill=(8, 20, 32), outline=(95, 112, 126), width=18)

    elif kind == "bg_facility":
        draw.rectangle((0, 250, WIDTH, 1480), fill=fill or (26, 32, 36))
        # Industrial hazard stripes & pipes
        draw.rectangle((60, 320, WIDTH - 60, 1360), fill=(20, 25, 30), outline=(78, 88, 98), width=8)
        for pipe_y in (390, 440):
            draw.line((60, pipe_y, WIDTH - 60, pipe_y), fill=(110, 120, 130), width=14)
        for hx in range(80, WIDTH - 100, 90):
            draw.polygon([(hx, 1300), (hx + 45, 1300), (hx + 20, 1350), (hx - 25, 1350)], fill=(225, 175, 45))

    elif kind == "building_grid":
        bx = int(command.get("x", 160)) + int(dx)
        by = int(command.get("y", 340)) + int(dy)
        bw = int(command.get("w", 760))
        bh = int(command.get("h", 820))
        draw.rectangle((bx, by, bx + bw, by + bh), fill=fill or (26, 34, 44), outline=(65, 78, 94), width=6)
        rows, cols = 5, 4
        cell_w, cell_h = bw // cols, bh // rows
        for r_idx in range(rows):
            for c_idx in range(cols):
                wx = bx + c_idx * cell_w + 22
                wy = by + r_idx * cell_h + 22
                lit = (r_idx + c_idx + int(phase * 2)) % 4 == 0
                w_col = (240, 195, 95) if lit else (38, 48, 62)
                draw.rounded_rectangle((wx, wy, wx + cell_w - 44, wy + cell_h - 44), radius=6, fill=w_col)

    elif kind == "door_closeup":
        dx_pos = int(command.get("x", 260)) + int(dx)
        dy_pos = int(command.get("y", 340)) + int(dy)
        dw = int(command.get("w", 560))
        dh = int(command.get("h", 920))
        draw.rectangle((dx_pos - 20, dy_pos - 20, dx_pos + dw + 20, dy_pos + dh),
                       fill=(42, 48, 58), outline=(95, 105, 118), width=8)
        door_col = fill or (58, 38, 30)
        draw.rectangle((dx_pos, dy_pos, dx_pos + dw, dy_pos + dh),
                       fill=door_col, outline=(24, 18, 15), width=6)
        # Door panels & viewing peephole / window
        draw.rectangle((dx_pos + 55, dy_pos + 70, dx_pos + dw - 55, dy_pos + 390),
                       fill=(22, 28, 36), outline=(90, 98, 110), width=6)
        if command.get("bars", True):
            for bx_bar in range(dx_pos + 110, dx_pos + dw - 80, 85):
                draw.line((bx_bar, dy_pos + 70, bx_bar, dy_pos + 390), fill=(130, 138, 148), width=8)
        # Handle & lock indicator
        hx = dx_pos + dw - 75
        hy = dy_pos + dh // 2
        lock_col = (225, 50, 50) if command.get("locked", True) else (55, 205, 95)
        draw.ellipse((hx - 28, hy - 28, hx + 28, hy + 28), fill=(30, 34, 40), outline=lock_col, width=5)
        draw.rounded_rectangle((hx - 65, hy - 12, hx + 10, hy + 12), radius=8, fill=(210, 180, 95))

    elif kind == "window_view":
        wx = int(command.get("x", 220)) + int(dx)
        wy = int(command.get("y", 360)) + int(dy)
        ww = int(command.get("w", 640))
        wh = int(command.get("h", 680))
        draw.rectangle((wx, wy, wx + ww, wy + wh), fill=fill or (14, 20, 32), outline=(200, 208, 220), width=12)
        draw.line((wx + ww // 2, wy, wx + ww // 2, wy + wh), fill=(200, 208, 220), width=10)
        draw.line((wx, wy + wh // 2, wx + ww, wy + wh // 2), fill=(200, 208, 220), width=10)
        if command.get("broken"):
            for shard in range(5):
                sx = wx + 80 + shard * 105
                sy = wy + 180 + (shard % 2) * 140
                draw.polygon([(sx, sy), (sx + 55, sy + 20), (sx + 20, sy + 110)],
                             fill=(165, 210, 235), outline="white", width=2)

    elif kind == "valve_diagram":
        cx = int(command.get("x", 540)) + int(dx)
        cy = int(command.get("y", 720)) + int(dy)
        r = int(command.get("r", 175))
        draw.rounded_rectangle((cx - r - 90, cy - r - 70, cx + r + 90, cy + r + 70),
                               radius=28, fill=(20, 26, 34), outline=(90, 102, 118), width=6)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(32, 38, 48), outline=(235, 175, 65), width=10)
        angle_offset = phase * 0.9
        for spoke in range(4):
            ang = angle_offset + spoke * (math.pi / 2)
            x1 = cx + int(math.cos(ang) * (r - 18))
            y1 = cy + int(math.sin(ang) * (r - 18))
            x2 = cx - int(math.cos(ang) * (r - 18))
            y2 = cy - int(math.sin(ang) * (r - 18))
            draw.line((x1, y1, x2, y2), fill=(225, 58, 52), width=18)
        draw.ellipse((cx - 36, cy - 36, cx + 36, cy + 36), fill=(235, 190, 75), outline=(25, 30, 36), width=5)

    elif kind == "dashed_zone":
        cx = int(command.get("x", 540)) + int(dx)
        cy = int(command.get("y", 760)) + int(dy)
        r = int(command.get("r", 230)) + int(math.sin(phase * math.pi * 2) * 10)
        col = fill or outline or accent
        for seg in range(0, 360, 24):
            start_a = seg + int(phase * 30) % 360
            draw.arc((cx - r, cy - r, cx + r, cy + r), start_a, start_a + 14, fill=col, width=7)

    elif kind == "highlight_ring":
        cx = int(command.get("x", 540)) + int(dx)
        cy = int(command.get("y", 740)) + int(dy)
        r = int(command.get("r", 115)) + int(math.sin(phase * math.pi * 3) * 12)
        col = fill or outline or (245, 65, 55)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=col, width=8)
        draw.ellipse((cx - r - 18, cy - r - 18, cx + r + 18, cy + r + 18), outline=(255, 200, 95), width=3)

    elif kind == "stickman":
        sx = int(command.get("x", 420)) + int(dx)
        sy = int(command.get("y", 820)) + int(dy)
        scale = float(command.get("scale", 1.05))
        shirt = _rgb(command.get("shirt") or command.get("fill"), (74, 134, 184))
        draw_stickman(draw, sx, sy, scale=scale, shirt=shirt, infected=bool(command.get("infected", False)))

    elif kind == "actor":
        _draw_actor(draw, command, round(dx), round(dy))

    elif kind == "entity":
        _draw_entity(draw, command, round(dx), round(dy), phase)

    elif kind == "hands_reach":
        slip = bool(command.get("slip", False))
        shift = int(phase * 22) if slip else int(math.sin(phase * math.pi * 2) * 14)
        y_pos = int(command.get("y", 780)) + int(dy)
        # Left arm reaching right
        draw.polygon([(-30, y_pos + 180), (360 - shift, y_pos + 10),
                      (410 - shift, y_pos + 85), (20, y_pos + 270)], fill=(228, 182, 62), outline=(22, 26, 32), width=5)
        draw.ellipse((330 - shift, y_pos - 25, 455 - shift, y_pos + 95),
                     fill=(238, 210, 185), outline=(22, 26, 32), width=5)
        # Right arm reaching left
        r_col = (95, 102, 108) if not slip else (62, 118, 178)
        draw.polygon([(WIDTH + 30, y_pos - 160), (680 + shift, y_pos - 20),
                      (735 + shift, y_pos + 65), (WIDTH + 40, y_pos - 55)], fill=r_col, outline=(22, 26, 32), width=5)
        draw.ellipse((605 + shift, y_pos - 45, 735 + shift, y_pos + 75),
                     fill=(185, 190, 195) if not slip else (238, 210, 185), outline=(22, 26, 32), width=5)

    elif kind == "key_palm":
        kx = int(command.get("x", 540)) + int(dx)
        ky = int(command.get("y", 800)) + int(dy)
        draw.ellipse((kx - 165, ky - 140, kx + 165, ky + 140),
                     fill=(235, 208, 182), outline=(30, 35, 42), width=6)
        # Golden key
        draw.ellipse((kx - 75, ky - 45, kx + 15, ky + 45), outline=(235, 188, 58), width=14)
        draw.line((kx + 12, ky, kx + 135, ky + 75), fill=(235, 188, 58), width=16)
        draw.line((kx + 95, ky + 52, kx + 70, ky + 90), fill=(235, 188, 58), width=12)
        draw.line((kx + 125, ky + 70, kx + 100, ky + 108), fill=(235, 188, 58), width=12)

    elif kind == "padlock_chain":
        px = int(command.get("x", 540)) + int(dx)
        py = int(command.get("y", 760)) + int(dy)
        # Heavy chains
        draw.line((120, py - 190, WIDTH - 120, py + 190), fill=(135, 142, 152), width=22)
        draw.line((120, py + 190, WIDTH - 120, py - 190), fill=(135, 142, 152), width=22)
        # Padlock shackle & body
        draw.arc((px - 65, py - 145, px + 65, py - 10), 180, 360, fill=(195, 202, 210), width=18)
        draw.rounded_rectangle((px - 95, py - 50, px + 95, py + 120),
                               radius=24, fill=(232, 182, 58), outline=(32, 28, 22), width=6)
        draw.ellipse((px - 18, py + 10, px + 18, py + 46), fill=(24, 26, 30))
        draw.polygon([(px - 8, py + 38), (px + 8, py + 38), (px + 14, py + 82), (px - 14, py + 82)], fill=(24, 26, 30))

    elif kind == "light_beam":
        lx = int(command.get("x", 540)) + int(dx)
        ly = int(command.get("y", 380)) + int(dy)
        tx = int(command.get("tx", 540)) + int(math.sin(phase * math.pi * 2) * 85)
        ty = int(command.get("ty", 1180))
        beam_col = fill or (245, 220, 130)
        # Soft semi-transparent outer & inner light cones so background remains clearly visible
        draw.polygon(
            [(lx - 28, ly), (lx + 28, ly), (tx + 260, ty), (tx - 260, ty)],
            fill=(*beam_col, 32),
        )
        draw.polygon(
            [(lx - 14, ly), (lx + 14, ly), (tx + 145, ty), (tx - 145, ty)],
            fill=(*beam_col, 52),
        )
        draw.ellipse((lx - 22, ly - 14, lx + 22, ly + 14), fill=(*beam_col, 135))

    elif kind == "weather":
        w_style = str(command.get("style", "rain")).lower()
        count = int(command.get("count", 22))
        for i in range(count):
            wx = (i * 137 + int(phase * 95)) % WIDTH
            wy = 260 + ((i * 193 + int(phase * 260)) % 1080)
            if w_style == "snow" or w_style == "ash":
                r_dot = 4 + (i % 4)
                col = (215, 225, 235) if w_style == "snow" else (165, 155, 148)
                draw.ellipse((wx - r_dot, wy - r_dot, wx + r_dot, wy + r_dot), fill=col)
            else:
                draw.line((wx, wy, wx - 12, wy + 42), fill=(135, 175, 215), width=3)

    elif kind == "polygon":
        pts = command.get("points", [])
        if len(pts) >= 3:
            shifted = [(int(px + dx), int(py + dy)) for px, py in pts]
            poly_col = fill or (60, 70, 85)
            # If polygon is used as a warm light/glow cone, render it semi-transparently
            if poly_col[0] > 190 and poly_col[1] > 160 and poly_col[2] < 170:
                draw.polygon(shifted, fill=(*poly_col, 48))
            else:
                draw.polygon(shifted, fill=poly_col, outline=outline)

    elif kind == "rect":
        x1 = round(int(command.get("x", 180)) + dx)
        y1 = round(int(command.get("y", 480)) + dy)
        x2 = x1 + int(command.get("w", 240))
        y2 = y1 + int(command.get("h", 220))
        draw.rounded_rectangle((x1, y1, x2, y2), radius=int(command.get("radius", 14)),
                               fill=fill or (42, 48, 58), outline=outline,
                               width=stroke_width if outline else 0)

    elif kind == "circle":
        cx = round(int(command.get("x", 540)) + dx)
        cy = round(int(command.get("y", 780)) + dy)
        radius = int(command.get("r", 85))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                     fill=fill, outline=outline or accent, width=stroke_width)

    elif kind == "line":
        draw.line((round(int(command.get("x1", 200)) + dx),
                   round(int(command.get("y1", 600)) + dy),
                   round(int(command.get("x2", 800)) + dx),
                   round(int(command.get("y2", 900)) + dy)),
                  fill=fill or (235, 185, 65), width=int(command.get("w_stroke", 8)))

    elif kind == "phone_ui":
        px, py = round(int(command.get("x", 270)) + dx), round(int(command.get("y", 360)) + dy)
        draw.rounded_rectangle((px, py, px + 540, py + 900), radius=54,
                               fill=(16, 20, 28), outline=(95, 105, 120), width=8)
        header = str(command.get("header", "112 · ЭКСТРЕННЫЙ ВЫЗОВ"))[:24]
        subtext = str(command.get("subtext", "Входящее сообщение..."))[:32]
        badge_color = _rgb(command.get("fill"), (205, 45, 45))
        draw.rounded_rectangle((px + 45, py + 170, px + 495, py + 430),
                               radius=28, fill=badge_color)
        draw.text((px + 270, py + 265), header.upper(), font=get_font(38),
                  fill="white", anchor="mm")
        draw.text((px + 270, py + 345), subtext, font=get_font(28),
                  fill=(235, 240, 245), anchor="mm")

    elif kind == "radio_kit":
        rx, ry = round(int(command.get("x", 180)) + dx), round(int(command.get("y", 450)) + dy)
        draw.rounded_rectangle(
            (rx, ry, rx + 720, ry + 580), radius=34,
            fill=(35, 40, 48), outline=(185, 195, 205), width=8,
        )
        draw.rectangle(
            (rx + 75, ry + 115, rx + 645, ry + 240),
            fill=(10, 15, 20), outline=(235, 175, 65), width=5,
        )
        draw.text(
            (rx + 360, ry + 178),
            str(command.get("frequency") or command.get("header") or "87.4 FM")[:16],
            font=get_font(52), fill=(245, 205, 95), anchor="mm",
        )
        draw.ellipse(
            (rx + 300, ry + 340, rx + 420, ry + 460),
            fill=(205, 45, 45), outline="white", width=4,
        )

    elif kind == "bg_lab_board":
        draw.rectangle((0, 250, WIDTH, 1480), fill=fill or (18, 26, 38))
        # Subtle infographic blueprint grid
        for gx in range(80, WIDTH, 120):
            draw.line((gx, 260, gx, 1460), fill=(28, 40, 56), width=2)
        for gy in range(300, 1460, 120):
            draw.line((40, gy, WIDTH - 40, gy), fill=(28, 40, 56), width=2)
        draw.rounded_rectangle((55, 295, WIDTH - 55, 1390), radius=24, outline=(58, 92, 136), width=5)

    elif kind == "bg_space":
        draw.rectangle((0, 250, WIDTH, 1480), fill=fill or (8, 12, 24))
        for i in range(36):
            sx = (i * 173 + int(phase * 18)) % WIDTH
            sy = 270 + ((i * 139) % 1120)
            sr = 2 + (i % 3)
            star_col = (255, 245, 200) if i % 4 == 0 else (175, 205, 245)
            draw.ellipse((sx - sr, sy - sr, sx + sr, sy + sr), fill=star_col)
        # Glowing planet / event horizon
        px, py, pr = int(command.get("x", 540)), int(command.get("y", 720)), int(command.get("r", 220))
        draw.ellipse((px - pr - 28, py - pr - 28, px + pr + 28, py + pr + 28), outline=(95, 165, 245), width=4)
        draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=(22, 46, 86), outline=(135, 205, 255), width=8)

    elif kind == "diagram_flow":
        # 2D Frontend <-> Backend / Process Flow diagram
        y_mid = int(command.get("y", 640)) + int(dy)
        left_lbl = str(command.get("left_label") or command.get("header") or "FRONTEND / CLIENT")[:18].upper()
        right_lbl = str(command.get("right_label") or command.get("subtext") or "BACKEND / SERVER")[:18].upper()
        draw.rounded_rectangle((75, y_mid - 115, 435, y_mid + 115), radius=22,
                               fill=(28, 52, 84), outline=(95, 175, 250), width=6)
        draw.rounded_rectangle((645, y_mid - 115, 1005, y_mid + 115), radius=22,
                               fill=(36, 68, 54), outline=(85, 225, 145), width=6)
        draw.text((255, y_mid), left_lbl, font=get_font(30), fill="white", anchor="mm")
        draw.text((825, y_mid), right_lbl, font=get_font(30), fill="white", anchor="mm")
        # Animated data packets between left and right boxes
        draw.line((445, y_mid - 25, 635, y_mid - 25), fill=(95, 175, 250), width=6)
        draw.line((635, y_mid + 25, 445, y_mid + 25), fill=(85, 225, 145), width=6)
        pkt_x = 455 + int((phase * 140) % 165)
        draw.ellipse((pkt_x - 14, y_mid - 39, pkt_x + 14, y_mid - 11), fill=(255, 215, 65))
        pkt_x2 = 625 - int((phase * 140) % 165)
        draw.ellipse((pkt_x2 - 14, y_mid + 11, pkt_x2 + 14, y_mid + 39), fill=(85, 245, 165))

    elif kind == "anatomy_xray":
        # 2D Medical / Biology X-Ray Zoom Scanner (bone fracture, blood vessels, brain neurons)
        cx = int(command.get("x", 540)) + int(dx)
        cy = int(command.get("y", 690)) + int(dy)
        r = int(command.get("r", 230))
        style = str(command.get("style", "bone")).lower()
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(12, 22, 34), outline=(75, 215, 235), width=10)
        if style in ("blood", "wound", "heart"):
            draw.line((cx - r + 35, cy, cx + r - 35, cy), fill=(195, 35, 45), width=46)
            for cell_i in range(6):
                cell_x = cx - r + 65 + ((cell_i * 65 + int(phase * 90)) % (r * 2 - 130))
                cell_y = cy + int(math.sin(phase * 3 + cell_i) * 14)
                draw.ellipse((cell_x - 22, cell_y - 16, cell_x + 22, cell_y + 16),
                             fill=(245, 55, 65), outline=(255, 165, 170), width=3)
        elif style in ("brain", "mind", "sleep"):
            draw.ellipse((cx - 135, cy - 105, cx + 135, cy + 105),
                         fill=(225, 145, 165), outline=(255, 205, 215), width=6)
            for n_i in range(5):
                ang = phase * 2 + n_i * 1.25
                nx, ny = cx + int(math.cos(ang) * 75), cy + int(math.sin(ang) * 55)
                draw.ellipse((nx - 12, ny - 12, nx + 12, ny + 12), fill=(255, 230, 75))
                draw.line((cx, cy, nx, ny), fill=(255, 230, 75), width=4)
        else:
            # Bone / fracture X-Ray
            draw.line((cx - 145, cy - 95, cx - 12, cy - 8), fill=(235, 240, 248), width=38)
            draw.line((cx + 12, cy + 8, cx + 145, cy + 95), fill=(235, 240, 248), width=38)
            # Fracture crack glow
            draw.line((cx - 28, cy + 28, cx + 28, cy - 28), fill=(255, 75, 65), width=8)
            pulse_r = 38 + int(math.sin(phase * math.pi * 3) * 10)
            draw.ellipse((cx - pulse_r, cy - pulse_r, cx + pulse_r, cy + pulse_r), outline=(255, 95, 75), width=4)

    elif kind == "chart_bars":
        # 2D Finance / Statistics Bar Chart + Trend Line
        bx, by = int(command.get("x", 160)), int(command.get("y", 860))
        heights = (140, 220, 190, 310, 420)
        pts = []
        for idx, h_val in enumerate(heights):
            x1 = bx + idx * 155
            h_anim = int(h_val * min(1.0, 0.75 + 0.25 * math.sin(phase + idx)))
            y1 = by - h_anim
            col = (65, 195, 115) if idx == len(heights) - 1 else (58, 128, 205)
            draw.rounded_rectangle((x1, y1, x1 + 105, by), radius=12, fill=col, outline="white", width=3)
            pts.append((x1 + 52, y1 - 20))
        for i in range(len(pts) - 1):
            draw.line((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]), fill=(255, 215, 65), width=8)

    elif kind == "math_board":
        # 2D School / Science / Math Concept Board
        bx, by = int(command.get("x", 140)), int(command.get("y", 380))
        draw.rounded_rectangle((bx, by, bx + 800, by + 460), radius=24,
                               fill=(20, 44, 36), outline=(185, 145, 75), width=10)
        formula = str(command.get("header") or command.get("label") or "E = mc²   |   f(x) = 2x + 1")[:28]
        draw.text((bx + 400, by + 135), formula, font=get_font(44), fill=(255, 235, 145), anchor="mm")
        # Geometric triangle & circle on board
        draw.polygon([(bx + 180, by + 360), (bx + 310, by + 220), (bx + 380, by + 360)],
                     outline=(95, 225, 185), width=6)
        draw.ellipse((bx + 480, by + 220, bx + 620, by + 360), outline=(255, 185, 75), width=6)

    elif kind == "nuke_mushroom":
        # Ядерный гриб (столб взрыва, купол, ударная волна, пламя)
        cx = int(command.get("x", 540)) + int(dx * 0.3)
        cy = int(command.get("y", 680)) + int(dy * 0.3)
        # 1. Сияние горизонта и зарево
        glow_r = 380 + int(math.sin(phase * 4) * 25)
        draw.ellipse((cx - glow_r, cy - 80, cx + glow_r, cy + 420), fill=(255, 120, 30))
        draw.ellipse((cx - glow_r + 80, cy - 20, cx + glow_r - 80, cy + 340), fill=(255, 200, 60))
        # 2. Столб ножки гриба
        stem_w = 110 + int(math.sin(phase * 3) * 12)
        draw.polygon([(cx - stem_w, cy + 350), (cx + stem_w, cy + 350),
                      (cx + stem_w // 2, cy - 80), (cx - stem_w // 2, cy - 80)],
                     fill=(75, 40, 30))
        draw.polygon([(cx - stem_w + 25, cy + 350), (cx + stem_w - 25, cy + 350),
                      (cx + 15, cy - 80), (cx - 15, cy - 80)],
                     fill=(245, 135, 45))
        # 3. Шляпка ядерного гриба (клубы дыма)
        cap_y = cy - 140
        cap_w = 340 + int(math.sin(phase * 2) * 18)
        cap_h = 190
        draw.ellipse((cx - cap_w, cap_y - cap_h, cx + cap_w, cap_y + cap_h // 2), fill=(55, 30, 26))
        draw.ellipse((cx - cap_w + 35, cap_y - cap_h + 20, cx + cap_w - 35, cap_y + 30), fill=(185, 65, 30))
        draw.ellipse((cx - cap_w // 2, cap_y - cap_h + 40, cx + cap_w // 2, cap_y), fill=(255, 175, 55))
        # 4. Кольцо ударной волны
        shock_r = int((phase * 480) % 480) + 120
        draw.ellipse((cx - shock_r, cy + 180 - shock_r // 4, cx + shock_r, cy + 180 + shock_r // 4),
                     outline=(255, 235, 180), width=6)

    elif kind == "thumb_rule":
        # Правило большого пальца (рука с вытянутым вверх пальцем, сравнивающим гриб)
        hx = int(command.get("x", 540)) + int(dx)
        hy = int(command.get("y", 840)) + int(dy)
        # Рука/кулак
        draw.rounded_rectangle((hx - 70, hy, hx + 70, hy + 220), radius=28, fill=(235, 175, 135), outline=(180, 115, 80), width=6)
        # Большой палец вверх
        draw.rounded_rectangle((hx - 35, hy - 190, hx + 35, hy + 20), radius=30, fill=(245, 185, 145), outline=(180, 115, 80), width=7)
        # Линии сравнения / видоискатель
        draw.line((hx - 160, hy - 190, hx + 160, hy - 190), fill=(255, 75, 65), width=5)
        draw.line((hx - 160, hy + 20, hx + 160, hy + 20), fill=(255, 75, 65), width=5)
        rule_tag = str(command.get("label") or "ПРАВИЛО БОЛЬШОГО ПАЛЬЦА")[:28]
        draw.rounded_rectangle((hx - 220, hy - 260, hx + 220, hy - 205), radius=12, fill=(20, 24, 32), outline=(255, 215, 65), width=3)
        draw.text((hx, hy - 233), rule_tag, font=get_font(24), fill="white", anchor="mm")

    elif kind == "callout":
        cx = round(int(command.get("x", 540)) + dx)
        cy = round(int(command.get("y", 520)) + dy)
        tx = round(int(command.get("tx", cx)) + dx)
        ty = round(int(command.get("ty", min(1150, cy + 140))) + dy)
        label = str(command.get("label") or command.get("header") or "ВНИМАНИЕ").upper()[:28]
        tag_color = _rgb(command.get("fill"), (215, 52, 48))
        draw.line((cx, cy, tx, ty), fill=tag_color, width=5)
        draw.ellipse((tx - 9, ty - 9, tx + 9, ty + 9), fill=tag_color)
        text_width = int(draw.textlength(label, font=get_font(30))) + 40
        draw.rounded_rectangle((cx - text_width // 2, cy - 28, cx + text_width // 2, cy + 28),
                               radius=12, fill=tag_color, outline="white", width=2)
        draw.text((cx, cy), label, font=get_font(30), fill="white", anchor="mm")


def draw_scene(
    scene: dict[str, Any],
    story_meta: dict[str, Any],
    phase: float,
    word_index: int = -1,
    words: list[str] | None = None,
    show_choice: bool | None = None,
    choice_source: dict[str, Any] | None = None,
    scene_number: int = 1,
    lang: str = "ru",
) -> Image.Image:
    phase = float(phase)
    accent = _rgb(scene.get("accent_color"), (235, 115, 45))

    # Full 2D Procedural Vector Canvas Renderer (cached gradient + RGBA alpha support)
    bg_top = _rgb(scene.get("bg_top"), (22, 26, 34))
    bg_bottom = _rgb(scene.get("bg_bottom"), (10, 13, 18))
    image = get_gradient_bg(bg_top, bg_bottom)
    draw = ImageDraw.Draw(image, "RGBA")

    canvas_ops = scene.get("canvas", [])
    # Ensure a background op is drawn first if none present
    bg_ops = {
        "bg_forest", "bg_room", "bg_corridor", "bg_street",
        "bg_water", "bg_facility", "bg_lab_board", "bg_space",
    }
    if not any(isinstance(c, dict) and c.get("op") in bg_ops for c in canvas_ops):
        draw_canvas_op(draw, {"op": "bg_lab_board"}, phase, accent)

    for command in canvas_ops:
        if isinstance(command, dict):
            draw_canvas_op(draw, command, phase, accent)

    # ========================================================================
    # TOP HUD: MAKE IT EASY AI badge, Topic/Location, Clock, Metric, Step Badge
    # ========================================================================
    rec_title = str(story_meta.get("series_tag", "MAKE IT EASY AI")).upper()[:30]
    location = str(story_meta.get("location", "")).upper()[:36]
    clock = str(scene.get("clock", "00:45"))
    bpm = int(scene.get("bpm", 120)) + int(round(math.sin(phase * math.pi) * 3))
    timer_label = str(scene.get("timer_label", "")).strip()

    draw.ellipse((42, 52, 66, 76),
                 fill=(55, 205, 245) if math.sin(phase * math.pi) >= 0 else (35, 125, 165))
    draw.text((82, 64), f"AI · {rec_title}", font=get_font(28),
              fill=(225, 235, 245), anchor="lm")
    if location:
        draw.text((540, 112), location, font=get_font(25),
                  fill=(195, 205, 218), anchor="mm")
    draw.text((WIDTH - 45, 64), clock, font=get_font(30), fill=(235, 115, 45), anchor="rm")
    draw.rounded_rectangle((WIDTH - 245, 102, WIDTH - 42, 164), radius=12,
                           fill=(14, 17, 22), outline=(210, 50, 50), width=3)
    draw.text((WIDTH - 220, 133), "♥", font=get_font(34), fill=(225, 50, 50), anchor="lm")
    draw.text((WIDTH - 58, 133), str(bpm), font=get_font(34), fill="white", anchor="rm")
    if timer_label:
        draw.rounded_rectangle((42, 102, 295, 164), radius=12,
                               fill=(14, 17, 22), outline=(235, 115, 45), width=3)
        draw.text((168, 133), timer_label.upper()[:18], font=get_font(28),
                  fill="white", anchor="mm")

    rule_badge = str(scene.get("rule_badge", "")).strip()
    visual_detail = str(scene.get("visual_detail", "")).strip()
    if scene_number == 1:
        # ====================================================================
        # 🔥 ПЕРВЫЕ 5 СЕКУНД — РАЗЪЁБ (COLD-OPEN RETENTION HOOK OVERLAY)
        # ====================================================================
        is_en_hook = (str(lang).lower() == "en")
        hook_title = str(visual_detail or story_meta.get("series_tag", "MAKE IT EASY AI")).upper()[:34]
        hook_prefix = "⚡ WAIT FOR IT:" if is_en_hook else "⚡ ПЕРВЫЕ 5 СЕКУНД:"
        ring_r = int(((phase * 220) % 380) + 110)
        draw.ellipse(
            (540 - ring_r, 760 - ring_r, 540 + ring_r, 760 + ring_r),
            outline=(245, 75, 55) if int(phase * 3) % 2 == 0 else (245, 216, 160),
            width=4,
        )
        draw.rounded_rectangle(
            (48, 185, WIDTH - 48, 285),
            radius=18,
            fill=(195, 32, 32),
            outline=(245, 216, 160),
            width=4,
        )
        draw.text((540, 222), f"{hook_prefix} {hook_title[:24]}", font=get_font(34), fill="white", anchor="mm")
        rem_ratio = max(0.05, min(1.0, 1.0 - (phase / 4.0)))
        bar_w = int((WIDTH - 136) * rem_ratio)
        draw.rounded_rectangle((68, 258, 68 + bar_w, 272), radius=6, fill=(245, 216, 160))
    elif rule_badge or visual_detail:
        header = f"{rule_badge} · {visual_detail}".strip(" ·").upper()[:38]
        box_width = min(WIDTH - 60, max(int(draw.textlength(header, font=get_font(34))) + 60, 360))
        draw.rounded_rectangle((540 - box_width // 2, 195, 540 + box_width // 2, 268),
                               radius=14, fill=(16, 20, 26), outline=accent, width=4)
        draw.text((540, 231), header, font=get_font(34), fill="white", anchor="mm")

    # ========================================================================
    # INTERACTIVE CHOICE OVERLAY ("ЧТО БЫ СДЕЛАЛ ТЫ?" / "WHAT WOULD YOU CHOOSE?")
    # ========================================================================
    if show_choice is None:
        show_choice = bool(scene.get("is_ending") or scene.get("is_choice_scene"))
    choice_data = choice_source or scene
    if show_choice:
        is_en = (str(lang).lower() == "en")
        choice_a = str(choice_data.get("choice_a") or ("RUN AWAY" if is_en else "УБЕЖАТЬ")).upper()[:18]
        choice_b = str(choice_data.get("choice_b") or ("STAY" if is_en else "ОСТАТЬСЯ")).upper()[:18]
        question = "WHAT WOULD YOU CHOOSE? COMMENT BELOW" if is_en else "ЧТО БЫ СДЕЛАЛ ТЫ? НАПИШИ В КОММЕНТАРИЯХ"
        vote_label = "VOTE" if is_en else "ГОЛОСУЙ"
        sec_label = "SEC" if is_en else "СЕК"
        question_width = min(
            WIDTH - 60,
            int(draw.textlength(question, font=get_font(28 if is_en else 30))) + 44,
        )
        draw.rounded_rectangle(
            (540 - question_width // 2, 965, 540 + question_width // 2, 1028),
            radius=12, fill=(14, 18, 24), outline=accent, width=3,
        )
        draw.text((540, 996), question, font=get_font(28 if is_en else 30),
                  fill=(255, 205, 95), anchor="mm")
        for box, label, color in (
            ((75, 1055, 515, 1215), f"1. {choice_a}", (215, 48, 48)),
            ((565, 1055, 1005, 1215), f"2. {choice_b}", (42, 145, 95)),
        ):
            draw.rounded_rectangle(box, radius=20, fill=color, outline="white", width=4)
            draw.text(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
                      label, font=get_font(38), fill="white", anchor="mm")
        timer_value = max(1, 3 - (int(math.floor(phase * 3)) % 3))
        draw.rounded_rectangle((400, 1245, 680, 1308), radius=15, fill=accent)
        draw.text((540, 1276), f"{vote_label}: {timer_value} {sec_label}",
                  font=get_font(32), fill="white", anchor="mm")
        clock_cx, clock_cy, clock_r = 750, 1276, 34
        draw.ellipse((clock_cx - clock_r, clock_cy - clock_r,
                      clock_cx + clock_r, clock_cy + clock_r),
                     fill=(18, 22, 28), outline=(255, 205, 95), width=4)
        angle = (phase * math.tau) - math.pi / 2
        hand_x = clock_cx + int(math.cos(angle) * 23)
        hand_y = clock_cy + int(math.sin(angle) * 23)
        draw.line((clock_cx, clock_cy, hand_x, hand_y), fill="white", width=5)
        draw.ellipse((clock_cx - 5, clock_cy - 5, clock_cx + 5, clock_cy + 5),
                     fill=(255, 205, 95))

    # ========================================================================
    # ANIMATED KARAOKE SUBTITLES
    # ========================================================================
    if words:
        lines: list[list[tuple[int, str]]] = [[]]
        for index, word in enumerate(words):
            if sum(len(item[1]) + 1 for item in lines[-1]) + len(word) > 20 and lines[-1]:
                lines.append([])
            lines[-1].append((index, word))
        font_sub = get_font(58)
        base_y = 1365
        for line in lines[-3:]:
            widths = [int(draw.textlength(word.upper(), font=font_sub)) for _, word in line]
            spacing = 18
            start_x = 540 - (sum(widths) + spacing * (len(line) - 1)) // 2
            for (word_index_value, word), width in zip(line, widths):
                color = (255, 145, 45) if word_index_value == word_index else "white"
                draw.text((start_x + width // 2, base_y), word.upper(), font=font_sub,
                          fill=color, anchor="mm", stroke_width=6, stroke_fill=(8, 10, 14))
                start_x += width + spacing
            base_y += 82
    return image


# ============================================================================
# UNRESTRICTED AI SHOWRUNNER & VISUAL COMPOSER
# ============================================================================

def infer_visual_ops(text: str, scene_description: str = "", location: str = "") -> list[dict[str, Any]]:
    """Builds a rich default procedural canvas for Make It Easy AI across science, biology, tech, finance, games, and stories."""
    content = f"{text} {scene_description} {location}".lower()
    ops: list[dict[str, Any]] = []

    # 1. Choose dynamic background matching any setting or topic
    if any(w in content for w in ("космос", "солнц", "планет", "дыр", "звезд", "вселен", "галактик", "орбит")):
        ops.append({"op": "bg_space"})
    elif any(w in content for w in ("фронт", "бэк", "бек", "сервер", "код", "математ", "формул", "график", "финанс", "банк", "схем", "ии", "нейро")):
        ops.append({"op": "bg_lab_board"})
    elif any(w in content for w in ("лес", "дерев", "парк", "троп", "чащ", "лагер", "озер", "болот")):
        ops.append({"op": "bg_forest"})
        if any(w in content for w in ("дожд", "ливень", "гроз", "шторм")):
            ops.append({"op": "weather", "style": "rain", "count": 24})
        elif any(w in content for w in ("снег", "метел", "мороз", "зима", "пепел", "туман")):
            ops.append({"op": "weather", "style": "snow", "count": 20})
    elif any(w in content for w in ("вод", "океан", "море", "глубин", "бассейн", "аква", "затоп", "подвод", "корабл", "батискаф")):
        ops.append({"op": "bg_water"})
    elif any(w in content for w in ("улиц", "город", "трасс", "дорог", "мост", "парковк", "двор", "площад", "автобус", "машин", "поезд", "велик", "велосипед")):
        ops.append({"op": "bg_street"})
    elif any(w in content for w in ("лаборатор", "бункер", "станц", "завод", "цех", "склад", "шахт", "ангар", "метро", "туннел")):
        ops.append({"op": "bg_facility"})
    elif any(w in content for w in ("коридор", "лестниц", "лифт", "подъезд", "больниц", "школ", "отел", "этаж", "торгов")):
        ops.append({"op": "bg_corridor"})
    else:
        ops.append({"op": "bg_lab_board"})

    # 2. Add Edutainment / Infographics / Story focal objects
    if any(w in content for w in ("кост", "перелом", "скелет", "сустав", "рентген", "хрящ", "мозол")):
        ops.append({"op": "anatomy_xray", "x": 580, "y": 660, "r": 220, "style": "bone"})
    elif any(w in content for w in ("кров", "ран", "тромбоцит", "сосуд", "сердц", "клетк", "иммун", "зажив")):
        ops.append({"op": "anatomy_xray", "x": 580, "y": 660, "r": 220, "style": "blood"})
    elif any(w in content for w in ("мозг", "нейрон", "сознан", "сон", "смерт", "мысл", "памят", "нерв")):
        ops.append({"op": "anatomy_xray", "x": 580, "y": 660, "r": 220, "style": "brain"})
    elif any(w in content for w in ("фронт", "бэк", "бек", "сервер", "клиент", "пинг", "тикрейт", "запрос", "база дан", "движ")):
        ops.append({"op": "diagram_flow", "y": 630, "left_label": "FRONTEND / ИГРОК", "right_label": "BACKEND / СЕРВЕР"})
    elif any(w in content for w in ("деньг", "банк", "инфляц", "доллар", "финанс", "акци", "экономик", "цен", "богат")):
        ops.append({"op": "chart_bars", "x": 160, "y": 840})
    elif any(w in content for w in ("математ", "учител", "урок", "формул", "алгебр", "геометр", "логик", "задач")):
        ops.append({"op": "math_board", "x": 140, "y": 380, "header": "ЛОГИКА · НЕЙРОНЫ · АЛГОРИТМ"})
    elif any(w in content for w in ("ядерн", "атом", "гриб", "радиац", "взрыв", "бомб", "детонац", "апокалипс")):
        ops.append({"op": "nuke_mushroom", "x": 540, "y": 660})
    elif any(w in content for w in ("палец", "пальц", "правил", "сравн", "кулак", "рука", "вытян")):
        ops.append({"op": "thumb_rule", "x": 540, "y": 820, "label": "ПРАВИЛО БОЛЬШОГО ПАЛЬЦА"})
    elif any(w in content for w in ("двер", "вход", "замок", "заперт", "стук", "глазок", "шлюз")):
        ops.append({"op": "door_closeup", "x": 280, "y": 350, "locked": True})
    elif any(w in content for w in ("окн", "стекл", "витрин", "иллюминатор")):
        ops.append({"op": "window_view", "x": 220, "y": 360, "broken": "разби" in content or "трещин" in content})
    elif any(w in content for w in ("здани", "многоэтаж", "дом", "крыш", "этаж")):
        ops.append({"op": "building_grid", "x": 160, "y": 340, "w": 760, "h": 800})

    if any(w in content for w in ("радио", "частот", "эфир", "раци", "сигнал", "динамик", "колонк", "запис")):
        ops.append({"op": "radio_kit", "x": 180, "y": 460, "frequency": "СИГНАЛ · ЭФИР"})
    elif any(w in content for w in ("телефон", "смартфон", "экран", "звон", "сообщен", "смс", "камер", "чат")):
        ops.append({"op": "phone_ui", "x": 270, "y": 360, "header": "MAKE IT EASY AI", "subtext": "НАГЛЯДНЫЙ РАЗБОР"})
    elif any(w in content for w in ("кран", "вентил", "рычаг", "клапан", "рубильник", "кнопк", "механизм")):
        ops.append({"op": "valve_diagram", "x": 540, "y": 700, "r": 165})
        ops.append({"op": "dashed_zone", "x": 540, "y": 700, "r": 230})

    if any(w in content for w in ("фонар", "луч", "свет", "прожектор", "фары")):
        ops.append({"op": "light_beam", "x": 540, "y": 360, "tx": 540, "ty": 1180})

    if any(w in content for w in ("сущност", "монстр", "силуэт", "тень", "фигур", "оно", "кто-то", "двойник", "существо")):
        ops.append({"op": "entity", "x": 680, "y": 680, "scale": 1.35, "eyes": 2})

    emotion = "curious"
    if any(w in content for w in ("бол", "упал", "перелом", "ран", "кров", "удар")):
        emotion = "pain"
    elif any(w in content for w in ("шок", "взрыв", "опасн", "смерт", "страх")):
        emotion = "shock"
    elif any(w in content for w in ("почему", "зачем", "секрет", "дума", "математ")):
        emotion = "thinking"

    ops.append({"op": "actor", "x": 270, "y": 920, "scale": 0.95, "emotion": emotion, "pose": "point"})
    return ops


def _normalize_episode(
    episode: dict[str, Any],
    index: int,
    total: int,
    final: bool,
) -> dict[str, Any]:
    scenes = episode.get("scenes")
    if not isinstance(scenes, list) or not 6 <= len(scenes) <= 12:
        raise ValueError(f"Акт {index + 1}: требуется от 6 до 12 сцен")
    total_scene_time = 50.0
    step = total_scene_time / len(scenes)
    normalized: list[dict[str, Any]] = []
    ep_location = str(episode.get("location", ""))
    for scene_index, raw in enumerate(scenes):
        if not isinstance(raw, dict) or not str(raw.get("text", "")).strip():
            raise ValueError(f"Акт {index + 1}, сцена {scene_index + 1}: пустая сцена")
        scene = dict(raw)
        start = round(scene_index * step, 2)
        end = round((scene_index + 1) * step, 2)
        scene["timing"] = f"{start:g}-{end:g}"
        scene["text"] = str(scene["text"]).strip()
        scene["rule_badge"] = str(scene.get("rule_badge", "")).strip()[:40]
        scene["visual_detail"] = str(scene.get("visual_detail", "")).strip()[:80]
        scene["scene_description"] = str(scene.get("scene_description", "")).strip()[:220]
        scene["image_prompt"] = str(scene.get("image_prompt", "")).strip()[:320]
        scene["clock"] = str(scene.get("clock") or f"0{index + 1}:{15 + scene_index * 4:02d}")[:8]
        scene["bpm"] = int(_clamp_number(scene.get("bpm", 120 + scene_index * 5), 90, 195, 130))
        scene["timer_label"] = str(scene.get("timer_label", "")).strip()[:20]
        scene["is_ending"] = bool(final and scene_index == len(scenes) - 1)
        scene["is_choice_scene"] = scene["is_ending"]
        if scene["is_ending"] and "что" not in scene["text"].lower()[-35:]:
            scene["text"] = f"{scene['text'].rstrip()} А что думаешь ты?"
        scene["choice_a"] = str(scene.get("choice_a") or "ВАРИАНТ 1").upper()[:18] if scene["is_ending"] else ""
        scene["choice_b"] = str(scene.get("choice_b") or "ВАРИАНТ 2").upper()[:18] if scene["is_ending"] else ""
        raw_canvas = sanitize_canvas(scene.get("canvas", []))
        if not raw_canvas:
            raw_canvas = infer_visual_ops(scene["text"], scene["scene_description"], ep_location)
        scene["canvas"] = raw_canvas
        scene["bg_top"] = _rgb(scene.get("bg_top") or scene.get("bg_color"), (18, 26, 38))
        scene["bg_bottom"] = _rgb(scene.get("bg_bottom"), (8, 12, 20))
        scene["accent_color"] = _rgb(scene.get("accent_color"), (55, 195, 245))
        normalized.append(scene)
    episode = dict(episode)
    episode["episode"] = index + 1
    episode["total_episodes"] = total
    episode["scenes"] = normalized
    return episode


def scan_youtube_market_trends() -> str:
    """Scans viral edutainment & storytelling trends across YouTube Shorts, TikTok, and Instagram Reels
    for the 'Make It Easy AI (Всё сложное — просто)' channel."""
    yt_queries = [
        "что происходит с телом когда объяснение на пальцах shorts",
        "как работают игры изнутри фронтенд и бэкенд простыми словами shorts",
        "что будет если научный разбор за 60 секунд shorts",
        "как работают деньги и финансы простыми словами shorts",
        "почему в школе нас учат математике на самом деле shorts",
        "правила выживания захватывающая история выбор shorts",
        "психологическая история загадка а что выберешь ты shorts",
        "what happens to your body when explained simply shorts",
        "how video games actually work under the hood shorts",
        "survival story dilemma what would you choose shorts",
    ]
    chosen_query = random.choice(yt_queries)
    print(f"📊 [Make It Easy AI Radar] Сканируем тренды YouTube Shorts + TikTok + Reels: «{chosen_query}»...")
    top_items: list[str] = []

    # 1. YouTube Shorts radar via yt-dlp
    try:
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch6:{chosen_query}", download=False)
            entries = info.get("entries", []) if isinstance(info, dict) else []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                title = str(entry.get("title") or "").strip()
                views = entry.get("view_count")
                if title:
                    views_str = f"{views:,} просм." if isinstance(views, int) else "в тренде"
                    top_items.append(f"- [YouTube Shorts] «{title}» ({views_str})")
    except Exception as err:
        print(f"ℹ️ Быстрый скан YouTube пропущен ({err}).")

    # 2. TikTok & Instagram Reels trend radar via DuckDuckGo (ddgs)
    try:
        from ddgs import DDGS

        social_query = random.choice([
            "site:tiktok.com what happens when you break a bone animation",
            "site:instagram.com/reel how things work simple animation",
            "site:tiktok.com почему так происходит простое объяснение",
            "site:tiktok.com правила выживания история выбор",
        ])
        with DDGS() as ddgs:
            for res in ddgs.text(social_query, max_results=4):
                title = str(res.get("title") or "").strip()
                if title:
                    platform = "TikTok" if "tiktok" in social_query else "Instagram Reels"
                    top_items.append(f"- [{platform}] «{title[:90]}»")
    except Exception:
        pass

    if top_items:
        summary = "\n".join(top_items[:8])
        print(f"🔥 Найдены горячие темы 3 платформ (YT + TikTok + Reels):\n{summary}")
        return summary

    return (
        "- [Биология и Тело] «Упал с велика: как за секунду ломается кость, почему течёт кровь и как тело склеивает себя обратно»\n"
        "- [Тайны жизни] «Что наступает в мозге и теле в первые минуты после смерти с точки зрения науки»\n"
        "- [Игры и IT] «Как общаются фронтенд и бэкенд в онлайн-игре за 20 миллисекунд и почему тебя убивают за стенкой»\n"
        "- [Школа и Мозг] «Почему учителя так хотят, чтобы ты понимал математику: что происходит с нейронами»\n"
        "- [Финансы] «Почему нельзя просто напечатать миллион долларов каждому и как работают деньги»\n"
        "- [Сюжетная История / Выживание] «Ночной лифт остановился между этажами: 3 правила и финальный выбор»"
    )


def generate_series_script(num_episodes: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    history = load_series_history()
    custom_topic = os.getenv("STORY_TOPIC", "").strip()
    market_trends = scan_youtube_market_trends() if not custom_topic else ""
    topic_instruction = (
        f"Тема выпуска от автора: «{custom_topic}»."
        if custom_topic
        else (
            "Проанализируй текущие вирусные тренды YouTube Shorts, TikTok и Reels ниже и выбери САМУЮ ЦЕПЛЯЮЩУЮ, "
            "высокооплачиваемую (высокий RPM YouTube) и залипательную тему для нашего 2D-мультфильма "
            "на канале «Make It Easy AI» («Всё сложное — просто»). Чередуй научно-популярные разборы И сюжетные истории (смотря что ещё не выходило в последних выпусках):\n"
            "  1) БИОЛОГИЯ И ЖИЗНЕННЫЕ СИТУАЦИИ («Что если ехал на велике и упал: как происходит перелом кости, почему течёт кровь и как клетки заживляют рану», «Что наступает в мозге и теле после смерти», «Почему возникает сонный паралич»);\n"
            "  2) ИГРЫ, ФРОНТЕНД И БЭКЕНД, ТЕХНОЛОГИИ («Как работает онлайн-игра изнутри: что делает фронтенд на твоём экране, а что считает бэкенд-сервер за 20 мс», «Как создаются игры и физика движка»);\n"
            "  3) ФИНАНСЫ И ЭКОНОМИКА НА ПАЛЬЦАХ («Почему нельзя просто напечатать всем миллион», «Как работают банки и деньги в телефоне»);\n"
            "  4) ШКОЛА, МАТЕМАТИКА, КОСМОС И ИСТОРИЯ («Почему учителя так хотят, чтобы ты понимал математику», «Что если Солнце погаснет на 1 час», «Как строили мосты в древности»);\n"
            "  5) СЮРРЕАЛИСТИЧНЫЕ ПАРАДОКСЫ, ВЫЖИВАНИЕ И АПОКАЛИПСИСЫ («Что делать, если увидел ядерный гриб: правило большого пальца», «Что наступает в первую секунду после смерти», «Что будет, если на Земле исчезнет кислород на 5 секунд»);\n"
            "  6) ЗАХВАТЫВАЮЩИЕ СЮЖЕТНЫЕ ИСТОРИИ И ТАЙНЫ («А что выберешь ты?»: детективные загадки, экстремальное выживание или городская мистика с 1 минималистичной сущностью и правилами выживания).\n\n"
            f"Свежие сигналы рынка YouTube / TikTok / Reels:\n{market_trends}"
        )
    )
    prompt = f"""
Ты — главный шоураннер, сценарист и 2D-арт-директор научно-популярного и сюжетного анимационного канала
«Make It Easy AI» (слоган: «Всё сложное — просто!»).
Наша аудитория — школьники, подростки и взрослые, которые обожают, когда сложные вещи (биология, устройство игр, фронтенд/бэкенд, финансы, космос, медицина, история или крутые сюжеты) объясняются на конкретной живой ситуации простым, ярким, увлекательным языком в формате 2D-мультика!
{topic_instruction}

ЖЕЛЕЗНЫЕ ПРАВИЛА СЦЕНАРИЯ «MAKE IT EASY AI»:
1. 🔥 ПЕРВЫЕ 5 СЕКУНД (СЦЕНА №1) — АБСОЛЮТНЫЙ РАЗЪЁБ И ЗАХВАТ ВНИМАНИЯ!
   - КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО начинать с «Привет» или медленного вступления!
   - Первая сцена (`scenes[0]`) должна длиться ровно 4–5 секунд (10–14 слов максимум!) и сразу бросать зрителя в самый шокирующий момент, парадокс или взрывную ситуацию (например: «Не смей закрывать глаза! Через три секунды после этой вспышки воздух вокруг раскалится до тысячи градусов!»).
   - Со второй сцены разложи объяснение пошагово (СТЕП-БАЙ-СТЕП): от зацепки в первой сцене к наглядному механизму шаг за шагом.
   - Длина предложений со 2-й сцены: 14-22 слова (чтобы диктор говорил спокойно, размеренно и понятно, а общая длительность ролика комфортно укладывалась в 45-55 секунд).
   Например:
   - Если тема про перелом и заживление раны — начни с того, как герой летит на велике, падает на асфальт, и дальше мы «ныряем» внутрь тела через рентген `anatomy_xray`, показывая трещину кости, поток эритроцитов и тромбоцитов (`style="blood"`) и как остеобласты строят костную мозоль!
   - Если тема про выживание или апокалипсис («Что делать, если увидел ядерный гриб») — покажи горизонт и ядерный гриб `nuke_mushroom`, затем как герой вытягивает руку с большим пальцем `thumb_rule` (если палец перекрывает гриб — ты вне зоны мгновенного поражения, если нет — немедленно падай лицом вниз и закрой уши от ударной волны), пошагово объясняя физику ударной волны и радиоактивного следа!
   - Если тема про создание игр (фронтенд и бэкенд) — покажи игрока за экраном, затем схему `diagram_flow` (`FRONTEND / КЛИЕНТ` <-> `BACKEND / СЕРВЕР`), как нажатие кнопки летит пакетом на сервер и возвращается!
   - Если тема про «Что наступает после смерти» или «Почему учителя хотят, чтобы ты понимал математику» — веди зрителя шаг за шагом через мозг (`anatomy_xray` со `style="brain"`), доску `math_board` и наглядные примеры!
2. Не повторяй прошлые названия: {history["used_titles"][-15:]}.
3. Создай выпуск из РОВНО {num_episodes} связанных актов; каждый акт содержит ровно 8-10 сцен для глубокого пошагового погружения.
4. Пиши живым, дружелюбным, чистым и понятным языком без спешки (ровные законченные мысли, чтобы диктор не тараторил).
5. В поле `rule_badge` пиши короткий шаг или факт (например: «ШАГ №1», «ФАКТ №1», «ВНУТРИ ТЕЛА», «СЕКРЕТ №2»).
6. Только ПОСЛЕДНЯЯ сцена ПОСЛЕДНЕГО акта заканчивается вопросом к зрителю («А что думаешь ты?» или «А какую тему разобрать следующей?») и содержит два варианта для голосования в комментариях `choice_a` и `choice_b` (до 16 символов каждый).
7. Для КАЖДОЙ сцены обязательно создай наглядный 2D-векторный `canvas` из 4–6 операций, активно используя новые инфографические и сюжетные примитивы:
   Доступные операции `op`:
   bg_lab_board, bg_space, bg_forest, bg_room, bg_corridor, bg_street, bg_water, bg_facility,
   anatomy_xray (параметр style: "bone" | "blood" | "brain"),
   diagram_flow (параметры left_label, right_label),
   nuke_mushroom, thumb_rule (параметр label),
   chart_bars, math_board (параметр header),
   door_closeup, window_view, valve_diagram, dashed_zone, stickman,
   actor (параметры emotion: "curious" | "happy" | "thinking" | "pain" | "shock", pose: "point" | "stand" | "lie" | "phone"),
   entity, hands_reach, key_palm, padlock_chain, building_grid, highlight_ring,
   light_beam, weather, rect, circle, line, phone_ui, radio_kit, callout.

Верни ТОЛЬКО валидный JSON:
{{
  "series_title": "Короткое цепляющее название выпуска",
  "location": "Рубрика · Наглядный разбор",
  "premise": "Краткая суть объяснения",
  "cliffhanger_summary": "Вопрос к зрителям",
  "episodes": [
    {{
      "episode": 1,
      "title": "Название акта",
      "location": "Локация / Слой разбора",
      "atmosphere": "Яркая 2D-инфографика",
      "scenes": [
        {{
          "text": "Увлекательный текст диктора простыми словами",
          "rule_badge": "ФАКТ №1",
          "visual_detail": "Короткая плашка-деталь (до 35 символов)",
          "scene_description": "Что именно рисуется в 2D-кадре",
          "clock": "00:15",
          "bpm": 115,
          "bg_top": [20, 32, 48],
          "bg_bottom": [10, 16, 26],
          "accent_color": [55, 195, 245],
          "choice_a": "",
          "choice_b": "",
          "canvas": [
            {{"op": "bg_lab_board"}},
            {{"op": "anatomy_xray", "x": 600, "y": 660, "r": 220, "style": "bone"}},
            {{"op": "actor", "x": 260, "y": 920, "scale": 0.95, "emotion": "curious", "pose": "point"}},
            {{"op": "callout", "x": 580, "y": 420, "tx": 600, "ty": 640, "label": "ТРЕЩИНА В КОСТИ"}}
          ]
        }}
      ]
    }}
  ]
}}
"""
    last_error: Exception | None = None
    candidate_models = [
        os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
    ]

    for attempt in range(5):
        model_name = candidate_models[min(attempt, len(candidate_models) - 1)]
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.9,
                ),
            )
            data: Any = json.loads(response.text)
            raw_episodes = data.get("episodes") if isinstance(data, dict) else None
            if not isinstance(raw_episodes, list) or len(raw_episodes) != num_episodes:
                raise ValueError(f"Ожидалось {num_episodes} актов, получено: {type(raw_episodes)}")
            episodes = [
                _normalize_episode(raw, index, num_episodes, index == num_episodes - 1)
                for index, raw in enumerate(raw_episodes)
            ]
            meta = {
                "series_title": str(data.get("series_title") or episodes[0].get("title", "Правила выживания")),
                "location": str(data.get("location") or episodes[0].get("location", "НЕИЗВЕСТНЫЙ СЕКТОР")),
                "premise": str(data.get("premise") or data.get("series_title") or "оригинальный выпуск"),
                "cliffhanger_summary": str(data.get("cliffhanger_summary", "")),
            }
            return episodes, meta
        except Exception as error:
            last_error = error
            print(f"⚠️ Попытка генерации сценария {attempt + 1}/5 ({model_name}): {error}")
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"Не удалось получить сценарий от Gemini после 5 попыток: {last_error}"
    )


def prepare_episode_visuals(episode: dict[str, Any], series_meta: dict[str, Any]) -> dict[str, Any]:
    """Ensures every scene in the episode has rich, distinct 2D procedural vector canvas layers."""
    ep_location = str(episode.get("location") or series_meta["location"])
    scenes = episode["scenes"]
    for idx, scene in enumerate(scenes):
        scene.pop("_cached_ai_image", None)
        inferred = infer_visual_ops(
            scene["text"],
            scene.get("scene_description", ""),
            ep_location,
        )
        current_canvas = scene.get("canvas") or []
        if len(current_canvas) < 3:
            existing_ops = {c.get("op") for c in current_canvas if isinstance(c, dict)}
            for op_item in inferred:
                if op_item["op"] not in existing_ops:
                    current_canvas.append(op_item)
                    existing_ops.add(op_item["op"])
        # Add subtle motion dx/dy and visual callout if missing so every scene feels alive
        for cmd in current_canvas:
            if isinstance(cmd, dict) and cmd.get("op") in {"actor", "entity", "stickman"} and "dx" not in cmd:
                cmd["dx"] = 14 if idx % 2 == 0 else -14
        if not any(isinstance(c, dict) and c.get("op") == "callout" for c in current_canvas):
            badge = str(scene.get("visual_detail") or scene.get("rule_badge") or "").strip()
            if badge:
                current_canvas.append({
                    "op": "callout",
                    "x": 540,
                    "y": 460 + (idx % 3) * 45,
                    "tx": 580,
                    "ty": 760,
                    "label": badge[:26],
                })
        scene["canvas"] = current_canvas
    return episode


def _synthesize_edge_or_gtts(text: str, path: Path, lang: str = "ru") -> None:
    import subprocess
    voice_name = "en-US-ChristopherNeural" if str(lang).lower() == "en" else "ru-RU-DmitryNeural"
    edge_bin = str(Path(sys.executable).parent / "edge-tts")
    if not os.path.exists(edge_bin):
        edge_bin = "edge-tts"
    res_edge = subprocess.run(
        [edge_bin, "--voice", voice_name, "--text", text, "--write-media", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if res_edge.returncode != 0 or not path.exists() or path.stat().st_size < 100:
        from gtts import gTTS
        gTTS(text=text, lang="en" if str(lang).lower() == "en" else "ru").save(str(path))


def generate_voice(text: str, filename: str, lang: str = "ru") -> Path:
    path = AUDIO_DIR / filename
    economy_mode = os.getenv("ELEVENLABS_ECONOMY", "0").lower() in {"1", "true", "yes"}
    en_edge = os.getenv("ELEVENLABS_EN_EDGE", "1").lower() in {"1", "true", "yes"}
    if economy_mode or (en_edge and str(lang).lower() == "en") or eleven_client is None:
        _synthesize_edge_or_gtts(text, path, lang=lang)
        return path

    try:
        stream = eleven_client.text_to_speech.convert(
            text=text,
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"),
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.82,
                similarity_boost=0.88,
                style=0.0,
                use_speaker_boost=True,
            ),
        )
        with path.open("wb") as file:
            for chunk in stream:
                if chunk:
                    file.write(chunk)
    except Exception as error:
        print(f"⚠️ Ошибка ElevenLabs ({error}), переключаемся на бесплатный нейроголос Edge-TTS...")
        _synthesize_edge_or_gtts(text, path, lang=lang)
    return path


def make_scene_clip(
    scene: dict[str, Any],
    story_meta: dict[str, Any],
    audio_path: Path | None,
    timestamp: str,
    scene_number: int,
    clips: list[Any],
    audio_clips: list[Any],
    duration_scale: float = 1.0,
    absolute_start: float = 0.0,
    choice_overlay_start: list[float] | None = None,
    choice_source: dict[str, Any] | None = None,
    lang: str = "ru",
) -> None:
    audio = None
    if audio_path is not None:
        # Keep natural audio speed and pitch (no with_speed_scaled pitch distortion)
        source_audio = AudioFileClip(str(audio_path))
        audio_clips.append(source_audio)
        audio = source_audio
    else:
        scene_duration_for_clip = get_scene_duration(scene) * max(duration_scale, 0.1)

    words = str(scene.get("text", "")).split()
    duration = max(float(audio.duration), 0.1) if audio is not None else scene_duration_for_clip
    if not words:
        words = [""]
    num_words = max(len(words), 1)
    word_duration = duration / num_words

    # Validate initial frame once
    test_frame = np.asarray(
        draw_scene(scene, story_meta, 0.0, 0, words, show_choice=False, scene_number=scene_number, lang=lang),
        dtype=np.uint8,
    )
    if test_frame.shape != (HEIGHT, WIDTH, 3) or not test_frame.any():
        raise RuntimeError(f"Пустой кадр сцены {scene_number}")

    def make_scene_frame(t: float) -> np.ndarray:
        clamped_t = min(max(float(t), 0.0), duration)
        word_index = min(int(clamped_t / word_duration), num_words - 1)
        motion_phase = (clamped_t / duration) * 4.0
        absolute_time = absolute_start + clamped_t
        show_choice = (
            choice_overlay_start is not None
            and absolute_time >= choice_overlay_start[0]
            and choice_source is not None
        )
        return np.asarray(
            draw_scene(
                scene,
                story_meta,
                motion_phase,
                word_index,
                words,
                show_choice=show_choice,
                choice_source=choice_source,
                scene_number=scene_number,
                lang=lang,
            ),
            dtype=np.uint8,
        )

    base_clip = VideoClip(frame_function=make_scene_frame, duration=duration)
    scene_clip = base_clip.with_audio(audio) if audio is not None else base_clip
    clips.append(scene_clip)


def translate_series_to_english(
    episodes: list[dict[str, Any]],
    series_meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Translates entire episode texts, HUD badges, and choices to English for the dual release."""
    print("🌐 Переводим историю на английский язык для EN-версии...")
    compact_payload = {
        "series_title": series_meta.get("series_title", ""),
        "location": series_meta.get("location", ""),
        "episodes": [
            {
                "episode": ep["episode"],
                "title": ep.get("title", ""),
                "location": ep.get("location", ""),
                "scenes": [
                    {
                        "text": sc["text"],
                        "rule_badge": sc.get("rule_badge", ""),
                        "visual_detail": sc.get("visual_detail", ""),
                        "timer_label": sc.get("timer_label", ""),
                        "choice_a": sc.get("choice_a", ""),
                        "choice_b": sc.get("choice_b", ""),
                    }
                    for sc in ep["scenes"]
                ],
            }
            for ep in episodes
        ],
    }

    prompt = f"""
Translate the following 'Make It Easy AI' edutainment & storytelling script JSON from Russian to clear, engaging, natural English for YouTube Shorts narration.
Rules:
1. Translate `text` naturally for a captivating English narrator voice (short, clear, punchy sentences).
2. If the scene ends with "А что думаешь ты?" or "А что выберешь ты?", translate it to "What do you think?" or "What would you choose?".
3. Translate `rule_badge` naturally (e.g. "STEP #1", "FACT #1", "INSIDE THE BODY").
4. Keep `choice_a` and `choice_b` short (1-3 words max).
5. Return strictly valid JSON preserving the exact array structures and scene counts.

JSON to translate:
{json.dumps(compact_payload, ensure_ascii=False)}
"""
    for candidate_model in [os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"), "gemini-3.1-flash-lite", "gemini-3.8-flash"]:
        try:
            response = gemini_client.models.generate_content(
                model=candidate_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            translated_data = json.loads(response.text)
            en_episodes: list[dict[str, Any]] = []
            for orig_ep, trans_ep in zip(episodes, translated_data.get("episodes", [])):
                ep_copy = dict(orig_ep)
                ep_copy["title"] = trans_ep.get("title") or orig_ep.get("title")
                ep_copy["location"] = trans_ep.get("location") or orig_ep.get("location")
                new_scenes = []
                for orig_sc, trans_sc in zip(orig_ep["scenes"], trans_ep.get("scenes", [])):
                    sc_copy = dict(orig_sc)
                    sc_copy["text"] = str(trans_sc.get("text") or orig_sc["text"]).strip()
                    sc_copy["rule_badge"] = str(trans_sc.get("rule_badge") or orig_sc.get("rule_badge", "")).strip()
                    sc_copy["visual_detail"] = str(trans_sc.get("visual_detail") or orig_sc.get("visual_detail", "")).strip()
                    sc_copy["timer_label"] = str(trans_sc.get("timer_label") or orig_sc.get("timer_label", "")).strip()
                    sc_copy["choice_a"] = str(trans_sc.get("choice_a") or orig_sc.get("choice_a", "")).upper()[:18]
                    sc_copy["choice_b"] = str(trans_sc.get("choice_b") or orig_sc.get("choice_b", "")).upper()[:18]
                    # Also adapt canvas callouts if present
                    new_canvas = []
                    for op in sc_copy.get("canvas", []):
                        op_copy = dict(op)
                        if op_copy.get("op") == "callout" and sc_copy["visual_detail"]:
                            op_copy["label"] = sc_copy["visual_detail"][:26].upper()
                        new_canvas.append(op_copy)
                    sc_copy["canvas"] = new_canvas
                    new_scenes.append(sc_copy)
                ep_copy["scenes"] = new_scenes
                en_episodes.append(ep_copy)

            en_meta = dict(series_meta)
            en_meta["series_title"] = translated_data.get("series_title") or series_meta["series_title"]
            en_meta["location"] = translated_data.get("location") or series_meta["location"]
            return en_episodes, en_meta
        except Exception as error:
            print(f"⚠️ Попытка перевода через {candidate_model}: {error}")
    print("⚠️ Не удалось перевести через Gemini, оставляем оригинал для EN.")
    return episodes, series_meta


def save_multiplatform_metadata(
    video_path: Path,
    series_meta: dict[str, Any],
    episodes: list[dict[str, Any]],
    lang: str = "ru",
) -> Path:
    """Generates ready-to-post titles, descriptions, captions, and hashtags for
    YouTube Shorts, TikTok (Creator Rewards >60s), and Instagram Reels."""
    is_en = (lang.lower() == "en")
    title = series_meta.get("series_title", "Make It Easy AI")
    loc = series_meta.get("location", "")
    final_scene = episodes[-1]["scenes"][-1] if episodes and episodes[-1].get("scenes") else {}
    choice_a = final_scene.get("choice_a") or ("OPTION 1" if is_en else "ВАРИАНТ 1")
    choice_b = final_scene.get("choice_b") or ("OPTION 2" if is_en else "ВАРИАНТ 2")

    if is_en:
        meta_bundle = {
            "language": "EN",
            "video_file": str(video_path),
            "youtube_shorts": {
                "title": f"{title} 🤯🧠 ({choice_a} vs {choice_b}) #shorts #explained #science",
                "description": (
                    f"{title} — Make It Easy AI ({loc}).\n"
                    f"Everything complex made simple! What do YOU think: 1) {choice_a} or 2) {choice_b}?\n"
                    "Drop your answer in the comments! 👇\n\n"
                    "#shorts #makeiteasyai #explained #science #tech #biology #animation #howitworks"
                ),
            },
            "tiktok": {
                "caption": (
                    f"{title} 🤯🧠 Explained simply! Vote in comments: 1 ({choice_a}) or 2 ({choice_b})? 👇 "
                    "#fyp #viral #explained #science #howitworks #animation #learnontiktok"
                ),
            },
            "instagram_reels": {
                "caption": (
                    f"1 ({choice_a}) or 2 ({choice_b})? 👇 Everything complex — made simple!\n.\n.\n"
                    f"Episode: {title} ({loc})\n"
                    "#reels #explorepage #explained #science #tech #animation #makeiteasyai"
                ),
            },
        }
    else:
        meta_bundle = {
            "language": "RU",
            "video_file": str(video_path),
            "youtube_shorts": {
                "title": f"{title} 🤯🧠 ({choice_a} или {choice_b}?) #shorts #наука #простымисловами",
                "description": (
                    f"{title} — Make It Easy AI: Всё сложное — просто! ({loc}).\n"
                    f"А что думаешь ТЫ: 1) {choice_a} или 2) {choice_b}?\n"
                    "Пиши свой вариант в комментариях! 👇\n\n"
                    "#shorts #makeiteasyai #простымисловами #наука #технологии #биология #анимация #факты"
                ),
            },
            "tiktok": {
                "caption": (
                    f"{title} 🤯🧠 Всё сложное — просто! Пиши в комменты: 1 ({choice_a}) или 2 ({choice_b})? 👇 "
                    "#рек #наука #простымисловами #факты #анимация #какэтоработает #интересно"
                ),
            },
            "instagram_reels": {
                "caption": (
                    f"1 ({choice_a}) или 2 ({choice_b})? 👇 Всё сложное — простыми словами!\n.\n.\n"
                    f"Выпуск: {title} ({loc})\n"
                    "#reels #рекомендации #наука #факты #простымисловами #анимация #технологии"
                ),
            },
        }

    meta_path = video_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta_bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path


def render_story_video(
    episodes: list[dict[str, Any]],
    series_meta: dict[str, Any],
    timestamp: str,
    index: int = 1,
    lang: str = "ru",
) -> Path:
    """Renders a single language edition (RU or EN) of the horror series."""
    tag = lang.upper()
    output_path = VIDEOS_DIR / f"survival_story_{timestamp}_{index}_{tag}.mp4"
    duration_scale = MAX_SHORT_SECONDS / (len(episodes) * 50.0)
    choice_overlay_start = [MAX_SHORT_SECONDS - 10.0]
    clips: list[Any] = []
    audio_clips: list[Any] = []

    print(
        f"🎙️🎬 [{tag}] Собираем сцены выпуска «{series_meta['series_title']}» "
        + ("с озвучкой ElevenLabs..." if VOICE_ENABLED else "без озвучки...")
    )
    global_scene_number = 0
    absolute_start = 0.0
    final_choice_source = episodes[-1]["scenes"][-1]

    try:
        for episode in episodes:
            story_meta = {
                "series_tag": f"{series_meta['series_title']} · ACT {episode['episode']}/{len(episodes)}",
                "location": episode.get("location") or series_meta["location"],
            }
            for scene in episode["scenes"]:
                global_scene_number += 1
                audio_path = (
                    generate_voice(
                        scene["text"],
                        f"{timestamp}_{lang}_scene{global_scene_number}.mp3",
                        lang=lang,
                    )
                    if VOICE_ENABLED else None
                )
                make_scene_clip(
                    scene,
                    story_meta,
                    audio_path,
                    timestamp,
                    global_scene_number,
                    clips,
                    audio_clips,
                    duration_scale=duration_scale,
                    absolute_start=absolute_start,
                    choice_overlay_start=choice_overlay_start,
                    choice_source=final_choice_source,
                    lang=lang,
                )
                absolute_start += clips[-1].duration

        total_duration = sum(clip.duration for clip in clips)
        choice_overlay_start[0] = max(0.0, min(total_duration, MAX_SHORT_SECONDS) - 10.0)
        print(f"🎬 [{tag}] Рендерим вертикальное видео ({total_duration:.1f} сек.)...")
        final_video = concatenate_videoclips(clips, method="chain")
        temp_audiofile = str(AUDIO_DIR / f"temp_{timestamp}_{index}_{lang}_audio.mp4")
        try:
            final_video.write_videofile(
                str(output_path),
                fps=FPS,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=temp_audiofile,
                preset="ultrafast",
                ffmpeg_params=["-crf", "28"],
                threads=4,
                logger="bar",
            )
        finally:
            final_video.close()
            if os.path.exists(temp_audiofile):
                try:
                    os.remove(temp_audiofile)
                except OSError:
                    pass
        meta_file = save_multiplatform_metadata(output_path, series_meta, episodes, lang=lang)
        print(f"✅ [{tag}] Видео готово: {output_path}")
        print(f"📋 [{tag}] Метаданные (YouTube + TikTok + Reels): {meta_file}")
        return output_path
    finally:
        for clip in clips + audio_clips:
            clip.close()


def build_single_short(index: int = 1) -> list[Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🤖 [1/5] ИИ-шоураннер пишет оригинальный сценарий ({EPISODES_PER_SHORT} акта)...")
    if VOICE_ENABLED:
        check_elevenlabs_quota(min_required_chars=800)
    episodes_ru, series_meta_ru = generate_series_script(EPISODES_PER_SHORT)
    for episode in episodes_ru:
        print(
            f"🎨 Подготовка визуала для Акта {episode['episode']}/{EPISODES_PER_SHORT}: "
            f"«{episode.get('title', series_meta_ru['series_title'])}» ({len(episode['scenes'])} сцен)..."
        )
        prepare_episode_visuals(episode, series_meta_ru)

    generated_videos: list[Path] = []
    # 1. Russian edition
    print("🇷🇺 [2/5] Рендер РУССКОЙ версии...")
    ru_video = render_story_video(episodes_ru, series_meta_ru, timestamp, index=index, lang="ru")
    generated_videos.append(ru_video)

    save_series_history(
        load_series_history(),
        series_meta_ru["series_title"],
        series_meta_ru["location"],
        series_meta_ru["premise"],
    )

    # 2. English edition (мультимедийка / двойной оклад)
    enable_english = os.getenv("ENABLE_ENGLISH", "1").lower() in {"1", "true", "yes"}
    if enable_english:
        print("🇺🇸 [3/5] Рендер АНГЛИЙСКОЙ версии...")
        episodes_en, series_meta_en = translate_series_to_english(episodes_ru, series_meta_ru)
        en_video = render_story_video(episodes_en, series_meta_en, timestamp, index=index, lang="en")
        generated_videos.append(en_video)

    if os.getenv("UPLOAD_TO_YOUTUBE", "1").lower() in {"1", "true", "yes"}:
        defer_en = os.getenv("DEFER_EN_UPLOAD", "0").lower() in {"1", "true", "yes"}
        print("📤 [4/5] Загружаем ролики на YouTube...")
        try:
            from uploader import upload_short

            for video_file in generated_videos:
                meta_path = video_file.with_suffix(".meta.json")
                yt_title = f"{series_meta_ru['series_title']} #shorts"
                yt_desc = "#shorts #makeiteasyai #explained"
                if meta_path.exists():
                    try:
                        meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
                        yt_info = meta_obj.get("youtube_shorts", {})
                        yt_title = yt_info.get("title") or yt_title
                        yt_desc = yt_info.get("description") or yt_desc
                    except Exception:
                        pass
                if defer_en and video_file.stem.endswith("_EN"):
                    queue_file = VIDEOS_DIR / "pending_en_upload.json"
                    queue_file.write_text(
                        json.dumps({"video_file": str(video_file), "title": yt_title, "description": yt_desc}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(f"⏳ [EN] Английский мультик сохранён в очередь на вечерний прайм-тайм (18:30): {video_file.name}")
                else:
                    upload_short(str(video_file), yt_title, yt_desc)
        except Exception as error:
            print(f"⚠️ Видео создано, но загрузка на YouTube потребовала авторизации/действия: {error}")
    else:
        print("⏭️ [4/5] Загрузка на YouTube отключена (UPLOAD_TO_YOUTUBE=1 для включения).")

    print(f"🏁 [5/5] Все видео созданы: {[v.name for v in generated_videos]}")
    return generated_videos


if __name__ == "__main__":
    for number in range(1, int(os.getenv("SHORTS_TO_GENERATE", "1")) + 1):
        build_single_short(number)
