import json
#!/usr/bin/env python3
"""
Make It Easy AI — Autonomous Long-Form + Shorts Waterfall Factory ("ЧТО, ЕСЛИ?" Style)
=======================================================================================
1. AI Trend Hunter: Scans YouTube & generates a viral "Что, если...?" / "Как устроено...?" topic.
2. Multi-Chapter Showrunner: Writes a deep, step-by-step multi-chapter script (RU + EN).
3. "ЧТО, ЕСЛИ?" 2D Vector Motion Engine:
   - Signature pitch-black circular heads (#0E0E13) with warm golden-cream rim light (#F5D8A0)
   - Expressive concentric cream-white eyes (#F4EFE3), eyebrows, cyan sweat drops, speech bubble pills
   - 3 Dynamic Camera Shots: Wide Establishing -> Medium Kinetic Text -> Extreme Bokeh Close-Up
   - Volumetric god-rays, glowing candles/lamps, floating dust specks, 2.35:1 cinematic letterbox
4. Memory-Safe Hardware Renderer: Streams frames chapter-by-chapter via FFmpeg pipe (<80MB RAM).
5. Shorts Waterfall Sniper: Automatically cuts 9:16 vertical Shorts from the episode linking to the Full Video!
"""

import argparse
import functools
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

LONGFORM_DIR = BASE_DIR.parent / "output" / "longform"
SHORTS_WATERFALL_DIR = BASE_DIR.parent / "output" / "waterfall_shorts"
TEMP_CHUNKS_DIR = BASE_DIR.parent / "output" / "temp_longform"
HISTORY_FILE = BASE_DIR / "longform_history.json"

for d in (LONGFORM_DIR, SHORTS_WATERFALL_DIR, TEMP_CHUNKS_DIR):
    d.mkdir(parents=True, exist_ok=True)

FFMPEG_BIN = "/opt/homebrew/bin/ffmpeg"
FFPROBE_BIN = "/opt/homebrew/bin/ffprobe"
if not os.path.exists(FFMPEG_BIN):
    FFMPEG_BIN = "ffmpeg"
if not os.path.exists(FFPROBE_BIN):
    FFPROBE_BIN = "ffprobe"

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY обязателен в scripts/.env")

gemini_client = genai.Client(
    api_key=GEMINI_KEY,
    http_options=types.HttpOptions(
        api_version="v1beta",
        timeout=60000,
        retry_options=types.HttpRetryOptions(attempts=2),
    ),
)

# Signature "ЧТО, ЕСЛИ?" Color Palette
HEAD_BLACK = (14, 14, 18)
RIM_GOLD = (245, 216, 160)
CREAM_WHITE = (244, 239, 227)
KINETIC_GOLD = (225, 198, 135)
SWEAT_CYAN = (130, 216, 230)


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


def load_history() -> list[str]:
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            return list(data.get("used_titles", []))
        except Exception:
            pass
    return []


def save_history(title: str) -> None:
    used = load_history()
    if title not in used:
        used.append(title)
    HISTORY_FILE.write_text(
        json.dumps({"used_titles": used[-60:]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================================
# 1. AI TREND HUNTER & MULTI-CHAPTER SHOWRUNNER
# ============================================================================
def scan_whatif_trends() -> str:
    """Scans YouTube for top-performing 'Что если' / 'What if' / Edutainment videos."""
    queries = [
        "что если бы ты мог анимация",
        "что будет если научпоп анимация",
        "what if science animation",
        "как работает изнутри на пальцах",
    ]
    q = random.choice(queries)
    try:
        yt_dlp_bin = str(Path(sys.executable).parent / "yt-dlp")
        if not os.path.exists(yt_dlp_bin):
            yt_dlp_bin = "yt-dlp"
        res = subprocess.run(
            [yt_dlp_bin, "--flat-playlist", "--dump-json", f"ytsearch8:{q}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        titles = []
        for line in res.stdout.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("title"):
                titles.append(f"- {item['title']} ({item.get('view_count', 'N/A')} views)")
        if titles:
            return "\n".join(titles[:6])
    except Exception:
        pass
    return (
        "- Что, если бы ты мог ПРЕВРАЩАТЬСЯ В КОГО УГОДНО?\n"
        "- Что, если бы ты мог ОСТАНАВЛИВАТЬ ВРЕМЯ, но с реальной физикой?\n"
        "- Что происходит с телом, если увидеть ядерный гриб (правило большого пальца)?\n"
        "- Что, если не спать 7 дней подряд: пошаговая хроника мозга?"
    )


def generate_longform_script(
    num_chapters: int = 5,
    scenes_per_chapter: int = 6,
    custom_topic: str = "",
) -> dict[str, Any]:
    """Generates a complete multi-chapter 'ЧТО, ЕСЛИ?' / 'Make It Easy AI' episode."""
    used_titles = load_history()
    trends = scan_whatif_trends()
    topic_block = (
        f"Сделай полный разбор на тему: «{custom_topic}»."
        if custom_topic
        else (
            "САМОСТОЯТЕЛЬНО выбери ОДНУ максимально залипательную, вирусную тему в стиле канала «ЧТО, ЕСЛИ?» и «Make It Easy AI»!\n"
            "Примеры форматов, которые собирают миллионы просмотров:\n"
            "  • «Что, если бы ты мог ОСТАНАВЛИВАТЬ ВРЕМЯ? (Почему реальная физика сделает тебя слепым, а воздух — твёрже бетона)»\n"
            "  • «Что, если бы ты мог ЧИТАТЬ МЫСЛИ каждого человека в радиусе 100 метров?»\n"
            "  • «Что, если температура твоего тела поднимется до 45 градусов или упадёт до 25?»\n"
            "  • «Что, если ты увидел ядерный гриб на горизонте: правило большого пальца и хроника первых 15 минут»\n"
            "  • «Что, если бы ты помнил КАЖДУЮ секунду своей жизни (гипертимезия) или перестал чувствовать боль?»\n"
            "  • «Что происходит внутри компьютера и сервера за 20 миллисекунд, когда ты нажимаешь выстрел в онлайн-игре?»\n"
            f"Ориентиры трендов YouTube:\n{trends}"
        )
    )

    prompt = f"""
Ты — главный шоураннер и сценарист культового анимационного канала «Make It Easy AI» в визуальном и повествовательном стиле канала «ЧТО, ЕСЛИ?» (What If / Meet Arnold).
{topic_block}
Не повторяй прошлые выпуски: {used_titles[-15:]}.

ПРАВИЛА СЦЕНАРИЯ БОЛЬШОГО ВЫПУСКА (ДО 15 МИНУТ МАКСИМУМ):
1. 🔥 ПЕРВЫЕ 5 СЕКУНД — АБСОЛЮТНЫЙ РАЗЪЁБ И ЗАХВАТ ВНИМАНИЯ (COLD OPEN HOOK):
   - КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО начинать с медленных приветствий («Привет», «Сегодня мы разберём...»).
   - Первая сцена первой главы (`chapters[0].scenes[0]`) — это ВЗРЫВНОЙ КЛИМАКС на 4–5 секунд (строго 10–14 слов!): мы сразу бросаем зрителя в эпицентр шока, парадокса или смертельной опасности (например: «Не смей двигаться! Если время остановилось — первый же вдох разорвёт твои лёгкие!»).
   - Для первой сцены обязательно укажи `hook_banner` (короткий ударный текст КАПСОМ до 32 символов, например `"⚡ НЕ ДЫШИ: ПЕРВЫЕ 5 СЕКУНД!"`).
2. Главного героя зовут Артём (или Тёма). Мы объясняем тему через его живую, пошаговую историю, совмещая юмор, бытовые ситуации, резкие повороты сюжета и точные научные факты (физику, биологию, анатомию, нейробиологию, устройство технологий).
3. Выпуск состоит из РОВНО {num_chapters} глав (`chapters`), и в КАЖДОЙ главе РОВНО {scenes_per_chapter} сцен (`scenes`).
4. Текст первой сцены (`chapters[0].scenes[0].text`) — 10–14 слов (ровно на 4–5 секунд взрывного хука!). Текст всех остальных сцен — 18–26 слов живого, увлекательного рассказа диктора.
5. Для каждой сцены укажи:
   - `shot_type`: один из `"wide"` (общий план локации с персонажами), `"medium"` (средний план с плавающей золотой надписью в кадре) или `"closeup"` (крупный портрет лица героя с размытым фоном боке, огромными глазами и облачком реплики). Первая сцена — всегда `"closeup"` или яркий `"wide"` с шоком!
   - `setting`: один из `"curtain_hall"`, `"room"`, `"lab"`, `"street"`, `"space"`, `"forest"`, `"bunker"`.
   - `prop`: один из `"portrait_coffin"`, `"xray_brain"`, `"xray_bone"`, `"xray_blood"`, `"nuke_mushroom"`, `"thumb_rule"`, `"diagram_flow"`, `"math_board"`, `"chart_bars"`, `"clock_paradox"`, `"none"`.
   - `emotion`: `"shock"`, `"curious"`, `"thinking"`, `"pain"`, `"happy"`, `"sleep"`.
   - `speech_bubble`: короткая живая реплика (1–2 слова, например `"Тёма?"`, `"ЧТО?!"`, `"СТОП!"`, `"КАК ТАК?"`, `"ОЙ..."`, `""`).
   - `kinetic_text`: короткий тезис/факт КАПСОМ (до 34 символов, например `"МОЧКА УХА · ТВОЯ ПРИВЫЧКА"` или `"ВОЗДУХ ТВЕРДЕЕТ КАК БЕТОН"`).
   - `hook_banner`: (для первой сцены каждой главы) взрывной заголовок первых 5 секунд КАПСОМ.

Верни СТРОГО валидный JSON:
{{
  "title": "Цепляющее название выпуска в стиле «Что, если бы ты...»",
  "subtitle": "Краткое описание интриги",
  "hero_name": "Артём",
  "teaser_chapter_index": 0,
  "chapters": [
    {{
      "chapter_number": 1,
      "chapter_title": "Глава 1. Случайное пробуждение",
      "short_hook": "ЧТО ЕСЛИ ОСТАНОВИТЬ ВРЕМЯ НА САМОМ ДЕЛЕ?!",
      "scenes": [
        {{
          "text": "Представь, что однажды утром Артём просыпается, щёлкает пальцами — и весь мир вокруг замирает. Но уже через три секунды законы физики превращают эту суперспособность в ловушку.",
          "shot_type": "wide",
          "setting": "room",
          "prop": "clock_paradox",
          "emotion": "shock",
          "speech_bubble": "СТОП!",
          "kinetic_text": "ЗАКОНЫ ФИЗИКИ ПРОТИВ ТЕБЯ"
        }}
      ]
    }}
  ]
}}
"""
    # 🌟 Priority 1: Claude Director (Opus/Sonnet) for viral storytelling
    try:
        import claude_director
        c_res = claude_director.ask_claude_director(prompt)
        if c_res:
            import re
            jm = re.search(r'\{.*\}', c_res, re.DOTALL)
            if jm:
                c_data = json.loads(jm.group(0))
                if 'chapters' in c_data and len(c_data['chapters']) >= 2:
                    print(f"🎭 [Claude Director] Сценарий от Claude Pro успешно внедрён: «{c_data.get('title')}» ({len(c_data['chapters'])} глав)")
                    save_history(c_data.get('title', ''))
                    return c_data
    except Exception as ce:
        print(f"⚠️ [Claude Director fallback]: {ce}")

    candidate_models = [
        os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
    ]
    last_err = None
    for attempt in range(4):
        model_name = candidate_models[min(attempt, len(candidate_models) - 1)]
        try:
            res = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.85,
                ),
            )
            data = json.loads(res.text)
            if isinstance(data, dict) and isinstance(data.get("chapters"), list) and len(data["chapters"]) > 0:
                print(f"✅ Сценарий сгенерирован ({model_name}): «{data.get('title')}» ({len(data['chapters'])} глав)")
                return data
        except Exception as e:
            last_err = e
            print(f"⚠️ Попытка {attempt + 1}/4 ({model_name}): {e}")
            time.sleep(3 * (attempt + 1))
            time.sleep(2)
    raise RuntimeError(f"Не удалось сгенерировать сценарий: {last_err}")


def translate_longform_to_english(script_ru: dict[str, Any]) -> dict[str, Any]:
    """Translates the entire multi-chapter script into English."""
    print("🌐 Переводим сценарий выпуска на английский язык (EN)...")
    prompt = f"""
Translate the following animated science/storytelling script JSON from Russian to natural, engaging English (in the style of 'What If' / 'Meet Arnold' YouTube channels).
Keep `shot_type`, `setting`, `prop`, `emotion` unchanged.
Translate `title`, `subtitle`, `chapter_title`, `short_hook`, `text`, `speech_bubble` (1-2 words max, e.g. 'Wait?!', 'Bro?', 'What?!'), and `kinetic_text` (ALL CAPS, max 34 chars).
Rename hero 'Артём'/'Тёма' to 'Arthur'/'Artie'.
Return strictly valid JSON:
{json.dumps(script_ru, ensure_ascii=False)}
"""
    for model_name in ("gemini-3.1-flash-lite", "gemini-3.8-flash"):
        try:
            res = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            data = json.loads(res.text)
            if isinstance(data, dict) and data.get("chapters"):
                return data
        except Exception as e:
            print(f"⚠️ Ошибка перевода ({model_name}): {e}")
    return script_ru


# ============================================================================
# 2. "ЧТО, ЕСЛИ?" SIGNATURE 2D VECTOR MOTION ENGINE
# ============================================================================
def _draw_whatif_character(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    scale: float = 1.0,
    shirt_color: tuple[int, int, int] = (42, 92, 68),
    emotion: str = "shock",
    hood: bool = False,
    speech_bubble: str = "",
    closeup_mode: bool = False,
    phase: float = 0.0,
) -> None:
    """Draws the iconic 'ЧТО, ЕСЛИ?' character:
    pitch-black circular head, golden-cream rim light crescent, curly hair / hood,
    concentric cream eyes, raised eyebrows, cyan sweat drop, and speech bubble pill."""
    bob_y = int(math.sin(phase * math.pi * 2) * (6 * scale))
    cy = cy + bob_y

    body_w = int(66 * scale)
    body_h = int(205 * scale)
    leg_h = int(165 * scale)

    if not closeup_mode:
        # Legs with golden-cream rim light
        for lx1, lx2 in (
            (cx - body_w + int(12 * scale), cx - int(6 * scale)),
            (cx + int(6 * scale), cx + body_w - int(12 * scale)),
        ):
            draw.rounded_rectangle(
                (lx1 + int(4 * scale), cy + body_h - int(15 * scale), lx2 + int(5 * scale), cy + body_h + leg_h),
                radius=int(12 * scale),
                fill=RIM_GOLD,
            )
            draw.rounded_rectangle(
                (lx1, cy + body_h - int(15 * scale), lx2, cy + body_h + leg_h),
                radius=int(12 * scale),
                fill=(26, 30, 38),
            )
            # Shoe with cream rim
            draw.rounded_rectangle(
                (lx1 - int(4 * scale), cy + body_h + leg_h - int(20 * scale), lx2 + int(16 * scale), cy + body_h + leg_h + int(6 * scale)),
                radius=int(10 * scale),
                fill=RIM_GOLD,
            )
            draw.rounded_rectangle(
                (lx1 - int(4 * scale), cy + body_h + leg_h - int(16 * scale), lx2 + int(14 * scale), cy + body_h + leg_h + int(6 * scale)),
                radius=int(10 * scale),
                fill=(36, 40, 48),
            )

    # Torso / shoulders
    torso_h = int(130 * scale) if closeup_mode else body_h
    draw.rounded_rectangle(
        (cx - body_w + int(5 * scale), cy + int(3 * scale), cx + body_w + int(6 * scale), cy + torso_h + int(3 * scale)),
        radius=int(28 * scale),
        fill=RIM_GOLD,
    )
    draw.rounded_rectangle(
        (cx - body_w, cy, cx + body_w, cy + torso_h),
        radius=int(28 * scale),
        fill=shirt_color,
    )
    # Collar / lapel line
    draw.line(
        (cx, cy + int(10 * scale), cx, cy + int(85 * scale)),
        fill=(max(0, shirt_color[0] - 18), max(0, shirt_color[1] - 18), max(0, shirt_color[2] - 18)),
        width=max(2, int(4 * scale)),
    )

    if not closeup_mode:
        # Side silhouette arm
        draw.rounded_rectangle(
            (cx + body_w - int(22 * scale), cy + int(18 * scale), cx + body_w + int(8 * scale), cy + int(130 * scale)),
            radius=int(12 * scale),
            fill=HEAD_BLACK,
        )

    radius = int(76 * scale)
    head_y = cy - radius + int(6 * scale)

    # Concentric shock/thought radar rings around head (signature shot_05.png effect)
    if closeup_mode or emotion in ("shock", "pain", "thinking"):
        for r_idx in (1, 2, 3):
            pulse = int(math.sin(phase * math.pi * 2 + r_idx) * (4 * scale))
            rr = radius + int((22 + r_idx * 22) * scale) + pulse
            draw.ellipse((cx - rr, head_y - rr, cx + rr, head_y + rr), outline=(105, 98, 92), width=1)

    # Hair / Hood
    if hood:
        hr = radius + int(16 * scale)
        draw.ellipse((cx - hr, head_y - hr, cx + hr, head_y + hr), fill=(214, 88, 84))
        for dot_ang in (-50, -25, 0, 25, 50):
            rad = math.radians(dot_ang - 90)
            dx_dot = cx + int(math.cos(rad) * (radius + int(4 * scale)))
            dy_dot = head_y + int(math.sin(rad) * (radius + int(4 * scale)))
            dr = max(3, int(5 * scale))
            draw.ellipse((dx_dot - dr, dy_dot - dr, dx_dot + dr, dy_dot + dr), fill=CREAM_WHITE)
    else:
        for puff_ang in range(190, 355, 26):
            rad = math.radians(puff_ang)
            px = cx + int(math.cos(rad) * (radius - int(6 * scale)))
            py = head_y + int(math.sin(rad) * (radius - int(6 * scale)))
            pr = int(27 * scale)
            draw.ellipse((px - pr, py - pr - int(3 * scale), px + pr, py + pr), fill=RIM_GOLD)
            draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=HEAD_BLACK)

    # Pitch-black circular head with warm golden-cream rim light crescent
    draw.ellipse(
        (cx - radius + int(5 * scale), head_y - radius + int(4 * scale), cx + radius + int(7 * scale), head_y + radius + int(7 * scale)),
        fill=RIM_GOLD,
    )
    draw.ellipse((cx - radius, head_y - radius, cx + radius, head_y + radius), fill=HEAD_BLACK)

    # Eyes (large concentric cream-white circles + black iris ring + cream pupil dot)
    eye_outer = int(22 * scale) if emotion in ("shock", "curious", "pain") else int(16 * scale)
    eye_inner = max(4, int(eye_outer * 0.56))
    pupil_r = max(2, int(eye_outer * 0.28))
    look_x = int(math.sin(phase * math.pi) * (3 * scale))

    for idx_e, eye_x in enumerate((cx - int(26 * scale), cx + int(26 * scale))):
        ey = head_y - int(4 * scale)
        if emotion == "sleep":
            draw.line((eye_x - int(12 * scale), ey, eye_x + int(12 * scale), ey), fill=CREAM_WHITE, width=max(3, int(5 * scale)))
        else:
            draw.ellipse((eye_x - eye_outer, ey - eye_outer, eye_x + eye_outer, ey + eye_outer), fill=CREAM_WHITE)
            draw.ellipse((eye_x + look_x - eye_inner, ey - eye_inner, eye_x + look_x + eye_inner, ey + eye_inner), fill=HEAD_BLACK)
            draw.ellipse((eye_x + look_x - pupil_r, ey - pupil_r, eye_x + look_x + pupil_r, ey + pupil_r), fill=CREAM_WHITE)

        # Eyebrows
        brow_y = ey - eye_outer - int(8 * scale)
        tilt = int(6 * scale) if (idx_e == 0 and emotion in ("shock", "pain")) else int(-6 * scale if emotion in ("shock", "pain") else 0)
        draw.line(
            (eye_x - int(15 * scale), brow_y + tilt, eye_x + int(15 * scale), brow_y - tilt),
            fill=CREAM_WHITE,
            width=max(3, int(6 * scale)),
        )

    # Cyan sweat drop on temple
    if emotion in ("shock", "pain", "thinking"):
        sx_d, sy_d = cx + int(48 * scale), head_y - int(26 * scale)
        draw.ellipse((sx_d - int(5 * scale), sy_d - int(9 * scale), sx_d + int(5 * scale), sy_d + int(9 * scale)), fill=SWEAT_CYAN)

    # Mouth
    if emotion == "shock":
        mr = int(13 * scale)
        draw.ellipse((cx - mr, head_y + int(26 * scale) - mr, cx + mr, head_y + int(26 * scale) + mr), fill=CREAM_WHITE)
    elif emotion == "pain":
        pts_m = [
            (cx - int(22 * scale), head_y + int(26 * scale)),
            (cx - int(8 * scale), head_y + int(35 * scale)),
            (cx + int(8 * scale), head_y + int(20 * scale)),
            (cx + int(22 * scale), head_y + int(29 * scale)),
        ]
        draw.line(pts_m, fill=CREAM_WHITE, width=max(3, int(6 * scale)))
    elif emotion == "happy":
        draw.arc((cx - int(18 * scale), head_y + int(14 * scale), cx + int(18 * scale), head_y + int(38 * scale)), 0, 180, fill=CREAM_WHITE, width=max(3, int(5 * scale)))
    else:
        draw.line((cx - int(15 * scale), head_y + int(28 * scale), cx + int(15 * scale), head_y + int(28 * scale)), fill=CREAM_WHITE, width=max(3, int(5 * scale)))

    # Speech bubble pill above head (like 'Тёма?' in shot_05.png)
    if speech_bubble:
        font_sz = max(22, int(24 * min(scale, 1.5)))
        by_pill = max(84, head_y - radius - int(38 * scale))
        tw = max(95, int(draw.textlength(speech_bubble[:18], font=get_font(font_sz))) + 40)
        ph = max(22, int(22 * min(scale, 1.4)))
        draw.rounded_rectangle((cx - tw // 2, by_pill - ph, cx + tw // 2, by_pill + ph), radius=ph, fill=CREAM_WHITE)
        draw.polygon([(cx - 10, by_pill + ph - 2), (cx + 8, by_pill + ph - 2), (cx - 16, by_pill + ph + 16)], fill=CREAM_WHITE)
        draw.text((cx, by_pill), speech_bubble[:18], font=get_font(font_sz), fill=(18, 18, 22), anchor="mm")


def _draw_environment_and_props(
    draw: ImageDraw.ImageDraw,
    w: int,
    h: int,
    setting: str,
    prop: str,
    phase: float,
) -> None:
    """Draws the background environment, volumetric light rays, candles/lamps, and scientific/story props."""
    floor_y = int(h * 0.78)

    if setting == "curtain_hall":
        # Rich crimson curtain hall (like shot_04.png)
        draw.rectangle((0, 0, w, floor_y), fill=(72, 26, 30))
        stripe_w = max(36, w // 18)
        for sx in range(0, w, stripe_w):
            col = (84, 32, 36) if (sx // stripe_w) % 2 == 0 else (62, 22, 26)
            draw.rectangle((sx, 0, sx + stripe_w // 2, floor_y), fill=col)
        draw.rectangle((0, floor_y, w, h), fill=(36, 26, 22))
    elif setting == "lab":
        draw.rectangle((0, 0, w, floor_y), fill=(22, 34, 46))
        for gx in range(0, w, 80):
            draw.line((gx, 0, gx, floor_y), fill=(28, 44, 58), width=2)
        for gy in range(0, floor_y, 80):
            draw.line((0, gy, w, gy), fill=(28, 44, 58), width=2)
        draw.rectangle((0, floor_y, w, h), fill=(16, 24, 32))
    elif setting == "space":
        draw.rectangle((0, 0, w, h), fill=(10, 12, 22))
        for s_i in range(28):
            sx = (s_i * 197) % w
            sy = (s_i * 131) % int(h * 0.85)
            sr = 2 + (s_i % 3)
            draw.ellipse((sx - sr, sy - sr, sx + sr, sy + sr), fill=CREAM_WHITE)
    elif setting == "street":
        draw.rectangle((0, 0, w, floor_y), fill=(20, 26, 38))
        for b_i, bx in enumerate(range(40, w - 80, 160)):
            bh = 180 + (b_i * 55) % 220
            draw.rectangle((bx, floor_y - bh, bx + 125, floor_y), fill=(28, 36, 50), outline=(45, 56, 74), width=2)
        draw.rectangle((0, floor_y, w, h), fill=(24, 26, 32))
    elif setting == "forest":
        draw.rectangle((0, 0, w, floor_y), fill=(18, 32, 28))
        for t_i, tx in enumerate(range(60, w, 150)):
            top_y = int(h * 0.22) + (t_i % 3) * 35
            draw.polygon([(tx, top_y), (tx - 95, floor_y), (tx + 95, floor_y)], fill=(14, 24, 20))
        draw.rectangle((0, floor_y, w, h), fill=(12, 18, 15))
    else:
        # Warm dark interior room
        draw.rectangle((0, 0, w, floor_y), fill=(48, 36, 34))
        for sx in range(0, w, 110):
            draw.line((sx, 0, sx, floor_y), fill=(42, 30, 28), width=3)
        draw.rectangle((0, floor_y, w, h), fill=(30, 22, 20))

    # Floor wood planks / horizon line
    draw.line((0, floor_y, w, floor_y), fill=(92, 68, 52), width=4)

    # Volumetric light rays from top center
    cx_light = w // 2
    for ray_offset in (-90, 0, 90):
        draw.polygon(
            [
                (cx_light + ray_offset - 35, 0),
                (cx_light + ray_offset + 35, 0),
                (cx_light + ray_offset * 2 + 140, floor_y),
                (cx_light + ray_offset * 2 - 140, floor_y),
            ],
            fill=(96, 52, 48) if setting == "curtain_hall" else (38, 54, 68),
        )

    # Warm glowing candles on stands (signature atmospheric lighting from shot_04.png)
    for candle_x in (int(w * 0.28), int(w * 0.72)):
        glow_r = 55 + int(math.sin(phase * math.pi * 4) * 6)
        draw.ellipse((candle_x - glow_r, floor_y - 165 - glow_r, candle_x + glow_r, floor_y - 165 + glow_r), fill=(135, 78, 38))
        draw.rectangle((candle_x - 24, floor_y - 95, candle_x + 24, floor_y), fill=(32, 22, 20))
        draw.rectangle((candle_x - 6, floor_y - 155, candle_x + 6, floor_y - 95), fill=CREAM_WHITE)
        draw.ellipse((candle_x - 6, floor_y - 172, candle_x + 6, floor_y - 154), fill=(255, 205, 65))

    # Centerpiece Prop
    px, py = int(w * 0.54), int(h * 0.44)
    if prop == "nuke_mushroom":
        draw.ellipse((px - 180, py - 40, px + 180, py + 190), fill=(235, 110, 30))
        draw.polygon([(px - 55, py + 180), (px + 55, py + 180), (px + 28, py - 50), (px - 28, py - 50)], fill=(85, 42, 32))
        draw.ellipse((px - 175, py - 145, px + 175, py + 10), fill=(65, 32, 28), outline=RIM_GOLD, width=4)
        draw.ellipse((px - 110, py - 115, px + 110, py - 15), fill=(245, 155, 50))
    elif prop == "thumb_rule":
        draw.rounded_rectangle((px - 45, py - 110, px + 45, py + 45), radius=24, fill=RIM_GOLD, outline=HEAD_BLACK, width=4)
        draw.line((px - 140, py - 110, px + 140, py - 110), fill=(245, 75, 65), width=4)
        draw.line((px - 140, py + 45, px + 140, py + 45), fill=(245, 75, 65), width=4)
    elif prop in ("xray_brain", "xray_bone", "xray_blood"):
        r_xr = min(155, int(min(w, h) * 0.2))
        draw.ellipse((px - r_xr, py - r_xr, px + r_xr, py + r_xr), fill=(14, 24, 36), outline=(85, 215, 235), width=6)
        if prop == "xray_brain":
            draw.ellipse((px - int(r_xr * 0.65), py - int(r_xr * 0.5), px + int(r_xr * 0.65), py + int(r_xr * 0.5)), fill=(225, 145, 165), outline=CREAM_WHITE, width=4)
            for n_i in range(5):
                ang = phase * 2 + n_i * 1.25
                nx, ny = px + int(math.cos(ang) * (r_xr * 0.38)), py + int(math.sin(ang) * (r_xr * 0.28))
                draw.ellipse((nx - 8, ny - 8, nx + 8, ny + 8), fill=(255, 225, 75))
                draw.line((px, py, nx, ny), fill=(255, 225, 75), width=3)
        elif prop == "xray_blood":
            draw.line((px - r_xr + 20, py, px + r_xr - 20, py), fill=(195, 35, 45), width=36)
            for c_i in range(4):
                cx_c = px - r_xr + 45 + ((c_i * 55 + int(phase * 70)) % max(40, r_xr * 2 - 90))
                draw.ellipse((cx_c - 16, py - 12, cx_c + 16, py + 12), fill=(245, 65, 75), outline=CREAM_WHITE, width=2)
        else:
            draw.line((px - int(r_xr * 0.65), py - int(r_xr * 0.45), px - 8, py - 5), fill=CREAM_WHITE, width=28)
            draw.line((px + 8, py + 5, px + int(r_xr * 0.65), py + int(r_xr * 0.45)), fill=CREAM_WHITE, width=28)
            draw.line((px - 22, py + 22, px + 22, py - 22), fill=(255, 75, 65), width=6)
    elif prop == "diagram_flow":
        draw.rounded_rectangle((px - 210, py - 70, px - 30, py + 70), radius=16, fill=(24, 38, 58), outline=(75, 195, 255), width=4)
        draw.rounded_rectangle((px + 30, py - 70, px + 210, py + 70), radius=16, fill=(24, 48, 38), outline=(85, 235, 145), width=4)
        draw.line((px - 30, py, px + 30, py), fill=RIM_GOLD, width=5)
        draw.text((px - 120, py), "CLIENT", font=get_font(22), fill=CREAM_WHITE, anchor="mm")
        draw.text((px + 120, py), "SERVER", font=get_font(22), fill=CREAM_WHITE, anchor="mm")
    elif prop == "math_board":
        draw.rounded_rectangle((px - 210, py - 110, px + 210, py + 110), radius=18, fill=(20, 44, 36), outline=(185, 145, 75), width=6)
        draw.text((px, py - 25), "E = mc²  |  Δt → 0", font=get_font(30), fill=RIM_GOLD, anchor="mm")
    elif prop == "chart_bars":
        for idx_b, bh_val in enumerate((65, 105, 85, 145, 190)):
            bx = px - 150 + idx_b * 65
            draw.rounded_rectangle((bx, py + 90 - bh_val, bx + 42, py + 90), radius=8, fill=(65, 195, 115) if idx_b == 4 else (58, 128, 205), outline=CREAM_WHITE, width=2)
    else:
        # Framed portrait on the wall (exactly like shot_04.png!)
        fw, fh = 150, 185
        fy = int(h * 0.12)
        draw.rounded_rectangle((px - fw // 2, fy, px + fw // 2, fy + fh), radius=8, fill=(32, 42, 54), outline=(205, 165, 75), width=8)
        pr = 44
        draw.ellipse((px - pr + 3, fy + 75 - pr + 3, px + pr + 3, fy + 75 + pr + 3), fill=RIM_GOLD)
        draw.ellipse((px - pr, fy + 75 - pr, px + pr, fy + 75 + pr), fill=HEAD_BLACK)
        draw.ellipse((px - 16, fy + 68, px - 6, fy + 78), fill=CREAM_WHITE)
        draw.ellipse((px + 6, fy + 68, px + 16, fy + 78), fill=CREAM_WHITE)
        draw.arc((px - 14, fy + 78, px + 14, fy + 96), 0, 180, fill=CREAM_WHITE, width=4)

    # Floating atmospheric dust particles (signature 'ЧТО, ЕСЛИ?' look)
    for d_i in range(18):
        dx_p = int((d_i * 173 + phase * 35 * ((d_i % 3) + 1)) % w)
        dy_p = int((d_i * 97 + math.sin(phase * 2 + d_i) * 18) % floor_y)
        r_p = 2 if d_i % 2 == 0 else 3
        draw.ellipse((dx_p - r_p, dy_p - r_p, dx_p + r_p, dy_p + r_p), fill=(155, 135, 125))


def render_whatif_frame(
    scene: dict[str, Any],
    chapter_title: str,
    phase: float,
    word_idx: int,
    words: list[str],
    w: int = 1280,
    h: int = 720,
    is_vertical: bool = False,
    cta_banner: str = "",
    blurred_bg_cache: dict[str, Image.Image] | None = None,
    is_intro_hook: bool = False,
    intro_elapsed_sec: float = 0.0,
) -> Image.Image:
    """Renders a single frame in the 'ЧТО, ЕСЛИ?' flat-vector + bokeh + rim-light style,
    with an explosive 5-second visual retention hook when is_intro_hook=True."""
    shot_type = str(scene.get("shot_type", "wide")).lower()
    setting = str(scene.get("setting", "room")).lower()
    prop = str(scene.get("prop", "none")).lower()
    emotion = str(scene.get("emotion", "shock")).lower()
    speech_bubble = str(scene.get("speech_bubble", "")).strip()
    kinetic_text = str(scene.get("kinetic_text", "")).strip().upper()
    hook_banner = str(scene.get("hook_banner") or kinetic_text or chapter_title).strip().upper()

    # Force high-energy expression & zoom boost during the first 5 seconds!
    hook_zoom = 1.0
    shake_x, shake_y = 0, 0
    if is_intro_hook and intro_elapsed_sec <= 5.0:
        emotion = "shock"
        hook_zoom = 1.0 + 0.14 * min(1.0, intro_elapsed_sec / 2.2)
        if intro_elapsed_sec < 0.9:
            amp = int((1.0 - intro_elapsed_sec) * 14)
            shake_x = int(math.sin(intro_elapsed_sec * 55) * amp)
            shake_y = int(math.cos(intro_elapsed_sec * 45) * amp)

    if shot_type == "closeup":
        # Bokeh blurred background (cached per setting+prop for instant speed!)
        cache_key = f"{w}x{h}_{setting}_{prop}"
        if blurred_bg_cache is not None and cache_key in blurred_bg_cache:
            img = blurred_bg_cache[cache_key].copy()
        else:
            bg_img = Image.new("RGB", (w, h), (24, 18, 20))
            bg_draw = ImageDraw.Draw(bg_img)
            _draw_environment_and_props(bg_draw, w, h, setting, prop, 0.25)
            bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=12))
            if blurred_bg_cache is not None:
                blurred_bg_cache[cache_key] = bg_img
            img = bg_img.copy()

        draw = ImageDraw.Draw(img)
        # Giant Close-Up Character in Foreground (like frame_01.png & shot_05.png)
        char_scale = (2.45 if not is_vertical else 2.85) * hook_zoom
        char_y = (int(h * 0.76) if not is_vertical else int(h * 0.66)) + shake_y
        _draw_whatif_character(
            draw,
            cx=(w // 2) + shake_x,
            cy=char_y,
            scale=char_scale,
            shirt_color=(42, 92, 68),
            emotion=emotion,
            hood=(setting == "curtain_hall"),
            speech_bubble=speech_bubble or "ЧТО?!",
            closeup_mode=True,
            phase=phase,
        )
    else:
        img = Image.new("RGB", (w, h), (24, 18, 20))
        draw = ImageDraw.Draw(img)
        _draw_environment_and_props(draw, w, h, setting, prop, phase)

        floor_y = int(h * 0.78)
        if shot_type == "medium":
            # Medium Tracking Shot + Floating Gold Kinetic Typography (like shot_03.png)
            char_x = (int(w * 0.34) if not is_vertical else int(w * 0.38)) + shake_x
            char_scale = (1.25 if not is_vertical else 1.50) * hook_zoom
            _draw_whatif_character(
                draw,
                cx=char_x,
                cy=floor_y - int(155 * char_scale) + shake_y,
                scale=char_scale,
                shirt_color=(110, 78, 55),
                emotion=emotion,
                hood=True,
                speech_bubble=speech_bubble,
                closeup_mode=False,
                phase=phase,
            )
            if kinetic_text:
                kx = int(w * 0.66) if not is_vertical else w // 2
                ky = int(h * 0.44) if not is_vertical else int(h * 0.26)
                font_k = get_font(32 if not is_vertical else 38)
                draw.text((kx + 2, ky + 2), kinetic_text[:32], font=font_k, fill=HEAD_BLACK, anchor="mm")
                draw.text((kx, ky), kinetic_text[:32], font=font_k, fill=KINETIC_GOLD, anchor="mm")
        else:
            # Wide Establishing Shot with 2 characters + dashed sightline (like shot_04.png)
            c1_x = (int(w * 0.20) if not is_vertical else int(w * 0.24)) + shake_x
            c2_x = (int(w * 0.82) if not is_vertical else int(w * 0.78)) - shake_x
            c_scale = (0.92 if not is_vertical else 1.15) * hook_zoom
            _draw_whatif_character(
                draw,
                cx=c1_x,
                cy=floor_y - int(225 * c_scale) + shake_y,
                scale=c_scale,
                shirt_color=(42, 92, 68),
                emotion=emotion,
                hood=False,
                speech_bubble=speech_bubble,
                closeup_mode=False,
                phase=phase,
            )
            _draw_whatif_character(
                draw,
                cx=c2_x,
                cy=floor_y - int(225 * c_scale) + shake_y,
                scale=c_scale,
                shirt_color=(110, 78, 55),
                emotion="thinking",
                hood=True,
                speech_bubble="",
                closeup_mode=False,
                phase=phase + 0.4,
            )
            # Dashed sightline between character and centerpiece
            for step_i in range(0, 14, 2):
                t1, t2 = step_i / 14.0, (step_i + 1) / 14.0
                x1 = int(c2_x * (1 - t1) + (w * 0.54) * t1)
                y1 = int((floor_y - 220) * (1 - t1) + (h * 0.30) * t1)
                x2 = int(c2_x * (1 - t2) + (w * 0.54) * t2)
                y2 = int((floor_y - 220) * (1 - t2) + (h * 0.30) * t2)
                draw.line((x1, y1, x2, y2), fill=(155, 130, 120), width=2)

    # ========================================================================
    # 🔥 FIRST 5 SECONDS EXPLOSIVE RETENTION HOOK OVERLAY
    # ========================================================================
    if is_intro_hook and intro_elapsed_sec <= 5.0:
        # Expanding crimson/gold shockwave ring
        ring_r = int(((intro_elapsed_sec * 260) % 420) + 90)
        draw.ellipse(
            (w // 2 - ring_r, h // 2 - ring_r, w // 2 + ring_r, h // 2 + ring_r),
            outline=(245, 75, 55) if int(intro_elapsed_sec * 4) % 2 == 0 else RIM_GOLD,
            width=4,
        )
        # High-contrast Hook Banner + Shrinking 5-Second Retention Bar
        hb_y = 85 if not is_vertical else 185
        hb_w = min(w - 80, 780 if not is_vertical else 940)
        x1_hb, x2_hb = w // 2 - hb_w // 2, w // 2 + hb_w // 2
        draw.rounded_rectangle(
            (x1_hb, hb_y - 34, x2_hb, hb_y + 42),
            radius=18,
            fill=(195, 32, 32),
            outline=RIM_GOLD,
            width=4,
        )
        draw.text(
            (w // 2, hb_y - 2),
            f"⚡ {hook_banner[:32]}",
            font=get_font(30 if not is_vertical else 36),
            fill=CREAM_WHITE,
            anchor="mm",
        )
        # Shrinking 5-second progress bar inside bottom of banner
        rem_ratio = max(0.0, min(1.0, (5.0 - intro_elapsed_sec) / 5.0))
        bar_fill_w = int((hb_w - 28) * rem_ratio)
        if bar_fill_w > 4:
            draw.rounded_rectangle(
                (x1_hb + 14, hb_y + 24, x1_hb + 14 + bar_fill_w, hb_y + 34),
                radius=5,
                fill=RIM_GOLD,
            )

    # Cinematic Letterbox Bars & Clean Subtitle Pill
    if not is_vertical:
        bar_h = int(h * 0.065)
        draw.rectangle((0, 0, w, bar_h), fill=(8, 8, 10))
        draw.rectangle((0, h - bar_h, w, h), fill=(8, 8, 10))
        # Subtle Chapter Badge in top-left letterbox
        draw.text((28, bar_h // 2), f"MAKE IT EASY AI  ·  {chapter_title.upper()[:42]}", font=get_font(18), fill=(185, 175, 160), anchor="lm")
    else:
        # Top Hook Banner for 9:16 Waterfall Short
        draw.rounded_rectangle((50, 70, w - 50, 155), radius=20, fill=(16, 18, 24), outline=RIM_GOLD, width=4)
        draw.text((w // 2, 112), chapter_title.upper()[:34], font=get_font(34), fill=CREAM_WHITE, anchor="mm")

    # Active 5-word Subtitle Window
    if words and 0 <= word_idx < len(words):
        chunk_size = 4 if is_vertical else 6
        start_w = (word_idx // chunk_size) * chunk_size
        chunk = words[start_w : start_w + chunk_size]
        active_pos = word_idx - start_w
        sub_font = get_font(44 if is_vertical else 30)
        sub_y = int(h * 0.82) if is_vertical else int(h * 0.88)

        full_line = " ".join(chunk)
        tw_line = int(draw.textlength(full_line, font=sub_font))
        pad_x = 28
        draw.rounded_rectangle(
            (w // 2 - tw_line // 2 - pad_x, sub_y - 32, w // 2 + tw_line // 2 + pad_x, sub_y + 32),
            radius=16,
            fill=(12, 12, 16),
            outline=(75, 70, 64),
            width=2,
        )
        cursor_x = w // 2 - tw_line // 2
        for i_w, w_str in enumerate(chunk):
            w_col = RIM_GOLD if i_w == active_pos else CREAM_WHITE
            draw.text((cursor_x, sub_y), w_str, font=sub_font, fill=w_col, anchor="lm")
            cursor_x += int(draw.textlength(w_str + " ", font=sub_font))

    # Waterfall CTA Banner on 9:16 Shorts ("▶ ПОЛНЫЙ ВЫПУСК УЖЕ НА КАНАЛЕ!")
    if is_vertical and cta_banner:
        draw.rounded_rectangle((65, h - 195, w - 65, h - 105), radius=24, fill=(215, 48, 45), outline=CREAM_WHITE, width=4)
        draw.text((w // 2, h - 150), cta_banner, font=get_font(34), fill=CREAM_WHITE, anchor="mm")

    return img


# ============================================================================
# 3. VOICEOVER & PROCEDURAL SOUNDTRACK
# ============================================================================
def synthesize_scene_voice(text: str, out_mp3: Path, lang: str = "ru", engine: str = "edge") -> float:
    """Generates neural voiceover via Edge-TTS (0 ElevenLabs tokens!) or ElevenLabs, returning audio duration in seconds."""
    if engine == "elevenlabs" and os.getenv("ELEVENLABS_API_KEY"):
        try:
            from elevenlabs import VoiceSettings
            from elevenlabs.client import ElevenLabs

            el_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
            stream = el_client.text_to_speech.convert(
                text=text,
                voice_id=os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"),
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
                voice_settings=VoiceSettings(stability=0.82, similarity_boost=0.88),
            )
            with out_mp3.open("wb") as f:
                for chunk in stream:
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"⚠️ ElevenLabs fallback to Edge-TTS: {e}")
            engine = "edge"

    if engine != "elevenlabs" or not out_mp3.exists():
        voice_name = "en-US-ChristopherNeural" if lang.lower() == "en" else "ru-RU-DmitryNeural"
        edge_bin = str(Path(sys.executable).parent / "edge-tts")
        if not os.path.exists(edge_bin):
            edge_bin = "edge-tts"
        edge_ok = False
        for _ in range(3):
            res_edge = subprocess.run(
                [edge_bin, "--voice", voice_name, "--text", text, "--write-media", str(out_mp3)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if res_edge.returncode == 0 and out_mp3.exists() and out_mp3.stat().st_size > 100:
                edge_ok = True
                break
            time.sleep(0.5)
            if res_edge.returncode == 0 and out_mp3.exists() and out_mp3.stat().st_size > 100:
                edge_ok = True
                break
            time.sleep(0.5)
        if not edge_ok:
            from gtts import gTTS
            gTTS(text=text, lang="en" if lang.lower() == "en" else "ru").save(str(out_mp3))

    # Probe exact audio duration via ffprobe
    try:
        res = subprocess.run(
            [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(out_mp3)],
            capture_output=True,
            text=True,
            check=True,
        )
        return max(2.5, float(res.stdout.strip()))
    except Exception:
        return max(4.0, len(text.split()) * 0.38)


def generate_ambient_bed_wav(duration_sec: float, out_wav: Path, include_intro_sfx: bool = False) -> Path:
    """Generates a warm, curious 'ЧТО, ЕСЛИ?' / Kurzgesagt style ambient synth pad +
    explosive 5-second audio hook (sub-bass impact drop + tension riser + heartbeat) when include_intro_sfx=True."""
    sr = 22050
    total_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, total_samples, dtype=np.float32)
    # Warm curiosity chord (D minor / D9 soft pad: 146.8Hz, 220Hz, 261.6Hz, 329.6Hz)
    pad = (
        0.35 * np.sin(2 * np.pi * 146.83 * t)
        + 0.25 * np.sin(2 * np.pi * 220.00 * t)
        + 0.20 * np.sin(2 * np.pi * 261.63 * t)
        + 0.12 * np.sin(2 * np.pi * 329.63 * t)
    )
    lfo = 0.65 + 0.35 * np.sin(2 * np.pi * 0.18 * t)
    audio = (pad * lfo * 0.08).astype(np.float32)

    if include_intro_sfx and duration_sec > 2.0:
        hook_len = min(total_samples, int(5.0 * sr))
        # 0. Sound Notch (Claude Audio Dip): Muffle ambient between 1.8s - 2.3s right before punchline
        dip_mask = (t >= 1.8) & (t <= 2.3)
        audio[dip_mask] *= 0.15
        th = t[:hook_len]
        # 1. Sub-bass impact drop (0.0s - 0.8s: 115Hz -> 38Hz)
        drop_env = np.exp(-th * 4.2)
        drop_phase = 2 * np.pi * (38.0 * th + 18.0 * (1.0 - np.exp(-th * 5.0)))
        sub_drop = 0.32 * np.sin(drop_phase) * drop_env
        # 2. Rising sci-fi tension whoosh (0.2s - 2.2s)
        riser_mask = ((th >= 0.15) & (th <= 2.2)).astype(np.float32)
        riser_env = np.sin(np.clip((th - 0.15) / 2.05, 0.0, 1.0) * np.pi) * riser_mask
        riser_freq = 160.0 + 380.0 * ((th - 0.15) / 2.05) ** 2
        riser = 0.11 * np.sin(2 * np.pi * riser_freq * th) * riser_env
        # 3. Urgent heartbeat pulses at 1.0s, 2.0s, 3.0s, 4.0s
        hb = np.zeros_like(th)
        for beat_t in (0.9, 1.12, 1.9, 2.12, 2.9, 3.12, 3.9, 4.12):
            dt_b = th - beat_t
            mask_b = (dt_b >= 0.0) & (dt_b <= 0.18)
            hb[mask_b] += 0.24 * np.sin(2 * np.pi * 56.0 * dt_b[mask_b]) * np.exp(-dt_b[mask_b] * 18.0)
        audio[:hook_len] += (sub_drop + riser + hb).astype(np.float32)

    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return out_wav


# ============================================================================
# 4. MEMORY-SAFE HARDWARE CHAPTER RENDERER (<80MB RAM)
# ============================================================================
def render_chapter_video(
    chapter: dict[str, Any],
    chapter_idx: int,
    out_mp4: Path,
    lang: str = "ru",
    voice_engine: str = "edge",
    w: int = 1280,
    h: int = 720,
    fps: int = 24,
    is_vertical: bool = False,
    cta_banner: str = "",
) -> tuple[Path, float]:
    """Renders a single chapter directly via FFmpeg pipe + hardware encoder so RAM stays flat."""
    scenes = chapter.get("scenes", [])
    chapter_title = str(chapter.get("chapter_title", f"Глава {chapter_idx + 1}"))
    blurred_bg_cache: dict[str, Image.Image] = {}

    scene_audios: list[Path] = []
    scene_durations: list[float] = []

    for s_idx, sc in enumerate(scenes):
        mp3_path = TEMP_CHUNKS_DIR / f"ch{chapter_idx}_sc{s_idx}_{lang}.mp3"
        dur = synthesize_scene_voice(str(sc.get("text", "")), mp3_path, lang=lang, engine=voice_engine)
        scene_audios.append(mp3_path)
        scene_durations.append(dur)

    # Concatenate chapter voiceover MP3s + mix with ambient pad
    concat_audio_list = TEMP_CHUNKS_DIR / f"audio_list_ch{chapter_idx}_{lang}.txt"
    concat_audio_list.write_text("\n".join(f"file '{p.resolve()}'" for p in scene_audios), encoding="utf-8")
    voice_Track = TEMP_CHUNKS_DIR / f"voice_ch{chapter_idx}_{lang}.m4a"
    subprocess.run(
        [
            FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_audio_list),
            "-af", "aresample=async=1:first_pts=0", "-c:a", "aac", "-b:a", "160k", str(voice_Track),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    total_dur = sum(scene_durations)
    ambient_wav = TEMP_CHUNKS_DIR / f"ambient_ch{chapter_idx}_{lang}.wav"
    generate_ambient_bed_wav(total_dur, ambient_wav, include_intro_sfx=(chapter_idx == 0 or is_vertical))
    mixed_audio = TEMP_CHUNKS_DIR / f"mixed_ch{chapter_idx}_{lang}.m4a"
    subprocess.run(
        [
            FFMPEG_BIN, "-y", "-i", str(voice_Track), "-i", str(ambient_wav),
            "-filter_complex", "[0:a]volume=1.15[v];[1:a]volume=0.55[a];[v][a]amix=inputs=2:duration=first:dropout_transition=1[out]",
            "-map", "[out]", "-c:a", "aac", "-b:a", "160k", str(mixed_audio),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Stream video frames directly into FFmpeg h264_videotoolbox pipe
    video_only_mp4 = TEMP_CHUNKS_DIR / f"vid_ch{chapter_idx}_{lang}.mp4"
    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        "-c:v", "h264_videotoolbox", "-b:v", "2800k", "-pix_fmt", "yuv420p",
        str(video_only_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None

    elapsed_chapter_sec = 0.0
    for s_idx, (sc, dur) in enumerate(zip(scenes, scene_durations)):
        words = str(sc.get("text", "")).split() or [""]
        n_frames = max(1, int(round(dur * fps)))
        # Render at 12 unique animation steps per second (doubled to 24fps) for 2x speed!
        last_bytes = b""
        for f_i in range(n_frames):
            if f_i % 2 == 0 or not last_bytes:
                t_norm = f_i / max(1, n_frames)
                cur_sec = elapsed_chapter_sec + (f_i / fps)
                w_idx = min(int(t_norm * len(words)), len(words) - 1)
                frame_img = render_whatif_frame(
                    sc,
                    chapter_title=chapter_title,
                    phase=t_norm * 2.0,
                    word_idx=w_idx,
                    words=words,
                    w=w,
                    h=h,
                    is_vertical=is_vertical,
                    cta_banner=cta_banner,
                    blurred_bg_cache=blurred_bg_cache,
                    is_intro_hook=((chapter_idx == 0 or is_vertical) and cur_sec <= 5.0),
                    intro_elapsed_sec=cur_sec,
                )
                last_bytes = frame_img.tobytes()
            proc.stdin.write(last_bytes)
        elapsed_chapter_sec += dur

    proc.stdin.close()
    proc.wait()

    # Mux video + mixed audio cleanly
    subprocess.run(
        [
            FFMPEG_BIN, "-y",
            "-i", str(video_only_mp4), "-i", str(mixed_audio),
            "-c:v", "copy", "-c:a", "copy", "-shortest",
            str(out_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    blurred_bg_cache.clear()
    gc.collect()
    return out_mp4, total_dur


# ============================================================================
# 5. FULL EPISODE + SHORTS WATERFALL ORCHESTRATOR
# ============================================================================
def format_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def build_longform_and_waterfall_shorts(
    num_chapters: int = 5,
    scenes_per_chapter: int = 6,
    custom_topic: str = "",
    voice_engine: str = "edge",
    enable_en: bool = True,
    upload: bool = True,
) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n🎬 [1/5] ИИ-Шоураннер «ЧТО, ЕСЛИ? / Make It Easy AI» создаёт большой выпуск ({num_chapters} глав по {scenes_per_chapter} сцен)...")
    script_ru = generate_longform_script(num_chapters=num_chapters, scenes_per_chapter=scenes_per_chapter, custom_topic=custom_topic)
    save_history(str(script_ru.get("title", "")))

    results: dict[str, Any] = {"ru": {}, "en": {}}

    for lang_code, script_data in [("ru", script_ru)] + ([("en", translate_longform_to_english(script_ru))] if enable_en else []):
        is_en = lang_code == "en"
        tag = "EN" if is_en else "RU"
        chapters = script_data.get("chapters", [])
        chapter_files: list[Path] = []
        chapter_timestamps: list[str] = []
        cursor_sec = 0.0

        MAX_LONGFORM_DURATION_SEC = 840.0  # Жёсткий потолок 14:00 (гарантия прохождения лимита YouTube в 15 минут)
        print(f"\n🎞️ [2/5] [{tag}] Поглавный рендер большого выпуска (16:9 Widescreen в стиле «ЧТО, ЕСЛИ?», лимит < 15 мин)...")
        for c_idx, ch in enumerate(chapters):
            if cursor_sec >= MAX_LONGFORM_DURATION_SEC - 60.0 and len(chapter_files) >= 2:
                print(f"   ⏹️ Достигнут безопасный порог длительности ({format_timestamp(cursor_sec)} < 15:00), завершаем сборку глав.")
                break
            ch_title = str(ch.get("chapter_title", f"Chapter {c_idx + 1}"))
            chapter_timestamps.append(f"{format_timestamp(cursor_sec)} {ch_title}")
            ch_mp4 = TEMP_CHUNKS_DIR / f"chapter_{timestamp}_{c_idx + 1}_{tag}.mp4"
            print(f"   ⏳ [{tag}] Рендер {c_idx + 1}/{len(chapters)}: {ch_title}...")
            _, ch_dur = render_chapter_video(
                ch, c_idx, ch_mp4, lang=lang_code, voice_engine=voice_engine, w=1280, h=720, is_vertical=False
            )
            chapter_files.append(ch_mp4)
            cursor_sec += ch_dur

        # Concatenate all chapters into the Full 16:9 Episode
        full_list = TEMP_CHUNKS_DIR / f"full_concat_{timestamp}_{tag}.txt"
        full_list.write_text("\n".join(f"file '{p.resolve()}'" for p in chapter_files), encoding="utf-8")
        full_video_path = LONGFORM_DIR / f"whatif_full_{timestamp}_{tag}.mp4"
        subprocess.run(
            [FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", str(full_list), "-c", "copy", str(full_video_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"✅ [{tag}] Большой выпуск готов ({format_timestamp(cursor_sec)}): {full_video_path}")

        # Build Vertical 9:16 Waterfall Short from the Teaser Chapter
        teaser_idx = min(int(script_data.get("teaser_chapter_index", 0)), len(chapters) - 1)
        teaser_ch = chapters[teaser_idx]
        short_hook = str(teaser_ch.get("short_hook") or script_data.get("title", "ЧТО, ЕСЛИ?!")).upper()
        cta_text = "▶ FULL VIDEO ON THE CHANNEL!" if is_en else "▶ ПОЛНЫЙ ВЫПУСК УЖЕ НА КАНАЛЕ!"
        short_video_path = SHORTS_WATERFALL_DIR / f"whatif_short_{timestamp}_{tag}.mp4"

        print(f"📱 [3/5] [{tag}] Shorts Sniper: создаём вертикальный 9:16 Шортс-тизер из Главы {teaser_idx + 1}...")
        shorts_voice = os.getenv("SHORTS_VOICE_ENGINE", "elevenlabs" if os.getenv("ELEVENLABS_API_KEY") else "edge")
        render_chapter_video(
            {**teaser_ch, "chapter_title": short_hook},
            teaser_idx,
            short_video_path,
            lang=lang_code,
            voice_engine=shorts_voice,
            w=1080,
            h=1920,
            is_vertical=True,
            cta_banner=cta_text,
        )
        print(f"✅ [{tag}] Вертикальный Шортс-тизер готов: {short_video_path}")

        full_yt_id = None
        short_yt_id = None
        if upload:
            try:
                from uploader import upload_short

                full_title = str(script_data.get("title", "Что, если?"))[:95]
                chapters_block = "\n".join(chapter_timestamps)
                full_desc = (
                    f"{script_data.get('subtitle', '')}\n\n"
                    f"{'TIMESTAMPS / CHAPTERS:' if is_en else 'ТАЙМКОДЫ ВЫПУСКА:'}\n{chapters_block}\n\n"
                    f"#makeiteasyai #whatif #чтоесли #научпоп #анимация"
                )
                print(f"📤 [4/5] [{tag}] Загружаем БОЛЬШОЙ ВЫПУСК на YouTube: {full_title}...")
                full_yt_id = upload_short(str(full_video_path), full_title, full_desc)

                full_url = f"https://youtu.be/{full_yt_id}" if full_yt_id else ""
                short_title = f"{short_hook[:75]} 😱 #shorts"
                short_desc = (
                    f"{'🔥 Watch the FULL episode here:' if is_en else '🔥 Смотри ПОЛНЫЙ выпуск на канале:'} {full_url}\n\n"
                    f"{script_data.get('title', '')}\n#shorts #whatif #чтоесли #makeiteasyai"
                )
                print(f"📤 [5/5] [{tag}] Загружаем ШОРТС-ТИЗЕР с привязкой к полному видео ({full_url})...")
                short_yt_id = upload_short(str(short_video_path), short_title, short_desc)
            except Exception as e:
                print(f"⚠️ Ошибка загрузки на YouTube ({tag}): {e}")

        results[lang_code] = {
            "full_video": str(full_video_path),
            "short_video": str(short_video_path),
            "full_youtube_url": f"https://youtu.be/{full_yt_id}" if full_yt_id else None,
            "short_youtube_url": f"https://youtu.be/{short_yt_id}" if short_yt_id else None,
            "duration_sec": round(cursor_sec, 1),
            "chapters": chapter_timestamps,
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make It Easy AI — Long-Form + Shorts Waterfall Factory")
    parser.add_argument("--chapters", type=int, default=int(os.getenv("LONGFORM_CHAPTERS", "5")), help="Количество глав (например, 5-12)")
    parser.add_argument("--scenes-per-chapter", type=int, default=int(os.getenv("SCENES_PER_CHAPTER", "6")), help="Сцен в каждой главе")
    parser.add_argument("--topic", type=str, default=os.getenv("CUSTOM_TOPIC", ""), help="Опциональная тема (если пусто — ИИ придумывает сам)")
    parser.add_argument("--voice", choices=["edge", "elevenlabs"], default=os.getenv("LONGFORM_VOICE", "edge"), help="edge (бесплатно 0 токенов) или elevenlabs")
    parser.add_argument("--lang", choices=["both", "ru", "en"], default="both", help="Языки релиза")
    parser.add_argument("--no-upload", action="store_true", help="Не загружать на YouTube")
    args = parser.parse_args()

    build_longform_and_waterfall_shorts(
        num_chapters=args.chapters,
        scenes_per_chapter=args.scenes_per_chapter,
        custom_topic=args.topic,
        voice_engine=args.voice,
        enable_en=(args.lang in ("both", "en")),
        upload=not args.no_upload,
    )
