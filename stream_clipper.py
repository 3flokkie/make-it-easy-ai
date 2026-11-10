#!/usr/bin/env python3
"""
Stream Clipper to Shorts Pipeline (4-Step Automated Engine)
1. Download Stream / VOD via yt-dlp (Twitch / YouTube)
2. Speech & Highlight Analysis (faster-whisper + Gemini highlight finder)
3. Smart Vertical 9:16 Crop (Streamer face/webcam top + gameplay bottom, or blurred backdrop)
4. Dynamic Karaoke Subtitles & 2-Second Hook Banner (MoviePy / FFmpeg)
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

OUTPUT_DIR = BASE_DIR.parent / "output" / "stream_shorts"
TEMP_DIR = BASE_DIR.parent / "output" / "temp_streams"
for d in (OUTPUT_DIR, TEMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

FFMPEG_BIN = "/opt/homebrew/bin/ffmpeg"
if not os.path.exists(FFMPEG_BIN):
    try:
        import imageio_ffmpeg
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG_BIN = "ffmpeg"


def get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ============================================================================
# STEP 1: DOWNLOAD STREAM OR CLIP VIA YT-DLP
# ============================================================================
def download_stream(url: str, max_duration_sec: int = 180) -> Path:
    """Downloads stream/VOD snippet using yt-dlp."""
    print(f"📥 [1/4] Скачивание видео/стрима по ссылке: {url}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_template = str(TEMP_DIR / f"raw_stream_{timestamp}.%(ext)s")

    ffmpeg_dir = str(Path(FFMPEG_BIN).parent)
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--ffmpeg-location",
        ffmpeg_dir,
        "-f",
        "bestvideo[vcodec^=avc1][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best[ext=mp4]",
        "--merge-output-format",
        "mp4",
        "--no-playlist",
        "--download-sections",
        f"*0-{max_duration_sec}",
        "-o",
        out_template,
        url,
    ]

    subprocess.run(cmd, check=True)
    # Prefer .mp4 merged file
    mp4_files = list(TEMP_DIR.glob(f"raw_stream_{timestamp}.mp4"))
    if mp4_files:
        target_file = mp4_files[0]
    else:
        downloaded_files = list(TEMP_DIR.glob(f"raw_stream_{timestamp}.*"))
        if not downloaded_files:
            raise RuntimeError(f"Не удалось найти скачанный файл для {url}")
        target_file = downloaded_files[0]
    print(f"✅ Видео скачано: {target_file} ({target_file.stat().st_size / (1024*1024):.1f} MB)")
    return target_file


# ============================================================================
# STEP 2: AUDIO TRANSCRIPTION & HIGHLIGHT ANALYSIS (FASTER-WHISPER + GEMINI)
# ============================================================================
def transcribe_audio_with_whisper(video_path: Path) -> list[dict[str, Any]]:
    """Runs faster-whisper to extract words with precise start/end timestamps."""
    print("🎙️ [2/4] Транскрибация речи и таймкодов через faster-whisper...")
    from faster_whisper import WhisperModel

    # Use CPU/NEON optimized 'base' model for speed on MacBook
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(video_path),
        word_timestamps=True,
        beam_size=1,
        condition_on_previous_text=False,
        no_speech_threshold=0.9,
        log_prob_threshold=-2.0,
    )

    words_timeline: list[dict[str, Any]] = []
    for segment in segments:
        for word in segment.words:
            if word.word.strip():
                words_timeline.append({
                    "word": word.word.strip(),
                    "start": round(word.start, 2),
                    "end": round(word.end, 2),
                    "probability": round(word.probability, 2),
                })
    print(f"✅ Найдено {len(words_timeline)} слов в аудиодорожке.")
    return words_timeline


def find_best_highlight(
    words_timeline: list[dict[str, Any]],
    target_duration: int = 50,
    lang: str = "ru",
) -> tuple[float, float, str]:
    """Uses Gemini to find the most viral, energetic, emotional moment (including action lead-ins)."""
    is_en = (lang.lower() == "en")
    default_hook = "CRAZIEST STREAM MOMENT!" if is_en else "САМЫЙ ДИКИЙ МОМЕНТ СТРИМА!"
    if not words_timeline:
        return 0.0, float(target_duration), default_hook

    max_end_time = words_timeline[-1]["end"]
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        client = genai.Client(
            api_key=gemini_key,
            http_options=types.HttpOptions(
                api_version="v1beta",
                timeout=30000,
                retry_options=types.HttpRetryOptions(attempts=2),
            ),
        )
        full_text = " ".join(f"[{w['start']}s]{w['word']}" for w in words_timeline[:600])
        lang_instruction = (
            "Write the `hook` in viral ALL-CAPS ENGLISH (3-5 words, e.g. 'HE ACTUALLY DID THIS?!' or 'WAIT FOR HIS REACTION!')."
            if is_en
            else "Придумай мощный вирусный хук-заголовок на РУССКОМ для первых секунд (3-5 слов, КАПСОМ, например 'ОН РЕАЛЬНО ЭТО СКАЗАЛ?!')."
        )

        prompt = f"""
Ты — топовый рилсмейкер и клиппер стримов.
Ниже расшифровка речи стримера с таймкодами:
{full_text}

Найди самый сочный, смешной, эмоциональный или безумный отрезок длительностью 35-55 секунд.
{lang_instruction}

Верни ТОЛЬКО валидный JSON:
{{
  "start": 12.5,
  "end": 57.0,
  "hook": "{default_hook}"
}}
"""
        for candidate in [os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"), "gemini-3.1-flash-lite", "gemini-3.8-flash"]:
            try:
                res = client.models.generate_content(
                    model=candidate,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                data = json.loads(res.text)
                start = max(0.0, float(data.get("start", 0.0)))
                end = max(start + 15.0, float(data.get("end", start + target_duration)))
                hook = str(data.get("hook", default_hook)).upper()[:36]
                words_in_window = sum(1 for w in words_timeline if start <= w["start"] <= end)
                if start + 10.0 <= max_end_time and words_in_window >= 15:
                    print(
                        f"🎯 ИИ выбрал лучший хайлайт ({lang.upper()} / {candidate}): "
                        f"{start:.1f}s - {end:.1f}s ({words_in_window} слов) | Хук: «{hook}»"
                    )
                    return start, min(end, max_end_time), hook
                default_hook = hook
                break
            except Exception as err:
                print(f"⚠️ Ошибка Gemini {candidate}: {err}")

    # Fallback: find the 45-second window with the highest speech density (preserving action lead-ins)
    best_start = 0.0
    best_count = -1
    step = 5.0
    probe = 0.0
    while probe + 15.0 <= max_end_time:
        count = sum(1 for w in words_timeline if probe <= w["start"] <= probe + target_duration)
        if count > best_count:
            best_count = count
            best_start = probe
        probe += step
    end = min(best_start + float(target_duration), max_end_time)
    print(f"🎯 Выбран самый насыщенный участок: {best_start:.1f}s - {end:.1f}s ({best_count} слов)")
    return best_start, end, default_hook


# ============================================================================
# STEP 3: SMART VERTICAL 9:16 CROP & FACE/WEBCAM DETECTION
# ============================================================================
def detect_webcam_box(video_path: Path, sample_time: float = 5.0) -> tuple[int, int, int, int] | None:
    """Detects streamer's face / webcam corner across multiple frames using OpenCV Haar Cascade."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        return None
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(str(video_path))
    for offset in (1.5, 4.0, 8.0, 14.0, 22.0):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, sample_time + offset) * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            pad_x = int(fw * 0.85)
            pad_y = int(fh * 0.95)
            bx = max(0, fx - pad_x)
            by = max(0, fy - pad_y)
            bw = min(w - bx, fw + pad_x * 2)
            bh = min(h - by, fh + pad_y * 2)
            cap.release()
            print(f"👤 Найдено лицо стримера (t={sample_time + offset:.1f}s): x={bx}, y={by}, w={bw}, h={bh}")
            return bx, by, bw, bh
    cap.release()
    return None


def render_split_vertical_short(
    input_video: Path,
    output_video: Path,
    start_time: float,
    duration: float,
    hook_text: str,
    words: list[dict[str, Any]],
) -> Path:
    """Builds vertical 1080x1920 Short using FFmpeg layout:
    - Top: Streamer webcam (or blurred background top)
    - Bottom: Gameplay screen (16:9 centered)
    - Hook banner in first 3 seconds
    """
    print(f"🎬 [3/4] Сборка вертикального ролика 9:16 (1080x1920) через FFmpeg...")

    webcam = detect_webcam_box(input_video, sample_time=start_time)

    # Filter graph construction (note: -ss and -t are already applied on input, so [0:v] starts at 0)
    if webcam:
        wx, wy, ww, wh = webcam
        # Layout: Top 40% (1080x768) = Face/Webcam | Bottom 60% (1080x1152) = Gameplay
        filter_complex = (
            f"[0:v]fps=30,split=2[cam_in][game_in];"
            f"[cam_in]crop={ww}:{wh}:{wx}:{wy},scale=1080:768:force_original_aspect_ratio=increase,crop=1080:768[cam];"
            f"[game_in]scale=1080:1152:force_original_aspect_ratio=decrease,pad=1080:1152:(ow-iw)/2:(oh-ih)/2:black[game];"
            f"[cam][game]vstack=inputs=2[stacked];"
            f"[stacked]drawbox=y=766:color=yellow@0.8:width=iw:height=4:t=fill[v]"
        )
    else:
        # Blurred backdrop layout (classic YouTube Shorts)
        filter_complex = (
            f"[0:v]fps=30,split=2[bg_in][fg_in];"
            f"[bg_in]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5[bg];"
            f"[fg_in]scale=1080:1080:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
        )

    temp_split_mp4 = TEMP_DIR / f"split_{output_video.name}"

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss",
        str(start_time),
        "-t",
        str(duration),
        "-i",
        str(input_video),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-af",
        "aresample=async=1:first_pts=0",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "26",
        "-c:a",
        "aac",
        "-ar",
        "44100",
        "-b:a",
        "160k",
        str(temp_split_mp4),
    ]

    subprocess.run(cmd, check=True)
    print(f"✅ Базовый вертикальный сплит готов: {temp_split_mp4}")

    # ========================================================================
    # STEP 4: OVERLAY HOOK & DYNAMIC KARAOKE SUBTITLES
    # ========================================================================
    print("✨ [4/4] Накладываем караоке-субтитры и хук...")
    apply_subtitles_and_hook(temp_split_mp4, output_video, hook_text, words, start_time, duration)

    if temp_split_mp4.exists():
        temp_split_mp4.unlink()

    return output_video


def apply_subtitles_and_hook(
    input_video: Path,
    output_video: Path,
    hook_text: str,
    words: list[dict[str, Any]],
    offset_start: float,
    duration: float,
) -> None:
    """Overlays dynamic karaoke subtitles and 2-second hook banner using MoviePy,
    then muxes clean FFmpeg audio directly without MoviePy audio resampling glitches."""
    from moviepy import VideoFileClip

    clip = VideoFileClip(str(input_video), audio=False)
    font_hook = get_font(52)
    font_sub = get_font(56)

    # Filter words inside highlight window
    active_words = [
        {
            "word": w["word"],
            "start": max(0.0, w["start"] - offset_start),
            "end": max(0.0, w["end"] - offset_start),
        }
        for w in words
        if offset_start - 0.5 <= w["start"] <= offset_start + duration
    ]

    def make_frame(get_frame, t: float) -> np.ndarray:
        frame_img = Image.fromarray(get_frame(t))
        draw = ImageDraw.Draw(frame_img, "RGBA")

        # 1. Hook banner (visible during first 2.8 seconds)
        if t <= 2.8:
            banner_y = 180
            tw = int(draw.textlength(hook_text, font=font_hook)) + 60
            draw.rounded_rectangle(
                (540 - tw // 2, banner_y - 45, 540 + tw // 2, banner_y + 45),
                radius=18,
                fill=(235, 45, 45, 230),
                outline=(255, 255, 255, 255),
                width=4,
            )
            draw.text((540, banner_y), hook_text, font=font_hook, fill="white", anchor="mm")

        # 2. Dynamic karaoke subtitles (centered at Y = 1380)
        current_word_idx = -1
        # Pass 1: exact word currently being spoken
        for idx, w in enumerate(active_words):
            if w["start"] <= t <= max(w["end"], w["start"] + 0.22):
                current_word_idx = idx
        # Pass 2: if in a pause between words, hold the most recently spoken word for up to 2.5s
        if current_word_idx == -1:
            for idx, w in enumerate(active_words):
                if w["end"] < t <= (w["end"] + 2.5):
                    current_word_idx = idx
        # Pass 3: if t is in the very first 0.5s right before word 0 starts, already show word 0
        if current_word_idx == -1 and active_words and t < active_words[0]["start"] <= 1.5:
            current_word_idx = 0

        if current_word_idx != -1:
            # Group into stable 3-word CapCut/Hormozi-style phrases so words don't jump horizontally
            start_i = (current_word_idx // 3) * 3
            end_i = min(len(active_words), start_i + 3)
            chunk = active_words[start_i:end_i]

            widths = [int(draw.textlength(w["word"].upper(), font=font_sub)) for w in chunk]
            spacing = 18
            total_w = sum(widths) + spacing * (len(chunk) - 1)
            # If 3 long words exceed screen width, show 2 words around current_word_idx
            if total_w > 960 and len(chunk) > 2:
                start_i = (current_word_idx // 2) * 2
                end_i = min(len(active_words), start_i + 2)
                chunk = active_words[start_i:end_i]
                widths = [int(draw.textlength(w["word"].upper(), font=font_sub)) for w in chunk]
                total_w = sum(widths) + spacing * (len(chunk) - 1)

            cur_x = 540 - total_w // 2
            sub_y = 1380

            # Semi-transparent dark backdrop pill for 100% contrast on any video frame
            pad_x, pad_y = 28, 42
            draw.rounded_rectangle(
                (540 - total_w // 2 - pad_x, sub_y - pad_y, 540 + total_w // 2 + pad_x, sub_y + pad_y),
                radius=16,
                fill=(10, 12, 18, 175),
                outline=(255, 215, 0, 90),
                width=2,
            )

            for idx, (w, w_len) in enumerate(zip(chunk, widths)):
                is_active = (start_i + idx) == current_word_idx
                color = (255, 215, 0) if is_active else (255, 255, 255)
                # Text stroke for legibility
                draw.text(
                    (cur_x + w_len // 2, sub_y),
                    w["word"].upper(),
                    font=font_sub,
                    fill=color,
                    anchor="mm",
                    stroke_width=6,
                    stroke_fill=(10, 12, 16),
                )
                cur_x += w_len + spacing

        return np.asarray(frame_img)

    temp_video_only = TEMP_DIR / f"video_only_{output_video.name}"
    subbed_clip = clip.transform(make_frame)
    try:
        subbed_clip.write_videofile(
            str(temp_video_only),
            fps=30,
            codec="libx264",
            audio=False,
            preset="ultrafast",
            ffmpeg_params=["-crf", "26"],
            threads=4,
            logger="bar",
        )
    finally:
        clip.close()
        subbed_clip.close()

    # Mux clean FFmpeg audio stream directly with subtitled video stream (zero MoviePy audio glitches!)
    mux_cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(temp_video_only),
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(output_video),
    ]
    subprocess.run(mux_cmd, check=True)
    if temp_video_only.exists():
        temp_video_only.unlink()
    print(f"🎉 Готовый Shorts-ролик сохранён (чистый звук + караоке с 0 сек): {output_video}")


# ============================================================================
# AUTONOMOUS STREAM HUNTER (ZERO TWITCH API KEYS REQUIRED)
# ============================================================================
STREAM_HISTORY_FILE = BASE_DIR / "stream_history.json"

STREAMERS_ROSTER = {
    "ru": [
        "Зубарев лучшая реакция стрим момент",
        "Зубарев смотрит смешной ролик реакция",
        "Папич Arthas смешной момент рейдж стрим",
        "Братишкин рофл реакция стрим нарезка",
        "Братишкин смотрит угарный момент стрим",
        "Бустер смешной момент стрим реакция",
        "Бустер и Плохой Парень Хазяева смешной момент",
        "Nix дота эмоции стрим момент",
        "Deepins реакция стрим момент",
        "Куплинов смешной момент испуг реакция",
        "T2x2 смешной момент история со стрима",
        "Каша Kussia смешной момент рейдж стрим",
        "Мазеллов Mzlff рофл реакция стрим",
        "SilverName рейдж смешной момент стрим",
        "Sasavot смешной момент нарезка со стрима",
    ],
    "en": [
        "IShowSpeed funniest rage moment stream highlight",
        "IShowSpeed funniest reaction clip viral",
        "Kai Cenat crazy stream moment highlight",
        "Kai Cenat funniest reaction clip",
        "xQc funny reaction stream highlight",
        "CaseOh funniest rage moment stream clip",
        "CaseOhchat roasts funniest moment highlight",
        "Jynxzi funniest rage reaction clip",
        "Tyler1 funniest rage moment stream highlight",
        "Plaqueboymax funny stream reaction highlight",
        "MoistCr1TiKaL penguinz0 funny stream moment",
        "Lacy and Marlon funny stream moment clip",
    ],
}


def load_used_stream_urls() -> set[str]:
    if STREAM_HISTORY_FILE.exists():
        try:
            data = json.loads(STREAM_HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return set(str(x) for x in data)
        except Exception:
            pass
    return set()


def save_used_stream_url(url: str) -> None:
    used = list(load_used_stream_urls())
    if url not in used:
        used.append(url)
    STREAM_HISTORY_FILE.write_text(json.dumps(used[-200:], ensure_ascii=False, indent=2), encoding="utf-8")


def auto_discover_viral_stream_url(lang: str = "ru") -> str:
    """Automatically finds a fresh, high-view streamer highlight video on YouTube/Twitch
    without needing any Twitch API keys or manual links."""
    import random
    import yt_dlp

    used_urls = load_used_stream_urls()
    queries = list(STREAMERS_ROSTER.get(lang.lower(), STREAMERS_ROSTER["ru"]))
    random.shuffle(queries)

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }

    for query in queries:
        print(f"🔎 [Auto-Hunter] Ищем вирусные моменты стримеров по запросу: «{query}»...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch20:{query}", download=False)
                entries = info.get("entries", []) if isinstance(info, dict) else []
                candidates = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    url = entry.get("url") or (f"https://www.youtube.com/watch?v={entry['id']}" if entry.get("id") else "")
                    duration = entry.get("duration") or 120
                    views = entry.get("view_count") or 0
                    # Pick clips/highlights between 25 seconds and 15 minutes that haven't been used yet
                    if url and url not in used_urls and 25 <= float(duration) <= 900:
                        candidates.append((int(views), url, entry.get("title", "")))
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    best_views, best_url, best_title = candidates[0]
                    print(f"🔥 Выбран вирусный ролик: «{best_title}» ({best_views:,} просм.) -> {best_url}")
                    return best_url
        except Exception as err:
            print(f"⚠️ Ошибка при поиске «{query}»: {err}")

    raise RuntimeError("Не удалось автоматически найти новый стрим-клип.")


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================
def process_stream_to_short(stream_url: str | None = None, duration: int = 45, lang: str = "ru") -> Path:
    if not stream_url or stream_url.lower() == "auto":
        stream_url = auto_discover_viral_stream_url(lang=lang)
    raw_video = download_stream(stream_url, max_duration_sec=240)
    words_timeline = transcribe_audio_with_whisper(raw_video)
    start_t, end_t, hook = find_best_highlight(words_timeline, target_duration=duration, lang=lang)
    clip_duration = min(end_t - start_t, float(duration))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"stream_short_{timestamp}_{lang.upper()}.mp4"

    render_split_vertical_short(
        input_video=raw_video,
        output_video=out_path,
        start_time=start_t,
        duration=clip_duration,
        hook_text=hook,
        words=words_timeline,
    )
    save_used_stream_url(stream_url)

    # Save multi-platform metadata (YouTube Shorts + TikTok + Instagram Reels)
    meta_bundle = {
        "language": lang.upper(),
        "source_url": stream_url,
        "video_file": str(out_path),
        "youtube_shorts": {
            "title": f"{hook} 😱🔥 #shorts #stream #clips",
            "description": f"{hook}\nSource: {stream_url}\n#shorts #stream #twitch #highlights #viral",
        },
        "tiktok": {
            "caption": f"{hook} 😭🔥 #fyp #viral #streamclips #twitch #funny",
        },
        "instagram_reels": {
            "caption": f"{hook} 👇 Wait till the end! 😭\n#reels #explore #streamer #clips #viral",
        },
    }
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(meta_bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if os.getenv("UPLOAD_TO_YOUTUBE", "1").lower() in {"1", "true", "yes"}:
        print("📤 Загружаем стрим-шортс на YouTube...")
        try:
            from uploader import upload_short
            upload_short(
                str(out_path),
                meta_bundle["youtube_shorts"]["title"],
                meta_bundle["youtube_shorts"]["description"],
            )
        except Exception as err:
            print(f"⚠️ Видео готово, но загрузка на YouTube потребовала авторизации: {err}")

    # Cleanup raw stream
    if raw_video.exists():
        raw_video.unlink()

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Twitch/YouTube Stream to Shorts AI Clipper (RU + EN)")
    parser.add_argument("url", nargs="?", default="auto", help="URL of Twitch/YouTube video, or 'auto' (default)")
    parser.add_argument("--duration", type=int, default=45, help="Target Short duration in seconds")
    parser.add_argument(
        "--lang",
        type=str,
        default="both",
        choices=["ru", "en", "both"],
        help="Streamer pool language: 'ru', 'en', or 'both' (default: both RU + EN)",
    )
    args = parser.parse_args()

    langs = ["ru", "en"] if args.lang == "both" else [args.lang]
    generated = []
    for language in langs:
        print(f"\n==================== СТРИМ-НАРЕЗКА [{language.upper()}] ====================")
        res = process_stream_to_short(args.url, duration=args.duration, lang=language)
        generated.append(res)
    print(f"\n🚀 ГОТОВО! Все стрим-ролики сохранены и загружены:\n" + "\n".join(str(p) for p in generated))
