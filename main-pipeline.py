import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from google import genai
from google.genai import types
from moviepy import AudioFileClip, ImageClip, concatenate_videoclips
from moviepy.audio.io.readers import FFMPEG_AudioReader
from PIL import Image, ImageDraw, ImageFont

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

WIDTH, HEIGHT, FPS = 1080, 1920, 30
ASSETS_DIR = BASE_DIR / "assets" / "images"
AUDIO_DIR = BASE_DIR / "output" / "audio"
VIDEOS_DIR = BASE_DIR / "output" / "videos"
for directory in (ASSETS_DIR, AUDIO_DIR, VIDEOS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
if not GEMINI_KEY or not ELEVEN_KEY:
    raise ValueError("Ключи GEMINI_API_KEY и ELEVENLABS_API_KEY должны быть в scripts/.env")

gemini_client = genai.Client(
    api_key=GEMINI_KEY,
    http_options=types.HttpOptions(
        timeout=int(os.getenv("GEMINI_TIMEOUT_MS", "20000")),
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)
eleven_client = ElevenLabs(api_key=ELEVEN_KEY)

FALLBACK_STORY: dict[str, Any] = {
    "title": "Последний автобус",
    "scenes": [
        {"timing": "0-5", "text": "В 2:17 ночи последний автобус остановился посреди леса.",
         "visual": "bus", "visual_detail": "Тусклый фонарь освещает пустой автобус среди мокрых деревьев."},
        {"timing": "5-10", "text": "Водитель исчез, а двери заблокировались изнутри.",
         "visual": "bus", "visual_detail": "Пустое водительское кресло, мигающая панель и закрытый замок двери."},
        {"timing": "10-16", "text": "За окном появились три силуэта, повторяющие каждое движение.",
         "visual": "forest", "visual_detail": "Три неподвижные фигуры стоят между деревьями за запотевшим окном."},
        {"timing": "16-22", "text": "Правило первое: не смотри им прямо в глаза.",
         "visual": "shadow", "visual_detail": "Тени приближаются к стеклу, а тусклый фонарик дрожит в руке героя."},
        {"timing": "22-28", "text": "Правило второе: держи руку на замке, но не открывай дверь.",
         "visual": "lock", "visual_detail": "Пальцы героя сжимают холодный металлический замок автобуса."},
        {"timing": "28-34", "text": "Ты услышал голос сестры снаружи, хотя она исчезла год назад.",
         "visual": "hands", "visual_detail": "Детская ладонь отпечаталась на запотевшем стекле рядом с рукой героя."},
        {"timing": "34-40", "text": "Автобус завёлся сам. Силуэты уже стояли в проходе.",
         "visual": "bus", "visual_detail": "Тёмные фигуры заполняют узкий проход, приборная панель мигает красным."},
        {"timing": "40-45", "text": "Выход был рядом, но за дверью не было земли.",
         "visual": "cave", "visual_detail": "Открытая дверь ведёт в чёрную пустоту, под автобусом видны камни."},
        {"timing": "45-50", "text": "Ты бы открыл дверь, чтобы спастись? Напиши ДА или НЕТ.",
         "visual": "choice", "visual_detail": "Крупные контрастные кнопки ДА и НЕТ на фоне открытой двери."},
    ],
}


def get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def wrapped(text: str, width: int = 23) -> str:
    return "\n".join(textwrap.wrap(text.upper(), width=width, break_long_words=False))


def draw_base(draw: ImageDraw.ImageDraw, phase: int) -> None:
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(9, 14, 29))
    draw.ellipse((-380, 250, 500, 1280), fill=(19, 34, 57))
    draw.ellipse((720, 1280, 1450, 2150), fill=(12, 49, 64))
    draw.line((0, 480 + phase * 4, WIDTH, 410 + phase * 4), fill=(46, 104, 119), width=12)
    draw.rounded_rectangle((38, 38, WIDTH - 38, HEIGHT - 38), radius=42,
                           outline=(255, 202, 64), width=12)


def draw_scene(scene_type: str, phase: int, subtitle: str = "", word_index: int = -1,
               words: list[str] | None = None, choice: bool = False) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (9, 14, 29))
    draw = ImageDraw.Draw(image)
    draw_base(draw, phase)
    accent = (255, 202, 64)
    shadow = (4, 7, 14)

    if scene_type == "bus":
        draw.rectangle((145, 600 + phase * 3, 935, 1350 + phase * 3), fill=(35, 44, 58), outline=accent, width=12)
        draw.rectangle((205, 710 + phase * 3, 875, 980 + phase * 3), fill=(12, 25, 45), outline=(104, 182, 199), width=8)
        for x in range(250, 850, 150):
            draw.rectangle((x, 755 + phase * 3, x + 90, 900 + phase * 3), fill=(91, 145, 155))
        draw.ellipse((220, 1250 + phase * 3, 370, 1400 + phase * 3), fill=shadow, outline=accent, width=8)
        draw.ellipse((710, 1250 + phase * 3, 860, 1400 + phase * 3), fill=shadow, outline=accent, width=8)
    elif scene_type == "forest":
        for x in range(100, 1050, 170):
            top = 560 + ((x + phase * 35) % 80)
            draw.rectangle((x - 18, top + 230, x + 18, 1370), fill=(41, 28, 29))
            draw.polygon((x, top, x - 125, top + 300, x + 125, top + 300), fill=(22, 65, 58))
            draw.polygon((x, top + 110, x - 105, top + 400, x + 105, top + 400), fill=(16, 48, 48))
        for x in (350, 540, 735):
            draw.ellipse((x - 32, 920, x + 32, 1000), fill=(222, 206, 117))
    elif scene_type == "lock":
        draw.rounded_rectangle((240, 650, 840, 1300), radius=60, fill=(40, 45, 54), outline=accent, width=14)
        draw.arc((375, 450, 705, 920), 180, 360, fill=accent, width=42)
        draw.ellipse((490, 870, 590, 970), fill=accent)
        draw.rectangle((525, 940, 555, 1110), fill=accent)
    elif scene_type == "hands":
        draw.ellipse((180, 790, 560, 1300), fill=(122, 70, 67), outline=accent, width=8)
        draw.ellipse((520, 730, 900, 1280), fill=(109, 62, 64), outline=accent, width=8)
        draw.line((440, 1040, 660, 930), fill=(214, 139, 112), width=50)
    elif scene_type == "cave":
        draw.polygon((80, 1450, 150, 500, 440, 250, 540, 660, 640, 250, 960, 500, 1010, 1450), fill=(31, 26, 42))
        draw.ellipse((380, 710, 700, 1250), fill=(1, 3, 8), outline=(93, 159, 170), width=12)
        draw.line((120, 1460, 960, 1460), fill=(55, 81, 92), width=16)
    else:
        draw.ellipse((300, 540, 780, 1100), fill=(18, 31, 54), outline=accent, width=12)
        draw.ellipse((440, 680, 640, 880), fill=(8, 12, 24), outline=(104, 182, 199), width=8)
        for x in (350, 540, 730):
            draw.ellipse((x - 35, 1000, x + 35, 1070), fill=accent)

    if choice:
        for box, label, color in (
            ((95, 1400, 490, 1630), "ДА", (40, 178, 104)),
            ((590, 1400, 985, 1630), "НЕТ", (220, 65, 66)),
        ):
            draw.rounded_rectangle(box, radius=36, fill=color, outline="white", width=8)
            draw.text(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), label,
                      font=get_font(82), fill="white", anchor="mm",
                      stroke_width=5, stroke_fill=shadow)

    if subtitle:
        draw.rounded_rectangle((90, 150, 990, 420), radius=28, fill=(5, 9, 20), outline=accent, width=5)
        if words:
            lines: list[list[str]] = [[]]
            for word in words:
                if sum(len(item) + 1 for item in lines[-1]) + len(word) > 24:
                    lines.append([])
                lines[-1].append(word)
            y = 210
            word_position = 0
            for line in lines:
                line_text = " ".join(line).upper()
                widths = [draw.textlength(item.upper(), font=get_font(52)) for item in line]
                spacing = 12
                start_x = 540 - (sum(widths) + spacing * (len(line) - 1)) / 2
                for item, item_width in zip(line, widths):
                    color = accent if word_position == word_index else "white"
                    draw.text((start_x + item_width / 2, y), item.upper(), font=get_font(52),
                              fill=color, anchor="mm", stroke_width=3, stroke_fill=shadow)
                    start_x += item_width + spacing
                    word_position += 1
                y += 70
        else:
            draw.multiline_text((540, 285), wrapped(subtitle, 25), font=get_font(53),
                                fill="white", anchor="mm", align="center",
                                stroke_width=3, stroke_fill=shadow)
    return image


def generate_story() -> dict[str, Any]:
    print("🤖 [1/5] Генерируем историю выживания 40–50 секунд...")
    prompt = """
Ты — профессиональный сценарист короткометражных хорроров и интерактивных
роликов для YouTube Shorts. Создай цельную, логичную и пугающую историю-
загадку с выбором «Да или Нет» длительностью 40–50 секунд.

Строгие правила:
1. Все сцены происходят в одной или максимум двух тесно связанных локациях.
Никаких резких переходов между несвязанными местами.
2. Сюжет развивается линейно:
- 0–5 секунд: крючок, проблема и смертельная угроза;
- 5–25 секунд: нарастание, правило выживания или улика;
- 25–40 секунд: кульминация, роковое действие или жуткая разгадка;
- 40–50 секунд: вопрос «Да или Нет?» и просьба написать ответ.
3. Используй конкретные визуальные детали: тусклый фонарик, мокрое стекло,
тени, руки, замки и другие предметы, которые можно нарисовать в 2D.
4. Пиши на русском языке. Текст каждой сцены должен быть коротким и удобным
для озвучки.

Верни ТОЛЬКО валидный JSON следующего вида:
{
  "title": "короткое название",
  "location": "единая локация истории",
  "scenes": [
    {
      "timing": "0-5",
      "text": "текст озвучки",
      "visual": "bus|forest|lock|hands|cave|shadow|choice",
      "visual_detail": "конкретное описание визуального кадра"
    }
  ]
}
Создай ровно 9 сцен с таймингами 0-5, 5-10, 10-16, 16-22, 22-28,
28-34, 34-40, 40-45 и 45-50. Последняя сцена обязана содержать вопрос
и призыв написать «ДА» или «НЕТ».
"""
    try:
        response = gemini_client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        data: Any = json.loads(response.text)
        scenes = data.get("scenes") if isinstance(data, dict) else None
        if not isinstance(scenes, list) or len(scenes) != 9:
            raise ValueError("Gemini вернул не 9 сцен")
        expected_timings = (
            "0-5", "5-10", "10-16", "16-22", "22-28",
            "28-34", "34-40", "40-45", "45-50",
        )
        clean = []
        for scene, expected_timing in zip(scenes, expected_timings):
            if not isinstance(scene, dict) or not isinstance(scene.get("text"), str):
                raise ValueError("Некорректная сцена")
            if str(scene.get("timing", "")).replace(" ", "") != expected_timing:
                raise ValueError(f"Некорректный тайминг сцены: {scene.get('timing')}")
            visual = scene.get("visual", "shadow")
            clean.append({
                "timing": str(scene.get("timing", "")),
                "text": scene["text"].strip(),
                "visual": visual if visual in
                {"bus", "forest", "lock", "hands", "cave", "shadow", "choice"} else "shadow",
                "visual_detail": str(scene.get("visual_detail", "")).strip(),
            })
        return {
            "title": str(data.get("title", "История выживания")),
            "location": str(data.get("location", "Единая локация")),
            "scenes": clean,
        }
    except Exception as error:
        print(f"⚠️ Gemini не ответил вовремя или недоступен ({error}); используем резервную историю.")
        return FALLBACK_STORY


def generate_voice(text: str, filename: str) -> Path:
    path = AUDIO_DIR / filename
    try:
        stream = eleven_client.text_to_speech.convert(
            text=text,
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"),
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        with path.open("wb") as file:
            for chunk in stream:
                if chunk:
                    file.write(chunk)
    except Exception as error:
        raise RuntimeError(f"Ошибка озвучки ElevenLabs: {error}") from error
    return path


def make_scene_clip(scene: dict[str, str], audio_path: Path, timestamp: str, scene_number: int,
                    clips: list[Any], audio_clips: list[Any]) -> None:
    source_audio = AudioFileClip(str(audio_path))
    # Leave a margin because compressed audio readers can report an imprecise end.
    safe_duration = max(source_audio.duration - 0.15, 0.1)
    audio = source_audio.subclipped(0, safe_duration)
    audio_clips.append(audio)
    words = scene["text"].split()
    word_duration = audio.duration / max(len(words), 1)
    word_clips = []
    for index, _ in enumerate(words):
        frame_path = ASSETS_DIR / f"{timestamp}_scene{scene_number}_word{index}.png"
        image = draw_scene(
            scene["visual"],
            index % 3,
            scene["text"],
            index,
            words,
            scene["visual"] == "choice",
        )
        image.save(frame_path)
        duration = (
            word_duration
            if index < len(words) - 1
            else audio.duration - word_duration * (len(words) - 1)
        )
        word_clips.append(ImageClip(str(frame_path)).with_duration(duration))
    base_clip = concatenate_videoclips(word_clips, method="compose")
    # Keep video strictly inside the audio interval; never extend either side.
    scene_duration = min(base_clip.duration, audio.duration)
    if base_clip.duration > scene_duration:
        base_clip = base_clip.subclipped(0, scene_duration)
    scene_clip = base_clip.with_audio(audio)
    clips.append(scene_clip)


def build_single_short(index: int = 1) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    story = generate_story()
    print(f"📜 {story['title']} ({len(story['scenes'])} сцен)")
    clips: list[Any] = []
    audio_clips: list[Any] = []
    try:
        print("🎙️🎨 [2/5] Озвучиваем сцены и создаём 2D-анимацию...")
        for scene_number, scene in enumerate(story["scenes"], 1):
            audio_path = generate_voice(scene["text"], f"{timestamp}_scene{scene_number}.mp3")
            make_scene_clip(scene, audio_path, timestamp, scene_number, clips, audio_clips)

        total_duration = sum(clip.duration for clip in clips)
        if total_duration < 40 and clips:
            extension = min(40 - total_duration, 50 - total_duration)
            clips[-1] = clips[-1].with_duration(clips[-1].duration + extension)
            total_duration += extension
        if not 40 <= total_duration <= 50:
            print(f"ℹ️ Длительность озвучки: {total_duration:.1f} секунд (цель 40–50).")
        output_path = VIDEOS_DIR / f"survival_story_{timestamp}_{index}.mp4"
        print(f"🎬 [3/5] Рендерим вертикальное видео ({total_duration:.1f} сек.)...")
        final_video = concatenate_videoclips(clips, method="compose")
        if final_video.duration > 50:
            final_video = final_video.subclipped(0, 50)
            total_duration = 50
        try:
            final_video.write_videofile(str(output_path), fps=FPS, codec="libx264",
                                        audio_codec="aac", preset="ultrafast", logger="bar")
        finally:
            final_video.close()
        print(f"✅ Видео готово: {output_path}")
    finally:
        for clip in clips + audio_clips:
            clip.close()

    print("📤 [4/5] Загружаем ролик на YouTube...")
    try:
        from uploader import upload_short
        upload_short(str(output_path), f"{story['title']} #shorts")
    except Exception as error:
        raise RuntimeError(f"Видео создано, но загрузка не выполнена: {error}") from error
    print("🏁 [5/5] Готово.")
    return output_path


if __name__ == "__main__":
    for number in range(1, int(os.getenv("SHORTS_TO_GENERATE", "1")) + 1):
        build_single_short(number)
