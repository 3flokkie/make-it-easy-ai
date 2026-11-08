"""Legacy demo pipeline.

The production horror-series pipeline is ``main-pipeline.py``. This file is
kept only as an old gTTS/MoviePy experiment and is not part of production.
"""

from pathlib import Path
from gtts import gTTS
from PIL import Image, ImageDraw
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

# Определяем корень проекта и нужные папки независимо от того, откуда запущен скрипт
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "assets" / "images"
AUDIO_DIR = BASE_DIR / "output" / "audio"
VIDEOS_DIR = BASE_DIR / "output" / "videos"

def create_test_images():
    """Создает 2 тестовые картинки 1080x1920, если их еще нет в assets/images"""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    slides = [
        (IMAGES_DIR / "slide1.png", (255, 87, 34), "SLIDE 1: HOOK"),
        (IMAGES_DIR / "slide2.png", (41, 121, 255), "SLIDE 2: ANSWER")
    ]
    image_paths = []
    for path, color, label in slides:
        if not path.exists():
            img = Image.new("RGB", (1080, 1920), color=color)
            draw = ImageDraw.Draw(img)
            draw.text((350, 930), label, fill="white")
            img.save(path)
        image_paths.append(str(path))
    return image_paths

def generate_short(image_paths, text_script, output_name="first_short.mp4"):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    audio_path = AUDIO_DIR / "voice.mp3"
    output_video_path = VIDEOS_DIR / output_name

    print("🎙️ 1/3 Генерируем озвучку...")
    tts = gTTS(text=text_script, lang="en", slow=False)
    tts.save(str(audio_path))

    audio_clip = AudioFileClip(str(audio_path))
    slide_duration = audio_clip.duration / len(image_paths)

    print("🎬 2/3 Собираем вертикальный видеоряд (9:16)...")
    clips = []
    for img_path in image_paths:
        # Синтаксис под MoviePy 2.2.1 (.with_duration и .resized)
        clip = (
            ImageClip(img_path)
            .with_duration(slide_duration)
            .resized(height=1920)
        )
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose").with_audio(audio_clip)

    print("💾 3/3 Рендерим финальный MP4...")
    video.write_videofile(
        str(output_video_path),
        fps=24,
        codec="libx264",
        audio_codec="aac"
    )

    audio_clip.close()
    video.close()
    print(f"✅ Готово! Открывай ролик: {output_video_path}")

if __name__ == "__main__":
    test_slides = create_test_images()
    script_text = "Everything complex made simple! Welcome to Make It Easy Ai. Can you guess this game in 3 seconds?"
    generate_short(test_slides, script_text)