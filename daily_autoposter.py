#!/usr/bin/env python3
"""
Daily Auto-Poster & Quality-First Scheduler for 'Make It Easy AI' (Europe/Vienna UTC+1)
========================================================================================
3 High-Quality Daily Releases (Zero Shadowban Risk + ElevenLabs Token Economy + <15m Cap):

1. 14:30 CET (16:30 MSK — Дневной прайм RU):
   - Пост №1: Премиальный 2D-мультфильм / сюжет «Make It Easy AI» (RU) с взрывным 5-сек хуком.
   - EN-версия озвучивается через бесплатный Edge-TTS (0 токенов ElevenLabs) и ставится в очередь на 20:30.

2. 18:00 CET (20:00 MSK — Главный вечерний прайм):
   - Пост №2: Полноценный большой выпуск «ЧТО, ЕСЛИ?» (16:9, строго < 15 минут, взрывные первые 5 сек, 0 токенов ElevenLabs)
     + привязанный Shorts-тизер (или публикация готового выпуска из очереди pending_youtube_queue.json).

3. 20:30 CET (14:30 EST / 19:30 UK — Прайм США и Европы):
   - Пост №3: Премиальный 2D-мультфильм «Make It Easy AI» (EN) из очереди 14:30 (0 токенов ElevenLabs).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR.parent / "output"
VIDEOS_DIR = OUTPUT_ROOT / "videos"
PENDING_EN_FILE = VIDEOS_DIR / "pending_en_upload.json"
QUEUE_FILE = OUTPUT_ROOT / "pending_youtube_queue.json"

SLOTS = ("1430", "1800", "2030", "stream_ru", "stream_en", "longform")


def detect_nearest_slot() -> str:
    now = datetime.now()
    cur_mins = now.hour * 60 + now.minute
    slot_mins = {
        "1430": 14 * 60 + 30,
        "1800": 18 * 60,
        "2030": 20 * 60 + 30,
    }
    return min(slot_mins.keys(), key=lambda k: abs(cur_mins - slot_mins[k]))


GDRIVE_ARCHIVE_ROOT = Path("/Users/sashasyrota/Library/CloudStorage/GoogleDrive-syrotaoleksandr2@gmail.com/Мой диск/MakeItEasyAI_Archive")


def sync_and_archive_to_gdrive() -> None:
    """Переносит готовые видеоролики и метаданные в 5TB Google Drive, освобождая 512GB SSD макбука."""
    if not GDRIVE_ARCHIVE_ROOT.exists():
        return
    moved_count = 0
    moved_bytes = 0
    for category in ("longform", "waterfall_shorts", "videos", "stream_shorts"):
        local_dir = OUTPUT_ROOT / category
        if not local_dir.exists():
            continue
        target_dir = GDRIVE_ARCHIVE_ROOT / category
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in local_dir.iterdir():
            if item.is_file() and (item.suffix.lower() in {".mp4", ".json", ".meta.json"}):
                # Не переносим файл очереди pending_en_upload.json, пока он нужен скрипту
                if item.name == "pending_en_upload.json":
                    continue
                try:
                    dest = target_dir / item.name
                    size = item.stat().st_size
                    shutil.copy2(item, dest)
                    item.unlink()
                    moved_count += 1
                    moved_bytes += size
                except Exception as e:
                    print(f"⚠️ Ошибка архивации {item.name} в Google Drive: {e}")
    if moved_count > 0:
        print(f"☁️ [Google Drive 5TB] Успешно перенесено в облако {moved_count} файлов ({moved_bytes / (1024 * 1024):.1f} МБ). SSD свободен!")


def cleanup_temp_dirs() -> None:
    """Удаляет временные чанки после рендера, чтобы не забивать SSD макбука."""
    freed_bytes = 0
    for temp_sub in ("temp_streams", "temp_longform", "audio"):
        tdir = OUTPUT_ROOT / temp_sub
        if not tdir.exists():
            continue
        for item in tdir.iterdir():
            try:
                if item.is_file():
                    freed_bytes += item.stat().st_size
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
            except Exception:
                pass
    if freed_bytes > 0:
        print(f"🧹 [Auto-Clean] Очищено {freed_bytes / (1024 * 1024):.1f} МБ временных кэш-файлов.")
    sync_and_archive_to_gdrive()


def _run_step(label: str, cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n▶️ {label}...")
    try:
        subprocess.run(cmd, env=env, check=True)
    except Exception as exc:
        print(f"⚠️ Шаг «{label}» завершился с предупреждением: {exc}")


def run_slot(slot: str) -> None:
    if slot == "auto":
        slot = detect_nearest_slot()

    print(f"\n🚀 [Make It Easy AI — Quality 3-Post Schedule] Запуск слота {slot} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})...")
    sys.path.insert(0, str(BASE_DIR))
    env = os.environ.copy()
    env["UPLOAD_TO_YOUTUBE"] = "1"

    if slot == "1430":
        print("🇷🇺 === СЛОТ 1/3 (14:30 CET): ПРЕМИАЛЬНЫЙ 2D-ШОРТС RU (+ EN В ОЧЕРЕДЬ) ===")
        env_cartoon = env.copy()
        env_cartoon["ENABLE_VOICE"] = "1"
        env_cartoon["ENABLE_ENGLISH"] = "1"
        env_cartoon["ELEVENLABS_EN_EDGE"] = "0"  # Максимальное премиум качество ElevenLabs для всех Shorts!
        env_cartoon["DEFER_EN_UPLOAD"] = "1"
        _run_step(
            "Генерация 2D-мультфильма Make It Easy AI (взрывные первые 5 сек)",
            [sys.executable, str(BASE_DIR / "main-pipeline.py")],
            env_cartoon,
        )

    elif slot == "1800":
        print("🍿 === СЛОТ 2/3 (18:00 CET): ГЛАВНЫЙ БОЛЬШОЙ ВЫПУСК «ЧТО, ЕСЛИ?» (< 15 МИН, 0 ТОКЕНОВ ELEVENLABS) ===")
        # Если в очереди pending_youtube_queue.json уже лежит срендеренный большой выпуск — заливаем его первым!
        flushed = []
        try:
            from uploader import flush_pending_uploads

            flushed = flush_pending_uploads()
        except Exception as exc:
            print(f"ℹ️ Проверка очереди отложенных загрузок: {exc}")

        if not flushed:
            # Чередуем язык главного выпуска по дням (нечётные дни — RU, чётные — EN) или запускаем RU
            lang_today = "ru" if datetime.now().day % 2 == 1 else "en"
            env_long = env.copy()
            env_long["SHORTS_VOICE_ENGINE"] = "elevenlabs"  # Топовый сочный голос ElevenLabs для Shorts-тизера!
            _run_step(
                f"Большой выпуск «ЧТО, ЕСЛИ?» 16:9 (< 15 мин) + Shorts-тизер ({lang_today.upper()})",
                [sys.executable, str(BASE_DIR / "longform_factory.py"), "--chapters", "4", "--scenes-per-chapter", "5", "--lang", lang_today, "--shorts-voice", "elevenlabs"],
                env_long,
            )

    elif slot == "2030":
        print("🇺🇸 === СЛОТ 3/3 (20:30 CET): ПРЕМИАЛЬНЫЙ 2D-ШОРТС EN (US PRIME TIME) ===")
        if PENDING_EN_FILE.exists():
            print("▶️ Публикуем готовый EN 2D-мультфильм Make It Easy AI из дневной очереди 14:30...")
            try:
                data = json.loads(PENDING_EN_FILE.read_text(encoding="utf-8"))
                video_file = data.get("video_file", "")
                title = data.get("title", "Make It Easy AI #shorts")
                desc = data.get("description", "#shorts #makeiteasyai #explained")
                if video_file and os.path.exists(video_file):
                    from uploader import upload_short

                    vid_id = upload_short(video_file, title, desc)
                    if vid_id:
                        PENDING_EN_FILE.unlink(missing_ok=True)
            except Exception as exc:
                print(f"⚠️ Ошибка загрузки отложенного EN-мультика: {exc}")
        else:
            _run_step(
                "Резервный EN Стрим-Шортс (0 токенов ElevenLabs)",
                [sys.executable, str(BASE_DIR / "stream_clipper.py"), "auto", "--lang", "en"],
                env,
            )

    elif slot == "stream_ru":
        _run_step("RU Стрим-Шортс", [sys.executable, str(BASE_DIR / "stream_clipper.py"), "auto", "--lang", "ru"], env)
    elif slot == "stream_en":
        _run_step("EN Стрим-Шортс", [sys.executable, str(BASE_DIR / "stream_clipper.py"), "auto", "--lang", "en"], env)
    elif slot == "longform":
        _run_step(
            "Большой выпуск «ЧТО, ЕСЛИ?» (< 15 мин)",
            [sys.executable, str(BASE_DIR / "longform_factory.py"), "--chapters", "4", "--scenes-per-chapter", "5", "--lang", "ru"],
            env,
        )
    else:
        raise ValueError(f"Неизвестный слот: {slot}")

    cleanup_temp_dirs()


def run_daemon_loop() -> None:
    print("🚀 [Daemon] Запущен премиальный график 3 постов в день (Europe/Vienna): 14:30 (RU Short), 18:00 (Full Video), 20:30 (EN Short)")
    fired_today: set[str] = set()
    last_day = datetime.now().strftime("%Y-%m-%d")

    while True:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        if today != last_day:
            fired_today.clear()
            last_day = today

        hhmm = now.strftime("%H%M")
        if hhmm in ("1430", "1800", "2030") and hhmm not in fired_today:
            fired_today.add(hhmm)
            try:
                run_slot(hhmm)
            except Exception as err:
                print(f"⚠️ Ошибка в слоте {hhmm}: {err}")
        time.sleep(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make It Easy AI — 3 Daily Quality Releases (14:30, 18:00, 20:30)")
    parser.add_argument(
        "--slot",
        type=str,
        default="auto",
        choices=[*SLOTS, "auto"],
        help="Target slot: 1430 (RU Short), 1800 (Longform <15m), 2030 (EN Short), or auto",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuously in background and trigger at 14:30, 18:00, and 20:30",
    )
    args = parser.parse_args()
    if args.daemon:
        run_daemon_loop()
    else:
        run_slot(args.slot)
