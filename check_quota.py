#!/usr/bin/env python3
"""Быстрая проверка остатка лимитов ElevenLabs."""
import os
import sys
import datetime
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("ELEVENLABS_API_KEY")
if not api_key:
    print("❌ ELEVENLABS_API_KEY не найден в scripts/.env")
    sys.exit(1)

try:
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=api_key)
    sub = client.user.subscription.get()

    used = sub.character_count
    limit = sub.character_limit
    remaining = max(0, limit - used)
    percent = (used / limit * 100.0) if limit else 0.0

    reset_str = "неизвестно"
    if getattr(sub, "next_character_count_reset_unix", None):
        reset_str = datetime.datetime.fromtimestamp(sub.next_character_count_reset_unix).strftime("%d.%m.%Y в %H:%M")

    print("\n" + "=" * 50)
    print(" 🎙️  СТАТУС ПОДПИСКИ ELEVENLABS")
    print("=" * 50)
    print(f" • Тарифный план:     {sub.tier.upper()} ({sub.status})")
    print(f" • Использовано:      {used:,} из {limit:,} симв. ({percent:.1f}%)")
    print(f" • ОСТАЛОСЬ:          {remaining:,} символов")
    print(f" • Дата сброса лимита: {reset_str}")
    
    # Расчет примерного запаса
    # 1 шортс на 45-50 сек = ~350-450 символов на 1 язык (или ~800 на RU+EN)
    shorts_left = remaining // 800
    print(f" • Хватит примерно на: ~{shorts_left} парных шортсов (RU + EN)")
    print("=" * 50 + "\n")

except Exception as err:
    print(f"❌ Ошибка проверки квоты ElevenLabs: {err}")
    sys.exit(1)
