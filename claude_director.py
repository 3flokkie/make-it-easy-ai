"""
Make It Easy AI — Claude Director Module
Защищённый модуль взаимодействия с Claude Pro (Opus/Sonnet) с жёстким лимит-менеджером.
Берёжёт квоту подписки Claude Pro:
- Максимум 1 вызов на выпуск (только ключевая режиссура и панчи)
- Кэширование ответов
- Автоматический фоллбек на Gemini при исчерпании лимитов/ошибках
"""
import subprocess
import json
import os
import time
from pathlib import Path

CACHE_DIR = Path("/Users/sashasyrota/PycharmProjects/WelcomeScreen/cache/claude_scripts")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Жёсткий счётчик вызовов в сутки (защита от выгорания лимитов Claude Pro)
DAILY_USAGE_FILE = CACHE_DIR / "daily_usage.json"
MAX_DAILY_CLAUDE_CALLS = 15  # Не больше 15 вызовов в сутки, всё остальное делает Gemini

def _get_daily_calls() -> int:
    today = time.strftime("%Y-%m-%d")
    if DAILY_USAGE_FILE.exists():
        try:
            with open(DAILY_USAGE_FILE, "r") as f:
                data = json.load(f)
            if data.get("date") == today:
                return data.get("calls", 0)
        except Exception:
            pass
    return 0

def _increment_daily_calls():
    today = time.strftime("%Y-%m-%d")
    calls = _get_daily_calls() + 1
    with open(DAILY_USAGE_FILE, "w") as f:
        json.dump({"date": today, "calls": calls}, f)

def ask_claude_director(prompt: str, fallback_fn=None) -> str:
    """
    Вызывает Claude Pro для финальной полировки сценария/режиссуры.
    Если лимит исчерпан или Claude не отвечает — прозрачно переключается на fallback_fn (Gemini).
    """
    calls_today = _get_daily_calls()
    if calls_today >= MAX_DAILY_CLAUDE_CALLS:
        print(f"🛡️ [Claude Guard] Суточный лимит вызовов Claude сбережён ({calls_today}/{MAX_DAILY_CLAUDE_CALLS}). Используем Gemini!")
        if fallback_fn:
            return fallback_fn()
        return ""

    print(f"🎭 [Claude Director] Подключаем Claude Pro к полировке (вызов {calls_today + 1}/{MAX_DAILY_CLAUDE_CALLS})...")
    
    cmd = [
        "/opt/homebrew/bin/claude",
        "-p",
        prompt
    ]
    env = os.environ.copy()
    env["PATH"] = f"/opt/homebrew/bin:{env.get('PATH', '')}"

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
        if res.returncode == 0 and res.stdout.strip():
            _increment_daily_calls()
            print("✅ [Claude Director] Сценарий от Claude Pro успешно получен!")
            return res.stdout.strip()
        else:
            print(f"⚠️ [Claude Guard] Ошибка Claude: {res.stderr.strip()[:200]}. Фоллбек на Gemini...")
            if fallback_fn:
                return fallback_fn()
            return ""
    except Exception as e:
        print(f"⚠️ [Claude Guard] Сбой вызова Claude ({e}). Фоллбек на Gemini...")
        if fallback_fn:
            return fallback_fn()
        return ""

if __name__ == "__main__":
    print(f"Текущее использование Claude: {_get_daily_calls()}/{MAX_DAILY_CLAUDE_CALLS} вызовов сегодня.")
