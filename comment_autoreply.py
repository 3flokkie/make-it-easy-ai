#!/usr/bin/env python3
"""
Automated YouTube Comment Responder for Make It Easy AI.
Polls recent videos, detects unanswered top-level comments, and replies
with helpful keyword-matched schemes (or Gemini-generated short replies).
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")
sys.path.insert(0, str(SCRIPT_DIR))
from uploader import get_authenticated_service  # noqa: E402

REPLIED_LOG = BASE_DIR / "output" / "replied_comments.json"

KEYWORD_REPLIES = {
    "бот": "Лови схему: 1) pytchat слушает живой чат и ловит 3x спайк сообщений 2) yt-dlp + ffmpeg держат кольцевой буфер и режут последние 40 сек в 9:16 3) faster-whisper накладывает субтитры 4) YouTube API заливает в Shorts. Закинь этот список в Claude Code — соберёт за 10 минут!",
    "голос": "Вот схема за 15 минут: 1) Загружаешь 30–60 сек своего голоса в ElevenLabs (Instant Voice Cloning). 2) Через API (модель eleven_multilingual_v2) озвучиваешь переведённый текст на EN/DE/ES с сохранением тембра. 3) Через faster-whisper делаешь тайминги субтитров и собираешь ролик!",
    "база": "Лови базу в 3 шага: 1) Выбери одну узкую рутину (разбор PDF, лидогенерация или автопостинг) 2) Собери связку на Python + Claude/Gemini API, а не просто переписывайся в чате 3) Сделай бесплатный тест для 1 бизнеса и покажи экономию времени!",
    "система": "Вот связка в 3 шага: 1) Парсер/поиск лидов на Python 2) Claude Code или Gemini собирает готовое решение 3) Авто-отправка и инвойс. Все наши инструменты и кейсы — на aingeneer.com!",
    "армия": "Вот состав цифровой армии из 3 агентов: 1) Радар трендов (парсит YouTube/TikTok) 2) Сборщик решений (Claude Code / Remotion) 3) Автопостер (YouTube API + cron на сервере). Всё крутится 24/7 без зарплат и созвонов!",
    "аудит": "Шаблон ИИ-аудита за 20 минут: 1) Берёшь 3 рутинных процесса компании (счета, заявки, перенос данных в таблицы) 2) Считаешь часы × ставку сотрудника 3) Рисуешь карту автоматизации на агентах и показываешь экономию за 1-й месяц!",
    "связка": "3 рабочие связки 2026: 1) PDF-накладная → Python + Vision → таблица в 1С/DATEV 2) Живой стрим → pytchat спайк → ffmpeg + Whisper → Shorts 3) Текст → ElevenLabs + Remotion → готовый 60fps ролик!",
    "автопилот": "Как собрать автопилот из 3 агентов: 1) Радар (поиск горячих лидов на картах/в соцсетях) 2) Фильтр + генератор персонального демо 3) Авто-дожим и ссылка на оплату. Подробнее и кейсы — на aingeneer.com!",
    "сотрудник": "Как собрать цифрового сотрудника за $20/мес: берёшь Python + API (Gemini/Claude), даёшь ему доступ к одной папке/почте/таблице, прописываешь жёсткие правила проверки и ставишь на cron 24/7!",
    "чертёж": "Архитектура ИИ-охотника: 1) Скрапер карт/сайтов ищет бизнес без автоматизации 2) Агент делает аудит слабых мест и мини-демо 3) Авто-отправка короткого письма владельцу напрямую мимо бирж!",
    "тест": "Напиши нам через контакты на aingeneer.com (там есть Telegram и почта) — присылай 1 свой документ или задачу, сделаем бесплатный пилот за 48 часов!",
}


def load_replied() -> set[str]:
    if REPLIED_LOG.exists():
        try:
            return set(json.loads(REPLIED_LOG.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def save_replied(ids: set[str]) -> None:
    REPLIED_LOG.parent.mkdir(parents=True, exist_ok=True)
    REPLIED_LOG.write_text(json.dumps(sorted(ids), indent=2), encoding="utf-8")


def generate_smart_reply(comment_text: str, video_title: str) -> str:
    low = comment_text.lower()
    for kw, ans in KEYWORD_REPLIES.items():
        if kw in low:
            return ans

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            prompt = (
                f"Ты автор YouTube-канала Make It Easy AI. Зритель оставил комментарий под роликом «{video_title}»:\n"
                f"«{comment_text}»\n"
                "Напиши короткий, дружелюбный, живой ответ на русском (1-2 предложения, без воды и официоза). "
                "Если это шутка или критика — отшутись по-доброму и поблагодари за активность."
            )
            res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            if res.text:
                return res.text.strip()[:450]
        except Exception:
            pass
    return "Спасибо за коммент! Если интересна конкретная схема — напиши кодовое слово из ролика, сразу скину 🔥"


def process_channel(token_name: str, my_handle: str, is_gaming_channel: bool = False) -> int:
    try:
        yt = get_authenticated_service(token_name)
        ch = yt.channels().list(part="contentDetails", mine=True).execute()
    except Exception as e:
        print(f"⚠️ Skip {token_name}: {e}")
        return 0

    uploads_id = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    vids = yt.playlistItems().list(part="snippet", playlistId=uploads_id, maxResults=15).execute()

    replied = load_replied()
    new_replies = 0

    for v in vids.get("items", []):
        vid = v["snippet"]["resourceId"]["videoId"]
        title = v["snippet"]["title"]
        if vid == "VuQk8XtsODw":
            continue
        try:
            res = yt.commentThreads().list(part="snippet", videoId=vid, maxResults=20, order="time").execute()
        except Exception:
            continue

        for item in res.get("items", []):
            cid = item["id"]
            if cid in replied:
                continue
            top = item["snippet"]["topLevelComment"]["snippet"]
            if item["snippet"].get("totalReplyCount", 0) > 0:
                replied.add(cid)
                continue
            if top.get("authorDisplayName") == my_handle:
                replied.add(cid)
                continue

            text_in = top.get("textDisplay", "")
            if is_gaming_channel:
                api_key = os.getenv("GEMINI_API_KEY", "").strip()
                reply_text = "Insane moment! 🔥 Subscribe for non-stop GTA 6 live highlights!"
                if api_key:
                    try:
                        client = genai.Client(api_key=api_key)
                        r = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=f"Reply shortly (1 friendly sentence, matching the comment's language) to this YouTube Shorts comment on '{title}': «{text_in}»",
                        )
                        if r.text:
                            reply_text = r.text.strip()[:300]
                    except Exception:
                        pass
            else:
                reply_text = generate_smart_reply(text_in, title)

            yt.comments().insert(
                part="snippet",
                body={"snippet": {"parentId": cid, "textOriginal": reply_text}},
            ).execute()
            replied.add(cid)
            new_replies += 1
            print(f"✅ [{token_name}] Replied to {top.get('authorDisplayName')} ({vid}): {reply_text[:60]}...")

    save_replied(replied)
    return new_replies


def main() -> None:
    total = 0
    total += process_channel("token.pickle", "@MakeItEasy-ai26", is_gaming_channel=False)
    total += process_channel("token_channel2.pickle", "@NightMode", is_gaming_channel=True)
    print(f"Done. Total new replies sent: {total}")


if __name__ == "__main__":
    main()
