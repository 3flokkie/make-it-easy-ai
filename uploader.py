import json
import os
import pickle
import re
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"
# Базовые скоупы, которые уже есть в старых token*.pickle (загрузка + комментарии).
CORE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]
# Новые авторизации сразу запрашивают и аналитику. Старые токены продолжают работать без re-auth:
# refresh использует скоупы, сохранённые внутри самого токена.
SCOPES = [*CORE_SCOPES, ANALYTICS_SCOPE]
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
QUEUE_FILE = BASE_DIR / "output" / "pending_youtube_queue.json"


def _resolve_client_secret(token_name: str) -> Path:
    """Подбирает отдельный client_secret_<channel>.json под каждый токен (своя квота GCP на канал)."""
    candidates: list[str] = []
    stem = Path(token_name).stem  # e.g. 'token_channel2' -> suffix 'channel2'
    if stem.startswith("token_") and len(stem) > 6:
        suffix = stem[6:]
        candidates.append(f"client_secret_{suffix}.json")
    candidates.append("client_secret.json")

    for fname in candidates:
        for folder in (BASE_DIR, SCRIPT_DIR):
            p = folder / fname
            if p.exists():
                return p
    raise FileNotFoundError(
        f"❌ Не найден {candidates[0]} или client_secret.json в {BASE_DIR} / {SCRIPT_DIR}."
    )


class ScopeMissingError(RuntimeError):
    """У сохранённого токена нет нужного OAuth-скоупа (например, yt-analytics.readonly)."""


def _token_path(token_name: str) -> Path:
    token_path = BASE_DIR / token_name
    if not token_path.exists() and (SCRIPT_DIR / token_name).exists():
        token_path = SCRIPT_DIR / token_name
    return token_path


def _creds_scopes(creds) -> set[str]:
    granted = getattr(creds, "granted_scopes", None) or getattr(creds, "scopes", None) or []
    if isinstance(granted, str):
        granted = granted.split()
    return set(granted)


def token_has_scope(token_name: str, scope: str = ANALYTICS_SCOPE) -> bool:
    """Проверяет скоуп сохранённого токена без запуска OAuth-флоу."""
    token_path = _token_path(token_name)
    if not token_path.exists():
        return False
    try:
        with open(token_path, "rb") as token:
            creds = pickle.load(token)
    except Exception:
        return False
    return scope in _creds_scopes(creds)


def _load_credentials(token_name: str, scopes: list[str]):
    creds = None
    token_path = _token_path(token_name)

    if token_path.exists():
        with open(token_path, "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_secret_path = _resolve_client_secret(token_name)
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), scopes)
            creds = flow.run_local_server(port=0, prompt="select_account")

        with open(token_path, "wb") as token:
            pickle.dump(creds, token)

    return creds


def get_authenticated_service(token_name: str = "token.pickle"):
    """Авторизуется в YouTube API (использует сохранённый токен канала и его GCP client_secret)."""
    creds = _load_credentials(token_name, SCOPES)
    return build("youtube", "v3", credentials=creds)


def get_analytics_service(token_name: str = "token.pickle"):
    """YouTube Analytics API v2. Бросает ScopeMissingError, если токен выдан без yt-analytics.readonly.

    Загрузки/комменты при этом не ломаются. Чтобы включить аналитику — удали token-файл
    и авторизуйся заново (новый флоу уже запрашивает ANALYTICS_SCOPE).
    """
    token_path = _token_path(token_name)
    if token_path.exists() and not token_has_scope(token_name, ANALYTICS_SCOPE):
        raise ScopeMissingError(
            f"⚠️ {token_name} не содержит {ANALYTICS_SCOPE}. Аналитика недоступна — "
            f"удали {token_path.name} и пройди авторизацию заново, чтобы выдать доступ."
        )
    creds = _load_credentials(token_name, SCOPES)
    if ANALYTICS_SCOPE not in _creds_scopes(creds):
        raise ScopeMissingError(f"⚠️ {token_name}: OAuth не выдал {ANALYTICS_SCOPE}.")
    return build("youtubeAnalytics", "v2", credentials=creds)


def enqueue_failed_upload(video_path: str, title: str, description: str) -> None:
    """Сохраняет готовый ролик в очередь отложенной загрузки, если YouTube вернул лимит."""
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    queue: list[dict[str, str]] = []
    if QUEUE_FILE.exists():
        try:
            queue = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
        except Exception:
            queue = []
    abs_path = str(Path(video_path).resolve())
    if not any(item.get("video_path") == abs_path for item in queue):
        queue.append({
            "video_path": abs_path,
            "title": str(title)[:98],
            "description": description,
        })
        QUEUE_FILE.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📥 Ролик сохранён в очередь авто-дозагрузки ({QUEUE_FILE.name}): {Path(video_path).name}")


def flush_pending_uploads() -> list[str]:
    """Публикует все видео из очереди отложенной загрузки, как только лимит YouTube освободился."""
    if not QUEUE_FILE.exists():
        return []
    try:
        queue = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not queue:
        return []

    print(f"\n📬 В очереди отложенной загрузки найдено {len(queue)} готовых видео. Пробуем залить...")
    uploaded_ids: list[str] = []
    remaining: list[dict[str, str]] = []

    for idx, item in enumerate(queue):
        vpath = item.get("video_path", "")
        if not vpath or not os.path.exists(vpath):
            continue
        vid_id = upload_short(vpath, item.get("title", "Make It Easy AI"), item.get("description", ""), enqueue_on_limit=False)
        if vid_id:
            uploaded_ids.append(vid_id)
        else:
            # Лимит всё ещё активен — оставляем оставшиеся в очереди
            remaining.extend(queue[idx:])
            break

    QUEUE_FILE.write_text(json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8")
    return uploaded_ids


def upload_short(
    video_path: str,
    title: str,
    description: str = "#shorts #ai #makeiteasyai",
    enqueue_on_limit: bool = True,
    token_name: str = "token.pickle",
) -> str | None:
    """Загружает готовый файл на YouTube с авто-определением языка (RU/EN) и защитой от потери при uploadLimitExceeded."""
    youtube = get_authenticated_service(token_name=token_name)

    has_cyrillic = bool(re.search(r"[а-яА-ЯёЁ]", f"{title} {description}"))
    is_en = str(video_path).endswith("_EN.mp4") or not has_cyrillic
    lang_code = "en" if is_en else "ru"

    if is_en:
        tags = ["what if", "make it easy ai", "science animation", "explained", "shorts", "viral"]
    else:
        tags = ["что если", "make it easy ai", "научпоп", "анимация", "всё сложное просто", "шортс"]

    body = {
        "snippet": {
            "title": str(title)[:98],
            "description": description,
            "tags": tags,
            "categoryId": "24",
            "defaultLanguage": lang_code,
            "defaultAudioLanguage": lang_code,
        },
        "status": {
            "privacyStatus": os.getenv("YOUTUBE_PRIVACY_STATUS", "public"),
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)

    print(f"📤 Заливаем на YouTube [{lang_code.upper()}]: {title}...")
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    try:
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"⏳ Загрузка: {int(status.progress() * 100)}%")

        print(f"🎉 Успешно залито! Ссылка: https://youtu.be/{response['id']}")
        return response["id"]
    except HttpError as err:
        if "uploadLimitExceeded" in str(err) and enqueue_on_limit:
            print("⚠️ Достигнут суточный лимит YouTube (uploadLimitExceeded) — ставим готовый файл в очередь!")
            enqueue_failed_upload(video_path, title, description)
            return None
        raise


def set_video_thumbnail(video_id: str, thumbnail_path: str, token_name: str = "token.pickle") -> bool:
    """Загружает кликабельное превью (Thumbnail) для видео на YouTube."""
    if not os.path.exists(thumbnail_path):
        print(f"⚠️ Файл превью не найден: {thumbnail_path}")
        return False
    try:
        youtube = get_authenticated_service(token_name=token_name)
        mime = "image/jpeg" if thumbnail_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        media = MediaFileUpload(thumbnail_path, mimetype=mime, resumable=True)
        req = youtube.thumbnails().set(videoId=video_id, media_body=media)
        req.execute()
        print(f"🖼️ Превью успешно установлено для https://youtu.be/{video_id}")
        return True
    except Exception as e:
        print(f"⚠️ Не удалось установить превью для {video_id} (нужен подтверждённый по SMS аккаунт YouTube): {e}")
        return False
