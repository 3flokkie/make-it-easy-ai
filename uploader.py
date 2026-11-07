import os
import pickle
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent


def get_authenticated_service():
    """Авторизуется в YouTube API (при первом запуске откроет браузер для входа)"""
    creds = None
    token_path = BASE_DIR / "token.pickle"
    client_secret_path = BASE_DIR / "client_secret.json"

    if token_path.exists():
        with open(token_path, "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not client_secret_path.exists():
                client_secret_path = SCRIPT_DIR / "client_secret.json"
            if not client_secret_path.exists():
                raise FileNotFoundError(
                    f"❌ Не найден client_secret.json в {BASE_DIR} или {SCRIPT_DIR}."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "wb") as token:
            pickle.dump(creds, token)

    return build("youtube", "v3", credentials=creds)


def upload_short(video_path: str, title: str, description: str = "#shorts #ai #quiz #makeiteasyai"):
    """Загружает готовый файл на YouTube"""
    youtube = get_authenticated_service()

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": ["shorts", "ai", "quiz", "makeiteasyai"],
            "categoryId": "28"  # Наука и технологии
        },
        "status": {
            "privacyStatus": os.getenv("YOUTUBE_PRIVACY_STATUS", "private"),
            "selfDeclaredMadeForKids": False
        }
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)

    print(f"📤 Заливаем на YouTube: {title}...")
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"⏳ Загрузка: {int(status.progress() * 100)}%")

    print(f"🎉 Успешно залито! Ссылка: https://youtu.be/{response['id']}")
    return response['id']
