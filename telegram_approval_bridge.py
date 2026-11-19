#!/usr/bin/env python3
"""
AIngeneer Telegram Approval Bridge & Remote Control
Uses native urllib for robust, socket-drop-proof communication with Telegram API.
"""

import sys
import os
import time
import json
import threading
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.parse
import urllib.error

BOT_TOKEN = "9174724584:AAFWBsUV5Tc8QNn3nXUtHsZ-ScztRbNxWIo"
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
STATE_FILE = os.path.expanduser("~/.config/aingeneer_bot/state.json")

os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"chat_id": None}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

state = load_state()
pending_requests = {}

def api_call(endpoint, data=None):
    url = f"{BASE_URL}/{endpoint}"
    req = urllib.request.Request(url)
    if data is not None:
        req.add_header("Content-Type", "application/json")
        encoded_data = json.dumps(data).encode("utf-8")
        req.data = encoded_data
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        # print(f"[API] Error calling {endpoint}: {e}")
        return None

def send_message(text, reply_markup=None):
    chat_id = state.get("chat_id")
    if not chat_id:
        print("[Bridge] No chat_id saved yet. Send /start to @Aingeneerbot first!")
        return None
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    return api_call("sendMessage", payload)

def send_photo(photo_path, caption=None):
    chat_id = state.get("chat_id")
    if not chat_id:
        return None
    # Use curl for multipart file upload - 100% reliable
    cmd = ["curl", "-s", "-X", "POST", f"{BASE_URL}/sendPhoto", "-F", f"chat_id={chat_id}", "-F", f"photo=@{photo_path}"]
    if caption:
        cmd.extend(["-F", f"caption={caption}"])
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"[Bridge] Send photo error: {e}")
        return False

def ask_approval_sync(question, timeout=300):
    req_id = str(int(time.time() * 1000))
    event = threading.Event()
    pending_requests[req_id] = {"event": event, "status": None}
    
    markup = {
        "inline_keyboard": [
            [
                {"text": "✅ Разрешить", "callback_data": f"approve:{req_id}"},
                {"text": "❌ Отклонить", "callback_data": f"deny:{req_id}"}
            ]
        ]
    }
    text = f"<b>⚠️ Запрос подтверждения:</b>\n\n<code>{question}</code>"
    send_message(text, reply_markup=markup)
    
    finished = event.wait(timeout)
    result = pending_requests.pop(req_id, {}).get("status")
    if not finished:
        send_message(f"⌛ Время ожидания запроса <code>{req_id}</code> истекло (таймаут).")
        return False
    return result == "approved"

class BridgeHTTPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        data = {}
        try:
            data = json.loads(body)
        except Exception:
            pass
            
        path = self.path
        if path == "/ask":
            question = data.get("question", "Требуется подтверждение")
            timeout = int(data.get("timeout", 300))
            approved = ask_approval_sync(question, timeout=timeout)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"approved": approved}).encode("utf-8"))
        elif path == "/notify":
            text = data.get("text", "Уведомление от системы")
            send_message(text)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        elif path == "/screen":
            tmp_screen = "/tmp/telegram_remote_screen.png"
            subprocess.run(["screencapture", "-x", tmp_screen])
            send_photo(tmp_screen, caption="📸 Текущий снимок экрана Mac")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return

def poll_telegram():
    last_update_id = 0
    print("[Bridge] Telegram polling loop started...")
    while True:
        try:
            params = "timeout=2"
            if last_update_id:
                params += f"&offset={last_update_id + 1}"
            url = f"{BASE_URL}/getUpdates?{params}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                
            if not data or not data.get("ok"):
                time.sleep(1)
                continue
                
            for upd in data.get("result", []):
                last_update_id = max(last_update_id, upd["update_id"])
                
                # Messages
                if "message" in upd:
                    msg = upd["message"]
                    chat_id = msg["chat"]["id"]
                    text = msg.get("text", "").strip()
                    user_name = msg["from"].get("first_name", "User")
                    print(f"[Bridge] Received from {user_name} ({chat_id}): {text}")
                    
                    state["chat_id"] = chat_id
                    save_state(state)
                    
                    if text.startswith("/start"):
                        send_message(f"👋 <b>Привет, {user_name}!</b>\nБот успешно привязан к твоему Mac!\n\nКоманды:\n/status - состояние Mac\n/screen - снимок экрана\n/exec &lt;команда&gt; - выполнить в терминале\n\n<i>Теперь подтверждения будут приходить прямо сюда кнопками!</i>")
                    elif text == "/status":
                        out = subprocess.check_output(["uptime"]).decode("utf-8").strip()
                        send_message(f"💻 <b>Mac Status:</b>\n<code>{out}</code>")
                    elif text == "/screen":
                        tmp_screen = "/tmp/telegram_remote_screen.png"
                        subprocess.run(["screencapture", "-x", tmp_screen])
                        send_photo(tmp_screen, caption="📸 Снимок экрана Mac")
                    elif text.startswith("/exec "):
                        cmd = text[6:].strip()
                        send_message(f"⚙️ Выполняю: <code>{cmd}</code>...")
                        try:
                            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
                            out = (res.stdout + "\n" + res.stderr).strip()
                            if len(out) > 3800:
                                out = out[:3800] + "\n... (truncated)"
                            send_message(f"<b>Результат:</b>\n<pre>{out or '(пустой вывод)'}</pre>")
                        except Exception as e:
                            send_message(f"❌ Ошибка: <code>{e}</code>")
                    else:
                        obs_file = os.path.expanduser("~/Documents/Obsidian Vault/Идеи и Правила Стиля.md")
                        try:
                            with open(obs_file, "a", encoding="utf-8") as f:
                                f.write(f"\n- **[{time.strftime('%d.%m %H:%M')} из TG]:** {text}")
                            send_message(f"📝 <b>Сохранено в Obsidian:</b>\n<code>{text}</code>")
                        except Exception as e:
                            send_message(f"🤖 Сообщение принято: <code>{text}</code>")

                # Inline buttons
                elif "callback_query" in upd:
                    cb = upd["callback_query"]
                    cb_data = cb.get("data", "")
                    cb_id = cb["id"]
                    msg_id = cb["message"]["message_id"]
                    chat_id = cb["message"]["chat"]["id"]
                    
                    action, req_id = cb_data.split(":", 1) if ":" in cb_data else (None, None)
                    if req_id in pending_requests:
                        if action == "approve":
                            pending_requests[req_id]["status"] = "approved"
                            api_call("editMessageText", {
                                "chat_id": chat_id,
                                "message_id": msg_id,
                                "text": f"{cb['message']['text']}\n\n<b>✅ РАЗРЕШЕНО (с телефона)</b>",
                                "parse_mode": "HTML"
                            })
                        else:
                            pending_requests[req_id]["status"] = "denied"
                            api_call("editMessageText", {
                                "chat_id": chat_id,
                                "message_id": msg_id,
                                "text": f"{cb['message']['text']}\n\n<b>❌ ОТКЛОНЕНО (с телефона)</b>",
                                "parse_mode": "HTML"
                            })
                        pending_requests[req_id]["event"].set()
                    
                    api_call("answerCallbackQuery", {"callback_query_id": cb_id})
                    
        except Exception as e:
            time.sleep(1)

def start_daemon():
    t = threading.Thread(target=poll_telegram, daemon=True)
    t.start()
    server = HTTPServer(("127.0.0.1", 8765), BridgeHTTPHandler)
    print("[Bridge] HTTP Server listening on http://127.0.0.1:8765")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--daemon":
        start_daemon()
    elif len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        # Send to daemon via HTTP
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8765/ask",
                data=json.dumps({"question": question}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=310) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                if res.get("approved"):
                    print("APPROVED")
                    sys.exit(0)
                else:
                    print("DENIED")
                    sys.exit(1)
        except Exception as e:
            # Fallback if daemon is not running
            approved = ask_approval_sync(question)
            if approved:
                print("APPROVED")
                sys.exit(0)
            else:
                print("DENIED")
                sys.exit(1)
    else:
        start_daemon()
