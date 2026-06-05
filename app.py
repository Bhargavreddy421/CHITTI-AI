from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import requests
import json
import os
import uuid
from io import BytesIO
from pathlib import Path
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

# load .env if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
except Exception:
    # python-dotenv not installed; environment variables may still be set externally
    pass

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.secret_key = os.getenv("FLASK_SECRET_KEY", "chitti-ai-super-secret-key-12345")

FILE = "chats.json"
USERS_FILE = "users.json"
MAX_PDF_TEXT_LENGTH = 8000

# ✅ API KEY (PREFER ENV VAR)
# Load from environment only. Do NOT keep real keys in source.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip() or None

URL = "https://api.groq.com/openai/v1/chat/completions"
PRIMARY_MODEL = "llama-3.1-8b-instant"
FALLBACK_MODEL = "llama-3.3-70b-versatile"


# ---------------- FILE HANDLING & AUTH HELPERS ----------------

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users):
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)
    except Exception as e:
        print("Error saving users:", e)


def load_chats():
    if not os.path.exists(FILE):
        return {}
    try:
        with open(FILE, "r") as f:
            data = json.load(f)
            if not data:
                return {}
            # Detect legacy flat schema (chat_id mapping to array)
            first_val = next(iter(data.values()))
            if isinstance(first_val, list):
                # Migrate to nested user-partitioned structure
                migrated = {"legacy_user": data}
                with open(FILE, "w") as f_out:
                    json.dump(migrated, f_out, indent=4)
                return migrated
            return data
    except Exception:
        return {}


def save_chats(data):
    try:
        with open(FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print("Error saving chats:", e)


def load_user_chats(username):
    chats = load_chats()
    return chats.get(username, {})


def save_user_chats(username, user_chats):
    chats = load_chats()
    chats[username] = user_chats
    save_chats(chats)


# Compatibility wrappers for older code that calls `load()` / `save()`
def load():
    return load_chats()


def save(data):
    return save_chats(data)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "username" not in session:
            # If it's an API request, return 401 JSON
            if request.path.startswith("/chat") or request.path in ["/new_chat", "/get_chats", "/upload_pdf", "/delete_chat", "/all_chats", "/clear_all"]:
                return jsonify({"reply": "Unauthorized. Please log in."}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def extract_pdf_text(file_storage):
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support is not installed. Add pypdf to requirements.txt and redeploy.") from exc

    pdf_bytes = file_storage.read()
    if not pdf_bytes:
        raise ValueError("Uploaded PDF is empty.")

    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(f"Page {index}:\n{text}")

    extracted_text = "\n\n".join(pages).strip()
    if not extracted_text:
        raise ValueError("Could not extract readable text from this PDF.")

    return extracted_text[:MAX_PDF_TEXT_LENGTH]


def prepare_history_for_api(history, max_chars=25000):
    # If the history is small enough, return as-is
    total_len = sum(len(m.get("content", "")) for m in history)
    if total_len <= max_chars:
        return history

    # We need to prune
    first_msg = history[0] if history else None
    
    keep_first = False
    if first_msg and ("PDF text:" in first_msg.get("content", "") or "I uploaded a PDF" in first_msg.get("content", "")):
        keep_first = True

    pruned = []
    current_len = 0
    
    first_msg_len = len(first_msg.get("content", "")) if keep_first else 0
    limit = max_chars - first_msg_len

    # Loop in reverse (newest first)
    msgs_to_process = history[1:] if keep_first else history
    for idx, msg in enumerate(reversed(msgs_to_process)):
        msg_len = len(msg.get("content", ""))
        if idx > 0 and current_len + msg_len > limit:
            break
        pruned.insert(0, msg)
        current_len += msg_len

    if keep_first:
        pruned.insert(0, first_msg)

    return pruned


def get_ai_reply(history):
    if not GROQ_API_KEY:
        return "API key not set", 500

    prepared_history = prepare_history_for_api(history)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": PRIMARY_MODEL,
        "messages": prepared_history,
        "temperature": 0.7
    }

    try:
        res = requests.post(URL, headers=headers, json=payload, timeout=60)
    except requests.RequestException as exc:
        return f"Could not connect to Groq: {exc}", 502

    print("STATUS:", res.status_code)
    print("RESPONSE:", res.text)

    if res.status_code != 200:
        print("Fallback to 70b...")
        payload["model"] = FALLBACK_MODEL
        try:
            res = requests.post(URL, headers=headers, json=payload, timeout=60)
        except requests.RequestException as exc:
            return f"Could not connect to Groq: {exc}", 502

        print("FALLBACK STATUS:", res.status_code)
        print("FALLBACK RESPONSE:", res.text)

        if res.status_code != 200:
            try:
                error_data = res.json()
                message = error_data.get("error", {}).get("message") or res.text
            except ValueError:
                message = res.text
            return f"Groq error: {message}", res.status_code

    data = res.json()
    return data["choices"][0]["message"]["content"], 200


# ---------------- ROUTES ----------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("home"))
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        username = data.get("username", "").strip()
        password = data.get("password", "")
        users = load_users()
        user = users.get(username)
        if user and check_password_hash(user["password"], password):
            session["username"] = username
            return jsonify({"success": True, "username": username})
        return jsonify({"success": False, "message": "Invalid username or password."}), 401
    return render_template("login.html")


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(force=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required."}), 400
    if len(username) < 3:
        return jsonify({"success": False, "message": "Username must be at least 3 characters."}), 400
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters."}), 400
    users = load_users()
    if username in users:
        return jsonify({"success": False, "message": "Username already taken. Choose another."}), 409
    users[username] = {
        "password": generate_password_hash(password),
        "created_at": __import__("datetime").datetime.utcnow().isoformat()
    }
    save_users(users)
    session["username"] = username
    return jsonify({"success": True, "username": username})


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    return render_template("index.html", username=session.get("username", "User"))


@app.route("/history_page")
@login_required
def history_page():
    return render_template("history.html", username=session.get("username", "User"))


@app.route("/new_chat", methods=["POST"])
@login_required
def new_chat():
    username = session["username"]
    user_chats = load_user_chats(username)
    chat_id = str(uuid.uuid4())
    user_chats[chat_id] = []
    save_user_chats(username, user_chats)
    return jsonify({"chat_id": chat_id})


@app.route("/get_chats")
@login_required
def get_chats():
    return jsonify(load_user_chats(session["username"]))


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    try:
        username = session["username"]
        data = request.get_json()
        try:
            print("CHAT REQUEST:", {"username": username, "remote": request.remote_addr, "path": request.path, "json": data})
        except Exception:
            print("CHAT REQUEST: (failed to print request info)")

        msg = data.get("message")
        chat_id = data.get("chat_id")

        if not msg:
            return jsonify({"reply": "Empty message"}), 400

        user_chats = load_user_chats(username)

        if not chat_id or chat_id not in user_chats:
            chat_id = str(uuid.uuid4())
            user_chats[chat_id] = []

        history = user_chats[chat_id]

        history.append({"role": "user", "content": msg})

        reply, status_code = get_ai_reply(history)
        if status_code != 200:
            return jsonify({"reply": reply}), status_code

        history.append({"role": "assistant", "content": reply})
        user_chats[chat_id] = history
        save_user_chats(username, user_chats)

        return jsonify({"reply": reply, "chat_id": chat_id})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"reply": "Server error"}), 500


@app.route("/upload_pdf", methods=["POST"])
@login_required
def upload_pdf():
    try:
        username = session["username"]
        uploaded_file = request.files.get("pdf")
        chat_id = request.form.get("chat_id")

        if not uploaded_file:
            return jsonify({"reply": "Please choose a PDF file first."}), 400

        filename = uploaded_file.filename or "document.pdf"
        if not filename.lower().endswith(".pdf"):
            return jsonify({"reply": "Only PDF files are supported."}), 400

        extracted_text = extract_pdf_text(uploaded_file)

        user_chats = load_user_chats(username)
        if not chat_id or chat_id not in user_chats:
            chat_id = str(uuid.uuid4())
            user_chats[chat_id] = []

        prompt = (
            f"I uploaded a PDF named '{filename}'. Extract the important points and explain it in a clear, simple way. "
            "Include a short summary first, then key points, and mention anything important the reader should notice.\n\n"
            f"PDF text:\n{extracted_text}"
        )

        history = user_chats[chat_id]
        history.append({"role": "user", "content": prompt})

        reply, status_code = get_ai_reply(history)
        if status_code != 200:
            return jsonify({"reply": reply}), status_code

        history.append({"role": "assistant", "content": reply})
        user_chats[chat_id] = history
        save_user_chats(username, user_chats)

        return jsonify({
            "chat_id": chat_id,
            "reply": reply,
            "filename": filename,
            "extracted_preview": extracted_text[:800]
        })

    except ValueError as e:
        return jsonify({"reply": str(e)}), 400
    except Exception as e:
        print("PDF ERROR:", str(e))
        return jsonify({"reply": "Error reading PDF file"}), 500


@app.route("/delete_chat", methods=["POST"])
@login_required
def delete_chat():
    username = session["username"]
    chat_id = request.json.get("chat_id")
    user_chats = load_user_chats(username)
    user_chats.pop(chat_id, None)
    save_user_chats(username, user_chats)
    return jsonify({"status": "deleted"})


@app.route("/all_chats")
@login_required
def all_chats():
    return jsonify(load_user_chats(session["username"]))


@app.route("/clear_all", methods=["POST"])
@login_required
def clear_all():
    save_user_chats(session["username"], {})
    return jsonify({"status": "cleared"})


# ---------------- RUN ----------------

if __name__ == "__main__":
    print("[*] Server starting...")

    if not GROQ_API_KEY:
        print("[ERROR] GROQ_API_KEY not set!")
    else:
        print("[OK] API Key loaded successfully")

    app.run(host="0.0.0.0", port=5000, debug=True)
    # Trigger auto-reload
