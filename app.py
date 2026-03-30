from flask import Flask, request, jsonify, render_template
import requests
import json
import os
import uuid
from pathlib import Path

# load .env if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except Exception:
    # python-dotenv not installed; environment variables may still be set externally
    pass

app = Flask(__name__)

FILE = "chats.json"

# ✅ API KEY (PREFER ENV VAR)
# Load from environment only. Do NOT keep real keys in source.
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or None

URL = "https://api.groq.com/openai/v1/chat/completions"


# ---------------- FILE HANDLING ----------------

def load():
    if not os.path.exists(FILE):
        return {}
    with open(FILE, "r") as f:
        return json.load(f)


def save(data):
    with open(FILE, "w") as f:
        json.dump(data, f, indent=4)


# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/history_page")
def history_page():
    return render_template("history.html")


@app.route("/new_chat", methods=["POST"])
def new_chat():
    chats = load()
    chat_id = str(uuid.uuid4())
    chats[chat_id] = []
    save(chats)
    return jsonify({"chat_id": chat_id})


@app.route("/get_chats")
def get_chats():
    return jsonify(load())


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        msg = data.get("message")
        chat_id = data.get("chat_id")

        if not msg:
            return jsonify({"reply": "Empty message"}), 400

        if not GROQ_API_KEY:
            return jsonify({"reply": "API key not set"}), 500

        chats = load()

        if chat_id not in chats:
            chats[chat_id] = []

        history = chats[chat_id]

        # ✅ Add user message
        history.append({
            "role": "user",
            "content": msg
        })

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        # ✅ MAIN MODEL
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": history,
            "temperature": 0.7
        }

        res = requests.post(URL, headers=headers, json=payload)

        print("STATUS:", res.status_code)
        print("RESPONSE:", res.text)

        # 🔁 FALLBACK MODEL
        if res.status_code != 200:
            print("⚠️ fallback to 70b...")
            payload["model"] = "llama-3.1-70b-versatile"
            res = requests.post(URL, headers=headers, json=payload)

            print("FALLBACK STATUS:", res.status_code)
            print("FALLBACK RESPONSE:", res.text)

            if res.status_code != 200:
                return jsonify({"reply": res.text}), 500

        # ✅ Parse response
        data = res.json()
        reply = data["choices"][0]["message"]["content"]

        # ✅ Save assistant reply
        history.append({
            "role": "assistant",
            "content": reply
        })

        chats[chat_id] = history
        save(chats)

        return jsonify({"reply": reply})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"reply": "Server error"}), 500


@app.route("/delete_chat", methods=["POST"])
def delete_chat():
    chat_id = request.json.get("chat_id")
    chats = load()
    chats.pop(chat_id, None)
    save(chats)
    return jsonify({"status": "deleted"})


@app.route("/all_chats")
def all_chats():
    return jsonify(load())


@app.route("/clear_all", methods=["POST"])
def clear_all():
    save({})
    return jsonify({"status": "cleared"})


# ---------------- RUN ----------------

if __name__ == "__main__":
    print("🚀 Server starting...")

    if not GROQ_API_KEY:
        print("❌ ERROR: GROQ_API_KEY not set!")
    else:
        print("✅ API Key loaded successfully")

    app.run(host="0.0.0.0", port=5000, debug=True)