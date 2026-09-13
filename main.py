import os
from fastapi import FastAPI, Request
from telegram import Update, Bot
from google import genai

app = FastAPI()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

bot = Bot(token=TELEGRAM_TOKEN)
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Persona dan aturan diskusi saham Anda
SYSTEM_INSTRUCTION = """
Kamu adalah partner diskusi dan asisten trading saham profesional di Bursa Efek Indonesia (IHSG).
Gaya bicaramu santai, to the point, objektif, dan suportif layaknya rekan diskusi trading.
Prinsip analisis:
- Fokus pada Price Action, Support & Resistance, Moving Average (MA20, MA50, MA200), RSI, dan Volume.
- Selalu utamakan risk management (Risk-to-Reward Ratio, Stop Loss, dan batas alokasi modal).
- Jawab pertanyaan umum maupun analisis saham secara fleksibel dan lugas.
"""

@app.get("/")
def home():
    return {"status": "Bot Aktif"}

@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, bot)

    if update.message and update.message.text:
        chat_id = update.message.chat_id
        user_message = update.message.text.strip()

        # Bot langsung memproses semua teks yang masuk sebagai obrolan
        prompt = f"{SYSTEM_INSTRUCTION}\n\nPertanyaan/Diskusi User: {user_message}"

        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        await bot.send_message(chat_id=chat_id, text=response.text)

    return {"status": "ok"}
