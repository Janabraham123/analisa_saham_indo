import os
from fastapi import FastAPI, Request
from telegram import Update, Bot
from google import genai

app = FastAPI()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

bot = Bot(token=TELEGRAM_TOKEN)
ai_client = genai.Client(api_key=GEMINI_API_KEY)

TRADING_RULES = """
Peran: Analis Saham Indonesia (IHSG).
Gaya Trading: Swing Trading / Price Action.
Indikator Acuan: Trend (MA20/MA50), Support & Resistance, Volume, RSI.
Format Respon:
1. Ringkasan Tren & Struktur Harga
2. Area Entry Beli (Buy on Weakness / Breakout)
3. Target Profit (TP) & Stop Loss (SL) tegas
4. Catatan Risiko
"""

@app.get("/")
def home():
    return {"message": "Bot Saham Aktif"}

@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, bot)

    if update.message and update.message.text:
        chat_id = update.message.chat_id
        text = update.message.text.strip()

        if text.startswith("/analisa"):
            parts = text.split()
            ticker = parts[1].upper() if len(parts) > 1 else "IHSG"
            
            await bot.send_message(chat_id=chat_id, text=f"Sedang menganalisa saham {ticker}...")

            prompt = f"{TRADING_RULES}\n\nBuat trading plan dan analisis teknikal untuk saham: {ticker} (Bursa Efek Indonesia)."
            
            response = ai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            await bot.send_message(chat_id=chat_id, text=response.text)
        else:
            await bot.send_message(chat_id=chat_id, text="Gunakan format: /analisa <KODE_SAHAM>\nContoh: /analisa BBRI")

    return {"status": "ok"}
