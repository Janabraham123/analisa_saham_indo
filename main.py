import os
from fastapi import FastAPI, Request
from telegram import Update, Bot
from groq import Groq

app = FastAPI()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

bot = Bot(token=TELEGRAM_TOKEN)
client = Groq(api_key=GROQ_API_KEY)

TELEGRAM_MAX_LENGTH = 4096

SYSTEM_INSTRUCTION = """
Peran: Partner dan Asisten Analisis Saham Indonesia (IHSG).
Gaya Komunikasi: Lugas, santai, profesional, dan to-the-point layaknya rekan diskusi trading.
Prinsip Analisis:
1. Prioritaskan Price Action, Support & Resistance, MA20/MA50, RSI, dan Konfirmasi Volume.
2. Selalu tekankan manajemen risiko (Risk-to-Reward Ratio, Stop Loss ketat, dan batas alokasi).
3. Respons pertanyaan umum seputar pasar modal maupun permintaan teknikal ticker saham tertentu secara terstruktur.
"""

def split_message(text: str, max_length: int = TELEGRAM_MAX_LENGTH):
    """Pecah teks panjang jadi beberapa bagian agar muat di batas pesan Telegram."""
    chunks = []
    while len(text) > max_length:
        # Coba potong di baris baru terdekat supaya tidak memotong kalimat
        split_at = text.rfind("\n", 0, max_length)
        if split_at == -1:
            split_at = max_length
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    chunks.append(text)
    return chunks

@app.get("/")
def home():
    return {"status": "Bot Groq Berjalan Aktif"}

@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, bot)

    if update.message and update.message.text:
        chat_id = update.message.chat_id
        user_message = update.message.text.strip()

        try:
            # Kirim request ke Groq API
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_message}
                ],
                model="openai/gpt-oss-120b",
            )

            jawaban = chat_completion.choices[0].message.content

            for chunk in split_message(jawaban):
                await bot.send_message(chat_id=chat_id, text=chunk)

        except Exception as e:
            await bot.send_message(chat_id=chat_id, text=f"⚠️ Kendala sistem: {str(e)}")

    return {"status": "ok"}
