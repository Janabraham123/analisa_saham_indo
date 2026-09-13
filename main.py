import os

from fastapi import FastAPI, Request
from telegram import Update, Bot
from groq import Groq
import yfinance as yf
import pandas as pd

from idx_data_resources import DataSourceManager

app = FastAPI()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

bot = Bot(token=TELEGRAM_TOKEN)
client = Groq(api_key=GROQ_API_KEY)

TELEGRAM_MAX_LENGTH = 4096

# Satu instance dipakai di seluruh bot (cache internal tersimpan di sini).
dsm = DataSourceManager()

SYSTEM_INSTRUCTION = """
Peran: Partner dan Asisten Analisis Saham Indonesia (IHSG).
Gaya Komunikasi: Lugas, santai, profesional, dan to-the-point layaknya rekan diskusi trading.
Prinsip Analisis:
1. Prioritaskan Price Action, Support & Resistance, MA20/MA50, RSI, dan Konfirmasi Volume.
2. Selalu tekankan manajemen risiko (Risk-to-Reward Ratio, Stop Loss ketat, dan batas alokasi).
3. Respons pertanyaan umum seputar pasar modal maupun permintaan teknikal ticker saham tertentu secara terstruktur.
"""

# Watchlist populer untuk fitur market overview (top gainers/losers).
# Yahoo Finance tidak menyediakan endpoint "top movers IDX" langsung,
# jadi kita hitung sendiri dari watchlist saham-saham besar/LQ45.
WATCHLIST = [
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP",
    "ADRO", "PGAS", "PTBA", "ANTM", "INCO", "MDKA", "GOTO", "BUKA",
    "EMTK", "KLBF", "INDF", "SMGR",
]

# Kamus kecil nama perusahaan -> ticker untuk fitur pencarian.
# Cakupan terbatas pada saham-saham populer/LQ45, bukan seluruh 958 saham IDX.
COMPANY_DIRECTORY = {
    "BBCA": "Bank Central Asia",
    "BBRI": "Bank Rakyat Indonesia",
    "BMRI": "Bank Mandiri",
    "BBNI": "Bank Negara Indonesia",
    "TLKM": "Telkom Indonesia",
    "ASII": "Astra International",
    "UNVR": "Unilever Indonesia",
    "ICBP": "Indofood CBP Sukses Makmur",
    "ADRO": "Adaro Energy",
    "PGAS": "Perusahaan Gas Negara",
    "PTBA": "Bukit Asam",
    "ANTM": "Aneka Tambang",
    "INCO": "Vale Indonesia",
    "MDKA": "Merdeka Copper Gold",
    "GOTO": "GoTo Gojek Tokopedia",
    "BUKA": "Bukalapak",
    "EMTK": "Elang Mahkota Teknologi",
    "KLBF": "Kalbe Farma",
    "INDF": "Indofood Sukses Makmur",
    "SMGR": "Semen Indonesia",
    "CPIN": "Charoen Pokphand Indonesia",
    "JPFA": "Japfa Comfeed Indonesia",
    "AKRA": "AKR Corporindo",
    "EXCL": "XL Axiata",
    "ISAT": "Indosat Ooredoo Hutchison",
}


def to_yahoo_ticker(ticker: str) -> str:
    """Tambahkan suffix .JK untuk saham Indonesia jika belum ada."""
    ticker = ticker.strip().upper()
    if not ticker.endswith(".JK"):
        ticker = f"{ticker}.JK"
    return ticker


def fmt_num(n, decimals=2):
    if n is None:
        return "N/A"
    try:
        return f"{n:,.{decimals}f}"
    except (TypeError, ValueError):
        return str(n)


def split_message(text: str, max_length: int = TELEGRAM_MAX_LENGTH):
    """Pecah teks panjang jadi beberapa bagian agar muat di batas pesan Telegram."""
    chunks = []
    while len(text) > max_length:
        split_at = text.rfind("\n", 0, max_length)
        if split_at == -1:
            split_at = max_length
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    chunks.append(text)
    return chunks


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    """Hitung RSI (Relative Strength Index) manual pakai pandas."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None


# ----------------------------------------------------------------------
# FITUR 1: MARKET OVERVIEW
# ----------------------------------------------------------------------
def get_market_overview() -> str:
    ihsg = dsm.yahoo.get_ihsg()
    if ihsg.empty:
        return "⚠️ Gagal mengambil data IHSG saat ini."

    last_close = ihsg["Close"].iloc[-1]
    prev_close = ihsg["Close"].iloc[-2] if len(ihsg) > 1 else last_close
    change = last_close - prev_close
    change_pct = (change / prev_close) * 100 if prev_close else 0
    volume = ihsg["Volume"].iloc[-1]

    # Hitung perubahan harian tiap saham di watchlist untuk top gainers/losers
    movers = []
    for t in WATCHLIST:
        try:
            hist = dsm.yahoo.get_history(t, period="5d")
            if len(hist) < 2:
                continue
            last = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2]
            pct = ((last - prev) / prev) * 100 if prev else 0
            movers.append((t, last, pct))
        except Exception:
            continue

    movers.sort(key=lambda x: x[2], reverse=True)
    top_gainers = movers[:3]
    top_losers = movers[-3:][::-1]

    lines = [
        "📊 *Ringkasan Pasar (IHSG)*",
        f"Terakhir: {fmt_num(last_close)}  ({'+' if change >= 0 else ''}{fmt_num(change)} / {change_pct:+.2f}%)",
        f"Volume: {fmt_num(volume, 0)}",
        "",
        "🟢 Top Gainers (watchlist):",
    ]
    for t, price, pct in top_gainers:
        lines.append(f"  {t}: {fmt_num(price)} ({pct:+.2f}%)")

    lines.append("")
    lines.append("🔴 Top Losers (watchlist):")
    for t, price, pct in top_losers:
        lines.append(f"  {t}: {fmt_num(price)} ({pct:+.2f}%)")

    lines.append("")
    lines.append("_Catatan: top gainers/losers dihitung dari watchlist saham populer, bukan seluruh saham IDX._")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# FITUR 2: INFO SAHAM
#
# PERUBAHAN PENTING: harga sekarang diambil dari YahooFinanceSource
# (yang menarik dari .history(), bukan dari tk.info yang sering
# basi/salah cache untuk saham IDX kurang likuid), lalu disilang-cek
# dengan hasil scraping Google Finance. Kalau selisihnya besar, bot
# kasih peringatan eksplisit alih-alih diam-diam menampilkan angka
# yang salah — inilah yang menyebabkan kasus harga DEWA ~1000-an
# kemarin.
# ----------------------------------------------------------------------
def get_stock_info(ticker: str) -> str:
    info = dsm.yahoo.get_stock_info(ticker)

    if not info or info.price is None:
        return f"⚠️ Data untuk ticker *{ticker.upper()}* tidak ditemukan. Pastikan kode saham benar (contoh: BBCA, TLKM)."

    verified = dsm.get_verified_price(ticker)
    price = verified["price_to_use"] or info.price

    lines = [
        f"📈 *{info.name} ({ticker.upper()})*",
        f"Harga: {fmt_num(price)}"
        + (f"  ({info.change_pct:+.2f}%)" if info.change_pct is not None else ""),
        f"Volume: {fmt_num(info.volume, 0)}",
        f"Market Cap: {fmt_num(info.market_cap, 0)}",
        f"P/E Ratio: {fmt_num(info.pe_ratio)}",
        f"52W High/Low: {fmt_num(info.week52_high)} / {fmt_num(info.week52_low)}",
    ]

    if verified["flagged"]:
        lines.append("")
        lines.append(
            f"⚠️ Harga Yahoo ({fmt_num(verified['yahoo_price'])}) beda "
            f"{verified['diff_pct']:.1f}% dari Google Finance "
            f"({fmt_num(verified['scraped_price'])}). Disarankan cek ulang manual."
        )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# FITUR 3: ANALISIS TEKNIKAL
# ----------------------------------------------------------------------
def get_technical_analysis(ticker: str, period: str = "6mo") -> str:
    hist = dsm.yahoo.get_history(ticker, period=period)

    if hist.empty or len(hist) < 20:
        return f"⚠️ Data historis untuk *{ticker.upper()}* tidak cukup untuk analisis teknikal."

    close = hist["Close"]
    last_price = close.iloc[-1]
    ma20 = close.rolling(window=20).mean().iloc[-1]
    ma50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else None
    rsi = compute_rsi(close)
    avg_volume = hist["Volume"].tail(20).mean()
    last_volume = hist["Volume"].iloc[-1]

    trend_ma20 = "di atas MA20 (bullish jangka pendek)" if last_price > ma20 else "di bawah MA20 (bearish jangka pendek)"
    trend_ma50 = ""
    if ma50 is not None:
        trend_ma50 = "di atas MA50 (bullish jangka menengah)" if last_price > ma50 else "di bawah MA50 (bearish jangka menengah)"

    if rsi is None:
        rsi_note = "N/A"
    elif rsi >= 70:
        rsi_note = f"{rsi:.1f} (overbought)"
    elif rsi <= 30:
        rsi_note = f"{rsi:.1f} (oversold)"
    else:
        rsi_note = f"{rsi:.1f} (netral)"

    volume_note = "di atas rata-rata 20 hari" if last_volume > avg_volume else "di bawah rata-rata 20 hari"

    lines = [
        f"🔍 *Analisis Teknikal: {ticker.upper()}*",
        f"Harga terakhir: {fmt_num(last_price)}",
        f"MA20: {fmt_num(ma20)} → harga {trend_ma20}",
    ]
    if ma50 is not None:
        lines.append(f"MA50: {fmt_num(ma50)} → harga {trend_ma50}")
    lines.append(f"RSI(14): {rsi_note}")
    lines.append(f"Volume: {fmt_num(last_volume, 0)} ({volume_note})")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# FITUR 4: DATA HISTORIS
# ----------------------------------------------------------------------
def get_historical_summary(ticker: str, period: str = "1y") -> str:
    hist = dsm.yahoo.get_history(ticker, period=period)

    if hist.empty:
        return f"⚠️ Data historis untuk *{ticker.upper()}* tidak ditemukan."

    start_price = hist["Close"].iloc[0]
    end_price = hist["Close"].iloc[-1]
    change_pct = ((end_price - start_price) / start_price) * 100
    high = hist["High"].max()
    low = hist["Low"].min()
    avg_volume = hist["Volume"].mean()

    lines = [
        f"📜 *Data Historis: {ticker.upper()}* (periode {period})",
        f"Harga awal: {fmt_num(start_price)}",
        f"Harga akhir: {fmt_num(end_price)}",
        f"Perubahan: {change_pct:+.2f}%",
        f"Tertinggi: {fmt_num(high)}",
        f"Terendah: {fmt_num(low)}",
        f"Rata-rata volume: {fmt_num(avg_volume, 0)}",
        f"Jumlah hari perdagangan: {len(hist)}",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# FITUR 5: BANDINGKAN SAHAM
# ----------------------------------------------------------------------
def compare_stocks(tickers: list, period: str = "1y") -> str:
    if len(tickers) < 2:
        return "⚠️ Masukkan minimal 2 ticker, dipisah koma. Contoh: /bandingkan BBCA,BBRI,BMRI"
    if len(tickers) > 5:
        tickers = tickers[:5]

    lines = [f"⚖️ *Perbandingan Performa* (periode {period})", ""]
    results = []
    for t in tickers:
        hist = dsm.yahoo.get_history(t, period=period)
        if hist.empty:
            lines.append(f"{t.upper()}: data tidak ditemukan")
            continue
        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        change_pct = ((end_price - start_price) / start_price) * 100
        results.append((t.upper(), change_pct))

    results.sort(key=lambda x: x[1], reverse=True)
    for t, pct in results:
        lines.append(f"  {t}: {pct:+.2f}%")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# FITUR 6: CARI SAHAM
# ----------------------------------------------------------------------
def search_stocks(query: str) -> str:
    query_lower = query.strip().lower()
    matches = [
        (ticker, name)
        for ticker, name in COMPANY_DIRECTORY.items()
        if query_lower in ticker.lower() or query_lower in name.lower()
    ]

    if matches:
        lines = [f"🔎 *Hasil pencarian: '{query}'*"]
        for ticker, name in matches:
            lines.append(f"  {ticker} — {name}")
        return "\n".join(lines)

    # Fallback: cocokkan ke kode ticker di seluruh dataset (958 saham),
    # tidak ada nama perusahaan untuk hasil ini karena hanya cocok kode.
    try:
        all_tickers = dsm.github.get_available_stocks()
        code_matches = [t for t in all_tickers if query_lower in t.lower()]
    except Exception:
        code_matches = []

    if code_matches:
        lines = [f"🔎 *Hasil pencarian kode: '{query}'*"]
        for t in code_matches[:20]:
            lines.append(f"  {t}")
        if len(code_matches) > 20:
            lines.append(f"  ... dan {len(code_matches) - 20} lainnya")
        lines.append("")
        lines.append("_Nama perusahaan tidak tersedia untuk hasil ini, hanya cocok kode ticker._")
        return "\n".join(lines)

    return (
        f"⚠️ Tidak ditemukan hasil untuk '{query}'.\n"
        "Kalau kamu tahu kode tickernya, langsung pakai /saham TICKER."
    )


# ----------------------------------------------------------------------
# DETEKSI KONTEKS UNTUK CHAT BEBAS
# ----------------------------------------------------------------------
# Semua ticker yang "dikenal" bot untuk auto-deteksi di chat bebas.
KNOWN_TICKERS = set(WATCHLIST) | set(COMPANY_DIRECTORY.keys())

MARKET_KEYWORDS = ["ihsg", "market", "pasar saham", "pasar modal", "bursa"]


def detect_tickers_in_text(text: str) -> list:
    """Cari kode ticker yang dikenal di dalam teks bebas (mis. 'gimana BBCA hari ini?')."""
    import re

    found = []
    upper_text = text.upper()
    for ticker in KNOWN_TICKERS:
        if re.search(rf"\b{re.escape(ticker)}\b", upper_text):
            found.append(ticker)

    # Juga cocokkan berdasarkan nama perusahaan (mis. 'saham bank central asia')
    text_lower = text.lower()
    for ticker, name in COMPANY_DIRECTORY.items():
        if name.lower() in text_lower and ticker not in found:
            found.append(ticker)

    return found[:2]  # batasi maksimal 2 ticker per pesan biar respons cepat


def build_data_context(text: str) -> str:
    """
    Bangun konteks data real-time berdasarkan isi pesan bebas.
    Kalau pesan menyebut IHSG/pasar dan/atau ticker saham tertentu,
    ambil datanya dari Yahoo Finance untuk dijadikan dasar jawaban LLM.
    """
    context_parts = []
    text_lower = text.lower()

    if any(keyword in text_lower for keyword in MARKET_KEYWORDS):
        try:
            context_parts.append(get_market_overview())
        except Exception:
            pass

    tickers = detect_tickers_in_text(text)
    for ticker in tickers:
        try:
            context_parts.append(get_stock_info(ticker))
        except Exception:
            pass
        try:
            context_parts.append(get_technical_analysis(ticker))
        except Exception:
            pass

    if not context_parts:
        return ""

    return "\n\n".join(context_parts)


# ----------------------------------------------------------------------
# INFO DATASET & DAFTAR SAHAM
# (dipindah ke idx_data_sources.GitHubDatasetSource, dipakai lewat dsm.github)
# ----------------------------------------------------------------------
def get_dataset_info_text() -> str:
    try:
        count = len(dsm.github.get_available_stocks())
    except Exception:
        count = "958 (perkiraan, gagal ambil data terbaru)"

    lines = [
        "🗂️ *Info Dataset*",
        f"Jumlah saham tercakup: {count}",
        "Cakupan historis: 2019 - sekarang",
        "Sumber data historis: wildangunawan/Dataset-Saham-IDX (GitHub, publik)",
        "Sumber data real-time: Yahoo Finance (disilang-cek dengan Google Finance)",
        "Cache internal: 24 jam (dataset) / 5 menit (harga) / 3 menit (scraping)",
    ]
    return "\n".join(lines)


def get_available_stocks_text() -> str:
    try:
        tickers = dsm.github.get_available_stocks()
    except Exception:
        return "⚠️ Gagal mengambil daftar saham saat ini, coba lagi nanti."

    sample = ", ".join(tickers[:30])
    lines = [
        "📋 *Daftar Saham Tersedia*",
        f"Total: {len(tickers)} saham (IDX)",
        "",
        f"Contoh 30 pertama: {sample}",
        "",
        "Gunakan /cari KATA_KUNCI untuk mencari ticker berdasarkan nama perusahaan.",
    ]
    return "\n".join(lines)


def get_sector_performance(sector_query: str) -> str:
    try:
        sectors = dsm.github.get_sectors()
    except Exception:
        return "⚠️ Gagal mengambil daftar sektor saat ini, coba lagi nanti."

    if not sectors:
        return "⚠️ Daftar sektor kosong/tidak ditemukan."

    query_lower = sector_query.strip().lower()
    matched = [s for s in sectors if query_lower in s.lower()]

    if not matched:
        available = ", ".join(sectors)
        return f"⚠️ Sektor '{sector_query}' tidak ditemukan.\nSektor tersedia: {available}"

    sector_name = matched[0]

    try:
        tickers = dsm.github.get_tickers_in_sector(sector_name, limit=10)
    except Exception:
        return f"⚠️ Gagal membaca daftar saham untuk sektor {sector_name}."

    if not tickers:
        return f"⚠️ Tidak ada data ticker yang terbaca untuk sektor {sector_name}."

    changes = []
    for t in tickers:
        try:
            hist = dsm.yahoo.get_history(t, period="5d")
            if len(hist) < 2:
                continue
            last = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2]
            pct = ((last - prev) / prev) * 100 if prev else 0
            changes.append((t, pct))
        except Exception:
            continue

    if not changes:
        return f"⚠️ Gagal mengambil data harga untuk saham-saham di sektor {sector_name}."

    changes.sort(key=lambda x: x[1], reverse=True)
    avg_change = sum(c[1] for c in changes) / len(changes)

    lines = [
        f"🏭 *Performa Sektor: {sector_name}*",
        f"Rata-rata perubahan harian: {avg_change:+.2f}% (sampel {len(changes)} saham)",
        "",
        "🟢 Terbaik:",
    ]
    for t, pct in changes[:3]:
        lines.append(f"  {t}: {pct:+.2f}%")
    lines.append("")
    lines.append("🔴 Terburuk:")
    for t, pct in changes[-3:][::-1]:
        lines.append(f"  {t}: {pct:+.2f}%")
    lines.append("")
    lines.append(f"_Catatan: dihitung dari sampel {len(changes)} saham dalam sektor ini, bukan keseluruhan anggota sektor._")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# HELP TEXT
# ----------------------------------------------------------------------
HELP_TEXT = """
🤖 *Bot Analisa Saham Indonesia*

Perintah yang tersedia:
/market — ringkasan IHSG + top gainers/losers
/saham TICKER — info saham (contoh: /saham BBCA)
/analisa TICKER — analisis teknikal (MA20/MA50, RSI, volume)
/historis TICKER PERIODE — data historis (contoh: /historis BBCA 1y)
/bandingkan TICKER1,TICKER2,... — bandingkan performa saham
/cari KATA_KUNCI — cari ticker dari nama perusahaan atau kode
/sektor NAMA_SEKTOR — performa saham per sektor (contoh: /sektor Energy)
/daftar_saham — jumlah & contoh saham yang tercakup di dataset
/dataset_info — info sumber & cakupan data

Periode yang didukung: 1mo, 3mo, 6mo, 1y, 2y, 5y

Selain perintah di atas, kamu juga bisa chat bebas untuk diskusi atau tanya opini analisis.
"""


# ----------------------------------------------------------------------
# TELEGRAM WEBHOOK
# ----------------------------------------------------------------------
@app.get("/")
def home():
    return {"status": "Bot Groq Berjalan Aktif"}


@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, bot)

    if not (update.message and update.message.text):
        return {"status": "ok"}

    chat_id = update.message.chat_id
    text = update.message.text.strip()

    try:
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            command = parts[0].lower().lstrip("/")
            arg_str = parts[1].strip() if len(parts) > 1 else ""

            if command in ("start", "help"):
                reply = HELP_TEXT

            elif command == "market":
                reply = get_market_overview()

            elif command == "saham":
                if not arg_str:
                    reply = "⚠️ Gunakan format: /saham TICKER (contoh: /saham BBCA)"
                else:
                    reply = get_stock_info(arg_str.split()[0])

            elif command == "analisa":
                if not arg_str:
                    reply = "⚠️ Gunakan format: /analisa TICKER (contoh: /analisa BBCA)"
                else:
                    args = arg_str.split()
                    ticker = args[0]
                    period = args[1] if len(args) > 1 else "6mo"
                    reply = get_technical_analysis(ticker, period)

            elif command == "historis":
                if not arg_str:
                    reply = "⚠️ Gunakan format: /historis TICKER PERIODE (contoh: /historis BBCA 1y)"
                else:
                    args = arg_str.split()
                    ticker = args[0]
                    period = args[1] if len(args) > 1 else "1y"
                    reply = get_historical_summary(ticker, period)

            elif command == "bandingkan":
                if not arg_str:
                    reply = "⚠️ Gunakan format: /bandingkan TICKER1,TICKER2,... (contoh: /bandingkan BBCA,BBRI,BMRI)"
                else:
                    tickers = [t.strip() for t in arg_str.split(",") if t.strip()]
                    reply = compare_stocks(tickers)

            elif command == "cari":
                if not arg_str:
                    reply = "⚠️ Gunakan format: /cari KATA_KUNCI (contoh: /cari bank)"
                else:
                    reply = search_stocks(arg_str)

            elif command == "sektor":
                if not arg_str:
                    reply = "⚠️ Gunakan format: /sektor NAMA_SEKTOR (contoh: /sektor Energy)"
                else:
                    reply = get_sector_performance(arg_str)

            elif command == "daftar_saham":
                reply = get_available_stocks_text()

            elif command == "dataset_info":
                reply = get_dataset_info_text()

            else:
                reply = "Perintah tidak dikenal. Ketik /help untuk daftar perintah."

        else:
            # Chat bebas -> deteksi apakah pesan menyebut ticker/pasar,
            # kalau ya ambil data real-time dan sertakan sebagai konteks ke Groq.
            data_context = build_data_context(text)

            messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

            if data_context:
                messages.append({
                    "role": "system",
                    "content": (
                        "DATA REAL-TIME (diambil dari Yahoo Finance, gunakan ini sebagai dasar "
                        "analisis, jangan mengarang angka lain di luar data berikut):\n\n"
                        f"{data_context}"
                    ),
                })

            messages.append({"role": "user", "content": text})

            chat_completion = client.chat.completions.create(
                messages=messages,
                model="openai/gpt-oss-120b",
            )
            reply = chat_completion.choices[0].message.content

        for chunk in split_message(reply):
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode="Markdown")

    except Exception as e:
        await bot.send_message(chat_id=chat_id, text=f"⚠️ Kendala sistem: {str(e)}")

    return {"status": "ok"}
