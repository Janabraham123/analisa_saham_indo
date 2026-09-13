import os
import re
import time

from fastapi import FastAPI, Request
from telegram import Update, Bot
from groq import Groq
import pandas as pd

from idx_data_resources import DataSourceManager


app = FastAPI()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

bot = Bot(token=TELEGRAM_TOKEN)
client = Groq(api_key=GROQ_API_KEY)

TELEGRAM_MAX_LENGTH = 4096

dsm = DataSourceManager()

SYSTEM_INSTRUCTION = """
Peran:
Kamu adalah Partner dan Asisten Analisis Saham Indonesia (IHSG).

Gaya komunikasi:
- Lugas
- Santai
- Profesional
- To-the-point
- Seperti rekan diskusi pasar modal

Prinsip analisis:
1. Prioritaskan Price Action.
2. Gunakan Support & Resistance.
3. Gunakan MA20 dan MA50.
4. Gunakan RSI.
5. Perhatikan konfirmasi volume.
6. Bedakan dengan jelas antara data harga saat ini,
   data historis, dan hasil analisis teknikal.
7. Jika DATA REAL-TIME diberikan oleh sistem,
   gunakan angka tersebut sebagai sumber utama.
8. Jangan mengarang harga, volume, market cap,
   atau angka teknikal yang tidak diberikan.
9. Jika data tidak tersedia, katakan bahwa data tidak tersedia.
10. Jangan mengklaim memiliki akses data yang sebenarnya tidak diberikan.
"""

WATCHLIST = [
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP",
    "ADRO", "PGAS", "PTBA", "ANTM", "INCO", "MDKA", "GOTO", "BUKA",
    "EMTK", "KLBF", "INDF", "SMGR",
]

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

ALL_TICKERS_CACHE = {"data": set(), "expires_at": 0}
ALL_TICKERS_CACHE_TTL = 24 * 60 * 60


def normalize_ticker(ticker: str) -> str:
    ticker = str(ticker).strip().upper()
    if ticker.endswith(".JK"):
        ticker = ticker[:-3]
    return ticker


def get_known_tickers() -> set:
    now = time.time()
    if ALL_TICKERS_CACHE["data"] and now < ALL_TICKERS_CACHE["expires_at"]:
        return ALL_TICKERS_CACHE["data"]

    known_tickers = set()
    known_tickers.update(normalize_ticker(t) for t in WATCHLIST)
    known_tickers.update(normalize_ticker(t) for t in COMPANY_DIRECTORY.keys())

    try:
        tickers = dsm.github.get_available_stocks()
        if tickers:
            for ticker in tickers:
                normalized = normalize_ticker(ticker)
                if normalized:
                    known_tickers.add(normalized)
    except Exception:
        pass

    ALL_TICKERS_CACHE["data"] = known_tickers
    ALL_TICKERS_CACHE["expires_at"] = now + ALL_TICKERS_CACHE_TTL
    return known_tickers


def fmt_num(n, decimals=2):
    if n is None:
        return "N/A"
    try:
        return f"{n:,.{decimals}f}"
    except (TypeError, ValueError):
        return str(n)


def split_message(text: str, max_length: int = TELEGRAM_MAX_LENGTH):
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
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    value = rsi.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def get_market_overview() -> str:
    ihsg = dsm.yahoo.get_ihsg()
    if ihsg.empty:
        return "⚠️ Gagal mengambil data IHSG saat ini."

    last_close = ihsg["Close"].iloc[-1]
    prev_close = ihsg["Close"].iloc[-2] if len(ihsg) > 1 else last_close
    change = last_close - prev_close
    change_pct = (change / prev_close) * 100 if prev_close else 0
    volume = ihsg["Volume"].iloc[-1]

    movers = []
    for ticker in WATCHLIST:
        try:
            hist = dsm.yahoo.get_history(ticker, period="5d")
            if len(hist) < 2:
                continue
            last = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2]
            pct = ((last - prev) / prev) * 100 if prev else 0
            movers.append((ticker, last, pct))
        except Exception:
            continue

    movers.sort(key=lambda x: x[2], reverse=True)
    top_gainers = movers[:3]
    top_losers = movers[-3:][::-1]

    lines = [
        "📊 *Ringkasan Pasar (IHSG)*",
        f"Terakhir: {fmt_num(last_close)} ({'+' if change >= 0 else ''}{fmt_num(change)} / {change_pct:+.2f}%)",
        f"Volume: {fmt_num(volume, 0)}",
        "",
        "🟢 Top Gainers (watchlist):",
    ]
    for ticker, price, pct in top_gainers:
        lines.append(f"  {ticker}: {fmt_num(price)} ({pct:+.2f}%)")
    lines.append("")
    lines.append("🔴 Top Losers (watchlist):")
    for ticker, price, pct in top_losers:
        lines.append(f"  {ticker}: {fmt_num(price)} ({pct:+.2f}%)")
    lines.append("")
    lines.append("_Catatan: top gainers/losers dihitung dari watchlist saham populer, bukan seluruh saham IDX._")
    return "\n".join(lines)


def get_stock_info(ticker: str) -> str:
    ticker = normalize_ticker(ticker)
    info = dsm.yahoo.get_stock_info(ticker)

    if not info or info.price is None:
        return f"⚠️ Data untuk ticker *{ticker}* tidak ditemukan.\n\nPastikan kode saham benar."

    verified = dsm.get_verified_price(ticker)
    price = verified["price_to_use"] or info.price

    lines = [
        f"📈 *{info.name} ({ticker})*",
        f"Harga: {fmt_num(price)}" + (f"  ({info.change_pct:+.2f}%)" if info.change_pct is not None else ""),
        f"Volume: {fmt_num(info.volume, 0)}",
        f"Market Cap: {fmt_num(info.market_cap, 0)}",
        f"P/E Ratio: {fmt_num(info.pe_ratio)}",
        f"52W High/Low: {fmt_num(info.week52_high)} / {fmt_num(info.week52_low)}",
    ]

    if verified["flagged"]:
        lines.append("")
        lines.append(
            f"⚠️ Harga Yahoo ({fmt_num(verified['yahoo_price'])}) beda {verified['diff_pct']:.1f}% "
            f"dari Google Finance ({fmt_num(verified['scraped_price'])})."
        )
        lines.append("Disarankan melakukan pengecekan manual.")

    return "\n".join(lines)


def get_technical_analysis(ticker: str, period: str = "6mo") -> str:
    ticker = normalize_ticker(ticker)
    hist = dsm.yahoo.get_history(ticker, period=period)

    if hist.empty or len(hist) < 20:
        return f"⚠️ Data historis untuk *{ticker}* tidak cukup untuk analisis teknikal."

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
        f"🔍 *Analisis Teknikal: {ticker}*",
        f"Harga terakhir: {fmt_num(last_price)}",
        f"MA20: {fmt_num(ma20)} → harga {trend_ma20}",
    ]
    if ma50 is not None:
        lines.append(f"MA50: {fmt_num(ma50)} → harga {trend_ma50}")
    lines.append(f"RSI(14): {rsi_note}")
    lines.append(f"Volume: {fmt_num(last_volume, 0)} ({volume_note})")
    return "\n".join(lines)


def get_historical_summary(ticker: str, period: str = "1y") -> str:
    ticker = normalize_ticker(ticker)
    hist = dsm.yahoo.get_history(ticker, period=period)

    if hist.empty:
        return f"⚠️ Data historis untuk *{ticker}* tidak ditemukan."

    start_price = hist["Close"].iloc[0]
    end_price = hist["Close"].iloc[-1]
    change_pct = ((end_price - start_price) / start_price) * 100
    high = hist["High"].max()
    low = hist["Low"].min()
    avg_volume = hist["Volume"].mean()

    lines = [
        f"📜 *Data Historis: {ticker}* (periode {period})",
        f"Harga awal: {fmt_num(start_price)}",
        f"Harga akhir: {fmt_num(end_price)}",
        f"Perubahan: {change_pct:+.2f}%",
        f"Tertinggi: {fmt_num(high)}",
        f"Terendah: {fmt_num(low)}",
        f"Rata-rata volume: {fmt_num(avg_volume, 0)}",
        f"Jumlah hari perdagangan: {len(hist)}",
    ]
    return "\n".join(lines)


def compare_stocks(tickers: list, period: str = "1y") -> str:
    if len(tickers) < 2:
        return "⚠️ Masukkan minimal 2 ticker, dipisah koma.\n\nContoh:\n/bandingkan BBCA,BBRI,BMRI"
    if len(tickers) > 5:
        tickers = tickers[:5]

    lines = [f"⚖️ *Perbandingan Performa* (periode {period})", ""]
    results = []
    for ticker in tickers:
        ticker = normalize_ticker(ticker)
        hist = dsm.yahoo.get_history(ticker, period=period)
        if hist.empty:
            lines.append(f"{ticker}: data tidak ditemukan")
            continue
        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        change_pct = ((end_price - start_price) / start_price) * 100
        results.append((ticker, change_pct))

    results.sort(key=lambda x: x[1], reverse=True)
    for ticker, pct in results:
        lines.append(f"  {ticker}: {pct:+.2f}%")
    return "\n".join(lines)


def search_stocks(query: str) -> str:
    query_lower = query.strip().lower()
    matches = [
        (ticker, name) for ticker, name in COMPANY_DIRECTORY.items()
        if query_lower in ticker.lower() or query_lower in name.lower()
    ]

    if matches:
        lines = [f"🔎 *Hasil pencarian: '{query}'*"]
        for ticker, name in matches:
            lines.append(f"  {ticker} — {name}")
        return "\n".join(lines)

    try:
        all_tickers = dsm.github.get_available_stocks()
        code_matches = [normalize_ticker(t) for t in all_tickers if query_lower in normalize_ticker(t).lower()]
    except Exception:
        code_matches = []

    if code_matches:
        lines = [f"🔎 *Hasil pencarian kode: '{query}'*"]
        for ticker in code_matches[:20]:
            lines.append(f"  {ticker}")
        if len(code_matches) > 20:
            lines.append(f"  ... dan {len(code_matches) - 20} lainnya")
        lines.append("")
        lines.append("_Nama perusahaan tidak tersedia untuk hasil ini, hanya cocok kode ticker._")
        return "\n".join(lines)

    return f"⚠️ Tidak ditemukan hasil untuk '{query}'.\n\nKalau kamu tahu kode tickernya, langsung pakai /saham TICKER."


def detect_tickers_in_text(text: str) -> list:
    found = []
    upper_text = text.upper()
    known_tickers = get_known_tickers()

    for ticker in known_tickers:
        if not ticker:
            continue
        pattern = rf"(?<![A-Z0-9]){re.escape(ticker)}(?:\.JK)?(?![A-Z0-9])"
        if re.search(pattern, upper_text):
            if ticker not in found:
                found.append(ticker)

    text_lower = text.lower()
    for ticker, name in COMPANY_DIRECTORY.items():
        if name.lower() in text_lower and ticker not in found:
            found.append(ticker)

    return found[:2]


def build_data_context(text: str) -> str:
    context_parts = []
    text_lower = text.lower()

    market_keywords = ["ihsg", "market", "pasar saham", "pasar modal", "bursa"]
    if any(keyword in text_lower for keyword in market_keywords):
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


def get_dataset_info_text() -> str:
    try:
        count = len(dsm.github.get_available_stocks())
    except Exception:
        count = "958 (perkiraan, gagal mengambil data terbaru)"

    lines = [
        "🗂️ *Info Dataset*",
        f"Jumlah saham tercakup: {count}",
        "Cakupan historis: 2019 - sekarang",
        "Sumber data historis: wildangunawan/Dataset-Saham-IDX (GitHub, publik)",
        "Sumber harga saat ini: Yahoo Finance",
        "Verifikasi tambahan: Google Finance",
        "Cache internal: 24 jam (dataset) / 60 detik (harga) / 3 menit (scraping)",
    ]
    return "\n".join(lines)


def get_available_stocks_text() -> str:
    try:
        tickers = dsm.github.get_available_stocks()
    except Exception:
        return "⚠️ Gagal mengambil daftar saham saat ini, coba lagi nanti."

    sample = ", ".join(normalize_ticker(t) for t in tickers[:30])
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
    matched = [sector for sector in sectors if query_lower in sector.lower()]

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
    for ticker in tickers:
        ticker = normalize_ticker(ticker)
        try:
            hist = dsm.yahoo.get_history(ticker, period="5d")
            if len(hist) < 2:
                continue
            last = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2]
            pct = ((last - prev) / prev) * 100 if prev else 0
            changes.append((ticker, pct))
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
    for ticker, pct in changes[:3]:
        lines.append(f"  {ticker}: {pct:+.2f}%")
    lines.append("")
    lines.append("🔴 Terburuk:")
    for ticker, pct in changes[-3:][::-1]:
        lines.append(f"  {ticker}: {pct:+.2f}%")
    lines.append("")
    lines.append(f"_Catatan: dihitung dari sampel {len(changes)} saham dalam sektor ini, bukan keseluruhan anggota sektor._")
    return "\n".join(lines)


HELP_TEXT = """
🤖 *Bot Analisa Saham Indonesia*

Perintah yang tersedia:

/market — Ringkasan IHSG + top gainers/losers
/saham TICKER — Info saham (contoh: /saham BBCA)
/analisa TICKER — Analisis teknikal MA20/MA50, RSI, volume
/historis TICKER PERIODE — Data historis (contoh: /historis BBCA 1y)
/bandingkan TICKER1,TICKER2,... — Bandingkan performa saham
/cari KATA_KUNCI — Cari ticker dari nama perusahaan atau kode
/sektor NAMA_SEKTOR — Performa saham per sektor
/daftar_saham — Jumlah & contoh saham yang tercakup
/dataset_info — Info sumber & cakupan data

Periode yang didukung: 1mo, 3mo, 6mo, 1y, 2y, 5y

Selain perintah di atas, kamu juga bisa chat bebas untuk diskusi dan analisis saham.
"""


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
                    reply = "⚠️ Gunakan format:\n/saham TICKER\n\nContoh:\n/saham BBCA"
                else:
                    reply = get_stock_info(arg_str.split()[0])
            elif command == "analisa":
                if not arg_str:
                    reply = "⚠️ Gunakan format:\n/analisa TICKER\n\nContoh:\n/analisa BBCA"
                else:
                    args = arg_str.split()
                    ticker = args[0]
                    period = args[1] if len(args) > 1 else "6mo"
                    reply = get_technical_analysis(ticker, period)
            elif command == "historis":
                if not arg_str:
                    reply = "⚠️ Gunakan format:\n/historis TICKER PERIODE\n\nContoh:\n/historis BBCA 1y"
                else:
                    args = arg_str.split()
                    ticker = args[0]
                    period = args[1] if len(args) > 1 else "1y"
                    reply = get_historical_summary(ticker, period)
            elif command == "bandingkan":
                if not arg_str:
                    reply = "⚠️ Gunakan format:\n/bandingkan TICKER1,TICKER2,...\n\nContoh:\n/bandingkan BBCA,BBRI,BMRI"
                else:
                    tickers = [t.strip() for t in arg_str.split(",") if t.strip()]
                    reply = compare_stocks(tickers)
            elif command == "cari":
                if not arg_str:
                    reply = "⚠️ Gunakan format:\n/cari KATA_KUNCI\n\nContoh:\n/cari bank"
                else:
                    reply = search_stocks(arg_str)
            elif command == "sektor":
                if not arg_str:
                    reply = "⚠️ Gunakan format:\n/sektor NAMA_SEKTOR\n\nContoh:\n/sektor Energy"
                else:
                    reply = get_sector_performance(arg_str)
            elif command == "daftar_saham":
                reply = get_available_stocks_text()
            elif command == "dataset_info":
                reply = get_dataset_info_text()
            else:
                reply = "Perintah tidak dikenal.\n\nKetik /help untuk daftar perintah."
        else:
            data_context = build_data_context(text)
            messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

            if data_context:
                messages.append({
                    "role": "system",
                    "content": (
                        "DATA SAHAM TERKINI/HISTORIS YANG BERHASIL DIAMBIL OLEH SISTEM:\n\n"
                        f"{data_context}\n\n"
                        "Gunakan data di atas sebagai dasar jawaban.\n"
                        "Jangan mengarang angka harga, volume, market cap, RSI, MA20, atau MA50 "
                        "yang tidak terdapat dalam data."
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
        error_message = f"⚠️ Kendala sistem: {str(e)}"
        await bot.send_message(chat_id=chat_id, text=error_message)

    return {"status": "ok"}
