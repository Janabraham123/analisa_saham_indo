"""
idx_data_sources.py

Port Python dari arsitektur data-source di baguskto/saham-mcp
(https://github.com/baguskto/saham-mcp, MIT License) untuk data
saham IDX, disesuaikan supaya bisa dipakai di dalam bot
analisa_saham_indo (main.py).

Tiga data source, mengikuti pola prioritas yang sama seperti versi
TypeScript aslinya:

  1. GitHubDatasetSource  - daftar ticker & sektor + data historis
                             (wildangunawan/Dataset-Saham-IDX)
  2. YahooFinanceSource   - harga & OHLCV via yfinance
  3. WebScrapingSource    - scraping Google Finance, dipakai sebagai
                             pembanding/silang-cek harga

CATATAN PENTING soal WebScrapingSource:
Di repo TypeScript aslinya, fungsi scraping untuk info saham individual
TIDAK PERNAH benar-benar diimplementasikan - cuma stub yang selalu
`return null` (ada komentar eksplisit "not fully implemented" di kode
sumbernya). Kalau cuma di-porting apa adanya, masalah harga yang salah
TIDAK akan terselesaikan, karena root cause-nya (ketergantungan penuh
ke Yahoo Finance yang kurang akurat untuk saham IDX kurang likuid)
tetap sama.

Port ini BENAR-BENAR mengimplementasikan scraping tsb, supaya bisa
dipakai DataSourceManager.get_verified_price() untuk mendeteksi kalau
harga dari Yahoo melenceng jauh dari sumber lain - bukan cuma diam-diam
menampilkan angka yang salah.
"""

import re
import time
import logging
import urllib.parse
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import requests
import pandas as pd
import yfinance as yf

logger = logging.getLogger("idx_data_sources")


# ---------------------------------------------------------------------
# Cache TTL sederhana di memori (setara cache/index.ts di repo asli)
# ---------------------------------------------------------------------
class TTLCache:
    def __init__(self):
        self._store: Dict[str, tuple] = {}

    def get(self, key: str):
        entry = self._store.get(key)
        if not entry:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value, ttl_seconds: float):
        self._store[key] = (value, time.time() + ttl_seconds)

    def clear(self):
        self._store.clear()


# ---------------------------------------------------------------------
# 1. GitHubDatasetSource - daftar ticker & sektor (958 saham, historis)
#    (logika sama seperti yang sudah ada di main.py Anda sekarang,
#    cuma dirapikan jadi class biar sejalan dengan struktur repo asli)
# ---------------------------------------------------------------------
class GitHubDatasetSource:
    API_BASE = "https://api.github.com/repos/wildangunawan/Dataset-Saham-IDX/contents"
    RAW_BASE = "https://raw.githubusercontent.com/wildangunawan/Dataset-Saham-IDX/master"
    CACHE_TTL = 24 * 60 * 60  # 24 jam, sama seperti repo asli

    def __init__(self, cache: TTLCache):
        self.cache = cache

    def get_available_stocks(self) -> List[str]:
        cached = self.cache.get("gh:available_stocks")
        if cached is not None:
            return cached
        resp = requests.get(f"{self.API_BASE}/Saham/Semua", timeout=15)
        resp.raise_for_status()
        files = resp.json()
        tickers = sorted(
            f["name"].replace(".csv", "") for f in files if f["name"].endswith(".csv")
        )
        self.cache.set("gh:available_stocks", tickers, self.CACHE_TTL)
        return tickers

    def get_sectors(self) -> List[str]:
        cached = self.cache.get("gh:sectors")
        if cached is not None:
            return cached
        resp = requests.get(f"{self.API_BASE}/List%20Emiten/Sectors", timeout=15)
        resp.raise_for_status()
        files = resp.json()
        sectors = sorted(
            f["name"].replace(".csv", "") for f in files if f["name"].endswith(".csv")
        )
        self.cache.set("gh:sectors", sectors, self.CACHE_TTL)
        return sectors

    def get_tickers_in_sector(self, sector_name: str, limit: int = 10) -> List[str]:
        url = f"{self.RAW_BASE}/List%20Emiten/Sectors/{urllib.parse.quote(sector_name)}.csv"
        df = pd.read_csv(url)
        for col in ["Kode", "kode", "Code", "code", "Ticker", "ticker", "Kode Saham", "Symbol"]:
            if col in df.columns:
                return df[col].astype(str).str.strip().str.upper().tolist()[:limit]
        for col in df.columns:
            sample = df[col].astype(str).head(10)
            if sample.str.match(r"^[A-Z]{4}$").sum() >= 5:
                return df[col].astype(str).str.strip().str.upper().tolist()[:limit]
        return []


# ---------------------------------------------------------------------
# 2. YahooFinanceSource - harga & OHLCV
#
# Perbedaan penting dari main.py Anda saat ini: harga SELALU diambil
# dari .history() (candle harian), BUKAN dari tk.info. Field .info
# ("currentPrice"/"regularMarketPrice") itu yang sering basi/salah
# cache untuk saham kecil/kurang likuid di luar AS - termasuk banyak
# saham IDX. .info di sini cuma dipakai untuk metadata (nama, market
# cap, PE ratio), bukan untuk angka harga.
# ---------------------------------------------------------------------
@dataclass
class StockInfo:
    ticker: str
    name: str
    price: Optional[float]
    prev_close: Optional[float]
    change_pct: Optional[float]
    volume: Optional[float]
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    week52_high: Optional[float]
    week52_low: Optional[float]
    source: str = "yahoo_history"


class YahooFinanceSource:
    CACHE_TTL = 5 * 60  # 5 menit, sama seperti repo asli

    def __init__(self, cache: TTLCache):
        self.cache = cache

    @staticmethod
    def to_yahoo_ticker(ticker: str) -> str:
        ticker = ticker.strip().upper()
        return ticker if ticker.endswith(".JK") else f"{ticker}.JK"

    def get_stock_info(self, ticker: str) -> Optional[StockInfo]:
        cache_key = f"yf:info:{ticker.upper()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        yt = self.to_yahoo_ticker(ticker)
        tk = yf.Ticker(yt)

        info: Dict[str, Any] = {}
        try:
            info = tk.info or {}
        except Exception:
            pass  # .info boleh gagal, harga tetap diambil dari history()

        hist = tk.history(period="5d")
        if hist.empty:
            return None

        price = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else None
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else None
        volume = float(hist["Volume"].iloc[-1])

        result = StockInfo(
            ticker=ticker.upper(),
            name=info.get("longName", ticker.upper()),
            price=price,
            prev_close=prev_close,
            change_pct=change_pct,
            volume=volume,
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            week52_high=info.get("fiftyTwoWeekHigh"),
            week52_low=info.get("fiftyTwoWeekLow"),
        )
        self.cache.set(cache_key, result, self.CACHE_TTL)
        return result

    def get_history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        return yf.Ticker(self.to_yahoo_ticker(ticker)).history(period=period)

    def get_ihsg(self) -> pd.DataFrame:
        return yf.Ticker("^JKSE").history(period="5d")


# ---------------------------------------------------------------------
# 3. WebScrapingSource - fallback / pembanding harga
#
# Ini bagian yang di repo TypeScript asli CUMA STUB (selalu None).
# Di sini benar-benar di-scrape dari Google Finance sebagai sumber
# kedua yang independen dari Yahoo, khusus untuk verifikasi harga.
# ---------------------------------------------------------------------
class WebScrapingSource:
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    CACHE_TTL = 3 * 60  # 3 menit - lebih pendek dari Yahoo karena ini pembanding

    def __init__(self, cache: TTLCache):
        self.cache = cache

    def _fetch(self, url: str) -> str:
        resp = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
        resp.raise_for_status()
        return resp.text

    def get_price(self, ticker: str) -> Optional[float]:
        """
        Scrape harga dari Google Finance sebagai pembanding independen.

        PENTING: struktur HTML Google Finance bisa berubah sewaktu-waktu
        tanpa pemberitahuan. Kalau fungsi ini mulai selalu gagal, buka
        https://www.google.com/finance/quote/{TICKER}:IDX manual di
        browser, inspect elemen harga, dan sesuaikan pola regex di bawah.
        """
        cache_key = f"scrape:price:{ticker.upper()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"https://www.google.com/finance/quote/{ticker.upper()}:IDX"
        try:
            html = self._fetch(url)
        except Exception as e:
            logger.warning("Scraping gagal untuk %s: %s", ticker, e)
            return None

        # Pola 1: atribut data-last-price (kalau tersedia di markup)
        match = re.search(r'data-last-price="([\d.]+)"', html)
        # Pola 2: fallback ke class harga utama Google Finance (format "Rp1.234,56")
        if not match:
            match = re.search(r'class="YMlKec[^"]*">Rp([\d.,]+)<', html)

        if not match:
            logger.warning("Tidak menemukan harga di halaman Google Finance untuk %s", ticker)
            return None

        raw = match.group(1)
        # Format Indonesia: titik = pemisah ribuan, koma = desimal
        raw = raw.replace(".", "").replace(",", ".")
        try:
            price = float(raw)
        except ValueError:
            return None

        self.cache.set(cache_key, price, self.CACHE_TTL)
        return price


# ---------------------------------------------------------------------
# DataSourceManager - orkestrasi + verifikasi silang harga
# ---------------------------------------------------------------------
class DataSourceManager:
    def __init__(self):
        self.cache = TTLCache()
        self.github = GitHubDatasetSource(self.cache)
        self.yahoo = YahooFinanceSource(self.cache)
        self.scraper = WebScrapingSource(self.cache)

    def get_verified_price(self, ticker: str, tolerance_pct: float = 5.0) -> Dict[str, Any]:
        """
        Ambil harga dari Yahoo, silang-cek dengan hasil scraping Google
        Finance. Kalau selisihnya lebih dari `tolerance_pct`, tandai
        sebagai `flagged` dan pakai harga hasil scraping (dianggap lebih
        bisa dipercaya untuk kasus seperti ini) alih-alih diam-diam
        menampilkan angka Yahoo yang mencurigakan.

        Return:
            {
                "ticker": str,
                "yahoo_price": float | None,
                "scraped_price": float | None,
                "diff_pct": float | None,
                "flagged": bool,
                "price_to_use": float | None,
            }
        """
        info = self.yahoo.get_stock_info(ticker)
        yahoo_price = info.price if info else None
        scraped_price = self.scraper.get_price(ticker)

        diff_pct = None
        flagged = False
        if yahoo_price and scraped_price:
            diff_pct = abs(yahoo_price - scraped_price) / scraped_price * 100
            flagged = diff_pct > tolerance_pct

        price_to_use = scraped_price if (flagged and scraped_price) else (yahoo_price or scraped_price)

        return {
            "ticker": ticker.upper(),
            "yahoo_price": yahoo_price,
            "scraped_price": scraped_price,
            "diff_pct": diff_pct,
            "flagged": flagged,
            "price_to_use": price_to_use,
        }
