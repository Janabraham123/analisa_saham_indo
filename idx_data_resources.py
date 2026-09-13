"""
idx_data_sources.py

Port Python dari arsitektur data-source di baguskto/saham-mcp
untuk data saham IDX, disesuaikan supaya bisa dipakai di dalam bot
analisa_saham_indo (main.py).

Data source:

  1. GitHubDatasetSource
     - daftar ticker & sektor
     - data historis

  2. YahooFinanceSource
     - harga quote terbaru
     - OHLCV
     - metadata saham

  3. WebScrapingSource
     - scraping Google Finance
     - digunakan sebagai pembanding/silang-cek harga

Prioritas harga Yahoo:

  1. fast_info["last_price"]
  2. info["regularMarketPrice"]
  3. history()["Close"] sebagai fallback

CATATAN:
Yahoo Finance bukan direct feed Bursa Efek Indonesia dan tidak menjamin
tick-by-tick real-time. Namun penggunaan last_price / regularMarketPrice
lebih tepat untuk harga quote terbaru dibanding selalu menggunakan
Close dari candle harian terakhir.
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
# Cache TTL sederhana di memori
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
        self._store[key] = (
            value,
            time.time() + ttl_seconds
        )

    def clear(self):
        self._store.clear()


# ---------------------------------------------------------------------
# 1. GitHubDatasetSource
# ---------------------------------------------------------------------
class GitHubDatasetSource:
    API_BASE = (
        "https://api.github.com/repos/"
        "wildangunawan/Dataset-Saham-IDX/contents"
    )

    RAW_BASE = (
        "https://raw.githubusercontent.com/"
        "wildangunawan/Dataset-Saham-IDX/master"
    )

    CACHE_TTL = 24 * 60 * 60

    def __init__(self, cache: TTLCache):
        self.cache = cache

    def get_available_stocks(self) -> List[str]:
        cached = self.cache.get("gh:available_stocks")

        if cached is not None:
            return cached

        resp = requests.get(
            f"{self.API_BASE}/Saham/Semua",
            timeout=15
        )

        resp.raise_for_status()

        files = resp.json()

        tickers = sorted(
            f["name"].replace(".csv", "")
            for f in files
            if f["name"].endswith(".csv")
        )

        self.cache.set(
            "gh:available_stocks",
            tickers,
            self.CACHE_TTL
        )

        return tickers

    def get_sectors(self) -> List[str]:
        cached = self.cache.get("gh:sectors")

        if cached is not None:
            return cached

        resp = requests.get(
            f"{self.API_BASE}/List%20Emiten/Sectors",
            timeout=15
        )

        resp.raise_for_status()

        files = resp.json()

        sectors = sorted(
            f["name"].replace(".csv", "")
            for f in files
            if f["name"].endswith(".csv")
        )

        self.cache.set(
            "gh:sectors",
            sectors,
            self.CACHE_TTL
        )

        return sectors

    def get_tickers_in_sector(
        self,
        sector_name: str,
        limit: int = 10
    ) -> List[str]:

        url = (
            f"{self.RAW_BASE}/List%20Emiten/Sectors/"
            f"{urllib.parse.quote(sector_name)}.csv"
        )

        df = pd.read_csv(url)

        for col in [
            "Kode",
            "kode",
            "Code",
            "code",
            "Ticker",
            "ticker",
            "Kode Saham",
            "Symbol",
        ]:
            if col in df.columns:
                return (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .tolist()[:limit]
                )

        for col in df.columns:
            sample = df[col].astype(str).head(10)

            if sample.str.match(
                r"^[A-Z]{4}$"
            ).sum() >= 5:

                return (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .tolist()[:limit]
                )

        return []


# ---------------------------------------------------------------------
# 2. YahooFinanceSource
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
    source: str = "yahoo_quote"


class YahooFinanceSource:

    # Sebelumnya 5 menit.
    # Sekarang 60 detik supaya harga tidak terlalu lama tertinggal.
    CACHE_TTL = 60

    def __init__(self, cache: TTLCache):
        self.cache = cache

    @staticmethod
    def to_yahoo_ticker(ticker: str) -> str:
        ticker = ticker.strip().upper()

        return (
            ticker
            if ticker.endswith(".JK")
            else f"{ticker}.JK"
        )

    def get_stock_info(
        self,
        ticker: str
    ) -> Optional[StockInfo]:

        cache_key = f"yf:info:{ticker.upper()}"

        cached = self.cache.get(cache_key)

        if cached is not None:
            return cached

        ticker_clean = ticker.strip().upper()
        yt = self.to_yahoo_ticker(ticker_clean)

        try:
            tk = yf.Ticker(yt)
        except Exception as e:
            logger.error(
                "Gagal membuat Yahoo Ticker %s: %s",
                yt,
                e
            )
            return None

        # =============================================================
        # Variabel hasil
        # =============================================================

        price: Optional[float] = None
        prev_close: Optional[float] = None
        volume: Optional[float] = None

        source = "yahoo_quote"

        info: Dict[str, Any] = {}

        # =============================================================
        # 1. Metadata dari Yahoo .info
        #
        # .info TIDAK dijadikan sumber utama harga.
        # =============================================================

        try:
            info = tk.info or {}

        except Exception as e:
            logger.warning(
                "Yahoo .info gagal untuk %s: %s",
                yt,
                e
            )

        # =============================================================
        # 2. PRIORITAS UTAMA:
        #    Yahoo fast_info -> last_price
        # =============================================================

        try:
            fast = tk.fast_info

            last_price = fast.get("last_price")

            if last_price is not None:
                price = float(last_price)
                source = "yahoo_quote"

            previous_close = fast.get("previous_close")

            if previous_close is not None:
                prev_close = float(previous_close)

            last_volume = fast.get("last_volume")

            if last_volume is not None:
                volume = float(last_volume)

        except Exception as e:
            logger.warning(
                "Yahoo fast_info gagal untuk %s: %s",
                yt,
                e
            )

        # =============================================================
        # 3. FALLBACK:
        #    regularMarketPrice
        # =============================================================

        if price is None:

            try:
                regular_market_price = info.get(
                    "regularMarketPrice"
                )

                if regular_market_price is not None:
                    price = float(
                        regular_market_price
                    )

                    source = (
                        "yahoo_regularMarketPrice"
                    )

            except Exception as e:
                logger.warning(
                    "regularMarketPrice gagal untuk %s: %s",
                    yt,
                    e
                )

        # previous close fallback dari .info
        if prev_close is None:

            try:
                regular_previous_close = info.get(
                    "regularMarketPreviousClose"
                )

                if regular_previous_close is not None:
                    prev_close = float(
                        regular_previous_close
                    )

            except Exception:
                pass

        # volume fallback dari .info
        if volume is None:

            try:
                regular_volume = info.get(
                    "regularMarketVolume"
                )

                if regular_volume is not None:
                    volume = float(
                        regular_volume
                    )

            except Exception:
                pass

        # =============================================================
        # 4. FALLBACK TERAKHIR:
        #    history().Close
        #
        # Ini bukan pilihan utama.
        # Dipakai kalau quote Yahoo gagal.
        # =============================================================

        if price is None:

            try:
                hist = tk.history(
                    period="5d",
                    auto_adjust=False
                )

                if not hist.empty:

                    price = float(
                        hist["Close"].iloc[-1]
                    )

                    source = (
                        "yahoo_history_fallback"
                    )

                    if prev_close is None and len(hist) > 1:
                        prev_close = float(
                            hist["Close"].iloc[-2]
                        )

                    if volume is None:
                        volume = float(
                            hist["Volume"].iloc[-1]
                        )

            except Exception as e:
                logger.warning(
                    "Yahoo history gagal untuk %s: %s",
                    yt,
                    e
                )

        # =============================================================
        # 5. Semua sumber Yahoo gagal
        # =============================================================

        if price is None:

            logger.warning(
                "Tidak berhasil mendapatkan harga untuk %s",
                ticker_clean
            )

            return None

        # =============================================================
        # 6. Hitung perubahan harga
        # =============================================================

        change_pct = None

        if (
            prev_close is not None
            and prev_close != 0
        ):
            change_pct = (
                (price - prev_close)
                / prev_close
                * 100
            )

        # =============================================================
        # 7. Buat StockInfo
        # =============================================================

        result = StockInfo(
            ticker=ticker_clean,
            name=info.get(
                "longName",
                ticker_clean
            ),
            price=price,
            prev_close=prev_close,
            change_pct=change_pct,
            volume=volume,
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            week52_high=info.get("fiftyTwoWeekHigh"),
            week52_low=info.get("fiftyTwoWeekLow"),
            source=source,
        )

        # =============================================================
        # 8. Cache hanya 60 detik
        # =============================================================

        self.cache.set(
            cache_key,
            result,
            self.CACHE_TTL
        )

        logger.info(
            "Harga %s = %.4f | source=%s",
            ticker_clean,
            price,
            source
        )

        return result

    # -----------------------------------------------------------------
    # Historical data
    # -----------------------------------------------------------------

    def get_history(
        self,
        ticker: str,
        period: str = "1y"
    ) -> pd.DataFrame:

        return yf.Ticker(
            self.to_yahoo_ticker(ticker)
        ).history(
            period=period,
            auto_adjust=False
        )

    # -----------------------------------------------------------------
    # IHSG
    # -----------------------------------------------------------------

    def get_ihsg(self) -> pd.DataFrame:

        return yf.Ticker(
            "^JKSE"
        ).history(
            period="5d",
            auto_adjust=False
        )


# ---------------------------------------------------------------------
# 3. WebScrapingSource
# ---------------------------------------------------------------------

class WebScrapingSource:

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )

    CACHE_TTL = 3 * 60

    def __init__(self, cache: TTLCache):
        self.cache = cache

    def _fetch(self, url: str) -> str:

        resp = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": (
                    "text/html,application/xhtml+xml,"
                    "application/xml;q=0.9,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.5",
            },
        )

        resp.raise_for_status()

        return resp.text

    def get_price(
        self,
        ticker: str
    ) -> Optional[float]:

        cache_key = (
            f"scrape:price:{ticker.upper()}"
        )

        cached = self.cache.get(cache_key)

        if cached is not None:
            return cached

        url = (
            "https://www.google.com/finance/quote/"
            f"{ticker.upper()}:IDX"
        )

        try:
            html = self._fetch(url)

        except Exception as e:

            logger.warning(
                "Scraping gagal untuk %s: %s",
                ticker,
                e
            )

            return None

        # =============================================================
        # Pola 1:
        # data-last-price
        # =============================================================

        match = re.search(
            r'data-last-price="([\d.]+)"',
            html
        )

        # =============================================================
        # Pola 2:
        # class YMlKec
        # =============================================================

        if not match:

            match = re.search(
                r'class="YMlKec[^"]*">Rp([\d.,]+)<',
                html
            )

        if not match:

            logger.warning(
                "Tidak menemukan harga Google Finance untuk %s",
                ticker
            )

            return None

        raw = match.group(1)

        # Format Indonesia:
        #
        # 1.234,56
        #
        # menjadi:
        #
        # 1234.56

        raw = (
            raw
            .replace(".", "")
            .replace(",", ".")
        )

        try:
            price = float(raw)

        except ValueError:

            logger.warning(
                "Format harga Google Finance tidak valid "
                "untuk %s: %s",
                ticker,
                raw
            )

            return None

        self.cache.set(
            cache_key,
            price,
            self.CACHE_TTL
        )

        return price


# ---------------------------------------------------------------------
# DataSourceManager
# ---------------------------------------------------------------------

class DataSourceManager:

    def __init__(self):

        self.cache = TTLCache()

        self.github = GitHubDatasetSource(
            self.cache
        )

        self.yahoo = YahooFinanceSource(
            self.cache
        )

        self.scraper = WebScrapingSource(
            self.cache
        )

    def get_verified_price(
        self,
        ticker: str,
        tolerance_pct: float = 5.0
    ) -> Dict[str, Any]:

        """
        Ambil harga dari Yahoo dan silang-cek dengan Google Finance.

        Yahoo quote tetap menjadi sumber utama.

        Google Finance hanya digunakan untuk:
        - membandingkan harga
        - mendeteksi perbedaan yang mencurigakan

        Google Finance TIDAK otomatis menggantikan harga Yahoo hanya
        karena terdapat selisih > tolerance_pct.

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

        # =============================================================
        # Ambil Yahoo
        # =============================================================

        info = self.yahoo.get_stock_info(
            ticker
        )

        yahoo_price = (
            info.price
            if info
            else None
        )

        # =============================================================
        # Ambil Google Finance sebagai cross-check
        # =============================================================

        scraped_price = self.scraper.get_price(
            ticker
        )

        # =============================================================
        # Hitung selisih
        # =============================================================

        diff_pct = None
        flagged = False

        if (
            yahoo_price is not None
            and scraped_price is not None
            and scraped_price != 0
        ):

            diff_pct = (
                abs(
                    yahoo_price
                    - scraped_price
                )
                / scraped_price
                * 100
            )

            flagged = (
                diff_pct > tolerance_pct
            )

        # =============================================================
        # Yahoo tetap menjadi sumber utama
        #
        # Google hanya menjadi fallback kalau Yahoo benar-benar gagal.
        # =============================================================

        if yahoo_price is not None:

            price_to_use = yahoo_price

        elif scraped_price is not None:

            price_to_use = scraped_price

        else:

            price_to_use = None

        # =============================================================
        # Log jika terdapat perbedaan besar
        # =============================================================

        if flagged:

            logger.warning(
                "Perbedaan harga besar %s: "
                "Yahoo=%.4f | Google=%.4f | diff=%.2f%%",
                ticker.upper(),
                yahoo_price,
                scraped_price,
                diff_pct,
            )

        return {
            "ticker": ticker.upper(),
            "yahoo_price": yahoo_price,
            "scraped_price": scraped_price,
            "diff_pct": diff_pct,
            "flagged": flagged,
            "price_to_use": price_to_use,
        }