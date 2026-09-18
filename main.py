"""
OKX Long Entry Scanner — v6
================================================================
Pembagian tugas antar timeframe (sengaja beda-beda, bukan tiga kali
pertanyaan yang sama):

  4H  (KUNCI)      -> apakah setup-nya layak?
                      WAJIB KEDUANYA: StochRSI masih <= 35 (oversold) DAN
                      golden cross (K memotong D) sudah/baru terjadi.
                      Kalau StochRSI sudah di atas 35 (sudah naik jauh),
                      TIDAK lolos lagi meski pernah golden cross.
  1H  (KONFIRMASI) -> apakah belum telat? momentum belum habis,
                      belum overbought, arah belum berbalik
  15M (KONFIRMASI ULANG) -> timing masuk. StochRSI/RSI sedang naik.

Konteks tambahan (TIDAK menggugurkan sinyal, cuma info untuk riset):
  - Volume 4H dibanding rata-rata 20 candle
  - Open Interest: naik/turun dalam ~24 jam terakhir
  - Bullish divergence di 4H (petunjuk lemah, sering salah deteksi)

Output untuk tiap kandidat:
  - Fibonacci 4H -> jaring entry 3 lapis (sesuai modal dibagi 3),
    SL, TP1/TP2/TP3, plus rasio risk/reward

ENV VARS (GitHub Secrets):
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""

import os
import json
import time
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd

# ============================== CONFIG ==============================

OKX_BASE_URL = "https://www.okx.com"
BINANCE_BASE_URL = "https://fapi.binance.com"
SETTLE_CCY = "USDT"
REQUEST_DELAY_SEC = 0.12

EXCHANGES = ["OKX", "BINANCE"]   # comment salah satu baris ini kalau mau nonaktifkan sementara

# ============================== FILTER KATEGORI PROYEK ==============================
# Diambil OTOMATIS tiap kali bot jalan dari CoinGecko (API publik, tanpa API
# key). Tidak perlu update manual lagi. Kalau CoinGecko gagal/rate-limit,
# otomatis jatuh balik ke daftar manual di bawah (STATIC_CATEGORY_FALLBACK)
# supaya bot tetap jalan.
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_CATEGORY_SLUGS = {
    "DeFi": "decentralized-finance-defi",
    "RWA": "real-world-assets-rwa",
    "L1": "layer-1",
    "L2": "layer-2",
    "AI": "artificial-intelligence",
    "Memecoin": "meme-token",
}
LARGE_CAP_MAX_RANK = 20    # market cap rank 1-20   -> Large-Cap
MID_CAP_MAX_RANK = 150     # market cap rank 21-150 -> Mid-Cap
COINGECKO_REQUEST_DELAY = 2.0  # jeda antar call biar tidak kena rate limit (~10-30/menit di tier gratis)

SELECTED_CATEGORIES = ["Large-Cap", "Mid-Cap", "DeFi", "RWA", "L1", "L2", "AI", "Memecoin"]

# Cadangan kalau CoinGecko tidak bisa diakses -- dikurasi manual, tidak lengkap
# 100% tapi cukup buat bot tetap jalan di run itu.
STATIC_CATEGORY_FALLBACK = {
    "Large-Cap": {
        "BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "AVAX", "DOT", "LINK",
        "LTC", "BCH", "TRX", "MATIC", "POL", "TON", "ATOM", "DOGE", "SHIB",
    },
    "Mid-Cap": {
        "UNI", "AAVE", "INJ", "SUI", "APT", "NEAR", "ARB", "OP", "FIL",
        "ICP", "HBAR", "VET", "ALGO", "FTM", "SAND", "MANA", "AXS", "EGLD",
        "XLM", "XMR", "ETC", "RUNE", "IMX", "GRT", "MKR", "SNX", "LDO",
        "CRV", "COMP", "TIA", "SEI", "JTO", "PYTH", "WLD", "ORDI", "STX",
        "RENDER", "FET", "TAO", "JUP", "STRK", "W", "ONDO", "GALA", "CHZ",
    },
    "DeFi": {
        "UNI", "AAVE", "CRV", "COMP", "MKR", "SNX", "SUSHI", "1INCH",
        "BAL", "YFI", "CAKE", "DYDX", "GMX", "LDO", "PENDLE", "JOE",
        "RUNE", "KNC", "LRC", "ZRX", "CVX", "FXS",
    },
    "RWA": {
        "ONDO", "POLYX", "CFG", "TRU", "MPL", "RIO", "OM", "PROPS",
        "CTC", "DUSK",
    },
    "L1": {
        "BTC", "ETH", "BNB", "SOL", "AVAX", "ADA", "DOT", "NEAR", "ATOM",
        "ALGO", "FTM", "EGLD", "ICP", "APT", "SUI", "SEI", "TIA", "TON",
        "KAVA", "ROSE", "ZIL", "KSM", "WAVES", "FLOW", "MINA", "ONE",
        "CELO", "XTZ", "INJ",
    },
    "L2": {
        "ARB", "OP", "MATIC", "POL", "STRK", "MANTA", "METIS", "IMX",
        "ZK", "LRC", "MNT", "BLAST",
    },
    "AI": {
        "FET", "AGIX", "OCEAN", "RENDER", "TAO", "RLC", "NMR", "GRT",
        "AKT", "WLD", "ARKM", "PHB",
    },
    "Memecoin": {
        "DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI", "MEME", "BOME",
        "SATS", "ORDI", "NEIRO", "POPCAT", "MOG", "TURBO", "BRETT",
        "DEGEN", "MYRO", "WOJAK", "LADYS",
    },
}

# Diisi oleh build_category_map() di awal main(). Sebelum itu dipanggil,
# pakai fallback statis supaya import module tidak pernah dalam keadaan kosong.
ACTIVE_CATEGORY_MAP = {k: set(v) for k, v in STATIC_CATEGORY_FALLBACK.items()}


def fetch_coingecko_markets(params: dict, retries: int = 3) -> list:
    url = f"{COINGECKO_BASE}/coins/markets"
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code == 429:
            time.sleep(15 + attempt * 15)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("CoinGecko rate limited terus-menerus")


def build_category_map() -> dict:
    """Bangun peta kategori LIVE dari CoinGecko. Jatuh balik ke daftar manual
    kalau gagal (network error, rate limit habis, dsb) -- tidak pernah
    membuat bot berhenti total gara-gara ini."""
    cat_map = {cat: set() for cat in COINGECKO_CATEGORY_SLUGS}
    cat_map["Large-Cap"] = set()
    cat_map["Mid-Cap"] = set()

    try:
        # 1) ranking market cap keseluruhan -> buat Large-Cap / Mid-Cap
        rows = fetch_coingecko_markets({
            "vs_currency": "usd", "order": "market_cap_desc",
            "per_page": 250, "page": 1, "sparkline": "false",
        })
        for r in rows:
            sym, rank = r.get("symbol", "").upper(), r.get("market_cap_rank")
            if not sym or rank is None:
                continue
            if rank <= LARGE_CAP_MAX_RANK:
                cat_map["Large-Cap"].add(sym)
            elif rank <= MID_CAP_MAX_RANK:
                cat_map["Mid-Cap"].add(sym)
        time.sleep(COINGECKO_REQUEST_DELAY)

        # 2) tiap kategori narasi (DeFi, RWA, L1, L2, AI, Memecoin)
        for cat_name, slug in COINGECKO_CATEGORY_SLUGS.items():
            rows = fetch_coingecko_markets({
                "vs_currency": "usd", "category": slug,
                "order": "market_cap_desc", "per_page": 250, "page": 1,
                "sparkline": "false",
            })
            for r in rows:
                sym = r.get("symbol", "").upper()
                if sym:
                    cat_map[cat_name].add(sym)
            time.sleep(COINGECKO_REQUEST_DELAY)

    except Exception as e:
        print(f"[WARNING] Gagal ambil kategori live dari CoinGecko ({e}) -- pakai daftar manual cadangan.")
        return {k: set(v) for k, v in STATIC_CATEGORY_FALLBACK.items()}

    if all(len(v) == 0 for v in cat_map.values()):
        print("[WARNING] Data kategori dari CoinGecko kosong (format API mungkin berubah) -- pakai daftar manual cadangan.")
        return {k: set(v) for k, v in STATIC_CATEGORY_FALLBACK.items()}

    total = sum(len(v) for v in cat_map.values())
    print(f"[INFO] Kategori live dari CoinGecko berhasil diambil ({total} entri ticker-kategori).")
    return cat_map


def get_base_ticker(exchange: str, inst_id: str) -> str:
    if exchange == "OKX":
        base = inst_id.split("-")[0]
    else:  # BINANCE, format "BTCUSDT"
        base = inst_id[:-4] if inst_id.endswith("USDT") else inst_id
    for prefix in ("1000000", "1000"):  # token meme yang di-scale (mis. 1000PEPEUSDT)
        if base.startswith(prefix) and len(base) > len(prefix):
            base = base[len(prefix):]
            break
    return base.upper()


def categories_for(base_ticker: str) -> list:
    return [cat for cat, tickers in ACTIVE_CATEGORY_MAP.items() if base_ticker in tickers]


def passes_category_filter(exchange: str, inst_id: str) -> list:
    """Return list kategori yang cocok (kosong = tidak lolos filter)."""
    base = get_base_ticker(exchange, inst_id)
    cats = categories_for(base)
    return [c for c in cats if c in SELECTED_CATEGORIES]

BAR_KEY = "4H"       # timeframe kunci
BAR_CONFIRM = "1H"   # konfirmasi
BAR_TIMING = "15m"   # konfirmasi ulang / timing

# Mapping nama timeframe generik -> format khusus tiap exchange
BINANCE_BAR_MAP = {"4H": "4h", "1H": "1h", "15m": "15m"}

LIMIT_KEY = 200
LIMIT_CONFIRM = 150
LIMIT_TIMING = 150

RSI_LENGTH = 14
STOCH_LENGTH = 14
STOCH_K_SMOOTH = 3
STOCH_D_SMOOTH = 3
EMA_FAST, EMA_SLOW = 9, 21

STOCH_OVERSOLD = 50        # 4H: StochRSI K wajib di bawah ini (potensi naik masih luas)
SWING_ORDER = 3            # candle kiri/kanan buat tentukan swing high/low
STRUCTURE_LOOKBACK = 80    # jumlah candle 4H yang dicek buat struktur
CROSS_RECENT_BARS = 3      # golden cross dianggap "baru" kalau terjadi <= N candle lalu
CONFIRM_RSI_MIN = 40       # 1H: di bawah ini artinya momentum masih lemah
CONFIRM_RSI_MAX = 72       # 1H: di atas ini artinya sudah telat / overbought
CONFIRM_STOCH_MAX = 85     # 1H: StochRSI sudah terlalu tinggi

VOL_MA_LENGTH = 20
FIB_LOOKBACK = 60          # candle 4H untuk cari swing high/low
MIN_RR = 1.5               # rasio risk/reward minimal untuk ditandai sehat

MAX_RESULTS_IN_MESSAGE = 10

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ============================== DATA FETCH: OKX ==============================

def okx_fetch_instruments(settle_ccy: str = SETTLE_CCY) -> list:
    """Futures perpetual (SWAP) USDT-margined di OKX -- semuanya pair crypto/USDT."""
    url = f"{OKX_BASE_URL}/api/v5/public/instruments"
    resp = requests.get(url, params={"instType": "SWAP"}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "0":
        raise RuntimeError(f"Gagal ambil daftar instrument OKX: {data}")
    return sorted([
        i["instId"] for i in data["data"]
        if i.get("settleCcy") == settle_ccy and i.get("state") == "live"
    ])


def okx_fetch_candles(inst_id: str, bar: str, limit: int, retries: int = 3) -> pd.DataFrame:
    url = f"{OKX_BASE_URL}/api/v5/market/candles"
    params = {"instId": inst_id, "bar": bar, "limit": str(limit)}

    data = None
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            time.sleep(1 + attempt)
            continue
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX error {inst_id} ({bar}): {data}")
        break
    if data is None or not data.get("data"):
        raise RuntimeError(f"Tidak ada data candle {inst_id} ({bar})")

    rows = data["data"]
    cols = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]
    df = pd.DataFrame(rows, columns=cols[: len(rows[0])])
    df = df.iloc[::-1].reset_index(drop=True)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms")
    return df


def okx_fetch_oi_change(inst_id: str) -> dict:
    """Perubahan Open Interest ~24 jam terakhir.
    Endpoint rubik OKX pakai ccy (mis. 'BTC'), bukan instId penuh.
    Tidak semua coin tersedia -> kalau gagal, return unavailable (bukan error)."""
    ccy = inst_id.split("-")[0]
    url = f"{OKX_BASE_URL}/api/v5/rubik/stat/contracts/open-interest-volume"
    try:
        resp = requests.get(url, params={"ccy": ccy, "period": "1H"}, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0" or not data.get("data"):
            return {"available": False}
        rows = data["data"]  # [ts, oi, vol], terbaru dulu
        if len(rows) < 25:
            return {"available": False}
        oi_now = float(rows[0][1])
        oi_24h_ago = float(rows[24][1])
        if oi_24h_ago == 0:
            return {"available": False}
        change_pct = (oi_now / oi_24h_ago - 1) * 100
        return {"available": True, "change_24h_pct": round(change_pct, 2), "rising": change_pct > 0}
    except Exception:
        return {"available": False}


# ============================== DATA FETCH: BINANCE ==============================

def binance_fetch_instruments() -> list:
    """Futures perpetual USDT-M di Binance -- disaring khusus pair CRYPTO/USDT:
    quoteAsset == USDT, contractType == PERPETUAL, status == TRADING.
    Ini otomatis membuang pair non-crypto/non-perpetual kalau ada."""
    url = f"{BINANCE_BASE_URL}/fapi/v1/exchangeInfo"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return sorted([
        s["symbol"] for s in data.get("symbols", [])
        if s.get("quoteAsset") == "USDT"
        and s.get("contractType") == "PERPETUAL"
        and s.get("status") == "TRADING"
    ])


def binance_fetch_candles(symbol: str, bar: str, limit: int, retries: int = 3) -> pd.DataFrame:
    url = f"{BINANCE_BASE_URL}/fapi/v1/klines"
    interval = BINANCE_BAR_MAP.get(bar, bar)
    params = {"symbol": symbol, "interval": interval, "limit": str(limit)}

    data = None
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429 or resp.status_code == 418:
            time.sleep(2 + attempt * 2)
            continue
        resp.raise_for_status()
        data = resp.json()
        break
    if not data:
        raise RuntimeError(f"Tidak ada data candle {symbol} ({bar}) dari Binance")

    # kline Binance: [openTime, open, high, low, close, volume, closeTime, ...]
    df = pd.DataFrame(data, columns=[
        "ts", "open", "high", "low", "close", "vol", "closeTime",
        "quoteVol", "trades", "takerBaseVol", "takerQuoteVol", "ignore"
    ])
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms")
    return df[["ts", "open", "high", "low", "close", "vol"]]


def binance_fetch_oi_change(symbol: str) -> dict:
    """Perubahan Open Interest ~24 jam terakhir lewat openInterestHist (period 1h)."""
    url = f"{BINANCE_BASE_URL}/futures/data/openInterestHist"
    try:
        resp = requests.get(url, params={"symbol": symbol, "period": "1h", "limit": "25"}, timeout=12)
        resp.raise_for_status()
        rows = resp.json()  # terlama -> terbaru
        if not isinstance(rows, list) or len(rows) < 25:
            return {"available": False}
        oi_24h_ago = float(rows[0]["sumOpenInterest"])
        oi_now = float(rows[-1]["sumOpenInterest"])
        if oi_24h_ago == 0:
            return {"available": False}
        change_pct = (oi_now / oi_24h_ago - 1) * 100
        return {"available": True, "change_24h_pct": round(change_pct, 2), "rising": change_pct > 0}
    except Exception:
        return {"available": False}


# ============================== DISPATCHER ANTAR EXCHANGE ==============================

def fetch_instruments(exchange: str) -> list:
    if exchange == "OKX":
        return okx_fetch_instruments()
    if exchange == "BINANCE":
        return binance_fetch_instruments()
    raise ValueError(f"Exchange tidak dikenal: {exchange}")


def fetch_candles(exchange: str, inst_id: str, bar: str, limit: int) -> pd.DataFrame:
    if exchange == "OKX":
        return okx_fetch_candles(inst_id, bar, limit)
    if exchange == "BINANCE":
        return binance_fetch_candles(inst_id, bar, limit)
    raise ValueError(f"Exchange tidak dikenal: {exchange}")


def fetch_oi_change(exchange: str, inst_id: str) -> dict:
    if exchange == "OKX":
        return okx_fetch_oi_change(inst_id)
    if exchange == "BINANCE":
        return binance_fetch_oi_change(inst_id)
    return {"available": False}


# ============================== INDIKATOR ==============================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = RSI_LENGTH) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def stoch_rsi(close: pd.Series):
    rsi_series = rsi(close, RSI_LENGTH)
    min_rsi = rsi_series.rolling(STOCH_LENGTH).min()
    max_rsi = rsi_series.rolling(STOCH_LENGTH).max()
    raw = (rsi_series - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan) * 100
    k = raw.rolling(STOCH_K_SMOOTH).mean()
    d = k.rolling(STOCH_D_SMOOTH).mean()
    return rsi_series, k.fillna(50), d.fillna(50)


def crossed_up_recently(fast: pd.Series, slow: pd.Series, bars: int = CROSS_RECENT_BARS) -> bool:
    """True kalau fast memotong slow dari bawah dalam N candle terakhir."""
    if len(fast) < bars + 2:
        return False
    for i in range(1, bars + 1):
        if fast.iloc[-i] > slow.iloc[-i] and fast.iloc[-i - 1] <= slow.iloc[-i - 1]:
            return True
    return False


def volume_context(df: pd.DataFrame) -> dict:
    vol_ma = df["vol"].rolling(VOL_MA_LENGTH).mean().iloc[-1]
    vol_now = float(df["vol"].iloc[-1])
    if not vol_ma or np.isnan(vol_ma) or vol_ma == 0:
        return {"ratio": 1.0, "rising": False}
    ratio = vol_now / vol_ma
    rising = df["vol"].iloc[-1] > df["vol"].iloc[-2] if len(df) > 2 else False
    return {"ratio": round(float(ratio), 2), "rising": bool(rising)}


def bullish_divergence(df: pd.DataFrame, lookback: int = 40) -> bool:
    """Harga bikin low lebih rendah, RSI bikin low lebih tinggi.
    Deteksi sederhana -- ini petunjuk lemah, jangan dijadikan penentu."""
    if len(df) < lookback + 5:
        return False
    sub = df.tail(lookback).reset_index(drop=True)
    rsi_series = rsi(sub["close"]).reset_index(drop=True)

    lows = []
    for i in range(2, len(sub) - 2):
        window = sub["low"].iloc[i - 2:i + 3]
        if sub["low"].iloc[i] == window.min():
            lows.append(i)
    if len(lows) < 2:
        return False

    i1, i2 = lows[-2], lows[-1]
    price_lower_low = sub["low"].iloc[i2] < sub["low"].iloc[i1]
    rsi_higher_low = rsi_series.iloc[i2] > rsi_series.iloc[i1]
    return bool(price_lower_low and rsi_higher_low)


def fibonacci_plan(df: pd.DataFrame, price: float) -> dict:
    """Jaring entry 3 lapis + SL + TP dari swing 4H."""
    recent = df.tail(FIB_LOOKBACK)
    swing_low = float(recent["low"].min())
    swing_high = float(recent["high"].max())
    diff = swing_high - swing_low
    if diff <= 0:
        return None

    lvl = {
        "0.382": swing_high - 0.382 * diff,
        "0.5": swing_high - 0.5 * diff,
        "0.618": swing_high - 0.618 * diff,
        "0.786": swing_high - 0.786 * diff,
    }

    # jaring entry: harga sekarang + dua level fib di bawahnya
    below = sorted([v for v in lvl.values() if v < price], reverse=True)
    entries = [price] + below[:2]
    while len(entries) < 3:
        entries.append(entries[-1] * 0.985)  # cadangan kalau level fib kurang
    entries = [round(e, 8) for e in entries[:3]]

    avg_entry = sum(entries) / len(entries)
    sl = min(lvl["0.786"], swing_low) * 0.995
    tp1 = swing_high
    tp2 = swing_high + 0.272 * diff
    tp3 = swing_high + 0.618 * diff

    risk = avg_entry - sl
    reward = tp1 - avg_entry
    rr = (reward / risk) if risk > 0 else 0

    return {
        "swing_low": swing_low,
        "swing_high": swing_high,
        "entries": entries,
        "avg_entry": avg_entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rr_to_tp1": round(float(rr), 2),
        "rr_healthy": bool(rr >= MIN_RR),
    }


# ============================== KRITERIA TIAP TIMEFRAME ==============================

def detect_structure(df: pd.DataFrame, order: int = SWING_ORDER, lookback: int = STRUCTURE_LOOKBACK) -> dict:
    """Deteksi struktur harga sederhana (proksi BOS/CHoCH, bukan replika
    persis LuxAlgo -- parameter internal mereka tidak dipublikasikan).

    - Cari swing high & swing low pakai metode fraktal (titik tertinggi/
      terendah dibanding N candle kiri-kanan).
    - Trend 'up' kalau swing high & low terbaru sama-sama lebih tinggi dari
      sebelumnya (HH+HL). Trend 'down' kalau sebaliknya (LH+LL).
    - BOS bullish: trend sudah 'up', lalu close menembus swing high
      terakhir lagi (lanjutan tren).
    - CHoCH bullish: trend masih 'down'/'mixed', tapi close menembus swing
      high terakhir (lower high) -- sinyal awal potensi pembalikan.
    """
    sub = df.tail(lookback).reset_index(drop=True)
    n = len(sub)
    if n < order * 2 + 10:
        return {"trend": "unknown", "bos_up": False, "choch_up": False,
                "last_swing_high": None, "last_swing_low": None}

    high_idx, low_idx = [], []
    for i in range(order, n - order):
        window_h = sub["high"].iloc[i - order:i + order + 1]
        window_l = sub["low"].iloc[i - order:i + order + 1]
        if sub["high"].iloc[i] == window_h.max():
            high_idx.append(i)
        if sub["low"].iloc[i] == window_l.min():
            low_idx.append(i)

    if len(high_idx) < 2 or len(low_idx) < 2:
        return {"trend": "unknown", "bos_up": False, "choch_up": False,
                "last_swing_high": None, "last_swing_low": None}

    last_high, prev_high = sub["high"].iloc[high_idx[-1]], sub["high"].iloc[high_idx[-2]]
    last_low, prev_low = sub["low"].iloc[low_idx[-1]], sub["low"].iloc[low_idx[-2]]

    higher_high, higher_low = last_high > prev_high, last_low > prev_low
    lower_high, lower_low = last_high < prev_high, last_low < prev_low

    if higher_high and higher_low:
        trend = "up"
    elif lower_high and lower_low:
        trend = "down"
    else:
        trend = "mixed"

    close_now = float(sub["close"].iloc[-1])
    broke_last_high = close_now > last_high

    return {
        "trend": trend,
        "bos_up": bool(broke_last_high and trend == "up"),
        "choch_up": bool(broke_last_high and trend in ("down", "mixed")),
        "last_swing_high": round(float(last_high), 8),
        "last_swing_low": round(float(last_low), 8),
    }


def check_key_4h(df: pd.DataFrame) -> dict:
    """4H = kunci. WAJIB SEMUA:
    1. StochRSI (K) di bawah 50 -- masih ada ruang naik, bukan sudah telat
    2. Golden cross sudah/baru terjadi (K memotong D dari bawah)
    3. Struktur harga menunjukkan BOS atau CHoCH bullish -- ada BUKTI
       pergerakan harga yang menembus level penting, bukan cuma oscillator
       yang naik sementara harga masih terkurung/turun.
    Kalau salah satu tidak terpenuhi (termasuk golden cross tanpa BOS/CHoCH,
    persis kasus token yang sudah naik jauh tapi struktur belum konfirmasi),
    TIDAK lolos."""
    close = df["close"]
    rsi_series, k, d = stoch_rsi(close)
    rsi_now = float(rsi_series.iloc[-1])
    k_now, d_now = float(k.iloc[-1]), float(d.iloc[-1])

    ema_fast, ema_slow = ema(close, EMA_FAST), ema(close, EMA_SLOW)

    still_room = k_now <= STOCH_OVERSOLD
    golden_cross = crossed_up_recently(k, d)
    structure = detect_structure(df)
    structure_ok = structure["bos_up"] or structure["choch_up"]

    reasons = []
    if still_room:
        reasons.append(f"StochRSI < {STOCH_OVERSOLD}")
    if golden_cross:
        reasons.append("golden cross")
    if structure["bos_up"]:
        reasons.append("BOS bullish")
    if structure["choch_up"]:
        reasons.append("CHoCH bullish")

    return {
        "ok": bool(still_room and golden_cross and structure_ok),
        "rsi": round(rsi_now, 1),
        "stoch_k": round(k_now, 1),
        "stoch_d": round(d_now, 1),
        "ema_bull": bool(ema_fast.iloc[-1] > ema_slow.iloc[-1]),
        "structure_trend": structure["trend"],
        "last_swing_high": structure["last_swing_high"],
        "last_swing_low": structure["last_swing_low"],
        "reasons": reasons,
    }


def check_confirm_1h(df: pd.DataFrame) -> dict:
    """1H = apakah belum telat. Momentum ada tapi belum overbought."""
    close = df["close"]
    rsi_series, k, d = stoch_rsi(close)
    rsi_now = float(rsi_series.iloc[-1])
    k_now, d_now = float(k.iloc[-1]), float(d.iloc[-1])

    not_late = rsi_now <= CONFIRM_RSI_MAX and k_now <= CONFIRM_STOCH_MAX
    has_momentum = rsi_now >= CONFIRM_RSI_MIN
    stoch_bullish = k_now >= d_now

    notes = []
    if not not_late:
        notes.append("sudah overbought")
    if not has_momentum:
        notes.append("momentum masih lemah")
    if stoch_bullish:
        notes.append("StochRSI bullish")

    return {
        "ok": bool(not_late and has_momentum),
        "rsi": round(rsi_now, 1),
        "stoch_k": round(k_now, 1),
        "stoch_d": round(d_now, 1),
        "notes": notes,
    }


def check_timing_15m(df: pd.DataFrame) -> dict:
    """15m = timing. Cari tanda momentum sedang berbelok naik."""
    close = df["close"]
    rsi_series, k, d = stoch_rsi(close)
    k_now, k_prev = float(k.iloc[-1]), float(k.iloc[-2])
    d_now = float(d.iloc[-1])
    rsi_now, rsi_prev = float(rsi_series.iloc[-1]), float(rsi_series.iloc[-2])

    k_rising = k_now > k_prev
    rsi_rising = rsi_now > rsi_prev
    stoch_bullish = k_now >= d_now

    if stoch_bullish and k_rising:
        verdict = "siap masuk"
    elif k_rising or rsi_rising:
        verdict = "mulai berbelok naik"
    else:
        verdict = "belum, tunggu dulu"

    return {
        "ok": bool(k_rising or rsi_rising),
        "rsi": round(rsi_now, 1),
        "stoch_k": round(k_now, 1),
        "stoch_d": round(d_now, 1),
        "verdict": verdict,
    }


# ============================== SCAN ==============================

def scan_instrument(exchange: str, inst_id: str):
    df_4h = fetch_candles(exchange, inst_id, BAR_KEY, LIMIT_KEY)
    if len(df_4h) < 60:
        return None
    key = check_key_4h(df_4h)
    if not key["ok"]:
        return None

    time.sleep(REQUEST_DELAY_SEC)
    df_1h = fetch_candles(exchange, inst_id, BAR_CONFIRM, LIMIT_CONFIRM)
    if len(df_1h) < 60:
        return None
    confirm = check_confirm_1h(df_1h)
    if not confirm["ok"]:
        return None

    time.sleep(REQUEST_DELAY_SEC)
    df_15m = fetch_candles(exchange, inst_id, BAR_TIMING, LIMIT_TIMING)
    if len(df_15m) < 60:
        return None
    timing = check_timing_15m(df_15m)
    if not timing["ok"]:
        return None

    price = float(df_4h["close"].iloc[-1])
    plan = fibonacci_plan(df_4h, price)
    if plan is None:
        return None

    time.sleep(REQUEST_DELAY_SEC)
    oi = fetch_oi_change(exchange, inst_id)

    return {
        "exchange": exchange,
        "inst_id": inst_id,
        "price": price,
        "key_4h": key,
        "confirm_1h": confirm,
        "timing_15m": timing,
        "volume_4h": volume_context(df_4h),
        "open_interest": oi,
        "bullish_divergence_4h": bullish_divergence(df_4h),
        "plan": plan,
    }


def rank_score(c: dict) -> float:
    """Skor untuk mengurutkan hasil (bukan sinyal beli)."""
    s = 0.0
    s += 20 * len(c["key_4h"]["reasons"])
    if c["key_4h"]["ema_bull"]:
        s += 10
    if c["volume_4h"]["ratio"] >= 1.5:
        s += 15
    elif c["volume_4h"]["ratio"] >= 1.0:
        s += 7
    oi = c["open_interest"]
    if oi.get("available") and oi.get("rising"):
        s += 12
    if c["bullish_divergence_4h"]:
        s += 5
    if c["plan"]["rr_healthy"]:
        s += 15
    if c["timing_15m"]["verdict"] == "siap masuk":
        s += 10
    return round(s, 1)


# ============================== OUTPUT ==============================

def fmt(v: float) -> str:
    return f"{v:,.6f}".rstrip("0").rstrip(".") if v < 1 else f"{v:,.4f}"


def format_candidate_block(c: dict) -> str:
    p = c["plan"]
    oi = c["open_interest"]
    oi_txt = (f"{oi['change_24h_pct']:+.2f}% (24j)" if oi.get("available") else "n/a")
    vol = c["volume_4h"]

    lines = [
        f"🟢 *[{c['exchange']}] {c['inst_id']}* ({'/'.join(c['categories'])}) — skor {c['rank_score']}",
        f"Harga: {fmt(c['price'])}",
        f"4H: {' + '.join(c['key_4h']['reasons'])} | RSI {c['key_4h']['rsi']} | StochRSI {c['key_4h']['stoch_k']}/{c['key_4h']['stoch_d']}",
        f"    Struktur: {c['key_4h']['structure_trend']} | swing high terakhir: {fmt(c['key_4h']['last_swing_high'])}",
        f"1H: RSI {c['confirm_1h']['rsi']} | StochRSI {c['confirm_1h']['stoch_k']}/{c['confirm_1h']['stoch_d']}",
        f"15m: {c['timing_15m']['verdict']} (K {c['timing_15m']['stoch_k']})",
        f"Volume 4H: {vol['ratio']}x rata-rata | OI: {oi_txt}",
    ]
    if c["bullish_divergence_4h"]:
        lines.append("Terdeteksi bullish divergence 4H (petunjuk lemah)")

    lines += [
        "",
        "Jaring entry (Fibonacci 4H):",
        f"  E1 {fmt(p['entries'][0])} | E2 {fmt(p['entries'][1])} | E3 {fmt(p['entries'][2])}",
        f"  Rata2 entry: {fmt(p['avg_entry'])}",
        f"  SL: {fmt(p['sl'])}",
        f"  TP1 {fmt(p['tp1'])} | TP2 {fmt(p['tp2'])} | TP3 {fmt(p['tp3'])}",
        f"  R:R ke TP1 = {p['rr_to_tp1']}" + ("" if p["rr_healthy"] else f" ⚠️ di bawah {MIN_RR}"),
    ]
    return "\n".join(lines)


def format_message(candidates: list, total_scanned: int) -> list:
    header = (
        f"📊 *OKX Long Setup Scan*\n"
        f"4H kunci → 1H konfirmasi → 15m timing\n"
        f"Discan: {total_scanned} pair\n"
    )
    if not candidates:
        return [header + "\nTidak ada setup yang lolos saat ini. Tidak ada setup juga sebuah informasi."]

    blocks = [header] + [format_candidate_block(c) for c in candidates[:MAX_RESULTS_IN_MESSAGE]]
    if len(candidates) > MAX_RESULTS_IN_MESSAGE:
        blocks.append(f"...dan {len(candidates) - MAX_RESULTS_IN_MESSAGE} pair lain juga lolos.")
    blocks.append("_Semua level cuma referensi teknikal, bukan rekomendasi. Riset manual sebelum entry._")

    text = "\n\n".join(blocks)
    if len(text) <= 3800:
        return [text]

    chunks, current = [], header + "\n"
    for b in blocks[1:]:
        if len(current) + len(b) + 2 > 3800:
            chunks.append(current)
            current = ""
        current += b + "\n\n"
    if current.strip():
        chunks.append(current)
    return chunks


def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()


def save_results_json(candidates: list, scanned: int, path: str = "docs/results.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    per_exchange = {}
    for c in candidates:
        per_exchange[c["exchange"]] = per_exchange.get(c["exchange"], 0) + 1

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_scanned": scanned,
        "exchanges": EXCHANGES,
        "candidates_per_exchange": per_exchange,
        "criteria": {
            "key": f"{BAR_KEY}: StochRSI < {STOCH_OVERSOLD} DAN golden cross DAN struktur BOS/CHoCH bullish (ketiganya wajib)",
            "confirm": f"{BAR_CONFIRM}: RSI {CONFIRM_RSI_MIN}-{CONFIRM_RSI_MAX}, StochRSI <= {CONFIRM_STOCH_MAX}",
            "timing": f"{BAR_TIMING}: StochRSI/RSI sedang naik",
        },
        "candidates": candidates,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Hasil disimpan ke {path}")


# ============================== MAIN ==============================

def main():
    global ACTIVE_CATEGORY_MAP
    print("Mengambil kategori proyek live dari CoinGecko...")
    ACTIVE_CATEGORY_MAP = build_category_map()

    all_targets = []
    for exch in EXCHANGES:
        try:
            symbols = fetch_instruments(exch)
            categorized = []
            for s in symbols:
                cats = passes_category_filter(exch, s)
                if cats:
                    categorized.append((exch, s, cats))
            print(f"[{exch}] Total pair: {len(symbols)} | Lolos filter kategori: {len(categorized)}")
            all_targets += categorized
        except Exception as e:
            print(f"[ERROR] Gagal ambil daftar instrument {exch}: {e}")

    candidates, scanned = [], 0
    for exch, inst_id, cats in all_targets:
        try:
            result = scan_instrument(exch, inst_id)
            scanned += 1
            if result:
                result["categories"] = cats
                result["rank_score"] = rank_score(result)
                candidates.append(result)
                print(f"[LOLOS] [{exch}] {inst_id} ({', '.join(cats)}) skor {result['rank_score']}")
        except Exception as e:
            print(f"[ERROR] [{exch}] {inst_id}: {e}")
        time.sleep(REQUEST_DELAY_SEC)

    candidates.sort(key=lambda c: c["rank_score"], reverse=True)
    print(f"Discan {scanned}/{len(all_targets)} | Lolos: {len(candidates)}")

    save_results_json(candidates, scanned)

    for chunk in format_message(candidates, scanned):
        send_telegram(chunk)
    print("Terkirim.")


if __name__ == "__main__":
    main()
