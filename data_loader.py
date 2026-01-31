import os
import time
import ssl
from typing import Optional

import pandas as pd
import yfinance as yf

from config import PRICES_DIR

ssl._create_default_https_context = ssl._create_unverified_context


def _finalize_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return None

    df = df[needed].copy()
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([pd.NA, float("inf"), float("-inf")], pd.NA).dropna()
    return df


def _parse_standard_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        if "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
        return _finalize_ohlcv(df)
    except Exception:
        return None


def _parse_yfinance_multiheader_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return _finalize_ohlcv(df)
    except Exception:
        return None


def load_price_csv(symbol: str) -> Optional[pd.DataFrame]:
    path = os.path.join(PRICES_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None

    df = _parse_standard_csv(path)
    if df is not None and not df.empty:
        return df

    df = _parse_yfinance_multiheader_csv(path)
    if df is not None and not df.empty:
        return df

    print(f"[WARN] Could not parse CSV format for {symbol}: {path}")
    return None


def yf_download_retry(symbol: str, start: str, end: str, retries: int = 3) -> Optional[pd.DataFrame]:
    last_err = None
    for i in range(1, retries + 1):
        try:
            df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            if df is not None and not df.empty:
                return _finalize_ohlcv(df)
            last_err = "empty"
        except Exception as e:
            last_err = str(e)
        print(f"[WARN] yfinance failed {symbol} ({i}/{retries}): {last_err}")
        time.sleep(1.5)
    return None


def get_price_data(symbol: str, start: str, end: str, offline_only: bool = True) -> Optional[pd.DataFrame]:
    df = load_price_csv(symbol)
    if df is not None and not df.empty:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        df = df[(df.index >= start_dt) & (df.index < end_dt)]
        if not df.empty:
            return df

    if offline_only:
        return None

    return yf_download_retry(symbol, start, end)