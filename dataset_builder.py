import numpy as np
import pandas as pd
from typing import List

from config import START_DATE, END_DATE, OUTPUT_DIR, TABLE3_TECH_COLS, TABLE2_MIN_COLS
from data_loader import get_price_data
from indicators import add_indicators_table3, compute_us_score


def _add_regime_feature(d: pd.DataFrame, lookback: int = 20, eps: float = 0.0005) -> pd.DataFrame:
    """
    Regime (past-only):
      2 = Bull, 1 = Sideways, 0 = Bear
    Based on rolling mean of log returns.
    """
    out = d.copy()
    r = np.log(out["Close"] / out["Close"].shift(1))
    mu = r.rolling(lookback).mean()

    regime = np.ones(len(out), dtype=int)
    regime[mu > eps] = 2
    regime[mu < -eps] = 0
    out["Regime"] = regime.astype(int)
    return out


def build_timeseries_dataset(
    tickers: List[str],
    horizon_days: int = 5,
    offline_only: bool = True,
    save_file: bool = False,
    add_regime: bool = True,
    regime_lookback: int = 20,
    regime_eps: float = 0.0005
) -> pd.DataFrame:
    frames = []

    for sym in tickers:
        df = get_price_data(sym, START_DATE, END_DATE, offline_only=offline_only)
        if df is None or df.empty or df.shape[0] < 400:
            continue

        d = add_indicators_table3(df).sort_index()
        if d.empty:
            continue

        if add_regime:
            d = _add_regime_feature(d, lookback=regime_lookback, eps=regime_eps)

        future_close = d["Close"].shift(-horizon_days)
        valid = future_close.notna()
        if valid.sum() == 0:
            continue

        d = d.loc[valid].copy()
        d["label"] = (future_close.loc[valid] > d["Close"]).astype(int)

        if "US_Score" in TABLE2_MIN_COLS:
            d["US_Score"] = d.apply(compute_us_score, axis=1).astype(float)

        if "Total_Trend" in TABLE2_MIN_COLS:
            first_close = d["Close"].iloc[0]
            d["Total_Trend"] = (d["Close"] / first_close - 1.0).astype(float)

        use_cols = (
            ["Close"]
            + TABLE3_TECH_COLS
            + (["Regime"] if add_regime else [])
            + [c for c in TABLE2_MIN_COLS if c in d.columns]
            + ["label"]
        )
        use_cols = [c for c in use_cols if c in d.columns]

        out = d[use_cols].copy()
        out["symbol"] = sym
        out["date"] = out.index
        frames.append(out.reset_index(drop=True))

    out_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if save_file and not out_all.empty:
        out_all.to_csv(f"{OUTPUT_DIR}/timeseries_dataset.csv", index=False)

    return out_all


def build_us_timeseries_dataset(
    tickers: List[str],
    offline_only: bool = True,
    us_threshold: int = 3,
    save_file: bool = False,
    add_regime: bool = True,
    regime_lookback: int = 20,
    regime_eps: float = 0.0005
) -> pd.DataFrame:
    frames = []

    for sym in tickers:
        df = get_price_data(sym, START_DATE, END_DATE, offline_only=offline_only)
        if df is None or df.empty or df.shape[0] < 400:
            continue

        d = add_indicators_table3(df).sort_index()
        if d.empty:
            continue

        if add_regime:
            d = _add_regime_feature(d, lookback=regime_lookback, eps=regime_eps)

        score = (
            (d["Close"] < d["SMA_20"]).astype(int)
            + (d["Close"] < d["SMA_50"]).astype(int)
            + (d["RSI"] < 40).astype(int)
        )
        disc = 1 - (d["Close"] / d["52w_high"])
        score = score + (disc > 0.20).astype(int)

        d["US_Score"] = score.astype(float)
        d["label"] = (score >= us_threshold).astype(int)

        if "Total_Trend" in TABLE2_MIN_COLS:
            first_close = d["Close"].iloc[0]
            d["Total_Trend"] = (d["Close"] / first_close - 1.0).astype(float)

        use_cols = (
            ["Close"]
            + TABLE3_TECH_COLS
            + (["Regime"] if add_regime else [])
            + [c for c in TABLE2_MIN_COLS if c in d.columns]
            + ["US_Score", "label"]
        )
        use_cols = [c for c in use_cols if c in d.columns]

        out = d[use_cols].copy()
        out["symbol"] = sym
        out["date"] = out.index
        frames.append(out.reset_index(drop=True))

    out_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if save_file and not out_all.empty:
        out_all.to_csv(f"{OUTPUT_DIR}/us_timeseries_dataset.csv", index=False)

    return out_all