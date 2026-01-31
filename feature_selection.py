import numpy as np
import pandas as pd
from typing import Tuple, List

from config import TABLE3_TECH_COLS, TABLE2_MIN_COLS, CORR_THRESHOLD


def paper_feature_select(df: pd.DataFrame, keep_id_cols: bool = True) -> Tuple[pd.DataFrame, List[str], List[str]]:
    d = df.copy()

    if "MA20_MA50" not in d.columns and ("SMA_20" in d.columns and "SMA_50" in d.columns):
        d["MA20_MA50"] = (d["SMA_20"] - d["SMA_50"]).astype(float)

    paper_cols = ["MA20_MA50", "RSI", "MACD", "BB_upper", "OBV", "ICHIMOKU_base"]
    keep = [c for c in paper_cols if c in d.columns]
    missing = [c for c in paper_cols if c not in d.columns]

    if "Regime" in d.columns:
        keep.append("Regime")

    cols = []
    if keep_id_cols:
        for c in ["symbol", "date"]:
            if c in d.columns:
                cols.append(c)

    if "Close" in d.columns:
        cols.append("Close")

    cols += keep

    for c in TABLE2_MIN_COLS:
        if c in d.columns:
            cols.append(c)

    cols.append("label")
    return d[cols].copy(), keep, missing


def correlation_select(df: pd.DataFrame, threshold: float = CORR_THRESHOLD, keep_id_cols: bool = True) -> Tuple[pd.DataFrame, List[str], List[str]]:
    tech_df = df[TABLE3_TECH_COLS].copy()
    corr = tech_df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    drop = [c for c in upper.columns if any(upper[c] > threshold)]
    keep = [c for c in TABLE3_TECH_COLS if c not in drop]

    cols = []
    if keep_id_cols:
        for c in ["symbol", "date"]:
            if c in df.columns:
                cols.append(c)

    if "Close" in df.columns:
        cols.append("Close")

    cols += keep
    if "Regime" in df.columns:
        cols.append("Regime")

    for c in TABLE2_MIN_COLS:
        if c in df.columns:
            cols.append(c)

    cols.append("label")
    return df[cols].copy(), keep, drop