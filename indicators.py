import numpy as np
import pandas as pd

def _kama(series: pd.Series, n=10) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def add_indicators_table3(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["STOCH_K"] = 100 * (c - low14) / (high14 - low14)

    tp = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    md = (tp - sma_tp).abs().rolling(20).mean()
    df["CCI"] = (tp - sma_tp) / (0.015 * md)

    sum_up = gain.rolling(14).sum()
    sum_down = loss.rolling(14).sum()
    df["CMO"] = 100 * (sum_up - sum_down) / (sum_up + sum_down)

    roc11 = c.pct_change(11) * 100
    roc14 = c.pct_change(14) * 100
    roc_sum = roc11 + roc14
    weights = np.arange(1, 11)
    df["COPP"] = roc_sum.rolling(10).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    df["EMA_20"] = c.ewm(span=20, adjust=False).mean()
    df["SMA_20"] = c.rolling(20).mean()
    df["SMA_50"] = c.rolling(50).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["PPO"] = (df["MACD"] / ema26) * 100

    df["KAMA_10"] = _kama(c, n=10)
    df["VAMA_20"] = (c * v).rolling(20).sum() / v.rolling(20).sum()
    df["TRIMA_20"] = c.rolling(20).mean().rolling(20).mean()

    conv = (h.rolling(9).max() + l.rolling(9).min()) / 2
    base = (h.rolling(26).max() + l.rolling(26).min()) / 2
    df["ICHIMOKU_conv"] = conv
    df["ICHIMOKU_base"] = base
    df["ICHIMOKU_span_a"] = (conv + base) / 2
    df["ICHIMOKU_span_b"] = (h.rolling(52).max() + l.rolling(52).min()) / 2

    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    df["BB_upper"] = mid + 2 * std

    obv = [0]
    for i in range(1, len(c)):
        if c.iloc[i] > c.iloc[i - 1]:
            obv.append(obv[-1] + v.iloc[i])
        elif c.iloc[i] < c.iloc[i - 1]:
            obv.append(obv[-1] - v.iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    tp2 = (h + l + c) / 3
    rmf = tp2 * v
    prev_tp = tp2.shift(1)
    pos = rmf.where(tp2 > prev_tp, 0.0)
    neg = rmf.where(tp2 < prev_tp, 0.0)
    mfr = pos.rolling(14).sum() / neg.rolling(14).sum().abs()
    df["MFI_14"] = 100 - (100 / (1 + mfr))

    df["52w_high"] = c.rolling(252).max()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def compute_us_score(last_row: pd.Series) -> int:
    score = 0
    if last_row["Close"] < last_row["SMA_20"]:
        score += 1
    if last_row["Close"] < last_row["SMA_50"]:
        score += 1
    if last_row["RSI"] < 40:
        score += 1
    if last_row["52w_high"] > 0:
        disc = 1 - last_row["Close"] / last_row["52w_high"]
        if disc > 0.20:
            score += 1
    return score