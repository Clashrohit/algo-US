import os
import pandas as pd
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
SP500_CSV = os.path.join(BASE_DIR, "SP500.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PRICES_DIR, exist_ok=True)

START_DATE = "2010-01-01"
END_DATE = "2023-12-31"

MAX_TICKERS = 50
HORIZON_DAYS = 5
N_SPLITS = 5

CORR_THRESHOLD = 0.75

# Signal thresholds
BUY_THR = 0.52
SELL_THR = 0.40

TABLE3_TECH_COLS = [
    "RSI", "STOCH_K", "CCI", "CMO", "COPP", "PPO", "MACD", "EMA_20", "KAMA_10",
    "SMA_20", "SMA_50", "VAMA_20", "TRIMA_20",
    "ICHIMOKU_conv", "ICHIMOKU_base", "ICHIMOKU_span_a", "ICHIMOKU_span_b",
    "BB_upper", "OBV", "MFI_14",
]

TABLE2_MIN_COLS = ["US_Score", "Total_Trend"]

def load_sp500_tickers() -> List[str]:
    sp500_df = pd.read_csv(SP500_CSV)
    if "Symbol" not in sp500_df.columns:
        raise ValueError("SP500.csv must contain 'Symbol' column.")
    tickers = sp500_df["Symbol"].astype(str).tolist()
    tickers = [t.strip().upper().replace(".", "-") for t in tickers if t and t.strip() and t != "nan"]
    # dedupe
    out, seen = [], set()
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:MAX_TICKERS]