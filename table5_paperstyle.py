import os
import ssl
import time
import numpy as np
import pandas as pd
import yfinance as yf

from report_paperstyle import save_table_png

# SSL workaround
ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = "data"
PRICES_DIR = os.path.join(DATA_DIR, "prices")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OFFLINE_ONLY = True   # keep True for your environment
EPS = 0.001           # neutral threshold

BACKWARD_START = "2022-01-01"
BACKWARD_END   = "2022-12-31"
FORWARD_START  = "2023-01-01"
FORWARD_END    = "2023-12-31"


# ---- CSV loader (supports normal + yfinance multi-header) ----
def load_price_csv_any(symbol: str):
    path = os.path.join(PRICES_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None

    # Standard format
    try:
        df0 = pd.read_csv(path)
        if "Date" in df0.columns:
            df0["Date"] = pd.to_datetime(df0["Date"], errors="coerce")
            df0 = df0.dropna(subset=["Date"]).set_index("Date")
            needed = ["Open", "High", "Low", "Close", "Volume"]
            if all(c in df0.columns for c in needed):
                df0 = df0[needed].apply(pd.to_numeric, errors="coerce").dropna()
                return df0
    except Exception:
        pass

    # Multi-header format
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]
        needed = ["Open", "High", "Low", "Close", "Volume"]
        for c in needed:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[needed].dropna()
        return df
    except Exception:
        return None


def yf_download_retry(symbol: str, start: str, end: str, retries: int = 3):
    try:
        import yfinance.shared as shared
        shared._DEFAULTS["timeout"] = 60
    except Exception:
        pass

    last = None
    for i in range(1, retries + 1):
        try:
            df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            if df is not None and not df.empty:
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                return df
            last = "empty"
        except Exception as e:
            last = str(e)
        print(f"[WARN] yfinance failed {symbol} ({i}/{retries}): {last}")
        time.sleep(1.5)
    return None


def get_close(symbol: str, start: str, end: str):
    df = load_price_csv_any(symbol)
    if df is not None and not df.empty:
        df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        if not df.empty:
            return df["Close"]

    if OFFLINE_ONLY:
        return None

    df = yf_download_retry(symbol, start, end, retries=3)
    if df is None or df.empty:
        return None
    df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
    return df["Close"] if not df.empty else None


def period_return(symbol: str, start: str, end: str):
    s = get_close(symbol, start, end)
    if s is None or s.empty:
        return None
    # entry = first available
    entry = float(s.iloc[0])
    exitp = float(s.iloc[-1])
    return (exitp - entry) / entry


def compute_win_neutral_loss(selected: list[str], benchmark_symbol: str, start: str, end: str):
    bench = period_return(benchmark_symbol, start, end)
    if bench is None:
        # If benchmark missing, return NA safely
        return (0, 0, 0), None

    win = neu = loss = 0
    rets = []
    for sym in selected:
        r = period_return(sym, start, end)
        if r is None:
            continue
        rets.append(r)
        diff = r - bench
        if diff > EPS:
            win += 1
        elif diff < -EPS:
            loss += 1
        else:
            neu += 1

    # equal-weight cumulative return
    cum_ret = float(np.mean(rets)) if rets else 0.0
    return (win, neu, loss), cum_ret


def main():
    """
    Paper Table-5 uses these selected tickers (from your Figures 1-3):
      S&P500:   HES, ON, SLB, XOM  (4)
      NASDAQ100: CDNS, MNST, ODFL, ROST, VRTX (5)
      Dow: CVX, MCD, TRV (3)

    Benchmarks (ETF) (offline CSV recommended):
      SPY for S&P500, QQQ for NASDAQ100, DIA for Dow
    """
    indices = [
        ("S&P 500", 503, ["HES", "ON", "SLB", "XOM"], "SPY"),
        ("Nasdaq100", 100, ["CDNS", "MNST", "ODFL", "ROST", "VRTX"], "QQQ"),
        ("Dow Jones", 30, ["CVX", "MCD", "TRV"], "DIA"),
    ]

    rows = []
    for idx_name, total, selected, bench in indices:
        # Backward
        (w1, n1, l1), cr1 = compute_win_neutral_loss(selected, bench, BACKWARD_START, BACKWARD_END)
        # Forward
        (w2, n2, l2), cr2 = compute_win_neutral_loss(selected, bench, FORWARD_START, FORWARD_END)

        rows.append({
            "Index": idx_name,
            "Total": total,
            "Hit": len(selected),
            "Backward WIN:Neutral:LOSS": f"{w1}:{n1}:{l1}",
            "Backward Cumulative Return": "" if cr1 is None else f"{cr1*100:.2f}%",
            "Forward WIN:Neutral:LOSS": f"{w2}:{n2}:{l2}",
            "Forward Cumulative Return": "" if cr2 is None else f"{cr2*100:.2f}%",
        })

    df = pd.DataFrame(rows)
    save_table_png(
        df,
        title="TABLE 5. Backward test and forward test.",
        out_path=os.path.join(OUTPUT_DIR, "TABLE_5.png"),
        font_size=10
    )
    print("Saved outputs/TABLE_5.png")
    print("Note: For accurate win/loss vs benchmark, keep SPY.csv, QQQ.csv, DIA.csv in data/prices/ (offline).")


if __name__ == "__main__":
    main()