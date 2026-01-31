import os
import time
import ssl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

plt.style.use("seaborn-v0_8")

# SSL workaround (proxy/self-signed)
ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = "data"
PRICES_DIR = os.path.join(DATA_DIR, "prices")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Period shown in your paper figures (change if needed)
START = "2022-01-01"
END   = "2023-01-31"


# ---------- CSV loader (supports both normal + your multi-header format) ----------

def load_price_csv_any(symbol: str):
    path = os.path.join(PRICES_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None

    # Try standard
    try:
        df0 = pd.read_csv(path)
        if "Date" in df0.columns:
            df0["Date"] = pd.to_datetime(df0["Date"], errors="coerce")
            df0 = df0.dropna(subset=["Date"]).set_index("Date")
            need = ["Open", "High", "Low", "Close", "Volume"]
            if all(c in df0.columns for c in need):
                df0 = df0[need].apply(pd.to_numeric, errors="coerce").dropna()
                return df0
    except Exception:
        pass

    # Try yfinance multi-header
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]
        need = ["Open", "High", "Low", "Close", "Volume"]
        for c in need:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[need].dropna()
        return df
    except Exception:
        return None


# ---------- yfinance fallback with retry (won't crash) ----------

def yf_download_retry(symbol: str, start: str, end: str, retries: int = 3):
    try:
        import yfinance.shared as shared
        shared._DEFAULTS["timeout"] = 60
    except Exception:
        pass

    last = None
    for i in range(1, retries + 1):
        try:
            df = yf.download(symbol, start=start, end=end, auto_adjust=True,
                             progress=False, threads=False)
            if df is not None and not df.empty:
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                return df
            last = "empty"
        except Exception as e:
            last = str(e)
        print(f"[WARN] Download failed {symbol} ({i}/{retries}): {last}")
        time.sleep(1.5)
    return None


def get_close_series(symbol: str, start: str, end: str):
    # offline first
    df = load_price_csv_any(symbol)
    if df is not None and not df.empty:
        df = df[(df.index >= pd.to_datetime(start)) & (df.index < pd.to_datetime(end))]
        if not df.empty:
            return df["Close"]

    # online fallback
    df = yf_download_retry(symbol, start, end, retries=3)
    if df is None or df.empty:
        return None
    return df["Close"]


# ---------- Plot like paper Figure 1/2/3 ----------

def plot_index_figure(tickers, title, out_path, start=START, end=END):
    series_list = []
    names = []

    for t in tickers:
        s = get_close_series(t, start, end)
        if s is None or s.empty:
            print(f"[SKIP] No data for {t}")
            continue
        series_list.append(s)
        names.append(t)

    if len(series_list) == 0:
        print(f"[ERROR] No ticker data available for: {title}")
        return

    # Align by date
    df_close = pd.concat(series_list, axis=1)
    df_close.columns = names
    df_close.dropna(inplace=True)

    # Normalized return (start=0)
    norm = (df_close / df_close.iloc[0]) - 1.0

    dates = norm.index
    x = np.arange(len(dates))

    plt.figure(figsize=(10, 3.5))
    for col in norm.columns:
        y = norm[col].values
        plt.plot(dates, y, label=col)

        # dotted linear trend
        m, b = np.polyfit(x, y, 1)
        plt.plot(dates, m * x + b, linestyle=":", color="k", linewidth=1)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Paper selections (change if you want)
    sp500 = ["HES", "ON", "SLB", "XOM"]
    nasdaq100 = ["CDNS", "MNST", "ODFL", "ROST", "VRTX"]
    dow = ["CVX", "MCD", "TRV"]

    plot_index_figure(
        sp500,
        title="FIGURE 1. S&P 500.",
        out_path=os.path.join(OUTPUT_DIR, "FIGURE_1_SP500.png")
    )

    plot_index_figure(
        nasdaq100,
        title="FIGURE 2. NASDAQ100.",
        out_path=os.path.join(OUTPUT_DIR, "FIGURE_2_NASDAQ100.png")
    )

    plot_index_figure(
        dow,
        title="FIGURE 3. Dow Jones.",
        out_path=os.path.join(OUTPUT_DIR, "FIGURE_3_DOWJONES.png")
    )


if __name__ == "__main__":
    main()