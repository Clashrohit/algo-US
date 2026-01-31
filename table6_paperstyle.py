import os
import ast
import ssl
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

ssl._create_default_https_context = ssl._create_unverified_context
plt.style.use("seaborn-v0_8")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# If your internet is bad, keep True (uses only local CSVs in data/prices/)
OFFLINE_ONLY = True

EVAL_DATES = [
    ("Return (%) on Dec 2021", "2021-12-31"),
    ("Return (%) on Dec 2022", "2022-12-31"),
    ("Return (%) on Dec 2023", "2023-12-31"),
    ("Return (%) on June 2024", "2024-06-30"),
]

# --- Hardcoded Table 6 recommended stocks (from your paper screenshots) ---
DCA_ROWS = [
    # No, DCA_date, Recommend_Stock
    (1,  "2020-01-01", ["AAPL","AMD","TSLA"]),
    (2,  "2020-02-01", ["AMD","NVDA","TSLA"]),
    (3,  "2020-03-01", []),
    (4,  "2020-04-01", ["AMD","MRNA","NVDA"]),
    (5,  "2020-05-01", ["AMD","MRNA","NVDA","TSLA"]),
    (6,  "2020-06-01", ["NVDA","TSLA"]),
    (7,  "2020-07-01", ["AAPL","AMD","MRNA","NVDA","PYPL","TSLA"]),
    (8,  "2020-08-01", ["AAPL","AMD","NOW","NVDA","PYPL","TSLA"]),
    (9,  "2020-09-01", ["NVDA","TSLA"]),
    (10, "2020-10-01", ["AAPL","CRM","NVDA","PYPL","TSLA"]),
    (11, "2020-11-01", ["AAPL","AMD","CRM","MRNA","NVDA","QCOM","TSLA"]),
    (12, "2020-12-01", ["AAPL","AMD","AVGO","CRWD","MRNA","NKE","PYPL","QCOM","TSLA","UBER"]),

    (13, "2021-01-01", ["AAPL","DD","ETSY","F","GM","GS","MU","PYPL","QCOM","TSLA","UBER"]),
    (14, "2021-02-01", ["DIS","GM","GOOG","GOOGL","LRCX","MRNA","MU","NVDA","PYPL","TSLA","UBER"]),
    (15, "2021-03-01", ["AAL","AMAT","AVGO","BA","BAC","C","CAT","CCL","CZR","DIS","F","GE","GM","GNRC","GS","MU","NXPI","PARA","UAL"]),
    (16, "2021-04-01", ["AMAT","BA","MRNA","MU","NVDA","PYPL","TSLA","UBER","WFC"]),
    (17, "2021-05-01", ["FCX","WFC"]),
    (18, "2021-06-01", ["F","GM","MRNA","NVDA"]),
    (19, "2021-07-01", ["MRNA"]),
    (20, "2021-08-01", ["WFC"]),
    (21, "2021-09-01", ["AMAT","GOOG","GOOGL","GS","MRNA","WFC"]),
    (22, "2021-10-01", ["F","GS","MS","WFC"]),
    (23, "2021-11-01", ["AMD","F","NVDA","TSLA","WFC"]),
    (24, "2021-12-01", ["EPAM","F","NVDA"]),

    (25, "2022-01-01", ["F"]),
    (26, "2022-02-01", []),
    (27, "2022-03-01", ["DVN","OXY"]),
    (28, "2022-04-01", ["OXY"]),
    (29, "2022-05-01", ["OXY"]),
    (30, "2022-06-01", ["OXY"]),
    (31, "2022-07-01", []),
    (32, "2022-08-01", ["OXY"]),
    (33, "2022-09-01", ["ENPH","OXY"]),
    (34, "2022-10-01", []),
    (35, "2022-11-01", ["OXY"]),
    (36, "2022-12-01", ["ENPH"]),

    (37, "2023-01-01", []),
    (38, "2023-02-01", []),
    (39, "2023-03-01", []),
    (40, "2023-04-01", ["NFLX","NVDA"]),
    (41, "2023-05-01", ["NVDA"]),
    (42, "2023-06-01", ["AMD","META","NFLX","NVDA"]),
    (43, "2023-07-01", ["AMD","CCL","META","NFLX","NVDA"]),
    (44, "2023-08-01", ["AVGO","META","NFLX","NVDA"]),
    (45, "2023-09-01", ["ADBE","META","NFLX","NVDA","ORCL"]),
    (46, "2023-10-01", ["META","TSLA"]),
    (47, "2023-11-01", ["META","NVDA"]),
    (48, "2023-12-01", ["META","NVDA"]),
]

# ---------------- helpers ----------------

def save_table_png(df: pd.DataFrame, title: str, out_path: str, font_size: int = 8):
    fig, ax = plt.subplots(figsize=(14, 1 + 0.45 * (len(df) + 2)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_price_csv_any(symbol: str):
    """
    Supports both standard and your multi-header format.
    data/prices/<SYMBOL>.csv
    """
    path = os.path.join(PRICES_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None

    # Standard
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

    # Multi-header
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


def portfolio_return(tickers, start_date, end_date):
    if not tickers:
        return None

    rets = []
    for sym in tickers:
        s = get_close(sym, start_date, end_date)
        if s is None or s.empty:
            continue
        entry = float(s.iloc[0])
        exitp = float(s.iloc[-1])
        rets.append((exitp - entry) / entry)

    if not rets:
        return None
    return float(np.mean(rets)) * 100.0


def main():
    # build base table
    df = pd.DataFrame({
        "No": [r[0] for r in DCA_ROWS],
        "DCA date": [r[1] for r in DCA_ROWS],
        "Total": [len(r[2]) for r in DCA_ROWS],
        "Recommend Stock": [str(r[2]) for r in DCA_ROWS],
    })
    df["DCA date"] = pd.to_datetime(df["DCA date"])

    # compute returns columns (blank if prices missing)
    for col, ed in EVAL_DATES:
        vals = []
        for d, tickers in zip(df["DCA date"], [r[2] for r in DCA_ROWS]):
            if d > pd.to_datetime(ed):
                vals.append("")
                continue
            r = portfolio_return(tickers, d.strftime("%Y-%m-%d"), ed)
            vals.append("" if r is None else round(r, 2))
        df[col] = vals

    # summary row (mean of available)
    summary = {"No": "", "DCA date": "DCA Total Return Year to Date (%)", "Total": "", "Recommend Stock": ""}
    for col, _ in EVAL_DATES:
        v = pd.to_numeric(df[col], errors="coerce")
        summary[col] = round(v.dropna().mean(), 2) if v.notna().any() else ""
    df_out = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    # split into two parts (like paper)
    part1 = df_out.iloc[:24].copy()
    part2 = df_out.iloc[24:].copy()

    p1 = os.path.join(OUTPUT_DIR, "TABLE_6_part1.png")
    p2 = os.path.join(OUTPUT_DIR, "TABLE_6_part2.png")

    save_table_png(part1, "TABLE 6. DCA performance and recommended stock overview (Part 1).", p1, font_size=8)
    save_table_png(part2, "TABLE 6. (Continued.) DCA performance and recommended stock overview (Part 2).", p2, font_size=8)

    print("Saved:", p1)
    print("Saved:", p2)
    print("Note: If returns are blank, add required ticker CSVs into data/prices/.")


if __name__ == "__main__":
    main()