import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HGB_AVAILABLE = True
except Exception:
    HGB_AVAILABLE = False

plt.style.use("seaborn-v0_8")

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRICES_DIR = os.path.join(BASE_DIR, "data", "prices")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYMBOL = "AAPL"

START_DISPLAY = "2014-01-02"
END_DISPLAY   = "2024-10-14"       # inclusive display
END_EXCL      = "2024-10-15"       # exclusive filter end

INITIAL_CAPITAL = 10_000.0

# -------- ML (fast + paper-like risk filter) --------
LABEL_HORIZON_DAYS = 5
LABEL_RET_THR = 0.0         # 5-day direction label: >0 means up
WARMUP_BARS = 252
RETRAIN_EVERY = 21          # monthly retrain (fast + better)
ENTER_THR = 0.53            # re-enter long if prob >= ENTER_THR
EXIT_THR  = 0.045          # exit to cash if prob <= EXIT_THR
START_IN_MARKET = 1         # start long like baseline
# ----------------------------------------------------


# ================= TABLE/FIGURE HELPERS =================
def save_table_png(df: pd.DataFrame, title: str, out_path: str, font_size: int = 10):
    fig, ax = plt.subplots(figsize=(14, 1 + 0.55 * (len(df) + 2)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)

    table = ax.table(
        cellText=df.values,
        rowLabels=df.index.tolist(),
        colLabels=df.columns.tolist(),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_trades(close: pd.Series, signal: pd.Series):
    sig = signal.fillna(0).astype(int)
    in_pos = False
    entry_price = None
    trades = []

    for dt, s in sig.items():
        if not in_pos and s == 1:
            in_pos = True
            entry_price = float(close.loc[dt])
        elif in_pos and s == 0:
            exit_price = float(close.loc[dt])
            pnl = (exit_price - entry_price) / entry_price
            trades.append((dt, pnl * 100.0))
            in_pos = False
            entry_price = None

    if in_pos and entry_price is not None:
        dt = sig.index[-1]
        exit_price = float(close.iloc[-1])
        pnl = (exit_price - entry_price) / entry_price
        trades.append((dt, pnl * 100.0))

    return trades


def backtest_equity(close: pd.Series, signal: pd.Series, initial=10000.0):
    s = signal.fillna(0).astype(int)
    ret = close.pct_change().fillna(0.0)
    pos = s.shift(1).fillna(0)           # next-day execution (no look-ahead)
    strat_ret = pos * ret
    equity = (1 + strat_ret).cumprod() * initial
    bh_equity = (1 + ret).cumprod() * initial
    return equity, bh_equity, strat_ret


def plot_strategy_3panel(df: pd.DataFrame, signal_col: str, title: str, out_path: str):
    d = df.copy().sort_index()
    close = d["Close"]
    sig = d[signal_col].fillna(0).astype(int)

    equity, bh_equity, _ = backtest_equity(close, sig, initial=INITIAL_CAPITAL)
    trades = compute_trades(close, sig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(close.index, close.values, label="Close", color="tab:blue")
    buy_idx = (sig.diff() == 1)
    sell_idx = (sig.diff() == -1)
    axes[0].scatter(close.index[buy_idx], close[buy_idx], marker="^", color="green", label="Buy", s=30)
    axes[0].scatter(close.index[sell_idx], close[sell_idx], marker="v", color="red", label="Sell", s=30)
    axes[0].set_title("Orders")
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="upper left")

    if trades:
        t_dates = [t[0] for t in trades]
        t_pnl = [t[1] for t in trades]
        colors = ["green" if p >= 0 else "red" for p in t_pnl]
        axes[1].scatter(t_dates, t_pnl, c=colors, s=30)
    axes[1].axhline(0, linestyle="--", color="gray")
    axes[1].set_title("Trade PnL")
    axes[1].set_ylabel("Trade PnL (%)")

    axes[2].plot(equity.index, equity.values / INITIAL_CAPITAL, label="Value", color="purple")
    axes[2].plot(bh_equity.index, bh_equity.values / INITIAL_CAPITAL, label="Benchmark", color="black", alpha=0.6)
    axes[2].fill_between(equity.index, 1.0, equity.values / INITIAL_CAPITAL, color="green", alpha=0.15)
    axes[2].set_title("Cumulative Returns")
    axes[2].set_ylabel("Cumulative returns (x)")
    axes[2].legend(loc="upper left")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ================= CSV LOADER (supports your AAPL format) =================
def load_price_csv_any(symbol: str) -> pd.DataFrame:
    path = os.path.join(PRICES_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    # Standard
    try:
        df0 = pd.read_csv(path)
        if "Date" in df0.columns:
            df0["Date"] = pd.to_datetime(df0["Date"], errors="coerce")
            df0 = df0.dropna(subset=["Date"]).set_index("Date")
            need = ["Open", "High", "Low", "Close", "Volume"]
            if all(c in df0.columns for c in need):
                df0 = df0[need].apply(pd.to_numeric, errors="coerce").dropna()
                df0 = df0[~df0.index.duplicated(keep="last")].sort_index()
                return df0
    except Exception:
        pass

    # Multi-header
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]

    need = ["Open", "High", "Low", "Close", "Volume"]
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[need].dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ================= INDICATORS =================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["SMA_20"] = c.rolling(20).mean()
    df["SMA_50"] = c.rolling(50).mean()

    d = c.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    df["BB_mid"] = mid
    df["BB_upper"] = mid + 2 * std
    df["BB_lower"] = mid - 2 * std

    obv = [0]
    for i in range(1, len(c)):
        if c.iloc[i] > c.iloc[i - 1]:
            obv.append(obv[-1] + v.iloc[i])
        elif c.iloc[i] < c.iloc[i - 1]:
            obv.append(obv[-1] - v.iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    df["OBV_MA20"] = df["OBV"].rolling(20).mean()

    conv = (h.rolling(9).max() + l.rolling(9).min()) / 2
    base = (h.rolling(26).max() + l.rolling(26).min()) / 2
    df["ICH_conv"] = conv
    df["ICH_base"] = base
    df["ICH_span_a"] = (conv + base) / 2
    df["ICH_span_b"] = (h.rolling(52).max() + l.rolling(52).min()) / 2

    df.dropna(inplace=True)
    return df.sort_index()


# ================= TRADITIONAL SIGNALS (FIXED EXIT LOGIC) =================
def sig_ma(df):
    return (df["SMA_20"] > df["SMA_50"]).astype(int)

def sig_macd(df):
    return (df["MACD"] > df["MACD_signal"]).astype(int)

def sig_obv(df):
    return (df["OBV"] > df["OBV_MA20"]).astype(int)

def sig_rsi(df):
    sig = pd.Series(np.nan, index=df.index, dtype=float)
    sig[df["RSI_14"] < 30] = 1
    sig[df["RSI_14"] > 70] = 0
    return sig.ffill().fillna(0).astype(int)

def sig_bb(df):
    sig = pd.Series(np.nan, index=df.index, dtype=float)
    sig[df["Close"] < df["BB_lower"]] = 1
    sig[df["Close"] > df["BB_mid"]] = 0
    return sig.ffill().fillna(0).astype(int)

def sig_ich(df):
    top = df[["ICH_span_a", "ICH_span_b"]].max(axis=1)
    bot = df[["ICH_span_a", "ICH_span_b"]].min(axis=1)
    sig = pd.Series(np.nan, index=df.index, dtype=float)
    sig[(df["Close"] > top) & (df["ICH_conv"] > df["ICH_base"])] = 1
    sig[(df["Close"] < bot) & (df["ICH_conv"] < df["ICH_base"])] = 0
    return sig.ffill().fillna(0).astype(int)


# ================= ML ALGORITHM SIGNAL (RISK FILTER) =================
def sig_algo_risk_filter(df: pd.DataFrame) -> pd.Series:
    d = df.copy().sort_index()

    # 5-day horizon label
    d["future_ret"] = d["Close"].shift(-LABEL_HORIZON_DAYS) / d["Close"] - 1
    d["y"] = (d["future_ret"] > LABEL_RET_THR).astype(int)

    feat_cols = [
        "SMA_20", "SMA_50", "RSI_14",
        "MACD", "MACD_signal",
        "BB_mid", "BB_upper", "BB_lower",
        "OBV", "OBV_MA20",
        "ICH_conv", "ICH_base", "ICH_span_a", "ICH_span_b",
    ]
    d = d.dropna(subset=feat_cols + ["y"]).copy()

    out = pd.Series(START_IN_MARKET, index=df.index, dtype=int)
    if len(d) < (WARMUP_BARS + 50):
        return out

    X = d[feat_cols].values.astype(np.float32)
    y = d["y"].values.astype(int)
    idx = d.index
    n = len(d)

    def make_model():
        if HGB_AVAILABLE:
            return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, random_state=42)
        return GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)

    i = WARMUP_BARS
    prev_pos = START_IN_MARKET

    while i < n - 1:
        model = make_model()
        model.fit(X[:i], y[:i])

        j = min(i + RETRAIN_EVERY, n)
        probs = model.predict_proba(X[i:j])[:, 1]

        block = []
        for p in probs:
            if p >= ENTER_THR:
                prev_pos = 1
            elif p <= EXIT_THR:
                prev_pos = 0
            block.append(prev_pos)

        out.loc[idx[i:j]] = np.array(block, dtype=int)
        i = j

    return out


def total_return(close: pd.Series, signal: pd.Series):
    equity, _, _ = backtest_equity(close.sort_index(), signal.sort_index(), initial=INITIAL_CAPITAL)
    end_val = float(equity.iloc[-1])
    ret_pct = (end_val / INITIAL_CAPITAL - 1) * 100.0
    return float(ret_pct), end_val


def main():
    df0 = load_price_csv_any(SYMBOL)
    df0 = df0[(df0.index >= pd.to_datetime(START_DISPLAY)) & (df0.index < pd.to_datetime(END_EXCL))].copy()
    df0 = df0.sort_index()

    df = add_indicators(df0)

    # -------- BUY & HOLD sanity check (ADDED) --------
    bh_end = INITIAL_CAPITAL * (df["Close"].iloc[-1] / df["Close"].iloc[0])
    bh_ret = (bh_end / INITIAL_CAPITAL - 1) * 100
    # -----------------------------------------------

    df["sig_ma"] = sig_ma(df)
    df["sig_rsi"] = sig_rsi(df)
    df["sig_macd"] = sig_macd(df)
    df["sig_bb"] = sig_bb(df)
    df["sig_obv"] = sig_obv(df)
    df["sig_ich"] = sig_ich(df)
    df["sig_algo"] = sig_algo_risk_filter(df)

    # Figures
    plot_strategy_3panel(df, "sig_ma",   f"FIGURE 5. MA20_MA50 ({SYMBOL})",             os.path.join(OUTPUT_DIR, "FIGURE_5_MA20_MA50.png"))
    plot_strategy_3panel(df, "sig_rsi",  f"FIGURE 6. RSI ({SYMBOL})",                   os.path.join(OUTPUT_DIR, "FIGURE_6_RSI.png"))
    plot_strategy_3panel(df, "sig_macd", f"FIGURE 7. MACD ({SYMBOL})",                  os.path.join(OUTPUT_DIR, "FIGURE_7_MACD.png"))
    plot_strategy_3panel(df, "sig_bb",   f"FIGURE 8. Bollinger Bands ({SYMBOL})",       os.path.join(OUTPUT_DIR, "FIGURE_8_BB.png"))
    plot_strategy_3panel(df, "sig_obv",  f"FIGURE 9. OBV ({SYMBOL})",                   os.path.join(OUTPUT_DIR, "FIGURE_9_OBV.png"))
    plot_strategy_3panel(df, "sig_ich",  f"FIGURE 10. Ichimoku ({SYMBOL})",             os.path.join(OUTPUT_DIR, "FIGURE_10_ICHIMOKU.png"))
    plot_strategy_3panel(df, "sig_algo", f"FIGURE 11. Approached Algorithm ({SYMBOL})", os.path.join(OUTPUT_DIR, "FIGURE_11_ALGORITHM.png"))

    # Table 8
    r_ma, end_ma = total_return(df["Close"], df["sig_ma"])
    r_rsi, end_rsi = total_return(df["Close"], df["sig_rsi"])
    r_macd, end_macd = total_return(df["Close"], df["sig_macd"])
    r_bb, end_bb = total_return(df["Close"], df["sig_bb"])
    r_obv, end_obv = total_return(df["Close"], df["sig_obv"])
    r_ich, end_ich = total_return(df["Close"], df["sig_ich"])
    r_algo, end_algo = total_return(df["Close"], df["sig_algo"])

    # ===== FINAL SUMMARY (for screenshot / report) =====
    print("\nFINAL SUMMARY")
    print("Period:", START_DISPLAY, "to", END_DISPLAY)
    print("Buy&Hold End Value:", int(bh_end))
    print("Buy&Hold Return %:", round(bh_ret, 2))

    print("Algorithm End Value:", int(end_algo))
    print("Algorithm Return %:", round(r_algo, 2))

    print("Algo in-market %:", round(df["sig_algo"].mean() * 100, 2))
    print("Algo trade count:", int((df["sig_algo"].diff().abs() == 1).sum() / 2))
    print("ENTER_THR:", ENTER_THR, "EXIT_THR:", EXIT_THR)
# ================================================

    table8 = pd.DataFrame({
        "MA20/MA50": [START_DISPLAY, END_DISPLAY, int(INITIAL_CAPITAL), int(end_ma), f"{r_ma:.0f}%"],
        "RSI":       [START_DISPLAY, END_DISPLAY, int(INITIAL_CAPITAL), int(end_rsi), f"{r_rsi:.0f}%"],
        "MACD":      [START_DISPLAY, END_DISPLAY, int(INITIAL_CAPITAL), int(end_macd), f"{r_macd:.0f}%"],
        "Bollinger": [START_DISPLAY, END_DISPLAY, int(INITIAL_CAPITAL), int(end_bb), f"{r_bb:.0f}%"],
        "OBV":       [START_DISPLAY, END_DISPLAY, int(INITIAL_CAPITAL), int(end_obv), f"{r_obv:.0f}%"],
        "Ichimoku":  [START_DISPLAY, END_DISPLAY, int(INITIAL_CAPITAL), int(end_ich), f"{r_ich:.0f}%"],
        "Algorithm": [START_DISPLAY, END_DISPLAY, int(INITIAL_CAPITAL), int(end_algo), f"{r_algo:.0f}%"],
    }, index=["Start", "End", "Start Value USD", "End Value USD", "Total Return"])

    save_table_png(table8, f"TABLE 8. Total return comparison ({SYMBOL})", os.path.join(OUTPUT_DIR, "TABLE_8.png"), font_size=10)
    print("Saved outputs/TABLE_8.png")


if __name__ == "__main__":
    main()