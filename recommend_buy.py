import os
import pandas as pd
import matplotlib.pyplot as plt

TOP_N = 10
SIGNALS_CSV = "outputs/trading_signals_best_model.csv"
SP500_CSV = "SP500.csv"
OUT_DIR = "outputs"

# keep consistent with your config.py
BUY_THR = 0.52
SELL_THR = 0.40

os.makedirs(OUT_DIR, exist_ok=True)

def save_table_png(df: pd.DataFrame, title: str, out_path: str, font_size: int = 10):
    plt.style.use("seaborn-v0_8")
    fig_w = 14
    fig_h = 1 + 0.45 * (len(df) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
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
    table.scale(1, 1.35)

    # header style
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E6E6E6")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def ensure_signal(df: pd.DataFrame) -> pd.DataFrame:
    """If signal missing, compute from prob_buy using thresholds."""
    if "signal" not in df.columns:
        if "prob_buy" not in df.columns:
            raise ValueError("Signals file must contain 'prob_buy' (and preferably 'signal').")
        df = df.copy()
        df["signal"] = df["prob_buy"].apply(lambda p: 1 if p >= BUY_THR else (-1 if p <= SELL_THR else 0))
    return df

def topn_or_all(df: pd.DataFrame, n: int, ascending: bool) -> pd.DataFrame:
    if df.empty:
        return df
    if len(df) <= n:
        return df.sort_values("prob_buy", ascending=ascending).copy()
    return df.sort_values("prob_buy", ascending=ascending).head(n).copy()

# --------- Load signals ----------
df = pd.read_csv(SIGNALS_CSV)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "symbol"])
df["symbol"] = df["symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)

df = ensure_signal(df)

latest_date = df["date"].max()
snap = df[df["date"] == latest_date].copy()

# --------- Merge company info ----------
sp = pd.read_csv(SP500_CSV)
sp = sp.rename(columns={"Symbol": "symbol", "Name": "company"})
sp["symbol"] = sp["symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)

merge_cols = ["symbol"]
if "company" in sp.columns:
    merge_cols.append("company")
if "Sector" in sp.columns:
    merge_cols.append("Sector")

snap = snap.merge(sp[merge_cols], on="symbol", how="left")

# --------- STRICT lists by signal ----------
buy_pool = snap[snap["signal"] == 1].copy()
sell_pool = snap[snap["signal"] == -1].copy()
hold_pool = snap[snap["signal"] == 0].copy()

buy = topn_or_all(buy_pool, TOP_N, ascending=False)
buy["final_action"] = "BUY (signal=1)"

sell = topn_or_all(sell_pool, TOP_N, ascending=True)
sell["final_action"] = "SELL/EXIT (signal=-1)"  # long-only exit/avoid

# HOLD: choose Top-N closest to mid (between thresholds)
mid = (BUY_THR + SELL_THR) / 2.0
hold_pool["dist_to_mid"] = (hold_pool["prob_buy"].astype(float) - mid).abs()
hold = hold_pool.sort_values("dist_to_mid", ascending=True).head(TOP_N).copy()
hold.drop(columns=["dist_to_mid"], inplace=True, errors="ignore")
hold["final_action"] = "HOLD (signal=0)"

# --------- Final output formatting ----------
cols = ["symbol"]
if "company" in snap.columns:
    cols.append("company")
if "Sector" in snap.columns:
    cols.append("Sector")
cols += ["date", "prob_buy", "signal", "final_action"]

def finalize(dfx: pd.DataFrame) -> pd.DataFrame:
    out = dfx[cols].copy() if not dfx.empty else pd.DataFrame(columns=cols)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out["prob_buy"] = out["prob_buy"].astype(float).round(4)
        out["signal"] = out["signal"].astype(int)
    return out

buy_out = finalize(buy)
sell_out = finalize(sell)
hold_out = finalize(hold)

# --------- Save CSVs ----------
buy_csv = os.path.join(OUT_DIR, "recommended_buy_list.csv")
sell_csv = os.path.join(OUT_DIR, "recommended_sell_list.csv")
hold_csv = os.path.join(OUT_DIR, "recommended_hold_list.csv")

buy_out.to_csv(buy_csv, index=False)
sell_out.to_csv(sell_csv, index=False)
hold_out.to_csv(hold_csv, index=False)

# --------- Save PNGs ----------
date_str = str(latest_date.date())
buy_png = os.path.join(OUT_DIR, "recommended_buy_list.png")
sell_png = os.path.join(OUT_DIR, "recommended_sell_list.png")
hold_png = os.path.join(OUT_DIR, "recommended_hold_list.png")

save_table_png(buy_out,  f"Top-{TOP_N} BUY List (signal=1, date: {date_str})",  buy_png)
save_table_png(sell_out, f"Top-{TOP_N} SELL/EXIT List (signal=-1, date: {date_str})", sell_png)
save_table_png(hold_out, f"Top-{TOP_N} HOLD List (signal=0, date: {date_str})", hold_png)

# --------- Print summary ----------
print("Latest date:", latest_date.date())
print(f"\nBUY count on latest date: {len(buy_pool)} (showing up to {TOP_N})")
print(f"SELL count on latest date: {len(sell_pool)} (showing up to {TOP_N})")
print(f"HOLD count on latest date: {len(hold_pool)} (showing up to {TOP_N})")

print("\nSaved CSVs:")
print(" ", buy_csv)
print(" ", sell_csv)
print(" ", hold_csv)

print("\nSaved PNGs:")
print(" ", buy_png)
print(" ", sell_png)
print(" ", hold_png)