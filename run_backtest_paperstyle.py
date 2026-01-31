import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from backtest import backtest_topn_portfolio

def perf_from_daily(daily: pd.DataFrame):
    # daily: columns: date, port_ret, equity
    equity = daily["equity"].astype(float)
    r = daily["port_ret"].astype(float).fillna(0.0)

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = float(dd.min())

    sharpe = 0.0 if r.std() == 0 else float((252 ** 0.5) * r.mean() / r.std())
    win_rate = float((r[r != 0] > 0).mean()) if (r != 0).any() else 0.0

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
    }

def run_period(df_all, start, end, top_n, fee_bps, initial, buffer_days=10):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # buffer to allow "t-1 execution" on the first day of the period
    buffer_start = start - pd.Timedelta(days=buffer_days)

    dfw = df_all[(df_all["date"] >= buffer_start) & (df_all["date"] <= end)].copy()
    if dfw.empty:
        raise ValueError(f"No rows in window {buffer_start.date()}..{end.date()}")

    daily, _ = backtest_topn_portfolio(
        dfw,
        top_n=top_n,
        price_col="Close",
        prob_col="prob_buy",
        date_col="date",
        symbol_col="symbol",
        initial=initial,
        fee_bps=fee_bps
    )

    # trim to exact period for reporting
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[(daily["date"] >= start) & (daily["date"] <= end)].reset_index(drop=True)
    if daily.empty:
        raise ValueError("Daily backtest is empty after trimming to period.")

    metrics = {"start": str(start.date()), "end": str(end.date()), "top_n": top_n, "fee_bps": fee_bps}
    metrics.update(perf_from_daily(daily))
    return daily, metrics

def save_metrics_table_png(rows, out_path):
    df = pd.DataFrame(rows)
    show = df[["period","start","end","top_n","fee_bps","total_return","max_drawdown","sharpe","win_rate"]].copy()
    # percentage formatting
    show["total_return"] = (show["total_return"] * 100).round(2).astype(str) + "%"
    show["max_drawdown"] = (show["max_drawdown"] * 100).round(2).astype(str) + "%"
    show["win_rate"] = (show["win_rate"] * 100).round(2).astype(str) + "%"

    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.axis("off")
    ax.set_title("Paper-style Backtest: Backward (2022) vs Forward (2023)", fontsize=12, pad=12)
    tbl = ax.table(
        cellText=show.values,
        colLabels=show.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", default="outputs/trading_signals_best_model.csv")
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--fee_bps", type=float, default=10.0)
    ap.add_argument("--initial", type=float, default=10000.0)
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.signals)
    need = {"symbol", "date", "Close", "prob_buy"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Signals file missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "Close", "prob_buy"]).copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)

    # --- Paper-style periods ---
    periods = [
        ("Backward_2022", "2022-01-01", "2022-12-31"),
        ("Forward_2023",  "2023-01-01", "2023-12-31"),
    ]

    all_rows = []
    equity_plot = []

    for name, start, end in periods:
        daily, m = run_period(df, start, end, args.top_n, args.fee_bps, args.initial, buffer_days=10)

        # save daily + metrics
        daily_path = os.path.join(args.out_dir, f"backtest_{name}_daily.csv")
        daily.to_csv(daily_path, index=False)

        metrics_path = os.path.join(args.out_dir, f"backtest_{name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(m, f, indent=2)

        print(f"\n[{name}] saved:")
        print("  daily  :", daily_path)
        print("  metrics:", metrics_path)
        print("  metrics:", m)

        all_rows.append({"period": name, **m})
        equity_plot.append((name, daily))

    # --- combined equity plot ---
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(10, 4))
    for name, daily in equity_plot:
        plt.plot(pd.to_datetime(daily["date"]), daily["equity"], label=name)
    plt.title(f"Top-{args.top_n} Portfolio Equity (fee={args.fee_bps} bps)")
    plt.xlabel("Date")
    plt.ylabel("Equity (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(args.out_dir, "portfolio_equity_curve_2022_2023.png")
    plt.savefig(fig_path, dpi=250, bbox_inches="tight")
    plt.close()
    print("\nSaved plot:", fig_path)

    # --- table png ---
    table_path = os.path.join(args.out_dir, "TABLE_backtest_2022_2023.png")
    save_metrics_table_png(all_rows, table_path)
    print("Saved table:", table_path)

if __name__ == "__main__":
    main()