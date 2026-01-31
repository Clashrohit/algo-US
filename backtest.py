import numpy as np
import pandas as pd


def backtest_equity_single(
    df: pd.DataFrame,
    symbol: str,
    price_col: str = "Close",
    signal_col: str = "signal",
    date_col: str = "date",
    initial: float = 10000.0,
    allow_short: bool = False
):
    """
    Single-stock backtest.
    Use when df has multiple symbols. We filter one symbol and backtest it.
    """

    d = df.copy()
    if "symbol" in d.columns:
        d = d[d["symbol"] == symbol].copy()
    if d.empty:
        raise ValueError(f"No rows found for symbol={symbol}")

    # sort by date
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    else:
        d = d.sort_index()

    # returns
    d["ret"] = d[price_col].pct_change().fillna(0.0)

    # position from signal (next day execution)
    raw_pos = d[signal_col].shift(1).fillna(0.0)

    if allow_short:
        d["pos"] = raw_pos.clip(-1, 1).astype(float)
    else:
        d["pos"] = (raw_pos > 0).astype(float)   # -1 => exit

    d["strat_ret"] = d["pos"] * d["ret"]
    d["equity"] = (1 + d["strat_ret"]).cumprod() * initial

    total_return = float(d["equity"].iloc[-1] / initial - 1)
    roll_max = d["equity"].cummax()
    dd = d["equity"] / roll_max - 1
    max_dd = float(dd.min())

    r = d["strat_ret"].fillna(0.0)
    sharpe = 0.0 if r.std() == 0 else float(np.sqrt(252) * r.mean() / r.std())
    win_rate = float((r[r != 0] > 0).mean()) if (r != 0).any() else 0.0

    metrics = {
        "symbol": symbol,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate
    }

    return d, metrics


def backtest_topn_portfolio(
    df: pd.DataFrame,
    top_n: int = 10,
    price_col: str = "Close",
    prob_col: str = "prob_buy",
    date_col: str = "date",
    symbol_col: str = "symbol",
    initial: float = 10000.0,
    fee_bps: float = 0.0,   # <-- ADD THIS
):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col, symbol_col, price_col, prob_col]).copy()

    d = d.sort_values([symbol_col, date_col]).reset_index(drop=True)
    d["ret"] = d.groupby(symbol_col)[price_col].pct_change().fillna(0.0)

    d = d.sort_values([date_col, symbol_col]).reset_index(drop=True)
    d["rank"] = d.groupby(date_col)[prob_col].rank(ascending=False, method="first")
    d["selected_today"] = (d["rank"] <= top_n).astype(int)

    # execute using previous DATE (not per-symbol row shift)
    dates = np.array(sorted(d[date_col].unique()))
    prev_map = {pd.Timestamp(dates[i]): pd.Timestamp(dates[i-1]) for i in range(1, len(dates))}
    d["prev_date"] = d[date_col].map(prev_map)

    prev_sel = d[[date_col, symbol_col, "selected_today"]].rename(
        columns={date_col: "prev_date", "selected_today": "pos"}
    )
    d = d.merge(prev_sel, on=["prev_date", symbol_col], how="left")
    d["pos"] = d["pos"].fillna(0).astype(int)

    # transaction cost on position change
    d = d.sort_values([symbol_col, date_col]).reset_index(drop=True)
    d["pos_change"] = d.groupby(symbol_col)["pos"].diff().abs().fillna(d["pos"].abs())
    d["cost"] = d["pos_change"] * (fee_bps / 10000.0)

    d["strategy_ret"] = d["pos"] * d["ret"] - d["cost"]

    held = d[d["pos"] == 1].copy()
    daily_ret = held.groupby(date_col)["strategy_ret"].mean()

    all_dates = pd.Series(sorted(d[date_col].unique()), name=date_col)
    daily = all_dates.to_frame()
    daily["port_ret"] = daily[date_col].map(daily_ret).fillna(0.0)
    daily = daily.sort_values(date_col).reset_index(drop=True)
    daily["equity"] = (1.0 + daily["port_ret"]).cumprod() * initial

    total_return = float(daily["equity"].iloc[-1] / daily["equity"].iloc[0] - 1.0)
    roll_max = daily["equity"].cummax()
    dd = daily["equity"] / roll_max - 1.0
    max_dd = float(dd.min())

    r = daily["port_ret"].fillna(0.0)
    sharpe = 0.0 if r.std() == 0 else float(np.sqrt(252) * r.mean() / r.std())
    win_rate = float((r[r != 0] > 0).mean()) if (r != 0).any() else 0.0

    metrics = {
        "top_n": top_n,
        "fee_bps": fee_bps,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate
    }
    return daily, metrics