import pandas as pd
from backtest import backtest_topn_portfolio

df = pd.read_csv("outputs/trading_signals_best_model.csv")
curve, metrics = backtest_topn_portfolio(df, top_n=10, initial=10000)

print(metrics)
curve.to_csv("outputs/portfolio_equity_curve.csv", index=False)
print(curve.tail())