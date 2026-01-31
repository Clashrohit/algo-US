"""Quick demo script: read sample outputs, print summaries and save a small equity curve plot.
Requires: pandas, matplotlib
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
SAMPLE_DIR = os.path.join(ROOT, 'sample_outputs')
OUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

print('Reading sample outputs from', SAMPLE_DIR)
ts = pd.read_csv(os.path.join(SAMPLE_DIR, 'sample_timeseries_dataset.csv'))
sigs = pd.read_csv(os.path.join(SAMPLE_DIR, 'sample_trading_signals_best_model.csv'))
eq = pd.read_csv(os.path.join(SAMPLE_DIR, 'sample_portfolio_equity_curve.csv'))

print('\nTimeseries (head):')
print(ts.head().to_string(index=False))

print('\nSignals (head):')
print(sigs.head().to_string(index=False))

print('\nEquity curve:')
print(eq.to_string(index=False))

# Plot equity curve
plt.figure(figsize=(6,3))
plt.plot(pd.to_datetime(eq['date']), eq['equity'], marker='o')
plt.title('Sample equity curve')
plt.xlabel('Date')
plt.ylabel('Equity')
plt.grid(True)
out_png = os.path.join(OUT_DIR, 'sample_equity_curve.png')
plt.savefig(out_png, bbox_inches='tight')
print('\nSaved sample equity plot to', out_png)
