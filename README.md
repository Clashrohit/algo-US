# algo-US 🚀

**Algorithmic trading research project for US equities.** This repo contains code to build datasets, select features, train models, and run backtests used to generate figures and tables for analysis and reports.

---

## 📌 Highlights

- Clean, reproducible scripts for data preparation, feature engineering, modelling and backtesting
- Reproducible experiments: see `main.py`, `run_backtest_paperstyle.py` and `backtest.py`
- Outputs (figures, datasets, model artifacts) saved to the `outputs/` folder

---

## 🧭 Quickstart

1. Create a Python virtual environment:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

1. Install dependencies (if you maintain `requirements.txt`):

```bash
pip install -r requirements.txt
```

1. Reproduce a backtest / generate outputs:

```bash
python run_backtest_paperstyle.py
# or
python main.py
```

Outputs will be written into `outputs/`.

---

## 📁 Data

- `data/prices/` holds historical price CSVs (one file per ticker). These are large and not strictly required to run the code if you provide a smaller sample.
- If you do not want to commit large raw data to GitHub, consider:
  - Using Git LFS for large files (recommended)
  - Storing large datasets in external storage (Google Drive, S3, Zenodo) and providing download scripts

---

## ⚠️ Notes on large files

Two files in `outputs/` currently exceed GitHub's recommended 50 MB size. If you plan to keep large files in the repo long-term, convert them to Git LFS:

```bash
# install git-lfs
git lfs install
# track CSV outputs
git lfs track "outputs/*.csv"
git add .gitattributes
# migrate existing big files (history rewrite)
git lfs migrate import --include="outputs/timeseries_dataset.csv,outputs/trading_signals_best_model.csv"
# force-push rewritten history
git push --force origin main
```

Be careful: migrating rewrites history and requires coordination with collaborators.

---

## 🧪 Reproducibility & Tips

- Set random seeds where applicable for deterministic model training.
- Keep dataset and model configs in `config.py` for easy experiment tracking.
- Use the `outputs/` folder to store results; don't rely on committing intermediate large files.

---

## 📦 Sample outputs & quick demo

To make it easy to try the project without large datasets, this repo includes a small `sample_outputs/` folder and a tiny demo script. These lightweight files let you verify the repo structure and run a minimal demo quickly.

**Included sample files**

- `sample_outputs/sample_timeseries_dataset.csv` — tiny timeseries dataset (date, ticker, OHLCV)
- `sample_outputs/sample_trading_signals_best_model.csv` — example buy/hold/sell signals
- `sample_outputs/sample_portfolio_equity_curve.csv` — small portfolio equity series

**Quick demo**

```bash
# copy sample files into outputs and run the demo
mkdir -p outputs
cp -r sample_outputs/* outputs/
python -m pip install -r requirements.txt
python scripts/quick_demo.py
# This prints summaries and saves outputs/sample_equity_curve.png
```

**Preview (first rows)**

```csv
# sample_timeseries_dataset.csv
date,ticker,open,high,low,close,volume
2025-12-29,AAPL,150,152,149,151,1000000
2025-12-30,AAPL,151,153,150,152,1100000
2025-12-29,MSFT,300,305,299,304,800000
...
```

> These samples are lightweight and intended for quick verification only. To run full experiments, use the full price dataset in `data/prices/`.

---

## 🤝 Contributing

Contributions are welcome. Please open issues or pull requests with clear descriptions and tests or reproducible steps.

---

## 📄 License

MIT — see `LICENSE`.

---

> Need the README in Tamil or want me to add a `requirements.txt` / CI workflow (`.github/workflows`) next? Say the word and I’ll add it. ✅
