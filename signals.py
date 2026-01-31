import os
import glob
import numpy as np
import pandas as pd
import joblib

from config import BUY_THR, SELL_THR, OUTPUT_DIR


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_signals(dataset_csv: str, model_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)

    # STRICT: we must have symbol/date to generate company-wise outputs
    if "symbol" not in df.columns or "date" not in df.columns:
        raise ValueError(
            f"Input dataset '{dataset_csv}' does NOT contain symbol/date. "
            "Use outputs/timeseries_dataset.csv (or outputs/us_timeseries_dataset.csv)."
        )

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # derive MA20_MA50 if training expects it
    if "MA20_MA50" in feature_cols and "MA20_MA50" not in df.columns:
        if "SMA_20" in df.columns and "SMA_50" in df.columns:
            df["MA20_MA50"] = (df["SMA_20"] - df["SMA_50"]).astype(float)
        else:
            raise ValueError("MA20_MA50 needed but SMA_20/SMA_50 not found in dataset.")

    # ensure required features exist now
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required feature columns: {missing}")

    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True)).values

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        proba = _sigmoid(model.decision_function(X))
    else:
        proba = model.predict(X).astype(float)

    signal = np.where(proba >= BUY_THR, 1,
             np.where(proba <= SELL_THR, -1, 0))

    out = df.copy()            # IMPORTANT: keeps symbol/date/Close/etc.
    out["prob_buy"] = proba
    out["signal"] = signal

    out_path = os.path.join(OUTPUT_DIR, "trading_signals_best_model.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    return out


if __name__ == "__main__":
    # 1) pick latest best_model_*.joblib automatically
    candidates = sorted(glob.glob(os.path.join(OUTPUT_DIR, "best_model_*.joblib")))
    if not candidates:
        raise FileNotFoundError(f"No best_model_*.joblib found in {OUTPUT_DIR}. Run main.py first.")
    model_path = candidates[-1]
    print("Using model:", model_path)

    # 2) ALWAYS use full dataset with symbol/date
    dataset_path = os.path.join(OUTPUT_DIR, "timeseries_dataset.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"{dataset_path} not found. Run main.py with save_file=True to generate it."
        )

    # show columns to confirm
    tmp = pd.read_csv(dataset_path, nrows=2)
    print("Dataset columns (sample):", tmp.columns.tolist())

    # 3) generate signals
    generate_signals(dataset_path, model_path)