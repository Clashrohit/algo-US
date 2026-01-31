import os
import time
import json
import numpy as np
import pandas as pd

from collections import Counter
from typing import Tuple, Dict, List, Optional

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE
import joblib

from config import OUTPUT_DIR


def get_models():
    return {
        "GBM": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42,
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("m", LinearSVC(class_weight="balanced", random_state=42, max_iter=5000))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("m", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1))
        ]),
        "NeuralNetwork": Pipeline([
            ("scaler", StandardScaler()),
            ("m", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=15,
                random_state=42
            ))
        ]),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("m", KNeighborsClassifier(n_neighbors=7, n_jobs=-1))
        ]),
        "GaussianNB": Pipeline([
            ("scaler", StandardScaler()),
            ("m", GaussianNB())
        ]),
    }


def _median_impute(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]):
    med = train_df[feature_cols].median(numeric_only=True)
    Xtr = train_df[feature_cols].fillna(med).values
    Xte = test_df[feature_cols].fillna(med).values
    return Xtr, Xte


def _smote_train_only(Xtr: np.ndarray, ytr: np.ndarray):
    dist = Counter(ytr)
    if len(dist) < 2:
        return Xtr, ytr
    minority = min(dist.values())
    if minority < 2:
        return Xtr, ytr
    k = min(5, minority - 1)
    if k < 1:
        return Xtr, ytr
    sm = SMOTE(random_state=42, k_neighbors=k)
    return sm.fit_resample(Xtr, ytr)


def _purge_last_horizon_per_symbol(train_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    if horizon_days <= 0:
        return train_df
    t = train_df.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    pos = t.groupby("symbol").cumcount()
    size = t.groupby("symbol")["symbol"].transform("size")
    keep = pos < (size - horizon_days)
    return t.loc[keep].copy()


def _prepare_features(df: pd.DataFrame):
    ignore = {"label", "symbol", "date", "US_Score", "Close", "Total_Trend"}
    feature_cols = [c for c in df.columns if c not in ignore]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise RuntimeError("No numeric features found after filtering.")
    return feature_cols


def _paper_holdout_split(df: pd.DataFrame, train_end_date: str, test_start_date: str, test_end_date: str):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")

    tr_end = pd.to_datetime(train_end_date)
    te_start = pd.to_datetime(test_start_date)
    te_end = pd.to_datetime(test_end_date)

    train_df = d[d["date"] <= tr_end].copy()
    test_df = d[(d["date"] >= te_start) & (d["date"] <= te_end)].copy()
    return train_df, test_df


def _class_weights_as_sample_weight(y: np.ndarray) -> np.ndarray:
    dist = Counter(y)
    n = len(y)
    w = {int(cls): n / (len(dist) * cnt) for cls, cnt in dist.items()}
    return np.array([w[int(v)] for v in y], dtype=float)


def _fit_try_sample_weight(model, X, y, sw):
    try:
        if isinstance(model, Pipeline):
            return model.fit(X, y, **{"m__sample_weight": sw})
        return model.fit(X, y, sample_weight=sw)
    except TypeError:
        return model.fit(X, y)


def run_assortment(
    df: pd.DataFrame,
    test_size: float = 0.3,
    apply_smote_on_train: bool = True,
    use_sample_weight: bool = True,
    split_strategy: str = "paper_holdout",
    train_end_date: str = "2021-12-31",
    test_start_date: str = "2022-01-01",
    test_end_date: str = "2022-12-31",
    horizon_days: int = 5,
    knn_max_train: int = 20000,
    sort_by: str = "BalancedAcc",
    save_csv: bool = False,
) -> Tuple[str, str, pd.DataFrame]:

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    feature_cols = _prepare_features(df)
    models = get_models()

    if split_strategy == "paper_holdout":
        train_df, test_df = _paper_holdout_split(df, train_end_date, test_start_date, test_end_date)
        train_df = _purge_last_horizon_per_symbol(train_df, horizon_days=horizon_days)
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=42)

    if train_df.empty or test_df.empty:
        raise RuntimeError("Train/Test empty. Check dates/data range.")

    ytr = train_df["label"].values
    yte = test_df["label"].values
    print("Train distribution:", Counter(ytr))
    print("Test  distribution:", Counter(yte))

    Xtr, Xte = _median_impute(train_df, test_df, feature_cols)

    if apply_smote_on_train:
        Xtr_fit, ytr_fit = _smote_train_only(Xtr, ytr)
        sw = None
    else:
        Xtr_fit, ytr_fit = Xtr, ytr
        sw = _class_weights_as_sample_weight(ytr_fit) if use_sample_weight else None

    rows = []
    for name, base_model in models.items():
        t0 = time.time()

        X_use, y_use = Xtr_fit, ytr_fit
        sw_use = sw

        if name == "KNN" and len(ytr_fit) > knn_max_train:
            idx = np.random.default_rng(42).choice(len(ytr_fit), size=knn_max_train, replace=False)
            X_use, y_use = Xtr_fit[idx], ytr_fit[idx]
            sw_use = None

        model = clone(base_model)
        if sw_use is not None:
            _fit_try_sample_weight(model, X_use, y_use, sw_use)
        else:
            model.fit(X_use, y_use)

        pred = model.predict(Xte)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()

        acc = accuracy_score(yte, pred)
        prec = precision_score(yte, pred, pos_label=1, zero_division=0)
        rec = recall_score(yte, pred, pos_label=1, zero_division=0)
        f1 = f1_score(yte, pred, pos_label=1, zero_division=0)

        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        balanced = 0.5 * (rec + specificity)
        pred_pos_rate = float(np.mean(pred))

        rows.append([name, acc, prec, rec, f1, specificity, balanced, pred_pos_rate])
        print(f"[DONE] {name:16s} Acc={acc:.3f} F1={f1:.3f} BalAcc={balanced:.3f} time={time.time()-t0:.1f}s")

    metrics = pd.DataFrame(rows, columns=["Model","Accuracy","Precision","Recall","F1","Specificity","BalancedAcc","PredPosRate"])
    if sort_by not in metrics.columns:
        sort_by = "BalancedAcc"
    metrics = metrics.sort_values(sort_by, ascending=False).reset_index(drop=True)

    if save_csv:
        metrics.to_csv(f"{OUTPUT_DIR}/table7_model_metrics.csv", index=False)

    best_name = str(metrics.iloc[0]["Model"])
    best_model = clone(models[best_name])

    # Fit best model on full data (deployment bundle)
    X_full = df[feature_cols].copy().fillna(df[feature_cols].median(numeric_only=True)).values
    y_full = df["label"].values
    if (not apply_smote_on_train) and use_sample_weight:
        sw_full = _class_weights_as_sample_weight(y_full)
        _fit_try_sample_weight(best_model, X_full, y_full, sw_full)
    else:
        best_model.fit(X_full, y_full)

    bundle = {
        "model": best_model,
        "feature_cols": feature_cols,
        "best_name": best_name,
    }

    model_path = f"{OUTPUT_DIR}/best_model_{best_name}.joblib"
    joblib.dump(bundle, model_path)

    with open(f"{OUTPUT_DIR}/best_model_{best_name}_meta.json", "w") as f:
        json.dump({"best_name": best_name, "feature_cols": feature_cols}, f, indent=2)

    print("\nSaved model bundle:", model_path)
    return best_name, model_path, metrics