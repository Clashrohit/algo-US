import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_table_png(df: pd.DataFrame, title: str, out_path: str, font_size: int = 10):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1, 1.3)
    ax.set_title(title, fontsize=12, pad=12)
    fig.set_constrained_layout(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_radar_from_metrics(metrics_df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = ["Accuracy","Precision","Recall","F1"]

    df = metrics_df.copy()
    if "Model" in df.columns:
        df = df.set_index("Model")
    df = df[cols]

    labels = cols
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    for name, row in df.iterrows():
        vals = row.values.astype(float).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.5, label=str(name))
        ax.fill(angles, vals, alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("FIGURE 4. Radar chart of ML test performance.", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))
    fig.set_constrained_layout(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)