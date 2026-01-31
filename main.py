from config import load_sp500_tickers, OUTPUT_DIR, HORIZON_DAYS
from dataset_builder import build_timeseries_dataset, build_us_timeseries_dataset
from feature_selection import paper_feature_select
from model_assortment import run_assortment
from report_paperstyle import save_table_png, plot_radar_from_metrics


def main():
    tickers = load_sp500_tickers()
    print("Tickers loaded:", len(tickers))

    EVAL_MODE = "REALISTIC"   # or "PAPER_US"

    if EVAL_MODE == "REALISTIC":
        H = HORIZON_DAYS
        df_ts = build_timeseries_dataset(
            tickers,
            horizon_days=H,
            offline_only=True,
            save_file=True,     # outputs/timeseries_dataset.csv
            add_regime=True
        )
        print("Time-series dataset rows:", len(df_ts))
        if df_ts.empty:
            print("No data generated.")
            return

        df_final, keep, missing = paper_feature_select(df_ts)
        print("Features kept:", keep)
        print("Missing:", missing)

        best_name, model_path, metrics = run_assortment(
            df_final,
            split_strategy="paper_holdout",
            train_end_date="2021-12-31",
            test_start_date="2022-01-01",
            test_end_date="2022-12-31",
            horizon_days=H,
            apply_smote_on_train=False,
            use_sample_weight=True,
            sort_by="BalancedAcc",
            save_csv=False
        )

    else:
        H = 0
        df_us = build_us_timeseries_dataset(
            tickers,
            offline_only=True,
            us_threshold=3,
            save_file=True,     # outputs/us_timeseries_dataset.csv
            add_regime=True
        )
        print("US dataset rows:", len(df_us))
        if df_us.empty:
            print("No data generated.")
            return

        df_final, keep, missing = paper_feature_select(df_us)
        print("Features kept:", keep)
        print("Missing:", missing)

        best_name, model_path, metrics = run_assortment(
            df_final,
            split_strategy="paper_holdout",
            train_end_date="2021-12-31",
            test_start_date="2022-01-01",
            test_end_date="2022-12-31",
            horizon_days=H,
            apply_smote_on_train=True,
            use_sample_weight=False,
            sort_by="BalancedAcc",
            save_csv=False
        )

    table7 = metrics[["Model","Accuracy","Precision","Recall","F1"]].copy()

    save_table_png(
        table7.set_index("Model").round(3),
        title="TABLE 7. Machine learning test performance details.",
        out_path=f"{OUTPUT_DIR}/TABLE_7.png",
        font_size=10
    )

    plot_radar_from_metrics(
        metrics_df=table7.round(3),
        out_path=f"{OUTPUT_DIR}/FIGURE_4_radar.png"
    )

    print("Saved: outputs/TABLE_7.png and outputs/FIGURE_4_radar.png")
    print("Best model:", best_name)
    print("Model file:", model_path)
    print("Next: run signals.py using outputs/timeseries_dataset.csv to keep symbol/date.")


if __name__ == "__main__":
    main()