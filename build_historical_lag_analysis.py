import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RUN_BACKTEST = True   # set False to skip the slow backtest step during fast iteration

from historical_lag_analysis import (
    ANALYSIS_WEATHER_END_DATE,
    DEFAULT_ANALYSIS_LAG_DAYS,
    ARIMAX_FUTURE_EXOG_FILENAME,
    ARIMAX_FORECAST_FILENAME,
    BEST_LAGS_FILENAME,
    LAG_SUMMARY_FILENAME,
    LITERATURE_SUMMARY_FILENAME,
    MANIFEST_FILENAME,
    QA_FILENAME,
    SARIMA_FORECAST_FILENAME,
    WEEKLY_BEST_LAGS_FILENAME,
    WEEKLY_LAG_SUMMARY_FILENAME,
    WEEKLY_BASE_FILENAME,
    artifact_paths,
    build_best_lags,
    build_default_forecast_artifacts,
    build_historical_analysis_qa,
    build_lag_correlation_summary,
    build_literature_comparison_summary,
    build_analysis_manifest,
    build_weekly_lag_correlation_summary,
    build_weekly_base_dataset,
    build_weekly_dengue_series,
    load_historical_weather_data,
    split_weekly_base_artifact,
    prepare_daily_weather_features,
    write_analysis_manifest,
    walk_forward_backtest,
    _diebold_mariano_hln,
    BACKTEST_HORIZON,
    BACKTEST_STEP_WEEKS,
    BACKTEST_MIN_TRAIN_WEEKS,
    BACKTEST_ALPHA,
    BACKTEST_EVAL_THRESHOLDS,
)


BASE_DIR = Path(__file__).resolve().parent
DENGUE_FILE = BASE_DIR / "singapore_dengue_raw_records.csv"
WEATHER_FILE = BASE_DIR / "singapore_weather_forecast_24hr_history.csv"
PATHS = artifact_paths(BASE_DIR)
BASE_OUTPUT = PATHS["base"]
LAG_SUMMARY_OUTPUT = PATHS["lag_summary"]
BEST_LAGS_OUTPUT = PATHS["best_lags"]
WEEKLY_LAG_SUMMARY_OUTPUT = PATHS["weekly_lag_summary"]
WEEKLY_BEST_LAGS_OUTPUT = PATHS["weekly_best_lags"]
LITERATURE_SUMMARY_OUTPUT = PATHS["literature_summary"]
QA_OUTPUT = PATHS["qa"]
MANIFEST_OUTPUT = PATHS["manifest"]
SARIMA_OUTPUT = PATHS["sarima_forecast"]
ARIMAX_OUTPUT = PATHS["arimax_forecast"]
ARIMAX_FUTURE_EXOG_OUTPUT = PATHS["arimax_future_exog"]


def main() -> None:
    records_df = pd.read_csv(DENGUE_FILE)
    weather_df = load_historical_weather_data(WEATHER_FILE, end_date=ANALYSIS_WEATHER_END_DATE)
    prepared_weather_df = prepare_daily_weather_features(weather_df)

    weekly_df, dengue_qa_df = build_weekly_dengue_series(records_df)
    base_df = build_weekly_base_dataset(weekly_df, prepared_weather_df, lag_days=DEFAULT_ANALYSIS_LAG_DAYS)
    lag_summary_df = build_lag_correlation_summary(weekly_df, prepared_weather_df)
    best_lags_df = build_best_lags(lag_summary_df)
    weekly_lag_summary_df = build_weekly_lag_correlation_summary(weekly_df, prepared_weather_df)
    weekly_best_lags_df = build_best_lags(weekly_lag_summary_df, lag_column="lag_weeks")
    literature_summary_df = build_literature_comparison_summary(weekly_lag_summary_df)
    qa_df = build_historical_analysis_qa(
        weekly_df,
        prepared_weather_df,
        base_df,
        lag_summary_df,
        weekly_lag_summary_df=weekly_lag_summary_df,
    )
    qa_df = pd.concat([dengue_qa_df, qa_df], ignore_index=True)
    weekly_base_df, weekly_weather_df = split_weekly_base_artifact(base_df)
    sarima_df, arimax_df, future_exog_df = build_default_forecast_artifacts(
        weekly_base_df,
        weekly_weather_df,
    )

    base_df.to_csv(BASE_OUTPUT, index=False)
    lag_summary_df.to_csv(LAG_SUMMARY_OUTPUT, index=False)
    best_lags_df.to_csv(BEST_LAGS_OUTPUT, index=False)
    weekly_lag_summary_df.to_csv(WEEKLY_LAG_SUMMARY_OUTPUT, index=False)
    weekly_best_lags_df.to_csv(WEEKLY_BEST_LAGS_OUTPUT, index=False)
    literature_summary_df.to_csv(LITERATURE_SUMMARY_OUTPUT, index=False)
    qa_df.to_csv(QA_OUTPUT, index=False)
    sarima_df.to_csv(SARIMA_OUTPUT, index=False)
    arimax_df.to_csv(ARIMAX_OUTPUT, index=False)
    future_exog_df.to_csv(ARIMAX_FUTURE_EXOG_OUTPUT, index=False)

    if RUN_BACKTEST:
        print("Running walk-forward backtest (this typically takes 3-6 minutes)...")
        sarima_results = walk_forward_backtest(weekly_base_df, model_kind="sarima")
        sarimax_results = walk_forward_backtest(
            weekly_base_df, weekly_weather_df, model_kind="sarimax"
        )

        for kind, results in [("sarima", sarima_results), ("sarimax", sarimax_results)]:
            results["per_origin"]["model_kind"] = kind
            results["horizon_metrics"]["model_kind"] = kind
            results["summary"]["model_kind"] = kind

        per_origin_combined = pd.concat(
            [sarima_results["per_origin"], sarimax_results["per_origin"]], ignore_index=True
        )
        horizon_metrics_combined = pd.concat(
            [sarima_results["horizon_metrics"], sarimax_results["horizon_metrics"]], ignore_index=True
        )
        summary_combined = pd.concat(
            [sarima_results["summary"], sarimax_results["summary"]], ignore_index=True
        )

        dm_rows = []
        for h_val in range(1, BACKTEST_HORIZON + 1):
            a = sarima_results["per_origin"].query("h == @h_val").sort_values("origin_week")
            b = sarimax_results["per_origin"].query("h == @h_val").sort_values("origin_week")
            merged = a.merge(b, on="origin_week", suffixes=("_sarima", "_sarimax"))
            if len(merged) < 8:
                dm_rows.append({"h": h_val, "dm_stat": np.nan, "p_value": np.nan, "n": len(merged)})
                continue
            err_sarima = (merged["predicted_sarima"] - merged["actual_sarima"]).to_numpy()
            err_sarimax = (merged["predicted_sarimax"] - merged["actual_sarimax"]).to_numpy()
            dm_stat, p_value = _diebold_mariano_hln(err_sarima, err_sarimax, h_val)
            dm_rows.append({"h": h_val, "dm_stat": dm_stat, "p_value": p_value, "n": len(merged)})
        dm_df = pd.DataFrame(dm_rows)

        per_origin_combined.to_csv(BASE_DIR / "backtest_per_origin.csv", index=False)
        horizon_metrics_combined.to_csv(BASE_DIR / "backtest_horizon_metrics.csv", index=False)
        summary_combined.to_csv(BASE_DIR / "backtest_summary.csv", index=False)
        dm_df.to_csv(BASE_DIR / "backtest_diebold_mariano.csv", index=False)

        backtest_meta = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_kinds": ["sarima", "sarimax"],
            "horizon": BACKTEST_HORIZON,
            "step_weeks": BACKTEST_STEP_WEEKS,
            "min_train_weeks": BACKTEST_MIN_TRAIN_WEEKS,
            "alpha": BACKTEST_ALPHA,
            "eval_thresholds": list(BACKTEST_EVAL_THRESHOLDS),
            "n_origins": int(per_origin_combined["origin_week"].nunique()),
            "origin_first": str(per_origin_combined["origin_week"].min()),
            "origin_last": str(per_origin_combined["origin_week"].max()),
        }
        print(
            f"Backtest done: {backtest_meta['n_origins']} origins from "
            f"{backtest_meta['origin_first']} to {backtest_meta['origin_last']}"
        )
    else:
        print("RUN_BACKTEST=False — preserving previous backtest manifest section")
        backtest_meta = None

    previous_manifest = {}
    if MANIFEST_OUTPUT.exists():
        try:
            with open(MANIFEST_OUTPUT) as f:
                previous_manifest = json.load(f)
        except Exception:
            previous_manifest = {}

    row_counts = {
        "base": len(base_df),
        "lag_summary": len(lag_summary_df),
        "best_lags": len(best_lags_df),
        "weekly_lag_summary": len(weekly_lag_summary_df),
        "weekly_best_lags": len(weekly_best_lags_df),
        "literature_summary": len(literature_summary_df),
        "qa": len(qa_df),
        "sarima_forecast": len(sarima_df),
        "arimax_forecast": len(arimax_df),
        "arimax_future_exog": len(future_exog_df),
    }
    manifest = build_analysis_manifest(
        base_dir=BASE_DIR,
        dengue_path=DENGUE_FILE,
        weather_path=WEATHER_FILE,
        artifact_row_counts=row_counts,
    )
    manifest["backtest"] = backtest_meta if backtest_meta is not None else previous_manifest.get("backtest", {})
    write_analysis_manifest(MANIFEST_OUTPUT, manifest)

    print(f"Wrote {BASE_OUTPUT}")
    print(f"Wrote {LAG_SUMMARY_OUTPUT}")
    print(f"Wrote {BEST_LAGS_OUTPUT}")
    print(f"Wrote {WEEKLY_LAG_SUMMARY_OUTPUT}")
    print(f"Wrote {WEEKLY_BEST_LAGS_OUTPUT}")
    print(f"Wrote {LITERATURE_SUMMARY_OUTPUT}")
    print(f"Wrote {QA_OUTPUT}")
    print(f"Wrote {SARIMA_OUTPUT}")
    print(f"Wrote {ARIMAX_OUTPUT}")
    print(f"Wrote {ARIMAX_FUTURE_EXOG_OUTPUT}")
    print(f"Wrote {MANIFEST_OUTPUT}")
    print(
        f"Base rows={len(base_df)} | "
        f"Lag summary rows={len(lag_summary_df)} | "
        f"Best lag rows={len(best_lags_df)} | "
        f"Weekly lag rows={len(weekly_lag_summary_df)} | "
        f"SARIMA rows={len(sarima_df)} | "
        f"ARIMAX rows={len(arimax_df)}"
    )


if __name__ == "__main__":
    main()
