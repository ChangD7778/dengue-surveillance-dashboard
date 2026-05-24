import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX


ANALYSIS_WEATHER_END_DATE = pd.Timestamp("2024-12-31")
DEFAULT_ANALYSIS_LAG_WEEKS = 18
DEFAULT_ANALYSIS_LAG_DAYS = DEFAULT_ANALYSIS_LAG_WEEKS * 7
LITERATURE_BENCHMARK_WEEKS = 18
DEFAULT_FORECAST_HORIZON = 12
DEFAULT_FORECAST_ALPHA = 0.2
DEFAULT_THRESHOLD = 400
DEFAULT_RISK_LOW_CUT = 0.30
DEFAULT_RISK_HIGH_CUT = 0.60
RISK_Z = 1.2816
SEASONAL_PERIOD = 52
FIXED_ORDER = (3, 1, 0)
FIXED_SEASONAL_ORDER = (0, 1, 0, SEASONAL_PERIOD)
BACKTEST_HORIZON = 12
BACKTEST_STEP_WEEKS = 4
BACKTEST_MIN_TRAIN_WEEKS = 156
BACKTEST_ALPHA = 0.20
BACKTEST_EVAL_THRESHOLDS = (200, 300, 400, 500)
LAG_SWEEP_DAYS = tuple(range(29))
LAG_SWEEP_WEEKS = tuple(range(53))
LAG_FEATURE_COLUMNS = [
    "avg_temp_c",
    "avg_relative_humidity_pct",
    "avg_wind_speed",
    "warm_day_share",
]
ARIMAX_EXOG_COLUMNS = list(LAG_FEATURE_COLUMNS)
LAG_FEATURE_LABELS = {
    "avg_temp_c": "Temperature",
    "avg_relative_humidity_pct": "Humidity",
    "avg_wind_speed": "Wind",
    "warm_day_share": "Warm-day share",
}
WEEKLY_BASE_FILENAME = "weekly_dengue_weather_base.csv"
LAG_SUMMARY_FILENAME = "weather_lag_correlation_summary.csv"
BEST_LAGS_FILENAME = "weather_lag_best_lags.csv"
QA_FILENAME = "historical_lag_analysis_qa.csv"
MANIFEST_FILENAME = "analysis_artifacts_manifest.json"
SARIMA_FORECAST_FILENAME = "sarima_default_forecast_12w.csv"
ARIMAX_FORECAST_FILENAME = "arimax_default_forecast_12w.csv"
ARIMAX_FUTURE_EXOG_FILENAME = "arimax_default_future_exog_12w.csv"
WEEKLY_LAG_SUMMARY_FILENAME = "weather_lag_correlation_summary_weekly.csv"
WEEKLY_BEST_LAGS_FILENAME = "weather_lag_best_lags_weekly.csv"
LITERATURE_SUMMARY_FILENAME = "weather_literature_comparison_weekly.csv"


def parse_epi_week_to_sunday(epi_week: str):
    match = pd.Series([str(epi_week).strip()]).str.extract(r"(\d{4})-W(\d{1,2})")
    if match.isna().any(axis=None):
        return pd.NaT
    year = int(match.iloc[0, 0])
    week = int(match.iloc[0, 1])
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    first_week_sunday = jan1 - pd.Timedelta(days=(jan1.weekday() + 1) % 7)
    return first_week_sunday + pd.Timedelta(weeks=week - 1)


def to_week_start_sunday(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, errors="coerce")
    day_offset = (ts.dt.dayofweek + 1) % 7
    return (ts - pd.to_timedelta(day_offset, unit="D")).dt.normalize()


def load_historical_weather_data(
    weather_path: Path,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp = ANALYSIS_WEATHER_END_DATE,
) -> pd.DataFrame:
    weather_path = Path(weather_path)
    if not weather_path.exists():
        raise FileNotFoundError(f"Missing weather history CSV: {weather_path}")

    weather_df = pd.read_csv(weather_path, low_memory=False)
    if "query_date" not in weather_df.columns:
        raise RuntimeError("Weather history CSV is missing the query_date column.")

    weather_df["query_date"] = pd.to_datetime(weather_df["query_date"], errors="coerce")
    weather_df = weather_df.dropna(subset=["query_date"]).copy()

    effective_end = pd.Timestamp(end_date).normalize()
    weather_df = weather_df[weather_df["query_date"] <= effective_end].copy()
    if start_date is not None:
        effective_start = pd.Timestamp(start_date).normalize()
        weather_df = weather_df[weather_df["query_date"] >= effective_start].copy()

    if weather_df.empty:
        raise RuntimeError(
            f"Historical weather history is empty after filtering through {effective_end.date()}."
        )

    return weather_df.sort_values("query_date").reset_index(drop=True)


def build_weekly_dengue_series(records_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = {"epi_week", "disease", "no._of_cases"}
    missing = required_cols.difference(records_df.columns)
    if missing:
        raise RuntimeError(f"Missing expected columns: {sorted(missing)}")

    work = records_df.copy()
    work["no._of_cases"] = pd.to_numeric(work["no._of_cases"], errors="coerce").fillna(0)
    work = work[work["disease"].isin(["Dengue Fever", "Dengue Haemorrhagic Fever"])].copy()
    if work.empty:
        raise RuntimeError("No dengue rows found in source data.")

    weekly = (
        work.groupby(["epi_week", "disease"], as_index=False)["no._of_cases"]
        .sum()
        .pivot(index="epi_week", columns="disease", values="no._of_cases")
        .fillna(0)
        .reset_index()
    )

    if "Dengue Fever" not in weekly.columns:
        weekly["Dengue Fever"] = 0
    if "Dengue Haemorrhagic Fever" not in weekly.columns:
        weekly["Dengue Haemorrhagic Fever"] = 0

    weekly["week_start"] = weekly["epi_week"].map(parse_epi_week_to_sunday)
    weekly = (
        weekly.dropna(subset=["week_start"])
        .groupby("week_start", as_index=False)[["Dengue Fever", "Dengue Haemorrhagic Fever"]]
        .sum()
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    weekly["Total Dengue Cases"] = weekly["Dengue Fever"] + weekly["Dengue Haemorrhagic Fever"]
    weekly["moving_avg_12w"] = weekly["Total Dengue Cases"].rolling(12, min_periods=1).mean()

    qa_df = pd.DataFrame(
        [
            {"metric": "dengue_rows", "value": int(len(weekly))},
            {"metric": "dengue_duplicate_week_start", "value": int(weekly["week_start"].duplicated().sum())},
            {"metric": "dengue_negative_cases", "value": int((weekly["Total Dengue Cases"] < 0).sum())},
            {"metric": "dengue_week_start_min", "value": weekly["week_start"].min()},
            {"metric": "dengue_week_start_max", "value": weekly["week_start"].max()},
        ]
    )

    return (
        weekly[
            [
                "week_start",
                "Dengue Fever",
                "Dengue Haemorrhagic Fever",
                "Total Dengue Cases",
                "moving_avg_12w",
            ]
        ],
        qa_df,
    )


def prepare_daily_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    if weather_df.empty:
        raise RuntimeError("Weather dataframe is empty.")

    work = weather_df.copy()
    work["query_date"] = pd.to_datetime(work["query_date"], errors="coerce")
    work = work.dropna(subset=["query_date"]).sort_values("query_date").reset_index(drop=True)
    if work.empty:
        raise RuntimeError("Weather dataframe does not contain valid query_date values.")

    for col in [
        "temperature_low_c",
        "temperature_high_c",
        "relative_humidity_low_pct",
        "relative_humidity_high_pct",
        "wind_speed_low",
        "wind_speed_high",
    ]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["avg_temp_c"] = work[["temperature_low_c", "temperature_high_c"]].mean(axis=1)
    work["avg_relative_humidity_pct"] = work[["relative_humidity_low_pct", "relative_humidity_high_pct"]].mean(axis=1)
    work["avg_wind_speed"] = work[["wind_speed_low", "wind_speed_high"]].mean(axis=1)
    work["warm_day_flag"] = (work["temperature_high_c"] >= 32).astype(float)
    return work


def _is_prepared_weather_df(weather_df: pd.DataFrame) -> bool:
    required_cols = {
        "query_date",
        "temperature_low_c",
        "temperature_high_c",
        "avg_temp_c",
        "avg_relative_humidity_pct",
        "avg_wind_speed",
        "warm_day_flag",
    }
    return required_cols.issubset(weather_df.columns)


def build_weekly_weather_features(weather_df: pd.DataFrame, lag_days: int = DEFAULT_ANALYSIS_LAG_DAYS) -> pd.DataFrame:
    work = weather_df.copy()
    if not _is_prepared_weather_df(work):
        work = prepare_daily_weather_features(work)
    else:
        work["query_date"] = pd.to_datetime(work["query_date"], errors="coerce")
        work = work.dropna(subset=["query_date"]).sort_values("query_date").reset_index(drop=True)

    work["lagged_query_date"] = work["query_date"] + pd.to_timedelta(int(lag_days), unit="D")
    work["week_start"] = to_week_start_sunday(work["lagged_query_date"])
    work = work.dropna(subset=["week_start"]).copy()

    weekly_weather_df = (
        work.groupby("week_start", as_index=False)
        .agg(
            avg_temp_c=("avg_temp_c", "mean"),
            min_temp_c=("temperature_low_c", "mean"),
            max_temp_c=("temperature_high_c", "mean"),
            avg_relative_humidity_pct=("avg_relative_humidity_pct", "mean"),
            avg_wind_speed=("avg_wind_speed", "mean"),
            warm_day_share=("warm_day_flag", "mean"),
            source_rows=("query_date", "size"),
            source_query_start=("query_date", "min"),
            source_query_end=("query_date", "max"),
        )
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    weekly_weather_df["weather_lag_days"] = int(lag_days)
    return weekly_weather_df


def build_weekly_weather_features_by_week(
    weather_df: pd.DataFrame,
    lag_weeks: int = 0,
) -> pd.DataFrame:
    work = weather_df.copy()
    if not _is_prepared_weather_df(work):
        work = prepare_daily_weather_features(work)
    else:
        work["query_date"] = pd.to_datetime(work["query_date"], errors="coerce")
        work = work.dropna(subset=["query_date"]).sort_values("query_date").reset_index(drop=True)

    work["lagged_query_date"] = work["query_date"] + pd.to_timedelta(int(lag_weeks), unit="W")
    work["week_start"] = to_week_start_sunday(work["lagged_query_date"])
    work = work.dropna(subset=["week_start"]).copy()

    weekly_weather_df = (
        work.groupby("week_start", as_index=False)
        .agg(
            avg_temp_c=("avg_temp_c", "mean"),
            min_temp_c=("temperature_low_c", "mean"),
            max_temp_c=("temperature_high_c", "mean"),
            avg_relative_humidity_pct=("avg_relative_humidity_pct", "mean"),
            avg_wind_speed=("avg_wind_speed", "mean"),
            warm_day_share=("warm_day_flag", "mean"),
            source_rows=("query_date", "size"),
            source_query_start=("query_date", "min"),
            source_query_end=("query_date", "max"),
        )
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    weekly_weather_df["weather_lag_weeks"] = int(lag_weeks)
    return weekly_weather_df


def build_weekly_base_dataset(
    weekly_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    lag_days: int = DEFAULT_ANALYSIS_LAG_DAYS,
) -> pd.DataFrame:
    weekly_weather_df = build_weekly_weather_features(weather_df, lag_days=lag_days)
    base_df = weekly_df.merge(weekly_weather_df, on="week_start", how="left")
    base_df["weather_overlap_flag"] = base_df["avg_temp_c"].notna().astype(int)
    return base_df.sort_values("week_start").reset_index(drop=True)


def _safe_corr(series_x: pd.Series, series_y: pd.Series, method: str) -> float:
    work = pd.DataFrame({"x": series_x, "y": series_y}).dropna()
    if len(work) < 2 or work["x"].nunique() < 2 or work["y"].nunique() < 2:
        return np.nan
    return float(work["x"].corr(work["y"], method=method))


def build_lag_correlation_summary(
    weekly_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    lag_days_iterable= LAG_SWEEP_DAYS,
) -> pd.DataFrame:
    prepared_weather_df = prepare_daily_weather_features(weather_df)
    rows: list[dict] = []
    for lag_days in lag_days_iterable:
        weekly_weather_df = build_weekly_weather_features(prepared_weather_df, lag_days=int(lag_days))
        for feature in LAG_FEATURE_COLUMNS:
            merged = (
                weekly_df[["week_start", "Total Dengue Cases"]]
                .merge(weekly_weather_df[["week_start", feature]], on="week_start", how="inner")
                .dropna(subset=[feature, "Total Dengue Cases"])
                .sort_values("week_start")
                .reset_index(drop=True)
            )
            rows.append(
                {
                    "feature": feature,
                    "feature_label": LAG_FEATURE_LABELS[feature],
                    "lag_days": int(lag_days),
                    "pearson_r": _safe_corr(merged[feature], merged["Total Dengue Cases"], "pearson"),
                    "spearman_rho": _safe_corr(merged[feature], merged["Total Dengue Cases"], "spearman"),
                    "overlap_weeks": int(len(merged)),
                    "analysis_start": merged["week_start"].min() if not merged.empty else pd.NaT,
                    "analysis_end": merged["week_start"].max() if not merged.empty else pd.NaT,
                }
            )

    return pd.DataFrame(rows).sort_values(["feature", "lag_days"]).reset_index(drop=True)


def build_weekly_lag_correlation_summary(
    weekly_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    lag_weeks_iterable=LAG_SWEEP_WEEKS,
) -> pd.DataFrame:
    prepared_weather_df = prepare_daily_weather_features(weather_df)
    rows: list[dict] = []
    for lag_weeks in lag_weeks_iterable:
        weekly_weather_df = build_weekly_weather_features_by_week(prepared_weather_df, lag_weeks=int(lag_weeks))
        for feature in LAG_FEATURE_COLUMNS:
            merged = (
                weekly_df[["week_start", "Total Dengue Cases"]]
                .merge(weekly_weather_df[["week_start", feature]], on="week_start", how="inner")
                .dropna(subset=[feature, "Total Dengue Cases"])
                .sort_values("week_start")
                .reset_index(drop=True)
            )
            rows.append(
                {
                    "feature": feature,
                    "feature_label": LAG_FEATURE_LABELS[feature],
                    "lag_weeks": int(lag_weeks),
                    "pearson_r": _safe_corr(merged[feature], merged["Total Dengue Cases"], "pearson"),
                    "spearman_rho": _safe_corr(merged[feature], merged["Total Dengue Cases"], "spearman"),
                    "overlap_weeks": int(len(merged)),
                    "analysis_start": merged["week_start"].min() if not merged.empty else pd.NaT,
                    "analysis_end": merged["week_start"].max() if not merged.empty else pd.NaT,
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "lag_weeks"]).reset_index(drop=True)


def build_best_lags(summary_df: pd.DataFrame, lag_column: str = "lag_days") -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "feature_label",
                lag_column,
                "pearson_r",
                "spearman_rho",
                "overlap_weeks",
                "analysis_start",
                "analysis_end",
                "abs_pearson_r",
            ]
        )

    work = summary_df.copy()
    work["abs_pearson_r"] = work["pearson_r"].abs()
    best_df = (
        work.sort_values(["feature", "abs_pearson_r", lag_column], ascending=[True, False, True])
        .groupby("feature", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best_df[
        [
            "feature",
            "feature_label",
            lag_column,
            "pearson_r",
            "spearman_rho",
            "overlap_weeks",
            "analysis_start",
            "analysis_end",
            "abs_pearson_r",
        ]
    ]


def build_literature_comparison_summary(
    summary_df: pd.DataFrame,
    benchmark_weeks: int = LITERATURE_BENCHMARK_WEEKS,
) -> pd.DataFrame:
    temp_df = summary_df[summary_df["feature"] == "avg_temp_c"].sort_values("lag_weeks").reset_index(drop=True)
    benchmark_df = temp_df[temp_df["lag_weeks"] == int(benchmark_weeks)].head(1)
    best_df = build_best_lags(temp_df, lag_column="lag_weeks")
    best_row = best_df.iloc[0] if not best_df.empty else None
    benchmark_row = benchmark_df.iloc[0] if not benchmark_df.empty else None
    return pd.DataFrame(
        [
            {
                "reading_anchor": "Koh et al. (2008)",
                "literature_benchmark_weeks": int(benchmark_weeks),
                "local_best_lag_weeks": int(best_row["lag_weeks"]) if best_row is not None else np.nan,
                "local_best_pearson_r": float(best_row["pearson_r"]) if best_row is not None else np.nan,
                "local_best_spearman_rho": float(best_row["spearman_rho"]) if best_row is not None else np.nan,
                "benchmark_pearson_r": float(benchmark_row["pearson_r"]) if benchmark_row is not None else np.nan,
                "benchmark_spearman_rho": float(benchmark_row["spearman_rho"]) if benchmark_row is not None else np.nan,
                "benchmark_overlap_weeks": int(benchmark_row["overlap_weeks"]) if benchmark_row is not None else 0,
                "analysis_start": benchmark_row["analysis_start"] if benchmark_row is not None else pd.NaT,
                "analysis_end": benchmark_row["analysis_end"] if benchmark_row is not None else pd.NaT,
                "literature_note": "Koh et al. (2008) reported the strongest temperature association at 18 weeks in Singapore.",
            }
        ]
    )


def build_historical_analysis_qa(
    weekly_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    base_df: pd.DataFrame,
    lag_summary_df: pd.DataFrame,
    weekly_lag_summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    weather_dates = pd.to_datetime(weather_df["query_date"], errors="coerce")
    overlap_mask = base_df["weather_overlap_flag"].astype(bool) if "weather_overlap_flag" in base_df.columns else base_df["avg_temp_c"].notna()
    overlap_rows = int(overlap_mask.sum())
    overlap_df = base_df.loc[overlap_mask, ["week_start"]].copy()

    qa_rows = [
        {"metric": "analysis_weather_end_date", "value": ANALYSIS_WEATHER_END_DATE},
        {"metric": "dengue_week_start_min", "value": weekly_df["week_start"].min()},
        {"metric": "dengue_week_start_max", "value": weekly_df["week_start"].max()},
        {"metric": "weather_query_start", "value": weather_dates.min()},
        {"metric": "weather_query_end", "value": weather_dates.max()},
        {"metric": "weekly_base_rows", "value": int(len(base_df))},
        {"metric": "weekly_base_duplicate_week_start", "value": int(base_df["week_start"].duplicated().sum())},
        {"metric": "weekly_base_overlap_rows", "value": overlap_rows},
        {"metric": "weekly_base_missing_weather_rows", "value": int(len(base_df) - overlap_rows)},
        {"metric": "weekly_base_overlap_start", "value": overlap_df["week_start"].min() if not overlap_df.empty else pd.NaT},
        {"metric": "weekly_base_overlap_end", "value": overlap_df["week_start"].max() if not overlap_df.empty else pd.NaT},
        {"metric": "lag_summary_rows", "value": int(len(lag_summary_df))},
        {"metric": "lag_unique_features", "value": int(lag_summary_df["feature"].nunique()) if not lag_summary_df.empty else 0},
        {"metric": "lag_unique_days", "value": int(lag_summary_df["lag_days"].nunique()) if not lag_summary_df.empty else 0},
    ]
    if weekly_lag_summary_df is not None:
        qa_rows.extend(
            [
                {"metric": "weekly_lag_summary_rows", "value": int(len(weekly_lag_summary_df))},
                {
                    "metric": "weekly_lag_unique_features",
                    "value": int(weekly_lag_summary_df["feature"].nunique()) if not weekly_lag_summary_df.empty else 0,
                },
                {
                    "metric": "weekly_lag_unique_weeks",
                    "value": int(weekly_lag_summary_df["lag_weeks"].nunique()) if not weekly_lag_summary_df.empty else 0,
                },
            ]
        )
    return pd.DataFrame(qa_rows)


def plot_lag_sweep_summary(summary_df: pd.DataFrame):
    fig = go.Figure()
    color_map = {
        "avg_temp_c": "#E76F51",
        "avg_relative_humidity_pct": "#2A9D8F",
        "avg_wind_speed": "#577590",
        "warm_day_share": "#B56576",
    }
    for feature in LAG_FEATURE_COLUMNS:
        tmp = summary_df[summary_df["feature"] == feature].sort_values("lag_days")
        fig.add_trace(
            go.Scatter(
                x=tmp["lag_days"],
                y=tmp["pearson_r"],
                mode="lines+markers",
                name=LAG_FEATURE_LABELS[feature],
                line=dict(color=color_map[feature], width=2.5),
            )
        )
    fig.add_hline(y=0, line_dash="dot")
    fig.add_vline(x=18, line_dash="dash", line_color="firebrick", annotation_text="18-day marker")
    fig.update_layout(
        title="Weather Lag Sweep Against Weekly Dengue Cases",
        xaxis_title="Lag days",
        yaxis_title="Pearson correlation",
        height=430,
        template="simple_white",
    )
    return fig


def plot_temperature_lag_focus(summary_df: pd.DataFrame):
    temp_df = summary_df[summary_df["feature"] == "avg_temp_c"].sort_values("lag_days").copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=temp_df["lag_days"],
            y=temp_df["pearson_r"],
            mode="lines+markers",
            name="Temperature",
            line=dict(color="#E76F51", width=3),
        )
    )
    fig.add_hline(y=0, line_dash="dot")
    fig.add_vline(x=18, line_dash="dash", line_color="firebrick", annotation_text="18-day hypothesis")

    if not temp_df.empty and temp_df["pearson_r"].notna().any():
        best_idx = temp_df["pearson_r"].abs().idxmax()
        best_row = temp_df.loc[best_idx]
        fig.add_trace(
            go.Scatter(
                x=[best_row["lag_days"]],
                y=[best_row["pearson_r"]],
                mode="markers",
                marker=dict(size=12, color="firebrick"),
                name=f"Best lag ({int(best_row['lag_days'])}d)",
            )
        )

    fig.update_layout(
        title="Temperature Lag Focus",
        xaxis_title="Lag days",
        yaxis_title="Pearson correlation",
        height=430,
        template="simple_white",
    )
    return fig


def plot_lag_correlation_heatmap(summary_df: pd.DataFrame):
    heat_df = (
        summary_df[["feature_label", "lag_days", "pearson_r"]]
        .pivot(index="feature_label", columns="lag_days", values="pearson_r")
        .reindex([LAG_FEATURE_LABELS[col] for col in LAG_FEATURE_COLUMNS])
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=heat_df.values,
            x=list(heat_df.columns),
            y=list(heat_df.index),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Pearson r"),
        )
    )
    fig.update_layout(
        title="Lag Correlation Heatmap",
        xaxis_title="Lag days",
        yaxis_title="Weather feature",
        height=420,
        template="simple_white",
    )
    return fig


def plot_weekly_temperature_lag_focus(
    summary_df: pd.DataFrame,
    benchmark_weeks: int = LITERATURE_BENCHMARK_WEEKS,
):
    temp_df = summary_df[summary_df["feature"] == "avg_temp_c"].sort_values("lag_weeks").copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=temp_df["lag_weeks"],
            y=temp_df["pearson_r"],
            mode="lines+markers",
            name="Temperature",
            line=dict(color="#E76F51", width=3),
        )
    )
    fig.add_hline(y=0, line_dash="dot")
    fig.add_vline(
        x=int(benchmark_weeks),
        line_dash="dash",
        line_color="firebrick",
        annotation_text=f"{int(benchmark_weeks)}-week literature benchmark",
    )

    benchmark_df = temp_df[temp_df["lag_weeks"] == int(benchmark_weeks)]
    if not benchmark_df.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_df["lag_weeks"],
                y=benchmark_df["pearson_r"],
                mode="markers",
                marker=dict(size=12, color="firebrick"),
                name=f"Pearson at {int(benchmark_weeks)}w",
            )
        )

    if not temp_df.empty and temp_df["pearson_r"].notna().any():
        best_idx = temp_df["pearson_r"].abs().idxmax()
        best_row = temp_df.loc[best_idx]
        fig.add_trace(
            go.Scatter(
                x=[best_row["lag_weeks"]],
                y=[best_row["pearson_r"]],
                mode="markers",
                marker=dict(size=12, color="#2A9D8F"),
                name=f"Local best lag ({int(best_row['lag_weeks'])}w)",
            )
        )

    fig.update_layout(
        title="Weekly Temperature Lag Analysis",
        xaxis_title="Lag weeks",
        yaxis_title="Pearson correlation",
        height=430,
        template="simple_white",
    )
    return fig


def plot_weekly_lag_correlation_heatmap(summary_df: pd.DataFrame):
    heat_df = (
        summary_df[["feature_label", "lag_weeks", "pearson_r"]]
        .pivot(index="feature_label", columns="lag_weeks", values="pearson_r")
        .reindex([LAG_FEATURE_LABELS[col] for col in LAG_FEATURE_COLUMNS])
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=heat_df.values,
            x=list(heat_df.columns),
            y=list(heat_df.index),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Pearson r"),
        )
    )
    fig.add_vline(x=LITERATURE_BENCHMARK_WEEKS, line_dash="dash", line_color="firebrick")
    fig.update_layout(
        title="Weekly Lag Correlation Heatmap",
        xaxis_title="Lag weeks",
        yaxis_title="Weather feature",
        height=420,
        template="simple_white",
    )
    return fig


def validate_weather_overlap(
    weekly_df: pd.DataFrame,
    weekly_weather_df: pd.DataFrame,
    lag_days: int,
    min_overlap_weeks: int = 26,
) -> None:
    overlap = weekly_df[["week_start"]].merge(weekly_weather_df[["week_start"]], on="week_start", how="inner")
    if len(overlap) >= int(min_overlap_weeks):
        return

    dengue_min = pd.to_datetime(weekly_df["week_start"], errors="coerce").min()
    dengue_max = pd.to_datetime(weekly_df["week_start"], errors="coerce").max()
    weather_min = pd.to_datetime(weekly_weather_df["week_start"], errors="coerce").min()
    weather_max = pd.to_datetime(weekly_weather_df["week_start"], errors="coerce").max()
    raise RuntimeError(
        "Weather/dengue overlap is insufficient for ARIMAX. "
        f"Dengue weeks: {dengue_min.date()} to {dengue_max.date()}. "
        f"Weather weeks after applying lag={lag_days}: {weather_min.date()} to {weather_max.date()}. "
        f"Overlap weeks found: {len(overlap)}."
    )


def build_arimax_inputs(weekly_df: pd.DataFrame, weekly_weather_df: pd.DataFrame) -> pd.DataFrame:
    merged = weekly_df.merge(weekly_weather_df, on="week_start", how="left")
    merged = merged.sort_values("week_start").reset_index(drop=True)
    for col in ARIMAX_EXOG_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[col] = merged[col].interpolate(limit_direction="both")
        if merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].median())
    return merged


def build_future_weather_exog(weekly_weather_df: pd.DataFrame, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    hist = weekly_weather_df.copy()
    hist["weekofyear"] = pd.to_datetime(hist["week_start"]).dt.isocalendar().week.astype(int)
    seasonal_template = hist.groupby("weekofyear", as_index=False)[ARIMAX_EXOG_COLUMNS].mean()
    fallback = hist[ARIMAX_EXOG_COLUMNS].mean()

    future_exog = pd.DataFrame({"week_start": future_index})
    future_exog["weekofyear"] = future_exog["week_start"].dt.isocalendar().week.astype(int)
    future_exog = future_exog.merge(seasonal_template, on="weekofyear", how="left")
    for col in ARIMAX_EXOG_COLUMNS:
        future_exog[col] = future_exog[col].fillna(fallback[col])
    return future_exog[["week_start"] + ARIMAX_EXOG_COLUMNS]


def fit_fixed_sarima_and_forecast(
    weekly_df: pd.DataFrame,
    steps: int = DEFAULT_FORECAST_HORIZON,
    alpha: float = DEFAULT_FORECAST_ALPHA,
) -> pd.DataFrame:
    y = weekly_df["Total Dengue Cases"].astype(float)
    model = SARIMAX(
        y,
        order=FIXED_ORDER,
        seasonal_order=FIXED_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(method="lbfgs", maxiter=200, disp=False)
    forecast = model.get_forecast(steps=int(steps))
    conf = forecast.conf_int(alpha=float(alpha))
    last_date = pd.to_datetime(weekly_df["week_start"]).iloc[-1]
    idx = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=int(steps), freq="W-SUN")
    return pd.DataFrame(
        {
            "week_start": idx,
            "predicted_cases": forecast.predicted_mean.values,
            "lower": conf.iloc[:, 0].values,
            "upper": conf.iloc[:, 1].values,
        }
    )


def fit_arimax_and_forecast(
    arimax_df: pd.DataFrame,
    weekly_weather_df: pd.DataFrame,
    steps: int = DEFAULT_FORECAST_HORIZON,
    alpha: float = DEFAULT_FORECAST_ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df = arimax_df.dropna(subset=["Total Dengue Cases"]).copy()
    y = model_df["Total Dengue Cases"].astype(float)
    exog = model_df[ARIMAX_EXOG_COLUMNS].astype(float)
    model = SARIMAX(
        y,
        exog=exog,
        order=FIXED_ORDER,
        seasonal_order=FIXED_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(method="lbfgs", maxiter=200, disp=False)
    last_date = pd.to_datetime(model_df["week_start"]).iloc[-1]
    idx = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=int(steps), freq="W-SUN")
    future_exog = build_future_weather_exog(weekly_weather_df, idx)
    forecast = model.get_forecast(steps=int(steps), exog=future_exog[ARIMAX_EXOG_COLUMNS].astype(float))
    conf = forecast.conf_int(alpha=float(alpha))
    fcst_df = pd.DataFrame(
        {
            "week_start": idx,
            "predicted_cases": forecast.predicted_mean.values,
            "lower": conf.iloc[:, 0].values,
            "upper": conf.iloc[:, 1].values,
        }
    )
    return fcst_df, future_exog


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return norm.cdf(x)


def assign_risk_bands(
    fcst_df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    low_cut: float = DEFAULT_RISK_LOW_CUT,
    high_cut: float = DEFAULT_RISK_HIGH_CUT,
) -> pd.DataFrame:
    out = fcst_df.copy()
    sigma = (out["upper"] - out["lower"]).abs() / (2 * RISK_Z)
    sigma = sigma.replace(0, np.nan).fillna(1.0)
    z = (float(threshold) - out["predicted_cases"]) / sigma
    p_exceed = 1 - normal_cdf(z.to_numpy())
    out["p_exceed_threshold"] = np.clip(p_exceed, 0.0, 1.0)
    out["risk_band"] = np.select(
        [out["p_exceed_threshold"] < float(low_cut), out["p_exceed_threshold"] < float(high_cut)],
        ["Low", "Medium"],
        default="High",
    )
    return out


def stl_diagnostics(weekly_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    ts = pd.Series(weekly_df["Total Dengue Cases"].values, index=weekly_df["week_start"])
    stl = STL(ts, period=SEASONAL_PERIOD, robust=True).fit()
    seasonal_strength = max(0.0, 1.0 - (np.var(stl.resid) / np.var(stl.seasonal + stl.resid)))
    trend_series = pd.Series(stl.trend, index=weekly_df["week_start"]).astype(float)
    recent_trend = trend_series.dropna().tail(6)
    trend_slope = float((recent_trend.iloc[-1] - recent_trend.iloc[0]) / max(1, len(recent_trend) - 1))
    stl_df = pd.DataFrame(
        {
            "week_start": weekly_df["week_start"].to_numpy(),
            "observed": ts.to_numpy(),
            "trend": np.asarray(stl.trend),
            "seasonal": np.asarray(stl.seasonal),
            "resid": np.asarray(stl.resid),
        }
    )
    diag_df = pd.DataFrame(
        [
            {"metric": "seasonal_period_weeks", "value": SEASONAL_PERIOD},
            {"metric": "seasonal_strength", "value": round(float(seasonal_strength), 4)},
            {"metric": "recent_trend_slope_per_week", "value": round(trend_slope, 4)},
        ]
    )
    return stl_df, diag_df, seasonal_strength, trend_slope


def build_seasonal_profile(stl_df: pd.DataFrame) -> pd.DataFrame:
    tmp = stl_df.copy()
    tmp["month"] = pd.to_datetime(tmp["week_start"]).dt.month
    prof = tmp.groupby("month", as_index=False)["seasonal"].mean()
    prof["month_name"] = pd.to_datetime(prof["month"], format="%m").dt.strftime("%b")
    prof["is_peak_window"] = prof["month"].isin([5, 6])
    return prof


def build_monthly_case_seasonality(weekly_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = (
        weekly_df.assign(
            year=pd.to_datetime(weekly_df["week_start"]).dt.year,
            month=pd.to_datetime(weekly_df["week_start"]).dt.month,
        )
        .groupby(["year", "month"], as_index=False)["Total Dengue Cases"]
        .mean()
        .rename(columns={"Total Dengue Cases": "avg_cases"})
    )
    monthly["month_name"] = pd.to_datetime(monthly["month"], format="%m").dt.strftime("%b")
    month_range = (
        monthly.groupby(["month", "month_name"], as_index=False)["avg_cases"]
        .agg(month_min="min", month_max="max", month_avg="mean")
        .sort_values("month")
        .reset_index(drop=True)
    )
    return monthly, month_range


def build_monthly_temperature_seasonality(weekly_weather_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = (
        weekly_weather_df.assign(
            year=pd.to_datetime(weekly_weather_df["week_start"]).dt.year,
            month=pd.to_datetime(weekly_weather_df["week_start"]).dt.month,
        )
        .groupby(["year", "month"], as_index=False)[["avg_temp_c", "min_temp_c", "max_temp_c"]]
        .mean()
    )
    monthly["month_name"] = pd.to_datetime(monthly["month"], format="%m").dt.strftime("%b")
    month_range = (
        monthly.groupby(["month", "month_name"], as_index=False)[["avg_temp_c", "min_temp_c", "max_temp_c"]]
        .agg({"avg_temp_c": "mean", "min_temp_c": "min", "max_temp_c": "max"})
        .rename(
            columns={
                "avg_temp_c": "month_avg_temp_c",
                "min_temp_c": "month_min_temp_c",
                "max_temp_c": "month_max_temp_c",
            }
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    return monthly, month_range


def build_driver_metrics(arimax_input_df: pd.DataFrame) -> pd.DataFrame:
    work = arimax_input_df.dropna(subset=["Total Dengue Cases"] + ARIMAX_EXOG_COLUMNS).copy()
    label_map = {
        "avg_temp_c": "Temperature correlation",
        "avg_relative_humidity_pct": "Humidity correlation",
        "avg_wind_speed": "Wind correlation",
        "warm_day_share": "Warm-day share correlation",
    }
    metrics = []
    for col in ARIMAX_EXOG_COLUMNS:
        corr = work["Total Dengue Cases"].corr(work[col])
        metrics.append({"metric": label_map[col], "value": round(float(corr), 3) if pd.notna(corr) else np.nan})
    return pd.DataFrame(metrics)


def artifact_paths(base_dir: Path) -> dict[str, Path]:
    root = Path(base_dir)
    return {
        "base": root / WEEKLY_BASE_FILENAME,
        "lag_summary": root / LAG_SUMMARY_FILENAME,
        "best_lags": root / BEST_LAGS_FILENAME,
        "weekly_lag_summary": root / WEEKLY_LAG_SUMMARY_FILENAME,
        "weekly_best_lags": root / WEEKLY_BEST_LAGS_FILENAME,
        "literature_summary": root / LITERATURE_SUMMARY_FILENAME,
        "qa": root / QA_FILENAME,
        "manifest": root / MANIFEST_FILENAME,
        "sarima_forecast": root / SARIMA_FORECAST_FILENAME,
        "arimax_forecast": root / ARIMAX_FORECAST_FILENAME,
        "arimax_future_exog": root / ARIMAX_FUTURE_EXOG_FILENAME,
    }


def _read_csv_artifact(path: Path, date_cols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in date_cols or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def split_weekly_base_artifact(base_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly_cols = [
        "week_start",
        "Dengue Fever",
        "Dengue Haemorrhagic Fever",
        "Total Dengue Cases",
        "moving_avg_12w",
    ]
    weather_cols = [
        "week_start",
        "avg_temp_c",
        "min_temp_c",
        "max_temp_c",
        "avg_relative_humidity_pct",
        "avg_wind_speed",
        "warm_day_share",
        "source_rows",
        "source_query_start",
        "source_query_end",
        "weather_lag_days",
    ]
    weekly_df = (
        base_df[[col for col in weekly_cols if col in base_df.columns]]
        .drop_duplicates(subset=["week_start"])
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    weekly_weather_df = base_df[[col for col in weather_cols if col in base_df.columns]].copy()
    weekly_weather_df = (
        weekly_weather_df.drop_duplicates(subset=["week_start"])
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    weekly_weather_df = weekly_weather_df.dropna(
        subset=["avg_temp_c", "avg_relative_humidity_pct", "avg_wind_speed", "warm_day_share"],
        how="all",
    ).reset_index(drop=True)
    if "weather_lag_days" not in weekly_weather_df.columns:
        weekly_weather_df["weather_lag_days"] = DEFAULT_ANALYSIS_LAG_DAYS
    return weekly_df, weekly_weather_df


def _source_metadata(path: Path) -> dict[str, str]:
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return {"path": str(path), "mtime_utc": ts}


def build_analysis_manifest(
    base_dir: Path,
    dengue_path: Path,
    weather_path: Path,
    artifact_row_counts: dict[str, int],
) -> dict:
    paths = artifact_paths(base_dir)
    artifact_meta = {}
    for key, path in paths.items():
        if key == "manifest":
            continue
        artifact_meta[key] = {
            "path": str(path),
            "rows": int(artifact_row_counts.get(key, 0)),
        }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_weather_end_date": ANALYSIS_WEATHER_END_DATE.strftime("%Y-%m-%d"),
        "default_weather_lag_days": DEFAULT_ANALYSIS_LAG_DAYS,
        "lag_sweep_days": {
            "min": min(LAG_SWEEP_DAYS),
            "max": max(LAG_SWEEP_DAYS),
            "count": len(LAG_SWEEP_DAYS),
        },
        "lag_sweep_weeks": {
            "min": min(LAG_SWEEP_WEEKS),
            "max": max(LAG_SWEEP_WEEKS),
            "count": len(LAG_SWEEP_WEEKS),
        },
        "literature_benchmark_weeks": LITERATURE_BENCHMARK_WEEKS,
        "default_forecast": {
            "horizon_weeks": DEFAULT_FORECAST_HORIZON,
            "alpha": DEFAULT_FORECAST_ALPHA,
            "threshold": DEFAULT_THRESHOLD,
        },
        "sources": {
            "dengue": _source_metadata(Path(dengue_path)),
            "weather": _source_metadata(Path(weather_path)),
        },
        "artifacts": artifact_meta,
    }


def write_analysis_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_default_forecast_artifacts(
    weekly_df: pd.DataFrame,
    weekly_weather_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_weather_overlap(weekly_df, weekly_weather_df, DEFAULT_ANALYSIS_LAG_DAYS)
    arimax_input_df = build_arimax_inputs(weekly_df, weekly_weather_df)
    sarima_df = fit_fixed_sarima_and_forecast(
        weekly_df,
        steps=DEFAULT_FORECAST_HORIZON,
        alpha=DEFAULT_FORECAST_ALPHA,
    )
    arimax_df, future_exog_df = fit_arimax_and_forecast(
        arimax_input_df,
        weekly_weather_df,
        steps=DEFAULT_FORECAST_HORIZON,
        alpha=DEFAULT_FORECAST_ALPHA,
    )
    return sarima_df, arimax_df, future_exog_df


def _winkler_score(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    below_penalty = (2.0 / alpha) * (lower - actual) * (actual < lower)
    above_penalty = (2.0 / alpha) * (actual - upper) * (actual > upper)
    return width + below_penalty + above_penalty


def _brier_score(p_exceed: np.ndarray, actual_exceeds: np.ndarray) -> float:
    p_exceed = np.asarray(p_exceed, dtype=float)
    actual_exceeds = np.asarray(actual_exceeds, dtype=float)
    if len(p_exceed) == 0:
        return float("nan")
    return float(np.mean((p_exceed - actual_exceeds) ** 2))


def _diebold_mariano_hln(errors_a: np.ndarray, errors_b: np.ndarray, h: int) -> tuple[float, float]:
    from scipy.stats import t as student_t

    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    n = len(errors_a)
    if n != len(errors_b) or n < max(8, 2 * h):
        return float("nan"), float("nan")

    d = errors_a ** 2 - errors_b ** 2
    d_mean = float(np.mean(d))

    gamma_0 = float(np.var(d, ddof=0))
    var_d = gamma_0
    for k in range(1, h):
        cov_k = float(np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)))
        var_d += 2.0 * cov_k

    if var_d <= 0:
        return float("nan"), float("nan")

    dm_stat = d_mean / np.sqrt(var_d / n)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat *= correction

    p_value = 2.0 * (1.0 - student_t.cdf(abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p_value)


def walk_forward_backtest(
    weekly_df: pd.DataFrame,
    weekly_weather_df: pd.DataFrame | None = None,
    *,
    model_kind: str,
    horizon: int = BACKTEST_HORIZON,
    step: int = BACKTEST_STEP_WEEKS,
    min_train_weeks: int = BACKTEST_MIN_TRAIN_WEEKS,
    alpha: float = BACKTEST_ALPHA,
    eval_thresholds: tuple[int, ...] = BACKTEST_EVAL_THRESHOLDS,
) -> dict[str, pd.DataFrame]:
    if model_kind not in {"sarima", "sarimax"}:
        raise ValueError(f"model_kind must be 'sarima' or 'sarimax', got {model_kind!r}")
    if model_kind == "sarimax" and weekly_weather_df is None:
        raise ValueError("weekly_weather_df is required for model_kind='sarimax'")

    work = (
        weekly_df.dropna(subset=["Total Dengue Cases"])
        .sort_values("week_start")
        .reset_index(drop=True)
        .copy()
    )

    n = len(work)
    first_origin_idx = min_train_weeks - 1
    last_origin_idx = n - horizon - 1
    if last_origin_idx <= first_origin_idx:
        raise RuntimeError(
            f"Not enough data for backtest: n={n}, min_train_weeks={min_train_weeks}, horizon={horizon}"
        )

    origin_indices = list(range(first_origin_idx, last_origin_idx + 1, step))
    if len(origin_indices) < 10:
        raise RuntimeError(
            f"Backtest produced only {len(origin_indices)} origins; need >= 10. "
            f"Reduce min_train_weeks or step."
        )

    per_origin_rows: list[dict] = []
    skipped_origins: list[dict] = []

    for origin_idx in origin_indices:
        origin_week = work["week_start"].iloc[origin_idx]
        train_df = work.iloc[: origin_idx + 1].copy()
        target_slice = work.iloc[origin_idx + 1 : origin_idx + 1 + horizon].copy()
        if len(target_slice) < horizon:
            continue
        target_weeks = target_slice["week_start"].to_numpy()
        actuals = target_slice["Total Dengue Cases"].to_numpy(dtype=float)

        try:
            if model_kind == "sarima":
                y = train_df["Total Dengue Cases"].astype(float)
                sm_model = SARIMAX(
                    y,
                    order=FIXED_ORDER,
                    seasonal_order=FIXED_SEASONAL_ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(method="lbfgs", maxiter=200, disp=False)
                forecast = sm_model.get_forecast(steps=horizon)
            else:
                train_weather = weekly_weather_df[
                    weekly_weather_df["week_start"] <= origin_week
                ].copy()
                if len(train_weather) < min_train_weeks // 2:
                    skipped_origins.append({
                        "origin_week": origin_week,
                        "reason": "insufficient_training_weather",
                    })
                    continue
                arimax_input = build_arimax_inputs(train_df, train_weather)
                model_df = arimax_input.dropna(subset=["Total Dengue Cases"])
                y = model_df["Total Dengue Cases"].astype(float)
                exog_train = model_df[ARIMAX_EXOG_COLUMNS].astype(float)

                future_index = pd.DatetimeIndex(target_weeks)
                future_exog = build_future_weather_exog(train_weather, future_index)

                sm_model = SARIMAX(
                    y,
                    exog=exog_train,
                    order=FIXED_ORDER,
                    seasonal_order=FIXED_SEASONAL_ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(method="lbfgs", maxiter=200, disp=False)
                forecast = sm_model.get_forecast(
                    steps=horizon,
                    exog=future_exog[ARIMAX_EXOG_COLUMNS].astype(float),
                )
        except Exception as exc:
            skipped_origins.append({
                "origin_week": origin_week,
                "reason": f"fit_or_forecast_failed: {type(exc).__name__}",
            })
            continue

        predicted = np.asarray(forecast.predicted_mean.values, dtype=float)
        conf = forecast.conf_int(alpha=alpha)
        lower = conf.iloc[:, 0].to_numpy(dtype=float)
        upper = conf.iloc[:, 1].to_numpy(dtype=float)

        sigma = np.maximum(np.abs(upper - lower) / (2.0 * 1.2816), 1e-9)
        for h_idx in range(horizon):
            row = {
                "origin_week": origin_week,
                "target_week": pd.Timestamp(target_weeks[h_idx]),
                "h": h_idx + 1,
                "predicted": predicted[h_idx],
                "lower": lower[h_idx],
                "upper": upper[h_idx],
                "actual": actuals[h_idx],
                "abs_error": abs(predicted[h_idx] - actuals[h_idx]),
                "signed_error": predicted[h_idx] - actuals[h_idx],
                "in_pi": bool(lower[h_idx] <= actuals[h_idx] <= upper[h_idx]),
                "winkler_alpha20": float(
                    _winkler_score(
                        np.array([actuals[h_idx]]),
                        np.array([lower[h_idx]]),
                        np.array([upper[h_idx]]),
                        alpha,
                    )[0]
                ),
            }
            for thr in eval_thresholds:
                z = (thr - predicted[h_idx]) / sigma[h_idx]
                row[f"p_exceed_{thr}"] = float(np.clip(1.0 - norm.cdf(z), 0.0, 1.0))
            per_origin_rows.append(row)

    per_origin_df = pd.DataFrame(per_origin_rows)
    if per_origin_df.empty:
        raise RuntimeError(f"Backtest for {model_kind} produced zero rows; all origins skipped.")

    horizon_rows = []
    for h_val in range(1, horizon + 1):
        sub = per_origin_df[per_origin_df["h"] == h_val]
        if sub.empty:
            continue
        denom = (np.abs(sub["predicted"]) + np.abs(sub["actual"])).replace(0, np.nan)
        smape = float((2.0 * sub["abs_error"] / denom).mean(skipna=True))
        horizon_rows.append({
            "h": h_val,
            "n_origins": int(len(sub)),
            "mae": float(sub["abs_error"].mean()),
            "rmse": float(np.sqrt(np.mean(sub["signed_error"] ** 2))),
            "smape": smape,
            "bias": float(sub["signed_error"].mean()),
            "pi_coverage": float(sub["in_pi"].mean()),
            "pi_width_mean": float((sub["upper"] - sub["lower"]).mean()),
            "winkler_alpha20": float(sub["winkler_alpha20"].mean()),
        })
    horizon_metrics_df = pd.DataFrame(horizon_rows)

    summary_rows = []
    for thr in eval_thresholds:
        pred_exceeds = per_origin_df["predicted"] >= thr
        actual_exceeds = per_origin_df["actual"] >= thr
        tp = int((pred_exceeds & actual_exceeds).sum())
        fp = int((pred_exceeds & ~actual_exceeds).sum())
        tn = int((~pred_exceeds & ~actual_exceeds).sum())
        fn = int((~pred_exceeds & actual_exceeds).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) and not np.isnan(precision) and not np.isnan(recall)
            else float("nan")
        )
        denom = (np.abs(per_origin_df["predicted"]) + np.abs(per_origin_df["actual"])).replace(0, np.nan)
        smape_overall = float((2.0 * per_origin_df["abs_error"] / denom).mean(skipna=True))
        summary_rows.append({
            "threshold": int(thr),
            "n_pairs": int(len(per_origin_df)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "brier_score": _brier_score(
                per_origin_df[f"p_exceed_{thr}"].to_numpy(),
                actual_exceeds.to_numpy(),
            ),
            "mae_overall": float(per_origin_df["abs_error"].mean()),
            "smape_overall": smape_overall,
        })
    summary_df = pd.DataFrame(summary_rows)

    return {
        "per_origin": per_origin_df,
        "horizon_metrics": horizon_metrics_df,
        "summary": summary_df,
        "skipped": pd.DataFrame(skipped_origins),
    }


def _safe_read_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates) if parse_dates else pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_analysis_artifacts(base_dir: Path) -> dict[str, object]:
    paths = artifact_paths(base_dir)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            "Analysis artifacts are missing. Run build_historical_lag_analysis.py first. "
            f"Missing: {missing_text}"
        )

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    base_df = _read_csv_artifact(
        paths["base"],
        date_cols=["week_start", "source_query_start", "source_query_end"],
    )
    lag_summary_df = _read_csv_artifact(
        paths["lag_summary"],
        date_cols=["analysis_start", "analysis_end"],
    )
    best_lags_df = _read_csv_artifact(
        paths["best_lags"],
        date_cols=["analysis_start", "analysis_end"],
    )
    weekly_lag_summary_df = _read_csv_artifact(
        paths["weekly_lag_summary"],
        date_cols=["analysis_start", "analysis_end"],
    )
    weekly_best_lags_df = _read_csv_artifact(
        paths["weekly_best_lags"],
        date_cols=["analysis_start", "analysis_end"],
    )
    literature_summary_df = _read_csv_artifact(
        paths["literature_summary"],
        date_cols=["analysis_start", "analysis_end"],
    )
    qa_df = _read_csv_artifact(paths["qa"])
    sarima_forecast_df = _read_csv_artifact(paths["sarima_forecast"], date_cols=["week_start"])
    arimax_forecast_df = _read_csv_artifact(paths["arimax_forecast"], date_cols=["week_start"])
    arimax_future_exog_df = _read_csv_artifact(paths["arimax_future_exog"], date_cols=["week_start"])

    row_expectations = {
        "base": len(base_df),
        "lag_summary": len(lag_summary_df),
        "best_lags": len(best_lags_df),
        "weekly_lag_summary": len(weekly_lag_summary_df),
        "weekly_best_lags": len(weekly_best_lags_df),
        "literature_summary": len(literature_summary_df),
        "qa": len(qa_df),
        "sarima_forecast": len(sarima_forecast_df),
        "arimax_forecast": len(arimax_forecast_df),
        "arimax_future_exog": len(arimax_future_exog_df),
    }
    for key, observed_rows in row_expectations.items():
        expected_rows = int(manifest.get("artifacts", {}).get(key, {}).get("rows", observed_rows))
        if expected_rows != int(observed_rows):
            raise RuntimeError(
                f"Artifact row count mismatch for {key}: manifest={expected_rows}, file={observed_rows}."
            )

    weekly_df, weekly_weather_df = split_weekly_base_artifact(base_df)
    return {
        "paths": paths,
        "manifest": manifest,
        "base_df": base_df,
        "weekly_df": weekly_df,
        "weekly_weather_df": weekly_weather_df,
        "lag_summary_df": lag_summary_df,
        "best_lags_df": best_lags_df,
        "weekly_lag_summary_df": weekly_lag_summary_df,
        "weekly_best_lags_df": weekly_best_lags_df,
        "literature_summary_df": literature_summary_df,
        "qa_df": qa_df,
        "sarima_forecast_df": sarima_forecast_df,
        "arimax_forecast_df": arimax_forecast_df,
        "arimax_future_exog_df": arimax_future_exog_df,
        "backtest_per_origin_df": _safe_read_csv(
            base_dir / "backtest_per_origin.csv",
            parse_dates=["origin_week", "target_week"],
        ),
        "backtest_horizon_metrics_df": _safe_read_csv(base_dir / "backtest_horizon_metrics.csv"),
        "backtest_summary_df": _safe_read_csv(base_dir / "backtest_summary.csv"),
        "backtest_diebold_mariano_df": _safe_read_csv(base_dir / "backtest_diebold_mariano.csv"),
    }
