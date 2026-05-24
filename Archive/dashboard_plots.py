"""
Notebook-style plot builder for the NEA dengue dashboard.

Open this file in VS Code or Jupyter-compatible editors and run the cells
individually. Each plot is designed to be easy to export into slides.
"""

# %%
from pathlib import Path
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from historical_lag_analysis import (
    ANALYSIS_WEATHER_END_DATE,
    build_arimax_inputs as shared_build_arimax_inputs,
    build_seasonal_profile as shared_build_seasonal_profile,
    load_analysis_artifacts,
    plot_lag_correlation_heatmap,
    plot_lag_sweep_summary,
    plot_temperature_lag_focus,
    stl_diagnostics as shared_stl_diagnostics,
)


# %%
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "singapore_dengue_raw_records.csv"
WEATHER_FILE = BASE_DIR / "singapore_weather_forecast_24hr_history.csv"
EXPORT_DIR = BASE_DIR / "dashboard_plots_exports"
WEATHER_FORECAST_URL = "https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast"
WEATHER_HISTORY_START_DATE = pd.Timestamp("2016-03-01")
MIN_OVERLAP_WEEKS = 26
SHOW_FIGURES_ENV = "DASHBOARD_PLOTS_SHOW"

SEASONAL_PERIOD = 52
FIXED_ORDER = (1, 1, 1)
FIXED_SEASONAL_ORDER = (0, 1, 1, SEASONAL_PERIOD)
FORECAST_ALPHA = 0.2
RISK_Z = 1.2816
WATCH_MONTHS = {4, 5, 6, 7}
DEFAULT_WEATHER_LAG_DAYS = 18
ARIMAX_EXOG_COLUMNS = [
    "avg_temp_c",
    "avg_relative_humidity_pct",
    "avg_wind_speed",
    "warm_day_share",
]
WEATHER_REGIONS = ("west", "east", "central", "north", "south")


# %%
def request_json(url: str, params: dict | None = None, max_attempts: int = 8, timeout: int = 30) -> dict:
    import requests
    import time

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt == max_attempts:
                    resp.raise_for_status()
                time.sleep(min(2**attempt, 60))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_err = exc
            if attempt == max_attempts:
                raise
            time.sleep(min(2**attempt, 60))
    raise RuntimeError(f"Request failed: {last_err}")


def request_json_allow_404(url: str, params: dict | None = None, max_attempts: int = 8, timeout: int = 30) -> dict | None:
    import requests
    import time

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
            if resp.status_code == 404:
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt == max_attempts:
                    resp.raise_for_status()
                time.sleep(min(2**attempt, 60))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_err = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 404:
                return None
            if attempt == max_attempts:
                raise
            time.sleep(min(2**attempt, 60))
    raise RuntimeError(f"Request failed: {last_err}")


def parse_epi_week_to_sunday(epi_week: str):
    match = pd.Series([str(epi_week).strip()]).str.extract(r"(\d{4})-W(\d{1,2})")
    if match.isna().any(axis=None):
        return pd.NaT
    year = int(match.iloc[0, 0])
    week = int(match.iloc[0, 1])
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    first_week_sunday = jan1 - pd.Timedelta(days=(jan1.weekday() + 1) % 7)
    return first_week_sunday + pd.Timedelta(weeks=week - 1)


def load_dengue_records() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing dengue cache: {DATA_FILE}")
    return pd.read_csv(DATA_FILE)


def merge_weather_cache(existing_df: pd.DataFrame, new_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not new_frames:
        return existing_df

    additions_df = pd.concat(new_frames, ignore_index=True)
    if existing_df.empty:
        combined_df = additions_df
    else:
        combined_df = pd.concat([existing_df, additions_df], ignore_index=True)

    dedupe_cols = [col for col in ["query_date", "timestamp", "updated_timestamp", "valid_period_start"] if col in combined_df.columns]
    if dedupe_cols:
        combined_df = combined_df.drop_duplicates(subset=dedupe_cols, keep="last")

    sort_cols = [col for col in ["query_date", "timestamp", "updated_timestamp"] if col in combined_df.columns]
    if sort_cols:
        combined_df = combined_df.sort_values(sort_cols).reset_index(drop=True)

    combined_df.to_csv(WEATHER_FILE, index=False)
    return combined_df


def required_weather_range(weekly_df: pd.DataFrame, lag_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    dengue_start = pd.to_datetime(weekly_df["week_start"], errors="coerce").min()
    dengue_end = pd.to_datetime(weekly_df["week_start"], errors="coerce").max()
    start_date = max(WEATHER_HISTORY_START_DATE, (dengue_start - pd.Timedelta(days=lag_days)).normalize())
    end_date = (dengue_end - pd.Timedelta(days=lag_days)).normalize()
    if end_date < start_date:
        raise RuntimeError("Computed weather range is invalid after applying the lag.")
    return start_date, end_date


def load_weather_history(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    effective_end = min(pd.Timestamp(end_date).normalize(), ANALYSIS_WEATHER_END_DATE)
    cached_df = load_historical_weather_data(
        WEATHER_FILE,
        start_date=WEATHER_HISTORY_START_DATE,
        end_date=effective_end,
    )
    cached_df["query_date"] = pd.to_datetime(cached_df["query_date"], errors="coerce")
    cached_df = cached_df.dropna(subset=["query_date"]).copy()
    cached_df = cached_df[
        (cached_df["query_date"] >= start_date) & (cached_df["query_date"] <= effective_end)
    ].reset_index(drop=True)
    if cached_df.empty:
        raise RuntimeError(
            f"No weather records were available for the required range {start_date.date()} to {effective_end.date()}."
        )
    return cached_df


def to_week_start_sunday(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, errors="coerce")
    day_offset = (ts.dt.dayofweek + 1) % 7
    return (ts - pd.to_timedelta(day_offset, unit="D")).dt.normalize()


def build_weekly_dengue_series(records_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = records_df.copy()
    work["no._of_cases"] = pd.to_numeric(work["no._of_cases"], errors="coerce").fillna(0)
    work = work[work["disease"].isin(["Dengue Fever", "Dengue Haemorrhagic Fever"])].copy()
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
            {"check": "non_empty", "value": int(len(weekly))},
            {"check": "duplicate_week_start", "value": int(weekly["week_start"].duplicated().sum())},
            {"check": "negative_cases", "value": int((weekly["Total Dengue Cases"] < 0).sum())},
        ]
    )
    return weekly[["week_start", "Dengue Fever", "Dengue Haemorrhagic Fever", "Total Dengue Cases", "moving_avg_12w"]], qa_df


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
    prof["is_peak_window"] = prof["month"].isin([5, 6, 7])
    return prof


def flatten_weather_record(record: dict) -> dict:
    general = record.get("general", {}) or {}
    valid_period = general.get("validPeriod", {}) or {}
    forecast = general.get("forecast", {}) or {}
    temperature = general.get("temperature", {}) or {}
    relative_humidity = general.get("relativeHumidity", {}) or {}
    wind = general.get("wind", {}) or {}
    wind_speed = wind.get("speed", {}) or {}

    row = {
        "query_date": record.get("query_date"),
        "record_date": record.get("date"),
        "updated_timestamp": record.get("updatedTimestamp"),
        "timestamp": record.get("timestamp"),
        "valid_period_start": valid_period.get("start"),
        "valid_period_end": valid_period.get("end"),
        "valid_period_text": valid_period.get("text"),
        "general_forecast_code": forecast.get("code"),
        "general_forecast_text": forecast.get("text"),
        "temperature_low_c": temperature.get("low"),
        "temperature_high_c": temperature.get("high"),
        "relative_humidity_low_pct": relative_humidity.get("low"),
        "relative_humidity_high_pct": relative_humidity.get("high"),
        "wind_speed_low": wind_speed.get("low"),
        "wind_speed_high": wind_speed.get("high"),
        "wind_direction": wind.get("direction"),
        "period_count": len(record.get("periods", []) or []),
    }
    for idx, period in enumerate(record.get("periods", []) or [], start=1):
        time_period = period.get("timePeriod", {}) or {}
        regions = period.get("regions", {}) or {}
        row[f"period_{idx}_start"] = time_period.get("start")
        row[f"period_{idx}_end"] = time_period.get("end")
        row[f"period_{idx}_text"] = time_period.get("text")
        for region in WEATHER_REGIONS:
            region_forecast = regions.get(region, {}) or {}
            row[f"period_{idx}_{region}_forecast_code"] = region_forecast.get("code")
            row[f"period_{idx}_{region}_forecast_text"] = region_forecast.get("text")
    return row


def build_weekly_weather_features(weather_df: pd.DataFrame, lag_days: int = DEFAULT_WEATHER_LAG_DAYS) -> pd.DataFrame:
    work = weather_df.copy()
    work["query_date"] = pd.to_datetime(work["query_date"], errors="coerce") + pd.Timedelta(days=lag_days)
    work = work.dropna(subset=["query_date"]).sort_values("query_date").reset_index(drop=True)
    for col in ["temperature_low_c", "temperature_high_c", "relative_humidity_low_pct", "relative_humidity_high_pct", "wind_speed_low", "wind_speed_high"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["avg_temp_c"] = work[["temperature_low_c", "temperature_high_c"]].mean(axis=1)
    work["avg_relative_humidity_pct"] = work[["relative_humidity_low_pct", "relative_humidity_high_pct"]].mean(axis=1)
    work["avg_wind_speed"] = work[["wind_speed_low", "wind_speed_high"]].mean(axis=1)
    work["warm_day_flag"] = (work["temperature_high_c"] >= 32).astype(float)
    work["week_start"] = to_week_start_sunday(work["query_date"])
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


def validate_weather_overlap(weekly_df: pd.DataFrame, weekly_weather_df: pd.DataFrame, lag_days: int) -> None:
    overlap = weekly_df[["week_start"]].merge(weekly_weather_df[["week_start"]], on="week_start", how="inner")
    if len(overlap) < MIN_OVERLAP_WEEKS:
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


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return norm.cdf(x)


def fit_fixed_sarima_and_forecast(weekly_df: pd.DataFrame, steps: int, alpha: float = FORECAST_ALPHA):
    y = weekly_df["Total Dengue Cases"].astype(float)
    model = SARIMAX(
        y,
        order=FIXED_ORDER,
        seasonal_order=FIXED_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(method="lbfgs", maxiter=30, disp=False)
    forecast = model.get_forecast(steps=steps)
    conf = forecast.conf_int(alpha=alpha)
    last_date = weekly_df["week_start"].iloc[-1]
    idx = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=steps, freq="W-SUN")
    return pd.DataFrame(
        {
            "week_start": idx,
            "predicted_cases": forecast.predicted_mean.values,
            "lower": conf.iloc[:, 0].values,
            "upper": conf.iloc[:, 1].values,
        }
    )


def fit_arimax_and_forecast(arimax_df: pd.DataFrame, weekly_weather_df: pd.DataFrame, steps: int, alpha: float = FORECAST_ALPHA):
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
    ).fit(method="lbfgs", maxiter=40, disp=False)
    last_date = model_df["week_start"].iloc[-1]
    idx = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=steps, freq="W-SUN")
    future_exog = build_future_weather_exog(weekly_weather_df, idx)
    forecast = model.get_forecast(steps=steps, exog=future_exog[ARIMAX_EXOG_COLUMNS].astype(float))
    conf = forecast.conf_int(alpha=alpha)
    return (
        pd.DataFrame(
            {
                "week_start": idx,
                "predicted_cases": forecast.predicted_mean.values,
                "lower": conf.iloc[:, 0].values,
                "upper": conf.iloc[:, 1].values,
            }
        ),
        future_exog,
    )


# %%
def plot_history(weekly_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_df["week_start"], y=weekly_df["Total Dengue Cases"], name="Weekly cases", line=dict(width=1.8, color="#0B3C49")))
    fig.add_trace(go.Scatter(x=weekly_df["week_start"], y=weekly_df["moving_avg_12w"], name="12-week MA", line=dict(width=2.6, color="#7A001F")))
    fig.update_layout(title="Weekly Dengue Cases", xaxis_title="Week", yaxis_title="Cases", height=430, template="simple_white")
    return fig


def plot_stl(stl_df: pd.DataFrame):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
    fig.add_trace(go.Scatter(x=stl_df["week_start"], y=stl_df["observed"], name="Observed", line=dict(color="#0B3C49")), row=1, col=1)
    fig.add_trace(go.Scatter(x=stl_df["week_start"], y=stl_df["trend"], name="Trend", line=dict(color="#7A001F")), row=2, col=1)
    fig.add_trace(go.Scatter(x=stl_df["week_start"], y=stl_df["seasonal"], name="Seasonal", line=dict(color="#B56576")), row=3, col=1)
    fig.add_trace(go.Scatter(x=stl_df["week_start"], y=stl_df["resid"], name="Residual", mode="markers", marker=dict(size=4, color="#355070")), row=4, col=1)
    fig.update_layout(height=760, title="STL Decomposition", template="simple_white")
    return fig


def plot_seasonal_profile(profile_df: pd.DataFrame):
    colors = np.where(profile_df["is_peak_window"], "#7A001F", "#0B3C49")
    fig = go.Figure(go.Bar(x=profile_df["month_name"], y=profile_df["seasonal"], marker_color=colors, name="Avg STL seasonal"))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title="Month-of-Year Seasonal Profile", xaxis_title="Month", yaxis_title="Average seasonal component", height=420, template="simple_white")
    return fig


def plot_temperature_history(weekly_weather_df: pd.DataFrame):
    hist = weekly_weather_df.tail(104).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(hist["week_start"]) + list(hist["week_start"])[::-1],
            y=list(hist["max_temp_c"]) + list(hist["min_temp_c"])[::-1],
            fill="toself",
            fillcolor="rgba(255,140,0,0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Weekly low-high envelope",
        )
    )
    fig.add_trace(go.Scatter(x=hist["week_start"], y=hist["avg_temp_c"], name="Weekly average temperature", line=dict(color="darkorange", width=2.6)))
    fig.update_layout(title="Singapore Weekly Temperature Trend", xaxis_title="Week", yaxis_title="Temperature (C)", height=430, template="simple_white")
    return fig


def plot_factors_used_for_forecast(arimax_input_df: pd.DataFrame):
    work = arimax_input_df.dropna(subset=["avg_temp_c", "avg_relative_humidity_pct", "Total Dengue Cases"]).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work["avg_temp_c"],
            y=work["Total Dengue Cases"],
            mode="markers",
            marker=dict(size=8, color=work["avg_relative_humidity_pct"], colorscale="YlOrRd", showscale=True, colorbar=dict(title="Avg humidity"), line=dict(width=0.5, color="white")),
            text=work["week_start"].dt.strftime("%Y-%m-%d"),
            hovertemplate="Week=%{text}<br>Avg temp=%{x:.2f} C<br>Cases=%{y:.0f}<br>Humidity=%{marker.color:.1f}%<extra></extra>",
            name="Weekly observations",
        )
    )
    if len(work) >= 2 and work["avg_temp_c"].nunique() > 1:
        slope, intercept = np.polyfit(work["avg_temp_c"], work["Total Dengue Cases"], 1)
        x_line = np.linspace(work["avg_temp_c"].min(), work["avg_temp_c"].max(), 100)
        fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept, mode="lines", line=dict(color="#7A001F", dash="dash"), name="Linear fit"))
    fig.update_layout(title="Weather Signal Against Weekly Dengue Cases", xaxis_title="Average weekly temperature (C)", yaxis_title="Weekly dengue cases", height=430, template="simple_white")
    return fig


def plot_sarima_projection_summary(weekly_df: pd.DataFrame, stl_df: pd.DataFrame, forecast_df: pd.DataFrame, threshold: float):
    hist = weekly_df.tail(104).copy()
    stl_hist = stl_df.tail(len(hist)).copy()
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Historical structure", "SARIMA forecast"], horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=hist["week_start"], y=hist["Total Dengue Cases"], name="Observed", line=dict(color="#0B3C49")), row=1, col=1)
    fig.add_trace(go.Scatter(x=stl_hist["week_start"], y=stl_hist["trend"], name="STL trend", line=dict(color="#7A001F", width=2.5)), row=1, col=1)
    recent_hist = weekly_df.tail(52).copy()
    fig.add_trace(go.Scatter(x=recent_hist["week_start"], y=recent_hist["Total Dengue Cases"], name="Historical", line=dict(color="#0B3C49")), row=1, col=2)
    fig.add_trace(go.Scatter(x=forecast_df["week_start"], y=forecast_df["predicted_cases"], name="SARIMA forecast", line=dict(color="#7A001F", dash="dash"), mode="lines+markers"), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=list(forecast_df["week_start"]) + list(forecast_df["week_start"])[::-1],
            y=list(forecast_df["upper"]) + list(forecast_df["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(122,0,31,0.14)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Forecast interval",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=threshold, line_dash="dot", annotation_text=f"Threshold {threshold}", row=1, col=2)
    fig.update_layout(height=470, title="Projection of Dengue Cases using SARIMA", template="simple_white")
    return fig


def plot_arimax_projection_summary(weekly_df: pd.DataFrame, arimax_input_df: pd.DataFrame, arimax_future_exog_df: pd.DataFrame, arimax_fcst_df: pd.DataFrame, threshold: float):
    hist_exog = arimax_input_df.tail(26).copy()
    future_exog = arimax_future_exog_df.copy()
    means = arimax_input_df[ARIMAX_EXOG_COLUMNS].mean()
    stds = arimax_input_df[ARIMAX_EXOG_COLUMNS].std().replace(0, 1).fillna(1)
    for df in [hist_exog, future_exog]:
        for col in ARIMAX_EXOG_COLUMNS:
            df[col] = (pd.to_numeric(df[col], errors="coerce") - means[col]) / stds[col]

    label_map = {
        "avg_temp_c": "Temperature",
        "avg_relative_humidity_pct": "Humidity",
        "avg_wind_speed": "Wind",
        "warm_day_share": "Warm-day share",
    }
    color_map = {
        "avg_temp_c": "#E76F51",
        "avg_relative_humidity_pct": "#2A9D8F",
        "avg_wind_speed": "#577590",
        "warm_day_share": "#B56576",
    }
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Exogenous driver path (normalized)", "ARIMAX forecast"], horizontal_spacing=0.12)
    for col in ARIMAX_EXOG_COLUMNS:
        fig.add_trace(go.Scatter(x=hist_exog["week_start"], y=hist_exog[col], name=f"{label_map[col]} (hist)", line=dict(color=color_map[col], width=2), legendgroup=col), row=1, col=1)
        fig.add_trace(go.Scatter(x=future_exog["week_start"], y=future_exog[col], name=f"{label_map[col]} (future)", line=dict(color=color_map[col], width=2, dash="dash"), legendgroup=col), row=1, col=1)
    recent_hist = weekly_df.tail(52).copy()
    fig.add_trace(go.Scatter(x=recent_hist["week_start"], y=recent_hist["Total Dengue Cases"], name="Historical", line=dict(color="#0B3C49")), row=1, col=2)
    fig.add_trace(go.Scatter(x=arimax_fcst_df["week_start"], y=arimax_fcst_df["predicted_cases"], name="ARIMAX forecast", line=dict(color="#2A9D8F", dash="dash"), mode="lines+markers"), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=list(arimax_fcst_df["week_start"]) + list(arimax_fcst_df["week_start"])[::-1],
            y=list(arimax_fcst_df["upper"]) + list(arimax_fcst_df["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(42,157,143,0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Forecast interval",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=threshold, line_dash="dot", annotation_text=f"Threshold {threshold}", row=1, col=2)
    fig.update_layout(height=470, title="Projection of Dengue Cases using ARIMAX", template="simple_white")
    return fig


def save_figure(fig, filename: str):
    EXPORT_DIR.mkdir(exist_ok=True)
    out = EXPORT_DIR / filename
    try:
        fig.write_image(str(out), width=1600, height=900, scale=2)
    except Exception:
        fig.write_html(str(out.with_suffix(".html")))
    return out


def should_show_figures() -> bool:
    return "ipykernel" in sys.modules or os.getenv(SHOW_FIGURES_ENV, "").strip() == "1"


# %%
artifacts = load_analysis_artifacts(BASE_DIR)
weekly_df = artifacts["weekly_df"]
weekly_weather_df = artifacts["weekly_weather_df"]
lag_summary_df = artifacts["lag_summary_df"]
best_lags_df = artifacts["best_lags_df"]
lag_qa_df = artifacts["qa_df"]
forecast_df = artifacts["sarima_forecast_df"]
arimax_forecast_df = artifacts["arimax_forecast_df"]
arimax_future_exog_df = artifacts["arimax_future_exog_df"]
arimax_input_df = shared_build_arimax_inputs(weekly_df, weekly_weather_df)
stl_df, stl_diag_df, seasonal_strength, trend_slope = shared_stl_diagnostics(weekly_df)
seasonal_profile_df = shared_build_seasonal_profile(stl_df)


# %%
fig1 = plot_history(weekly_df)
fig2 = plot_stl(stl_df)
fig3 = plot_seasonal_profile(seasonal_profile_df)
fig4 = plot_temperature_history(weekly_weather_df)
fig5 = plot_factors_used_for_forecast(arimax_input_df)
fig6 = plot_sarima_projection_summary(weekly_df, stl_df, forecast_df, threshold=400)
fig7 = plot_arimax_projection_summary(weekly_df, arimax_input_df, arimax_future_exog_df, arimax_forecast_df, threshold=400)
fig8 = plot_lag_sweep_summary(lag_summary_df)
fig9 = plot_temperature_lag_focus(lag_summary_df)
fig10 = plot_lag_correlation_heatmap(lag_summary_df)

if should_show_figures():
    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()
    fig5.show()
    fig6.show()
    fig7.show()
    fig8.show()
    fig9.show()
    fig10.show()


# %%
# Optional exports for slides.
save_figure(fig1, "01_weekly_dengue_cases.png")
save_figure(fig2, "02_stl_decomposition.png")
save_figure(fig3, "03_seasonal_profile.png")
save_figure(fig4, "04_temperature_trend.png")
save_figure(fig5, "05_weather_signal.png")
save_figure(fig6, "06_sarima_projection.png")
save_figure(fig7, "07_arimax_projection.png")
save_figure(fig8, "08_weather_lag_sweep.png")
save_figure(fig9, "09_temperature_lag_focus.png")
save_figure(fig10, "10_weather_lag_heatmap.png")
