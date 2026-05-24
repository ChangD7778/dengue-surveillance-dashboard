import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from historical_lag_analysis import (
    ANALYSIS_WEATHER_END_DATE,
    LITERATURE_BENCHMARK_WEEKS,
    assign_risk_bands as shared_assign_risk_bands,
    build_arimax_inputs as shared_build_arimax_inputs,
    build_driver_metrics as shared_build_driver_metrics,
    build_monthly_case_seasonality as shared_build_monthly_case_seasonality,
    build_monthly_temperature_seasonality as shared_build_monthly_temperature_seasonality,
    build_seasonal_profile as shared_build_seasonal_profile,
    build_weekly_weather_features as shared_build_weekly_weather_features,
    fit_arimax_and_forecast as shared_fit_arimax_and_forecast,
    fit_fixed_sarima_and_forecast as shared_fit_fixed_sarima_and_forecast,
    load_analysis_artifacts,
    load_historical_weather_data,
    prepare_daily_weather_features,
    plot_lag_correlation_heatmap,
    plot_lag_sweep_summary,
    plot_weekly_lag_correlation_heatmap,
    plot_weekly_temperature_lag_focus,
    stl_diagnostics as shared_stl_diagnostics,
    validate_weather_overlap as shared_validate_weather_overlap,
)

st.set_page_config(page_title="NEA Dengue Surveillance and Early Warning Dashboard", layout="wide")

DATASET_ID = "d_ca168b2cb763640d72c4600a68f9909e"
DATASTORE_URL = "https://data.gov.sg/api/action/datastore_search"
WEATHER_FORECAST_URL = "https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast"
BASE_DIR = Path(__file__).resolve().parent
CACHE_FILE = BASE_DIR / "singapore_dengue_raw_records.csv"
WEATHER_CACHE_FILE = BASE_DIR / "singapore_weather_forecast_24hr_history.csv"
WEATHER_HISTORY_START_DATE = pd.Timestamp("2016-03-01")
WEATHER_REGIONS = ("west", "east", "central", "north", "south")
DEFAULT_WEATHER_LAG_WEEKS = 18
MIN_OVERLAP_WEEKS = 26
ARIMAX_EXOG_COLUMNS = [
    "avg_temp_c",
    "avg_relative_humidity_pct",
    "avg_wind_speed",
    "warm_day_share",
]

SEASONAL_PERIOD = 52
FIXED_ORDER = (3, 1, 0)
FIXED_SEASONAL_ORDER = (0, 1, 0, SEASONAL_PERIOD)
FORECAST_ALPHA = 0.2  # 80% interval
RISK_Z = 1.2816

WATCH_MONTHS = {4, 5, 6, 7}  # Apr-Jul


def _headers() -> dict:
    api_key = os.getenv("DATA_GOV_SG_API_KEY", "").strip()
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def request_json(url: str, params: dict | None = None, max_attempts: int = 10, timeout: int = 30) -> dict:
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=timeout)
            if resp.status_code == 429:
                if attempt == max_attempts:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After", "").strip()
                wait_s = float(retry_after) if retry_after.isdigit() else min(2 ** attempt, 120)
                time.sleep(wait_s)
                continue
            if resp.status_code in (500, 502, 503, 504):
                if attempt == max_attempts:
                    resp.raise_for_status()
                time.sleep(min(2 ** attempt, 60))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_err = exc
            if attempt == max_attempts:
                raise
            time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"Request failed: {last_err}")


def request_json_allow_404(url: str, params: dict | None = None, max_attempts: int = 10, timeout: int = 30) -> dict | None:
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=timeout)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                if attempt == max_attempts:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After", "").strip()
                wait_s = float(retry_after) if retry_after.isdigit() else min(2 ** attempt, 120)
                time.sleep(wait_s)
                continue
            if resp.status_code in (500, 502, 503, 504):
                if attempt == max_attempts:
                    resp.raise_for_status()
                time.sleep(min(2 ** attempt, 60))
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
            time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"Request failed: {last_err}")


def fetch_all_records(resource_id: str) -> list[dict]:
    payload = request_json(DATASTORE_URL, params={"resource_id": resource_id, "limit": 50000, "offset": 0})
    if not payload.get("success"):
        raise RuntimeError("data.gov.sg datastore_search returned success=false")

    result = payload.get("result", {})
    total = int(result.get("total", 0))
    records = list(result.get("records", []))
    if len(records) >= total:
        return records

    offset = len(records)
    page_size = 5000
    while offset < total:
        payload = request_json(DATASTORE_URL, params={"resource_id": resource_id, "limit": page_size, "offset": offset})
        if not payload.get("success"):
            raise RuntimeError("Paged datastore_search returned success=false")
        page = payload.get("result", {}).get("records", [])
        if not page:
            break
        records.extend(page)
        offset += len(page)
        time.sleep(0.5)

    if not records:
        raise RuntimeError("No records returned from datastore_search")
    return records


def parse_epi_week_to_sunday(epi_week: str):
    match = re.fullmatch(r"(\d{4})-W(\d{1,2})", str(epi_week).strip())
    if not match:
        return pd.NaT
    year = int(match.group(1))
    week = int(match.group(2))
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    first_week_sunday = jan1 - pd.Timedelta(days=(jan1.weekday() + 1) % 7)
    return first_week_sunday + pd.Timedelta(weeks=week - 1)


def load_records() -> pd.DataFrame:
    if not CACHE_FILE.exists():
        records = fetch_all_records(DATASET_ID)
        raw_df = pd.DataFrame(records)
        raw_df.to_csv(CACHE_FILE, index=False)

    return pd.read_csv(CACHE_FILE)


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
        "temperature_unit": temperature.get("unit"),
        "relative_humidity_low_pct": relative_humidity.get("low"),
        "relative_humidity_high_pct": relative_humidity.get("high"),
        "relative_humidity_unit": relative_humidity.get("unit"),
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


def fetch_weather_records_for_day(query_date: pd.Timestamp) -> list[dict]:
    date_str = pd.Timestamp(query_date).strftime("%Y-%m-%d")
    params = {"date": date_str}
    records: list[dict] = []

    while True:
        payload = request_json_allow_404(WEATHER_FORECAST_URL, params=params)
        if not payload:
            break

        data = payload.get("data", {}) or {}
        batch = data.get("records", []) or []
        for record in batch:
            enriched = dict(record)
            enriched["query_date"] = date_str
            records.append(enriched)

        pagination_token = data.get("paginationToken")
        if not pagination_token:
            break

        params = {"date": date_str, "paginationToken": pagination_token}
        time.sleep(0.2)

    return records


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

    combined_df.to_csv(WEATHER_CACHE_FILE, index=False)
    return combined_df


def load_weather_forecast_history() -> pd.DataFrame:
    return load_historical_weather_data(
        WEATHER_CACHE_FILE,
        start_date=WEATHER_HISTORY_START_DATE,
        end_date=ANALYSIS_WEATHER_END_DATE,
    )


@st.cache_data(show_spinner=False)
def load_precomputed_analysis_artifacts() -> dict:
    return load_analysis_artifacts(BASE_DIR)


def to_week_start_sunday(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, errors="coerce")
    day_offset = (ts.dt.dayofweek + 1) % 7
    return (ts - pd.to_timedelta(day_offset, unit="D")).dt.normalize()


@st.cache_data(show_spinner=False)
def build_weekly_weather_features(weather_df: pd.DataFrame, lag_days: int = DEFAULT_WEATHER_LAG_WEEKS * 7) -> pd.DataFrame:
    if weather_df.empty:
        raise RuntimeError("Weather cache is empty")

    work = weather_df.copy()
    work["query_date"] = pd.to_datetime(work["query_date"], errors="coerce")
    work = work.dropna(subset=["query_date"]).sort_values("query_date").reset_index(drop=True)
    if work.empty:
        raise RuntimeError("Weather cache does not contain valid query_date values")

    for col in [
        "temperature_low_c",
        "temperature_high_c",
        "relative_humidity_low_pct",
        "relative_humidity_high_pct",
        "wind_speed_low",
        "wind_speed_high",
    ]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["avg_temp_c"] = work[["temperature_low_c", "temperature_high_c"]].mean(axis=1)
    work["avg_relative_humidity_pct"] = work[["relative_humidity_low_pct", "relative_humidity_high_pct"]].mean(axis=1)
    work["avg_wind_speed"] = work[["wind_speed_low", "wind_speed_high"]].mean(axis=1)
    work["warm_day_flag"] = (work["temperature_high_c"] >= 32).astype(float)
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


@st.cache_data(show_spinner=False)
def build_arimax_inputs(weekly_df: pd.DataFrame, weekly_weather_df: pd.DataFrame) -> pd.DataFrame:
    merged = weekly_df.merge(weekly_weather_df, on="week_start", how="left")
    merged = merged.sort_values("week_start").reset_index(drop=True)

    for col in ARIMAX_EXOG_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[col] = merged[col].interpolate(limit_direction="both")
        if merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].median())

    return merged


def validate_weather_overlap(weekly_df: pd.DataFrame, weekly_weather_df: pd.DataFrame, lag_days: int) -> None:
    overlap = weekly_df[["week_start"]].merge(weekly_weather_df[["week_start"]], on="week_start", how="inner")
    if len(overlap) < MIN_OVERLAP_WEEKS:
        dengue_min = pd.to_datetime(weekly_df["week_start"], errors="coerce").min()
        dengue_max = pd.to_datetime(weekly_df["week_start"], errors="coerce").max()
        weather_min = pd.to_datetime(weekly_weather_df["week_start"], errors="coerce").min()
        weather_max = pd.to_datetime(weekly_weather_df["week_start"], errors="coerce").max()
        raise RuntimeError(
            "Weather cache does not overlap enough with the dengue history for a valid ARIMAX fit. "
            f"Dengue weeks: {dengue_min.date()} to {dengue_max.date()}. "
            f"Weather weeks after applying lag={lag_days}: {weather_min.date()} to {weather_max.date()}. "
            f"Overlap weeks found: {len(overlap)}."
        )


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


@st.cache_data(show_spinner=False)
def fit_arimax_and_forecast(arimax_df: pd.DataFrame, weekly_weather_df: pd.DataFrame, steps: int, alpha: float = FORECAST_ALPHA):
    model_df = arimax_df.dropna(subset=["Total Dengue Cases"]).copy()
    y = model_df["Total Dengue Cases"].astype(float)
    exog = model_df[ARIMAX_EXOG_COLUMNS].astype(float)

    sm_model = SARIMAX(
        y,
        exog=exog,
        order=FIXED_ORDER,
        seasonal_order=FIXED_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(method="lbfgs", maxiter=200, disp=False)

    last_date = model_df["week_start"].iloc[-1]
    idx = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=steps, freq="W-SUN")
    future_exog = build_future_weather_exog(weekly_weather_df, idx)

    forecast = sm_model.get_forecast(steps=steps, exog=future_exog[ARIMAX_EXOG_COLUMNS].astype(float))
    conf = forecast.conf_int(alpha=alpha)

    fcst_df = pd.DataFrame(
        {
            "week_start": idx,
            "predicted_cases": forecast.predicted_mean.values,
            "lower": conf.iloc[:, 0].values,
            "upper": conf.iloc[:, 1].values,
        }
    )

    return fcst_df, future_exog


@st.cache_data(show_spinner=False)
def build_weekly_dengue_series(records_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = {"epi_week", "disease", "no._of_cases"}
    missing = required_cols.difference(records_df.columns)
    if missing:
        raise RuntimeError(f"Missing expected columns: {sorted(missing)}")

    work = records_df.copy()
    work["no._of_cases"] = pd.to_numeric(work["no._of_cases"], errors="coerce").fillna(0)
    work = work[work["disease"].isin(["Dengue Fever", "Dengue Haemorrhagic Fever"])].copy()
    if work.empty:
        raise RuntimeError("No dengue rows found in source data")

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

    checks = []
    checks.append({"check": "non_empty", "value": int(len(weekly)), "status": "PASS" if len(weekly) > 0 else "FAIL"})
    checks.append({"check": "duplicate_week_start", "value": int(weekly["week_start"].duplicated().sum()), "status": "PASS"})
    checks.append({"check": "negative_cases", "value": int((weekly["Total Dengue Cases"] < 0).sum()), "status": "PASS"})

    return weekly[["week_start", "Dengue Fever", "Dengue Haemorrhagic Fever", "Total Dengue Cases", "moving_avg_12w"]], pd.DataFrame(checks)


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def fit_fixed_sarima_and_forecast(weekly_df: pd.DataFrame, steps: int, alpha: float = FORECAST_ALPHA):
    y = weekly_df["Total Dengue Cases"].astype(float)
    sm_model = SARIMAX(
        y,
        order=FIXED_ORDER,
        seasonal_order=FIXED_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(method="lbfgs", maxiter=200, disp=False)

    forecast = sm_model.get_forecast(steps=steps)
    conf = forecast.conf_int(alpha=alpha)

    last_date = weekly_df["week_start"].iloc[-1]
    idx = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=steps, freq="W-SUN")

    fcst_df = pd.DataFrame(
        {
            "week_start": idx,
            "predicted_cases": forecast.predicted_mean.values,
            "lower": conf.iloc[:, 0].values,
            "upper": conf.iloc[:, 1].values,
        }
    )
    return fcst_df


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return norm.cdf(x)


@st.cache_data(show_spinner=False)
def assign_risk_bands(fcst_df: pd.DataFrame, threshold: float, low_cut: float, high_cut: float) -> pd.DataFrame:
    out = fcst_df.copy()
    sigma = (out["upper"] - out["lower"]).abs() / (2 * RISK_Z)
    sigma = sigma.replace(0, np.nan).fillna(1.0)
    z = (threshold - out["predicted_cases"]) / sigma
    p_exceed = 1 - normal_cdf(z.to_numpy())
    out["p_exceed_threshold"] = np.clip(p_exceed, 0.0, 1.0)

    out["risk_band"] = np.select(
        [out["p_exceed_threshold"] < low_cut, out["p_exceed_threshold"] < high_cut],
        ["Low", "Medium"],
        default="High",
    )
    return out


def surveillance_recommendation(risk_df: pd.DataFrame, trend_slope: float, horizon_mode: int, watch_enabled: bool, threshold: float) -> dict:
    decision_window = risk_df.head(min(4, horizon_mode)).copy()
    high_mask = decision_window["risk_band"].eq("High")

    month_now = pd.Timestamp.today().month
    in_watch_window = month_now in WATCH_MONTHS if watch_enabled else False

    baseline = bool(high_mask.any() and trend_slope > 0)
    medium_or_high = decision_window["risk_band"].isin(["Medium", "High"]).any()
    adjusted = bool(baseline or (in_watch_window and medium_or_high and trend_slope > 0))

    if adjusted and not decision_window.empty:
        week = decision_window[decision_window["risk_band"].isin(["High", "Medium"])]["week_start"].iloc[0]
        conf = float(decision_window["p_exceed_threshold"].max())
        reason = "Enhanced surveillance: elevated risk signal + positive trend"
        action = "Enhanced Surveillance"
    else:
        week = pd.NaT
        conf = float(decision_window["p_exceed_threshold"].max()) if not decision_window.empty else 0.0
        reason = "Routine surveillance: no qualifying trigger in the next 2-4 weeks"
        action = "Routine Surveillance"

    return {
        "action": action,
        "recommended_week": week,
        "confidence": conf,
        "reason": reason,
        "current_risk": decision_window["risk_band"].iloc[0] if not decision_window.empty else "Low",
        "in_watch_window": in_watch_window,
        "threshold": threshold,
    }


RISK_BAND_FILL_COLORS = {
    "Low": "rgba(0,180,0,0.10)",
    "Medium": "rgba(255,165,0,0.12)",
    "High": "rgba(255,0,0,0.12)",
}


def _overlay_risk_bands(fig, risk_df: pd.DataFrame, row: int | None = None, col: int | None = None) -> None:
    if risk_df.empty or "risk_band" not in risk_df.columns:
        return

    add_kwargs = {}
    if row is not None and col is not None:
        add_kwargs = {"row": row, "col": col}

    for _, r in risk_df.iterrows():
        band = r.get("risk_band", "Low")
        fillcolor = RISK_BAND_FILL_COLORS.get(band, RISK_BAND_FILL_COLORS["Low"])
        fig.add_vrect(
            x0=r["week_start"] - pd.Timedelta(days=3),
            x1=r["week_start"] + pd.Timedelta(days=3),
            fillcolor=fillcolor,
            line_width=0,
            layer="below",
            **add_kwargs,
        )


def plot_history(weekly_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_df["week_start"], y=weekly_df["Total Dengue Cases"], name="Weekly cases", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=weekly_df["week_start"], y=weekly_df["moving_avg_12w"], name="12-week MA", line=dict(width=2.8)))
    fig.update_layout(title="Historical Dengue Cases", xaxis_title="Week", yaxis_title="Cases", height=420)
    return fig


def plot_stl(stl_df: pd.DataFrame):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
    fig.add_trace(go.Scatter(x=stl_df["week_start"], y=stl_df["observed"], name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=stl_df["week_start"], y=stl_df["trend"], name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=stl_df["week_start"], y=stl_df["seasonal"], name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=stl_df["week_start"], y=stl_df["resid"], name="Residual"), row=4, col=1)
    fig.update_layout(height=760, title="STL Decomposition")
    return fig


def plot_future_forecast(weekly_df: pd.DataFrame, risk_df: pd.DataFrame, threshold: float):
    hist = weekly_df.tail(52).copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["week_start"], y=hist["Total Dengue Cases"], name="Actual (last 52 weeks)", line=dict(color="royalblue")))
    fig.add_trace(go.Scatter(x=risk_df["week_start"], y=risk_df["predicted_cases"], name="Forecast", line=dict(color="firebrick", dash="dash"), mode="lines+markers"))
    fig.add_trace(
        go.Scatter(
            x=list(risk_df["week_start"]) + list(risk_df["week_start"])[::-1],
            y=list(risk_df["upper"]) + list(risk_df["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(255,0,0,0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Forecast interval (80%)",
        )
    )
    fig.add_hline(y=threshold, line_dash="dot", annotation_text=f"Threshold {threshold}")

    _overlay_risk_bands(fig, risk_df)
    fig.update_layout(title="SARIMA Forecast (last 1 year context)", xaxis_title="Week", yaxis_title="Cases", height=480)
    return fig


def build_seasonal_profile(stl_df: pd.DataFrame) -> pd.DataFrame:
    tmp = stl_df.copy()
    tmp["month"] = pd.to_datetime(tmp["week_start"]).dt.month
    prof = tmp.groupby("month", as_index=False)["seasonal"].mean()
    prof["month_name"] = pd.to_datetime(prof["month"], format="%m").dt.strftime("%b")
    prof["is_peak_window"] = prof["month"].isin([5, 6])
    return prof


def plot_seasonal_profile(profile_df: pd.DataFrame):
    colors = np.where(profile_df["is_peak_window"], "crimson", "steelblue")
    fig = go.Figure(go.Bar(x=profile_df["month_name"], y=profile_df["seasonal"], marker_color=colors, name="Avg STL seasonal"))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title="Month-of-Year Seasonal Profile (STL)", xaxis_title="Month", yaxis_title="Average seasonal component", height=380)
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
    fig.add_trace(
        go.Scatter(
            x=hist["week_start"],
            y=hist["avg_temp_c"],
            name="Weekly average temperature",
            line=dict(color="darkorange", width=2.5),
        )
    )
    fig.update_layout(
        title="Singapore Weekly Temperature Trend",
        xaxis_title="Week",
        yaxis_title="Temperature (C)",
        height=420,
    )
    return fig


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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
        .rename(columns={"avg_temp_c": "month_avg_temp_c", "min_temp_c": "month_min_temp_c", "max_temp_c": "month_max_temp_c"})
        .sort_values("month")
        .reset_index(drop=True)
    )
    return monthly, month_range


@st.cache_data(show_spinner=False)
def build_driver_metrics(arimax_input_df: pd.DataFrame) -> pd.DataFrame:
    work = arimax_input_df.dropna(subset=["Total Dengue Cases"] + ARIMAX_EXOG_COLUMNS).copy()
    metrics = []
    label_map = {
        "avg_temp_c": "Temperature correlation",
        "avg_relative_humidity_pct": "Humidity correlation",
        "avg_wind_speed": "Wind correlation",
        "warm_day_share": "Warm-day share correlation",
    }
    for col in ARIMAX_EXOG_COLUMNS:
        corr = work["Total Dengue Cases"].corr(work[col])
        metrics.append({"metric": label_map[col], "value": round(float(corr), 3) if pd.notna(corr) else np.nan})
    return pd.DataFrame(metrics)


def plot_factors_used_for_forecast(arimax_input_df: pd.DataFrame):
    work = arimax_input_df.dropna(subset=["avg_temp_c", "avg_relative_humidity_pct", "Total Dengue Cases"]).copy()
    work["year"] = pd.to_datetime(work["week_start"]).dt.year.astype(str)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work["avg_temp_c"],
            y=work["Total Dengue Cases"],
            mode="markers",
            marker=dict(
                size=8,
                color=work["avg_relative_humidity_pct"],
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(title="Avg humidity"),
                line=dict(width=0.5, color="white"),
            ),
            text=work["week_start"].dt.strftime("%Y-%m-%d"),
            hovertemplate="Week=%{text}<br>Avg temp=%{x:.2f} C<br>Cases=%{y:.0f}<br>Humidity=%{marker.color:.1f}%<extra></extra>",
            name="Weekly observations",
        )
    )

    if len(work) >= 2 and work["avg_temp_c"].nunique() > 1:
        slope, intercept = np.polyfit(work["avg_temp_c"], work["Total Dengue Cases"], 1)
        x_line = np.linspace(work["avg_temp_c"].min(), work["avg_temp_c"].max(), 100)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="firebrick", dash="dash"),
                name="Linear fit",
            )
        )
        corr = work["avg_temp_c"].corr(work["Total Dengue Cases"])
        fig.add_annotation(
            x=0.98,
            y=0.03,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            align="right",
            bgcolor="rgba(255,255,255,0.85)",
            text=f"Temp vs dengue correlation: {corr:.2f}<br>Color indicates humidity",
        )

    fig.update_layout(
        title="Weather signal against weekly dengue cases",
        xaxis_title="Average weekly temperature (C)",
        yaxis_title="Weekly dengue cases",
        height=420,
    )
    return fig


def plot_dengue_seasonality(monthly_case_df: pd.DataFrame, case_range_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(case_range_df["month"]) + list(case_range_df["month"])[::-1],
            y=list(case_range_df["month_max"]) + list(case_range_df["month_min"])[::-1],
            fill="toself",
            fillcolor="rgba(220, 20, 60, 0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Historical range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=case_range_df["month"],
            y=case_range_df["month_avg"],
            mode="lines",
            line=dict(color="indianred", dash="dot"),
            name="Monthly average",
        )
    )

    recent_years = sorted(monthly_case_df["year"].unique())[-4:]
    palette = ["#0B3C49", "#7A001F", "#B56576", "#E76F51"]
    for color, year in zip(palette, recent_years):
        tmp = monthly_case_df[monthly_case_df["year"] == year]
        fig.add_trace(
            go.Scatter(
                x=tmp["month"],
                y=tmp["avg_cases"],
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                name=str(year),
            )
        )

    fig.update_layout(
        title="Dengue seasonality across recent years",
        xaxis_title="Month",
        yaxis_title="Average weekly dengue cases",
        xaxis=dict(tickmode="array", tickvals=list(range(1, 13)), ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]),
        height=430,
    )
    return fig


def plot_temperature_seasonality(monthly_temp_df: pd.DataFrame, temp_range_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(temp_range_df["month"]) + list(temp_range_df["month"])[::-1],
            y=list(temp_range_df["month_max_temp_c"]) + list(temp_range_df["month_min_temp_c"])[::-1],
            fill="toself",
            fillcolor="rgba(255, 140, 0, 0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Historical range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=temp_range_df["month"],
            y=temp_range_df["month_avg_temp_c"],
            mode="lines",
            line=dict(color="darkorange", dash="dot"),
            name="Monthly average",
        )
    )

    recent_years = sorted(monthly_temp_df["year"].unique())[-4:]
    palette = ["#355070", "#6D597A", "#B56576", "#E56B6F"]
    for color, year in zip(palette, recent_years):
        tmp = monthly_temp_df[monthly_temp_df["year"] == year]
        fig.add_trace(
            go.Scatter(
                x=tmp["month"],
                y=tmp["avg_temp_c"],
                mode="lines+markers",
                line=dict(color=color, width=2.3),
                name=str(year),
            )
        )

    fig.update_layout(
        title="Singapore temperature seasonality",
        xaxis_title="Month",
        yaxis_title="Average temperature (C)",
        xaxis=dict(tickmode="array", tickvals=list(range(1, 13)), ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]),
        height=430,
    )
    return fig


def plot_sarima_projection_summary(weekly_df: pd.DataFrame, stl_df: pd.DataFrame, forecast_df: pd.DataFrame, threshold: float):
    hist = weekly_df.tail(104).copy()
    stl_hist = stl_df.tail(len(hist)).copy()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Historical structure", "SARIMA forecast"],
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Scatter(x=hist["week_start"], y=hist["Total Dengue Cases"], name="Observed", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=stl_hist["week_start"], y=stl_hist["trend"], name="STL trend", line=dict(color="firebrick", width=2.5)), row=1, col=1)

    recent_hist = weekly_df.tail(52).copy()
    fig.add_trace(go.Scatter(x=recent_hist["week_start"], y=recent_hist["Total Dengue Cases"], name="Historical", line=dict(color="royalblue")), row=1, col=2)
    fig.add_trace(go.Scatter(x=forecast_df["week_start"], y=forecast_df["predicted_cases"], name="SARIMA forecast", line=dict(color="firebrick", dash="dash"), mode="lines+markers"), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=list(forecast_df["week_start"]) + list(forecast_df["week_start"])[::-1],
            y=list(forecast_df["upper"]) + list(forecast_df["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(220,20,60,0.14)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Forecast interval",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=threshold, line_dash="dot", annotation_text=f"Threshold {threshold}", row=1, col=2)
    fig.update_layout(height=460, title="Projection of Dengue Cases using SARIMA")
    fig.update_xaxes(title_text="Week", row=1, col=1)
    fig.update_xaxes(title_text="Week", row=1, col=2)
    fig.update_yaxes(title_text="Cases", row=1, col=1)
    fig.update_yaxes(title_text="Cases", row=1, col=2)
    return fig


def plot_arimax_projection_summary(
    weekly_df: pd.DataFrame,
    arimax_input_df: pd.DataFrame,
    arimax_future_exog_df: pd.DataFrame,
    arimax_risk_df: pd.DataFrame,
    threshold: float,
):
    hist_exog = arimax_input_df.tail(26).copy()
    future_exog = arimax_future_exog_df.copy()

    norm_source = arimax_input_df[ARIMAX_EXOG_COLUMNS].copy()
    means = norm_source.mean()
    stds = norm_source.std().replace(0, 1).fillna(1)

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

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Exogenous driver path (standardized)", "SARIMAX forecast"],
        horizontal_spacing=0.12,
    )
    for col in ARIMAX_EXOG_COLUMNS:
        fig.add_trace(
            go.Scatter(
                x=hist_exog["week_start"],
                y=hist_exog[col],
                name=f"{label_map[col]} (hist)",
                line=dict(color=color_map[col], width=2),
                legendgroup=col,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=future_exog["week_start"],
                y=future_exog[col],
                name=f"{label_map[col]} (future)",
                line=dict(color=color_map[col], width=2, dash="dash"),
                legendgroup=col,
            ),
            row=1,
            col=1,
        )

    recent_hist = weekly_df.tail(52).copy()
    fig.add_trace(go.Scatter(x=recent_hist["week_start"], y=recent_hist["Total Dengue Cases"], name="Historical", line=dict(color="royalblue")), row=1, col=2)
    fig.add_trace(go.Scatter(x=arimax_risk_df["week_start"], y=arimax_risk_df["predicted_cases"], name="SARIMAX forecast", line=dict(color="seagreen", dash="dash"), mode="lines+markers"), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=list(arimax_risk_df["week_start"]) + list(arimax_risk_df["week_start"])[::-1],
            y=list(arimax_risk_df["upper"]) + list(arimax_risk_df["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(46,139,87,0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Forecast interval",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=threshold, line_dash="dot", annotation_text=f"Threshold {threshold}", row=1, col=2)
    _overlay_risk_bands(fig, arimax_risk_df, row=1, col=2)
    fig.update_layout(height=470, title="Projection of Dengue Cases using SARIMAX")
    fig.update_xaxes(title_text="Week", row=1, col=1)
    fig.update_xaxes(title_text="Week", row=1, col=2)
    fig.update_yaxes(title_text="Standard deviations from historical mean", row=1, col=1)
    fig.update_yaxes(title_text="Cases", row=1, col=2)
    return fig


def plot_arimax_forecast(weekly_df: pd.DataFrame, arimax_risk_df: pd.DataFrame, threshold: float):
    hist = weekly_df.tail(52).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["week_start"],
            y=hist["Total Dengue Cases"],
            name="Actual (last 52 weeks)",
            line=dict(color="royalblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=arimax_risk_df["week_start"],
            y=arimax_risk_df["predicted_cases"],
            name="SARIMAX forecast",
            line=dict(color="seagreen", dash="dash"),
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(arimax_risk_df["week_start"]) + list(arimax_risk_df["week_start"])[::-1],
            y=list(arimax_risk_df["upper"]) + list(arimax_risk_df["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(46,139,87,0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Forecast interval (80%)",
        )
    )
    fig.add_hline(y=threshold, line_dash="dot", annotation_text=f"Threshold {threshold}")
    _overlay_risk_bands(fig, arimax_risk_df)
    fig.update_layout(title="SARIMAX Forecast with Weather Exogenous Inputs", xaxis_title="Week", yaxis_title="Cases", height=480)
    return fig


def _plot_backtest_skill_by_horizon(horizon_metrics_df: pd.DataFrame, dm_df: pd.DataFrame):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Forecast error by horizon (sMAPE)", "PI coverage by horizon (target 0.80)"],
        horizontal_spacing=0.12,
    )
    color_map = {"sarima": "firebrick", "sarimax": "seagreen"}
    label_map = {"sarima": "SARIMA", "sarimax": "SARIMAX"}

    for kind in ["sarima", "sarimax"]:
        sub = horizon_metrics_df[horizon_metrics_df["model_kind"] == kind].sort_values("h")
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["h"], y=sub["smape"], name=label_map[kind],
                mode="lines+markers", line=dict(color=color_map[kind], width=2.5),
                legendgroup=kind,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["h"], y=sub["pi_coverage"], name=label_map[kind],
                mode="lines+markers", line=dict(color=color_map[kind], width=2.5),
                legendgroup=kind, showlegend=False,
            ),
            row=1, col=2,
        )

    if not dm_df.empty:
        sig = dm_df[dm_df["p_value"] < 0.05]
        if not sig.empty:
            sm = horizon_metrics_df.pivot(index="h", columns="model_kind", values="smape")
            for _, r in sig.iterrows():
                h_val = int(r["h"])
                if h_val in sm.index:
                    y_marker = float(sm.loc[h_val].min()) * 0.92
                    fig.add_annotation(
                        x=h_val, y=y_marker, text="*", showarrow=False,
                        font=dict(size=18, color="black"), row=1, col=1,
                    )

    fig.add_hline(y=0.80, line_dash="dash", line_color="gray", row=1, col=2)
    fig.update_xaxes(title_text="Weeks ahead", row=1, col=1, dtick=1)
    fig.update_xaxes(title_text="Weeks ahead", row=1, col=2, dtick=1)
    fig.update_yaxes(title_text="sMAPE", row=1, col=1)
    fig.update_yaxes(title_text="Coverage", range=[0.5, 1.0], row=1, col=2)
    fig.update_layout(height=420, title="Walk-forward forecast skill")
    return fig


def _render_backtest_panel(
    per_origin_df: pd.DataFrame,
    horizon_metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    dm_df: pd.DataFrame,
    meta: dict,
    threshold: int,
    weekly_df: pd.DataFrame,
) -> None:
    st.caption(
        "Walk-forward expanding-window backtest. At each origin, the model is refit on all "
        "available data and forecasts 12 weeks ahead. Origins spaced 4 weeks apart, minimum "
        "3 years of training. Backtest uses default 18-week weather lag — sidebar controls "
        "do not affect this section."
    )
    st.markdown(
        f"**Origins evaluated:** {meta.get('n_origins', 'n/a')} from "
        f"{meta.get('origin_first', 'n/a')} to {meta.get('origin_last', 'n/a')}. "
        f"**Generated:** {meta.get('generated_at_utc', 'n/a')}"
    )

    # --- Train / last-window / test split chart ---
    st.markdown("### Most recent backtest split: train / last window / test")
    if not weekly_df.empty and not per_origin_df.empty:
        _wdf = weekly_df.copy()
        _wdf["week_start"] = pd.to_datetime(_wdf["week_start"])
        _wdf = _wdf.sort_values("week_start").reset_index(drop=True)

        _last_origin = pd.to_datetime(per_origin_df["origin_week"].max())

        _CONTEXT_WEEKS = 78  # ~1.5 years for top panel context
        _LAST_WINDOW_WEEKS = 16

        _train_all = _wdf[_wdf["week_start"] <= _last_origin]
        _train_cropped = _train_all.tail(_CONTEXT_WEEKS)
        if len(_train_cropped) >= _LAST_WINDOW_WEEKS:
            _far_train = _train_cropped.iloc[:-_LAST_WINDOW_WEEKS]
            _last_window = _train_cropped.iloc[-_LAST_WINDOW_WEEKS:]
        else:
            _far_train = pd.DataFrame()
            _last_window = _train_cropped

        _per_origin_copy = per_origin_df.copy()
        _per_origin_copy["origin_week"] = pd.to_datetime(_per_origin_copy["origin_week"])
        _test_rows = _per_origin_copy[_per_origin_copy["origin_week"] == _last_origin].copy()

        _sarima_test = _test_rows[_test_rows["model_kind"] == "sarima"].sort_values("h")
        _sarimax_test = _test_rows[_test_rows["model_kind"] == "sarimax"].sort_values("h")

        _forecast_end = _last_origin + pd.Timedelta(weeks=12)
        _future_dates = (
            [_last_origin + pd.Timedelta(weeks=int(h)) for h in _sarima_test["h"]]
            if not _sarima_test.empty else []
        )
        _future_dates_sx = (
            [_last_origin + pd.Timedelta(weeks=int(h)) for h in _sarimax_test["h"]]
            if not _sarimax_test.empty else []
        )

        _train = _train_all.iloc[:-_LAST_WINDOW_WEEKS].copy() if len(_train_all) > _LAST_WINDOW_WEEKS else pd.DataFrame()
        _last_window = _train_all.iloc[-_LAST_WINDOW_WEEKS:].copy()

        if not _sarima_test.empty and "target_week" in _sarima_test.columns:
            _test_dates = pd.to_datetime(_sarima_test["target_week"], errors="coerce").dropna().sort_values().unique()
            _test = _wdf[_wdf["week_start"].isin(_test_dates)].copy()
        else:
            _test = _wdf[(_wdf["week_start"] > _last_origin) & (_wdf["week_start"] <= _forecast_end)].copy()

        def _date_range_text(df: pd.DataFrame) -> str:
            if df.empty:
                return "n/a --- n/a  (n=0)"
            return f"{df['week_start'].min()} --- {df['week_start'].max()}  (n={len(df)})"

        st.text(
            "Train dates       : "
            f"{_date_range_text(_train)}\n"
            "Last window dates : "
            f"{_date_range_text(_last_window)}\n"
            "Test dates        : "
            f"{_date_range_text(_test)}"
        )

        fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
        if not _train.empty:
            _train.set_index("week_start")["Total Dengue Cases"].plot(ax=ax, label="train")
        if not _last_window.empty:
            _last_window.set_index("week_start")["Total Dengue Cases"].plot(ax=ax, label="last window")
        if not _test.empty:
            _test.set_index("week_start")["Total Dengue Cases"].plot(ax=ax, label="test")

        ax.set_xlabel("week_start")
        ax.set_ylabel("Total Dengue Cases")
        ax.legend(fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        _unused_old_backtest_plot = """
        # Build two-row figure: top = full context, bottom = forecast zoom
        _fig_ts = make_subplots(
            rows=2, cols=1,
            shared_xaxes=False,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.14,
            subplot_titles=[
                "Historical context (full y-scale)",
                "Forecast region — actuals vs model predictions (zoomed)",
            ],
        )

        # ===== TOP PANEL: full context, native y-scale =====
        if not _far_train.empty:
            _fig_ts.add_trace(go.Scatter(
                x=_far_train["week_start"],
                y=_far_train["Total Dengue Cases"],
                mode="lines",
                line=dict(color="steelblue", width=1.5),
                name="Training history",
                legendgroup="train",
            ), row=1, col=1)

        if not _last_window.empty:
            _fig_ts.add_trace(go.Scatter(
                x=_last_window["week_start"],
                y=_last_window["Total Dengue Cases"],
                mode="lines",
                line=dict(color="darkorange", width=2),
                name="Last 16-week window",
                legendgroup="last_window",
            ), row=1, col=1)

        if not _sarima_test.empty:
            _fig_ts.add_trace(go.Scatter(
                x=_future_dates,
                y=_sarima_test["actual"],
                mode="lines+markers",
                line=dict(color="black", width=2.5),
                marker=dict(size=6, color="black"),
                name="Actual (test)",
                legendgroup="actual",
            ), row=1, col=1)
            _fig_ts.add_trace(go.Scatter(
                x=_future_dates,
                y=_sarima_test["predicted"],
                mode="lines+markers",
                line=dict(color="firebrick", width=2, dash="dash"),
                marker=dict(size=6, color="firebrick", symbol="diamond"),
                name="SARIMA",
                legendgroup="sarima",
            ), row=1, col=1)

        if not _sarimax_test.empty:
            _fig_ts.add_trace(go.Scatter(
                x=_future_dates_sx,
                y=_sarimax_test["predicted"],
                mode="lines+markers",
                line=dict(color="seagreen", width=2, dash="dash"),
                marker=dict(size=6, color="seagreen", symbol="diamond"),
                name="SARIMAX",
                legendgroup="sarimax",
            ), row=1, col=1)

        _fig_ts.add_vline(
            x=int(_last_origin.timestamp() * 1000),
            line_dash="dash", line_color="dimgray", line_width=1.2,
            row=1, col=1,
        )
        _fig_ts.add_vrect(
            x0=_last_origin, x1=_forecast_end,
            fillcolor="rgba(255, 215, 0, 0.18)",
            line_width=0, layer="below",
            row=1, col=1,
        )

        # ===== BOTTOM PANEL: forecast region only, auto-scaled =====
        # Hide last-window and far-train; show only actual + forecasts
        if not _sarima_test.empty:
            _fig_ts.add_trace(go.Scatter(
                x=_future_dates,
                y=_sarima_test["actual"],
                mode="lines+markers",
                line=dict(color="black", width=4),
                marker=dict(size=14, color="black"),
                name="Actual (test)",
                legendgroup="actual",
                showlegend=False,
            ), row=2, col=1)
            _fig_ts.add_trace(go.Scatter(
                x=_future_dates,
                y=_sarima_test["predicted"],
                mode="lines+markers",
                line=dict(color="firebrick", width=3.5, dash="dash"),
                marker=dict(size=12, color="firebrick", symbol="diamond"),
                name="SARIMA",
                legendgroup="sarima",
                showlegend=False,
            ), row=2, col=1)

        if not _sarimax_test.empty:
            _fig_ts.add_trace(go.Scatter(
                x=_future_dates_sx,
                y=_sarimax_test["predicted"],
                mode="lines+markers",
                line=dict(color="seagreen", width=3.5, dash="dash"),
                marker=dict(size=12, color="seagreen", symbol="diamond"),
                name="SARIMAX",
                legendgroup="sarimax",
                showlegend=False,
            ), row=2, col=1)

        # Top-panel x range
        if not _far_train.empty:
            _x_start_top = _far_train["week_start"].min()
        elif not _last_window.empty:
            _x_start_top = _last_window["week_start"].min()
        else:
            _x_start_top = _last_origin - pd.Timedelta(weeks=_CONTEXT_WEEKS)

        # Bottom-panel x range — origin -2w buffer to forecast_end +2w buffer
        _x_start_bot = _last_origin - pd.Timedelta(weeks=2)
        _x_end_bot = _forecast_end + pd.Timedelta(weeks=2)

        _fig_ts.update_xaxes(
            range=[_x_start_top, _forecast_end + pd.Timedelta(weeks=2)],
            title_text="Week",
            row=1, col=1,
        )
        _fig_ts.update_xaxes(
            range=[_x_start_bot, _x_end_bot],
            title_text="Week (forecast region only)",
            row=2, col=1,
        )
        _fig_ts.update_yaxes(title_text="Cases", row=1, col=1)
        _fig_ts.update_yaxes(title_text="Cases", row=2, col=1)

        _fig_ts.update_layout(
            height=720,
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(_fig_ts, use_container_width=True)
        st.caption(
            f"**Top panel** shows the last 1.5 years of history at full scale, with the 2022 outbreak "
            f"peak visible. The yellow shading and vertical line mark where the 12-week forecast begins. "
            f"**Bottom panel** zooms into just the forecast region so you can compare model accuracy directly. "
            f"**Black** = actual cases. **Red dashed** = SARIMA forecast. **Green dashed** = SARIMAX forecast. "
            f"Origin: {_last_origin.date()}."
        )
        """
    else:
        st.info("Historical dengue series not available for this chart.")

    selected_threshold = 400

    st.markdown("### Headline comparison")
    rows = []
    for kind in ["sarima", "sarimax"]:
        hsub = horizon_metrics_df[horizon_metrics_df["model_kind"] == kind]
        ssub = summary_df[
            (summary_df["model_kind"] == kind) & (summary_df["threshold"] == selected_threshold)
        ]
        if hsub.empty or ssub.empty:
            continue
        rows.append({
            "Model": kind.upper(),
            "sMAPE": round(float(hsub["smape"].mean()), 3),
            "MAE": round(float(hsub["mae"].mean()), 1),
            "RMSE": round(float(hsub["rmse"].mean()), 1),
            "Bias": round(float(hsub["bias"].mean()), 1),
            "PI coverage": round(float(hsub["pi_coverage"].mean()), 3),
            "Winkler": round(float(hsub["winkler_alpha20"].mean()), 1),
            "Brier": round(float(ssub["brier_score"].iloc[0]), 4),
            f"F1@{selected_threshold}": round(float(ssub["f1"].iloc[0]), 3) if pd.notna(ssub["f1"].iloc[0]) else float("nan"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Aggregations across horizons use simple unweighted means. Lower is better for sMAPE/MAE/RMSE/Winkler/Brier; higher is better for PI coverage and F1; bias closer to 0 is better.")

    st.markdown("### Skill by horizon")
    st.plotly_chart(
        _plot_backtest_skill_by_horizon(horizon_metrics_df, dm_df),
        use_container_width=True,
    )
    st.caption("Stars (*) on the left panel mark horizons where Diebold-Mariano test rejects equal accuracy at p<0.05. Right panel: PI coverage close to 0.80 indicates well-calibrated 80% intervals.")

    st.markdown(f"### Decision skill at threshold = {selected_threshold} cases")
    cm_cols = st.columns(2)
    for col_widget, kind in zip(cm_cols, ["sarima", "sarimax"]):
        ssub = summary_df[
            (summary_df["model_kind"] == kind) & (summary_df["threshold"] == selected_threshold)
        ]
        if ssub.empty:
            continue
        s = ssub.iloc[0]
        with col_widget:
            st.markdown(f"**{kind.upper()}**")
            cm = pd.DataFrame(
                {"Actual >= T": [int(s["tp"]), int(s["fn"])], "Actual < T": [int(s["fp"]), int(s["tn"])]},
                index=["Predicted >= T", "Predicted < T"],
            )
            st.dataframe(cm, use_container_width=True)
            st.caption(
                f"Precision: {s['precision']:.2f}  |  Recall: {s['recall']:.2f}  |  F1: {s['f1']:.2f}"
            )
    st.caption("In surveillance, recall (catching real outbreaks) usually matters more than precision (avoiding false alarms). Prefer the model with higher recall unless its false-alarm rate is operationally untenable.")

    st.markdown("### Predicted vs actual at h=4 (operational lead time)")

    H_FIXED = 4
    sub = per_origin_df[per_origin_df["h"] == H_FIXED].copy()

    if sub.empty:
        st.info(f"No backtest data available at h={H_FIXED}.")
        return

    all_actuals = sub["actual"].to_numpy(dtype=float)
    all_predicted = sub["predicted"].to_numpy(dtype=float)
    ref_max = float(max(all_actuals.max(), all_predicted.max())) * 1.05
    ref_min = 0.0

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[ref_min, ref_max],
        y=[ref_min, ref_max],
        mode="lines",
        line=dict(color="gray", dash="dot", width=1),
        name="Perfect forecast",
        hoverinfo="skip",
    ))

    model_specs = [
        ("sarima", "firebrick", "SARIMA"),
        ("sarimax", "seagreen", "SARIMAX"),
    ]
    for kind, color, label in model_specs:
        s = sub[sub["model_kind"] == kind].copy()
        if s.empty:
            continue

        fig.add_trace(go.Scatter(
            x=s["actual"],
            y=s["predicted"],
            mode="markers",
            marker=dict(size=6, color=color, opacity=0.55),
            name=label,
            hovertemplate=(
                "Origin: %{customdata}<br>"
                "Actual: %{x:.0f}<br>"
                "Predicted: %{y:.0f}<extra></extra>"
            ),
            customdata=pd.to_datetime(s["origin_week"]).dt.strftime("%Y-%m-%d"),
        ))

        if len(s) >= 2 and s["actual"].nunique() > 1:
            slope, intercept = np.polyfit(s["actual"].astype(float), s["predicted"].astype(float), 1)
            x_line = np.linspace(ref_min, ref_max, 100)
            y_line = slope * x_line + intercept
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color=color, width=1.5),
                name=f"{label} fit",
                hoverinfo="skip",
                opacity=0.6,
            ))

    fig.add_vline(
        x=threshold, line_dash="dot", line_color="gray",
        annotation_text=f"T={threshold}", annotation_position="top",
    )
    fig.add_hline(y=threshold, line_dash="dot", line_color="gray")

    y_offset_step = (ref_max - ref_min) * 0.06
    for i, (kind, color, label) in enumerate(model_specs):
        s = sub[sub["model_kind"] == kind]
        if s.empty:
            continue
        actual_high = s["actual"] >= threshold
        pred_high = s["predicted"] >= threshold
        tp = int((actual_high & pred_high).sum())
        fn = int((actual_high & ~pred_high).sum())
        fp = int((~actual_high & pred_high).sum())
        tn = int((~actual_high & ~pred_high).sum())
        fig.add_annotation(
            x=ref_max * 0.98,
            y=ref_max * 0.98 - i * y_offset_step,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(size=11, color=color),
            text=f"<b>{label}</b>  TP={tp}  FN={fn}  FP={fp}  TN={tn}",
        )

    fig.update_layout(
        title="Predicted vs actual cases — all backtest origins at h=4",
        xaxis_title="Actual cases",
        yaxis_title="Predicted cases",
        xaxis=dict(range=[ref_min, ref_max]),
        yaxis=dict(range=[ref_min, ref_max]),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Each point is one backtest origin at h=4 weeks ahead — the operational lead time "
        f"used by the surveillance recommendation engine. Diagonal = perfect forecast. "
        f"Above the diagonal = over-forecast; below = under-forecast. The fit lines (one per "
        f"model) reveal systematic bias: a fit below the diagonal indicates consistent "
        f"under-forecasting. Threshold lines (T={threshold}) divide the plane into "
        f"outbreak-classification outcomes: top-right = correct outbreak alert (TP), "
        f"bottom-left = correct quiet week (TN), top-left = false alarm (FP), "
        f"bottom-right = missed outbreak (FN). Per-horizon skill is shown in the chart above."
    )


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("NEA Controls")
horizon = st.sidebar.selectbox("Future forecast horizon (weeks)", [4, 8, 12], index=2)
threshold = st.sidebar.number_input("Outbreak threshold", min_value=50, max_value=1000, value=400, step=10)
weather_lag_weeks = st.sidebar.slider(
    "Weather-to-dengue lag (weeks)",
    min_value=0,
    max_value=26,
    value=DEFAULT_WEATHER_LAG_WEEKS,
    step=1,
    help="Weather features are shifted forward by this many weeks before being aligned to weekly dengue cases.",
)

sensitivity = st.sidebar.selectbox("Risk sensitivity", ["Conservative", "Balanced", "Aggressive"], index=1)
if sensitivity == "Conservative":
    low_cut, high_cut = 0.40, 0.70
elif sensitivity == "Aggressive":
    low_cut, high_cut = 0.20, 0.50
else:
    low_cut, high_cut = 0.30, 0.60


# -----------------------------
# Pipeline
# -----------------------------
st.title("Dengue Surveillance and Early Warning Dashboard")
st.caption("Data source: Singapore MOH weekly dengue cases.")

try:
    with st.spinner("Loading precomputed historical artifacts and preparing the forecast view..."):
        artifacts = load_precomputed_analysis_artifacts()
        manifest = artifacts["manifest"]
        weekly_df = artifacts["weekly_df"]
        lag_summary_df = artifacts["weekly_lag_summary_df"]
        best_lags_df = artifacts["weekly_best_lags_df"]
        literature_summary_df = artifacts["literature_summary_df"]
        lag_qa_df = artifacts["qa_df"]
        default_weekly_weather_df = artifacts["weekly_weather_df"]
        backtest_per_origin_df = artifacts.get("backtest_per_origin_df", pd.DataFrame())
        backtest_horizon_metrics_df = artifacts.get("backtest_horizon_metrics_df", pd.DataFrame())
        backtest_summary_df = artifacts.get("backtest_summary_df", pd.DataFrame())
        backtest_dm_df = artifacts.get("backtest_diebold_mariano_df", pd.DataFrame())
        backtest_meta = manifest.get("backtest", {})
        backtests_available = not backtest_horizon_metrics_df.empty

        use_precomputed_default_forecasts = (
            int(weather_lag_weeks) == int(DEFAULT_WEATHER_LAG_WEEKS) and int(horizon) == 12
        )

        if use_precomputed_default_forecasts:
            weekly_weather_df = default_weekly_weather_df.copy()
            forecast_df = artifacts["sarima_forecast_df"].copy()
            arimax_forecast_df = artifacts["arimax_forecast_df"].copy()
            arimax_future_exog_df = artifacts["arimax_future_exog_df"].copy()
            weather_df = pd.DataFrame()
            forecast_mode = "precomputed-default-artifacts"
        else:
            weather_df = load_weather_forecast_history()
            weather_df = weather_df[
                pd.to_datetime(weather_df["query_date"], errors="coerce") <= ANALYSIS_WEATHER_END_DATE
            ].copy()
            weekly_weather_df = shared_build_weekly_weather_features(weather_df, lag_days=weather_lag_weeks * 7)
            shared_validate_weather_overlap(weekly_df, weekly_weather_df, weather_lag_weeks * 7, MIN_OVERLAP_WEEKS)
            forecast_df = (
                artifacts["sarima_forecast_df"].copy()
                if int(horizon) == 12
                else shared_fit_fixed_sarima_and_forecast(weekly_df, steps=horizon, alpha=FORECAST_ALPHA)
            )
            arimax_input_for_forecast = shared_build_arimax_inputs(weekly_df, weekly_weather_df)
            arimax_forecast_df, arimax_future_exog_df = shared_fit_arimax_and_forecast(
                arimax_input_for_forecast,
                weekly_weather_df,
                steps=horizon,
                alpha=FORECAST_ALPHA,
            )
            forecast_mode = "custom-forecast-recompute"

        arimax_input_df = shared_build_arimax_inputs(weekly_df, weekly_weather_df)
        monthly_case_df, case_range_df = shared_build_monthly_case_seasonality(weekly_df)
        monthly_temp_df, temp_range_df = shared_build_monthly_temperature_seasonality(weekly_weather_df)
        driver_metrics_df = shared_build_driver_metrics(arimax_input_df)
        stl_df, stl_diag_df, seasonal_strength, trend_slope_now = shared_stl_diagnostics(weekly_df)
        seasonal_profile_df = shared_build_seasonal_profile(stl_df)
        sarima_risk_df = shared_assign_risk_bands(forecast_df, threshold=threshold, low_cut=low_cut, high_cut=high_cut)
        arimax_risk_df = shared_assign_risk_bands(arimax_forecast_df, threshold=threshold, low_cut=low_cut, high_cut=high_cut)
        qa_df = lag_qa_df.copy()

        rec = surveillance_recommendation(
            sarima_risk_df,
            trend_slope=trend_slope_now,
            horizon_mode=horizon,
            watch_enabled=False,
            threshold=threshold,
        )

        weather_cache_diag_df = pd.DataFrame(
            [
                {"metric": "forecast_mode", "value": forecast_mode},
                {"metric": "artifact_manifest_generated_at", "value": manifest.get("generated_at_utc", "")},
                {"metric": "weather_cache_file", "value": manifest.get("sources", {}).get("weather", {}).get("path", str(WEATHER_CACHE_FILE))},
                {"metric": "weather_cache_mtime_utc", "value": manifest.get("sources", {}).get("weather", {}).get("mtime_utc", "")},
                {"metric": "weather_cache_rows_loaded", "value": int(len(weather_df))},
                {"metric": "analysis_weather_end_date", "value": ANALYSIS_WEATHER_END_DATE},
                {"metric": "weather_lag_weeks", "value": int(weather_lag_weeks)},
                {
                    "metric": "weather_cache_query_start",
                    "value": pd.to_datetime(weather_df["query_date"], errors="coerce").min() if not weather_df.empty else default_weekly_weather_df["source_query_start"].min(),
                },
                {
                    "metric": "weather_cache_query_end",
                    "value": pd.to_datetime(weather_df["query_date"], errors="coerce").max() if not weather_df.empty else default_weekly_weather_df["source_query_end"].max(),
                },
                {"metric": "weather_weekly_rows", "value": int(len(weekly_weather_df))},
                {
                    "metric": "weather_weekly_source_start",
                    "value": weekly_weather_df["source_query_start"].min() if not weekly_weather_df.empty else pd.NaT,
                },
                {
                    "metric": "weather_weekly_source_end",
                    "value": weekly_weather_df["source_query_end"].max() if not weekly_weather_df.empty else pd.NaT,
                },
                {"metric": "arimax_training_rows", "value": int(len(arimax_input_df))},
                {
                    "metric": "arimax_overlap_start",
                    "value": arimax_input_df["week_start"].min() if not arimax_input_df.empty else pd.NaT,
                },
                {
                    "metric": "arimax_overlap_end",
                    "value": arimax_input_df["week_start"].max() if not arimax_input_df.empty else pd.NaT,
                },
                {
                    "metric": "arimax_future_exog_start",
                    "value": arimax_future_exog_df["week_start"].min() if not arimax_future_exog_df.empty else pd.NaT,
                },
                {
                    "metric": "arimax_future_exog_end",
                    "value": arimax_future_exog_df["week_start"].max() if not arimax_future_exog_df.empty else pd.NaT,
                },
                {"metric": "driver_metrics_count", "value": int(len(driver_metrics_df))},
            ]
        )
except Exception as exc:
    st.error(f"Failed to load dashboard data/model pipeline: {exc}")
    st.stop()


# -----------------------------
# Header cards
# -----------------------------
sarimax_current_band = (
    arimax_risk_df["risk_band"].iloc[0] if not arimax_risk_df.empty else "Low"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("SARIMA risk band", rec["current_risk"])
col2.metric("SARIMAX risk band", sarimax_current_band)
col3.metric("Recommended posture", rec["action"])
col4.metric("Model confidence", f"{rec['confidence']:.2f}")

st.info(
    f"STL seasonal strength={seasonal_strength:.2f} | "
    f"Current trend slope={trend_slope_now:.2f} | SARIMA={FIXED_ORDER}x{FIXED_SEASONAL_ORDER}"
)


# -----------------------------
# Presentation flow
# -----------------------------
st.markdown("## Weekly Dengue Cases")
st.plotly_chart(plot_history(weekly_df), use_container_width=True)

st.markdown("## STL Decomposition")
st.plotly_chart(plot_stl(stl_df), use_container_width=True)

st.markdown("## SARIMA Forecast (Last 1 Year Context)")
st.plotly_chart(plot_future_forecast(weekly_df, sarima_risk_df, threshold=threshold), use_container_width=True)

st.markdown("## Seasonal Profile (Operational Insight)")
st.plotly_chart(plot_seasonal_profile(seasonal_profile_df), use_container_width=True)

st.markdown("## Factors Used for Forecast")
driver_text_col, driver_plot_col = st.columns([1.05, 1.55])
with driver_text_col:
    st.markdown(
        """
**Temperature**
Warmer weeks can create more favorable mosquito conditions and amplify near-term transmission pressure.

**Humidity**
Humidity helps explain why similar temperature weeks do not always produce the same dengue outcome.

**Wind and Weather**
Short-term weather shifts complement the baseline seasonal pattern and can change operational risk.

"""
    )
with driver_plot_col:
    st.plotly_chart(plot_factors_used_for_forecast(arimax_input_df), use_container_width=True)

st.markdown("## Weekly Temperature Lag Analysis")
st.plotly_chart(plot_weekly_temperature_lag_focus(lag_summary_df), use_container_width=True)

st.markdown("## Weekly Lag Surface")
st.plotly_chart(plot_weekly_lag_correlation_heatmap(lag_summary_df), use_container_width=True)

st.markdown("## Seasonality Effects in Dengue Cases")
st.plotly_chart(plot_dengue_seasonality(monthly_case_df, case_range_df), use_container_width=True)

st.markdown("## Temperature Seasonality in Singapore")
st.plotly_chart(plot_temperature_seasonality(monthly_temp_df, temp_range_df), use_container_width=True)

st.markdown("## Projection of Dengue Cases using SARIMAX (with Weather Inputs)")
st.caption(
    "The left panel standardizes temperature, humidity, wind, and warm-day share onto the same scale. "
    "A value above 0 means that driver is above its historical mean; below 0 means below its historical mean."
)
st.plotly_chart(
    plot_arimax_projection_summary(
        weekly_df,
        arimax_input_df,
        arimax_future_exog_df,
        arimax_risk_df,
        threshold=threshold,
    ),
    use_container_width=True,
)

with st.expander("QA + Diagnostics"):
    st.write("### STL diagnostics")
    st.dataframe(stl_diag_df, use_container_width=True)
    st.write("### QA checks")
    st.dataframe(qa_df, use_container_width=True)
    st.write("### Lag analysis QA")
    st.dataframe(lag_qa_df, use_container_width=True)
    st.write("### Weather cache")
    st.dataframe(weather_cache_diag_df, use_container_width=True)
    st.write("### Best lag summary")
    st.dataframe(best_lags_df, use_container_width=True)
    st.write("### Driver correlations")
    st.dataframe(driver_metrics_df, use_container_width=True)

with st.expander("Model Backtest (Walk-Forward Evaluation)", expanded=False):
    if not backtests_available:
        st.info(
            "Backtest artifacts not found. Run `build_historical_lag_analysis.py` "
            "with `RUN_BACKTEST=True` to populate this section."
        )
    else:
        _render_backtest_panel(
            backtest_per_origin_df,
            backtest_horizon_metrics_df,
            backtest_summary_df,
            backtest_dm_df,
            backtest_meta,
            threshold=int(threshold),
            weekly_df=weekly_df,
        )

with st.expander("Appendix: How to Read This Dashboard"):
    st.markdown("""
### What is a risk signal?
A risk signal is the model-estimated probability that weekly dengue cases will exceed the selected threshold.

### How risk is quantified
- `p_exceed_threshold`: probability that forecasted weekly cases exceed the threshold (default 150).
- Risk bands:
  - Low: probability < lower cutoff
  - Medium: lower cutoff <= probability < upper cutoff
  - High: probability >= upper cutoff
- Default (Balanced) cutoffs:
  - Low < 0.30
  - Medium 0.30 to < 0.60
  - High >= 0.60

### Recommendation logic shown in the header
The recommendation combines:
- forecast risk in the next few weeks, and
- STL trend slope direction.

### How to read each chart
- Weekly Dengue Cases: historical level and 12-week moving average.
- STL Decomposition: observed, trend, seasonal, and residual components.
- SARIMA Forecast: next-week projections with confidence interval and threshold line.
- Seasonal Profile: average STL seasonal component by month (May/June highlighted).
- Factors Used for Forecast: weather drivers compared against weekly dengue cases.
- Literature Benchmark: Weekly Temperature Lag Analysis: weekly temperature correlation sweep with the 18-week Koh et al. benchmark and the local best lag shown together.
- Weekly Lag Surface: full weekly lag heatmap across weather drivers, with lag measured in weeks.
- Seasonality Effects in Dengue Cases: monthly pattern by year, with historical range.
- Temperature Seasonality in Singapore: monthly temperature pattern by year, with historical range.
- Projection of Dengue Cases using SARIMAX (with Weather Inputs): exogenous weather drivers alongside the weather-adjusted forecast. Background shading shows risk bands using the same Low/Medium/High cutoffs as the SARIMA panel.
- Model Backtest: walk-forward evaluation comparing how SARIMA and SARIMAX would have predicted the past. Lower sMAPE = more accurate point forecasts. PI coverage near 0.80 = well-calibrated intervals. Brier score and F1 measure decision skill at the chosen threshold.

### Important note on dates
If your source data is historical (for example ending in 2022), all recommendations are relative to that last available week.
""")

# Run:
# & C:/Users/davin/anaconda3/python.exe -m streamlit run "C:/Users/davin/OneDrive/Documents/Python stuffs/NEA Dengue Model/nea_dashboard_static.py"
