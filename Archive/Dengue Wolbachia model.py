#%% Imports
import os
import re
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import pmdarima as pm
except Exception:
    pm = None

try:
    from IPython.display import display
except Exception:
    display = None


#%% Config
SHOW_INLINE = True
SAVE_OUTPUTS = False

DATASET_ID = "d_ca168b2cb763640d72c4600a68f9909e"
DATASTORE_URL = "https://data.gov.sg/api/action/datastore_search"

DENGUE_TYPES = ["Dengue Fever", "Dengue Haemorrhagic Fever"]
FORECAST_HORIZON = 4
HOLDOUT_WEEKS = 52
BACKTEST_STRIDE = 2
MAX_BACKTEST_ORIGINS = 24
SEASONAL_PERIOD = 52
OUTBREAK_THRESHOLD = 150
AUTO_SARIMA_TRACE = False

OUT_MODELING_CSV = Path("phase1_weekly_modeling_table.csv")
OUT_FORECAST_CSV = Path("phase1_forecast_backtest.csv")
OUT_SUMMARY_PNG = Path("phase1_sarima_summary.png")
LOCAL_RAW_CACHE = Path("singapore_dengue_raw_records.csv")
USE_LOCAL_CACHE_IF_AVAILABLE = True
CACHE_RAW_DATA = True

warnings.filterwarnings("ignore")


#%% Shared display helpers

def _show_table(df: pd.DataFrame, label: str) -> None:
    print(f"\n{label}")
    if SHOW_INLINE and display is not None:
        display(df)
    else:
        print(df.to_string(index=False))


def _headers() -> dict:
    api_key = os.getenv("DATA_GOV_SG_API_KEY", "").strip()
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


#%% Data fetch

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


def fetch_all_records(resource_id: str) -> list[dict]:
    payload = request_json(
        DATASTORE_URL,
        params={"resource_id": resource_id, "limit": 50000, "offset": 0},
    )

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
        payload = request_json(
            DATASTORE_URL,
            params={"resource_id": resource_id, "limit": page_size, "offset": offset},
        )
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


def load_raw_records() -> pd.DataFrame:
    if USE_LOCAL_CACHE_IF_AVAILABLE and LOCAL_RAW_CACHE.exists():
        df = pd.read_csv(LOCAL_RAW_CACHE)
        print(f"Loaded cached raw data: {LOCAL_RAW_CACHE.resolve()}")
        return df

    records = fetch_all_records(DATASET_ID)
    df = pd.DataFrame(records)

    if CACHE_RAW_DATA:
        df.to_csv(LOCAL_RAW_CACHE, index=False)
        print(f"Cached raw data: {LOCAL_RAW_CACHE.resolve()}")

    return df


#%% Data preparation and QA

def parse_epi_week_to_sunday(epi_week: str):
    match = re.fullmatch(r"(\d{4})-W(\d{1,2})", str(epi_week).strip())
    if not match:
        return pd.NaT

    year = int(match.group(1))
    week = int(match.group(2))
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    first_week_sunday = jan1 - pd.Timedelta(days=(jan1.weekday() + 1) % 7)
    return first_week_sunday + pd.Timedelta(weeks=week - 1)


def build_weekly_dengue_series(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = {"epi_week", "disease", "no._of_cases"}
    missing = required_cols.difference(raw_df.columns)
    if missing:
        raise RuntimeError(f"Missing expected columns in source data: {sorted(missing)}")

    work = raw_df.copy()
    work["no._of_cases"] = pd.to_numeric(work["no._of_cases"], errors="coerce").fillna(0)
    work = work[work["disease"].isin(DENGUE_TYPES)].copy()
    if work.empty:
        raise RuntimeError("No dengue rows found in source data")

    grouped = work.groupby(["epi_week", "disease"], as_index=False)["no._of_cases"].sum()

    weekly = grouped.pivot(index="epi_week", columns="disease", values="no._of_cases").fillna(0).reset_index()

    if "Dengue Fever" not in weekly.columns:
        weekly["Dengue Fever"] = 0
    if "Dengue Haemorrhagic Fever" not in weekly.columns:
        weekly["Dengue Haemorrhagic Fever"] = 0

    parts = weekly["epi_week"].str.extract(r"(?P<year>\d{4})-W(?P<week>\d{1,2})")
    parts["year"] = pd.to_numeric(parts["year"], errors="coerce")
    parts["week"] = pd.to_numeric(parts["week"], errors="coerce")
    weekly = pd.concat([weekly, parts], axis=1)
    weekly = weekly.dropna(subset=["year", "week"]).copy()
    weekly["year"] = weekly["year"].astype(int)
    weekly["week"] = weekly["week"].astype(int)
    weekly = weekly.sort_values(["year", "week"]).reset_index(drop=True)

    weekly["week_start"] = weekly["epi_week"].map(parse_epi_week_to_sunday)
    weekly["Total Dengue Cases"] = weekly["Dengue Fever"] + weekly["Dengue Haemorrhagic Fever"]
    weekly["12-week Moving Average"] = weekly["Total Dengue Cases"].rolling(12, min_periods=1).mean()

    weekly["model_week_idx"] = np.arange(len(weekly), dtype=int)
    if weekly["week_start"].notna().any():
        base_week = pd.to_datetime(weekly.loc[weekly["week_start"].notna(), "week_start"].iloc[0])
    else:
        base_week = pd.Timestamp("2012-01-01")
    weekly["model_week_start"] = base_week + pd.to_timedelta(weekly["model_week_idx"] * 7, unit="D")

    weekly = weekly[
        [
            "epi_week",
            "week_start",
            "model_week_idx",
            "model_week_start",
            "Dengue Fever",
            "Dengue Haemorrhagic Fever",
            "Total Dengue Cases",
            "12-week Moving Average",
        ]
    ]

    return work, weekly


def run_qa_checks(weekly_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if weekly_df.empty:
        raise RuntimeError("Weekly dengue table is empty")

    checks = []
    checks.append({"check": "non_empty", "value": int(len(weekly_df)), "status": "PASS" if len(weekly_df) > 0 else "FAIL"})

    dup_epi = int(weekly_df["epi_week"].duplicated().sum())
    checks.append({"check": "duplicate_epi_week", "value": dup_epi, "status": "PASS" if dup_epi == 0 else "FAIL"})

    monotonic_idx = bool(weekly_df["model_week_idx"].is_monotonic_increasing)
    checks.append({"check": "model_index_monotonic", "value": int(monotonic_idx), "status": "PASS" if monotonic_idx else "FAIL"})

    negative_cases = int((weekly_df["Total Dengue Cases"] < 0).sum())
    checks.append({"check": "negative_cases", "value": negative_cases, "status": "PASS" if negative_cases == 0 else "FAIL"})

    zero_weeks = int((weekly_df["Total Dengue Cases"] == 0).sum())
    checks.append({"check": "zero_case_weeks", "value": zero_weeks, "status": "PASS"})

    display_gaps = weekly_df["week_start"].dropna().sort_values().diff().dt.days
    non_7day_gaps = int((display_gaps.dropna() != 7).sum())
    checks.append({"check": "display_week_gaps_non7days", "value": non_7day_gaps, "status": "PASS" if non_7day_gaps <= 3 else "WARN"})

    q1 = weekly_df["Total Dengue Cases"].quantile(0.25)
    q3 = weekly_df["Total Dengue Cases"].quantile(0.75)
    iqr = q3 - q1
    spike_cutoff = float(q3 + 3.0 * iqr)
    spike_df = weekly_df[weekly_df["Total Dengue Cases"] > spike_cutoff][
        ["epi_week", "model_week_start", "Total Dengue Cases"]
    ].copy()
    checks.append({"check": "extreme_spikes_count", "value": int(len(spike_df)), "status": "PASS"})

    check_df = pd.DataFrame(checks)
    return check_df, spike_df.head(10)


#%% Seasonality diagnostics

def periodogram_table(series: pd.Series, top_k: int = 5) -> pd.DataFrame:
    arr = series.astype(float).to_numpy()
    arr = arr - np.mean(arr)
    n = len(arr)

    freq = np.fft.rfftfreq(n, d=1.0)
    power = np.abs(np.fft.rfft(arr)) ** 2

    mask = freq > 0
    freq = freq[mask]
    power = power[mask]
    period = 1.0 / freq

    out = pd.DataFrame({"frequency": freq, "period_weeks": period, "power": power})
    out = out.sort_values("power", ascending=False).head(top_k).reset_index(drop=True)
    return out


def run_seasonality_module(weekly_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ts = pd.Series(
        weekly_df["Total Dengue Cases"].to_numpy(dtype=float),
        index=weekly_df["model_week_start"],
        name="Total Dengue Cases",
    )

    stl = STL(ts, period=SEASONAL_PERIOD, robust=True).fit()

    seasonal_strength = max(0.0, 1.0 - (np.var(stl.resid) / np.var(stl.seasonal + stl.resid)))
    trend_strength = max(0.0, 1.0 - (np.var(stl.resid) / np.var(stl.trend + stl.resid)))

    period_df = periodogram_table(ts, top_k=6)
    dominant_period = float(period_df.iloc[0]["period_weeks"]) if not period_df.empty else float("nan")

    diag_df = pd.DataFrame(
        [
            {"metric": "seasonal_period_assumed_weeks", "value": SEASONAL_PERIOD},
            {"metric": "dominant_periodogram_period_weeks", "value": round(dominant_period, 2)},
            {"metric": "seasonal_strength_stl", "value": round(float(seasonal_strength), 4)},
            {"metric": "trend_strength_stl", "value": round(float(trend_strength), 4)},
        ]
    )

    stl_fig = stl.plot()
    stl_fig.set_size_inches(13, 8)
    stl_fig.suptitle("STL Decomposition - Weekly Total Dengue Cases", fontsize=13)
    stl_fig.tight_layout()
    if SHOW_INLINE:
        plt.show()

    lags = min(104, max(20, len(ts) // 3))
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    plot_acf(ts, ax=axes[0], lags=lags)
    axes[0].set_title("ACF")

    plot_pacf(ts, ax=axes[1], lags=min(lags, len(ts) // 2 - 1), method="ywm")
    axes[1].set_title("PACF")

    axes[2].plot(period_df["period_weeks"], period_df["power"], marker="o")
    axes[2].set_title("Top Periodogram Peaks")
    axes[2].set_xlabel("Period (weeks)")
    axes[2].set_ylabel("Power")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    if SHOW_INLINE:
        plt.show()

    figs = {"stl": stl_fig, "acf_pacf_periodogram": fig}
    return diag_df, {"periodogram_top": period_df, "figures": figs}


#%% Auto-SARIMA and backtest

def _fit_sarima(train_series: pd.Series, order: tuple, seasonal_order: tuple):
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def tune_sarima_auto(train_series: pd.Series) -> tuple[tuple, tuple, pd.DataFrame]:
    if pm is not None:
        auto_model = pm.auto_arima(
            train_series,
            start_p=0,
            start_q=0,
            max_p=2,
            max_d=2,
            max_q=2,
            start_P=0,
            start_Q=0,
            max_P=1,
            max_D=1,
            max_Q=1,
            seasonal=True,
            m=SEASONAL_PERIOD,
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True,
            trace=AUTO_SARIMA_TRACE,
            information_criterion="aic",
            with_intercept=False,
        )

        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        tune_df = pd.DataFrame(
            [
                {
                    "method": "pmdarima.auto_arima",
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "aic": float(auto_model.aic()),
                }
            ]
        )
        return order, seasonal_order, tune_df

    fallback_candidates = [
        ((0, 1, 1), (0, 1, 1, SEASONAL_PERIOD)),
        ((1, 1, 0), (1, 1, 0, SEASONAL_PERIOD)),
        ((1, 1, 1), (1, 1, 1, SEASONAL_PERIOD)),
        ((2, 1, 1), (1, 1, 1, SEASONAL_PERIOD)),
        ((1, 0, 1), (1, 1, 1, SEASONAL_PERIOD)),
        ((1, 2, 1), (1, 1, 1, SEASONAL_PERIOD)),
        ((2, 1, 2), (1, 0, 1, SEASONAL_PERIOD)),
        ((0, 1, 2), (0, 1, 1, SEASONAL_PERIOD)),
        ((2, 1, 0), (1, 1, 0, SEASONAL_PERIOD)),
    ]

    rows = []
    for order, seasonal_order in fallback_candidates:
        try:
            res = _fit_sarima(train_series, order, seasonal_order)
            rows.append(
                {
                    "method": "fallback_auto_search",
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "aic": float(res.aic),
                    "order_t": order,
                    "seasonal_t": seasonal_order,
                }
            )
        except Exception:
            continue

    if not rows:
        raise RuntimeError("Auto-SARIMA fallback failed: no candidate model converged")

    tune_df = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    order = tune_df.loc[0, "order_t"]
    seasonal_order = tune_df.loc[0, "seasonal_t"]
    tune_df = tune_df.drop(columns=["order_t", "seasonal_t"])
    return order, seasonal_order, tune_df


def walk_forward_backtest(series: pd.Series, train_end_idx: int, order: tuple, seasonal_order: tuple) -> pd.DataFrame:
    rows = []

    origins = list(range(train_end_idx, len(series) - FORECAST_HORIZON + 1, BACKTEST_STRIDE))
    if len(origins) > MAX_BACKTEST_ORIGINS:
        origins = origins[-MAX_BACKTEST_ORIGINS:]

    for origin_idx in origins:
        train_series = series.iloc[:origin_idx]
        try:
            res = _fit_sarima(train_series, order, seasonal_order)
            pred = res.forecast(FORECAST_HORIZON)
        except Exception:
            continue

        for lead in range(1, FORECAST_HORIZON + 1):
            target_idx = origin_idx + lead - 1
            actual = float(series.iloc[target_idx])
            predicted = float(pred.iloc[lead - 1])

            bench_idx = target_idx - SEASONAL_PERIOD
            seasonal_naive = float(series.iloc[bench_idx]) if bench_idx >= 0 else np.nan

            rows.append(
                {
                    "origin_idx": origin_idx,
                    "target_idx": target_idx,
                    "lead_weeks": lead,
                    "actual_cases": actual,
                    "predicted_cases": predicted,
                    "seasonal_naive_cases": seasonal_naive,
                }
            )

    if not rows:
        raise RuntimeError("Walk-forward backtest produced no forecasts")

    return pd.DataFrame(rows)


#%% Metrics and outbreak utility

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(np.where(denom == 0, 0.0, 2.0 * np.abs(y_true - y_pred) / denom)) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "sMAPE_pct": smape}


def classification_metrics(actual_bool: np.ndarray, pred_bool: np.ndarray) -> dict:
    tp = int(np.sum((pred_bool == 1) & (actual_bool == 1)))
    fp = int(np.sum((pred_bool == 1) & (actual_bool == 0)))
    tn = int(np.sum((pred_bool == 0) & (actual_bool == 0)))
    fn = int(np.sum((pred_bool == 0) & (actual_bool == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarm_rate": float(false_alarm_rate),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def build_metrics_tables(forecast_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows = []

    for lead in sorted(forecast_df["lead_weeks"].unique()):
        part = forecast_df[forecast_df["lead_weeks"] == lead].dropna(subset=["seasonal_naive_cases"])
        if part.empty:
            continue

        y = part["actual_cases"].to_numpy(float)
        m = part["predicted_cases"].to_numpy(float)
        b = part["seasonal_naive_cases"].to_numpy(float)

        mm = regression_metrics(y, m)
        bm = regression_metrics(y, b)

        all_rows.append({"lead_weeks": lead, "model": "SARIMA", **mm})
        all_rows.append({"lead_weeks": lead, "model": "SeasonalNaive", **bm})

    reg_df = pd.DataFrame(all_rows)

    lead4 = forecast_df[forecast_df["lead_weeks"] == FORECAST_HORIZON].dropna(subset=["seasonal_naive_cases"]).copy()
    lead4["actual_outbreak"] = (lead4["actual_cases"] >= OUTBREAK_THRESHOLD).astype(int)
    lead4["pred_outbreak"] = (lead4["predicted_cases"] >= OUTBREAK_THRESHOLD).astype(int)
    lead4["naive_outbreak"] = (lead4["seasonal_naive_cases"] >= OUTBREAK_THRESHOLD).astype(int)

    cls_sarima = classification_metrics(lead4["actual_outbreak"].to_numpy(), lead4["pred_outbreak"].to_numpy())
    cls_naive = classification_metrics(lead4["actual_outbreak"].to_numpy(), lead4["naive_outbreak"].to_numpy())
    cls_df = pd.DataFrame(
        [
            {"model": "SARIMA", **cls_sarima},
            {"model": "SeasonalNaive", **cls_naive},
        ]
    )

    return reg_df, cls_df, lead4


def compute_lead_time_utility(forecast_df: pd.DataFrame, full_series: pd.Series, train_end_idx: int) -> pd.DataFrame:
    actual_outbreak = (full_series >= OUTBREAK_THRESHOLD).astype(int)

    onset_idxs = []
    for i in range(train_end_idx, len(full_series)):
        prev = int(actual_outbreak.iloc[i - 1]) if i > 0 else 0
        curr = int(actual_outbreak.iloc[i])
        if curr == 1 and prev == 0:
            onset_idxs.append(i)

    rows = []
    for onset_idx in onset_idxs:
        preds = forecast_df[
            (forecast_df["target_idx"] == onset_idx)
            & (forecast_df["lead_weeks"].between(1, FORECAST_HORIZON))
            & (forecast_df["predicted_cases"] >= OUTBREAK_THRESHOLD)
        ]

        if preds.empty:
            rows.append({"onset_idx": onset_idx, "flagged_early": 0, "best_lead_weeks": 0})
        else:
            best_lead = int(preds["lead_weeks"].max())
            rows.append({"onset_idx": onset_idx, "flagged_early": 1, "best_lead_weeks": best_lead})

    util_df = pd.DataFrame(rows)
    if util_df.empty:
        return pd.DataFrame([
            {"metric": "outbreak_onsets_in_holdout", "value": 0},
            {"metric": "onsets_flagged_1_to_4_weeks_early", "value": 0},
            {"metric": "lead_time_hit_rate", "value": 0.0},
            {"metric": "avg_best_lead_weeks_when_hit", "value": 0.0},
        ])

    hits = int(util_df["flagged_early"].sum())
    total = int(len(util_df))
    avg_lead = float(util_df.loc[util_df["flagged_early"] == 1, "best_lead_weeks"].mean()) if hits else 0.0

    return pd.DataFrame([
        {"metric": "outbreak_onsets_in_holdout", "value": total},
        {"metric": "onsets_flagged_1_to_4_weeks_early", "value": hits},
        {"metric": "lead_time_hit_rate", "value": round(hits / total, 4) if total else 0.0},
        {"metric": "avg_best_lead_weeks_when_hit", "value": round(avg_lead, 4)},
    ])


#%% Visualization of forecasts and alerts

def plot_forecast_and_alerts(weekly_df: pd.DataFrame, forecast_df: pd.DataFrame, train_end_idx: int):
    lead4 = forecast_df[forecast_df["lead_weeks"] == FORECAST_HORIZON].copy()
    lead4 = lead4.drop_duplicates(subset=["target_idx"], keep="last")

    holdout = weekly_df.iloc[train_end_idx:].copy()
    holdout = holdout[["model_week_idx", "model_week_start", "Total Dengue Cases"]].rename(
        columns={"Total Dengue Cases": "actual_cases"}
    )

    merged = holdout.merge(
        lead4[["target_idx", "predicted_cases", "seasonal_naive_cases"]],
        left_on="model_week_idx",
        right_on="target_idx",
        how="left",
    )

    merged["actual_outbreak"] = (merged["actual_cases"] >= OUTBREAK_THRESHOLD).astype(int)
    merged["pred_outbreak_4w"] = (merged["predicted_cases"] >= OUTBREAK_THRESHOLD).astype(int)

    any_lead_alert = (
        forecast_df.assign(pred_outbreak_any=(forecast_df["predicted_cases"] >= OUTBREAK_THRESHOLD).astype(int))
        .groupby("target_idx", as_index=False)["pred_outbreak_any"]
        .max()
    )
    merged = merged.merge(
        any_lead_alert,
        left_on="model_week_idx",
        right_on="target_idx",
        how="left",
    )
    merged["pred_outbreak_any"] = merged["pred_outbreak_any"].fillna(0).astype(int)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(weekly_df["model_week_start"], weekly_df["Total Dengue Cases"], color="tab:blue", alpha=0.45, label="Actual")
    axes[0].plot(merged["model_week_start"], merged["predicted_cases"], color="tab:red", linestyle="--", label="SARIMA 4-week ahead")
    axes[0].plot(merged["model_week_start"], merged["seasonal_naive_cases"], color="tab:green", linestyle=":", label="Seasonal naive")
    axes[0].axhline(OUTBREAK_THRESHOLD, color="black", linestyle="--", linewidth=1.0, label=f"Outbreak threshold ({OUTBREAK_THRESHOLD})")
    axes[0].axvline(weekly_df.iloc[train_end_idx]["model_week_start"], color="grey", linestyle="--", alpha=0.7)
    axes[0].set_title("Forecast vs Actual (Holdout) - 4-week horizon")
    axes[0].set_ylabel("Cases")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].step(merged["model_week_start"], merged["actual_outbreak"], where="mid", label="Actual outbreak", linewidth=2)
    axes[1].step(merged["model_week_start"], merged["pred_outbreak_4w"], where="mid", label="Pred outbreak (4-week lead)", linewidth=2)
    axes[1].step(merged["model_week_start"], merged["pred_outbreak_any"], where="mid", label="Pred outbreak (any lead 1-4)", linewidth=2)
    axes[1].set_title("Outbreak Alert Timeline")
    axes[1].set_ylabel("Alert (0/1)")
    axes[1].set_xlabel("Model week")
    axes[1].set_yticks([0, 1])
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    if SHOW_INLINE:
        plt.show()

    return fig


#%% End-to-end workflow

def run_phase1_pipeline():
    raw_df = load_raw_records()
    print(f"Source dataset id: {DATASET_ID}")
    print(f"Rows fetched (all diseases): {len(raw_df):,}")

    dengue_rows, weekly_df = build_weekly_dengue_series(raw_df)
    print(f"Rows kept (dengue only): {len(dengue_rows):,}")
    print(f"Weekly points: {len(weekly_df):,}")

    _show_table(weekly_df.head(12), "Weekly modeling table preview (head 12):")

    qa_df, spike_df = run_qa_checks(weekly_df)
    _show_table(qa_df, "QA checks:")
    _show_table(spike_df, "Top detected extreme spikes (if any):")

    season_df, season_artifacts = run_seasonality_module(weekly_df)
    _show_table(season_df, "Seasonality diagnostics summary:")
    _show_table(season_artifacts["periodogram_top"], "Periodogram top peaks:")

    series = weekly_df["Total Dengue Cases"].astype(float).reset_index(drop=True)
    if len(series) <= HOLDOUT_WEEKS + FORECAST_HORIZON + SEASONAL_PERIOD:
        raise RuntimeError("Not enough observations for requested holdout/horizon/seasonal setup")

    train_end_idx = len(series) - HOLDOUT_WEEKS
    train_series = series.iloc[:train_end_idx]

    best_order, best_seasonal, tune_df = tune_sarima_auto(train_series)
    print(f"Best SARIMA order: {best_order}, seasonal_order: {best_seasonal}")
    _show_table(tune_df.head(10), "Auto-SARIMA selection summary:")

    forecast_df = walk_forward_backtest(series, train_end_idx, best_order, best_seasonal)
    forecast_df["target_week_start"] = weekly_df.loc[forecast_df["target_idx"].astype(int), "model_week_start"].values

    reg_df, cls_df, _lead4_df = build_metrics_tables(forecast_df)
    _show_table(reg_df, "Backtest regression metrics by lead:")
    _show_table(cls_df, f"Outbreak classification metrics at {FORECAST_HORIZON}-week lead:")

    lead_time_df = compute_lead_time_utility(forecast_df, series, train_end_idx)
    _show_table(lead_time_df, "Lead-time utility (1-4 week early warning):")

    summary_fig = plot_forecast_and_alerts(weekly_df, forecast_df, train_end_idx)

    lead4_metrics = reg_df[reg_df["lead_weeks"] == FORECAST_HORIZON].set_index("model")
    beats_mae = lead4_metrics.loc["SARIMA", "MAE"] < lead4_metrics.loc["SeasonalNaive", "MAE"]
    beats_rmse = lead4_metrics.loc["SARIMA", "RMSE"] < lead4_metrics.loc["SeasonalNaive", "RMSE"]
    print(f"\nAcceptance check (lead={FORECAST_HORIZON}): beats seasonal-naive MAE={beats_mae}, RMSE={beats_rmse}")

    if SAVE_OUTPUTS:
        weekly_df.to_csv(OUT_MODELING_CSV, index=False)
        forecast_df.to_csv(OUT_FORECAST_CSV, index=False)
        summary_fig.savefig(OUT_SUMMARY_PNG, dpi=180)

        print(f"Saved modeling table: {OUT_MODELING_CSV.resolve()}")
        print(f"Saved forecast table: {OUT_FORECAST_CSV.resolve()}")
        print(f"Saved summary figure: {OUT_SUMMARY_PNG.resolve()}")
    else:
        print("SAVE_OUTPUTS=False: skipping file exports.")

    return {
        "weekly_df": weekly_df,
        "forecast_df": forecast_df,
        "reg_metrics": reg_df,
        "cls_metrics": cls_df,
        "lead_time": lead_time_df,
        "best_order": best_order,
        "best_seasonal": best_seasonal,
    }


#%% Run
if __name__ == "__main__":
    try:
        run_phase1_pipeline()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
