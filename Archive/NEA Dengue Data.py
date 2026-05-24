#%% Imports
import os
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

try:
    from IPython.display import display
except Exception:
    display = None


#%% Config
SHOW_INLINE = True
SAVE_OUTPUTS = False

DATASET_ID = "d_ca168b2cb763640d72c4600a68f9909e"
DATASTORE_URL = "https://data.gov.sg/api/action/datastore_search"

OUT_RAW_CSV = Path("singapore_dengue_raw_records.csv")
OUT_FILTERED_CSV = Path("singapore_dengue_rows_filtered.csv")
OUT_WEEKLY_CSV = Path("singapore_dengue_weekly_series.csv")
OUT_PLOT_PNG = Path("singapore_dengue_weekly_line.png")


#%% API helpers

def _headers():
    api_key = os.getenv("DATA_GOV_SG_API_KEY", "").strip()
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def request_json(url, params=None, max_attempts=10, timeout=30):
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=timeout)

            if resp.status_code == 429:
                if attempt == max_attempts:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After", "").strip()
                wait_s = float(retry_after) if retry_after.isdigit() else min(2**attempt, 120)
                time.sleep(wait_s)
                continue

            if resp.status_code in (500, 502, 503, 504):
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


def fetch_all_records(resource_id):
    # Single large call first to reduce rate-limit risk.
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

    # Fallback pagination if dataset grows beyond first pull.
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


#%% Transform helpers

def parse_epi_week_to_sunday(epi_week):
    match = re.fullmatch(r"(\d{4})-W(\d{1,2})", str(epi_week).strip())
    if not match:
        return pd.NaT

    year = int(match.group(1))
    week = int(match.group(2))
    # Epi-week here is Sunday-Saturday, with week 1 being the week that contains Jan 1.
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    first_week_sunday = jan1 - pd.Timedelta(days=(jan1.weekday() + 1) % 7)
    return first_week_sunday + pd.Timedelta(weeks=week - 1)


def build_weekly_dengue_series(records_df):
    required_cols = {"epi_week", "disease", "no._of_cases"}
    missing = required_cols.difference(records_df.columns)
    if missing:
        raise RuntimeError(f"Missing expected columns: {sorted(missing)}")

    work = records_df.copy()
    work["no._of_cases"] = pd.to_numeric(work["no._of_cases"], errors="coerce").fillna(0)

    target = ["Dengue Fever", "Dengue Haemorrhagic Fever"]
    work = work[work["disease"].isin(target)].copy()
    if work.empty:
        raise RuntimeError("No dengue rows found in source data")

    work["week_start"] = work["epi_week"].map(parse_epi_week_to_sunday)
    work = work.dropna(subset=["week_start"])
    if work.empty:
        raise RuntimeError("Failed to parse epi_week into dates")

    weekly = (
        work.groupby(["week_start", "disease"], as_index=False)["no._of_cases"]
        .sum()
        .pivot(index="week_start", columns="disease", values="no._of_cases")
        .fillna(0)
        .sort_index()
    )

    if "Dengue Fever" not in weekly.columns:
        weekly["Dengue Fever"] = 0
    if "Dengue Haemorrhagic Fever" not in weekly.columns:
        weekly["Dengue Haemorrhagic Fever"] = 0

    weekly.columns.name = None
    weekly["Total Dengue Cases"] = weekly["Dengue Fever"] + weekly["Dengue Haemorrhagic Fever"]
    weekly["12-week Moving Average"] = weekly["Total Dengue Cases"].rolling(12, min_periods=1).mean()

    return work, weekly.reset_index()


#%% Plot helpers

def plot_weekly_series(weekly_df):
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        weekly_df["week_start"],
        weekly_df["Total Dengue Cases"],
        linewidth=1.2,
        alpha=0.45,
        color="tab:blue",
        label="Weekly total dengue cases",
    )
    ax.plot(
        weekly_df["week_start"],
        weekly_df["12-week Moving Average"],
        linewidth=2.6,
        color="tab:red",
        label="12-week moving average",
    )

    ax.set_title("Singapore Weekly Dengue Cases (Official MOH Data)", fontsize=14, weight="bold")
    ax.set_xlabel("Epi-week start date")
    ax.set_ylabel("Number of cases")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    if SHOW_INLINE:
        plt.show()
    return fig


def _show_table(df, label):
    print(f"\n{label}")
    if SHOW_INLINE and display is not None:
        display(df)
    else:
        print(df)


#%% Fetch data
def step_fetch():
    records = fetch_all_records(DATASET_ID)
    raw_df = pd.DataFrame(records)

    print(f"Source dataset id: {DATASET_ID}")
    print(f"Rows fetched (all diseases): {len(raw_df):,}")
    _show_table(raw_df.head(20), "Raw data preview (head 20):")

    return raw_df


#%% Clean/transform data
def step_transform(raw_df):
    dengue_rows, weekly_df = build_weekly_dengue_series(raw_df)

    print(f"Rows kept (dengue only): {len(dengue_rows):,}")
    print(f"Weekly points: {len(weekly_df):,}")

    _show_table(dengue_rows.head(20), "Filtered dengue rows (head 20):")
    _show_table(weekly_df.head(20), "Weekly series (head 20):")
    _show_table(weekly_df[["Total Dengue Cases", "12-week Moving Average"]].describe(), "Weekly series summary:")

    return dengue_rows, weekly_df


#%% Plot/display
def step_plot(weekly_df):
    fig = plot_weekly_series(weekly_df)
    return fig


#%% Optional export
def step_export(raw_df, dengue_rows, weekly_df, fig):
    if not SAVE_OUTPUTS:
        print("SAVE_OUTPUTS=False: skipping CSV/PNG file writes.")
        return

    raw_df.to_csv(OUT_RAW_CSV, index=False)
    dengue_rows.to_csv(OUT_FILTERED_CSV, index=False)
    weekly_df.to_csv(OUT_WEEKLY_CSV, index=False)
    fig.savefig(OUT_PLOT_PNG, dpi=180)

    print(f"Saved raw records: {OUT_RAW_CSV.resolve()}")
    print(f"Saved filtered rows: {OUT_FILTERED_CSV.resolve()}")
    print(f"Saved weekly series: {OUT_WEEKLY_CSV.resolve()}")
    print(f"Saved line plot: {OUT_PLOT_PNG.resolve()}")


#%% Run full workflow
def run_workflow():
    raw_df = step_fetch()
    dengue_rows, weekly_df = step_transform(raw_df)
    fig = step_plot(weekly_df)
    step_export(raw_df, dengue_rows, weekly_df, fig)


if __name__ == "__main__":
    try:
        run_workflow()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise



