from pathlib import Path
import time

import pandas as pd

from weather_history_loader import (
    WEATHER_HISTORY_METADATA_URL,
    _extract_dataset_year_pairs,
    _historical_raw_file,
    load_weather_history_cache,
    request_json,
)


BASE_DIR = Path(__file__).resolve().parent
OUT_FILE = BASE_DIR / "singapore_weather_forecast_24hr_history.csv"
WEATHER_HISTORY_START_DATE = pd.Timestamp("2016-03-01")
WEATHER_REGIONS = ("west", "east", "central", "north", "south")
MAX_PASSES = 6
PASS_COOLDOWN_SECONDS = 20


def main() -> None:
    raw_cache_dir = BASE_DIR / "weather_history_raw"

    for attempt in range(1, MAX_PASSES + 1):
        print(f"Historical weather backfill pass {attempt}/{MAX_PASSES}")
        df = load_weather_history_cache(
            cache_path=OUT_FILE,
            history_start_date=WEATHER_HISTORY_START_DATE,
            regions=WEATHER_REGIONS,
            logger=print,
        )

        metadata = request_json(WEATHER_HISTORY_METADATA_URL)
        dataset_year_pairs = _extract_dataset_year_pairs(metadata)
        missing_years = [
            year
            for year, dataset_id in dataset_year_pairs
            if not _historical_raw_file(raw_cache_dir, year, dataset_id).exists()
        ]

        if not missing_years:
            print("All historical weather years are now cached locally.")
            break

        print(f"Still missing historical weather years after pass {attempt}: {missing_years}")
        if attempt < MAX_PASSES:
            print(f"Cooling down for {PASS_COOLDOWN_SECONDS} seconds before retrying...")
            time.sleep(PASS_COOLDOWN_SECONDS)

    query_dates = pd.to_datetime(df["query_date"], errors="coerce")
    years = sorted(query_dates.dt.year.dropna().astype(int).unique().tolist())

    print("")
    print(f"Wrote merged weather history CSV: {OUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Coverage: {query_dates.min()} -> {query_dates.max()}")
    print(f"Years present: {years}")


if __name__ == "__main__":
    main()
