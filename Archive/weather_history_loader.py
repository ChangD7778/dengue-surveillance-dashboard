import io
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests


WEATHER_HISTORY_COLLECTION_ID = 2213
WEATHER_HISTORY_METADATA_URL = (
    "https://api-production.data.gov.sg/v2/public/api/collections/"
    f"{WEATHER_HISTORY_COLLECTION_ID}/metadata?withDatasetMetadata=true"
)
DATASET_POLL_DOWNLOAD_URL = "https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download"
WEATHER_FORECAST_URL = "https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast"
HISTORICAL_RAW_DIRNAME = "weather_history_raw"


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


def request_text(url: str, params: dict | None = None, max_attempts: int = 10, timeout: int = 60) -> str:
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
            return resp.text
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


def _canonicalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _extract_csv_urls(metadata: dict) -> list[str]:
    text_blob = json.dumps(metadata)
    matches = re.findall(r"https?://[^\"'\\\s>]+\.csv(?:\?[^\"'\\\s>]*)?", text_blob, flags=re.IGNORECASE)
    urls: list[str] = []
    seen = set()
    for url in matches:
        clean = url.rstrip(".,)")
        if clean not in seen:
            seen.add(clean)
            urls.append(clean)
    return urls


def _extract_collection_years(metadata: dict) -> set[int]:
    text_blob = json.dumps(metadata)
    return {int(year) for year in re.findall(r"Historical 24-hour Weather Forecast \((\d{4})\)", text_blob)}


def _extract_dataset_year_pairs(metadata: dict) -> list[tuple[int, str]]:
    dataset_metadata = metadata.get("data", {}).get("datasetMetadata", []) or []
    pairs: list[tuple[int, str]] = []
    seen = set()
    for item in dataset_metadata:
        dataset_id = str(item.get("datasetId", "")).lower()
        name = str(item.get("name", ""))
        match = re.search(r"\((\d{4})\)", name)
        if not dataset_id or not match:
            continue
        pair = (int(match.group(1)), dataset_id)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return sorted(pairs, key=lambda x: x[0])


def _extract_dataset_ids(metadata: dict) -> list[str]:
    text_blob = json.dumps(metadata)
    matches = re.findall(r"\bd_[a-f0-9]{32}\b", text_blob, flags=re.IGNORECASE)
    ids: list[str] = []
    seen = set()
    for dataset_id in matches:
        dataset_id = dataset_id.lower()
        if dataset_id not in seen:
            seen.add(dataset_id)
            ids.append(dataset_id)
    return ids


def _historical_raw_file(raw_cache_dir: Path, year: int, dataset_id: str) -> Path:
    return raw_cache_dir / f"historical_24hr_weather_forecast_{year}_{dataset_id}.csv"


def _format_year_list(years: list[int] | set[int]) -> str:
    values = sorted(int(year) for year in years)
    return ", ".join(str(year) for year in values) if values else "none"


def _poll_download_url(dataset_id: str) -> str:
    payload = request_json(DATASET_POLL_DOWNLOAD_URL.format(dataset_id=dataset_id))
    if payload.get("code") != 0:
        raise RuntimeError(
            f"poll-download failed for dataset {dataset_id}: "
            f"{payload.get('errMsg', 'unknown error')}"
        )
    data = payload.get("data") or {}
    download_url = data.get("url")
    if not download_url:
        raise RuntimeError(f"poll-download succeeded for dataset {dataset_id} but data.url was missing")
    return str(download_url)


def _normalize_historical_weather_frame(raw_df: pd.DataFrame, csv_url: str) -> pd.DataFrame:
    work = raw_df.copy()
    work.columns = [_canonicalize_column(col) for col in work.columns]

    rename_map = {
        "date": "query_date",
        "update_timestamp": "updated_timestamp",
        "forecast_code": "general_forecast_code",
        "forecast_text": "general_forecast_text",
        "temperature_low": "temperature_low_c",
        "temperature_high": "temperature_high_c",
        "relative_humidity_low": "relative_humidity_low_pct",
        "relative_humidity_high": "relative_humidity_high_pct",
        "wind_speed_direction": "wind_direction",
        "time_period_start": "period_1_start",
        "time_period_end": "period_1_end",
        "time_period_text": "period_1_text",
        "south_forecast_code": "period_1_south_forecast_code",
        "south_forecast_text": "period_1_south_forecast_text",
        "north_forecast_code": "period_1_north_forecast_code",
        "north_forecast_text": "period_1_north_forecast_text",
        "east_forecast_code": "period_1_east_forecast_code",
        "east_forecast_text": "period_1_east_forecast_text",
        "central_forecast_code": "period_1_central_forecast_code",
        "central_forecast_text": "period_1_central_forecast_text",
        "west_forecast_code": "period_1_west_forecast_code",
        "west_forecast_text": "period_1_west_forecast_text",
    }
    work = work.rename(columns=rename_map)

    if "query_date" not in work.columns:
        raise RuntimeError(f"Historical weather CSV is missing a date column: {csv_url}")

    work["record_date"] = work.get("record_date", work["query_date"])
    work["period_count"] = work.get("period_count", 1)
    work["weather_source"] = work.get("weather_source", "historical_collection")
    work["weather_source_url"] = work.get("weather_source_url", csv_url)

    expected_columns = [
        "query_date",
        "record_date",
        "updated_timestamp",
        "timestamp",
        "valid_period_start",
        "valid_period_end",
        "valid_period_text",
        "general_forecast_code",
        "general_forecast_text",
        "temperature_low_c",
        "temperature_high_c",
        "relative_humidity_low_pct",
        "relative_humidity_high_pct",
        "wind_speed_low",
        "wind_speed_high",
        "wind_direction",
        "period_count",
        "period_1_start",
        "period_1_end",
        "period_1_text",
        "period_1_south_forecast_code",
        "period_1_south_forecast_text",
        "period_1_north_forecast_code",
        "period_1_north_forecast_text",
        "period_1_east_forecast_code",
        "period_1_east_forecast_text",
        "period_1_central_forecast_code",
        "period_1_central_forecast_text",
        "period_1_west_forecast_code",
        "period_1_west_forecast_text",
        "weather_source",
        "weather_source_url",
    ]
    for col in expected_columns:
        if col not in work.columns:
            work[col] = pd.NA

    return work[expected_columns].copy()


def _flatten_realtime_weather_record(record: dict, regions: tuple[str, ...]) -> dict:
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
        "weather_source": "realtime_api",
        "weather_source_url": WEATHER_FORECAST_URL,
    }

    for idx, period in enumerate(record.get("periods", []) or [], start=1):
        time_period = period.get("timePeriod", {}) or {}
        region_values = period.get("regions", {}) or {}
        row[f"period_{idx}_start"] = time_period.get("start")
        row[f"period_{idx}_end"] = time_period.get("end")
        row[f"period_{idx}_text"] = time_period.get("text")
        for region in regions:
            forecast_values = region_values.get(region, {}) or {}
            row[f"period_{idx}_{region}_forecast_code"] = forecast_values.get("code")
            row[f"period_{idx}_{region}_forecast_text"] = forecast_values.get("text")

    return row


def _dedupe_sort_weather(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    dedupe_cols = [col for col in ["query_date", "timestamp", "updated_timestamp", "valid_period_start"] if col in out.columns]
    if dedupe_cols:
        out = out.drop_duplicates(subset=dedupe_cols, keep="last")

    sort_cols = [col for col in ["query_date", "timestamp", "updated_timestamp"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def _merge_weather_cache(existing_df: pd.DataFrame, new_frames: list[pd.DataFrame], cache_path: Path) -> pd.DataFrame:
    if not new_frames:
        return existing_df

    additions_df = pd.concat(new_frames, ignore_index=True)
    if existing_df.empty:
        combined_df = additions_df
    else:
        combined_df = pd.concat([existing_df, additions_df], ignore_index=True)

    combined_df = _dedupe_sort_weather(combined_df)
    combined_df.to_csv(cache_path, index=False)
    return combined_df


def _fetch_historical_weather_frames(
    metadata: dict,
    start_date: pd.Timestamp,
    raw_cache_dir: Path,
    dataset_year_pairs: list[tuple[int, str]] | None = None,
    logger=print,
) -> tuple[pd.DataFrame, list[str]]:
    csv_urls = _extract_csv_urls(metadata)
    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    if csv_urls:
        for csv_url in csv_urls:
            csv_text = request_text(csv_url)
            frame = pd.read_csv(io.StringIO(csv_text))
            normalized = _normalize_historical_weather_frame(frame, csv_url)
            normalized["query_date"] = pd.to_datetime(normalized["query_date"], errors="coerce")
            normalized = normalized.dropna(subset=["query_date"]).copy()
            normalized = normalized[normalized["query_date"] >= start_date].reset_index(drop=True)
            if not normalized.empty:
                frames.append(normalized)
    else:
        dataset_year_pairs = dataset_year_pairs or _extract_dataset_year_pairs(metadata)
        if not dataset_year_pairs:
            dataset_ids = _extract_dataset_ids(metadata)
            dataset_year_pairs = [(idx, dataset_id) for idx, dataset_id in enumerate(dataset_ids, start=1)]
        if not dataset_year_pairs:
            raise RuntimeError(
                "Historical weather collection metadata did not contain CSV download URLs or dataset IDs."
            )

        missing_years: list[str] = []
        for year, dataset_id in dataset_year_pairs:
            raw_file = _historical_raw_file(raw_cache_dir, year, dataset_id)
            source_url = str(raw_file)
            if raw_file.exists():
                if logger:
                    logger(f"Using cached historical weather dataset for {year}.")
                frame = pd.read_csv(raw_file)
            else:
                try:
                    download_url = _poll_download_url(dataset_id)
                    csv_text = request_text(download_url)
                    raw_file.write_text(csv_text, encoding="utf-8")
                    source_url = download_url
                    if logger:
                        logger(f"Downloaded historical weather dataset for {year}.")
                    time.sleep(1.0)
                    frame = pd.read_csv(io.StringIO(csv_text))
                except Exception as exc:
                    missing_years.append(f"{year} ({dataset_id}: {exc})")
                    continue

            normalized = _normalize_historical_weather_frame(frame, source_url)
            normalized["query_date"] = pd.to_datetime(normalized["query_date"], errors="coerce")
            normalized = normalized.dropna(subset=["query_date"]).copy()
            normalized = normalized[normalized["query_date"] >= start_date].reset_index(drop=True)
            if not normalized.empty:
                frames.append(normalized)

        if missing_years:
            warnings.append(
                "Historical weather download incomplete. Missing years: "
                + "; ".join(missing_years)
                + ". Set DATA_GOV_SG_API_KEY or rerun to resume from the yearly raw CSVs already downloaded."
            )

    for normalized in frames:
        normalized["query_date"] = pd.to_datetime(normalized["query_date"], errors="coerce")
        normalized = normalized.dropna(subset=["query_date"]).copy()

    if not frames:
        raise RuntimeError("Historical weather collection returned no usable rows after normalization.")
    return _dedupe_sort_weather(pd.concat(frames, ignore_index=True)), warnings


def _fetch_realtime_weather_records_for_day(query_date: pd.Timestamp, regions: tuple[str, ...]) -> pd.DataFrame:
    date_str = pd.Timestamp(query_date).strftime("%Y-%m-%d")
    params = {"date": date_str}
    rows: list[dict] = []

    while True:
        payload = request_json_allow_404(WEATHER_FORECAST_URL, params=params)
        if not payload:
            break

        data = payload.get("data", {}) or {}
        batch = data.get("records", []) or []
        for record in batch:
            enriched = dict(record)
            enriched["query_date"] = date_str
            rows.append(_flatten_realtime_weather_record(enriched, regions))

        pagination_token = data.get("paginationToken")
        if not pagination_token:
            break

        params = {"date": date_str, "paginationToken": pagination_token}
        time.sleep(0.2)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_weather_history_cache(
    cache_path: Path,
    history_start_date: pd.Timestamp,
    regions: tuple[str, ...],
    logger=print,
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    today = pd.Timestamp.today().normalize()
    history_start_date = pd.Timestamp(history_start_date).normalize()

    cache_df = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    if not cache_df.empty and "query_date" in cache_df.columns:
        cache_df["query_date"] = pd.to_datetime(cache_df["query_date"], errors="coerce")
        cache_df = cache_df.dropna(subset=["query_date"]).copy()
        cache_df = _dedupe_sort_weather(cache_df)

    metadata = request_json(WEATHER_HISTORY_METADATA_URL)
    raw_cache_dir = cache_path.parent / HISTORICAL_RAW_DIRNAME
    expected_years = _extract_collection_years(metadata)
    dataset_year_pairs = _extract_dataset_year_pairs(metadata)
    if dataset_year_pairs:
        expected_years = {year for year, _ in dataset_year_pairs}

    raw_cached_years = {
        year for year, dataset_id in dataset_year_pairs if _historical_raw_file(raw_cache_dir, year, dataset_id).exists()
    }
    cached_years: set[int] = set()
    if not cache_df.empty:
        cached_years = set(cache_df["query_date"].dt.year.dropna().astype(int).tolist())

    if logger and expected_years:
        logger(f"Historical weather years in collection: {_format_year_list(expected_years)}.")
        logger(f"Historical weather raw files already downloaded: {_format_year_list(raw_cached_years)}.")
        logger(f"Historical weather years still missing raw files: {_format_year_list(expected_years.difference(raw_cached_years))}.")

    cache_min_date = cache_df["query_date"].min().normalize() if not cache_df.empty else pd.NaT
    missing_historical_years = expected_years.difference(cached_years)
    need_historical_rebuild = (
        cache_df.empty
        or pd.isna(cache_min_date)
        or cache_min_date > history_start_date
        or bool(missing_historical_years)
    )

    if need_historical_rebuild:
        if logger:
            logger(
                "Refreshing historical weather cache from data.gov.sg collection 2213 "
                f"starting {history_start_date.date()}."
            )
        historical_df, historical_warnings = _fetch_historical_weather_frames(
            metadata,
            history_start_date,
            raw_cache_dir=raw_cache_dir,
            dataset_year_pairs=dataset_year_pairs,
            logger=logger,
        )
        cache_df = _merge_weather_cache(cache_df, [historical_df], cache_path)
        cache_df["query_date"] = pd.to_datetime(cache_df["query_date"], errors="coerce")
        if logger:
            refreshed_years = set(cache_df["query_date"].dt.year.dropna().astype(int).tolist())
            logger(f"Historical weather years now present in merged cache: {_format_year_list(refreshed_years.intersection(expected_years))}.")
            for warning in historical_warnings:
                logger(f"WARNING: {warning}")

    cached_max_date = cache_df["query_date"].max().normalize() if not cache_df.empty else history_start_date - pd.Timedelta(days=1)
    next_fetch_date = max(history_start_date, cached_max_date + pd.Timedelta(days=1))

    if next_fetch_date <= today:
        if logger:
            logger(
                f"Fetching real-time weather history from {next_fetch_date.date()} to {today.date()}."
            )
        new_frames: list[pd.DataFrame] = []
        for idx, query_date in enumerate(pd.date_range(next_fetch_date, today, freq="D"), start=1):
            daily_df = _fetch_realtime_weather_records_for_day(pd.Timestamp(query_date), regions)
            if not daily_df.empty:
                new_frames.append(daily_df)

            if idx % 14 == 0 and new_frames:
                cache_df = _merge_weather_cache(cache_df, new_frames, cache_path)
                new_frames = []
                if logger:
                    logger(f"Cached weather through {pd.Timestamp(query_date).date()}.")

            time.sleep(0.05)

        if new_frames:
            cache_df = _merge_weather_cache(cache_df, new_frames, cache_path)

    if cache_df.empty:
        raise RuntimeError("Weather cache is empty after historical + real-time loading.")

    cache_df["query_date"] = pd.to_datetime(cache_df["query_date"], errors="coerce")
    cache_df = cache_df.dropna(subset=["query_date"]).copy()
    cache_df = cache_df[cache_df["query_date"] >= history_start_date].reset_index(drop=True)
    cache_df = _dedupe_sort_weather(cache_df)
    cache_df.to_csv(cache_path, index=False)
    return cache_df
