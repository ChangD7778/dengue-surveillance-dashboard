"""
Run this script once to find the best ARIMA orders for your dengue data.

How to run in VSCode:
  1. Open this file
  2. Press Ctrl+F5  (Run Without Debugging)
     OR open a terminal (Ctrl+`) and type:
     python "tune_arima.py"

Output: prints the best (p,d,q)(P,D,Q,52) orders and AIC to the terminal.
"""

import subprocess
import sys

# Install pmdarima if not already present
try:
    import pmdarima
except ImportError:
    print("Installing pmdarima...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pmdarima"])
    import pmdarima

import pandas as pd
from pmdarima import auto_arima
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARIMAX_EXOG_COLUMNS = [
    "avg_temp_c",
    "avg_relative_humidity_pct",
    "avg_wind_speed",
    "warm_day_share",
]

# ------------------------------------------------------------------
# Load the pre-merged weekly dengue + weather data
# ------------------------------------------------------------------
data_path = BASE_DIR / "weekly_dengue_weather_base.csv"
if not data_path.exists():
    print(f"ERROR: {data_path} not found.")
    print("Make sure you have run the build script at least once so the CSV exists.")
    sys.exit(1)

df = pd.read_csv(data_path, parse_dates=["week_start"])
df = df.sort_values("week_start").reset_index(drop=True)

# Drop rows where the target or any exog column is missing
required = ["Total Dengue Cases"] + ARIMAX_EXOG_COLUMNS
df = df.dropna(subset=required)

y    = df["Total Dengue Cases"].astype(float)
exog = df[ARIMAX_EXOG_COLUMNS].astype(float)

print(f"\nTraining rows: {len(y)}")
print(f"Date range:    {df['week_start'].min().date()} → {df['week_start'].max().date()}")

# ------------------------------------------------------------------
# 1. SARIMA search (no exogenous variables)
# ------------------------------------------------------------------
print("\n" + "="*60)
print("Searching SARIMA orders (no weather inputs)...")
print("This may take a few minutes.")
print("="*60)

sarima_result = auto_arima(
    y,
    m=52,                   # yearly seasonal period (weekly data)
    seasonal=True,
    d=1,  D=1,              # fix differencing — standard for this type of series
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    start_P=0, max_P=1,
    start_Q=0, max_Q=1,
    information_criterion="aic",
    stepwise=True,          # stepwise=True is much faster than a full grid search
    trace=True,             # prints each candidate model as it's tested
    error_action="ignore",
    suppress_warnings=True,
)

print("\n--- SARIMA result ---")
print(f"  Best order:          {sarima_result.order}")
print(f"  Best seasonal order: {sarima_result.seasonal_order}")
print(f"  AIC:                 {sarima_result.aic():.2f}")

# ------------------------------------------------------------------
# 2. ARIMAX search (with weather exogenous variables)
# ------------------------------------------------------------------
print("\n" + "="*60)
print("Searching ARIMAX orders (with weather inputs)...")
print("="*60)

arimax_result = auto_arima(
    y,
    exogenous=exog,
    m=52,
    seasonal=True,
    d=1, D=1,
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    start_P=0, max_P=1,
    start_Q=0, max_Q=1,
    information_criterion="aic",
    stepwise=True,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
)

print("\n--- ARIMAX result ---")
print(f"  Best order:          {arimax_result.order}")
print(f"  Best seasonal order: {arimax_result.seasonal_order}")
print(f"  AIC:                 {arimax_result.aic():.2f}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY — copy these values into nea_dashboard_static.py")
print("="*60)
print(f"\n  # From SARIMA search")
print(f"  FIXED_ORDER          = {sarima_result.order}")
print(f"  FIXED_SEASONAL_ORDER = {sarima_result.seasonal_order}")
print(f"\n  # From ARIMAX search (use these instead if AIC is lower)")
print(f"  FIXED_ORDER          = {arimax_result.order}")
print(f"  FIXED_SEASONAL_ORDER = {arimax_result.seasonal_order}")
print()
