"""Validate the raw stock prices CSV without modifying it."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raw_prices.csv"
REQUIRED_COLUMN_GROUPS = {
    "date": ("date",),
    "ticker": ("ticker",),
    "close price": ("close", "Value"),
    "volume": ("volume", "Volume"),
}
MIN_ROWS_PER_TICKER = 100
MIN_GLOBAL_ROWS = 300


def _has_any_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> bool:
    """Return True when at least one candidate column exists in the DataFrame."""
    return any(column in df.columns for column in candidates)


def _print_required_column_status(df: pd.DataFrame) -> list[str]:
    """Print required column checks and return missing logical column names."""
    missing = []
    print("\nRequired column checks:")

    for label, candidates in REQUIRED_COLUMN_GROUPS.items():
        if _has_any_column(df, candidates):
            matched = [column for column in candidates if column in df.columns]
            print(f"- OK: {label} ({', '.join(matched)})")
        else:
            missing.append(label)
            print(f"- MISSING: {label} (expected one of: {', '.join(candidates)})")

    return missing


def main() -> int:
    """Run validation checks for data/raw/raw_prices.csv."""
    print(f"Raw data file: {RAW_DATA_PATH}")

    if not RAW_DATA_PATH.exists():
        print("ERROR: data/raw/raw_prices.csv does not exist.")
        return 1

    if RAW_DATA_PATH.stat().st_size == 0:
        print("ERROR: data/raw/raw_prices.csv is empty.")
        return 1

    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except pd.errors.EmptyDataError:
        print("ERROR: data/raw/raw_prices.csv has no readable rows.")
        return 1

    if df.empty:
        print("ERROR: data/raw/raw_prices.csv loaded successfully but contains no rows.")
        return 1

    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

    missing_required = _print_required_column_status(df)
    if missing_required:
        print("\nERROR: required raw data columns are missing.")
        return 1

    print("\nFirst rows:")
    print(df.head().to_string(index=False))

    dates = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    valid_dates = dates.dropna()
    if valid_dates.empty:
        print("\nWARNING: no valid dates could be parsed from the date column.")
    else:
        print(f"\nDate range: {valid_dates.min().date()} to {valid_dates.max().date()}")

    rows_per_ticker = df.groupby("ticker", dropna=False).size().sort_index()
    print("\nRows per ticker:")
    print(rows_per_ticker.to_string())

    if len(df) < MIN_GLOBAL_ROWS:
        print(f"\nWARNING: global dataset has fewer than {MIN_GLOBAL_ROWS} rows.")

    small_tickers = rows_per_ticker[rows_per_ticker < MIN_ROWS_PER_TICKER]
    if not small_tickers.empty:
        print(f"\nWARNING: these tickers have fewer than {MIN_ROWS_PER_TICKER} rows:")
        print(small_tickers.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
