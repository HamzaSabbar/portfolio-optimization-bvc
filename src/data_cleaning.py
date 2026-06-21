"""Data cleaning utilities for raw Casablanca Stock Exchange prices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_PATH = Path("data") / "processed" / "clean_prices.csv"
COLUMN_RENAME_MAP = {
    "date": "date",
    "ticker": "ticker",
    "value": "close",
    "close": "close",
    "min": "low",
    "low": "low",
    "max": "high",
    "high": "high",
    "volume": "volume",
    "variation": "variation",
}
REQUIRED_COLUMNS = ["date", "ticker", "close", "low", "high", "volume", "variation"]
PRICE_VOLUME_COLUMNS = ["close", "low", "high", "volume"]
NUMERIC_COLUMNS = PRICE_VOLUME_COLUMNS + ["variation"]


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed_columns = {}
    for column in df.columns:
        normalized = str(column).strip().lower()
        renamed_columns[column] = COLUMN_RENAME_MAP.get(normalized, str(column).strip())

    standardized = df.rename(columns=renamed_columns)
    if standardized.columns.duplicated().any():
        duplicates = sorted(set(standardized.columns[standardized.columns.duplicated()]))
        raise ValueError(f"Duplicate columns after standardization: {', '.join(duplicates)}.")

    return standardized


def _validate_prices_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df must not be empty.")


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw price data before feature engineering.

    The function standardizes BVCscrap column names, converts dates and numeric
    fields, removes unusable rows, deduplicates observations, and sorts the
    result chronologically by ticker.

    Parameters
    ----------
    df:
        Raw prices DataFrame with columns from BVCscrap or the Phase 1 export.

    Returns
    -------
    pandas.DataFrame
        Cleaned prices sorted by ``ticker`` and ``date``.

    Raises
    ------
    ValueError
        If the input is invalid or required columns are missing.
    """
    _validate_prices_dataframe(df)

    cleaned = _standardize_column_names(df.copy())
    _validate_required_columns(cleaned)

    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce", dayfirst=True)
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["date", "ticker", "close"])
    cleaned = cleaned[cleaned["close"] > 0]
    cleaned = cleaned.drop_duplicates(subset=["date", "ticker"], keep="last")
    cleaned = cleaned.sort_values(["ticker", "date"]).reset_index(drop=True)

    return cleaned


def save_clean_prices(df: pd.DataFrame, output_path=DEFAULT_OUTPUT_PATH) -> Path:
    """Save cleaned prices to a CSV file.

    Parameters
    ----------
    df:
        Cleaned prices DataFrame returned by ``clean_prices``.
    output_path:
        Destination CSV path, typically ``data/processed/clean_prices.csv``.

    Returns
    -------
    pathlib.Path
        Path where the cleaned data was saved.

    Raises
    ------
    ValueError
        If the dataframe or output path is invalid.
    RuntimeError
        If writing the CSV fails.
    """
    _validate_prices_dataframe(df)
    _validate_required_columns(df)

    if not output_path:
        raise ValueError("output_path must be provided.")

    path = Path(output_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("output_path must be a CSV file path ending with `.csv`.")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to save cleaned prices to '{path}'.") from exc

    print(f"Saved cleaned prices to: {path}")
    print(f"Output shape: {df.shape}")
    print("Rows per ticker:")
    print(df.groupby("ticker").size().sort_index().to_string())

    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean raw BVC stock prices and save data/processed/clean_prices.csv."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input raw CSV path, for example data\\raw\\raw_prices.csv.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output cleaned CSV path. Defaults to data\\processed\\clean_prices.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 2 data cleaning command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Data cleaning failed: input file does not exist: {input_path}", file=sys.stderr)
        return 1
    if input_path.stat().st_size == 0:
        print(f"Data cleaning failed: input file is empty: {input_path}", file=sys.stderr)
        return 1

    try:
        raw_prices = pd.read_csv(input_path)
        cleaned_prices = clean_prices(raw_prices)
        save_clean_prices(cleaned_prices, args.output)
    except Exception as exc:
        print(f"Data cleaning failed: {exc}", file=sys.stderr)
        return 1

    return 0


__all__ = ["clean_prices", "save_clean_prices"]


if __name__ == "__main__":
    raise SystemExit(main())
