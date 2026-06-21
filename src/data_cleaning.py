"""Data cleaning utilities for raw Casablanca Stock Exchange prices."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


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
NUMERIC_COLUMNS = ["close", "low", "high", "volume", "variation"]


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
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned[cleaned["close"] > 0]
    cleaned = cleaned.sort_values(["ticker", "date"]).reset_index(drop=True)

    return cleaned


def save_clean_prices(df: pd.DataFrame, output_path) -> Path:
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

    return path


__all__ = ["clean_prices", "save_clean_prices"]
