"""Data collection utilities for Casablanca Stock Exchange assets.

This module delegates historical market data retrieval to BVCscrap. It never
generates synthetic stock prices.
"""

from __future__ import annotations

from datetime import date
from importlib import import_module
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = ["date", "ticker", "close", "low", "high", "volume", "variation"]


def _load_pandas():
    """Load pandas lazily and raise a clear error if it is unavailable."""
    try:
        return import_module("pandas")
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to collect and save stock data. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc


def _load_bvcscrap():
    """Load BVCscrap lazily and raise a clear error if it is unavailable."""
    try:
        return import_module("BVCscrap")
    except ImportError as exc:
        raise RuntimeError(
            "BVCscrap is required for BVC data collection but is not installed. "
            "Install it with `pip install BVCscrap` and try again."
        ) from exc


def _normalize_tickers(tickers: str | Iterable[str]) -> list[str]:
    if isinstance(tickers, str):
        ticker_list = [tickers.strip()]
    else:
        try:
            ticker_list = []
            for ticker in tickers:
                if not isinstance(ticker, str):
                    raise ValueError("each ticker must be a string.")
                ticker_list.append(ticker.strip())
        except TypeError as exc:
            raise ValueError("tickers must be a ticker string or an iterable of strings.") from exc

    ticker_list = [ticker for ticker in ticker_list if ticker]
    if not ticker_list:
        raise ValueError("tickers must contain at least one non-empty ticker.")

    return ticker_list


def _validate_date(value: str | None, name: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty YYYY-MM-DD string or None.")

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must use the YYYY-MM-DD format.") from exc


def _validate_dates(start_date: str | None, end_date: str | None) -> None:
    start = _validate_date(start_date, "start_date")
    end = _validate_date(end_date, "end_date")

    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be earlier than or equal to end_date.")


def _standardize_columns(df, ticker: str):
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    has_date_column = any(column.lower() in {"date", "labels"} for column in df.columns)
    if has_date_column:
        df = df.reset_index(drop=True)
    else:
        df = df.reset_index()
        df.columns = [str(column).strip() for column in df.columns]

    rename_map = {
        "date": "date",
        "labels": "date",
        "index": "date",
        "level_0": "date",
        "value": "close",
        "min": "low",
        "max": "high",
        "volume": "volume",
        "variation": "variation",
    }

    columns_to_rename = {}
    for column in df.columns:
        normalized_name = column.strip().lower()
        if normalized_name in rename_map:
            columns_to_rename[column] = rename_map[normalized_name]

    df = df.rename(columns=columns_to_rename)

    if "ticker" not in df.columns:
        df.insert(1 if "date" in df.columns else 0, "ticker", ticker)
    else:
        df["ticker"] = ticker

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise RuntimeError(
            f"BVCscrap data for '{ticker}' is missing required columns: "
            f"{', '.join(missing_columns)}. Expected columns are: "
            f"{', '.join(REQUIRED_COLUMNS)}."
        )

    extra_columns = [column for column in df.columns if column not in REQUIRED_COLUMNS]
    return df[REQUIRED_COLUMNS + extra_columns]


def collect_stock_data(tickers, start_date=None, end_date=None):
    """Collect historical stock data from BVCscrap.

    Parameters
    ----------
    tickers:
        A ticker string or an iterable of ticker strings using BVCscrap notation.
    start_date:
        Optional start date in ``YYYY-MM-DD`` format.
    end_date:
        Optional end date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pandas.DataFrame
        Historical rows with at least these columns: ``date``, ``ticker``,
        ``close``, ``low``, ``high``, ``volume`` and ``variation``.

    Raises
    ------
    ValueError
        If tickers or dates are invalid.
    RuntimeError
        If BVCscrap is not installed, unavailable, or returns unusable data.
    """
    ticker_list = _normalize_tickers(tickers)
    _validate_dates(start_date, end_date)

    bvc = _load_bvcscrap()
    pd = _load_pandas()

    if not hasattr(bvc, "loadata"):
        raise RuntimeError(
            "BVCscrap is installed but does not expose `loadata`. "
            "Install a compatible BVCscrap version."
        )

    collected_frames = []
    for ticker in ticker_list:
        try:
            raw_df = bvc.loadata(ticker, start=start_date, end=end_date)
        except Exception as exc:
            raise RuntimeError(
                f"BVCscrap failed to collect data for '{ticker}'. "
                "Verify the ticker notation, the date range, and internet access."
            ) from exc

        if raw_df is None:
            raise RuntimeError(f"BVCscrap returned no data object for '{ticker}'.")
        if not isinstance(raw_df, pd.DataFrame):
            raise RuntimeError(
                f"BVCscrap returned {type(raw_df).__name__} for '{ticker}', "
                "but a pandas DataFrame was expected."
            )
        if raw_df.empty:
            raise RuntimeError(
                f"BVCscrap returned an empty DataFrame for '{ticker}'. "
                "Verify the ticker notation and date range."
            )

        collected_frames.append(_standardize_columns(raw_df, ticker))

    result = pd.concat(collected_frames, ignore_index=True)
    if result.empty:
        raise RuntimeError("BVCscrap did not return any stock rows.")

    return result


def save_raw_data(df, output_path):
    """Save raw collected data to a CSV file.

    Parameters
    ----------
    df:
        The pandas DataFrame returned by ``collect_stock_data``.
    output_path:
        Destination path, typically ``data/raw/raw_prices.csv``.

    Returns
    -------
    pathlib.Path
        The path where the file was saved.

    Raises
    ------
    ValueError
        If the dataframe or output path is invalid.
    RuntimeError
        If writing the CSV file fails.
    """
    pd = _load_pandas()

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df must not be empty.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "df is missing required columns before saving: "
            f"{', '.join(missing_columns)}."
        )

    if not output_path:
        raise ValueError("output_path must be provided.")

    path = Path(output_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("output_path must be a CSV file path ending with `.csv`.")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to save raw data to '{path}'.") from exc

    return path


__all__ = ["collect_stock_data", "save_raw_data"]
