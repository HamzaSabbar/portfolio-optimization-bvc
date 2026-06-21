"""Data collection utilities for Casablanca Stock Exchange assets.

This module delegates historical market data retrieval to BVCscrap. It never
generates synthetic stock prices.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from importlib import import_module
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = ["date", "ticker", "close", "low", "high", "volume", "variation"]
DEFAULT_RAW_OUTPUT_PATH = Path("data") / "raw" / "raw_prices.csv"
DEFAULT_REPORT_OUTPUT_PATH = Path("data") / "outputs" / "collection_report.csv"
DEFAULT_TICKERS_FILE_PATH = Path("config") / "bvc_tickers.txt"
NON_COMPANY_SYMBOLS = {"MASI", "MSI20"}
REPORT_COLUMNS = [
    "ticker",
    "status",
    "rows_collected",
    "min_date",
    "max_date",
    "error_message",
]


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


def _read_tickers_file(path: str | Path) -> list[str]:
    """Read one ticker per line from a text file."""
    tickers_path = Path(path)
    if not tickers_path.is_file() or tickers_path.stat().st_size == 0:
        raise ValueError(f"tickers file is missing or empty: {tickers_path}")

    tickers = []
    for line in tickers_path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        tickers.append(value)

    return _normalize_tickers(tickers)


def _filter_company_tickers(tickers: Iterable[str]) -> list[str]:
    """Remove non-company symbols such as market indexes and de-duplicate names."""
    filtered_tickers = []
    seen = set()
    for ticker in tickers:
        ticker = str(ticker).strip()
        if not ticker or ticker in NON_COMPANY_SYMBOLS or ticker in seen:
            continue
        seen.add(ticker)
        filtered_tickers.append(ticker)

    if not filtered_tickers:
        raise ValueError("No company tickers were found.")

    return filtered_tickers


def get_all_available_tickers() -> list[str]:
    """Return all available BVC company tickers from BVCscrap or fallback config."""
    bvc = _load_bvcscrap()

    try:
        if hasattr(bvc, "notation") and callable(bvc.notation):
            tickers = _filter_company_tickers(bvc.notation())
            if tickers:
                return tickers
    except Exception:
        pass

    return _filter_company_tickers(_read_tickers_file(DEFAULT_TICKERS_FILE_PATH))


def _resolve_tickers(
    tickers: list[str] | None = None,
    tickers_file: str | Path | None = None,
    collect_all: bool = False,
) -> list[str]:
    selected_sources = sum(
        [
            bool(tickers),
            bool(tickers_file),
            bool(collect_all),
        ]
    )
    if selected_sources > 1:
        raise ValueError("Use only one of --all, --tickers, or --tickers-file.")

    if tickers:
        return _normalize_tickers(tickers)
    if tickers_file:
        return _read_tickers_file(tickers_file)

    return get_all_available_tickers()


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


def _empty_raw_dataframe():
    pd = _load_pandas()
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def _report_record(
    ticker: str,
    status: str,
    rows_collected: int = 0,
    min_date: str = "",
    max_date: str = "",
    error_message: str = "",
) -> dict:
    return {
        "ticker": ticker,
        "status": status,
        "rows_collected": rows_collected,
        "min_date": min_date,
        "max_date": max_date,
        "error_message": error_message,
    }


def _collect_single_ticker(bvc, pd, ticker: str, start_date=None, end_date=None):
    raw_df = bvc.loadata(ticker, start=start_date, end=end_date)
    if raw_df is None:
        raise RuntimeError("BVCscrap returned no data object.")
    if not isinstance(raw_df, pd.DataFrame):
        raise RuntimeError(
            f"BVCscrap returned {type(raw_df).__name__}, but a pandas DataFrame was expected."
        )
    if raw_df.empty:
        raise RuntimeError("BVCscrap returned an empty DataFrame.")

    return _standardize_columns(raw_df, ticker)


def collect_stock_data_with_report(tickers=None, start_date=None, end_date=None):
    """Collect BVC stock data and return successful rows with a per-ticker report."""
    ticker_list = _resolve_tickers(tickers=tickers)
    _validate_dates(start_date, end_date)

    bvc = _load_bvcscrap()
    pd = _load_pandas()

    if not hasattr(bvc, "loadata"):
        raise RuntimeError(
            "BVCscrap is installed but does not expose `loadata`. "
            "Install a compatible BVCscrap version."
        )

    collected_frames = []
    report_records = []

    for ticker in ticker_list:
        try:
            ticker_df = _collect_single_ticker(
                bvc,
                pd,
                ticker,
                start_date=start_date,
                end_date=end_date,
            )
            min_date, max_date = _date_range_for_display(ticker_df)
            rows_collected = len(ticker_df)
            collected_frames.append(ticker_df)
            report_records.append(
                _report_record(
                    ticker=ticker,
                    status="success",
                    rows_collected=rows_collected,
                    min_date=min_date,
                    max_date=max_date,
                )
            )
            print(
                f"{ticker}: success | rows={rows_collected} | "
                f"date_range={min_date} to {max_date}",
                flush=True,
            )
        except Exception as exc:
            error_message = str(exc)
            report_records.append(
                _report_record(
                    ticker=ticker,
                    status="failure",
                    error_message=error_message,
                )
            )
            print(
                f"{ticker}: failure | rows=0 | date_range=unknown | "
                f"error={error_message}",
                flush=True,
            )

    if collected_frames:
        result = pd.concat(collected_frames, ignore_index=True)
    else:
        result = _empty_raw_dataframe()

    report = pd.DataFrame(report_records, columns=REPORT_COLUMNS)
    return result, report


def collect_stock_data(tickers=None, start_date=None, end_date=None):
    """Collect historical stock data from BVCscrap.

    Parameters
    ----------
    tickers:
        Optional ticker string or iterable of ticker strings using BVCscrap
        notation. If omitted, all available BVC company tickers are collected.
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
    result, _ = collect_stock_data_with_report(tickers, start_date=start_date, end_date=end_date)
    if result.empty:
        raise RuntimeError("BVCscrap did not return any stock rows.")

    return result


def save_raw_data(df, output_path=DEFAULT_RAW_OUTPUT_PATH):
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


def save_collection_report(report_df, output_path=DEFAULT_REPORT_OUTPUT_PATH):
    """Save the per-ticker collection report to CSV."""
    pd = _load_pandas()

    if not isinstance(report_df, pd.DataFrame):
        raise ValueError("report_df must be a pandas DataFrame.")
    if report_df.empty:
        raise ValueError("report_df must not be empty.")

    missing_columns = [column for column in REPORT_COLUMNS if column not in report_df.columns]
    if missing_columns:
        raise ValueError(
            "report_df is missing required columns: "
            f"{', '.join(missing_columns)}."
        )

    path = Path(output_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("output_path must be a CSV file path ending with `.csv`.")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        report_df[REPORT_COLUMNS].to_csv(path, index=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to save collection report to '{path}'.") from exc

    return path


def _validate_collected_data(df) -> None:
    pd = _load_pandas()

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Collected data must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Collected data is empty. No CSV was produced.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Collected data is missing required columns: "
            f"{', '.join(missing_columns)}."
        )


def _date_range_for_display(df) -> tuple[str, str]:
    pd = _load_pandas()
    parsed_dates = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    if parsed_dates.notna().any():
        return (
            parsed_dates.min().strftime("%Y-%m-%d"),
            parsed_dates.max().strftime("%Y-%m-%d"),
        )

    date_values = df["date"].dropna().astype(str)
    if date_values.empty:
        return "unknown", "unknown"
    return date_values.min(), date_values.max()


def _print_collection_summary(df, saved_path: Path) -> None:
    _validate_collected_data(df)
    min_date, max_date = _date_range_for_display(df)
    rows_per_ticker = df.groupby("ticker").size().sort_index()

    print(f"Saved raw data to: {saved_path}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of tickers: {df['ticker'].nunique()}")
    print(f"Date range: {min_date} to {max_date}")
    print("Rows per ticker:")
    for ticker, row_count in rows_per_ticker.items():
        print(f"  {ticker}: {row_count}")
    print(f"CSV columns: {', '.join(REQUIRED_COLUMNS)}")


def _print_report_summary(report_df, report_path: Path) -> None:
    status_counts = report_df["status"].value_counts().to_dict()
    successes = status_counts.get("success", 0)
    failures = status_counts.get("failure", 0)
    print(f"Saved collection report to: {report_path}")
    print(f"Successful tickers: {successes}")
    print(f"Failed tickers: {failures}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Collect raw BVC stock data with BVCscrap and save it as CSV."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Optional ticker names using BVCscrap notation for testing a small subset.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Collect all available BVC company tickers.",
    )
    parser.add_argument(
        "--tickers-file",
        default=None,
        help="Optional text file with one BVC ticker per line.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RAW_OUTPUT_PATH),
        help="CSV output path. Defaults to data/raw/raw_prices.csv.",
    )
    parser.add_argument(
        "--report-output",
        default=str(DEFAULT_REPORT_OUTPUT_PATH),
        help="Collection report output path. Defaults to data/outputs/collection_report.csv.",
    )
    return parser.parse_args()


def _main():
    args = _parse_args()
    try:
        ticker_list = _resolve_tickers(
            tickers=args.tickers,
            tickers_file=args.tickers_file,
            collect_all=args.all,
        )
        raw_data, collection_report = collect_stock_data_with_report(
            ticker_list,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        report_path = save_collection_report(collection_report, args.report_output)

        if raw_data.empty:
            _print_report_summary(collection_report, report_path)
            raise RuntimeError("No ticker was collected successfully. No raw CSV was produced.")

        _validate_collected_data(raw_data)
        saved_path = save_raw_data(raw_data, args.output)
        _print_collection_summary(raw_data, saved_path)
        _print_report_summary(collection_report, report_path)
    except (RuntimeError, ValueError) as exc:
        print(f"Data collection failed: {exc}", file=sys.stderr)
        print(
            "How to fix it: make sure BVCscrap is installed with "
            "`python -m pip install -r requirements.txt`, verify the ticker symbols, "
            "check the date range, and confirm that the BVC source website is reachable.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


__all__ = [
    "collect_stock_data",
    "collect_stock_data_with_report",
    "get_all_available_tickers",
    "save_collection_report",
    "save_raw_data",
]


if __name__ == "__main__":
    _main()
