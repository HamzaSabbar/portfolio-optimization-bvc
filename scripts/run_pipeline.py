"""Run the full BVC portfolio optimization data pipeline.

The script starts from the raw Phase 1 export and regenerates every downstream
file used by the final Streamlit app. It does not collect data itself, generate
fake data, or change the financial methodology.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import scoring
from src.data_cleaning import clean_prices, save_clean_prices
from src.features import (
    MIN_FEATURE_OBSERVATIONS_PER_TICKER,
    add_target_variable,
    build_financial_features,
    calculate_daily_returns,
    save_features,
    save_returns,
    save_scoring_dataset,
)


RAW_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "raw_prices.csv"
CLEAN_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "clean_prices.csv"
RETURNS_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "returns.csv"
FEATURES_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
SCORING_DATASET_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "scoring_dataset.csv"
EXCLUDED_TICKERS_OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "excluded_tickers_report.csv"
LATEST_SCORES_OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "latest_scores.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"
SCALER_OUTPUT_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

EXCLUDED_REPORT_COLUMNS = ["ticker", "stage", "reason", "rows_available"]


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} file is empty: {path}")
    return pd.read_csv(path)


def _require_ticker_column(df: pd.DataFrame, label: str) -> None:
    if "ticker" not in df.columns:
        raise ValueError(f"{label} must contain a ticker column.")


def _ticker_set(df: pd.DataFrame) -> set[str]:
    _require_ticker_column(df, "DataFrame")
    return set(df["ticker"].dropna().astype(str))


def _ticker_count(df: pd.DataFrame) -> int:
    return len(_ticker_set(df))


def _wide_ticker_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if str(column) != "date"]


def _wide_ticker_count(df: pd.DataFrame) -> int:
    return len(_wide_ticker_columns(df))


def _append_missing_ticker_records(
    records: list[dict],
    before_counts: pd.Series,
    after_tickers: set[str],
    stage: str,
    reason: str,
) -> None:
    for ticker, row_count in before_counts.items():
        ticker_name = str(ticker)
        if ticker_name not in after_tickers:
            records.append(
                {
                    "ticker": ticker_name,
                    "stage": stage,
                    "reason": reason,
                    "rows_available": int(row_count),
                }
            )


def _filter_eligible_clean_prices(
    clean_prices_df: pd.DataFrame,
    records: list[dict],
) -> pd.DataFrame:
    clean_counts = clean_prices_df.groupby("ticker").size().sort_index()
    short_counts = clean_counts[clean_counts < MIN_FEATURE_OBSERVATIONS_PER_TICKER]

    for ticker, row_count in short_counts.items():
        records.append(
            {
                "ticker": str(ticker),
                "stage": "features",
                "reason": (
                    f"Fewer than {MIN_FEATURE_OBSERVATIONS_PER_TICKER} cleaned observations; "
                    "rolling financial features need at least 70+ observations."
                ),
                "rows_available": int(row_count),
            }
        )

    if short_counts.empty:
        return clean_prices_df.copy()

    eligible = clean_prices_df[
        ~clean_prices_df["ticker"].astype(str).isin(short_counts.index.astype(str))
    ].copy()
    if eligible.empty:
        raise ValueError(
            "No tickers have enough observations for feature engineering. "
            f"At least {MIN_FEATURE_OBSERVATIONS_PER_TICKER} observations are required."
        )

    return eligible


def _save_excluded_report(records: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(records, columns=EXCLUDED_REPORT_COLUMNS)
    report = report.drop_duplicates(subset=["ticker", "stage", "reason"], keep="first")
    report = report.sort_values(["stage", "ticker"]).reset_index(drop=True)
    report.to_csv(output_path, index=False)
    print(f"Saved excluded tickers report to: {output_path}")
    print(f"Excluded tickers: {report['ticker'].nunique() if not report.empty else 0}")
    return output_path


def _log_count(label: str, count: int) -> None:
    print(f"{label}: {count}")


def _warn_if_three_scores_with_large_raw(raw_ticker_count: int, latest_score_count: int) -> None:
    if raw_ticker_count > 3 and latest_score_count == 3:
        print(
            "WARNING: latest_scores.csv contains only 3 tickers while "
            f"raw_prices.csv contains {raw_ticker_count}. "
            "This usually means downstream files were generated from an old "
            "test subset. Rerun this pipeline after full data collection."
        )


def run_pipeline(args: argparse.Namespace) -> None:
    """Regenerate processed data, trained model, and latest scores from raw data."""
    raw_prices = _read_csv(args.raw_input, "Raw prices")
    _require_ticker_column(raw_prices, "Raw prices")
    raw_ticker_count = _ticker_count(raw_prices)
    _log_count("Number of raw tickers", raw_ticker_count)

    excluded_records: list[dict] = []

    clean_df = clean_prices(raw_prices)
    raw_counts = raw_prices.groupby(raw_prices["ticker"].astype(str)).size().sort_index()
    _append_missing_ticker_records(
        excluded_records,
        raw_counts,
        _ticker_set(clean_df),
        stage="cleaning",
        reason="No valid rows remain after cleaning date, ticker, close, and positive close.",
    )
    save_clean_prices(clean_df, args.clean_output)
    cleaned_ticker_count = _ticker_count(clean_df)
    _log_count("Number of cleaned tickers", cleaned_ticker_count)

    eligible_clean_df = _filter_eligible_clean_prices(clean_df, excluded_records)
    if _ticker_count(eligible_clean_df) != cleaned_ticker_count:
        _log_count("Number of eligible cleaned tickers", _ticker_count(eligible_clean_df))

    returns_df = calculate_daily_returns(eligible_clean_df)
    save_returns(returns_df, args.returns_output)
    return_tickers = {
        ticker for ticker in _wide_ticker_columns(returns_df) if returns_df[ticker].notna().any()
    }
    _log_count("Number of tickers with returns", len(return_tickers))

    features_df = build_financial_features(eligible_clean_df)
    eligible_counts = eligible_clean_df.groupby(eligible_clean_df["ticker"].astype(str)).size()
    _append_missing_ticker_records(
        excluded_records,
        eligible_counts,
        _ticker_set(features_df),
        stage="features",
        reason="No complete feature rows remain after rolling indicators and missing-value filtering.",
    )
    save_features(features_df, args.features_output)
    feature_ticker_count = _ticker_count(features_df)
    _log_count("Number of tickers with features", feature_ticker_count)

    scoring_dataset = add_target_variable(
        features_df,
        horizon_days=args.horizon_days,
        threshold=args.threshold,
    )
    feature_counts = features_df.groupby(features_df["ticker"].astype(str)).size()
    _append_missing_ticker_records(
        excluded_records,
        feature_counts,
        _ticker_set(scoring_dataset),
        stage="target",
        reason="No rows remain after future-return target construction.",
    )
    save_scoring_dataset(scoring_dataset, args.scoring_output)
    scoring_ticker_count = _ticker_count(scoring_dataset)
    _log_count("Number of tickers in scoring_dataset", scoring_ticker_count)

    scoring.MODEL_OUTPUT_PATH = args.model_output
    scoring.SCALER_OUTPUT_PATH = args.scaler_output
    scoring.LATEST_SCORES_OUTPUT_PATH = args.latest_scores_output
    model, scaler, metrics = scoring.train_scoring_model(
        scoring_dataset,
        test_size=args.test_size,
    )
    latest_scores = scoring.score_latest_stocks(features_df, model, scaler)
    feature_counts = features_df.groupby(features_df["ticker"].astype(str)).size()
    _append_missing_ticker_records(
        excluded_records,
        feature_counts,
        _ticker_set(latest_scores),
        stage="scoring",
        reason="No valid latest feature row remains for model scoring.",
    )

    latest_score_count = _ticker_count(latest_scores)
    _log_count("Number of tickers in latest_scores", latest_score_count)

    displayed_tickers = sorted(set(latest_scores["ticker"].astype(str)).intersection(return_tickers))
    _log_count("Number of tickers displayed in Streamlit", len(displayed_tickers))
    _warn_if_three_scores_with_large_raw(raw_ticker_count, latest_score_count)

    _save_excluded_report(excluded_records, args.excluded_output)

    print(f"Saved model to: {args.model_output}")
    print(f"Saved scaler to: {args.scaler_output}")
    print(f"Saved latest scores to: {args.latest_scores_output}")
    print("Model metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline from raw BVC prices to latest scores."
    )
    parser.add_argument("--raw-input", type=Path, default=RAW_INPUT_PATH)
    parser.add_argument("--clean-output", type=Path, default=CLEAN_OUTPUT_PATH)
    parser.add_argument("--returns-output", type=Path, default=RETURNS_OUTPUT_PATH)
    parser.add_argument("--features-output", type=Path, default=FEATURES_OUTPUT_PATH)
    parser.add_argument("--scoring-output", type=Path, default=SCORING_DATASET_OUTPUT_PATH)
    parser.add_argument("--excluded-output", type=Path, default=EXCLUDED_TICKERS_OUTPUT_PATH)
    parser.add_argument("--latest-scores-output", type=Path, default=LATEST_SCORES_OUTPUT_PATH)
    parser.add_argument("--model-output", type=Path, default=MODEL_OUTPUT_PATH)
    parser.add_argument("--scaler-output", type=Path, default=SCALER_OUTPUT_PATH)
    parser.add_argument("--horizon-days", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line pipeline."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        run_pipeline(args)
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
