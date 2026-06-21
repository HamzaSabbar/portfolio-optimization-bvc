"""Feature engineering utilities for portfolio optimization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RETURNS_OUTPUT_PATH = Path("data") / "processed" / "returns.csv"
DEFAULT_FEATURES_OUTPUT_PATH = Path("data") / "processed" / "features.csv"
DEFAULT_SCORING_DATASET_OUTPUT_PATH = Path("data") / "processed" / "scoring_dataset.csv"
MIN_FEATURE_OBSERVATIONS_PER_TICKER = 70
REQUIRED_PRICE_COLUMNS = ["date", "ticker", "close"]
REQUIRED_FEATURE_INPUT_COLUMNS = ["date", "ticker", "close", "volume"]
FEATURE_COLUMNS = [
    "past_return_20d",
    "momentum_20d",
    "volatility_20d",
    "rsi_14d",
    "macd",
    "macd_signal",
    "moving_average_20d",
    "moving_average_50d",
    "average_volume_20d",
    "drawdown",
]
TARGET_COLUMN = "target"
MIN_POSSIBLE_RETURN = -1.0
TARGET_IMBALANCE_WARNING_THRESHOLD = 0.05


def _is_missing_ticker(tickers: pd.Series) -> pd.Series:
    return tickers.isna() | tickers.astype(str).str.strip().isin(["", "nan", "None"])


def _print_price_quality_report(invalid_rows: pd.DataFrame, context: str) -> None:
    if invalid_rows.empty:
        return

    print(
        f"WARNING: removed {len(invalid_rows)} rows with invalid required market data "
        f"before {context}."
    )
    if "ticker" in invalid_rows.columns:
        print("Invalid price rows per ticker:")
        print(invalid_rows.groupby("ticker", dropna=False).size().sort_index().to_string())


def _print_impossible_returns_report(impossible_rows: pd.DataFrame) -> None:
    if impossible_rows.empty:
        return

    print(
        "WARNING: removed impossible daily returns caused by invalid price data. "
        f"Rows removed: {len(impossible_rows)}"
    )
    print("Impossible returns per ticker:")
    print(impossible_rows.groupby("ticker", dropna=False).size().sort_index().to_string())


def _prepare_price_input_for_returns(clean_prices_df: pd.DataFrame) -> pd.DataFrame:
    _validate_clean_prices(clean_prices_df)

    prices = clean_prices_df[REQUIRED_PRICE_COLUMNS].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")

    invalid_mask = (
        prices["date"].isna()
        | _is_missing_ticker(prices["ticker"])
        | prices["close"].isna()
        | (prices["close"] <= 0)
    )
    invalid_rows = prices.loc[invalid_mask, REQUIRED_PRICE_COLUMNS]
    _print_price_quality_report(invalid_rows, "daily return calculation")

    prices = prices.loc[~invalid_mask].copy()
    prices["ticker"] = prices["ticker"].astype(str).str.strip()

    if prices.empty:
        raise ValueError("No valid price rows remain after date, ticker, and close validation.")
    if prices.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("clean_prices_df contains duplicate rows for the same date and ticker.")

    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def _validate_returns_output(returns_df: pd.DataFrame) -> None:
    if not isinstance(returns_df, pd.DataFrame):
        raise ValueError("returns_df must be a pandas DataFrame.")
    if returns_df.empty:
        raise ValueError("returns_df must not be empty.")

    numeric_returns = returns_df.apply(pd.to_numeric, errors="coerce")
    return_values = numeric_returns.to_numpy(dtype=float)
    finite_values = return_values[~np.isnan(return_values)]
    if finite_values.size == 0:
        raise ValueError("returns_df contains no numeric return values.")
    if not np.isfinite(finite_values).all():
        raise ValueError("returns_df contains infinite or non-finite return values.")
    if (finite_values <= MIN_POSSIBLE_RETURN).any():
        raise ValueError("returns_df contains impossible returns less than or equal to -100%.")


def _validate_feature_output(features_df: pd.DataFrame) -> None:
    required_columns = ["date", "ticker", "close", *FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in features_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")

    numeric_columns = ["close", *FEATURE_COLUMNS]
    numeric_features = features_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    finite_values = numeric_features.to_numpy(dtype=float).ravel()
    if not np.isfinite(finite_values).all():
        raise ValueError("features_df contains missing, infinite, or non-finite feature values.")

    impossible_masks = {
        "close": numeric_features["close"] <= 0,
        "past_return_20d": numeric_features["past_return_20d"] <= MIN_POSSIBLE_RETURN,
        "volatility_20d": numeric_features["volatility_20d"] < 0,
        "rsi_14d": ~numeric_features["rsi_14d"].between(0, 100),
        "moving_average_20d": numeric_features["moving_average_20d"] <= 0,
        "moving_average_50d": numeric_features["moving_average_50d"] <= 0,
        "average_volume_20d": numeric_features["average_volume_20d"] < 0,
        "drawdown": (numeric_features["drawdown"] < MIN_POSSIBLE_RETURN)
        | (numeric_features["drawdown"] > 0),
    }
    impossible_counts = {
        column: int(mask.sum()) for column, mask in impossible_masks.items() if mask.any()
    }
    if impossible_counts:
        formatted_counts = ", ".join(
            f"{column}={count}" for column, count in impossible_counts.items()
        )
        raise ValueError(f"features_df contains impossible feature values: {formatted_counts}.")


def _feature_distribution_summary(features_df: pd.DataFrame) -> pd.DataFrame:
    distribution = features_df[FEATURE_COLUMNS].agg(["min", "max", "mean", "median"]).T
    return distribution[["min", "max", "mean", "median"]]


def _print_feature_diagnostics(features_df: pd.DataFrame) -> None:
    print("Rows per ticker after feature generation:")
    print(features_df.groupby("ticker").size().sort_index().to_string())
    print("Feature distributions:")
    print(_feature_distribution_summary(features_df).to_string())


def _validate_clean_prices(clean_prices_df: pd.DataFrame) -> None:
    if not isinstance(clean_prices_df, pd.DataFrame):
        raise ValueError("clean_prices_df must be a pandas DataFrame.")
    if clean_prices_df.empty:
        raise ValueError("clean_prices_df must not be empty.")

    missing_columns = [
        column for column in REQUIRED_PRICE_COLUMNS if column not in clean_prices_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")


def _validate_feature_input(clean_prices_df: pd.DataFrame) -> None:
    if not isinstance(clean_prices_df, pd.DataFrame):
        raise ValueError("clean_prices_df must be a pandas DataFrame.")
    if clean_prices_df.empty:
        raise ValueError("clean_prices_df must not be empty.")

    missing_columns = [
        column
        for column in REQUIRED_FEATURE_INPUT_COLUMNS
        if column not in clean_prices_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")


def _prepare_feature_input(clean_prices_df: pd.DataFrame) -> pd.DataFrame:
    _validate_feature_input(clean_prices_df)

    prices = clean_prices_df[REQUIRED_FEATURE_INPUT_COLUMNS].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce")
    invalid_mask = (
        prices["date"].isna()
        | _is_missing_ticker(prices["ticker"])
        | prices["close"].isna()
        | prices["volume"].isna()
        | (prices["close"] <= 0)
        | (prices["volume"] < 0)
    )
    invalid_rows = prices.loc[invalid_mask, REQUIRED_FEATURE_INPUT_COLUMNS]
    _print_price_quality_report(invalid_rows, "financial feature generation")
    prices = prices.loc[~invalid_mask].copy()
    prices["ticker"] = prices["ticker"].astype(str).str.strip()

    if prices.empty:
        raise ValueError("No valid price rows remain after date, close, and volume conversion.")
    if prices.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("clean_prices_df contains duplicate rows for the same date and ticker.")

    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def _validate_feature_history(prices: pd.DataFrame) -> None:
    rows_per_ticker = prices.groupby("ticker").size().sort_index()
    short_tickers = rows_per_ticker[rows_per_ticker < MIN_FEATURE_OBSERVATIONS_PER_TICKER]
    if not short_tickers.empty:
        ticker_counts = ", ".join(
            f"{ticker}={count}" for ticker, count in short_tickers.items()
        )
        raise ValueError(
            "Insufficient history to build financial features. "
            f"At least 70+ observations per ticker are needed; found {ticker_counts}."
        )


def _validate_target_input(features_df: pd.DataFrame, horizon_days: int, threshold: float) -> None:
    if not isinstance(features_df, pd.DataFrame):
        raise ValueError("features_df must be a pandas DataFrame.")
    if features_df.empty:
        raise ValueError("features_df must not be empty.")
    if not isinstance(horizon_days, int) or horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer.")
    if not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be numeric.")

    missing_columns = [
        column for column in REQUIRED_PRICE_COLUMNS if column not in features_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")


def _validate_target_not_used_as_feature(future_return_column: str | None = None) -> None:
    forbidden_feature_columns = {TARGET_COLUMN}
    if future_return_column is not None:
        forbidden_feature_columns.add(future_return_column)
    forbidden_feature_columns.update(
        column for column in FEATURE_COLUMNS if column.startswith("future_return_")
    )

    leaked_columns = sorted(set(FEATURE_COLUMNS).intersection(forbidden_feature_columns))
    if leaked_columns:
        raise ValueError(
            "Target or future-return columns must never be used as model features: "
            f"{', '.join(leaked_columns)}."
        )


def _target_distribution_overall(scoring_df: pd.DataFrame) -> pd.DataFrame:
    target = pd.to_numeric(scoring_df[TARGET_COLUMN], errors="coerce").dropna().astype(int)
    counts = target.value_counts().reindex([0, 1], fill_value=0).sort_index()
    rates = target.value_counts(normalize=True).reindex([0, 1], fill_value=0).sort_index()
    return pd.DataFrame({"count": counts.astype(int), "rate": rates})


def _target_distribution_by_ticker(scoring_df: pd.DataFrame) -> pd.DataFrame:
    target_df = scoring_df[["ticker", TARGET_COLUMN]].copy()
    target_df["ticker"] = target_df["ticker"].astype(str)
    target_df[TARGET_COLUMN] = pd.to_numeric(target_df[TARGET_COLUMN], errors="coerce")
    target_df = target_df.dropna(subset=["ticker", TARGET_COLUMN])
    target_df[TARGET_COLUMN] = target_df[TARGET_COLUMN].astype(int)

    grouped = target_df.groupby("ticker")[TARGET_COLUMN]
    distribution = pd.DataFrame(
        {
            "rows": grouped.count(),
            "target_0": grouped.apply(lambda values: int((values == 0).sum())),
            "target_1": grouped.apply(lambda values: int((values == 1).sum())),
            "target_1_rate": grouped.mean(),
        }
    )
    return distribution.sort_index()


def _print_target_balance_warnings(scoring_df: pd.DataFrame) -> None:
    overall = _target_distribution_overall(scoring_df)
    non_empty_classes = overall[overall["count"] > 0]
    if len(non_empty_classes) < 2:
        present_classes = ", ".join(str(index) for index in non_empty_classes.index)
        print(
            "WARNING: target contains only one class "
            f"({present_classes or 'none'}). The scoring dataset can be saved for "
            "analysis, but model training requires both classes."
        )
        return

    minority_rate = float(non_empty_classes["rate"].min())
    if minority_rate < TARGET_IMBALANCE_WARNING_THRESHOLD:
        print(
            "WARNING: target is extremely imbalanced overall. "
            f"Minority class rate is {minority_rate:.2%}."
        )

    by_ticker = _target_distribution_by_ticker(scoring_df)
    one_class_tickers = by_ticker[(by_ticker["target_0"] == 0) | (by_ticker["target_1"] == 0)]
    if not one_class_tickers.empty:
        tickers = ", ".join(one_class_tickers.index.astype(str))
        print(f"WARNING: these tickers have only one target class: {tickers}")

    ticker_minority_rate = by_ticker[["target_0", "target_1"]].min(axis=1) / by_ticker["rows"]
    imbalanced_tickers = ticker_minority_rate[
        (ticker_minority_rate > 0) & (ticker_minority_rate < TARGET_IMBALANCE_WARNING_THRESHOLD)
    ]
    if not imbalanced_tickers.empty:
        details = ", ".join(
            f"{ticker}={rate:.2%}" for ticker, rate in imbalanced_tickers.sort_values().items()
        )
        print(f"WARNING: target is extremely imbalanced for tickers: {details}")


def _print_target_diagnostics(scoring_df: pd.DataFrame) -> None:
    print("Target distribution overall:")
    print(_target_distribution_overall(scoring_df).to_string())
    print("Target distribution by ticker:")
    print(_target_distribution_by_ticker(scoring_df).to_string())
    _print_target_balance_warnings(scoring_df)


def _calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.rolling(window=window, min_periods=window).mean()
    average_loss = losses.rolling(window=window, min_periods=window).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)


def _build_single_ticker_features(ticker_prices: pd.DataFrame) -> pd.DataFrame:
    ticker_prices = ticker_prices.sort_values("date").copy()
    close = ticker_prices["close"]
    volume = ticker_prices["volume"]
    daily_returns = close.pct_change(fill_method=None)

    features = ticker_prices[["date", "ticker", "close"]].copy()
    features["past_return_20d"] = close / close.shift(20) - 1
    features["momentum_20d"] = close - close.shift(20)
    features["volatility_20d"] = daily_returns.rolling(window=20, min_periods=20).std()
    features["rsi_14d"] = _calculate_rsi(close, window=14)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    features["macd"] = ema_12 - ema_26
    features["macd_signal"] = features["macd"].ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    features["moving_average_20d"] = close.rolling(window=20, min_periods=20).mean()
    features["moving_average_50d"] = close.rolling(window=50, min_periods=50).mean()
    features["average_volume_20d"] = volume.rolling(window=20, min_periods=20).mean()
    features["drawdown"] = close / close.cummax() - 1

    return features


def calculate_daily_returns(clean_prices_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from cleaned close prices.

    Parameters
    ----------
    clean_prices_df:
        Cleaned prices DataFrame containing at least ``date``, ``ticker``, and
        ``close`` columns.

    Returns
    -------
    pandas.DataFrame
        Wide DataFrame with dates as index, tickers as columns, and daily
        percentage returns as values.

    Raises
    ------
    ValueError
        If the input dataframe is invalid, missing required columns, or cannot
        produce any returns.
    """
    prices = _prepare_price_input_for_returns(clean_prices_df)
    prices["daily_return"] = prices.groupby("ticker")["close"].pct_change(fill_method=None)
    return_values = prices["daily_return"].to_numpy(dtype=float)
    impossible_return_mask = prices["daily_return"].notna() & (
        (~np.isfinite(return_values)) | (return_values <= MIN_POSSIBLE_RETURN)
    )
    impossible_return_rows = prices.loc[
        impossible_return_mask,
        ["date", "ticker", "close", "daily_return"],
    ]
    _print_impossible_returns_report(impossible_return_rows)
    prices.loc[impossible_return_mask, "daily_return"] = pd.NA

    returns = prices.pivot(index="date", columns="ticker", values="daily_return")
    returns = returns.sort_index().dropna(how="all")
    returns.columns.name = None

    if returns.empty:
        raise ValueError("At least two close prices per ticker are required to calculate returns.")

    _validate_returns_output(returns)
    return returns


def build_financial_features(clean_prices_df: pd.DataFrame) -> pd.DataFrame:
    """Build financial features from cleaned prices without look-ahead bias.

    All rolling, exponential, and cumulative calculations are performed within
    each ticker using observations available up to and including date ``t``.
    No future returns or target variables are created here.

    Parameters
    ----------
    clean_prices_df:
        Cleaned prices DataFrame containing at least ``date``, ``ticker``,
        ``close``, and ``volume`` columns.

    Returns
    -------
    pandas.DataFrame
        Long-format dataframe with ``date``, ``ticker``, ``close``, and the
        Phase 4 financial feature columns.

    Raises
    ------
    ValueError
        If the input is invalid or there is insufficient history to calculate
        the required rolling features.
    """
    prices = _prepare_feature_input(clean_prices_df)
    _validate_feature_history(prices)

    feature_frames = [
        _build_single_ticker_features(ticker_prices)
        for _, ticker_prices in prices.groupby("ticker", sort=True)
    ]
    features = pd.concat(feature_frames, ignore_index=True)
    features = features.replace([float("inf"), float("-inf")], pd.NA)
    features = features.dropna(subset=FEATURE_COLUMNS)
    features = features.sort_values(["ticker", "date"]).reset_index(drop=True)

    if features.empty:
        raise ValueError(
            "Insufficient history to build financial features. "
            "At least 70+ observations per ticker are needed."
        )

    features = features[["date", "ticker", "close", *FEATURE_COLUMNS]]
    _validate_feature_output(features)
    return features


def add_target_variable(
    features_df: pd.DataFrame,
    horizon_days: int = 20,
    threshold: float = 0.02,
) -> pd.DataFrame:
    """Add a future-return target for supervised scoring.

    The target is calculated per ticker as ``close.shift(-horizon_days) / close
    - 1``. The resulting future return column is kept for auditability, but it
    must not be included in model feature columns during training.

    Parameters
    ----------
    features_df:
        Long-format features DataFrame with at least ``date``, ``ticker``, and
        ``close`` columns.
    horizon_days:
        Number of future rows per ticker used to calculate the target return.
    threshold:
        Return threshold above which ``target`` is set to 1.

    Returns
    -------
    pandas.DataFrame
        Features with ``future_return_{horizon_days}d`` and ``target`` columns,
        sorted by ticker and date, with rows lacking future returns removed.

    Raises
    ------
    ValueError
        If inputs are invalid or no target rows can be built.
    """
    _validate_target_input(features_df, horizon_days, threshold)

    target_df = features_df.copy()
    target_df["date"] = pd.to_datetime(target_df["date"], errors="coerce")
    target_df["close"] = pd.to_numeric(target_df["close"], errors="coerce")
    target_df = target_df.dropna(subset=["date", "ticker", "close"])

    if target_df.empty:
        raise ValueError("No valid rows remain after date and close conversion.")
    if (target_df["close"] <= 0).any():
        raise ValueError("close prices must be strictly positive.")
    if target_df.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("features_df contains duplicate rows for the same date and ticker.")

    target_df = target_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    future_return_column = f"future_return_{horizon_days}d"
    _validate_target_not_used_as_feature(future_return_column)
    future_close = target_df.groupby("ticker")["close"].shift(-horizon_days)
    target_df[future_return_column] = future_close / target_df["close"] - 1
    target_df = target_df.dropna(subset=[future_return_column]).copy()

    if target_df.empty:
        raise ValueError("Insufficient future observations to build the target variable.")

    target_df[TARGET_COLUMN] = (target_df[future_return_column] > threshold).astype(int)
    _print_target_balance_warnings(target_df)
    return target_df.sort_values(["ticker", "date"]).reset_index(drop=True)


def save_returns(returns_df: pd.DataFrame, output_path=DEFAULT_RETURNS_OUTPUT_PATH) -> Path:
    """Save daily returns to a CSV file.

    Parameters
    ----------
    returns_df:
        Daily returns DataFrame returned by ``calculate_daily_returns``.
    output_path:
        Destination CSV path. Defaults to ``data/processed/returns.csv``.

    Returns
    -------
    pathlib.Path
        Path where the returns were saved.

    Raises
    ------
    ValueError
        If the returns dataframe or output path is invalid.
    RuntimeError
        If writing the CSV file fails.
    """
    _validate_returns_output(returns_df)
    if not output_path:
        raise ValueError("output_path must be provided.")

    path = Path(output_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("output_path must be a CSV file path ending with `.csv`.")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        returns_df.to_csv(path, index_label="date")
    except Exception as exc:
        raise RuntimeError(f"Failed to save returns to '{path}'.") from exc

    print(f"Saved daily returns to: {path}")
    print(f"Output shape: {returns_df.shape}")
    print("Missing values summary:")
    print(returns_df.isna().sum().sort_index().to_string())

    return path


def save_features(features_df: pd.DataFrame, output_path=DEFAULT_FEATURES_OUTPUT_PATH) -> Path:
    """Save Phase 4 financial features to a CSV file.

    Parameters
    ----------
    features_df:
        Long-format features DataFrame returned by ``build_financial_features``.
    output_path:
        Destination CSV path. Defaults to ``data/processed/features.csv``.

    Returns
    -------
    pathlib.Path
        Path where the features were saved.

    Raises
    ------
    ValueError
        If the features dataframe or output path is invalid.
    RuntimeError
        If writing the CSV file fails.
    """
    if not isinstance(features_df, pd.DataFrame):
        raise ValueError("features_df must be a pandas DataFrame.")
    if features_df.empty:
        raise ValueError("features_df must not be empty.")
    if not output_path:
        raise ValueError("output_path must be provided.")
    _validate_feature_output(features_df)

    path = Path(output_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("output_path must be a CSV file path ending with `.csv`.")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(path, index=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to save features to '{path}'.") from exc

    print(f"Saved financial features to: {path}")
    print(f"Output shape: {features_df.shape}")
    _print_feature_diagnostics(features_df)

    return path


def save_scoring_dataset(
    scoring_df: pd.DataFrame,
    output_path=DEFAULT_SCORING_DATASET_OUTPUT_PATH,
) -> Path:
    """Save the Phase 5 scoring dataset with target diagnostics.

    The saved CSV keeps the ``future_return_*d`` column for analysis, but model
    training code must exclude it from feature columns to avoid look-ahead bias.

    Parameters
    ----------
    scoring_df:
        Dataset returned by ``add_target_variable``.
    output_path:
        Destination CSV path. Defaults to ``data/processed/scoring_dataset.csv``.

    Returns
    -------
    pathlib.Path
        Path where the scoring dataset was saved.

    Raises
    ------
    ValueError
        If the dataframe, target classes, or output path are invalid.
    RuntimeError
        If writing the CSV file fails.
    """
    if not isinstance(scoring_df, pd.DataFrame):
        raise ValueError("scoring_df must be a pandas DataFrame.")
    if scoring_df.empty:
        raise ValueError("scoring_df must not be empty.")
    if not output_path:
        raise ValueError("output_path must be provided.")

    future_return_columns = [
        column for column in scoring_df.columns if column.startswith("future_return_")
    ]
    required_columns = ["date", "ticker", "close", TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in scoring_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")
    if not future_return_columns:
        raise ValueError("Missing required future_return_* column.")

    for future_return_column in future_return_columns:
        _validate_target_not_used_as_feature(future_return_column)

    path = Path(output_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("output_path must be a CSV file path ending with `.csv`.")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        scoring_df.to_csv(path, index=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to save scoring dataset to '{path}'.") from exc

    print(f"Saved scoring dataset to: {path}")
    print(f"Output shape: {scoring_df.shape}")
    _print_target_diagnostics(scoring_df)

    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run feature pipeline steps: returns, features, or target dataset."
    )
    parser.add_argument(
        "--mode",
        choices=["returns", "features", "target"],
        default="returns",
        help="Pipeline step to run. Defaults to returns.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input cleaned CSV path, for example data\\processed\\clean_prices.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. Defaults to data\\processed\\returns.csv for returns "
            "data\\processed\\features.csv for features, and "
            "data\\processed\\scoring_dataset.csv for target."
        ),
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=20,
        help="Future horizon used by target mode. Defaults to 20.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Future return threshold used by target mode. Defaults to 0.02.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 3 or Phase 4 command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"{args.mode.title()} failed: input file does not exist: {input_path}", file=sys.stderr)
        return 1
    if input_path.stat().st_size == 0:
        print(f"{args.mode.title()} failed: input file is empty: {input_path}", file=sys.stderr)
        return 1

    try:
        clean_prices = pd.read_csv(input_path)
        if args.mode == "features":
            output_path = args.output or DEFAULT_FEATURES_OUTPUT_PATH
            features = build_financial_features(clean_prices)
            save_features(features, output_path)
        elif args.mode == "target":
            output_path = args.output or DEFAULT_SCORING_DATASET_OUTPUT_PATH
            scoring_dataset = add_target_variable(
                clean_prices,
                horizon_days=args.horizon_days,
                threshold=args.threshold,
            )
            save_scoring_dataset(scoring_dataset, output_path)
        else:
            output_path = args.output or DEFAULT_RETURNS_OUTPUT_PATH
            returns = calculate_daily_returns(clean_prices)
            save_returns(returns, output_path)
    except Exception as exc:
        print(f"{args.mode.title()} failed: {exc}", file=sys.stderr)
        return 1

    return 0


__all__ = [
    "calculate_daily_returns",
    "build_financial_features",
    "add_target_variable",
    "save_returns",
    "save_features",
    "save_scoring_dataset",
]


if __name__ == "__main__":
    raise SystemExit(main())
