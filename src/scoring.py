"""Logistic-regression scoring utilities for stock attractiveness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


MODEL_OUTPUT_PATH = Path("models/logistic_model.pkl")
SCALER_OUTPUT_PATH = Path("models/scaler.pkl")
LATEST_SCORES_OUTPUT_PATH = Path("data/outputs/latest_scores.csv")
DEFAULT_SCORING_DATASET_INPUT_PATH = Path("data/processed/scoring_dataset.csv")
DEFAULT_FEATURES_INPUT_PATH = Path("data/processed/features.csv")
REQUIRED_METADATA_COLUMNS = ["date", "ticker"]
TARGET_COLUMN = "target"


def get_feature_columns() -> list[str]:
    """Return the financial feature columns used by the scoring model.

    ``future_return_*`` columns and ``target`` are intentionally excluded to
    avoid data leakage during model training.
    """
    return [
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


def _validate_dataframe(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame.")
    if df.empty:
        raise ValueError(f"{name} must not be empty.")


def _validate_feature_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in get_feature_columns() if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {', '.join(missing_columns)}.")


def _coerce_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    _validate_feature_columns(df)
    features = df[get_feature_columns()].copy()

    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    return features.replace([float("inf"), float("-inf")], pd.NA)


def _prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    features = _coerce_feature_matrix(df)

    if features.isna().any().any():
        missing_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(
            "Feature columns must be numeric and non-missing. Invalid columns: "
            f"{', '.join(missing_columns)}."
        )

    return features.astype(float)


def _prepare_scoring_dataset(scoring_dataset: pd.DataFrame) -> pd.DataFrame:
    _validate_dataframe(scoring_dataset, "scoring_dataset")
    required_columns = [*REQUIRED_METADATA_COLUMNS, TARGET_COLUMN, *get_feature_columns()]
    missing_columns = [
        column for column in required_columns if column not in scoring_dataset.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")

    dataset = scoring_dataset[required_columns].copy()
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    dataset[TARGET_COLUMN] = pd.to_numeric(dataset[TARGET_COLUMN], errors="coerce")
    dataset[get_feature_columns()] = _coerce_feature_matrix(dataset)
    dataset = dataset.dropna(subset=["date", "ticker", TARGET_COLUMN, *get_feature_columns()])

    if dataset.empty:
        raise ValueError(
            "No valid scoring rows remain after date, target, and feature conversion."
        )
    if not dataset[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("target must contain only 0 and 1 values.")
    if dataset[TARGET_COLUMN].nunique() < 2:
        raise ValueError("scoring_dataset must contain both target classes 0 and 1.")

    return dataset.sort_values(["date", "ticker"]).reset_index(drop=True)


def _chronological_split(dataset: pd.DataFrame, test_size: float):
    if not isinstance(test_size, (int, float)) or not 0 < float(test_size) < 1:
        raise ValueError("test_size must be a float between 0 and 1.")

    split_index = int(len(dataset) * (1 - float(test_size)))
    if split_index <= 0 or split_index >= len(dataset):
        raise ValueError("test_size leaves an empty train or test set.")

    train_df = dataset.iloc[:split_index].copy()
    test_df = dataset.iloc[split_index:].copy()

    if train_df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training data must contain both target classes 0 and 1.")

    return train_df, test_df


def _calculate_metrics(y_true, y_pred, y_proba) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    if pd.Series(y_true).nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def _save_pickle(obj, output_path: Path) -> Path:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, output_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save object to '{output_path}'.") from exc

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Saved file is missing or empty: '{output_path}'.")

    return output_path


def train_scoring_model(scoring_dataset, test_size=0.2):
    """Train a logistic-regression stock scoring model.

    The split is chronological: earliest rows are used for training and latest
    rows for testing. ``StandardScaler`` is fitted only on the training feature
    matrix, then reused for the test set.

    Parameters
    ----------
    scoring_dataset:
        DataFrame containing Phase 4 features and the Phase 5 ``target``.
    test_size:
        Fraction of the latest rows reserved for testing.

    Returns
    -------
    tuple
        ``(model, scaler, metrics)`` where metrics contains accuracy,
        precision, recall, f1_score, roc_auc when possible, and confusion_matrix.

    Raises
    ------
    ValueError
        If inputs are invalid or training cannot be performed safely.
    RuntimeError
        If model or scaler saving fails.
    """
    dataset = _prepare_scoring_dataset(scoring_dataset)
    train_df, test_df = _chronological_split(dataset, test_size)

    x_train = _prepare_feature_matrix(train_df)
    y_train = train_df[TARGET_COLUMN].astype(int)
    x_test = _prepare_feature_matrix(test_df)
    y_test = test_df[TARGET_COLUMN].astype(int)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)
    y_proba = model.predict_proba(x_test_scaled)[:, 1]
    metrics = _calculate_metrics(y_test, y_pred, y_proba)

    _save_pickle(model, MODEL_OUTPUT_PATH)
    _save_pickle(scaler, SCALER_OUTPUT_PATH)

    return model, scaler, metrics


def score_latest_stocks(features_df, model, scaler):
    """Score the latest available feature row for each ticker.

    Parameters
    ----------
    features_df:
        Long-format features DataFrame containing ``date``, ``ticker`` and the
        model feature columns.
    model:
        Trained ``LogisticRegression`` model.
    scaler:
        Fitted ``StandardScaler``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``date``, ``ticker`` and ``score`` columns, saved to
        ``data/outputs/latest_scores.csv``.

    Raises
    ------
    ValueError
        If inputs or feature rows are invalid.
    RuntimeError
        If saving scores fails.
    """
    _validate_dataframe(features_df, "features_df")
    required_columns = [*REQUIRED_METADATA_COLUMNS, *get_feature_columns()]
    missing_columns = [column for column in required_columns if column not in features_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")
    if not hasattr(model, "predict_proba"):
        raise ValueError("model must expose predict_proba.")
    if not hasattr(scaler, "transform"):
        raise ValueError("scaler must expose transform.")

    features = features_df[required_columns].copy()
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    features[get_feature_columns()] = _coerce_feature_matrix(features)
    features = features.dropna(subset=["date", "ticker", *get_feature_columns()])

    if features.empty:
        raise ValueError("No valid feature rows remain after date and feature conversion.")

    features = features.sort_values(["ticker", "date"]).reset_index(drop=True)
    latest_rows = features.groupby("ticker", as_index=False, sort=True).tail(1)
    latest_rows = latest_rows.sort_values(["ticker", "date"]).reset_index(drop=True)
    x_latest = _prepare_feature_matrix(latest_rows)
    scores = model.predict_proba(scaler.transform(x_latest))[:, 1]

    scores_df = latest_rows[["date", "ticker"]].copy()
    scores_df["score"] = scores
    scores_df = scores_df.sort_values("score", ascending=False).reset_index(drop=True)

    try:
        LATEST_SCORES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(LATEST_SCORES_OUTPUT_PATH, index=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save latest scores to '{LATEST_SCORES_OUTPUT_PATH}'."
        ) from exc

    return scores_df


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Phase 6 logistic scoring model and score latest stocks."
    )
    parser.add_argument(
        "--scoring-input",
        default=str(DEFAULT_SCORING_DATASET_INPUT_PATH),
        help="Input scoring dataset CSV. Defaults to data\\processed\\scoring_dataset.csv.",
    )
    parser.add_argument(
        "--features-input",
        default=str(DEFAULT_FEATURES_INPUT_PATH),
        help="Input features CSV for latest scores. Defaults to data\\processed\\features.csv.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Chronological test fraction. Defaults to 0.2.",
    )
    return parser


def _format_metric_value(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    """Run Phase 6 model training and latest-stock scoring from CSV inputs."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    scoring_path = Path(args.scoring_input)
    features_path = Path(args.features_input)
    for path in [scoring_path, features_path]:
        if not path.exists():
            print(f"Scoring failed: input file does not exist: {path}", file=sys.stderr)
            return 1
        if path.stat().st_size == 0:
            print(f"Scoring failed: input file is empty: {path}", file=sys.stderr)
            return 1

    try:
        scoring_dataset = pd.read_csv(scoring_path)
        features = pd.read_csv(features_path)
        model, scaler, metrics = train_scoring_model(
            scoring_dataset,
            test_size=args.test_size,
        )
        latest_scores = score_latest_stocks(features, model, scaler)
    except Exception as exc:
        print(f"Scoring failed: {exc}", file=sys.stderr)
        return 1

    print(f"Saved model to: {MODEL_OUTPUT_PATH}")
    print(f"Saved scaler to: {SCALER_OUTPUT_PATH}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {_format_metric_value(value)}")
    print(f"Saved latest scores to: {LATEST_SCORES_OUTPUT_PATH}")
    print(f"Latest scores shape: {latest_scores.shape}")

    return 0


__all__ = [
    "get_feature_columns",
    "train_scoring_model",
    "score_latest_stocks",
]


if __name__ == "__main__":
    raise SystemExit(main())
