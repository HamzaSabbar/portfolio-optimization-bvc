from pathlib import Path
import sys

import joblib
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.scoring as scoring


def _make_scoring_dataset(rows=100):
    feature_columns = scoring.get_feature_columns()
    data = {
        "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
        "ticker": ["BCP" if index % 2 == 0 else "CIH" for index in range(rows)],
        "target": [index % 2 for index in range(rows)],
        "future_return_20d": [0.03 if index % 2 else 0.01 for index in range(rows)],
    }
    for feature_index, column in enumerate(feature_columns, start=1):
        data[column] = [
            (index * feature_index) / 100.0 + (index % 2) * 0.1
            for index in range(rows)
        ]
    return pd.DataFrame(data)


def test_get_feature_columns_excludes_target_and_future_returns():
    feature_columns = scoring.get_feature_columns()

    assert "target" not in feature_columns
    assert "future_return_20d" not in feature_columns
    assert "close" not in feature_columns
    assert "date" not in feature_columns
    assert "ticker" not in feature_columns
    assert feature_columns == [
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


def test_train_scoring_model_uses_chronological_split_and_saves_artifacts(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "logistic_model.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    monkeypatch.setattr(scoring, "MODEL_OUTPUT_PATH", model_path)
    monkeypatch.setattr(scoring, "SCALER_OUTPUT_PATH", scaler_path)
    dataset = _make_scoring_dataset(rows=100)

    model, scaler, metrics = scoring.train_scoring_model(dataset, test_size=0.2)

    expected_train = dataset.sort_values(["date", "ticker"]).iloc[:80]
    expected_mean = expected_train[scoring.get_feature_columns()].mean().to_numpy()

    assert model_path.exists()
    assert scaler_path.exists()
    assert model_path.stat().st_size > 0
    assert scaler_path.stat().st_size > 0
    assert joblib.load(model_path).__class__ is model.__class__
    assert joblib.load(scaler_path).__class__ is scaler.__class__
    assert scaler.mean_ == pytest.approx(expected_mean)
    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "confusion_matrix",
    }
    assert metrics["roc_auc"] is not None
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2


def test_score_latest_stocks_scores_one_latest_row_per_ticker(tmp_path, monkeypatch):
    model_path = tmp_path / "logistic_model.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    scores_path = tmp_path / "latest_scores.csv"
    monkeypatch.setattr(scoring, "MODEL_OUTPUT_PATH", model_path)
    monkeypatch.setattr(scoring, "SCALER_OUTPUT_PATH", scaler_path)
    monkeypatch.setattr(scoring, "LATEST_SCORES_OUTPUT_PATH", scores_path)
    dataset = _make_scoring_dataset(rows=100)
    model, scaler, _ = scoring.train_scoring_model(dataset, test_size=0.2)

    latest_scores = scoring.score_latest_stocks(dataset, model, scaler)
    saved_scores = pd.read_csv(scores_path)

    assert scores_path.exists()
    assert latest_scores.columns.tolist() == ["date", "ticker", "score"]
    assert set(latest_scores["ticker"]) == {"BCP", "CIH"}
    assert len(latest_scores) == 2
    assert latest_scores["score"].between(0, 1).all()
    assert saved_scores.columns.tolist() == ["date", "ticker", "score"]


def test_score_latest_stocks_uses_latest_valid_row_per_ticker(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "MODEL_OUTPUT_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(scoring, "SCALER_OUTPUT_PATH", tmp_path / "scaler.pkl")
    monkeypatch.setattr(scoring, "LATEST_SCORES_OUTPUT_PATH", tmp_path / "scores.csv")
    dataset = _make_scoring_dataset(rows=100)
    model, scaler, _ = scoring.train_scoring_model(dataset, test_size=0.2)

    latest_bcp_index = dataset[dataset["ticker"] == "BCP"].index.max()
    previous_bcp_date = dataset.loc[latest_bcp_index - 2, "date"]
    dataset.loc[latest_bcp_index, "drawdown"] = None

    latest_scores = scoring.score_latest_stocks(dataset, model, scaler)
    bcp_date = pd.to_datetime(
        latest_scores.loc[latest_scores["ticker"] == "BCP", "date"].iloc[0]
    )

    assert bcp_date == previous_bcp_date


def test_train_scoring_model_drops_rows_with_missing_features(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "MODEL_OUTPUT_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(scoring, "SCALER_OUTPUT_PATH", tmp_path / "scaler.pkl")
    dataset = _make_scoring_dataset(rows=100)
    dataset.loc[0, "drawdown"] = None

    model, scaler, metrics = scoring.train_scoring_model(dataset, test_size=0.2)

    assert model is not None
    assert scaler is not None
    assert metrics["confusion_matrix"]


def test_train_scoring_model_rejects_missing_feature_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "MODEL_OUTPUT_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(scoring, "SCALER_OUTPUT_PATH", tmp_path / "scaler.pkl")
    dataset = _make_scoring_dataset(rows=40).drop(columns=["drawdown"])

    with pytest.raises(ValueError, match="Missing required columns"):
        scoring.train_scoring_model(dataset)
