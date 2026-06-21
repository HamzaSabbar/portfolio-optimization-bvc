from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.optimization as optimization


def _make_scores_and_returns():
    tickers = ["A", "B", "C", "D", "E"]
    scores = pd.DataFrame(
        {
            "ticker": tickers,
            "score": [0.80, 0.65, 0.50, 0.35, 0.20],
        }
    )
    base_returns = np.array(
        [
            [0.010, 0.004, 0.002, -0.001, 0.003],
            [0.006, 0.002, 0.001, 0.000, 0.002],
            [-0.004, 0.001, 0.003, 0.002, -0.001],
            [0.008, 0.003, 0.002, 0.001, 0.004],
            [0.002, -0.002, 0.001, 0.002, 0.000],
            [0.005, 0.004, -0.001, 0.001, 0.003],
        ]
    )
    returns = pd.DataFrame(base_returns, columns=tickers)
    return scores, returns


def test_optimize_portfolio_weights_sum_to_one(tmp_path, monkeypatch):
    monkeypatch.setattr(
        optimization,
        "ALLOCATION_OUTPUT_PATH",
        tmp_path / "optimal_allocation.csv",
    )
    scores, returns = _make_scores_and_returns()

    allocation = optimization.optimize_portfolio(scores, returns, risk_profile="balanced")

    assert allocation["weight"].sum() == pytest.approx(1.0, abs=1e-6)
    assert (tmp_path / "optimal_allocation.csv").exists()


def test_optimize_portfolio_has_no_negative_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(
        optimization,
        "ALLOCATION_OUTPUT_PATH",
        tmp_path / "optimal_allocation.csv",
    )
    scores, returns = _make_scores_and_returns()

    allocation = optimization.optimize_portfolio(scores, returns, risk_profile="dynamic")

    assert (allocation["weight"] >= -1e-8).all()


def test_optimize_portfolio_respects_max_weight_by_risk_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(
        optimization,
        "ALLOCATION_OUTPUT_PATH",
        tmp_path / "optimal_allocation.csv",
    )
    scores, returns = _make_scores_and_returns()

    allocation = optimization.optimize_portfolio(scores, returns, risk_profile="conservative")

    max_weight = optimization.RISK_PROFILES["conservative"]["max_weight"]
    assert (allocation["weight"] <= max_weight + 1e-6).all()


def test_optimize_portfolio_aligns_tickers(tmp_path, monkeypatch):
    monkeypatch.setattr(
        optimization,
        "ALLOCATION_OUTPUT_PATH",
        tmp_path / "optimal_allocation.csv",
    )
    scores, returns = _make_scores_and_returns()
    scores = pd.concat(
        [scores, pd.DataFrame({"ticker": ["NOT_IN_RETURNS"], "score": [1.0]})],
        ignore_index=True,
    )

    allocation = optimization.optimize_portfolio(scores, returns, risk_profile="balanced")

    assert "NOT_IN_RETURNS" not in set(allocation["ticker"])
    assert set(allocation["ticker"]) == set(returns.columns)


def test_optimize_portfolio_rejects_infeasible_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(
        optimization,
        "ALLOCATION_OUTPUT_PATH",
        tmp_path / "optimal_allocation.csv",
    )
    scores, returns = _make_scores_and_returns()
    scores = scores.head(3)
    returns = returns[scores["ticker"]]

    with pytest.raises(ValueError, match="infeasible"):
        optimization.optimize_portfolio(scores, returns, risk_profile="balanced")


def test_optimize_portfolio_rejects_unknown_risk_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(
        optimization,
        "ALLOCATION_OUTPUT_PATH",
        tmp_path / "optimal_allocation.csv",
    )
    scores, returns = _make_scores_and_returns()

    with pytest.raises(ValueError, match="risk_profile must be one of"):
        optimization.optimize_portfolio(scores, returns, risk_profile="aggressive")


def test_optimize_portfolio_rejects_no_overlapping_tickers(tmp_path, monkeypatch):
    monkeypatch.setattr(
        optimization,
        "ALLOCATION_OUTPUT_PATH",
        tmp_path / "optimal_allocation.csv",
    )
    scores = pd.DataFrame({"ticker": ["X", "Y"], "score": [0.8, 0.6]})
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.02, 0.03]})

    with pytest.raises(ValueError, match="No overlapping tickers"):
        optimization.optimize_portfolio(scores, returns, risk_profile="balanced")


def test_optimize_portfolio_rejects_infinite_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(
        optimization,
        "ALLOCATION_OUTPUT_PATH",
        tmp_path / "optimal_allocation.csv",
    )
    scores, returns = _make_scores_and_returns()
    returns.loc[0, "A"] = np.inf

    with pytest.raises(ValueError, match="infinite return values"):
        optimization.optimize_portfolio(scores, returns, risk_profile="balanced")


def test_compute_investment_amounts_adds_amount_columns_and_saves(tmp_path, monkeypatch):
    monkeypatch.setattr(
        optimization,
        "INVESTMENT_AMOUNTS_OUTPUT_PATH",
        tmp_path / "investment_amounts.csv",
    )
    allocation = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "weight": [0.50, 0.333333, 0.166667],
            "score": [0.80, 0.60, 0.40],
        }
    )

    investment_amounts = optimization.compute_investment_amounts(allocation, 10000)
    saved = pd.read_csv(tmp_path / "investment_amounts.csv")

    assert investment_amounts.columns.tolist() == [
        "ticker",
        "weight",
        "score",
        "amount",
        "amount_rounded",
    ]
    assert investment_amounts["ticker"].tolist() == ["A", "B", "C"]
    assert investment_amounts["weight"].tolist() == pytest.approx([0.50, 0.333333, 0.166667])
    assert investment_amounts["score"].tolist() == pytest.approx([0.80, 0.60, 0.40])
    assert investment_amounts["amount"].tolist() == pytest.approx([5000.0, 3333.33, 1666.67])
    assert investment_amounts["amount_rounded"].tolist() == pytest.approx(
        [5000.0, 3333.33, 1666.67]
    )
    assert saved.columns.tolist() == investment_amounts.columns.tolist()


def test_compute_investment_amounts_rejects_non_positive_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(
        optimization,
        "INVESTMENT_AMOUNTS_OUTPUT_PATH",
        tmp_path / "investment_amounts.csv",
    )
    allocation = pd.DataFrame(
        {
            "ticker": ["A"],
            "weight": [1.0],
            "score": [0.80],
        }
    )

    with pytest.raises(ValueError, match="budget must be greater than 0"):
        optimization.compute_investment_amounts(allocation, 0)


def test_compute_investment_amounts_rejects_negative_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(
        optimization,
        "INVESTMENT_AMOUNTS_OUTPUT_PATH",
        tmp_path / "investment_amounts.csv",
    )
    allocation = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "weight": [1.1, -0.1],
            "score": [0.80, 0.60],
        }
    )

    with pytest.raises(ValueError, match="non-negative"):
        optimization.compute_investment_amounts(allocation, 10000)
