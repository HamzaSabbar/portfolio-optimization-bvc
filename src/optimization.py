"""Constrained portfolio optimization utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


RISK_PROFILES = {
    "conservative": {"lambda_risk": 10.0, "max_weight": 0.20},
    "balanced": {"lambda_risk": 5.0, "max_weight": 0.30},
    "dynamic": {"lambda_risk": 2.0, "max_weight": 0.40},
}
ALLOCATION_OUTPUT_PATH = Path("data/outputs/optimal_allocation.csv")
INVESTMENT_AMOUNTS_OUTPUT_PATH = Path("data/outputs/investment_amounts.csv")


def _validate_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(scores_df, pd.DataFrame):
        raise ValueError("scores_df must be a pandas DataFrame.")
    if scores_df.empty:
        raise ValueError("scores_df must not be empty.")

    missing_columns = [column for column in ["ticker", "score"] if column not in scores_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required score columns: {', '.join(missing_columns)}.")

    scores = scores_df[["ticker", "score"]].copy()
    scores["ticker"] = scores["ticker"].astype(str).str.strip()
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scores = scores.dropna(subset=["ticker", "score"])
    scores = scores[scores["ticker"] != ""]

    if scores.empty:
        raise ValueError("No valid score rows remain after validation.")

    scores = scores.sort_values("ticker").drop_duplicates(subset=["ticker"], keep="last")
    return scores.reset_index(drop=True)


def _validate_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns_df, pd.DataFrame):
        raise ValueError("returns_df must be a pandas DataFrame.")
    if returns_df.empty:
        raise ValueError("returns_df must not be empty.")

    returns = returns_df.copy()
    for column in returns.columns:
        returns[column] = pd.to_numeric(returns[column], errors="coerce")

    if np.isinf(returns.to_numpy(dtype=float)).any():
        raise ValueError("returns_df contains infinite return values.")

    returns = returns.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if returns.empty:
        raise ValueError("returns_df contains no numeric return values.")

    return returns


def _get_risk_profile(risk_profile: str) -> dict[str, float]:
    if risk_profile not in RISK_PROFILES:
        allowed_profiles = ", ".join(RISK_PROFILES)
        raise ValueError(f"risk_profile must be one of: {allowed_profiles}.")
    return RISK_PROFILES[risk_profile]


def _align_scores_and_returns(scores_df: pd.DataFrame, returns_df: pd.DataFrame):
    scores = _validate_scores(scores_df)
    returns = _validate_returns(returns_df)

    score_tickers = set(scores["ticker"])
    return_tickers = set(map(str, returns.columns))
    aligned_tickers = sorted(score_tickers.intersection(return_tickers))

    if not aligned_tickers:
        raise ValueError("No overlapping tickers between scores_df and returns_df.")

    aligned_scores = scores.set_index("ticker").loc[aligned_tickers, "score"]
    aligned_returns = returns[aligned_tickers].dropna(how="any")

    if aligned_returns.empty:
        raise ValueError("No complete return rows remain for aligned tickers.")
    if len(aligned_returns) < 2:
        raise ValueError("At least two return observations are required.")

    return aligned_tickers, aligned_scores.to_numpy(dtype=float), aligned_returns


def _save_allocation(allocation_df: pd.DataFrame) -> Path:
    try:
        ALLOCATION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        allocation_df.to_csv(ALLOCATION_OUTPUT_PATH, index=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save optimal allocation to '{ALLOCATION_OUTPUT_PATH}'."
        ) from exc

    return ALLOCATION_OUTPUT_PATH


def _validate_allocation(allocation_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(allocation_df, pd.DataFrame):
        raise ValueError("allocation_df must be a pandas DataFrame.")
    if allocation_df.empty:
        raise ValueError("allocation_df must not be empty.")

    required_columns = ["ticker", "weight", "score"]
    missing_columns = [
        column for column in required_columns if column not in allocation_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required allocation columns: {', '.join(missing_columns)}.")

    allocation = allocation_df[required_columns].copy()
    allocation["ticker"] = allocation["ticker"].astype(str).str.strip()
    allocation["weight"] = pd.to_numeric(allocation["weight"], errors="coerce")
    allocation["score"] = pd.to_numeric(allocation["score"], errors="coerce")
    allocation = allocation.dropna(subset=required_columns)
    allocation = allocation[allocation["ticker"] != ""]

    if allocation.empty:
        raise ValueError("No valid allocation rows remain after validation.")
    if (allocation["weight"] < 0).any():
        raise ValueError("allocation weights must be non-negative.")

    return allocation.reset_index(drop=True)


def _validate_budget(budget) -> float:
    if isinstance(budget, bool):
        raise ValueError("budget must be a positive number.")

    try:
        numeric_budget = float(budget)
    except (TypeError, ValueError) as exc:
        raise ValueError("budget must be a positive number.") from exc

    if not np.isfinite(numeric_budget) or numeric_budget <= 0:
        raise ValueError("budget must be greater than 0.")

    return numeric_budget


def _save_investment_amounts(investment_df: pd.DataFrame) -> Path:
    try:
        INVESTMENT_AMOUNTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        investment_df.to_csv(INVESTMENT_AMOUNTS_OUTPUT_PATH, index=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save investment amounts to '{INVESTMENT_AMOUNTS_OUTPUT_PATH}'."
        ) from exc

    return INVESTMENT_AMOUNTS_OUTPUT_PATH


def optimize_portfolio(scores_df, returns_df, risk_profile="balanced"):
    """Optimize portfolio weights from stock scores and historical returns.

    The optimizer maximizes ``portfolio_score - lambda_risk *
    portfolio_variance`` by minimizing its negative with SciPy's SLSQP method.
    It enforces fully invested long-only weights and the maximum weight allowed
    by the selected risk profile.

    Parameters
    ----------
    scores_df:
        DataFrame containing ``ticker`` and ``score`` columns.
    returns_df:
        Wide daily returns DataFrame with tickers as columns.
    risk_profile:
        One of ``conservative``, ``balanced`` or ``dynamic``.

    Returns
    -------
    pandas.DataFrame
        Allocation with ``ticker``, ``weight`` and ``score`` columns.

    Raises
    ------
    ValueError
        If inputs are invalid, constraints are infeasible, or optimization fails.
    RuntimeError
        If saving the allocation fails.
    """
    profile = _get_risk_profile(risk_profile)
    lambda_risk = profile["lambda_risk"]
    max_weight = profile["max_weight"]
    tickers, scores, aligned_returns = _align_scores_and_returns(scores_df, returns_df)

    n_assets = len(tickers)
    if n_assets * max_weight < 1 - 1e-12:
        raise ValueError(
            "The selected risk profile is infeasible for the number of aligned assets. "
            f"Profile '{risk_profile}' allows max_weight={max_weight:.2f}, but only "
            f"{n_assets} aligned assets are available. Add more assets or choose a "
            "profile with a higher max_weight."
        )

    covariance_matrix = aligned_returns.cov().to_numpy(dtype=float)
    if not np.isfinite(covariance_matrix).all():
        raise ValueError("Covariance matrix contains missing or non-finite values.")

    def objective(weights):
        portfolio_score = float(weights @ scores)
        portfolio_variance = float(weights.T @ covariance_matrix @ weights)
        return -(portfolio_score - lambda_risk * portfolio_variance)

    bounds = [(0.0, max_weight) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]
    initial_weight = 1.0 / n_assets
    initial_weights = np.full(n_assets, initial_weight)

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(f"Portfolio optimization failed: {result.message}")
    if not np.isfinite(result.x).all():
        raise ValueError("Portfolio optimization failed: optimized weights are non-finite.")

    weights = np.clip(result.x, 0.0, max_weight)
    weights = weights / weights.sum()

    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("Optimized weights do not sum to 1.")
    if (weights < -1e-8).any():
        raise ValueError("Optimized weights contain negative values.")
    if (weights > max_weight + 1e-6).any():
        raise ValueError("Optimized weights exceed the maximum allowed weight.")

    allocation = pd.DataFrame(
        {
            "ticker": tickers,
            "weight": weights,
            "score": scores,
        }
    )
    allocation = allocation.sort_values("weight", ascending=False).reset_index(drop=True)
    _save_allocation(allocation)

    return allocation


def compute_investment_amounts(allocation_df, budget):
    """Convert optimized weights into investment amounts.

    Parameters
    ----------
    allocation_df:
        Allocation DataFrame containing ``ticker``, ``weight`` and ``score``.
    budget:
        Positive total investment budget.

    Returns
    -------
    pandas.DataFrame
        Allocation with preserved ``ticker``, ``weight`` and ``score`` columns,
        plus ``amount`` and ``amount_rounded``.

    Raises
    ------
    ValueError
        If the allocation or budget is invalid.
    RuntimeError
        If saving the investment amounts fails.
    """
    allocation = _validate_allocation(allocation_df)
    numeric_budget = _validate_budget(budget)

    investment_amounts = allocation.copy()
    investment_amounts["amount"] = investment_amounts["weight"] * numeric_budget
    investment_amounts["amount_rounded"] = investment_amounts["amount"].round(2)
    investment_amounts = investment_amounts[
        ["ticker", "weight", "score", "amount", "amount_rounded"]
    ]

    _save_investment_amounts(investment_amounts)
    return investment_amounts


__all__ = ["optimize_portfolio", "compute_investment_amounts"]
