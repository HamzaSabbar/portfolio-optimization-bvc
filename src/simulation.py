"""Historical bootstrap Monte Carlo simulation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SIMULATION_PATHS_OUTPUT_PATH = Path("data/outputs/simulation_paths.csv")
SIMULATION_SUMMARY_OUTPUT_PATH = Path("data/outputs/simulation_summary.csv")


def _validate_positive_number(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number.") from exc

    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be greater than 0.")

    return numeric_value


def _validate_positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns_df, pd.DataFrame):
        raise ValueError("returns_df must be a pandas DataFrame.")
    if returns_df.empty:
        raise ValueError("returns_df must not be empty.")

    returns = returns_df.copy()
    returns.columns = returns.columns.astype(str)
    for column in returns.columns:
        returns[column] = pd.to_numeric(returns[column], errors="coerce")

    returns = returns.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if returns.empty:
        raise ValueError("returns_df contains no numeric return values.")

    return returns


def _validate_allocation(allocation_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(allocation_df, pd.DataFrame):
        raise ValueError("allocation_df must be a pandas DataFrame.")
    if allocation_df.empty:
        raise ValueError("allocation_df must not be empty.")

    required_columns = ["ticker", "weight"]
    missing_columns = [
        column for column in required_columns if column not in allocation_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required allocation columns: {', '.join(missing_columns)}.")

    allocation = allocation_df[required_columns].copy()
    allocation["ticker"] = allocation["ticker"].astype(str).str.strip()
    allocation["weight"] = pd.to_numeric(allocation["weight"], errors="coerce")
    allocation = allocation.dropna(subset=required_columns)
    allocation = allocation[allocation["ticker"] != ""]

    if allocation.empty:
        raise ValueError("No valid allocation rows remain after validation.")
    if (allocation["weight"] < 0).any():
        raise ValueError("allocation weights must be non-negative.")
    if not np.isfinite(allocation["weight"].to_numpy(dtype=float)).all():
        raise ValueError("allocation weights must be finite.")

    allocation = allocation.groupby("ticker", as_index=False)["weight"].sum()
    weight_sum = allocation["weight"].sum()
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError("allocation weights must sum close to 1.")

    return allocation


def _align_returns_and_allocation(returns_df: pd.DataFrame, allocation_df: pd.DataFrame):
    returns = _validate_returns(returns_df)
    allocation = _validate_allocation(allocation_df)

    return_tickers = set(returns.columns)
    allocation_tickers = set(allocation["ticker"])
    aligned_tickers = sorted(return_tickers.intersection(allocation_tickers))

    if not aligned_tickers:
        raise ValueError("No overlapping tickers between returns_df and allocation_df.")

    aligned_returns = returns[aligned_tickers].dropna(how="any")
    if aligned_returns.empty:
        raise ValueError("No complete return rows remain for aligned tickers.")

    aligned_weights = allocation.set_index("ticker").loc[aligned_tickers, "weight"]
    weight_sum = aligned_weights.sum()
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(
            "Aligned allocation weights must sum close to 1. "
            "Use only tickers present in both returns_df and allocation_df."
        )

    return aligned_returns, aligned_weights.to_numpy(dtype=float)


def _save_simulation_results(paths_df: pd.DataFrame, summary: dict) -> None:
    try:
        SIMULATION_PATHS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        paths_df.to_csv(SIMULATION_PATHS_OUTPUT_PATH, index_label="day")
        pd.DataFrame([summary]).to_csv(SIMULATION_SUMMARY_OUTPUT_PATH, index=False)
    except Exception as exc:
        raise RuntimeError("Failed to save simulation results.") from exc


def historical_bootstrap_monte_carlo(
    returns_df,
    allocation_df,
    initial_value,
    horizon_days=252,
    n_simulations=5000,
    target_value=None,
    random_state=42,
):
    """Run a historical bootstrap Monte Carlo simulation.

    The simulation samples observed historical portfolio daily returns with
    replacement. It does not use GBM and does not assume returns are normally
    distributed.

    Parameters
    ----------
    returns_df:
        Wide DataFrame of historical daily returns with tickers as columns.
    allocation_df:
        DataFrame containing at least ``ticker`` and ``weight`` columns.
    initial_value:
        Initial portfolio value.
    horizon_days:
        Number of future days to simulate.
    n_simulations:
        Number of bootstrap simulation paths.
    target_value:
        Optional target portfolio value for success probability.
    random_state:
        Seed passed to NumPy's ``default_rng`` for reproducible sampling.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Simulation paths with day 0 included and a summary metrics dictionary.

    Raises
    ------
    ValueError
        If inputs are invalid or there are no usable historical returns.
    RuntimeError
        If saving simulation outputs fails.
    """
    numeric_initial_value = _validate_positive_number(initial_value, "initial_value")
    horizon_days = _validate_positive_integer(horizon_days, "horizon_days")
    n_simulations = _validate_positive_integer(n_simulations, "n_simulations")
    numeric_target_value = None
    if target_value is not None:
        numeric_target_value = _validate_positive_number(target_value, "target_value")

    aligned_returns, weights = _align_returns_and_allocation(returns_df, allocation_df)
    portfolio_returns = aligned_returns.to_numpy(dtype=float) @ weights
    portfolio_returns = portfolio_returns[np.isfinite(portfolio_returns)]

    if portfolio_returns.size == 0:
        raise ValueError("No valid historical portfolio returns are available.")
    if (portfolio_returns <= -1).any():
        raise ValueError("Historical portfolio returns must be greater than -100%.")

    rng = np.random.default_rng(random_state)
    sampled_returns = rng.choice(
        portfolio_returns,
        size=(horizon_days, n_simulations),
        replace=True,
    )
    simulated_values = numeric_initial_value * np.cumprod(1 + sampled_returns, axis=0)
    paths = np.vstack([np.full((1, n_simulations), numeric_initial_value), simulated_values])
    paths_df = pd.DataFrame(
        paths,
        index=pd.Index(range(horizon_days + 1), name="day"),
        columns=[f"simulation_{index + 1}" for index in range(n_simulations)],
    )

    final_values = paths_df.iloc[-1].to_numpy(dtype=float)
    summary = {
        "mean_final_value": float(np.mean(final_values)),
        "median_final_value": float(np.median(final_values)),
        "percentile_5": float(np.percentile(final_values, 5)),
        "percentile_95": float(np.percentile(final_values, 95)),
        "max_loss_simulated": float(max(0.0, numeric_initial_value - paths_df.min().min())),
    }
    if numeric_target_value is not None:
        summary["probability_target_reached"] = float(
            np.mean(final_values >= numeric_target_value)
        )

    _save_simulation_results(paths_df, summary)
    return paths_df, summary


__all__ = ["historical_bootstrap_monte_carlo"]
