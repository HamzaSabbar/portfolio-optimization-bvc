from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.simulation as simulation


def _make_returns_and_allocation():
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03, 0.00],
            "B": [0.00, 0.01, -0.01, 0.02],
            "EXTRA": [0.10, 0.10, 0.10, 0.10],
        }
    )
    allocation = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "weight": [0.60, 0.40],
        }
    )
    return returns, allocation


def test_historical_bootstrap_monte_carlo_output_shape_and_saved_files(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        simulation,
        "SIMULATION_PATHS_OUTPUT_PATH",
        tmp_path / "simulation_paths.csv",
    )
    monkeypatch.setattr(
        simulation,
        "SIMULATION_SUMMARY_OUTPUT_PATH",
        tmp_path / "simulation_summary.csv",
    )
    returns, allocation = _make_returns_and_allocation()

    paths, summary = simulation.historical_bootstrap_monte_carlo(
        returns,
        allocation,
        initial_value=1000,
        horizon_days=10,
        n_simulations=25,
        target_value=1010,
        random_state=7,
    )

    assert paths.shape == (11, 25)
    assert paths.index.name == "day"
    assert paths.iloc[0].tolist() == pytest.approx([1000.0] * 25)
    assert (tmp_path / "simulation_paths.csv").exists()
    assert (tmp_path / "simulation_summary.csv").exists()
    assert set(summary) == {
        "mean_final_value",
        "median_final_value",
        "percentile_5",
        "percentile_95",
        "max_loss_simulated",
        "probability_target_reached",
    }


def test_historical_bootstrap_monte_carlo_summary_metrics_match_paths(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        simulation,
        "SIMULATION_PATHS_OUTPUT_PATH",
        tmp_path / "simulation_paths.csv",
    )
    monkeypatch.setattr(
        simulation,
        "SIMULATION_SUMMARY_OUTPUT_PATH",
        tmp_path / "simulation_summary.csv",
    )
    returns, allocation = _make_returns_and_allocation()

    paths, summary = simulation.historical_bootstrap_monte_carlo(
        returns,
        allocation,
        initial_value=1000,
        horizon_days=5,
        n_simulations=10,
        target_value=1000,
        random_state=3,
    )
    final_values = paths.iloc[-1].to_numpy(dtype=float)

    assert summary["mean_final_value"] == pytest.approx(np.mean(final_values))
    assert summary["median_final_value"] == pytest.approx(np.median(final_values))
    assert summary["percentile_5"] == pytest.approx(np.percentile(final_values, 5))
    assert summary["percentile_95"] == pytest.approx(np.percentile(final_values, 95))
    assert summary["max_loss_simulated"] == pytest.approx(max(0.0, 1000 - paths.min().min()))
    assert summary["probability_target_reached"] == pytest.approx(
        np.mean(final_values >= 1000)
    )


def test_historical_bootstrap_uses_empirical_portfolio_returns_only(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        simulation,
        "SIMULATION_PATHS_OUTPUT_PATH",
        tmp_path / "simulation_paths.csv",
    )
    monkeypatch.setattr(
        simulation,
        "SIMULATION_SUMMARY_OUTPUT_PATH",
        tmp_path / "simulation_summary.csv",
    )
    returns, allocation = _make_returns_and_allocation()
    expected_portfolio_returns = {
        round(value, 10)
        for value in (returns[["A", "B"]].to_numpy() @ np.array([0.60, 0.40]))
    }

    paths, _ = simulation.historical_bootstrap_monte_carlo(
        returns,
        allocation,
        initial_value=1000,
        horizon_days=8,
        n_simulations=12,
        random_state=11,
    )
    simulated_daily_returns = paths.pct_change().iloc[1:].to_numpy().ravel()

    assert {
        round(value, 10) for value in simulated_daily_returns
    }.issubset(expected_portfolio_returns)


def test_historical_bootstrap_omits_probability_when_no_target(tmp_path, monkeypatch):
    monkeypatch.setattr(
        simulation,
        "SIMULATION_PATHS_OUTPUT_PATH",
        tmp_path / "simulation_paths.csv",
    )
    monkeypatch.setattr(
        simulation,
        "SIMULATION_SUMMARY_OUTPUT_PATH",
        tmp_path / "simulation_summary.csv",
    )
    returns, allocation = _make_returns_and_allocation()

    _, summary = simulation.historical_bootstrap_monte_carlo(
        returns,
        allocation,
        initial_value=1000,
        horizon_days=3,
        n_simulations=5,
    )

    assert "probability_target_reached" not in summary


def test_historical_bootstrap_rejects_invalid_initial_value(tmp_path, monkeypatch):
    monkeypatch.setattr(
        simulation,
        "SIMULATION_PATHS_OUTPUT_PATH",
        tmp_path / "simulation_paths.csv",
    )
    monkeypatch.setattr(
        simulation,
        "SIMULATION_SUMMARY_OUTPUT_PATH",
        tmp_path / "simulation_summary.csv",
    )
    returns, allocation = _make_returns_and_allocation()

    with pytest.raises(ValueError, match="initial_value must be greater than 0"):
        simulation.historical_bootstrap_monte_carlo(
            returns,
            allocation,
            initial_value=0,
        )


def test_historical_bootstrap_rejects_invalid_horizon_days(tmp_path, monkeypatch):
    monkeypatch.setattr(
        simulation,
        "SIMULATION_PATHS_OUTPUT_PATH",
        tmp_path / "simulation_paths.csv",
    )
    monkeypatch.setattr(
        simulation,
        "SIMULATION_SUMMARY_OUTPUT_PATH",
        tmp_path / "simulation_summary.csv",
    )
    returns, allocation = _make_returns_and_allocation()

    with pytest.raises(ValueError, match="horizon_days must be a positive integer"):
        simulation.historical_bootstrap_monte_carlo(
            returns,
            allocation,
            initial_value=1000,
            horizon_days=0,
        )


def test_historical_bootstrap_rejects_invalid_n_simulations(tmp_path, monkeypatch):
    monkeypatch.setattr(
        simulation,
        "SIMULATION_PATHS_OUTPUT_PATH",
        tmp_path / "simulation_paths.csv",
    )
    monkeypatch.setattr(
        simulation,
        "SIMULATION_SUMMARY_OUTPUT_PATH",
        tmp_path / "simulation_summary.csv",
    )
    returns, allocation = _make_returns_and_allocation()

    with pytest.raises(ValueError, match="n_simulations must be a positive integer"):
        simulation.historical_bootstrap_monte_carlo(
            returns,
            allocation,
            initial_value=1000,
            n_simulations=0,
        )


def test_historical_bootstrap_rejects_allocation_weights_not_close_to_one(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        simulation,
        "SIMULATION_PATHS_OUTPUT_PATH",
        tmp_path / "simulation_paths.csv",
    )
    monkeypatch.setattr(
        simulation,
        "SIMULATION_SUMMARY_OUTPUT_PATH",
        tmp_path / "simulation_summary.csv",
    )
    returns, allocation = _make_returns_and_allocation()
    allocation["weight"] = [0.60, 0.30]

    with pytest.raises(ValueError, match="allocation weights must sum close to 1"):
        simulation.historical_bootstrap_monte_carlo(
            returns,
            allocation,
            initial_value=1000,
        )


def test_historical_bootstrap_rejects_aligned_weights_not_close_to_one(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        simulation,
        "SIMULATION_PATHS_OUTPUT_PATH",
        tmp_path / "simulation_paths.csv",
    )
    monkeypatch.setattr(
        simulation,
        "SIMULATION_SUMMARY_OUTPUT_PATH",
        tmp_path / "simulation_summary.csv",
    )
    returns, allocation = _make_returns_and_allocation()
    allocation = pd.concat(
        [
            allocation,
            pd.DataFrame({"ticker": ["NOT_IN_RETURNS"], "weight": [0.20]}),
        ],
        ignore_index=True,
    )
    allocation.loc[allocation["ticker"] == "A", "weight"] = 0.48
    allocation.loc[allocation["ticker"] == "B", "weight"] = 0.32

    with pytest.raises(ValueError, match="Aligned allocation weights must sum close to 1"):
        simulation.historical_bootstrap_monte_carlo(
            returns,
            allocation,
            initial_value=1000,
        )
