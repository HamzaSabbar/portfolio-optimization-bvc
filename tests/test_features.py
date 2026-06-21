from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    FEATURE_COLUMNS,
    add_target_variable,
    build_financial_features,
    calculate_daily_returns,
    main as features_main,
    save_features,
    save_returns,
    save_scoring_dataset,
)


def test_calculate_daily_returns_returns_dates_index_and_tickers_columns():
    clean_prices = pd.DataFrame(
        {
            "date": [
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-01",
            ],
            "ticker": ["BCP", "BCP", "BCP", "CIH", "CIH"],
            "close": [121.0, 100.0, 110.0, 180.0, 200.0],
        }
    )

    returns = calculate_daily_returns(clean_prices)

    assert returns.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert returns.columns.tolist() == ["BCP", "CIH"]
    assert returns.index.name == "date"
    assert returns.loc[pd.Timestamp("2024-01-02"), "BCP"] == pytest.approx(0.10)
    assert returns.loc[pd.Timestamp("2024-01-02"), "CIH"] == pytest.approx(-0.10)
    assert returns.loc[pd.Timestamp("2024-01-03"), "BCP"] == pytest.approx(0.10)
    assert pd.isna(returns.loc[pd.Timestamp("2024-01-03"), "CIH"])


def test_calculate_daily_returns_rejects_missing_columns():
    clean_prices = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "ticker": ["BCP"],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        calculate_daily_returns(clean_prices)


def test_calculate_daily_returns_rejects_duplicate_ticker_dates():
    clean_prices = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "ticker": ["BCP", "BCP"],
            "close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate rows"):
        calculate_daily_returns(clean_prices)


def test_calculate_daily_returns_reports_and_ignores_invalid_prices(capsys):
    clean_prices = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "ticker": ["BCP", "BCP", "BCP"],
            "close": [100.0, 0.0, 110.0],
        }
    )

    returns = calculate_daily_returns(clean_prices)
    stdout = capsys.readouterr().out

    assert "WARNING: removed 1 rows with invalid required market data" in stdout
    assert returns.index.tolist() == [pd.Timestamp("2024-01-03")]
    assert returns.loc[pd.Timestamp("2024-01-03"), "BCP"] == pytest.approx(0.10)


def test_save_returns_writes_csv(tmp_path):
    returns = pd.DataFrame(
        {"BCP": [0.1], "CIH": [-0.1]},
        index=pd.Index([pd.Timestamp("2024-01-02")], name="date"),
    )
    output_path = tmp_path / "returns.csv"

    saved_path = save_returns(returns, output_path)
    saved = pd.read_csv(saved_path)

    assert saved_path == output_path
    assert saved.columns.tolist() == ["date", "BCP", "CIH"]
    assert saved.loc[0, "BCP"] == pytest.approx(0.1)


def test_daily_returns_cli_writes_output_and_prints_summary(tmp_path, capsys):
    clean_prices = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "ticker": ["BCP", "BCP", "CIH", "CIH"],
            "close": [100.0, 110.0, 200.0, 180.0],
        }
    )
    input_path = tmp_path / "clean_prices.csv"
    output_path = tmp_path / "returns.csv"
    clean_prices.to_csv(input_path, index=False)

    exit_code = features_main(["--input", str(input_path), "--output", str(output_path)])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert output_path.exists()
    assert "Output shape:" in stdout
    assert "Missing values summary:" in stdout


def _make_feature_prices(ticker="BCP", periods=80, future_multiplier=1.0):
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    close = pd.Series([100.0 + index for index in range(periods)])
    if future_multiplier != 1.0:
        close.iloc[60:] = close.iloc[60:] * future_multiplier

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "close": close,
            "volume": [1000 + index for index in range(periods)],
        }
    )


def test_build_financial_features_returns_expected_long_format_columns():
    clean_prices = pd.concat(
        [
            _make_feature_prices("BCP"),
            _make_feature_prices("CIH"),
        ],
        ignore_index=True,
    )

    features = build_financial_features(clean_prices)

    assert features.columns.tolist() == [
        "date",
        "ticker",
        "close",
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
    assert features["ticker"].tolist() == sorted(features["ticker"].tolist())
    assert not features.drop(columns=["date", "ticker"]).isna().any().any()


def test_build_financial_features_calculates_values_using_past_data_only():
    clean_prices = _make_feature_prices("BCP")
    features = build_financial_features(clean_prices)
    target_date = pd.Timestamp("2024-02-29")
    row = features.loc[features["date"] == target_date].iloc[0]

    price_history = clean_prices.sort_values("date").reset_index(drop=True)
    target_index = price_history.index[price_history["date"] == target_date][0]
    close = price_history["close"]
    volume = price_history["volume"]
    daily_returns = close.pct_change(fill_method=None)

    assert row["close"] == pytest.approx(close.iloc[target_index])
    assert row["past_return_20d"] == pytest.approx(
        close.iloc[target_index] / close.iloc[target_index - 20] - 1
    )
    assert row["momentum_20d"] == pytest.approx(
        close.iloc[target_index] - close.iloc[target_index - 20]
    )
    assert row["volatility_20d"] == pytest.approx(
        daily_returns.iloc[target_index - 19 : target_index + 1].std()
    )
    assert row["moving_average_20d"] == pytest.approx(
        close.iloc[target_index - 19 : target_index + 1].mean()
    )
    assert row["moving_average_50d"] == pytest.approx(
        close.iloc[target_index - 49 : target_index + 1].mean()
    )
    assert row["average_volume_20d"] == pytest.approx(
        volume.iloc[target_index - 19 : target_index + 1].mean()
    )
    assert row["drawdown"] == pytest.approx(0.0)
    assert row["rsi_14d"] == pytest.approx(100.0)


def test_build_financial_features_does_not_change_past_features_when_future_changes():
    base_prices = _make_feature_prices("BCP")
    changed_future_prices = _make_feature_prices("BCP", future_multiplier=10.0)
    target_date = pd.Timestamp("2024-02-29")

    base_features = build_financial_features(base_prices)
    changed_features = build_financial_features(changed_future_prices)

    base_row = base_features.loc[base_features["date"] == target_date].iloc[0]
    changed_row = changed_features.loc[changed_features["date"] == target_date].iloc[0]

    feature_columns = [
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
    for column in feature_columns:
        assert changed_row[column] == pytest.approx(base_row[column])


def test_build_financial_features_rejects_missing_volume():
    clean_prices = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60),
            "ticker": ["BCP"] * 60,
            "close": [100.0 + index for index in range(60)],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        build_financial_features(clean_prices)


def test_build_financial_features_rejects_short_history_with_clear_error():
    clean_prices = _make_feature_prices("BCP", periods=69)

    with pytest.raises(ValueError, match="At least 70\\+ observations per ticker"):
        build_financial_features(clean_prices)


def test_save_features_writes_csv_and_prints_summary(tmp_path, capsys):
    features = build_financial_features(_make_feature_prices("BCP", periods=80))
    output_path = tmp_path / "features.csv"

    saved_path = save_features(features, output_path)
    stdout = capsys.readouterr().out
    saved = pd.read_csv(saved_path)

    assert saved_path == output_path
    assert saved.columns.tolist() == ["date", "ticker", "close", *FEATURE_COLUMNS]
    assert "Output shape:" in stdout
    assert "Rows per ticker after feature generation:" in stdout
    assert "Feature distributions:" in stdout


def test_financial_features_cli_writes_output(tmp_path):
    clean_prices = _make_feature_prices("BCP", periods=80)
    input_path = tmp_path / "clean_prices.csv"
    output_path = tmp_path / "features.csv"
    clean_prices.to_csv(input_path, index=False)

    exit_code = features_main(
        ["--mode", "features", "--input", str(input_path), "--output", str(output_path)]
    )

    assert exit_code == 0
    assert output_path.exists()


def _make_scoring_features(ticker, closes):
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
            "ticker": [ticker] * len(closes),
            "close": closes,
            "past_return_20d": [0.01] * len(closes),
        }
    )


def test_add_target_variable_builds_future_return_and_target_by_ticker():
    bcp_closes = [100.0] * 20 + [103.0, 101.0]
    cih_closes = [200.0] * 20 + [210.0, 201.0]
    features = pd.concat(
        [
            _make_scoring_features("CIH", cih_closes),
            _make_scoring_features("BCP", bcp_closes),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=1)

    scoring_dataset = add_target_variable(features)

    assert "future_return_20d" in scoring_dataset.columns
    assert "target" in scoring_dataset.columns
    assert len(scoring_dataset) == 4
    assert scoring_dataset["ticker"].tolist() == ["BCP", "BCP", "CIH", "CIH"]
    assert scoring_dataset.groupby("ticker")["date"].is_monotonic_increasing.all()

    bcp_rows = scoring_dataset[scoring_dataset["ticker"] == "BCP"].reset_index(drop=True)
    cih_rows = scoring_dataset[scoring_dataset["ticker"] == "CIH"].reset_index(drop=True)

    assert bcp_rows.loc[0, "future_return_20d"] == pytest.approx(0.03)
    assert bcp_rows.loc[0, "target"] == 1
    assert bcp_rows.loc[1, "future_return_20d"] == pytest.approx(0.01)
    assert bcp_rows.loc[1, "target"] == 0
    assert cih_rows.loc[0, "future_return_20d"] == pytest.approx(0.05)
    assert cih_rows.loc[0, "target"] == 1
    assert cih_rows.loc[1, "future_return_20d"] == pytest.approx(0.005)
    assert cih_rows.loc[1, "target"] == 0


def test_add_target_variable_supports_custom_horizon_and_threshold():
    features = _make_scoring_features("BCP", [100.0, 101.0, 104.0, 103.0])

    scoring_dataset = add_target_variable(features, horizon_days=2, threshold=0.03)

    assert "future_return_2d" in scoring_dataset.columns
    assert len(scoring_dataset) == 2
    assert scoring_dataset.loc[0, "future_return_2d"] == pytest.approx(0.04)
    assert scoring_dataset.loc[0, "target"] == 1
    assert scoring_dataset.loc[1, "future_return_2d"] == pytest.approx(2 / 101)
    assert scoring_dataset.loc[1, "target"] == 0


def test_add_target_variable_does_not_add_future_return_to_feature_columns():
    features = _make_scoring_features("BCP", [100.0] * 20 + [103.0, 101.0])

    scoring_dataset = add_target_variable(features)

    assert "future_return_20d" in scoring_dataset.columns
    assert "future_return_20d" not in FEATURE_COLUMNS
    assert "target" not in FEATURE_COLUMNS


def test_add_target_variable_warns_single_target_class(capsys):
    features = _make_scoring_features("BCP", [100.0] * 25)

    scoring_dataset = add_target_variable(features)
    stdout = capsys.readouterr().out

    assert scoring_dataset["target"].nunique() == 1
    assert "WARNING: target contains only one class" in stdout


def test_add_target_variable_rejects_duplicate_ticker_dates():
    features = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "ticker": ["BCP", "BCP"],
            "close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate rows"):
        add_target_variable(features)


def test_save_scoring_dataset_writes_csv_and_prints_target_distribution(tmp_path, capsys):
    features = _make_scoring_features("BCP", [100.0] * 20 + [103.0, 101.0])
    scoring_dataset = add_target_variable(features)
    output_path = tmp_path / "scoring_dataset.csv"

    saved_path = save_scoring_dataset(scoring_dataset, output_path)
    stdout = capsys.readouterr().out
    saved = pd.read_csv(saved_path)

    assert saved_path == output_path
    assert "future_return_20d" in saved.columns
    assert "target" in saved.columns
    assert "Output shape:" in stdout
    assert "Target distribution overall:" in stdout
    assert "Target distribution by ticker:" in stdout


def test_target_cli_writes_scoring_dataset(tmp_path):
    features = _make_scoring_features("BCP", [100.0] * 20 + [103.0, 101.0])
    input_path = tmp_path / "features.csv"
    output_path = tmp_path / "scoring_dataset.csv"
    features.to_csv(input_path, index=False)

    exit_code = features_main(
        ["--mode", "target", "--input", str(input_path), "--output", str(output_path)]
    )

    assert exit_code == 0
    assert output_path.exists()
