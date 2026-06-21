from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import clean_prices, save_clean_prices


def test_clean_prices_standardizes_converts_filters_and_sorts():
    raw_data = pd.DataFrame(
        {
            "Date": [
                "03/01/2024",
                "02/01/2024",
                "02/01/2024",
                "04/01/2024",
                "05/01/2024",
                None,
            ],
            "ticker": ["BCP", "BCP", "BCP", "CIH", "CIH", "BOA"],
            "Value": ["276.0", "275.0", "275.0", "0", None, "10"],
            "Min": ["275.0", "274.5", "274.5", "0", "290.0", "9.0"],
            "Max": ["276.0", "275.0", "275.0", "0", "300.0", "11.0"],
            "Volume": ["1000", "500", "500", "100", "200", "300"],
            "Variation": ["0.36", "0.1", "0.1", "0", "1.0", "2.0"],
        }
    )

    cleaned = clean_prices(raw_data)

    assert list(cleaned.columns) == [
        "date",
        "ticker",
        "close",
        "low",
        "high",
        "volume",
        "variation",
    ]
    assert len(cleaned) == 2
    assert cleaned["ticker"].tolist() == ["BCP", "BCP"]
    assert cleaned["date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert cleaned["close"].tolist() == [275.0, 276.0]
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    for column in ["close", "low", "high", "volume", "variation"]:
        assert pd.api.types.is_numeric_dtype(cleaned[column])
    assert (cleaned["close"] > 0).all()
    assert not cleaned.duplicated().any()


def test_clean_prices_rejects_missing_required_columns():
    raw_data = pd.DataFrame({"date": ["2024-01-02"], "Value": [275.0]})

    with pytest.raises(ValueError, match="Missing required columns"):
        clean_prices(raw_data)


def test_clean_prices_accepts_already_standardized_columns_and_deduplicates_by_date_ticker():
    raw_data = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "ticker": ["BCP", "BCP", "BCP"],
            "close": ["270", "275", "280"],
            "low": ["269", "274", "279"],
            "high": ["271", "276", "281"],
            "volume": ["100", "200", "300"],
            "variation": ["0.1", "0.2", "0.3"],
        }
    )

    cleaned = clean_prices(raw_data)

    assert cleaned["close"].tolist() == [275, 280]
    assert cleaned["volume"].tolist() == [200, 300]
    assert not cleaned.duplicated(subset=["date", "ticker"]).any()


def test_save_clean_prices_writes_csv(tmp_path):
    cleaned = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")],
            "ticker": ["BCP"],
            "close": [275.0],
            "low": [274.5],
            "high": [275.0],
            "volume": [500],
            "variation": [0.1],
        }
    )
    output_path = tmp_path / "clean_prices.csv"

    saved_path = save_clean_prices(cleaned, output_path)
    saved = pd.read_csv(saved_path)

    assert saved_path == output_path
    assert saved.columns.tolist() == [
        "date",
        "ticker",
        "close",
        "low",
        "high",
        "volume",
        "variation",
    ]
    assert saved.loc[0, "ticker"] == "BCP"
