"""Tests for data ingestion and cleaning."""
from pathlib import Path

import pandas as pd
import pytest

RAW_DATA_PATH = Path("data/raw/industrial_equipment_anomaly_data.csv")
INTERIM_DATA_PATH = Path("data/interim/cleaned_data.csv")


@pytest.fixture
def raw_data():
    """Load raw data if available, skip otherwise."""
    if not RAW_DATA_PATH.exists():
        pytest.skip("Raw data not available (run 'dvc pull' first)")
    return pd.read_csv(RAW_DATA_PATH)


@pytest.fixture
def cleaned_data():
    """Load cleaned data if available, skip otherwise."""
    if not INTERIM_DATA_PATH.exists():
        pytest.skip("Cleaned data not available (run 'dvc repro' first)")
    return pd.read_csv(INTERIM_DATA_PATH)


class TestRawData:
    def test_expected_columns_exist(self, raw_data):
        expected = {"temperature", "pressure", "vibration", "humidity", "equipment", "location", "faulty"}
        assert expected == set(raw_data.columns)

    def test_no_null_values(self, raw_data):
        assert raw_data.isnull().sum().sum() == 0

    def test_target_is_binary(self, raw_data):
        assert set(raw_data["faulty"].unique()).issubset({0, 1})

    def test_valid_equipment_types(self, raw_data):
        valid = {"Turbine", "Compressor", "Pump"}
        assert set(raw_data["equipment"].str.strip().unique()).issubset(valid)

    def test_valid_locations(self, raw_data):
        valid = {"Atlanta", "Chicago", "San Francisco", "New York", "Houston"}
        assert set(raw_data["location"].str.strip().unique()).issubset(valid)


class TestCleanedData:
    def test_target_is_integer(self, cleaned_data):
        assert cleaned_data["faulty"].dtype in ("int64", "int32")

    def test_no_leading_trailing_whitespace(self, cleaned_data):
        for col in cleaned_data.select_dtypes(include=["object"]).columns:
            assert (cleaned_data[col] == cleaned_data[col].str.strip()).all()

    def test_row_count_matches_raw(self, raw_data, cleaned_data):
        assert len(raw_data) == len(cleaned_data)
