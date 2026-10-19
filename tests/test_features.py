"""Tests for feature engineering pipeline."""
from pathlib import Path

import joblib
import pandas as pd
import pytest

PROCESSED_DATA_PATH = Path("data/processed/featured_data.csv")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")


@pytest.fixture
def processed_data():
    if not PROCESSED_DATA_PATH.exists():
        pytest.skip("Processed data not available (run 'dvc repro' first)")
    return pd.read_csv(PROCESSED_DATA_PATH)


@pytest.fixture
def preprocessor():
    if not PREPROCESSOR_PATH.exists():
        pytest.skip("Preprocessor not available (run 'dvc repro' first)")
    return joblib.load(PREPROCESSOR_PATH)


class TestProcessedData:
    def test_target_column_present(self, processed_data):
        assert "faulty" in processed_data.columns

    def test_no_null_values(self, processed_data):
        assert processed_data.isnull().sum().sum() == 0

    def test_no_raw_categorical_columns(self, processed_data):
        """After encoding, no object-type columns should remain (except target)."""
        object_cols = processed_data.select_dtypes(include=["object"]).columns.tolist()
        assert len(object_cols) == 0

    def test_numerical_features_are_scaled(self, processed_data):
        """Scaled features should have mean near 0 and std near 1."""
        for col in ["temperature", "pressure", "vibration", "humidity"]:
            assert abs(processed_data[col].mean()) < 0.5
            assert 0.5 < processed_data[col].std() < 2.0


class TestPreprocessor:
    def test_preprocessor_has_transformers(self, preprocessor):
        assert hasattr(preprocessor, "transformers_")

    def test_preprocessor_transforms_valid_input(self, preprocessor):
        sample = pd.DataFrame([{
            "temperature": 70.0,
            "pressure": 35.0,
            "vibration": 1.5,
            "humidity": 50.0,
            "equipment": "Turbine",
            "location": "Atlanta",
        }])
        result = preprocessor.transform(sample)
        assert result.shape[0] == 1
        assert result.shape[1] > 4  # 4 numerical + one-hot encoded categoricals
