"""Unit tests for the preprocessing module."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.ingestion import load_data, split_data
from src.preprocessing import (
    ALL_NUMERICAL,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    add_features,
    build_preprocessor,
    prepare_data,
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "input.csv")


@pytest.fixture
def sample_df():
    """Load a small sample of the dataset for testing."""
    df = load_data(DATA_PATH)
    ml_data, _ = split_data(df, holdout_size=200)
    return ml_data


class TestAddFeatures:
    """Tests for the add_features function."""

    def test_creates_total_income(self, sample_df):
        """add_features should create a TotalIncome column."""
        X, _ = prepare_data(sample_df)
        assert "TotalIncome" in X.columns

    def test_creates_income_to_loan(self, sample_df):
        """add_features should create an IncomeToLoan column."""
        X, _ = prepare_data(sample_df)
        assert "IncomeToLoan" in X.columns

    def test_creates_loan_amount_log(self, sample_df):
        """add_features should create a LoanAmountLog column."""
        X, _ = prepare_data(sample_df)
        assert "LoanAmountLog" in X.columns

    def test_total_income_calculation(self):
        """TotalIncome should equal ApplicantIncome + CoapplicantIncome."""
        df = pd.DataFrame(
            {
                "ApplicantIncome": [5000, 3000],
                "CoapplicantIncome": [1000, 2000],
                "LoanAmount": [100, 200],
            }
        )
        result = add_features(df)
        assert list(result["TotalIncome"]) == [6000, 5000]

    def test_income_to_loan_handles_zero(self):
        """IncomeToLoan should be 0 when LoanAmount is 0."""
        df = pd.DataFrame(
            {
                "ApplicantIncome": [5000],
                "CoapplicantIncome": [1000],
                "LoanAmount": [0],
            }
        )
        result = add_features(df)
        assert result["IncomeToLoan"].iloc[0] == 0

    def test_income_to_loan_handles_nan(self):
        """IncomeToLoan should be 0 when LoanAmount is NaN."""
        df = pd.DataFrame(
            {
                "ApplicantIncome": [5000],
                "CoapplicantIncome": [1000],
                "LoanAmount": [np.nan],
            }
        )
        result = add_features(df)
        assert result["IncomeToLoan"].iloc[0] == 0


class TestPrepareData:
    """Tests for the prepare_data function."""

    def test_returns_x_and_y(self, sample_df):
        """prepare_data should return a tuple of (X, y)."""
        X, y = prepare_data(sample_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_loan_id_dropped(self, sample_df):
        """X should not contain Loan_ID."""
        X, _ = prepare_data(sample_df)
        assert "Loan_ID" not in X.columns

    def test_target_encoded(self, sample_df):
        """y should contain only 0 and 1 values."""
        _, y = prepare_data(sample_df)
        assert set(y.unique()).issubset({0, 1})

    def test_loan_status_not_in_x(self, sample_df):
        """X should not contain the target column Loan_Status."""
        X, _ = prepare_data(sample_df)
        assert "Loan_Status" not in X.columns


class TestBuildPreprocessor:
    """Tests for the build_preprocessor function."""

    def test_returns_column_transformer(self):
        """build_preprocessor should return a ColumnTransformer."""
        preprocessor = build_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_has_two_transformers(self):
        """Preprocessor should have numerical and categorical transformers."""
        preprocessor = build_preprocessor()
        assert len(preprocessor.transformers) == 2

    def test_fit_transform_produces_output(self, sample_df):
        """Preprocessor should fit_transform without error and produce a 2D array."""
        X, _ = prepare_data(sample_df)
        preprocessor = build_preprocessor()
        X_processed = preprocessor.fit_transform(X)
        assert X_processed.ndim == 2
        assert X_processed.shape[0] == len(X)
