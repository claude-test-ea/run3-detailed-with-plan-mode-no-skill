"""Unit tests for the ingestion module."""
import os
import sys

import pandas as pd
import pytest

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.ingestion import load_data, split_data

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "input.csv")


class TestLoadData:
    """Tests for the load_data function."""

    def test_returns_dataframe(self):
        """load_data should return a pandas DataFrame."""
        df = load_data(DATA_PATH)
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_not_empty(self):
        """Loaded DataFrame should contain rows."""
        df = load_data(DATA_PATH)
        assert len(df) > 0

    def test_expected_columns_present(self):
        """DataFrame should contain key columns from the loan dataset."""
        df = load_data(DATA_PATH)
        expected = ["Loan_ID", "Gender", "Loan_Status", "ApplicantIncome"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"


class TestSplitData:
    """Tests for the split_data function."""

    def test_split_shapes(self):
        """ML data + holdout row counts should equal original row count."""
        df = load_data(DATA_PATH)
        ml_data, holdout = split_data(df, holdout_size=200)
        assert len(ml_data) + len(holdout) == len(df)

    def test_holdout_size(self):
        """Holdout should have exactly the requested number of rows."""
        df = load_data(DATA_PATH)
        ml_data, holdout = split_data(df, holdout_size=200)
        assert len(holdout) == 200

    def test_ml_data_size(self):
        """ML data should have total rows minus holdout_size rows."""
        df = load_data(DATA_PATH)
        holdout_size = 200
        ml_data, holdout = split_data(df, holdout_size=holdout_size)
        assert len(ml_data) == len(df) - holdout_size

    def test_no_index_overlap(self):
        """ML data and holdout should not share indices."""
        df = load_data(DATA_PATH)
        ml_data, holdout = split_data(df, holdout_size=200)
        overlap = set(ml_data.index) & set(holdout.index)
        assert len(overlap) == 0
