"""Unit tests for the training module."""
import os
import sys

import pytest

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.training import get_grid_params, get_models


class TestGetModels:
    """Tests for the get_models function."""

    def test_returns_dict(self):
        """get_models should return a dictionary."""
        models = get_models()
        assert isinstance(models, dict)

    def test_contains_five_models(self):
        """There should be exactly 5 candidate models."""
        models = get_models()
        assert len(models) == 5

    def test_expected_model_names(self):
        """All five expected model names should be present."""
        models = get_models()
        expected_names = {
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting",
            "SVM",
            "KNN",
        }
        assert set(models.keys()) == expected_names

    def test_models_have_fit_method(self):
        """Each model should have a .fit() method."""
        models = get_models()
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} missing .fit()"


class TestGetGridParams:
    """Tests for the get_grid_params function."""

    def test_returns_dict(self):
        """get_grid_params should return a dictionary."""
        grid_params = get_grid_params()
        assert isinstance(grid_params, dict)

    def test_contains_random_forest(self):
        """Grid params should include Random Forest."""
        grid_params = get_grid_params()
        assert "Random Forest" in grid_params

    def test_contains_gradient_boosting(self):
        """Grid params should include Gradient Boosting."""
        grid_params = get_grid_params()
        assert "Gradient Boosting" in grid_params

    def test_random_forest_params(self):
        """Random Forest grid should have n_estimators, max_depth, min_samples_split."""
        grid_params = get_grid_params()
        rf_params = grid_params["Random Forest"]
        assert "n_estimators" in rf_params
        assert "max_depth" in rf_params
        assert "min_samples_split" in rf_params

    def test_gradient_boosting_params(self):
        """Gradient Boosting grid should have n_estimators, max_depth, learning_rate."""
        grid_params = get_grid_params()
        gb_params = grid_params["Gradient Boosting"]
        assert "n_estimators" in gb_params
        assert "max_depth" in gb_params
        assert "learning_rate" in gb_params
