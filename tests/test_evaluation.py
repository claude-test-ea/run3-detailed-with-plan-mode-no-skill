"""Unit tests for the evaluation module."""
import os
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation import evaluate_model, evaluate_all_models, get_best_model


@pytest.fixture
def dummy_model_and_data():
    """Create a simple logistic regression model and synthetic data for testing."""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_val = np.random.randn(30, 5)
    y_val = (X_val[:, 0] > 0).astype(int)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model, X_val, y_val


class TestEvaluateModel:
    """Tests for the evaluate_model function."""

    def test_returns_dict(self, dummy_model_and_data):
        """evaluate_model should return a dictionary."""
        model, X_val, y_val = dummy_model_and_data
        result = evaluate_model("Test Model", model, X_val, y_val)
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, dummy_model_and_data):
        """Result should contain all expected metric keys."""
        model, X_val, y_val = dummy_model_and_data
        result = evaluate_model("Test Model", model, X_val, y_val)
        expected_keys = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "confusion_matrix",
            "classification_report",
            "feature_importances",
            "roc_curve",
            "precision_recall_curve",
            "y_pred",
            "y_proba",
            "y_val",
        }
        assert expected_keys == set(result.keys())

    def test_metrics_are_floats(self, dummy_model_and_data):
        """Scalar metrics should be float values."""
        model, X_val, y_val = dummy_model_and_data
        result = evaluate_model("Test Model", model, X_val, y_val)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert isinstance(result[key], float), f"{key} is not a float"

    def test_metrics_in_valid_range(self, dummy_model_and_data):
        """All scalar metrics should be between 0 and 1."""
        model, X_val, y_val = dummy_model_and_data
        result = evaluate_model("Test Model", model, X_val, y_val)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of range"

    def test_classification_report_is_string(self, dummy_model_and_data):
        """classification_report should be a string."""
        model, X_val, y_val = dummy_model_and_data
        result = evaluate_model("Test Model", model, X_val, y_val)
        assert isinstance(result["classification_report"], str)

    def test_y_pred_shape(self, dummy_model_and_data):
        """y_pred should have the same length as y_val."""
        model, X_val, y_val = dummy_model_and_data
        result = evaluate_model("Test Model", model, X_val, y_val)
        assert len(result["y_pred"]) == len(y_val)

    def test_roc_curve_is_tuple(self, dummy_model_and_data):
        """roc_curve should be a tuple of (fpr, tpr, thresholds)."""
        model, X_val, y_val = dummy_model_and_data
        result = evaluate_model("Test Model", model, X_val, y_val)
        assert isinstance(result["roc_curve"], tuple)
        assert len(result["roc_curve"]) == 3


class TestEvaluateAllModels:
    """Tests for evaluate_all_models."""

    def test_returns_dict_per_model(self, dummy_model_and_data):
        """Should return one result dict per model."""
        model, X_val, y_val = dummy_model_and_data
        models = {"Model A": model, "Model B": model}
        results = evaluate_all_models(models, X_val, y_val)
        assert set(results.keys()) == {"Model A", "Model B"}


class TestGetBestModel:
    """Tests for get_best_model."""

    def test_returns_best_by_f1(self, dummy_model_and_data):
        """Should return the model with the highest F1 score."""
        model, X_val, y_val = dummy_model_and_data
        results = {
            "Low F1": {"f1": 0.5},
            "High F1": {"f1": 0.9},
        }
        models = {"Low F1": model, "High F1": model}
        best_name, best_model = get_best_model(results, models)
        assert best_name == "High F1"
