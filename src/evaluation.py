"""Evaluation module.

Provides functions for evaluating trained models, selecting the best model,
and saving results and model artifacts.
"""
import pickle

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(name, model, X_val, y_val):
    """Evaluate a single trained model on the validation set.

    Computes a comprehensive set of classification metrics, predicted labels,
    predicted probabilities, and curve data for ROC and Precision-Recall plots.

    Parameters
    ----------
    name : str
        Name of the model (for display/logging purposes).
    model : estimator
        Trained sklearn-compatible model with predict and predict_proba methods.
    X_val : array-like
        Validation feature matrix (already preprocessed).
    y_val : array-like
        Validation target vector.

    Returns
    -------
    dict
        Dictionary containing:
        - accuracy, precision, recall, f1, roc_auc (float)
        - confusion_matrix (np.ndarray)
        - classification_report (str)
        - feature_importances (np.ndarray or None)
        - roc_curve (tuple of fpr, tpr, thresholds)
        - precision_recall_curve (tuple of precision, recall, thresholds)
        - y_pred (np.ndarray)
        - y_proba (np.ndarray)
        - y_val (np.ndarray)
    """
    y_pred = model.predict(X_val)

    # Get predicted probabilities for the positive class
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_proba = np.zeros(len(y_val))

    # Core metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    roc = roc_auc_score(y_val, y_proba)

    # Confusion matrix and classification report
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, zero_division=0)

    # Feature importances (if available)
    if hasattr(model, "feature_importances_"):
        feat_imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        feat_imp = model.coef_[0]
    else:
        feat_imp = None

    # Curve data
    fpr, tpr, roc_thresholds = roc_curve(y_val, y_proba)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_val, y_proba)

    print(f"\n--- {name} ---")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importances": feat_imp,
        "roc_curve": (fpr, tpr, roc_thresholds),
        "precision_recall_curve": (pr_precision, pr_recall, pr_thresholds),
        "y_pred": y_pred,
        "y_proba": y_proba,
        "y_val": np.array(y_val),
    }


def evaluate_all_models(models, X_val, y_val):
    """Evaluate all trained models on the validation set.

    Parameters
    ----------
    models : dict
        Mapping of model name (str) to trained model instance.
    X_val : array-like
        Validation feature matrix (already preprocessed).
    y_val : array-like
        Validation target vector.

    Returns
    -------
    dict
        Mapping of model name (str) to evaluation results dict.
    """
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(name, model, X_val, y_val)
    return results


def get_best_model(results, models):
    """Select the best model based on highest F1 score.

    Parameters
    ----------
    results : dict
        Mapping of model name to evaluation results dict (must contain 'f1' key).
    models : dict
        Mapping of model name to trained model instance.

    Returns
    -------
    tuple of (str, estimator)
        (best_model_name, best_model_instance).
    """
    best_name = max(results, key=lambda name: results[name]["f1"])
    best_model = models[best_name]
    print(f"\nBest model: {best_name} (F1 = {results[best_name]['f1']:.4f})")
    return best_name, best_model


def save_results(results, path):
    """Save evaluation results dictionary to a pickle file.

    Parameters
    ----------
    results : dict
        Evaluation results to persist.
    path : str
        File path for the pickle output.
    """
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {path}")


def save_model(model, preprocessor, path, best_name=None, feature_names=None):
    """Save the trained model and preprocessor to a pickle file.

    The saved dictionary contains keys 'model', 'preprocessor',
    'best_name', and 'feature_names' so that the dashboard can display
    feature importance labels and hyperparameter details.

    Parameters
    ----------
    model : estimator
        Trained model instance.
    preprocessor : transformer
        Fitted sklearn preprocessor (ColumnTransformer).
    path : str
        File path for the pickle output.
    best_name : str, optional
        Name of the best model (e.g. "Random Forest").
    feature_names : array-like, optional
        Feature names from the fitted preprocessor.
    """
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "best_name": best_name,
        "feature_names": feature_names,
    }
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Model saved to {path}")
