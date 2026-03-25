"""Full ML pipeline for loan eligibility prediction.

This standalone script orchestrates the entire workflow:
1. Data ingestion and train/holdout split
2. Feature engineering and preprocessing
3. Model training with hyperparameter tuning
4. Evaluation and best-model selection
5. Saving artifacts (models, results, predictions)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Ensure src/ is importable regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.ingestion import load_data, split_data
from src.preprocessing import build_preprocessor, prepare_data
from src.training import train_all_models
from src.evaluation import (
    evaluate_all_models,
    get_best_model,
    save_model,
    save_results,
)
from src.utils import ensure_dir


def main():
    """Run the complete loan-prediction ML pipeline."""

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data_path = os.path.join(PROJECT_ROOT, "data", "input.csv")
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    df = load_data(data_path)

    # ------------------------------------------------------------------
    # 2. Split into ML data and holdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Splitting data (holdout = last 200 rows)")
    print("=" * 60)
    ml_data, holdout = split_data(df, holdout_size=200)

    # ------------------------------------------------------------------
    # 3. Prepare features and target for ML data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Preparing features and target")
    print("=" * 60)
    X, y = prepare_data(ml_data)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # ------------------------------------------------------------------
    # 4. Train / validation split (BEFORE fitting the preprocessor)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Train / validation split (80/20, stratified)")
    print("=" * 60)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"X_train_raw: {X_train_raw.shape}, X_val_raw: {X_val_raw.shape}")
    print(f"y_train distribution:\n{y_train.value_counts()}")
    print(f"y_val distribution:\n{y_val.value_counts()}")

    # ------------------------------------------------------------------
    # 5. Build preprocessor and fit on TRAINING data only (no leakage)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Building preprocessor and transforming features")
    print("=" * 60)
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

    # ------------------------------------------------------------------
    # 6. Train all models
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Training models")
    print("=" * 60)
    trained_models = train_all_models(X_train, y_train)

    # ------------------------------------------------------------------
    # 7. Evaluate all models on validation set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: Evaluating models on validation set")
    print("=" * 60)
    results = evaluate_all_models(trained_models, X_val, y_val)

    # ------------------------------------------------------------------
    # 8. Select best model by F1
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8: Selecting best model")
    print("=" * 60)
    best_name, best_model = get_best_model(results, trained_models)

    # ------------------------------------------------------------------
    # 9. Save model artifacts
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 9: Saving model artifacts")
    print("=" * 60)
    models_dir = os.path.join(PROJECT_ROOT, "models")
    ensure_dir(models_dir)

    save_results(results, os.path.join(models_dir, "model_results.pkl"))

    # Get feature names from the fitted preprocessor for the dashboard
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = None

    save_model(
        best_model,
        preprocessor,
        os.path.join(models_dir, "best_model.pkl"),
        best_name=best_name,
        feature_names=feature_names,
    )

    # ------------------------------------------------------------------
    # 10. Generate and save predictions
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 10: Generating predictions")
    print("=" * 60)
    preds_dir = os.path.join(PROJECT_ROOT, "predictions")
    ensure_dir(preds_dir)

    # --- Train predictions ---
    train_preds = best_model.predict(X_train)
    train_proba = best_model.predict_proba(X_train)[:, 1]
    train_df = pd.DataFrame(
        {
            "y_true": y_train.values,
            "y_pred": train_preds,
            "y_proba": train_proba,
        }
    )
    train_csv_path = os.path.join(preds_dir, "train_predictions.csv")
    train_df.to_csv(train_csv_path, index=False)
    print(f"Train predictions saved to {train_csv_path}")

    # --- Validation predictions ---
    val_preds = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]
    val_df = pd.DataFrame(
        {
            "y_true": y_val.values,
            "y_pred": val_preds,
            "y_proba": val_proba,
        }
    )
    val_csv_path = os.path.join(preds_dir, "validation_predictions.csv")
    val_df.to_csv(val_csv_path, index=False)
    print(f"Validation predictions saved to {val_csv_path}")

    # --- Holdout predictions (NO fitting on holdout!) ---
    X_holdout, y_holdout = prepare_data(holdout)
    X_holdout_processed = preprocessor.transform(X_holdout)  # transform only
    holdout_preds = best_model.predict(X_holdout_processed)
    holdout_proba = best_model.predict_proba(X_holdout_processed)[:, 1]
    holdout_df = pd.DataFrame(
        {
            "y_true": y_holdout.values,
            "y_pred": holdout_preds,
            "y_proba": holdout_proba,
        }
    )
    holdout_csv_path = os.path.join(preds_dir, "holdout_predictions.csv")
    holdout_df.to_csv(holdout_csv_path, index=False)
    print(f"Holdout predictions saved to {holdout_csv_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Best model: {best_name}")
    print(f"  F1:       {results[best_name]['f1']:.4f}")
    print(f"  Accuracy: {results[best_name]['accuracy']:.4f}")
    print(f"  ROC AUC:  {results[best_name]['roc_auc']:.4f}")
    print(f"\nArtifacts saved in: {models_dir}")
    print(f"Predictions saved in: {preds_dir}")

    return results, trained_models, best_name, best_model


if __name__ == "__main__":
    main()
