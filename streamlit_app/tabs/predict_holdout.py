"""Predict on Hold-Out tab for the Streamlit dashboard.

Loads holdout predictions and displays them alongside actual labels,
summary metrics, and a downloadable CSV. Gilded Observatory theme.
"""
import os

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def render(base_dir):
    """Render the Predict on Hold-Out tab.

    Parameters
    ----------
    base_dir : str
        Project root directory path, used to locate the
        predictions/holdout_predictions.csv file.
    """
    st.header("Predict on Hold-Out Set")

    predictions_path = os.path.join(base_dir, "predictions", "holdout_predictions.csv")

    if not os.path.exists(predictions_path):
        st.warning(
            "Holdout predictions file not found at "
            f"`predictions/holdout_predictions.csv`. "
            "Run the pipeline first to generate predictions."
        )
        return

    holdout_df = pd.read_csv(predictions_path)

    if holdout_df.empty:
        st.warning("Holdout predictions file is empty.")
        return

    st.subheader("Hold-Out Predictions")

    # Build display DataFrame
    display_df = holdout_df.copy()

    # Detect column names (handle both possible naming conventions)
    pred_col = None
    proba_col = None
    actual_col = None

    for col in display_df.columns:
        col_lower = col.lower()
        if "pred" in col_lower and "proba" not in col_lower:
            pred_col = col
        elif "proba" in col_lower or "probability" in col_lower:
            proba_col = col
        elif col == "Loan_Status" or "actual" in col_lower or "true" in col_lower:
            actual_col = col

    # Add match indicator if both predicted and actual columns exist
    if pred_col and actual_col:
        def _normalize_label(val):
            """Normalize loan status labels to Y/N string."""
            if isinstance(val, (int, float, np.integer, np.floating)):
                return "Y" if val == 1 else "N"
            return str(val).strip()

        actual_norm = display_df[actual_col].apply(_normalize_label)
        pred_norm = display_df[pred_col].apply(_normalize_label)

        display_df["Match"] = [
            "Correct" if a == p else "Wrong"
            for a, p in zip(actual_norm, pred_norm)
        ]

    st.dataframe(display_df, use_container_width=True, height=450)

    # ---- Summary Metrics ----
    if pred_col and actual_col:
        st.subheader("Hold-Out Summary Metrics")

        def _to_numeric(series):
            """Convert a label series to numeric 0/1."""
            return series.apply(
                lambda v: 1
                if (isinstance(v, (int, float, np.integer, np.floating)) and v == 1)
                or str(v).strip().upper() == "Y"
                else 0
            )

        y_actual = _to_numeric(display_df[actual_col])
        y_pred = _to_numeric(display_df[pred_col])

        acc = accuracy_score(y_actual, y_pred)
        f1 = f1_score(y_actual, y_pred, zero_division=0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{acc:.4f}")
        with col2:
            st.metric("F1 Score", f"{f1:.4f}")
        with col3:
            total = len(display_df)
            correct = int((y_actual == y_pred).sum())
            st.metric("Correct / Total", f"{correct} / {total}")

    # ---- Download Button ----
    st.subheader("Download Predictions")
    csv_data = display_df.to_csv(index=False)
    st.download_button(
        label="Download Hold-Out Predictions as CSV",
        data=csv_data,
        file_name="holdout_predictions.csv",
        mime="text/csv",
    )
