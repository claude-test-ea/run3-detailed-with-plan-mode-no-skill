"""Model Deep Dive tab for the Streamlit dashboard.

Provides per-model analysis including feature importances,
hyperparameters, precision-recall curves, and threshold tuning.
All visualizations use Plotly with the Gilded Observatory dark theme.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score


# ---------------------------------------------------------------------------
# Gilded Observatory chart constants
# ---------------------------------------------------------------------------
COLOR_APPROVED = "#00E5A0"
COLOR_REJECTED = "#FF6161"
ACCENT_GOLD = "#C9A96E"
ACCENT_TEAL = "#00B4D8"
CHART_COLORS = ['#C9A96E', '#00B4D8', '#00E5A0', '#FF6161', '#A78BFA', '#F472B6']

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,22,29,0.6)",
    font=dict(family="Sora, sans-serif", color="#E8E4DD", size=12),
    title_font=dict(family="Cormorant Garamond, serif", size=20, color="#E8E4DD"),
    margin=dict(t=50, b=30, l=40, r=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(gridcolor="rgba(42,42,53,0.5)", zerolinecolor="rgba(42,42,53,0.8)"),
    yaxis=dict(gridcolor="rgba(42,42,53,0.5)", zerolinecolor="rgba(42,42,53,0.8)"),
)

TREE_MODELS = ["Random Forest", "Gradient Boosting"]


def _apply_layout(fig):
    """Apply the standard Gilded Observatory chart layout to a figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to style.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig.update_layout(**CHART_LAYOUT)
    return fig


def _compute_metrics_at_threshold(y_true, y_proba, threshold):
    """Compute precision, recall, and F1 at a given probability threshold.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    threshold : float
        Decision threshold.

    Returns
    -------
    tuple of (float, float, float)
        (precision, recall, f1)
    """
    y_pred = (y_proba >= threshold).astype(int)
    # Handle edge case where all predictions are the same class
    if len(np.unique(y_pred)) == 1:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    return prec, rec, f1


def render(results, best_model_data):
    """Render the Model Deep Dive tab.

    Parameters
    ----------
    results : dict
        Model results dictionary from model_results.pkl.
    best_model_data : dict
        Best model dictionary from best_model.pkl, containing
        'model', 'preprocessor', 'best_name', 'feature_names'.
    """
    st.header("Model Deep Dive")

    model_names = list(results.keys())
    selected = st.selectbox(
        "Select a model to explore",
        model_names,
        key="deep_dive_selector",
    )

    if not selected:
        return

    model_result = results[selected]

    # ---- 1. Feature Importances (gold-to-teal gradient) ----
    st.subheader("Feature Importances")
    feat_imp = model_result.get("feature_importances")
    feature_names = best_model_data.get("feature_names", [])

    if feat_imp is not None and len(feat_imp) > 0:
        # Build feature importance DataFrame
        if feature_names is not None and len(feature_names) == len(feat_imp):
            names = list(feature_names)
        else:
            names = [f"Feature {i}" for i in range(len(feat_imp))]

        imp_df = pd.DataFrame(
            {"Feature": names, "Importance": feat_imp}
        )
        imp_df = imp_df.sort_values("Importance", ascending=True).tail(15)

        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top 15 Feature Importances — {selected}",
            color="Importance",
            color_continuous_scale=[[0, ACCENT_TEAL], [1, ACCENT_GOLD]],
        )
        _apply_layout(fig_imp)
        fig_imp.update_layout(
            coloraxis_showscale=False,
            yaxis=dict(
                categoryorder="total ascending",
                gridcolor="rgba(42,42,53,0.5)",
                zerolinecolor="rgba(42,42,53,0.8)",
            ),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info(
            f"{selected} does not provide feature importances. "
            "This is typical for models like SVM and KNN."
        )

    # ---- 2. Hyperparameters for tree-based models ----
    if selected in TREE_MODELS:
        st.subheader("Model Hyperparameters")
        model_obj = best_model_data.get("model")
        if model_obj is not None and selected == best_model_data.get("best_name"):
            # If the best model matches the selected, show its params
            if hasattr(model_obj, "best_params_"):
                params = model_obj.best_params_
                st.markdown(
                    '<p style="font-family: Sora, sans-serif; color: #C9A96E; '
                    'font-weight: 500; font-size: 0.9rem;">Best GridSearchCV Parameters</p>',
                    unsafe_allow_html=True,
                )
            elif hasattr(model_obj, "get_params"):
                params = model_obj.get_params()
                st.markdown(
                    '<p style="font-family: Sora, sans-serif; color: #C9A96E; '
                    'font-weight: 500; font-size: 0.9rem;">Model Parameters</p>',
                    unsafe_allow_html=True,
                )
            else:
                params = {}
            if params:
                params_df = pd.DataFrame(
                    list(params.items()), columns=["Parameter", "Value"]
                )
                st.dataframe(params_df, use_container_width=True)
        else:
            st.info(
                f"Detailed hyperparameters are available for the best model "
                f"({best_model_data.get('best_name', 'N/A')}). "
                f"Select it to view GridSearch results."
            )

    # ---- 3. Precision-Recall Curve (gold fill) ----
    st.subheader("Precision-Recall Curve")
    pr_data = model_result.get("precision_recall_curve")
    if pr_data is not None:
        precision_vals, recall_vals, _ = pr_data
        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(
                x=recall_vals,
                y=precision_vals,
                mode="lines",
                name="Precision-Recall",
                line=dict(color=ACCENT_GOLD, width=2),
                fill="tozeroy",
                fillcolor="rgba(201, 169, 110, 0.12)",
            )
        )
        _apply_layout(fig_pr)
        fig_pr.update_layout(
            title=f"Precision-Recall Curve — {selected}",
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        st.plotly_chart(fig_pr, use_container_width=True)
    else:
        st.warning("Precision-Recall curve data not available for this model.")

    # ---- 4. Threshold Tuning ----
    st.subheader("Threshold Tuning")
    st.markdown(
        '<p style="font-family: Sora, sans-serif; color: #8B8680; font-size: 0.88rem;">'
        'Adjust the decision threshold and see how it affects Precision, Recall, and F1 Score.</p>',
        unsafe_allow_html=True,
    )

    y_proba = model_result.get("y_proba")
    y_val = model_result.get("y_val")

    if y_proba is not None and y_val is not None:
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            key="threshold_slider",
        )

        prec, rec, f1 = _compute_metrics_at_threshold(y_val, y_proba, threshold)

        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Precision", f"{prec:.4f}")
        with metric_cols[1]:
            st.metric("Recall", f"{rec:.4f}")
        with metric_cols[2]:
            st.metric("F1 Score", f"{f1:.4f}")

        # Show metrics across all thresholds as a line chart
        thresholds_range = np.arange(0.10, 0.91, 0.05)
        metrics_over_thresh = []
        for t in thresholds_range:
            p, r, f = _compute_metrics_at_threshold(y_val, y_proba, t)
            metrics_over_thresh.append(
                {"Threshold": round(t, 2), "Precision": p, "Recall": r, "F1": f}
            )
        thresh_df = pd.DataFrame(metrics_over_thresh)

        fig_thresh = go.Figure()
        metric_colors = {
            "Precision": CHART_COLORS[0],  # Gold
            "Recall": CHART_COLORS[1],     # Teal
            "F1": CHART_COLORS[2],         # Emerald
        }
        for metric_name, color in metric_colors.items():
            fig_thresh.add_trace(
                go.Scatter(
                    x=thresh_df["Threshold"],
                    y=thresh_df[metric_name],
                    mode="lines+markers",
                    name=metric_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=5),
                )
            )
        # Mark the selected threshold
        fig_thresh.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#8B8680",
            annotation_text=f"Threshold = {threshold:.2f}",
            annotation_font=dict(
                family="JetBrains Mono, monospace",
                size=11,
                color="#E8E4DD",
            ),
        )
        _apply_layout(fig_thresh)
        fig_thresh.update_layout(
            title="Metrics vs Decision Threshold",
            xaxis_title="Threshold",
            yaxis_title="Score",
            legend=dict(x=0.01, y=0.01, bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_thresh, use_container_width=True)
    else:
        st.warning(
            "Probability predictions or validation labels not available "
            "for threshold tuning."
        )
