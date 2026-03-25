"""Model Performance tab for the Streamlit dashboard.

Displays comparison tables, ROC curves, confusion matrices,
F1 bar charts, and per-model classification reports.
All visualizations use Plotly with the Gilded Observatory dark theme.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def _highlight_best(s):
    """Highlight the maximum value in each column with gold accent.

    Parameters
    ----------
    s : pd.Series
        A column from the comparison DataFrame.

    Returns
    -------
    list of str
        CSS styles for each cell in the column.
    """
    is_best = s == s.max()
    return [
        "background-color: rgba(201,169,110,0.2); color: #C9A96E; font-weight: bold"
        if v else ""
        for v in is_best
    ]


def render(results):
    """Render the Model Performance tab.

    Parameters
    ----------
    results : dict
        Model results dictionary loaded from model_results.pkl.
        Keys are model names; values are dicts with metrics,
        confusion matrices, ROC curves, etc.
    """
    st.header("Model Performance")

    model_names = list(results.keys())

    # ---- 1. Comparison Table ----
    st.subheader("Model Comparison")
    comparison_data = []
    for name in model_names:
        r = results[name]
        comparison_data.append(
            {
                "Model": name,
                "Accuracy": round(r["accuracy"], 4),
                "Precision": round(r["precision"], 4),
                "Recall": round(r["recall"], 4),
                "F1": round(r["f1"], 4),
                "AUC": round(r["roc_auc"], 4),
            }
        )

    comp_df = pd.DataFrame(comparison_data).set_index("Model")
    styled = comp_df.style.apply(_highlight_best, axis=0).format("{:.4f}")
    st.dataframe(styled, use_container_width=True)

    # ---- 2. ROC Curves (all models overlaid) ----
    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    for idx, name in enumerate(model_names):
        fpr, tpr, _ = results[name]["roc_curve"]
        auc_val = results[name]["roc_auc"]
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC={auc_val:.3f})",
                line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)], width=2),
            )
        )
    # Diagonal reference line
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="#8B8680", width=1, dash="dash"),
        )
    )
    _apply_layout(fig_roc)
    fig_roc.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.05, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # ---- 3. Confusion Matrices (2-column grid) ----
    st.subheader("Confusion Matrices")
    cols = st.columns(2)
    for idx, name in enumerate(model_names):
        cm = results[name]["confusion_matrix"]
        labels = ["Rejected (N)", "Approved (Y)"]

        # Dark gold colorscale for confusion matrix
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=labels,
            y=labels,
            color_continuous_scale=[[0, '#16161D'], [0.5, '#5C4A2A'], [1, '#C9A96E']],
            title=name,
            aspect="auto",
        )
        _apply_layout(fig_cm)
        fig_cm.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            margin=dict(t=50, b=30, l=20, r=20),
            coloraxis_showscale=False,
        )
        fig_cm.update_traces(
            textfont=dict(family="JetBrains Mono, monospace", size=14, color="#E8E4DD")
        )
        with cols[idx % 2]:
            st.plotly_chart(fig_cm, use_container_width=True)

    # ---- 4. F1 Score Bar Chart ----
    st.subheader("F1 Scores Comparison")
    f1_data = pd.DataFrame(
        {
            "Model": model_names,
            "F1 Score": [results[n]["f1"] for n in model_names],
        }
    )
    f1_data = f1_data.sort_values("F1 Score", ascending=True)
    fig_f1 = px.bar(
        f1_data,
        x="F1 Score",
        y="Model",
        orientation="h",
        title="F1 Score by Model",
        color="F1 Score",
        color_continuous_scale=[[0, '#5C4A2A'], [1, '#C9A96E']],
    )
    _apply_layout(fig_f1)
    fig_f1.update_layout(
        coloraxis_showscale=False,
        yaxis=dict(
            categoryorder="total ascending",
            gridcolor="rgba(42,42,53,0.5)",
            zerolinecolor="rgba(42,42,53,0.8)",
        ),
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    # ---- 5. Classification Report Selector ----
    st.subheader("Detailed Classification Report")
    selected_model = st.selectbox(
        "Select a model to view its classification report",
        model_names,
        key="perf_model_selector",
    )
    if selected_model:
        st.code(results[selected_model]["classification_report"], language="text")
