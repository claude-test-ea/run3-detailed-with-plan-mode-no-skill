"""EDA & Visualizations tab for the Streamlit dashboard.

All charts use Plotly with the Gilded Observatory dark theme for
interactive visualizations of the full loan application dataset.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


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
        The styled figure.
    """
    fig.update_layout(**CHART_LAYOUT)
    return fig


def _approval_rate_chart(df, group_col, title):
    """Create a grouped bar chart showing approval rate by a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Loan_Status column.
    group_col : str
        Column name to group by.
    title : str
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    counts = (
        df.groupby([group_col, "Loan_Status"])
        .size()
        .reset_index(name="Count")
    )
    fig = px.bar(
        counts,
        x=group_col,
        y="Count",
        color="Loan_Status",
        barmode="group",
        title=title,
        color_discrete_map={"Y": COLOR_APPROVED, "N": COLOR_REJECTED},
    )
    _apply_layout(fig)
    fig.update_layout(legend_title_text="Loan Status")
    return fig


def render(df):
    """Render the EDA & Visualizations tab.

    Displays distribution charts, approval-rate breakdowns, income
    and loan-amount distributions, a correlation heatmap, and a
    credit-history vs loan-status stacked bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        The full loan application dataset.
    """
    st.header("EDA & Visualizations")

    # ---- 1. Loan Status Distribution: bar + pie side by side ----
    st.subheader("Loan Status Distribution")
    col1, col2 = st.columns(2)

    status_counts = df["Loan_Status"].value_counts().reset_index()
    status_counts.columns = ["Loan_Status", "Count"]

    with col1:
        fig_bar = px.bar(
            status_counts,
            x="Loan_Status",
            y="Count",
            color="Loan_Status",
            color_discrete_map={"Y": COLOR_APPROVED, "N": COLOR_REJECTED},
            title="Loan Status — Bar Chart",
        )
        _apply_layout(fig_bar)
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            status_counts,
            values="Count",
            names="Loan_Status",
            title="Loan Status — Breakdown",
            color="Loan_Status",
            color_discrete_map={"Y": COLOR_APPROVED, "N": COLOR_REJECTED},
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont=dict(family="Sora, sans-serif", size=13),
        )
        _apply_layout(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ---- 2. Approval Rate by categorical features (2-col grid) ----
    st.subheader("Approval Rate by Category")
    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(
            _approval_rate_chart(df, "Property_Area", "By Property Area"),
            use_container_width=True,
        )
        st.plotly_chart(
            _approval_rate_chart(df, "Gender", "By Gender"),
            use_container_width=True,
        )

    with col4:
        st.plotly_chart(
            _approval_rate_chart(df, "Education", "By Education"),
            use_container_width=True,
        )
        st.plotly_chart(
            _approval_rate_chart(df, "Married", "By Marital Status"),
            use_container_width=True,
        )

    # ---- 3. Income Distribution by Loan Status (box plot) ----
    st.subheader("Income Distribution by Loan Status")
    fig_box = px.box(
        df,
        x="Loan_Status",
        y="ApplicantIncome",
        color="Loan_Status",
        color_discrete_map={"Y": COLOR_APPROVED, "N": COLOR_REJECTED},
        title="Applicant Income Distribution by Loan Status",
    )
    _apply_layout(fig_box)
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # ---- 4. Loan Amount Distribution (histogram + box marginal) ----
    st.subheader("Loan Amount Distribution")
    fig_hist = px.histogram(
        df.dropna(subset=["LoanAmount"]),
        x="LoanAmount",
        marginal="box",
        nbins=40,
        title="Loan Amount Distribution",
        color_discrete_sequence=[ACCENT_TEAL],
    )
    _apply_layout(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True)

    # ---- 5. Correlation Heatmap ----
    st.subheader("Correlation Heatmap (Numerical Features)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_df = df[numeric_cols].corr()

    fig_heatmap = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="Cividis",
        title="Feature Correlation Matrix",
        aspect="auto",
    )
    _apply_layout(fig_heatmap)
    fig_heatmap.update_layout(
        coloraxis_colorbar=dict(
            tickfont=dict(color="#8B8680"),
            title=dict(font=dict(color="#8B8680")),
        )
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ---- 6. Credit History vs Loan Status (stacked bar) ----
    st.subheader("Credit History vs Loan Status")
    credit_df = (
        df.dropna(subset=["Credit_History"])
        .groupby(["Credit_History", "Loan_Status"])
        .size()
        .reset_index(name="Count")
    )
    credit_df["Credit_History"] = credit_df["Credit_History"].astype(str)

    fig_credit = px.bar(
        credit_df,
        x="Credit_History",
        y="Count",
        color="Loan_Status",
        barmode="stack",
        title="Credit History vs Loan Status",
        color_discrete_map={"Y": COLOR_APPROVED, "N": COLOR_REJECTED},
        labels={"Credit_History": "Credit History (0 = No, 1 = Yes)"},
    )
    _apply_layout(fig_credit)
    st.plotly_chart(fig_credit, use_container_width=True)
