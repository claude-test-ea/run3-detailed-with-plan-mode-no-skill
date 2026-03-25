"""Data Explorer tab for the Streamlit dashboard.

Provides interactive filtering, search, and summary statistics
for the full loan application dataset. Gilded Observatory theme.
"""
import streamlit as st
import pandas as pd


def render(df):
    """Render the Data Explorer tab.

    Displays the full dataset with sidebar filters for categorical
    and numerical columns, a text search across all columns, and
    summary statistics for the filtered results.

    Parameters
    ----------
    df : pd.DataFrame
        The full loan application dataset.
    """
    st.header("Data Explorer")

    # --- Search box ---
    search = st.text_input(
        "Search across all columns",
        "",
        placeholder="Type to search rows...",
    )

    # --- Sidebar filters ---
    st.sidebar.markdown(
        """
        <div style="
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.4rem;
            font-weight: 600;
            color: #C9A96E;
            margin-bottom: 0.4rem;
            letter-spacing: 0.02em;
        ">Filters</div>
        <div style="
            width: 40px;
            height: 2px;
            background: linear-gradient(90deg, #C9A96E, transparent);
            margin-bottom: 1rem;
        "></div>
        """,
        unsafe_allow_html=True,
    )

    # Categorical filters
    gender_options = df["Gender"].dropna().unique().tolist()
    gender_filter = st.sidebar.multiselect(
        "Gender",
        options=gender_options,
        default=gender_options,
    )

    married_options = df["Married"].dropna().unique().tolist()
    married_filter = st.sidebar.multiselect(
        "Married",
        options=married_options,
        default=married_options,
    )

    education_options = df["Education"].dropna().unique().tolist()
    education_filter = st.sidebar.multiselect(
        "Education",
        options=education_options,
        default=education_options,
    )

    property_options = df["Property_Area"].dropna().unique().tolist()
    property_filter = st.sidebar.multiselect(
        "Property Area",
        options=property_options,
        default=property_options,
    )

    status_options = df["Loan_Status"].dropna().unique().tolist()
    status_filter = st.sidebar.multiselect(
        "Loan Status",
        options=status_options,
        default=status_options,
    )

    # Numerical sliders
    income_min = int(df["ApplicantIncome"].min())
    income_max = int(df["ApplicantIncome"].max())
    income_range = st.sidebar.slider(
        "Applicant Income Range",
        min_value=income_min,
        max_value=income_max,
        value=(income_min, income_max),
    )

    loan_amt_clean = df["LoanAmount"].dropna()
    loan_min = int(loan_amt_clean.min())
    loan_max = int(loan_amt_clean.max())
    loan_range = st.sidebar.slider(
        "Loan Amount Range",
        min_value=loan_min,
        max_value=loan_max,
        value=(loan_min, loan_max),
    )

    # --- Apply filters ---
    filtered = df.copy()

    # Categorical filters (preserve NaN rows when category values are all selected)
    filtered = filtered[
        filtered["Gender"].isin(gender_filter) | filtered["Gender"].isna()
    ]
    filtered = filtered[
        filtered["Married"].isin(married_filter) | filtered["Married"].isna()
    ]
    filtered = filtered[filtered["Education"].isin(education_filter)]
    filtered = filtered[filtered["Property_Area"].isin(property_filter)]
    filtered = filtered[filtered["Loan_Status"].isin(status_filter)]

    # Numerical range filters
    filtered = filtered[
        (filtered["ApplicantIncome"] >= income_range[0])
        & (filtered["ApplicantIncome"] <= income_range[1])
    ]
    filtered = filtered[
        (
            (filtered["LoanAmount"] >= loan_range[0])
            & (filtered["LoanAmount"] <= loan_range[1])
        )
        | filtered["LoanAmount"].isna()
    ]

    # Text search filter
    if search:
        mask = filtered.astype(str).apply(
            lambda row: row.str.contains(search, case=False).any(), axis=1
        )
        filtered = filtered[mask]

    # --- Prominent filtered-row metric ---
    st.metric("Filtered Rows", len(filtered))

    if filtered.empty:
        st.warning("No data matches the current filters.")
    else:
        st.dataframe(filtered, use_container_width=True, height=450)
        st.subheader("Summary Statistics")
        st.dataframe(filtered.describe(), use_container_width=True)
