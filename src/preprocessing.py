"""Preprocessing module.

Handles feature engineering, data preparation, and building sklearn
preprocessing pipelines for numerical and categorical columns.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Original numerical and categorical column definitions
NUMERICAL_COLS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

CATEGORICAL_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

# Full numerical list including engineered features
ALL_NUMERICAL = NUMERICAL_COLS + ["TotalIncome", "IncomeToLoan", "LoanAmountLog"]


def add_features(df):
    """Add engineered features to the DataFrame.

    Creates the following columns:
    - TotalIncome: ApplicantIncome + CoapplicantIncome
    - IncomeToLoan: TotalIncome / LoanAmount (0 when LoanAmount is 0 or NaN)
    - LoanAmountLog: np.log1p(LoanAmount)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least ApplicantIncome, CoapplicantIncome, LoanAmount.

    Returns
    -------
    pd.DataFrame
        DataFrame with the three new feature columns added.
    """
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    # Handle division by zero / NaN in LoanAmount
    df["IncomeToLoan"] = np.where(
        (df["LoanAmount"].isna()) | (df["LoanAmount"] == 0),
        0,
        df["TotalIncome"] / df["LoanAmount"],
    )
    df["LoanAmountLog"] = np.log1p(df["LoanAmount"])
    return df


def prepare_data(df):
    """Prepare features and target from raw DataFrame.

    Steps:
    1. Drop Loan_ID column.
    2. Encode Loan_Status: Y -> 1, N -> 0.
    3. Add engineered features via add_features().
    4. Separate into X (features) and y (target).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with Loan_ID, Loan_Status, and feature columns.

    Returns
    -------
    tuple of (pd.DataFrame, pd.Series)
        (X, y) where X contains all feature columns and y is the binary target.
    """
    df = df.copy()
    df = df.drop(columns=["Loan_ID"])
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
    df = add_features(df)
    y = df["Loan_Status"]
    X = df.drop(columns=["Loan_Status"])
    return X, y


def build_preprocessor():
    """Build a ColumnTransformer for preprocessing numerical and categorical features.

    Numerical pipeline: SimpleImputer(strategy='median') -> StandardScaler
    Categorical pipeline: SimpleImputer(strategy='most_frequent') -> OneHotEncoder

    The numerical pipeline operates on ALL_NUMERICAL (original + engineered features).
    The categorical pipeline operates on CATEGORICAL_COLS.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Fitted-ready preprocessor.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, ALL_NUMERICAL),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )

    return preprocessor
