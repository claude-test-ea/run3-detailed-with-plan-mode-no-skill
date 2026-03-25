"""Data ingestion module.

Handles loading CSV data and splitting into ML training data and holdout sets.
"""
import pandas as pd


def load_data(filepath):
    """Load CSV data, print shape/dtypes/preview, return DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file to load.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nPreview:\n{df.head()}")
    return df


def split_data(df, holdout_size=200):
    """Split DataFrame into ml_data and holdout sets.

    The last `holdout_size` rows are reserved as a holdout set that should
    never be used during model fitting or preprocessing fitting.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset.
    holdout_size : int, optional
        Number of rows to reserve for holdout (default 200).

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (ml_data, holdout) DataFrames.
    """
    ml_data = df[:-holdout_size].copy()
    holdout = df[-holdout_size:].copy()
    print(f"ML data: {ml_data.shape}, Holdout: {holdout.shape}")
    return ml_data, holdout
