"""Utility functions for the loan prediction pipeline."""
import os
from pathlib import Path


def ensure_dir(path):
    """Create a directory (and parents) if it does not already exist.

    Parameters
    ----------
    path : str or Path
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def get_project_root():
    """Return the project root directory.

    The project root is defined as the parent of the `src/` package,
    i.e., two levels up from this file (src/utils.py -> src -> project_root).

    Returns
    -------
    pathlib.Path
        Absolute path to the project root.
    """
    return Path(__file__).resolve().parent.parent
