"""NumPy-only input/output helpers for NumCompute."""

from __future__ import annotations

from pathlib import Path
import numpy as np


def _as_path(path):
    return Path(path)


def load_array(path, *, dtype=None, delimiter=",", skiprows=0):
    """
    Load a numeric array from a text file.

    Supports CSV-like files saved with save_array()/save_csv().
    """
    path = _as_path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return np.loadtxt(path, dtype=dtype, delimiter=delimiter, skiprows=skiprows)


def save_array(path, arr, *, delimiter=",", fmt="%.18g", header=None, comments=""):
    """
    Save a NumPy array to a text file.
    """
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(arr), delimiter=delimiter, fmt=fmt, header=header or "", comments=comments)


def load_csv(path, *, dtype=None, skiprows=0):
    """
    Convenience wrapper for comma-separated files.
    """
    return load_array(path, dtype=dtype, delimiter=",", skiprows=skiprows)


def save_csv(path, arr, *, fmt="%.18g", header=None, comments=""):
    """
    Convenience wrapper for comma-separated files.
    """
    save_array(path, arr, delimiter=",", fmt=fmt, header=header, comments=comments)
