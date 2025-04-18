"""
utils.py

Utility functions for parsing CSV data and supporting the signal viewer.
"""

import pandas as pd
import numpy as np
import warnings

def parse_csv_file(path: str):
    """
    Parses a CSV file and returns timestamp array and signal data.

    Args:
        path (str): Path to the CSV file.

    Returns:
        tuple:
            - np.ndarray: Array of datetime timestamps as float seconds.
            - dict[str, np.ndarray]: Dictionary of signal name to signal values.

    Raises:
        ValueError: If timestamp cannot be parsed or signals are missing.
    """
    try:
        df = pd.read_csv(path, sep=';', engine='python')
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    # === Parse timestamp ===
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Timestamp'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'].str.replace(',', '.', regex=False),
            format='%Y-%m-%d %H:%M:%S.%f',
            errors='coerce'
        )
    else:
        raise ValueError("CSV must contain 'Date' and 'Time' columns.")

    if df['Timestamp'].isnull().all():
        raise ValueError("All timestamps failed to parse.")

    df.dropna(subset=['Timestamp'], inplace=True)
    timestamps = df['Timestamp'].astype(np.int64) / 1e9
    timestamps = timestamps.to_numpy()

    # === Parse signals ===
    signal_cols = [col for col in df.columns if col not in ('Date', 'Time', 'Timestamp')]
    signals = {}
    skipped = []

    for col in signal_cols:
        # Replace commas with dots and force string to convert properly
        cleaned = df[col].astype(str).str.replace(',', '.', regex=False)
        numeric = pd.to_numeric(cleaned, errors='coerce')
        if numeric.isnull().all():
            skipped.append(col)
        else:
            signals[col] = numeric.to_numpy(dtype=np.float32)

    if not signals:
        raise ValueError("All signal columns failed to convert.")

    if skipped:
        for col in skipped:
            warnings.warn(f"Column '{col}' could not be converted to float and was skipped.")

    return timestamps, signals


def find_nearest_index(array: np.ndarray, value: float) -> int:
    """
    Finds the index of the closest value in an array.

    Args:
        array (np.ndarray): The array to search.
        value (float): The value to find the closest match to.

    Returns:
        int: Index of the closest value in the array.
    """
    idx = (np.abs(array - value)).argmin()
    return idx
