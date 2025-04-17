"""
Utility functions for CSV parsing and signal time processing.
"""
import pandas as pd
import numpy as np


def parse_csv_file(file_path):
    """
    Parses a CSV file with 'Date' and 'Time' columns into timestamps and signals.

    Returns:
        time_stamps (np.ndarray): Timestamps in Unix time (float seconds).
        signals (dict): Dictionary of signal_name -> np.ndarray of float values.
    """
    df = pd.read_csv(file_path, sep=';', decimal=',', engine='python')
    df['Timestamp'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%Y-%m-%d %H:%M:%S,%f',
        errors='coerce'
    )
    df.drop(columns=['Date', 'Time'], inplace=True)
    df.dropna(subset=['Timestamp'], inplace=True)

    df['UnixTime'] = df['Timestamp'].astype('int64') // 10**9
    time_stamps = df['UnixTime'].to_numpy(dtype=float)

    df.drop(columns=['Timestamp', 'UnixTime'], inplace=True)
    df.columns = df.columns.str.strip()

    signals = {col: df[col].astype(float).to_numpy() for col in df.columns}
    return time_stamps, signals


def find_nearest_index(array, value):
    idx = np.searchsorted(array, value, side="left")
    idx = np.clip(idx, 0, len(array) - 1)
    return idx
