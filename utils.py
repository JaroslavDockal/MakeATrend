"""Helper functions for parsing CSV and working with cursors."""

import pandas as pd
import numpy as np


def parse_csv_file(file_path):
    """
    Load a CSV file with 'Date' and 'Time' columns, parse to datetime,
    and return time array and dictionary of signal arrays.

    Parameters:
    - file_path: str

    Returns:
    - time: np.ndarray (seconds from start)
    - signals: dict[str, np.ndarray]
    """
    # Načti CSV s oddělovačem středník
    df = pd.read_csv(file_path, sep=';', engine='python')

    # Spoj Date + Time do jednoho datetime sloupce
    df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # Vypočítej relativní čas od začátku
    df['RelativeTime'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()

    # Extrahuj časovou osu
    time = df['RelativeTime'].to_numpy()

    # Odeber nepoužité sloupce
    df.drop(columns=['Date', 'Time', 'Timestamp', 'RelativeTime'], inplace=True)

    # Vyčisti názvy signálů
    df.columns = df.columns.str.strip()

    # Ulož signály do dictu
    signals = {col: df[col].to_numpy(dtype=float) for col in df.columns}

    return time, signals


def get_signal_values_at_time(time_array, signals_dict, target_time):
    """
    Return the nearest value from each signal at the given time.
    """
    idx = np.searchsorted(time_array, target_time)
    idx = np.clip(idx, 0, len(time_array) - 1)
    return {
        name: signal[idx]
        for name, signal in signals_dict.items()
    }
