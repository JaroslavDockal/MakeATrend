"""
utils.py

Utility functions for parsing CSV data and supporting the signal viewer.
"""

import pandas as pd
import numpy as np
import warnings
import re
from datetime import datetime, timedelta


def parse_csv_or_recorder(path: str):
    """
    Parses either a standard CSV file or a proprietary recorder TXT file.

    Args:
        path (str): Path to the file.

    Returns:
        tuple:
            - np.ndarray: Array of timestamps (float, seconds since epoch).
            - dict[str, np.ndarray]: Dictionary of signal name -> signal values.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "RECORDER VALUES" in content and "Interval:" in content:
        return parse_recorder_format(content)
    else:
        return parse_csv_file(path)

def parse_csv_file(path):
    """
    Parses a standard CSV file and returns timestamp array and signal data.

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
        raise ValueError("Missing 'Date' and 'Time' columns.")

    df.dropna(subset=['Timestamp'], inplace=True)
    timestamps = df['Timestamp'].astype(np.int64) / 1e9
    timestamps = timestamps.to_numpy()

    signals = {}
    for col in df.columns:
        if col in ('Date', 'Time', 'Timestamp'):
            continue
        cleaned = df[col].astype(str).str.replace(',', '.', regex=False)
        numeric = pd.to_numeric(cleaned, errors='coerce')
        if not numeric.isnull().all():
            signals[col] = numeric.to_numpy(dtype=np.float32)

    if not signals:
        raise ValueError("No signals could be parsed.")

    return timestamps, signals


def parse_recorder_format(text):
    """
    Parses a text file in the special "Drive Window" format.

    Args:
        text (str): Full content of the file.

    Returns:
        tuple:
            - np.ndarray: Array of timestamps (float).
            - dict[str, np.ndarray]: Dictionary of signal name -> values.
    """
    lines = text.strip().splitlines()
    item_map = {}
    start_time_str = None
    interval_sec = None
    data_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith("Item "):
            match = re.match(r"Item\s+(\d+)\s*=\s*(.+)", line)
            if match:
                item_map[int(match.group(1))] = match.group(2).strip()
        elif "Time of Interval" in line:
            start_time_str = line.split(":", 1)[1].strip()
        elif "Interval:" in line:
            m = re.search(r"([\d.]+)\s*sec", line)
            if m:
                interval_sec = float(m.group(1))
        elif re.match(r"\s*\d+\s+", line):
            parts = re.split(r'\s+', line)
            try:
                idx = int(parts[0])
                values = [float(p.replace(',', '.')) for p in parts[1:]]
                data_lines.append([idx] + values)
            except ValueError:
                continue

    if not (start_time_str and interval_sec and item_map and data_lines):
        raise ValueError("Invalid recorder format.")

    start_dt = datetime.strptime(start_time_str, "%y/%m/%d %H:%M:%S")
    data_lines.sort(key=lambda row: row[0])
    timestamps = [(start_dt - timedelta(seconds=row[0] * interval_sec)).timestamp() for row in data_lines]

    signals = {}
    for i in range(len(data_lines[0]) - 1):
        name = item_map.get(i + 1, f"Signal {i + 1}")
        signals[name] = [row[i + 1] for row in data_lines]

    for name in signals:
        signals[name] = np.array(signals[name], dtype=np.float32)

    return np.array(timestamps, dtype=np.float64), signals

def find_nearest_index(array, value):
    """
    Finds the index of the closest value in an array.

    Args:
        array (np.ndarray): The array to search.
        value (float): The value to find the closest match to.

    Returns:
        int: Index of the closest value in the array.
    """
    return (np.abs(array - value)).argmin()

def is_digital_signal(arr):
    """
    Determines whether a signal is digital (boolean-like).

    A signal is considered digital if it only contains values like:
    - 'TRUE' / 'FALSE' (case-insensitive)
    - Not numeric 0/1, to avoid misclassification of analog signals

    Args:
        arr (np.ndarray): Signal values (can be numeric or string).

    Returns:
        bool: True if the signal is clearly boolean (TRUE/FALSE), False otherwise.
    """
    unique = set(str(v).strip().lower() for v in np.unique(arr))
    return unique.issubset({'true', 'false'})
