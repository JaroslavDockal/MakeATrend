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


def parse_csv_file(path: str):
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


def parse_recorder_format(text: str):
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
    item_map = {}  # item number -> signal name
    start_time_str = None
    interval_sec = None
    data_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith("Item "):
            match = re.match(r"Item\s+(\d+)\s*=\s*(.+)", line)
            if match:
                item_num = int(match.group(1))
                item_name = match.group(2).strip()
                item_map[item_num] = item_name
        elif "Time of Interval" in line:
            parts = line.split(":", 1)
            if len(parts) >= 2:
                start_time_str = parts[1].strip()
        elif "Interval:" in line:
            interval_match = re.search(r"([\d.]+)\s*sec", line)
            if interval_match:
                interval_sec = float(interval_match.group(1))
        elif re.match(r"\s*\d+\s+", line):
            parts = re.split(r'\s+', line)
            try:
                index = int(parts[0])
                values = [float(p.replace(',', '.')) for p in parts[1:]]
                data_lines.append([index] + values)
            except ValueError:
                continue

    if not (start_time_str and interval_sec and item_map and data_lines):
        raise ValueError("Invalid recorder format: missing required information.")

    # Normalize whitespaces in time string
    cleaned_time = re.sub(r"\s+", " ", start_time_str.strip())

    try:
        start_dt = datetime.strptime(cleaned_time, "%y/%m/%d %H:%M:%S")
    except ValueError as e:
        raise ValueError(f"Failed to parse start time '{cleaned_time}': {e}")

    data_lines.sort(key=lambda row: row[0])  # sort by sample index ascending

    # Generate timestamps
    timestamps = [
        (start_dt - timedelta(seconds=row[0] * interval_sec)).timestamp()
        for row in data_lines
    ]

    # Parse signals
    num_signals = len(data_lines[0]) - 1
    signals = {}

    for i in range(num_signals):
        signal_name = item_map.get(i + 1, f"Signal {i + 1}")
        signals[signal_name] = [row[i + 1] if i + 1 < len(row) else np.nan for row in data_lines]

    for name in signals:
        signals[name] = np.array(signals[name], dtype=np.float32)

    return np.array(timestamps, dtype=np.float64), signals


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
