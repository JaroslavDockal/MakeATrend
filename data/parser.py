"""
Parser module for handling different data file formats.

This module provides functionality for parsing various file formats:
1. CSV files with different delimiters (semicolon or comma)
2. Drive Debug recorder files (proprietary format with timestamps and signal data)

The module handles various timestamp formats, numeric/boolean values, and supports
automatic detection of file format based on content patterns.

Functions:
    parse_csv_or_recorder: Main entry point to parse either CSV or recorder format
    parse_csv_file: Parse standard CSV files with different delimiters
    parse_recorder_format: Parse Drive Debug recorder format
"""

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils.logger import Logger

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
    Logger.log_message_static(f"Starting to parse file: {os.path.basename(path)}", Logger.DEBUG)

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            Logger.log_message_static(f"Successfully read file: {os.path.basename(path)}", Logger.DEBUG)
    except UnicodeDecodeError:
        Logger.log_message_static(f"UTF-8 encoding failed, trying with latin1 for file: {os.path.basename(path)}", Logger.WARNING)
        try:
            with open(path, "r", encoding="latin1") as f:
                content = f.read()
        except Exception as e:
            Logger.log_message_static(f"Failed to read file {os.path.basename(path)}: {str(e)}", Logger.ERROR)
            raise ValueError(f"Cannot read file: {str(e)}")
    except Exception as e:
        Logger.log_message_static(f"Failed to read file {os.path.basename(path)}: {str(e)}", Logger.ERROR)
        raise ValueError(f"Cannot read file: {str(e)}")

    if ("RECORDER VALUES" in content or "TREND VALUES" in content) and "Interval:" in content:
        Logger.log_message_static(f"Detected Drive Debug format for file: {os.path.basename(path)}", Logger.INFO)
        return parse_recorder_format(content)
    else:
        Logger.log_message_static(f"Treating file as standard CSV: {os.path.basename(path)}", Logger.INFO)
        return parse_csv_file(path)

def parse_csv_file(path):
    """
    Parses a standard CSV file and returns timestamp array and signal data.
    Handles both numeric values and boolean string values ('TRUE'/'FALSE').

    Args:
        path (str): Path to the CSV file.

    Returns:
        tuple:
            - np.ndarray: Array of datetime timestamps as float seconds.
            - dict[str, np.ndarray]: Dictionary of signal name to signal values.

    Raises:
        ValueError: If timestamp cannot be parsed or signals are missing.
    """
    Logger.log_message_static(f"Parsing CSV file: {os.path.basename(path)}", Logger.DEBUG)

    try:
        Logger.log_message_static("Attempting to parse with semicolon delimiter", Logger.DEBUG)
        df = pd.read_csv(path, sep=';', engine='python')
    except pd.errors.ParserError as e:
        Logger.log_message_static(f"CSV parsing error with semicolon delimiter: {str(e)}", Logger.WARNING)
        try:
            Logger.log_message_static("Trying with comma delimiter", Logger.DEBUG)
            df = pd.read_csv(path, sep=',', engine='python')
        except pd.errors.ParserError as e:
            Logger.log_message_static(f"CSV parsing error with comma delimiter: {str(e)}", Logger.ERROR)
            raise ValueError(f"CSV parsing error: {e}")
    except FileNotFoundError:
        Logger.log_message_static(f"File not found: {path}", Logger.ERROR)
        raise ValueError(f"File not found: {path}")
    except Exception as e:
        Logger.log_message_static(f"Failed to read CSV: {str(e)}", Logger.ERROR)
        raise ValueError(f"Failed to read CSV: {e}")

    # === Parse timestamp ===
    Logger.log_message_static("Checking for Date and Time columns", Logger.DEBUG)
    if 'Date' in df.columns and 'Time' in df.columns:
        Logger.log_message_static("Found Date and Time columns, parsing timestamps", Logger.DEBUG)
        try:
            df['Timestamp'] = pd.to_datetime(
                df['Date'] + ' ' + df['Time'].str.replace(',', '.', regex=False),
                format='%Y-%m-%d %H:%M:%S.%f',
                errors='coerce'
            )
            Logger.log_message_static("Successfully parsed timestamps", Logger.DEBUG)
        except Exception as e:
            Logger.log_message_static(f"Error parsing timestamps: {str(e)}, trying flexible parsing", Logger.WARNING)
            try:
                df['Timestamp'] = pd.to_datetime(
                    df['Date'] + ' ' + df['Time'].str.replace(',', '.', regex=False),
                    errors='coerce'
                )
                Logger.log_message_static("Successfully parsed timestamps with flexible format", Logger.DEBUG)
            except Exception as e:
                Logger.log_message_static(f"Failed to parse timestamps: {str(e)}", Logger.ERROR)
                raise ValueError("Failed to parse timestamps")
    else:
        Logger.log_message_static("Missing 'Date' and 'Time' columns", Logger.ERROR)
        raise ValueError("Missing 'Date' and 'Time' columns.")

    # Drop rows with invalid timestamps
    invalid_count = df['Timestamp'].isna().sum()
    if invalid_count > 0:
        Logger.log_message_static(f"Found {invalid_count} rows with invalid timestamps, dropping them", Logger.WARNING)

    df.dropna(subset=['Timestamp'], inplace=True)
    timestamps = df['Timestamp'].astype(np.int64) / 1e9
    timestamps = timestamps.to_numpy()
    Logger.log_message_static(f"Created timestamp array with {len(timestamps)} points", Logger.DEBUG)

    signals = {}
    Logger.log_message_static("Starting to parse signal columns", Logger.DEBUG)
    for col in df.columns:
        if col in ('Date', 'Time', 'Timestamp'):
            continue

        # Direct approach for boolean values - check if any values are TRUE/FALSE
        values = df[col].astype(str).str.upper()
        true_mask = values == "TRUE"
        false_mask = values == "FALSE"

        # If column contains TRUE/FALSE values, treat as boolean
        if true_mask.any() or false_mask.any():
            Logger.log_message_static(f"Detected boolean signal in column '{col}'", Logger.DEBUG)
            # Force a refresh of the plot
            signals[col] = values.to_numpy()  # Preserve TRUE/FALSE strings
        else:
            # Try numeric conversion for non-boolean columns
            try:
                Logger.log_message_static(f"Converting column '{col}' to numeric", Logger.DEBUG)
                cleaned = df[col].astype(str).str.replace(',', '.', regex=False)
                numeric = pd.to_numeric(cleaned, errors='coerce')
                if not numeric.isnull().all():
                    signals[col] = numeric.to_numpy(dtype=np.float32)
                    Logger.log_message_static(f"Successfully converted '{col}' to numeric", Logger.DEBUG)
                else:
                    Logger.log_message_static(f"Column '{col}' contains only non-numeric values", Logger.WARNING)
            except Exception as e:
                Logger.log_message_static(f"Failed to convert column '{col}' to numeric: {str(e)}", Logger.WARNING)
                # Skip columns that can't be parsed

    if not signals:
        Logger.log_message_static("No signals could be parsed from the file", Logger.ERROR)
        raise ValueError("No signals could be parsed.")

    Logger.log_message_static(f"Successfully parsed {len(signals)} signals from CSV file", Logger.INFO)
    return timestamps, signals

def parse_recorder_format(text):
    """
    Parses a text file in the special "Drive Debug" format.

    Args:
        text (str): Full content of the file.

    Returns:
        tuple:
            - np.ndarray: Array of timestamps (float, seconds since UNIX epoch).
            - dict[str, np.ndarray]: Dictionary of signal name -> signal values as float32 arrays.
    """
    Logger.log_message_static("Parsing Drive Debug format file", Logger.DEBUG)

    lines = text.strip().splitlines()
    item_map = {}
    start_time_str = None
    interval_sec = None
    data_lines = []

    Logger.log_message_static(f"File contains {len(lines)} lines", Logger.DEBUG)

    for line in lines:
        line = line.strip()
        if line.startswith("Item "):
            match = re.match(r"Item\s+(\d+)\s*=\s*(.+)", line)
            if match:
                item_id = int(match.group(1))
                item_name = match.group(2).strip()
                item_map[item_id] = item_name
                Logger.log_message_static(f"Found Item {item_id} = {item_name}", Logger.DEBUG)
        elif "Time of Interval" in line:
            start_time_str = line.split(":", 1)[1].strip()
            Logger.log_message_static(f"Found start time: {start_time_str}", Logger.DEBUG)
        elif "Interval:" in line:
            m = re.search(r"([\d.]+)\s*sec", line)
            if m:
                interval_sec = float(m.group(1))
                Logger.log_message_static(f"Found interval: {interval_sec} seconds", Logger.DEBUG)
        elif re.match(r"\s*\d+\s+", line):
            parts = re.split(r'\s+', line)
            try:
                idx = int(parts[0])
                values = [float(p.replace(',', '.')) for p in parts[1:]]
                data_lines.append([idx] + values)
            except ValueError:
                Logger.log_message_static(f"Skipping invalid data line: {line}", Logger.WARNING)
                continue

    if not start_time_str:
        Logger.log_message_static("Missing start time in the file", Logger.ERROR)
        raise ValueError("Invalid recorder format: missing start time.")
    if not interval_sec:
        Logger.log_message_static("Missing interval in the file", Logger.ERROR)
        raise ValueError("Invalid recorder format: missing interval.")
    if not item_map:
        Logger.log_message_static("No signal items found in the file", Logger.ERROR)
        raise ValueError("Invalid recorder format: no signal items found.")
    if not data_lines:
        Logger.log_message_static("No data lines found in the file", Logger.ERROR)
        raise ValueError("Invalid recorder format: no data found.")

    try:
        start_dt = datetime.strptime(start_time_str, "%m/%d/%y %H:%M:%S")
        Logger.log_message_static(f"Parsed start time: {start_dt}", Logger.DEBUG)
    except ValueError as e:
        Logger.log_message_static(f"Failed to parse start time '{start_time_str}': {str(e)}", Logger.ERROR)
        raise ValueError(f"Invalid start time format: {start_time_str}")

    Logger.log_message_static(f"Sorting {len(data_lines)} data lines", Logger.DEBUG)
    data_lines.sort(key=lambda row: row[0])

    Logger.log_message_static("Calculating timestamps", Logger.DEBUG)
    timestamps = [(start_dt - timedelta(seconds=row[0] * interval_sec)).timestamp() for row in data_lines]

    signals = {}
    Logger.log_message_static("Extracting signal values", Logger.DEBUG)
    for i in range(len(data_lines[0]) - 1):
        name = item_map.get(i + 1, f"Signal {i + 1}")
        signals[name] = [row[i + 1] for row in data_lines]

    for name in signals:
        signals[name] = np.array(signals[name], dtype=np.float32)
        Logger.log_message_static(f"Created signal '{name}' with {len(signals[name])} points", Logger.DEBUG)

    Logger.log_message_static(f"Successfully parsed {len(signals)} signals from Drive Debug file", Logger.INFO)
    return np.array(timestamps, dtype=np.float64), signals