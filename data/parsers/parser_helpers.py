"""
Helper functions for CSV file parsing shared across different parser implementations.
"""

import os
import csv
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from utils.logger import Logger


def detect_delimiter(file_path: str, sample_size: int = 5) -> str:
    """
    Detect the delimiter used in a CSV file.

    Args:
        file_path: Path to the CSV file
        sample_size: Number of lines to sample for detection

    Returns:
        The detected delimiter character
    """
    potential_delimiters = [',', ';', '\t', '|', ' ']
    delimiter_counts = {d: 0 for d in potential_delimiters}

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [f.readline() for _ in range(sample_size) if f.readline()]

    for line in lines:
        for delimiter in potential_delimiters:
            delimiter_counts[delimiter] += line.count(delimiter)

    # Choose delimiter with highest count
    detected_delimiter = max(delimiter_counts.items(), key=lambda x: x[1])[0]
    Logger.log_message_static(f"Parser-Helper: Detected delimiter '{detected_delimiter}'", Logger.DEBUG)

    # Default to comma if no clear delimiter is found
    if delimiter_counts[detected_delimiter] == 0:
        Logger.log_message_static("Parser-Helper: No clear delimiter detected, defaulting to comma", Logger.WARNING)
        return ','

    return detected_delimiter


def normalize_column_names(headers: List[str]) -> List[str]:
    """
    Normalize column names by stripping whitespace and quotes.

    Args:
        headers: List of column header strings

    Returns:
        List of normalized header strings
    """
    normalized = []
    for header in headers:
        # Strip whitespace and quotes
        clean_header = header.strip()
        for quote in ['"', "'"]:
            if clean_header.startswith(quote) and clean_header.endswith(quote):
                clean_header = clean_header[1:-1]
        normalized.append(clean_header)

    return normalized


def create_time_column(num_points: int, sample_rate: float) -> np.ndarray:
    """
    Create a time column based on number of points and sample rate.

    Args:
        num_points: Number of data points
        sample_rate: Sample rate in Hz

    Returns:
        NumPy array representing the time values
    """
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")

    time_step = 1.0 / sample_rate
    return np.arange(0, num_points * time_step, time_step)


def extract_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract basic metadata from a file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary of metadata
    """
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    modification_time = os.path.getmtime(file_path)

    return {
        "file_name": file_name,
        "file_path": file_path,
        "file_size": file_size,
        "modification_time": modification_time
    }


def is_numeric_column(values: List[str]) -> bool:
    """
    Check if a column contains numeric data.

    Args:
        values: List of string values from the column

    Returns:
        True if the column appears to be numeric, False otherwise
    """
    try:
        # Sample up to 100 non-empty values
        sample = [v for v in values[:100] if v.strip()]
        if not sample:
            return False

        # Check if values can be converted to float
        numeric_count = 0
        for value in sample:
            try:
                float(value.replace(',', '.'))  # Handle European number format
                numeric_count += 1
            except ValueError:
                pass

        # Consider column numeric if at least 90% of values are numeric
        return numeric_count >= 0.9 * len(sample)
    except Exception as e:
        Logger.log_message_static(f"Parser-Helper: Error checking if column is numeric: {str(e)}", Logger.WARNING)
        return False