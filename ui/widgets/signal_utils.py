"""
Data Utilities Module

This module provides core data manipulation functionality for signal processing:

1. Signal Classification:
   - Detection of digital/boolean signals vs. analog signals
   - Support for various boolean representations (TRUE/FALSE strings, 0/1 values)

2. Data Navigation:
   - Finding nearest indices in time series data
   - Timestamp alignment and synchronization

3. File Parsing:
   - CSV file parsing with configurable dialects
   - Proprietary recorder file format support
   - Timestamp normalization and standardization

The module serves as a foundation for the application's data processing pipeline,
providing low-level utilities that are used by higher-level modules like loader.py
and parser.py.

Functions:
    find_nearest_index: Locate the closest value in an array
    is_digital_signal: Determine if a signal contains boolean-like values
"""
import numpy as np

from utils.logger import Logger


def find_nearest_index(array, value):
    """
    Find the index of the closest value in an array.

    Args:
        array (np.ndarray): Array to search in.
        value (float): Target value to find.

    Returns:
        int: Index of the closest value.
    """
    if array is None:
        Logger.log_message_static("Data-Signal: Cannot find nearest index in None array", Logger.WARNING)
        return None

    if len(array) == 0:
        Logger.log_message_static("Data-Signal: Cannot find nearest index in empty array", Logger.WARNING)
        return None

    try:
        idx = (np.abs(array - value)).argmin()
        Logger.log_message_static(f"Data-Signal: Found nearest index {idx} with value {array[idx]:.6f} for target {value:.6f}", Logger.DEBUG)
        return idx
    except Exception as e:
        Logger.log_message_static(f"Data-Signal: Error finding nearest index: {str(e)}", Logger.ERROR)
        return None

def is_digital_signal(arr):
    """
    Determines whether a signal is digital (boolean-like).

    A signal is considered digital if it:
    - Has boolean dtype (from astype(bool))
    - Contains 'TRUE' / 'FALSE' (case-insensitive) string values
    - Contains only 0/1 values

    Args:
        arr: NumPy array or tuple of (time_array, values_array)

    Returns:
        bool: True if signal appears to be digital/binary
    """
    Logger.log_message_static("Data-Signal: Checking if signal is digital", Logger.DEBUG)

    # Handle case where argument is a tuple (time, values)
    if isinstance(arr, tuple) and len(arr) == 2:
        Logger.log_message_static("Data-Signal: Processing (time, values) tuple, extracting values", Logger.DEBUG)
        _, arr = arr  # Extract just the values

    if arr is None:
        Logger.log_message_static("Data-Signal: Cannot determine if None signal is digital", Logger.WARNING)
        return False

    if not isinstance(arr, np.ndarray):
        try:
            Logger.log_message_static("Data-Signal: Converting input to numpy array", Logger.DEBUG)
            arr = np.array(arr)
        except Exception as e:
            Logger.log_message_static(f"Data-Signal: Failed to convert input to array: {str(e)}", Logger.ERROR)
            return False

    if len(arr) == 0:
        Logger.log_message_static("Data-Signal: Empty signal, cannot determine if digital", Logger.WARNING)
        return False

    try:
        # Check if the array has boolean dtype (from astype(bool))
        if arr.dtype == np.bool_ or arr.dtype == bool:
            Logger.log_message_static("Data-Signal: Signal has boolean dtype, classified as digital", Logger.DEBUG)
            return True

        # Check for boolean-like strings
        if arr.dtype == object:
            Logger.log_message_static("Data-Signal: Signal has object dtype, checking for TRUE/FALSE values", Logger.DEBUG)
            unique_vals = set(str(val).upper() for val in arr if val is not None)
            if unique_vals.issubset({'TRUE', 'FALSE'}):
                Logger.log_message_static(f"Data-Signal: Signal classified as digital with values: {unique_vals}", Logger.DEBUG)
                return True
            else:
                Logger.log_message_static(f"Data-Signal: Signal contains non-boolean values: {unique_vals}", Logger.DEBUG)

        # Check for numeric values (e.g., 0/1)
        elif np.issubdtype(arr.dtype, np.number):
            unique_vals = np.unique(arr[~np.isnan(arr)])
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                Logger.log_message_static(f"Data-Signal: Signal classified as digital with numeric values: {unique_vals}",
                                          Logger.INFO)
                return True
            else:
                Logger.log_message_static(f"Data-Signal: Signal contains non-binary numeric values: {unique_vals}", Logger.DEBUG)

        Logger.log_message_static("Data-Signal: Signal classified as analog", Logger.DEBUG)
        return False
    except Exception as e:
        Logger.log_message_static(f"Data-Signal: Error determining if signal is digital: {str(e)}", Logger.ERROR)
        return False