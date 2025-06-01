"""
Common utilities for signal preparation and validation.

This module provides essential functions used across all signal analysis calculations:
- Signal validation and cleaning
- Sampling rate estimation
- Input parameter validation

All calculation modules should import from this module for consistent signal handling.
"""

import numpy as np
from PySide6.QtWidgets import QMessageBox

from utils.logger import Logger


def safe_sample_rate(time_arr):
    """
    Calculate sampling rate from time array with robust error handling.

    Computes sampling rate as 1/mean(time_differences). Handles edge cases like
    insufficient data points and non-finite values. Works with both ascending and
    descending time arrays.

    Args:
        time_arr (np.ndarray): Time values array (at least 2 elements required).

    Returns:
        float: Sampling rate in inverse time units, or 0.0 if calculation fails.

    Example:
        >>> time_arr = np.linspace(0, 1, 1000)  # 1 second, 1000 samples
        >>> fs = safe_sample_rate(time_arr)
        >>> print(f"Sampling rate: {fs:.1f} Hz")
        Sampling rate: 999.0 Hz
    """
    Logger.log_message_static(
        f"Calculations-Common: Calculating sampling rate from time array with {len(time_arr)} points", Logger.DEBUG)

    # Validate input array length
    if len(time_arr) < 2:
        Logger.log_message_static(
            f"Insufficient data points for sampling rate calculation: {len(time_arr)} < 2",
            Logger.WARNING
        )
        return 0.0

    # Calculate time differences between consecutive samples
    dt = np.diff(time_arr)
    Logger.log_message_static(
        f"Time differences calculated - Min: {np.min(dt):.6f}, Max: {np.max(dt):.6f}, "
        f"Std: {np.std(dt):.6f}",
        Logger.DEBUG
    )

    # Check if time array is in descending order and adjust accordingly
    if np.mean(dt) < 0:
        dt = np.abs(dt)
        Logger.log_message_static(
            "Detected descending time array. Using absolute time differences.",
            Logger.INFO
        )

    # Calculate mean time difference
    mean_dt = np.mean(dt)
    Logger.log_message_static(f"Calculations-Common: Mean time difference: {mean_dt:.6f}", Logger.DEBUG)

    # Validate mean time difference
    if mean_dt <= 0:
        Logger.log_message_static(
            f"Invalid mean time difference: {mean_dt} (non-positive)",
            Logger.ERROR
        )
        return 0.0

    if not np.isfinite(mean_dt):
        Logger.log_message_static(
            f"Invalid mean time difference: {mean_dt} (non-finite)",
            Logger.ERROR
        )
        return 0.0

    # Calculate and return sampling rate
    sample_rate = 1.0 / mean_dt
    Logger.log_message_static(
        f"Successfully calculated sampling rate: {sample_rate:.2f} Hz",
        Logger.INFO
    )

    # Check for unusually high or low sampling rates
    if sample_rate > 1e6:
        Logger.log_message_static(
            f"Warning: Very high sampling rate detected ({sample_rate:.0f} Hz). "
            "Please verify time units are correct.",
            Logger.WARNING
        )
    elif sample_rate < 0.1:
        Logger.log_message_static(
            f"Warning: Very low sampling rate detected ({sample_rate:.3f} Hz). "
            "Please verify time units are correct.",
            Logger.WARNING
        )

    return sample_rate


def safe_prepare_signal(values, dialog, title="Signal Validation"):
    """
    Validate and clean signal data by handling non-finite values and zero-only signals.

    This is the primary signal preprocessing function used by all analysis calculations.
    It handles common data quality issues and provides user interaction for problematic data.

    Args:
        values (np.ndarray): Input signal values to validate and clean.
        dialog (QDialog): Parent window for user interaction dialogs.
        title (str, optional): Dialog title for user messages. Defaults to "Signal Validation".

    Returns:
        np.ndarray or None: Cleaned signal array, or None if validation fails
                           or user cancels the operation.

    Raises:
        None: All exceptions are caught and logged. Returns None on error.

    Example:
        >>> values = np.array([1, 2, np.nan, 4, 5])
        >>> clean_values = safe_prepare_signal(values, dialog, "FFT Analysis")
        >>> # User will be prompted to replace NaN with zeros
        >>> print(clean_values)  # [1, 2, 0, 4, 5] if user accepts
    """
    Logger.log_message_static(f"Calculations-Common: Starting signal validation for '{title}'", Logger.DEBUG)

    try:
        values = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as e:
        Logger.log_message_static(f"Calculations-Common: Cannot convert input to numpy array: {e}", Logger.ERROR)
        if dialog:
            QMessageBox.critical(dialog, title, f"Invalid signal data: {str(e)}")
        return None

    Logger.log_message_static(
        f"Input signal characteristics - Shape: {values.shape}, dtype: {values.dtype}, "
        f"Min: {np.nanmin(values):.6f}, Max: {np.nanmax(values):.6f}",
        Logger.DEBUG
    )

    # Check for non-finite values (NaN, +Inf, -Inf)
    non_finite_mask = ~np.isfinite(values)
    non_finite_count = np.sum(non_finite_mask)

    if non_finite_count > 0:
        # Count specific types of non-finite values
        nan_count = np.sum(np.isnan(values))
        posinf_count = np.sum(np.isposinf(values))
        neginf_count = np.sum(np.isneginf(values))

        Logger.log_message_static(
            f"Detected {non_finite_count} non-finite values - "
            f"NaN: {nan_count}, +Inf: {posinf_count}, -Inf: {neginf_count}",
            Logger.WARNING
        )

        # Create detailed user dialog
        if dialog:
            msg = QMessageBox(dialog)
            msg.setWindowTitle(title)
            msg.setText("Signal Validation Issue Detected")
            msg.setInformativeText(
                f"Your signal contains {non_finite_count} non-finite values:\n"
                f"• NaN (Not a Number): {nan_count}\n"
                f"• Positive Infinity: {posinf_count}\n"
                f"• Negative Infinity: {neginf_count}\n\n"
                f"These values must be replaced with 0.0 for analysis to continue.\n"
                f"Do you want to proceed with this replacement?"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)

            # Set button text for clarity
            yes_button = msg.button(QMessageBox.StandardButton.Yes)
            yes_button.setText("Replace with Zeros")
            cancel_button = msg.button(QMessageBox.StandardButton.Cancel)
            cancel_button.setText("Cancel Processing")

            response = msg.exec()

            if response == QMessageBox.StandardButton.Cancel:
                Logger.log_message_static("Calculations-Common: User canceled non-finite value replacement",
                                          Logger.WARNING)
                return None

        # Replace non-finite values with zeros
        Logger.log_message_static("Calculations-Common: Replacing non-finite values with zeros", Logger.INFO)
        values_cleaned = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        # Verify replacement was successful
        remaining_non_finite = np.sum(~np.isfinite(values_cleaned))
        if remaining_non_finite > 0:
            Logger.log_message_static(
                f"Error: {remaining_non_finite} non-finite values remain after replacement",
                Logger.ERROR
            )
            if dialog:
                QMessageBox.critical(
                    dialog, title,
                    f"Failed to replace all non-finite values. {remaining_non_finite} values remain problematic.\n"
                    "Please check your input data."
                )
            return None

        Logger.log_message_static(f"Calculations-Common: Successfully replaced {non_finite_count} non-finite values",
                                  Logger.INFO)
        values = values_cleaned
    else:
        Logger.log_message_static("Calculations-Common: No non-finite values detected in signal", Logger.DEBUG)

    # Check for zero-only signal
    zero_count = np.sum(values == 0)
    total_count = len(values)

    if np.all(values == 0):
        Logger.log_message_static(
            f"Signal validation failed: All {total_count} values are zero",
            Logger.ERROR
        )
        if dialog:
            QMessageBox.information(
                dialog, title,
                f"Signal Processing Error\n\n"
                f"All {total_count} signal values are zero. Analysis cannot continue.\n\n"
                f"Possible causes:\n"
                f"• Sensor disconnection or malfunction\n"
                f"• Incorrect data scaling or units\n"
                f"• Data acquisition issues\n\n"
                f"Please verify your signal source and data."
            )
        return None

    # Log signal statistics after validation
    Logger.log_message_static(
        f"Signal validation completed successfully - "
        f"Non-zero values: {total_count - zero_count}/{total_count}, "
        f"Range: [{np.min(values):.6f}, {np.max(values):.6f}]",
        Logger.INFO
    )

    if zero_count > 0:
        zero_percentage = (zero_count / total_count) * 100
        Logger.log_message_static(
            f"Signal contains {zero_count} zero values ({zero_percentage:.1f}% of total)",
            Logger.INFO
        )

        # Warn if high percentage of zeros
        if zero_percentage > 50:
            Logger.log_message_static(
                f"Warning: High percentage of zero values ({zero_percentage:.1f}%) may affect analysis quality",
                Logger.WARNING
            )

    return values


def extended_prepare_signal(values, dialog, title="Signal Analysis"):
    """
    Extended signal preparation with polarity handling for specialized analyses.

    This function provides additional preprocessing for analyses that require positive values
    or have specific requirements for signal polarity (like logarithmic operations).

    Args:
        values (np.ndarray): Pre-validated signal values (should come from safe_prepare_signal).
        dialog (QDialog): Parent dialog for user interaction.
        title (str): Title for user dialogs.

    Returns:
        np.ndarray or None: Processed signal or None if user cancels.

    Note:
        This function should only be called after safe_prepare_signal() validation.
    """
    Logger.log_message_static(f"Calculations-Common: Starting extended signal preparation for '{title}'", Logger.DEBUG)

    # This should only receive pre-validated signals
    if values is None:
        return None

    total_values = len(values)
    negative_count = np.sum(values < 0)
    positive_count = np.sum(values > 0)
    zero_count = np.sum(values == 0)

    Logger.log_message_static(
        f"Signal composition - Total: {total_values}, Positive: {positive_count}, "
        f"Negative: {negative_count}, Zero: {zero_count}",
        Logger.DEBUG
    )

    # Skip processing if signal is entirely positive (most common case)
    if negative_count == 0:
        Logger.log_message_static("Calculations-Common: Signal is entirely positive, no preprocessing needed",
                                  Logger.DEBUG)
        return values

    # Calculate ratio for decision making
    if positive_count == 0:
        negative_ratio = 1.0
    else:
        negative_ratio = negative_count / positive_count

    # Handle negligible negative values (< 5% compared to positives)
    if negative_ratio < 0.05 and negative_count > 0:
        Logger.log_message_static(
            f"Detected negligible negative values ({negative_count}/{total_values}), "
            "automatically replacing with near-zero values",
            Logger.INFO
        )
        if dialog:
            QMessageBox.information(
                dialog, title,
                f"Detected {negative_count} negligible negative values out of {total_values} total.\n"
                "These will be automatically replaced with near-zero positive values (1e-10) "
                "to enable logarithmic analysis."
            )
        values = values.copy()
        values[values <= 0] = 1e-10
        return values

    # Handle negligible positive values (< 5% compared to negatives)
    if negative_ratio > 0.95 and positive_count > 0:
        Logger.log_message_static(
            f"Detected negligible positive values ({positive_count}/{total_values}), "
            "automatically flipping signal polarity",
            Logger.INFO
        )
        if dialog:
            QMessageBox.information(
                dialog, title,
                f"Detected {positive_count} negligible positive values out of {total_values} total.\n"
                "Signal polarity will be flipped (multiplied by -1) and positive values "
                "will be replaced with near-zero values."
            )
        values = values.copy()
        values[values >= 0] = -1e-10
        return -values

    # Handle mixed positive/negative values - require user decision
    Logger.log_message_static(
        f"Signal contains significant mix of positive ({positive_count}) and negative ({negative_count}) values, "
        "requiring user intervention",
        Logger.INFO
    )

    if not dialog:
        # If no dialog available, use absolute values as safe default
        Logger.log_message_static("Calculations-Common: No dialog available, using absolute values as default",
                                  Logger.INFO)
        return np.abs(values)

    msg = QMessageBox(dialog)
    msg.setWindowTitle(title)
    msg.setText("Signal Processing Decision Required")
    msg.setInformativeText(
        f"Your signal contains {positive_count} positive and {negative_count} negative values.\n"
        f"Ratio of negative to positive: {negative_ratio:.2f}\n\n"
        "Please choose how to process the signal for analysis:"
    )

    # Add processing option buttons
    abs_button = msg.addButton("Use Absolute Values", QMessageBox.ButtonRole.AcceptRole)
    abs_button.setToolTip("Convert all values to their absolute values (removes sign information)")
    pos_button = msg.addButton("Keep Positives Only", QMessageBox.ButtonRole.AcceptRole)
    pos_button.setToolTip("Replace negative values with near-zero, preserve positive values")
    neg_button = msg.addButton("Keep Negatives Only", QMessageBox.ButtonRole.AcceptRole)
    neg_button.setToolTip("Replace positive values with near-zero, flip and keep negative values")
    cancel_button = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    cancel_button.setToolTip("Cancel signal processing and return to previous step")

    msg.exec()

    # Process user choice
    if msg.clickedButton() == cancel_button:
        Logger.log_message_static("Calculations-Common: User canceled signal processing", Logger.DEBUG)
        return None
    elif msg.clickedButton() == abs_button:
        Logger.log_message_static("Calculations-Common: User selected absolute value transformation", Logger.INFO)
        result = np.abs(values)
        if dialog:
            QMessageBox.information(
                dialog, title,
                f"Applied absolute value transformation.\n"
                f"Result: {len(result)} values, range [{np.min(result):.6f}, {np.max(result):.6f}]"
            )
        return result
    elif msg.clickedButton() == pos_button:
        Logger.log_message_static("Calculations-Common: User selected positive value preservation", Logger.INFO)
        values_copy = values.copy()
        original_negatives = np.sum(values_copy < 0)
        values_copy[values_copy < 0] = 1e-10
        if dialog:
            QMessageBox.information(
                dialog, title,
                f"Replaced {original_negatives} negative values with 1e-10.\n"
                f"Preserved {positive_count} positive values."
            )
        return values_copy
    elif msg.clickedButton() == neg_button:
        Logger.log_message_static("Calculations-Common: User selected negative value preservation", Logger.INFO)
        values_copy = values.copy()
        original_positives = np.sum(values_copy > 0)
        values_copy[values_copy > 0] = 1e-10
        if dialog:
            QMessageBox.information(
                dialog, title,
                f"Replaced {original_positives} positive values with 1e-10.\n"
                f"Flipped and preserved {negative_count} negative values."
            )
        return -values_copy

    # Fallback case (should not occur with proper button handling)
    Logger.log_message_static("Calculations-Common: Unexpected dialog result, returning original values",
                              Logger.WARNING)
    return values


def validate_analysis_inputs(time_arr, values, min_length=2, require_positive_sample_rate=True):
    """
    Validate common inputs for analysis functions.

    Args:
        time_arr (np.ndarray): Time array
        values (np.ndarray): Signal values
        min_length (int): Minimum required signal length
        require_positive_sample_rate (bool): Whether to require valid sample rate

    Returns:
        tuple: (is_valid, error_message, sample_rate)
    """
    try:
        if len(values) < min_length:
            return False, f"Signal too short: {len(values)} < {min_length} samples", 0.0

        if len(time_arr) != len(values):
            return False, f"Time and value arrays have different lengths: {len(time_arr)} != {len(values)}", 0.0

        sample_rate = safe_sample_rate(time_arr)
        if require_positive_sample_rate and sample_rate <= 0:
            return False, "Cannot determine valid sampling rate", sample_rate

        return True, "", sample_rate

    except Exception as e:
        return False, f"Validation error: {str(e)}", 0.0


def format_array_for_display(arr, max_items=5, precision=3):
    """
    Format a numpy array for user-friendly display with limited items.

    Args:
        arr (np.ndarray): Array to format
        max_items (int): Maximum number of items to show
        precision (int): Number of decimal places

    Returns:
        str: Formatted string representation
    """
    if len(arr) == 0:
        return "[]"

    if len(arr) <= max_items:
        formatted = [f"{x:.{precision}f}" for x in arr]
        return f"[{', '.join(formatted)}]"
    else:
        formatted = [f"{x:.{precision}f}" for x in arr[:max_items]]
        return f"[{', '.join(formatted)}, ... ({len(arr) - max_items} more)]"


def calculate_bandwidth(freqs, psd, db_level=-3):
    """
    Calculate bandwidth at specified dB level below peak.

    Args:
        freqs (np.ndarray): Frequency values array in Hz.
        psd (np.ndarray): Power spectral density values.
        db_level (float): dB level below peak (default -3dB for half-power bandwidth).

    Returns:
        dict: Bandwidth information including lower/upper frequencies and bandwidth value.
    """
    try:
        # Find peak power and its frequency
        max_idx = np.argmax(psd)
        max_power = psd[max_idx]
        max_freq = freqs[max_idx]

        # Calculate threshold (e.g., -3dB = half power)
        threshold_linear = max_power * (10 ** (db_level / 10))

        # Find lower frequency (below the peak)
        lower_indices = np.where((freqs < max_freq) & (psd > threshold_linear))[0]
        if len(lower_indices) > 0:
            f_low = freqs[lower_indices[0]]
        else:
            f_low = freqs[0]  # Use minimum frequency as fallback

        # Find upper frequency (above the peak)
        upper_indices = np.where((freqs > max_freq) & (psd > threshold_linear))[0]
        if len(upper_indices) > 0:
            f_high = freqs[upper_indices[-1]]
        else:
            f_high = freqs[-1]  # Use maximum frequency as fallback

        # Calculate bandwidth
        bandwidth = f_high - f_low

        Logger.log_message_static(
            f"Bandwidth calculation ({db_level}dB): Peak at {max_freq:.2f} Hz, "
            f"threshold points at {f_low:.2f} Hz and {f_high:.2f} Hz, "
            f"bandwidth = {bandwidth:.2f} Hz",
            Logger.DEBUG
        )

        return {
            "bandwidth_hz": bandwidth,
            "lower_freq_hz": f_low,
            "upper_freq_hz": f_high,
            "peak_freq_hz": max_freq,
            "db_level": db_level
        }

    except Exception as e:
        Logger.log_message_static(f"Calculations-Common: Error calculating bandwidth: {str(e)}", Logger.ERROR)
        return {
            "bandwidth_hz": 0.0,
            "lower_freq_hz": 0.0,
            "upper_freq_hz": 0.0,
            "peak_freq_hz": 0.0,
            "db_level": db_level,
            "error": str(e)
        }