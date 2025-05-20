import numpy as np
from PySide6.QtWidgets import QMessageBox, QDialog

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
    """
    Logger.log_message_static(f"Calculating sampling rate from time array with {len(time_arr)} points", Logger.DEBUG)

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
    time_direction = 1
    if np.mean(dt) < 0:
        time_direction = -1
        dt = np.abs(dt)
        Logger.log_message_static(
            "Detected descending time array. Using absolute time differences.",
            Logger.INFO
        )

    # Calculate mean time difference
    mean_dt = np.mean(dt)
    Logger.log_message_static(f"Mean time difference: {mean_dt:.6f}", Logger.DEBUG)

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

    First stage of signal preprocessing. Detects NaN/Inf values and prompts user
    for replacement with zeros. Rejects signals containing only zeros.

    Args:
        values (np.ndarray): Input signal values to validate and clean.
        dialog (QDialog): Parent window for user interaction dialogs.
        title (str, optional): Dialog title. Defaults to "Signal Validation".

    Returns:
        np.ndarray or None: Cleaned signal array, or None if validation fails
                           or user cancels the operation.
    """
    Logger.log_message_static(f"Starting signal validation for '{title}'", Logger.DEBUG)
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
            Logger.log_message_static("User canceled non-finite value replacement - signal validation failed",
                                      Logger.WARNING)
            return None

        # Replace non-finite values with zeros
        Logger.log_message_static("Replacing non-finite values with zeros", Logger.INFO)
        values_cleaned = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        # Verify replacement was successful
        remaining_non_finite = np.sum(~np.isfinite(values_cleaned))
        if remaining_non_finite > 0:
            Logger.log_message_static(
                f"Error: {remaining_non_finite} non-finite values remain after replacement",
                Logger.ERROR
            )
            QMessageBox.critical(
                dialog,
                title,
                f"Failed to replace all non-finite values. {remaining_non_finite} values remain problematic.\n"
                "Please check your input data."
            )
            return None

        Logger.log_message_static(f"Successfully replaced {non_finite_count} non-finite values with zeros", Logger.INFO)
        values = values_cleaned
    else:
        Logger.log_message_static("No non-finite values detected in signal", Logger.DEBUG)

    # Check for zero-only signal
    zero_count = np.sum(values == 0)
    total_count = len(values)

    if np.all(values == 0):
        Logger.log_message_static(
            f"Signal validation failed: All {total_count} values are zero",
            Logger.ERROR
        )
        QMessageBox.information(
            dialog,
            title,
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
    Interactive post-processor for handling signal polarity and zero-value cases.

    This function provides an extended validation and preprocessing pipeline for signal data,
    specifically designed to handle edge cases involving negative values, zero values, and
    mixed-polarity signals. It offers interactive user dialogs to determine the appropriate
    preprocessing strategy based on the signal characteristics.

    The function analyzes the signal composition and applies different strategies:
    - For negligible negative values (<5%): Automatically replaces with near-zero
    - For negligible positive values (<5%): Automatically flips and offsets the signal
    - For significant mixed values: Prompts user to choose processing method
    - For all-zero signals: Returns None with warning

    Args:
        dialog (QDialog): Parent dialog.
        values (np.ndarray): Signal values.
        title (str): Dialog title.

    Returns:
        np.ndarray or None: Processed signal or None if canceled.

    Raises:
        AttributeError: If dialog parameter doesn't support QMessageBox parent functionality.
        ValueError: If values parameter cannot be processed as NumPy array.
    """
    Logger.log_message_static(f"Starting extended signal preparation for '{title}'", Logger.DEBUG)
    Logger.log_message_static(f"Input signal shape: {values.shape}, dtype: {values.dtype}", Logger.DEBUG)

    # Calculate signal composition statistics
    total_values = len(values)
    negative_count = np.sum(values < 0)
    positive_count = np.sum(values > 0)
    zero_count = np.sum(values == 0)

    Logger.log_message_static(
        f"Signal composition - Total: {total_values}, Positive: {positive_count}, "
        f"Negative: {negative_count}, Zero: {zero_count}",
        Logger.DEBUG
    )

    # Handle empty signal
    if total_values == 0:
        Logger.log_message_static("Signal preprocessing failed: Empty signal detected", Logger.ERROR)
        QMessageBox.critical(dialog, title, "Cannot process empty signal. Please provide valid signal data.")
        return None

    # Handle all-zero signal
    if zero_count == total_values:
        Logger.log_message_static("Signal preprocessing stopped: All values are zero", Logger.WARNING)
        QMessageBox.information(dialog, title, "All values are zero. Cannot proceed.")
        return None

    # Calculate negative-to-positive ratio for decision making
    if positive_count == 0:
        negative_ratio = 1
        Logger.log_message_static("Signal contains only negative and zero values", Logger.DEBUG)
    else :
        negative_ratio = negative_count / positive_count
        Logger.log_message_static(f"Negative-to-positive ratio: {negative_ratio:.4f}", Logger.DEBUG)

    # Skip dialog if the signal is entirely positive
    if negative_count == 0:
        Logger.log_message_static("Signal is entirely positive. No further processing required.", Logger.DEBUG)
        return values

    # Handle negligible negative values (< 5% compared to positives)
    if negative_ratio < 0.05 and negative_count > 0:
        Logger.log_message_static(
            f"Detected negligible negative values ({negative_count}/{total_values}), "
            "automatically replacing with near-zero values",
            Logger.INFO
        )
        QMessageBox.information(
            dialog,
            title,
            f"Detected {negative_count} negligible negative values out of {total_values} total.\n"
            "These will be automatically replaced with near-zero positive values (1e-10) "
            "to enable logarithmic analysis."
        )
        values[values <= 0] = 1e-10
        return values

    # Handle negligible positive values (< 5% compared to negatives)
    if negative_ratio > 0.95 and positive_count > 0:
        Logger.log_message_static(
            f"Detected negligible positive values ({positive_count}/{total_values}), "
            "automatically flipping signal polarity",
            Logger.INFO
        )
        QMessageBox.information(
            dialog,
            title,
            f"Detected {positive_count} negligible positive values out of {total_values} total.\n"
            "Signal polarity will be flipped (multiplied by -1) and positive offsets "
            "will be replaced with near-zero values."
        )
        values[values >= 0] = -1e-10
        return -values

    # Handle mixed positive/negative values - require user decision
    Logger.log_message_static(
        f"Signal contains significant mix of positive ({positive_count}) and negative ({negative_count}) values, "
        "requiring user intervention",
        Logger.INFO
    )

    msg = QMessageBox(dialog)
    msg.setWindowTitle(title)
    msg.setText("Signal Processing Decision Required")
    msg.setInformativeText(
        f"Your signal contains {positive_count} positive and {negative_count} negative values.\n"
        f"Ratio of negative to positive: {negative_ratio:.2f}\n\n"
        "Please choose how to process the signal for analysis:"
    )

    # Add processing option buttons and execute dialog
    abs_button = msg.addButton("Use Absolute Values", QMessageBox.ButtonRole.AcceptRole)
    abs_button.setToolTip("Convert all values to their absolute values (removes sign information)")
    pos_button = msg.addButton("Keep Positives Only", QMessageBox.ButtonRole.AcceptRole)
    pos_button.setToolTip("Replace negative values with near-zero, preserve positive values")
    neg_button = msg.addButton("Keep Negatives Only", QMessageBox.ButtonRole.AcceptRole)
    neg_button.setToolTip("Replace positive values with near-zero, and flip negative values")
    cancel_button = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    cancel_button.setToolTip("Cancel signal processing and return to previous step")
    msg.exec()

    # Process user choice
    if msg.clickedButton() == cancel_button:
        Logger.log_message_static("User canceled signal processing - operation aborted", Logger.DEBUG)
        return None
    elif msg.clickedButton() == abs_button:
        Logger.log_message_static("User selected absolute value transformation", Logger.INFO)
        result = np.abs(values)
        QMessageBox.information(
            dialog,
            title,
            f"Applied absolute value transformation.\n"
            f"Result: {len(result)} values, range [{np.min(result):.6f}, {np.max(result):.6f}]"
        )
        return result
    elif msg.clickedButton() == pos_button:
        Logger.log_message_static("User selected positive value preservation", Logger.INFO)
        values_copy = values.copy()
        original_negatives = np.sum(values_copy < 0)
        values_copy[values_copy < 0] = 1e-10
        QMessageBox.information(
            dialog,
            title,
            f"Replaced {original_negatives} negative values with 1e-10.\n"
            f"Preserved {positive_count} positive values."
        )
        Logger.log_message_static(f"Replaced {original_negatives} negative values with near-zero", Logger.DEBUG)
        return values_copy
    elif msg.clickedButton() == neg_button:
        Logger.log_message_static("User selected negative value preservation", Logger.INFO)
        values_copy = values.copy()
        original_positives = np.sum(values_copy > 0)
        values_copy[values_copy > 0] = 1e-10
        QMessageBox.information(
            dialog,
            title,
            f"Replaced {original_positives} positive values with 1e-10.\n"
            f"Preserved {negative_count} negative values."
        )
        Logger.log_message_static(f"Replaced {original_positives} positive values with near-zero", Logger.DEBUG)
        return values_copy

    # Fallback case (should not occur with proper button handling)
    Logger.log_message_static("Unexpected dialog result, returning original values", Logger.WARNING)
    return values

def calculate_bandwidth(freqs, psd):
    """
    Calculate the 3dB bandwidth of a power spectral density.

    Finds the frequency width between points where the power drops to half (-3dB)
    of the maximum value. This is a common measure of signal bandwidth in
    frequency domain analysis.

    Args:
        freqs (np.ndarray): Frequency values array in Hz.
        psd (np.ndarray): Power spectral density values corresponding to freqs.

    Returns:
        str: Formatted bandwidth value in Hz, or "N/A"/"Error" if calculation fails.
    """
    try:
        # Find peak power and its frequency
        max_idx = np.argmax(psd)
        max_power = psd[max_idx]
        max_freq = freqs[max_idx]

        # Calculate half-power (-3dB) threshold
        threshold = max_power / 2

        # Find lower 3dB frequency (below the peak)
        lower_indices = np.where((freqs < max_freq) & (psd > threshold))[0]
        if len(lower_indices) > 0:
            lower_idx = lower_indices[0]  # First crossing from left
            f_low = freqs[lower_idx]
        else:
            Logger.log_message_static("Could not find lower 3dB point", Logger.WARNING)
            return "N/A"

        # Find upper 3dB frequency (above the peak)
        upper_indices = np.where((freqs > max_freq) & (psd > threshold))[0]
        if len(upper_indices) > 0:
            upper_idx = upper_indices[-1]  # Last crossing from right
            f_high = freqs[upper_idx]
        else:
            Logger.log_message_static("Could not find upper 3dB point", Logger.WARNING)
            return "N/A"

        # Calculate bandwidth
        bandwidth = f_high - f_low

        Logger.log_message_static(
            f"Bandwidth calculation: Peak at {max_freq:.2f} Hz, 3dB points at "
            f"{f_low:.2f} Hz and {f_high:.2f} Hz, width = {bandwidth:.2f} Hz",
            Logger.DEBUG
        )

        return f"{bandwidth:.2f} Hz"

    except Exception as e:
        Logger.log_message_static(f"Error calculating bandwidth: {str(e)}", Logger.ERROR)
        return "Error"

def format_array_for_display(arr, max_items=5):
    """Format a numpy array for user-friendly display with limited items."""
    if len(arr) <= max_items:
        return str([float(f"{x:.3f}") for x in arr])
    else:
        formatted = [float(f"{x:.3f}") for x in arr[:max_items]]
        return f"{formatted} ... ({len(arr)-max_items} more items)"