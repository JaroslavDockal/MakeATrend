"""
MAT parser implementation.

This parser handles MATLAB .mat files and returns data
in the same format as the StandardParser for compatibility.
"""

import os
import numpy as np
from typing import Tuple, Dict, Any, List
from datetime import datetime

try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from utils.logger import Logger


class MATParser:
    """
    Parser for MATLAB .mat files.
    Returns data in the same format as StandardParser for compatibility.
    """

    def __init__(self):
        """Initialize the MAT parser."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy library is required for MAT parsing. Install with: pip install scipy")

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this parser.

        Returns:
            List of supported file extensions
        """
        return ['.mat']

    def parse_file(self, file_path: str) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
        """
        Parse a MAT file and extract timestamps and signals.

        Args:
            file_path: Path to the MAT file to parse

        Returns:
            Tuple containing:
            - Dictionary mapping signal names to tuples of (time_array, values_array)
            - Dictionary of metadata about the file

        Raises:
            ValueError: If the file cannot be parsed
        """
        Logger.log_message_static(f"Parser-MAT: Parsing MAT file: {os.path.basename(file_path)}", Logger.DEBUG)

        if not os.path.exists(file_path):
            Logger.log_message_static(f"Parser-MAT: File not found: {file_path}", Logger.ERROR)
            raise ValueError(f"File not found: {file_path}")

        try:
            # Load MAT file
            Logger.log_message_static("Parser-MAT: Loading MAT file", Logger.DEBUG)
            mat_data = loadmat(file_path, squeeze_me=True)

            Logger.log_message_static(f"Parser-MAT: Loaded MAT file with {len(mat_data)} variables", Logger.DEBUG)

        except Exception as e:
            Logger.log_message_static(f"Parser-MAT: Failed to load MAT file: {str(e)}", Logger.ERROR)
            raise ValueError(f"Failed to load MAT file: {e}")

        # Extract timestamps and signals
        timestamps, signals = self._extract_data_from_mat(mat_data)

        if not signals:
            Logger.log_message_static("Parser-MAT: No signals could be parsed from the file", Logger.ERROR)
            raise ValueError("No signals could be parsed from the file")

        # Ensure all signals have the same length as timestamps
        timestamps, signals = self._align_data_lengths(timestamps, signals)

        # Convert to master parser format: Dict[str, Tuple[np.ndarray, np.ndarray]]
        result_signals = {}
        metadata = {"source_file": file_path, "parser": "MATParser"}

        for name, values in signals.items():
            result_signals[name] = (timestamps, values)

        Logger.log_message_static(f"Parser-MAT: Successfully parsed {len(signals)} signals from MAT file", Logger.INFO)
        return result_signals, metadata

    def _extract_data_from_mat(self, mat_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract data from MAT file dictionary.

        Args:
            mat_data: Dictionary from loadmat

        Returns:
            Tuple of timestamps and signals
        """
        timestamps = None
        signals = {}

        # Filter out MATLAB metadata
        filtered_data = {k: v for k, v in mat_data.items()
                         if not k.startswith('__') and not k in ['#refs#', '#subsystem#']}

        Logger.log_message_static(f"Parser-MAT: Processing {len(filtered_data)} variables", Logger.DEBUG)

        for var_name, var_data in filtered_data.items():
            Logger.log_message_static(f"Parser-MAT: Processing variable '{var_name}'", Logger.DEBUG)

            try:
                # Convert to numpy array if not already
                if not isinstance(var_data, np.ndarray):
                    if np.isscalar(var_data):
                        Logger.log_message_static(f"Parser-MAT: Variable '{var_name}' is scalar, skipping",
                                                  Logger.DEBUG)
                        continue
                    var_data = np.array(var_data)

                # Skip empty arrays
                if var_data.size == 0:
                    Logger.log_message_static(f"Parser-MAT: Variable '{var_name}' is empty, skipping", Logger.WARNING)
                    continue

                # Check if this could be timestamp data (only if we don't have timestamps yet)
                if timestamps is None and self._could_be_timestamps(var_data, var_name):
                    timestamps = self._process_timestamp_data(var_data, var_name)
                    if timestamps is not None:
                        Logger.log_message_static(f"Parser-MAT: Using variable '{var_name}' as timestamps",
                                                  Logger.DEBUG)
                        continue

                # Process as signal data
                processed_signal = self._process_signal_data(var_data, var_name)
                if processed_signal is not None:
                    signals[var_name] = processed_signal
                    Logger.log_message_static(
                        f"Parser-MAT: Added signal '{var_name}' with {len(processed_signal)} points", Logger.DEBUG)

            except Exception as e:
                Logger.log_message_static(f"Parser-MAT: Error processing variable '{var_name}': {str(e)}",
                                          Logger.WARNING)

        # If no timestamps found, generate synthetic ones
        if timestamps is None and signals:
            Logger.log_message_static("Parser-MAT: No timestamps found, generating synthetic timestamps",
                                      Logger.WARNING)
            timestamps = self._generate_synthetic_timestamps(signals)

        return timestamps, signals

    def _could_be_timestamps(self, data: np.ndarray, name: str) -> bool:
        """
        Check if a variable could contain timestamp data.

        Args:
            data: Variable array
            name: Variable name

        Returns:
            True if data could be timestamps
        """
        # Check name hints
        name_lower = name.lower()
        time_keywords = ['time', 'timestamp', 'datetime', 'date', 't', 'zeit', 'cas', 'doba']
        if any(keyword in name_lower for keyword in time_keywords):
            return True

        # Check data characteristics for numeric timestamps
        if np.issubdtype(data.dtype, np.number) and data.ndim <= 2:
            # Flatten if multidimensional
            flat_data = data.flatten()

            # Need at least 2 points for timestamps and reasonable length
            if 2 <= len(flat_data) <= 1000000:
                try:
                    min_val, max_val = np.min(flat_data), np.max(flat_data)

                    # Skip arrays with all same values
                    if min_val == max_val:
                        return False

                    # Unix timestamp range (1970-2100)
                    if 0 < min_val < 4102444800 and max_val > min_val:
                        return True

                    # MATLAB datenum range (days since year 0)
                    if 719529 < min_val < 767011 and max_val > min_val:  # Roughly 1970-2100 in datenum
                        return True

                    # Relative time (starting from 0 or small value, reasonable increments)
                    if 0 <= min_val < 1000 and max_val - min_val > 0:
                        # Check if it looks like a time series (monotonic increase)
                        if len(flat_data) > 2:
                            diffs = np.diff(flat_data)
                            # At least 80% of differences should be positive (mostly increasing)
                            positive_ratio = np.sum(diffs > 0) / len(diffs)
                            if positive_ratio > 0.8:
                                return True

                except Exception:
                    pass

        return False

    def _process_timestamp_data(self, data: np.ndarray, name: str) -> np.ndarray:
        """
        Process timestamp data.

        Args:
            data: Raw timestamp data
            name: Variable name

        Returns:
            Processed timestamp array or None if processing fails
        """
        try:
            # Handle different array shapes
            if data.ndim > 1:
                if data.shape[1] == 1:
                    data = data.flatten()
                elif data.shape[0] == 1:
                    data = data.flatten()
                else:
                    Logger.log_message_static(
                        f"Parser-MAT: Timestamp variable '{name}' is multidimensional, taking first column",
                        Logger.WARNING)
                    data = data[:, 0]

            if np.issubdtype(data.dtype, np.number):
                # Convert to float64 for timestamps
                timestamps = data.astype(np.float64)

                # Remove any NaN or infinite values
                if np.any(~np.isfinite(timestamps)):
                    Logger.log_message_static(f"Parser-MAT: Removing {np.sum(~np.isfinite(timestamps))} invalid timestamp values", Logger.WARNING)
                    valid_mask = np.isfinite(timestamps)
                    timestamps = timestamps[valid_mask]

                if len(timestamps) < 2:
                    Logger.log_message_static(f"Parser-MAT: Not enough valid timestamps in '{name}'", Logger.WARNING)
                    return None

                # Check timestamp format and convert if needed
                min_val = np.min(timestamps)
                max_val = np.max(timestamps)

                if min_val > 1e9:  # Looks like Unix timestamp (after ~1973)
                    return timestamps
                elif min_val > 719529:  # MATLAB datenum (days since year 0)
                    # Convert MATLAB datenum to Unix timestamp
                    Logger.log_message_static(f"Parser-MAT: Converting MATLAB datenum to Unix timestamps", Logger.DEBUG)
                    unix_timestamps = (timestamps - 719529) * 86400
                    return unix_timestamps
                elif 0 <= min_val < 1000 and max_val > min_val:
                    # Relative timestamps - convert to absolute
                    Logger.log_message_static(f"Parser-MAT: Converting relative timestamps to absolute", Logger.DEBUG)
                    current_time = datetime.now().timestamp()
                    return current_time + timestamps
                else:
                    Logger.log_message_static(f"Parser-MAT: Timestamp range ({min_val:.2f} to {max_val:.2f}) not recognized", Logger.WARNING)
                    return None

            return None

        except Exception as e:
            Logger.log_message_static(f"Parser-MAT: Error processing timestamps from '{name}': {str(e)}",
                                      Logger.WARNING)
            return None

    def _process_signal_data(self, data: np.ndarray, name: str) -> np.ndarray:
        """
        Process signal data to match StandardParser format.

        Args:
            data: Raw signal data
            name: Variable name

        Returns:
            Processed signal array or None if processing fails
        """
        try:
            # Handle different data types and shapes
            if data.ndim > 2:
                Logger.log_message_static(
                    f"Parser-MAT: Variable '{name}' has too many dimensions ({data.ndim}), skipping", Logger.WARNING)
                return None

            if data.ndim == 2:
                # For 2D data, check if it's a single column/row or matrix
                if data.shape[1] == 1:
                    data = data.flatten()
                elif data.shape[0] == 1:
                    data = data.flatten()
                elif min(data.shape) <= 3:
                    # Small matrix - take the largest dimension as the signal
                    if data.shape[0] > data.shape[1]:
                        Logger.log_message_static(f"Parser-MAT: Variable '{name}' is a matrix, taking first column", Logger.WARNING)
                        data = data[:, 0]
                    else:
                        Logger.log_message_static(f"Parser-MAT: Variable '{name}' is a matrix, taking first row", Logger.WARNING)
                        data = data[0, :]
                else:
                    Logger.log_message_static(f"Parser-MAT: Variable '{name}' is a large matrix ({data.shape}), skipping", Logger.WARNING)
                    return None

            # Skip if too few data points
            if len(data) < 2:
                Logger.log_message_static(f"Parser-MAT: Variable '{name}' has too few data points", Logger.WARNING)
                return None

            if data.dtype == bool:
                # Convert boolean to string format like StandardParser
                Logger.log_message_static(f"Parser-MAT: Detected boolean signal in variable '{name}'", Logger.DEBUG)
                return np.array(['TRUE' if x else 'FALSE' for x in data])

            elif np.issubdtype(data.dtype, np.number):
                # Numeric data - handle NaN and infinite values
                Logger.log_message_static(f"Parser-MAT: Converting variable '{name}' to numeric", Logger.DEBUG)

                # Replace inf with NaN, then handle NaN
                data_clean = np.where(np.isfinite(data), data, np.nan)

                # If too many NaN values, skip
                nan_ratio = np.sum(np.isnan(data_clean)) / len(data_clean)
                if nan_ratio > 0.5:
                    Logger.log_message_static(f"Parser-MAT: Variable '{name}' has too many NaN values ({nan_ratio:.1%}), skipping", Logger.WARNING)
                    return None

                # Fill remaining NaN with interpolation or zero
                if nan_ratio > 0:
                    Logger.log_message_static(f"Parser-MAT: Filling {nan_ratio:.1%} NaN values in '{name}'", Logger.DEBUG)
                    # Simple forward fill, then backward fill, then zero fill
                    mask = ~np.isnan(data_clean)
                    if np.any(mask):
                        # Forward fill
                        for i in range(1, len(data_clean)):
                            if np.isnan(data_clean[i]) and not np.isnan(data_clean[i-1]):
                                data_clean[i] = data_clean[i-1]
                        # Backward fill
                        for i in range(len(data_clean)-2, -1, -1):
                            if np.isnan(data_clean[i]) and not np.isnan(data_clean[i+1]):
                                data_clean[i] = data_clean[i+1]
                        # Zero fill any remaining
                        data_clean = np.where(np.isnan(data_clean), 0, data_clean)

                return data_clean.astype(np.float32)

            elif data.dtype.kind in ['U', 'S', 'O']:  # String data
                # Handle MATLAB cell arrays and string arrays
                if data.dtype == object:
                    # Try to extract strings from cell array
                    try:
                        str_data = np.array([str(item) if item is not None else '' for item in data.flatten()])
                    except Exception:
                        Logger.log_message_static(f"Parser-MAT: Cannot convert object array '{name}' to strings",
                                                  Logger.WARNING)
                        return None
                else:
                    str_data = data.astype(str)

                # Try to detect if strings represent boolean values
                upper_data = np.char.upper(str_data)
                unique_vals = np.unique(upper_data)
                bool_vals = set(['TRUE', 'FALSE', '1', '0', 'YES', 'NO', 'ON', 'OFF'])

                if len(unique_vals) <= 2 and any(val in bool_vals for val in unique_vals):
                    Logger.log_message_static(f"Parser-MAT: Detected boolean strings in variable '{name}'", Logger.DEBUG)
                    # Normalize to TRUE/FALSE
                    result = np.where(np.isin(upper_data, ['TRUE', '1', 'YES', 'ON']), 'TRUE', 'FALSE')
                    return result

                # Try numeric conversion
                try:
                    # Replace common decimal separators and clean whitespace
                    cleaned_data = np.char.strip(str_data)
                    cleaned_data = np.char.replace(cleaned_data, ',', '.')

                    # Handle empty strings
                    cleaned_data = np.where(cleaned_data == '', '0', cleaned_data)

                    # Try to convert to numeric
                    numeric_data = []
                    for val in cleaned_data:
                        try:
                            numeric_data.append(float(val))
                        except:
                            numeric_data.append(np.nan)

                    numeric_data = np.array(numeric_data)

                    # Check if conversion was successful for most values
                    valid_ratio = np.sum(np.isfinite(numeric_data)) / len(numeric_data)
                    if valid_ratio > 0.7:  # At least 70% of values converted successfully
                        Logger.log_message_static(
                            f"Parser-MAT: Successfully converted string variable '{name}' to numeric ({valid_ratio:.1%} success rate)",
                            Logger.DEBUG)
                        # Fill NaN with 0
                        numeric_data = np.where(np.isfinite(numeric_data), numeric_data, 0)
                        return numeric_data.astype(np.float32)
                    else:
                        Logger.log_message_static(
                            f"Parser-MAT: Variable '{name}' has too many non-numeric strings ({valid_ratio:.1%} success rate), skipping",
                            Logger.WARNING)
                        return None

                except Exception as conv_e:
                    Logger.log_message_static(
                        f"Parser-MAT: Failed to convert string variable '{name}' to numeric: {str(conv_e)}",
                        Logger.WARNING)
                    return None

            else:
                Logger.log_message_static(f"Parser-MAT: Unsupported data type in variable '{name}': {data.dtype}",
                                          Logger.WARNING)
                return None

        except Exception as e:
            Logger.log_message_static(f"Parser-MAT: Error processing signal data for '{name}': {str(e)}",
                                      Logger.ERROR)
            return None

    def _generate_synthetic_timestamps(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate synthetic timestamps when none are available.

        Args:
            signals: Dictionary of signals

        Returns:
            Synthetic timestamp array
        """
        if not signals:
            return np.array([])

        # Get length from first signal
        signal_length = len(next(iter(signals.values())))

        # Generate timestamps at 1 Hz starting from current time
        current_time = datetime.now().timestamp()
        timestamps = current_time + np.arange(signal_length, dtype=float)

        Logger.log_message_static(f"Parser-MAT: Generated {len(timestamps)} synthetic timestamps", Logger.DEBUG)
        return timestamps

    def _align_data_lengths(self, timestamps: np.ndarray, signals: Dict[str, np.ndarray]) -> Tuple[
        np.ndarray, Dict[str, np.ndarray]]:
        """
        Ensure all signals have the same length as timestamps.

        Args:
            timestamps: Timestamp array
            signals: Dictionary of signals

        Returns:
            Tuple of aligned timestamps and signals
        """
        if not signals:
            return timestamps, signals

        target_length = len(timestamps)
        aligned_signals = {}

        for signal_name, signal_data in signals.items():
            if len(signal_data) == target_length:
                aligned_signals[signal_name] = signal_data
            elif len(signal_data) > target_length:
                # Truncate signal
                Logger.log_message_static(
                    f"Parser-MAT: Truncating signal '{signal_name}' from {len(signal_data)} to {target_length} points",
                    Logger.WARNING)
                aligned_signals[signal_name] = signal_data[:target_length]
            else:
                # Signal too short - pad with last value or NaN
                Logger.log_message_static(
                    f"Parser-MAT: Signal '{signal_name}' too short ({len(signal_data)} < {target_length}), padding",
                    Logger.WARNING)
                if signal_data.dtype.kind in ['U', 'S']:  # String data
                    padded = np.pad(signal_data, (0, target_length - len(signal_data)), mode='edge')
                else:  # Numeric data
                    padded = np.pad(signal_data, (0, target_length - len(signal_data)), mode='constant',
                                    constant_values=np.nan)
                aligned_signals[signal_name] = padded

        return timestamps, aligned_signals