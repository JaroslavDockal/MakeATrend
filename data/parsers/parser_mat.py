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

    def parse_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Parse a MAT file and extract timestamps and signals.

        Args:
            file_path: Path to the MAT file to parse

        Returns:
            Tuple containing:
            - np.ndarray: Array of timestamps (float, seconds since epoch)
            - Dict[str, np.ndarray]: Dictionary of signal name -> signal values

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

        Logger.log_message_static(f"Parser-MAT: Successfully parsed {len(signals)} signals from MAT file", Logger.INFO)
        return timestamps, signals

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

                # Check if this could be timestamp data
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
        time_keywords = ['time', 'timestamp', 'datetime', 'date', 't', 'zeit']
        if any(keyword in name_lower for keyword in time_keywords):
            return True

        # Check data characteristics for numeric timestamps
        if np.issubdtype(data.dtype, np.number) and data.ndim <= 2:
            # Flatten if multidimensional
            flat_data = data.flatten()

            # Check if values are in reasonable timestamp range
            if len(flat_data) > 1:  # Need at least 2 points for timestamps
                min_val, max_val = np.min(flat_data), np.max(flat_data)

                # Unix timestamp range (1970-2100)
                if 0 < min_val < 4102444800 and max_val > min_val:
                    return True

                # Relative time (starting from 0, reasonable increments)
                if min_val >= 0 and max_val - min_val > 0:
                    # Check if it looks like a time series (monotonic increase)
                    if len(flat_data) > 2:
                        diffs = np.diff(flat_data)
                        if np.all(diffs >= 0):  # Monotonic increase
                            return True

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

                # Check if timestamps look like Unix timestamps
                min_val = np.min(timestamps)
                if min_val > 1e9:  # Looks like Unix timestamp
                    return timestamps
                elif min_val > 719529:  # MATLAB datenum (days since year 0)
                    # Convert MATLAB datenum to Unix timestamp
                    unix_timestamps = (timestamps - 719529) * 86400
                    return unix_timestamps
                else:
                    # Relative timestamps - convert to absolute
                    current_time = datetime.now().timestamp()
                    return current_time + timestamps

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
                else:
                    # Matrix data - take first column or create multiple signals would be better
                    # but for simplicity, we'll take the first column
                    Logger.log_message_static(f"Parser-MAT: Variable '{name}' is a matrix, taking first column",
                                              Logger.WARNING)
                    data = data[:, 0]

            if data.dtype == bool:
                # Convert boolean to string format like StandardParser
                Logger.log_message_static(f"Parser-MAT: Detected boolean signal in variable '{name}'", Logger.DEBUG)
                return np.array(['TRUE' if x else 'FALSE' for x in data])

            elif np.issubdtype(data.dtype, np.number):
                # Numeric data - convert to float32 like StandardParser
                Logger.log_message_static(f"Parser-MAT: Converting variable '{name}' to numeric", Logger.DEBUG)
                return data.astype(np.float32)

            elif data.dtype.kind in ['U', 'S', 'O']:  # String data
                # Handle MATLAB cell arrays and string arrays
                if data.dtype == object:
                    # Try to extract strings from cell array
                    try:
                        str_data = np.array([str(item) for item in data.flatten()])
                    except Exception:
                        Logger.log_message_static(f"Parser-MAT: Cannot convert object array '{name}' to strings",
                                                  Logger.WARNING)
                        return None
                else:
                    str_data = data.astype(str)

                # Try to detect if strings represent boolean values
                upper_data = np.char.upper(str_data)

                if np.any(np.isin(upper_data, ['TRUE', 'FALSE'])):
                    Logger.log_message_static(f"Parser-MAT: Detected boolean strings in variable '{name}'",
                                              Logger.DEBUG)
                    return upper_data

                # Try numeric conversion
                try:
                    # Replace common decimal separators
                    cleaned_data = np.char.replace(str_data, ',', '.')
                    numeric_data = cleaned_data.astype(float)
                    Logger.log_message_static(f"Parser-MAT: Successfully converted string variable '{name}' to numeric",
                                              Logger.DEBUG)
                    return numeric_data.astype(np.float32)
                except (ValueError, TypeError):
                    Logger.log_message_static(f"Parser-MAT: Variable '{name}' contains non-numeric strings, skipping",
                                              Logger.WARNING)
                    return None

            else:
                Logger.log_message_static(f"Parser-MAT: Unsupported data type in variable '{name}': {data.dtype}",
                                          Logger.WARNING)
                return None

        except Exception as e:
            Logger.log_message_static(f"Parser-MAT: Error processing signal data for '{name}': {str(e)}",
                                      Logger.WARNING)
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