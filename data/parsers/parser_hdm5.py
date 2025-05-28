"""
HDF5 parser implementation.

This parser handles HDF5 files and returns data
in the same format as the StandardParser for compatibility.
"""

import os
import numpy as np
from typing import Tuple, Dict, Any, List
from datetime import datetime

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

from utils.logger import Logger


class HDF5Parser:
    """
    Parser for HDF5 files.
    Returns data in the same format as StandardParser for compatibility.
    """

    def __init__(self):
        """Initialize the HDF5 parser."""
        if not H5PY_AVAILABLE:
            raise ImportError("h5py library is required for HDF5 parsing. Install with: pip install h5py")

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this parser.

        Returns:
            List of supported file extensions
        """
        return ['.h5', '.hdf5', '.hdf']

    def parse_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Parse a HDF5 file and extract timestamps and signals.

        Args:
            file_path: Path to the HDF5 file to parse

        Returns:
            Tuple containing:
            - np.ndarray: Array of timestamps (float, seconds since epoch)
            - Dict[str, np.ndarray]: Dictionary of signal name -> signal values

        Raises:
            ValueError: If the file cannot be parsed
        """
        Logger.log_message_static(f"Parser-HDF5: Parsing HDF5 file: {os.path.basename(file_path)}", Logger.DEBUG)

        if not os.path.exists(file_path):
            Logger.log_message_static(f"Parser-HDF5: File not found: {file_path}", Logger.ERROR)
            raise ValueError(f"File not found: {file_path}")

        try:
            with h5py.File(file_path, 'r') as f:
                Logger.log_message_static("Parser-HDF5: Successfully opened HDF5 file", Logger.DEBUG)

                # Extract timestamps and signals
                timestamps, signals = self._extract_data_from_hdf5(f)

        except Exception as e:
            Logger.log_message_static(f"Parser-HDF5: Failed to open HDF5 file: {str(e)}", Logger.ERROR)
            raise ValueError(f"Failed to open HDF5 file: {e}")

        if not signals:
            Logger.log_message_static("Parser-HDF5: No signals could be parsed from the file", Logger.ERROR)
            raise ValueError("No signals could be parsed from the file")

        # Ensure all signals have the same length as timestamps
        timestamps, signals = self._align_data_lengths(timestamps, signals)

        Logger.log_message_static(f"Parser-HDF5: Successfully parsed {len(signals)} signals from HDF5 file",
                                  Logger.INFO)
        return timestamps, signals

    def _extract_data_from_hdf5(self, h5_file) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract data from HDF5 file structure.

        Args:
            h5_file: Open HDF5 file object

        Returns:
            Tuple of timestamps and signals
        """
        timestamps = None
        signals = {}

        # Recursively traverse HDF5 structure
        def visit_item(name, obj):
            nonlocal timestamps, signals

            if isinstance(obj, h5py.Dataset):
                Logger.log_message_static(f"Parser-HDF5: Processing dataset '{name}'", Logger.DEBUG)

                try:
                    data = obj[:]

                    # Skip empty datasets
                    if data.size == 0:
                        Logger.log_message_static(f"Parser-HDF5: Dataset '{name}' is empty, skipping", Logger.WARNING)
                        return

                    # Check if this could be timestamp data
                    if timestamps is None and self._could_be_timestamps(data, name):
                        timestamps = self._process_timestamp_data(data, name)
                        if timestamps is not None:
                            Logger.log_message_static(f"Parser-HDF5: Using dataset '{name}' as timestamps",
                                                      Logger.DEBUG)
                            return

                    # Process as signal data
                    processed_signal = self._process_signal_data(data, name)
                    if processed_signal is not None:
                        signals[name] = processed_signal
                        Logger.log_message_static(
                            f"Parser-HDF5: Added signal '{name}' with {len(processed_signal)} points", Logger.DEBUG)

                except Exception as e:
                    Logger.log_message_static(f"Parser-HDF5: Error processing dataset '{name}': {str(e)}",
                                              Logger.WARNING)

        # Visit all items in the HDF5 file
        h5_file.visititems(visit_item)

        # If no timestamps found, generate synthetic ones
        if timestamps is None and signals:
            Logger.log_message_static("Parser-HDF5: No timestamps found, generating synthetic timestamps",
                                      Logger.WARNING)
            timestamps = self._generate_synthetic_timestamps(signals)

        return timestamps, signals

    def _could_be_timestamps(self, data: np.ndarray, name: str) -> bool:
        """
        Check if a dataset could contain timestamp data.

        Args:
            data: Dataset array
            name: Dataset name

        Returns:
            True if data could be timestamps
        """
        # Check name hints
        name_lower = name.lower()
        time_keywords = ['time', 'timestamp', 'datetime', 'date', 't']
        if any(keyword in name_lower for keyword in time_keywords):
            return True

        # Check data characteristics for numeric timestamps
        if np.issubdtype(data.dtype, np.number):
            # Flatten if multidimensional
            flat_data = data.flatten()

            # Check if values are in reasonable timestamp range
            if len(flat_data) > 0:
                min_val, max_val = np.min(flat_data), np.max(flat_data)

                # Unix timestamp range (1970-2100)
                if 0 < min_val < 4102444800 and max_val > min_val:
                    return True

                # Relative time (starting from 0, reasonable increments)
                if min_val >= 0 and max_val - min_val > 0:
                    return True

        return False

    def _process_timestamp_data(self, data: np.ndarray, name: str) -> np.ndarray:
        """
        Process timestamp data.

        Args:
            data: Raw timestamp data
            name: Dataset name

        Returns:
            Processed timestamp array or None if processing fails
        """
        try:
            # Flatten if multidimensional
            flat_data = data.flatten()

            if np.issubdtype(data.dtype, np.number):
                # Convert to float64 for timestamps
                timestamps = flat_data.astype(np.float64)

                # Check if timestamps look like Unix timestamps
                min_val = np.min(timestamps)
                if min_val > 1e9:  # Looks like Unix timestamp
                    return timestamps
                else:
                    # Relative timestamps - convert to absolute
                    current_time = datetime.now().timestamp()
                    return current_time + timestamps

            elif data.dtype.kind in ['U', 'S']:  # String timestamps
                # Try to parse string dates
                try:
                    # Common string timestamp formats
                    str_data = data.astype(str)
                    parsed_times = []

                    for time_str in str_data:
                        # Try different formats
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f',
                                    '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
                            try:
                                dt = datetime.strptime(time_str.strip(), fmt)
                                parsed_times.append(dt.timestamp())
                                break
                            except ValueError:
                                continue
                        else:
                            # If no format worked, use current time + index
                            parsed_times.append(datetime.now().timestamp() + len(parsed_times))

                    return np.array(parsed_times, dtype=np.float64)

                except Exception:
                    return None

            return None

        except Exception as e:
            Logger.log_message_static(f"Parser-HDF5: Error processing timestamps from '{name}': {str(e)}",
                                      Logger.WARNING)
            return None

    def _process_signal_data(self, data: np.ndarray, name: str) -> np.ndarray:
        """
        Process signal data to match StandardParser format.

        Args:
            data: Raw signal data
            name: Dataset name

        Returns:
            Processed signal array or None if processing fails
        """
        try:
            # Handle different data types and shapes
            if data.ndim > 1:
                # For multidimensional data, flatten or take first column
                if data.shape[1] == 1:
                    data = data.flatten()
                else:
                    Logger.log_message_static(f"Parser-HDF5: Dataset '{name}' is multidimensional, taking first column",
                                              Logger.WARNING)
                    data = data[:, 0]

            if data.dtype == bool:
                # Convert boolean to string format like StandardParser
                Logger.log_message_static(f"Parser-HDF5: Detected boolean signal in dataset '{name}'", Logger.DEBUG)
                return np.array(['TRUE' if x else 'FALSE' for x in data])

            elif np.issubdtype(data.dtype, np.number):
                # Numeric data - convert to float32 like StandardParser
                Logger.log_message_static(f"Parser-HDF5: Converting dataset '{name}' to numeric", Logger.DEBUG)
                return data.astype(np.float32)

            elif data.dtype.kind in ['U', 'S', 'O']:  # String data
                # Try to detect if strings represent boolean values
                str_data = data.astype(str)
                upper_data = np.char.upper(str_data)

                if np.any(np.isin(upper_data, ['TRUE', 'FALSE'])):
                    Logger.log_message_static(f"Parser-HDF5: Detected boolean strings in dataset '{name}'",
                                              Logger.DEBUG)
                    return upper_data

                # Try numeric conversion
                try:
                    # Replace common decimal separators
                    cleaned_data = np.char.replace(str_data, ',', '.')
                    numeric_data = cleaned_data.astype(float)
                    Logger.log_message_static(f"Parser-HDF5: Successfully converted string dataset '{name}' to numeric",
                                              Logger.DEBUG)
                    return numeric_data.astype(np.float32)
                except (ValueError, TypeError):
                    Logger.log_message_static(f"Parser-HDF5: Dataset '{name}' contains non-numeric strings, skipping",
                                              Logger.WARNING)
                    return None

            else:
                Logger.log_message_static(f"Parser-HDF5: Unsupported data type in dataset '{name}': {data.dtype}",
                                          Logger.WARNING)
                return None

        except Exception as e:
            Logger.log_message_static(f"Parser-HDF5: Error processing signal data for '{name}': {str(e)}",
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

        Logger.log_message_static(f"Parser-HDF5: Generated {len(timestamps)} synthetic timestamps", Logger.DEBUG)
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
                    f"Parser-HDF5: Truncating signal '{signal_name}' from {len(signal_data)} to {target_length} points",
                    Logger.WARNING)
                aligned_signals[signal_name] = signal_data[:target_length]
            else:
                # Signal too short - pad with last value or NaN
                Logger.log_message_static(
                    f"Parser-HDF5: Signal '{signal_name}' too short ({len(signal_data)} < {target_length}), padding",
                    Logger.WARNING)
                if signal_data.dtype.kind in ['U', 'S']:  # String data
                    padded = np.pad(signal_data, (0, target_length - len(signal_data)), mode='edge')
                else:  # Numeric data
                    padded = np.pad(signal_data, (0, target_length - len(signal_data)), mode='constant',
                                    constant_values=np.nan)
                aligned_signals[signal_name] = padded

        return timestamps, aligned_signals