"""
TDMS parser implementation.

This parser handles TDMS files (National Instruments format) and returns data
in the same format as the StandardParser for compatibility.
"""

import os
import numpy as np
from typing import Tuple, Dict, Any, List
from datetime import datetime

try:
    from nptdms import TdmsFile

    NPTDMS_AVAILABLE = True
except ImportError:
    NPTDMS_AVAILABLE = False

from utils.logger import Logger


class TDMSParser:
    """
    Parser for TDMS files (National Instruments format).
    Returns data in the same format as StandardParser for compatibility.
    """

    def __init__(self):
        """Initialize the TDMS parser."""
        if not NPTDMS_AVAILABLE:
            raise ImportError("npTDMS library is required for TDMS parsing. Install with: pip install npTDMS")

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this parser.

        Returns:
            List of supported file extensions
        """
        return ['.tdms']

    def parse_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Parse a TDMS file and extract timestamps and signals.

        Args:
            file_path: Path to the TDMS file to parse

        Returns:
            Tuple containing:
            - np.ndarray: Array of timestamps (float, seconds since epoch)
            - Dict[str, np.ndarray]: Dictionary of signal name -> signal values

        Raises:
            ValueError: If the file cannot be parsed
        """
        Logger.log_message_static(f"Parser-TDMS: Parsing TDMS file: {os.path.basename(file_path)}", Logger.DEBUG)

        if not os.path.exists(file_path):
            Logger.log_message_static(f"Parser-TDMS: File not found: {file_path}", Logger.ERROR)
            raise ValueError(f"File not found: {file_path}")

        try:
            # Open TDMS file
            Logger.log_message_static("Parser-TDMS: Opening TDMS file", Logger.DEBUG)
            tdms_file = TdmsFile.read(file_path)

            if not tdms_file.groups():
                Logger.log_message_static("Parser-TDMS: No groups found in TDMS file", Logger.ERROR)
                raise ValueError("No groups found in TDMS file")

            Logger.log_message_static(f"Parser-TDMS: Found {len(tdms_file.groups())} group(s)", Logger.DEBUG)

        except Exception as e:
            Logger.log_message_static(f"Parser-TDMS: Failed to open TDMS file: {str(e)}", Logger.ERROR)
            raise ValueError(f"Failed to open TDMS file: {e}")

        # Extract timestamps and signals
        timestamps = None
        signals = {}

        try:
            # Iterate through all groups and channels
            for group in tdms_file.groups():
                Logger.log_message_static(f"Parser-TDMS: Processing group '{group.name}'", Logger.DEBUG)

                for channel in group.channels():
                    channel_name = f"{group.name}/{channel.name}" if group.name else channel.name
                    Logger.log_message_static(f"Parser-TDMS: Processing channel '{channel_name}'", Logger.DEBUG)

                    # Get channel data
                    data = channel[:]

                    if data is None or len(data) == 0:
                        Logger.log_message_static(f"Parser-TDMS: Channel '{channel_name}' is empty, skipping",
                                                  Logger.WARNING)
                        continue

                    # Handle timestamps
                    if timestamps is None:
                        timestamps = self._extract_timestamps(channel)
                        if timestamps is not None:
                            Logger.log_message_static(
                                f"Parser-TDMS: Extracted {len(timestamps)} timestamps from channel '{channel_name}'",
                                Logger.DEBUG)

                    # Process signal data
                    processed_data = self._process_signal_data(data, channel_name)
                    if processed_data is not None:
                        signals[channel_name] = processed_data
                        Logger.log_message_static(
                            f"Parser-TDMS: Added signal '{channel_name}' with {len(processed_data)} points",
                            Logger.DEBUG)

        except Exception as e:
            Logger.log_message_static(f"Parser-TDMS: Error processing TDMS data: {str(e)}", Logger.ERROR)
            raise ValueError(f"Error processing TDMS data: {e}")

        # Validate results
        if timestamps is None:
            Logger.log_message_static("Parser-TDMS: No timestamps found, generating synthetic timestamps",
                                      Logger.WARNING)
            timestamps = self._generate_synthetic_timestamps(signals)

        if not signals:
            Logger.log_message_static("Parser-TDMS: No signals could be parsed from the file", Logger.ERROR)
            raise ValueError("No signals could be parsed from the file")

        # Ensure all signals have the same length as timestamps
        timestamps, signals = self._align_data_lengths(timestamps, signals)

        Logger.log_message_static(f"Parser-TDMS: Successfully parsed {len(signals)} signals from TDMS file",
                                  Logger.INFO)
        return timestamps, signals

    def _extract_timestamps(self, channel) -> np.ndarray:
        """
        Extract timestamps from a TDMS channel.

        Args:
            channel: TDMS channel object

        Returns:
            np.ndarray: Array of timestamps in seconds since epoch, or None if not found
        """
        try:
            # Check if channel has time information
            if hasattr(channel, 'time_track') and channel.time_track() is not None:
                time_data = channel.time_track()
                # Convert datetime objects to timestamps
                if len(time_data) > 0 and hasattr(time_data[0], 'timestamp'):
                    timestamps = np.array([t.timestamp() for t in time_data])
                    return timestamps

            # Check channel properties for timing information
            properties = channel.properties
            if 'wf_start_time' in properties and 'wf_increment' in properties:
                start_time = properties['wf_start_time']
                increment = properties['wf_increment']
                data_length = len(channel[:])

                # Convert start_time to timestamp if it's a datetime object
                if hasattr(start_time, 'timestamp'):
                    start_timestamp = start_time.timestamp()
                elif isinstance(start_time, (int, float)):
                    start_timestamp = start_time
                else:
                    return None

                # Generate timestamps
                timestamps = start_timestamp + np.arange(data_length) * increment
                return timestamps

            return None

        except Exception as e:
            Logger.log_message_static(f"Parser-TDMS: Error extracting timestamps: {str(e)}", Logger.WARNING)
            return None

    def _process_signal_data(self, data: np.ndarray, channel_name: str) -> np.ndarray:
        """
        Process signal data to match StandardParser format.

        Args:
            data: Raw channel data
            channel_name: Name of the channel

        Returns:
            np.ndarray: Processed signal data, or None if processing fails
        """
        try:
            # Handle different data types
            if data.dtype == bool:
                # Convert boolean to string format like StandardParser
                Logger.log_message_static(f"Parser-TDMS: Detected boolean signal in channel '{channel_name}'",
                                          Logger.DEBUG)
                return np.array(['TRUE' if x else 'FALSE' for x in data])

            elif np.issubdtype(data.dtype, np.number):
                # Numeric data - convert to float32 like StandardParser
                Logger.log_message_static(f"Parser-TDMS: Converting channel '{channel_name}' to numeric", Logger.DEBUG)
                return data.astype(np.float32)

            elif data.dtype.kind in ['U', 'S', 'O']:  # String data
                # Try to detect if strings represent boolean values
                str_data = data.astype(str)
                upper_data = np.char.upper(str_data)

                if np.any(np.isin(upper_data, ['TRUE', 'FALSE'])):
                    Logger.log_message_static(f"Parser-TDMS: Detected boolean strings in channel '{channel_name}'",
                                              Logger.DEBUG)
                    return upper_data

                # Try numeric conversion
                try:
                    # Replace common decimal separators
                    cleaned_data = np.char.replace(str_data, ',', '.')
                    numeric_data = cleaned_data.astype(float)
                    Logger.log_message_static(
                        f"Parser-TDMS: Successfully converted string channel '{channel_name}' to numeric", Logger.DEBUG)
                    return numeric_data.astype(np.float32)
                except (ValueError, TypeError):
                    Logger.log_message_static(
                        f"Parser-TDMS: Channel '{channel_name}' contains non-numeric strings, skipping", Logger.WARNING)
                    return None

            else:
                Logger.log_message_static(
                    f"Parser-TDMS: Unsupported data type in channel '{channel_name}': {data.dtype}", Logger.WARNING)
                return None

        except Exception as e:
            Logger.log_message_static(f"Parser-TDMS: Error processing signal data for '{channel_name}': {str(e)}",
                                      Logger.WARNING)
            return None

    def _generate_synthetic_timestamps(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate synthetic timestamps when none are available.

        Args:
            signals: Dictionary of signals

        Returns:
            np.ndarray: Synthetic timestamp array
        """
        if not signals:
            return np.array([])

        # Get length from first signal
        signal_length = len(next(iter(signals.values())))

        # Generate timestamps at 1 Hz starting from current time
        current_time = datetime.now().timestamp()
        timestamps = current_time + np.arange(signal_length, dtype=float)

        Logger.log_message_static(f"Parser-TDMS: Generated {len(timestamps)} synthetic timestamps", Logger.DEBUG)
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
                    f"Parser-TDMS: Truncating signal '{signal_name}' from {len(signal_data)} to {target_length} points",
                    Logger.WARNING)
                aligned_signals[signal_name] = signal_data[:target_length]
            else:
                # Signal too short - pad with last value or NaN
                Logger.log_message_static(
                    f"Parser-TDMS: Signal '{signal_name}' too short ({len(signal_data)} < {target_length}), padding",
                    Logger.WARNING)
                if signal_data.dtype.kind in ['U', 'S']:  # String data
                    padded = np.pad(signal_data, (0, target_length - len(signal_data)), mode='edge')
                else:  # Numeric data
                    padded = np.pad(signal_data, (0, target_length - len(signal_data)), mode='constant',
                                    constant_values=np.nan)
                aligned_signals[signal_name] = padded

        return timestamps, aligned_signals