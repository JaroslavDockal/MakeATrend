"""
LVM parser implementation.

This parser handles LabVIEW Measurement (LVM) files and returns data
in the same format as the StandardParser for compatibility.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from datetime import datetime
import re

from utils.logger import Logger


class LVMParser:
    """
    Parser for LabVIEW Measurement (LVM) files.
    Returns data in the same format as StandardParser for compatibility.
    """

    def __init__(self):
        """Initialize the LVM parser."""
        pass

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this parser.

        Returns:
            List of supported file extensions
        """
        return ['.lvm']

    def parse_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Parse a LVM file and extract timestamps and signals.

        Args:
            file_path: Path to the LVM file to parse

        Returns:
            Tuple containing:
            - np.ndarray: Array of timestamps (float, seconds since epoch)
            - Dict[str, np.ndarray]: Dictionary of signal name -> signal values

        Raises:
            ValueError: If the file cannot be parsed
        """
        Logger.log_message_static(f"Parser-LVM: Parsing LVM file: {os.path.basename(file_path)}", Logger.DEBUG)

        if not os.path.exists(file_path):
            Logger.log_message_static(f"Parser-LVM: File not found: {file_path}", Logger.ERROR)
            raise ValueError(f"File not found: {file_path}")

        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    content = f.read()
            except Exception as e:
                Logger.log_message_static(f"Parser-LVM: Failed to read file: {str(e)}", Logger.ERROR)
                raise ValueError(f"Failed to read file: {e}")

        # Parse LVM header and data
        header_info, data_lines = self._parse_lvm_content(content)

        if not data_lines:
            Logger.log_message_static("Parser-LVM: No data found in LVM file", Logger.ERROR)
            raise ValueError("No data found in LVM file")

        # Parse data section
        timestamps, signals = self._parse_data_section(data_lines, header_info)

        if not signals:
            Logger.log_message_static("Parser-LVM: No signals could be parsed from the file", Logger.ERROR)
            raise ValueError("No signals could be parsed from the file")

        Logger.log_message_static(f"Parser-LVM: Successfully parsed {len(signals)} signals from LVM file", Logger.INFO)
        return timestamps, signals

    def _parse_lvm_content(self, content: str) -> Tuple[Dict[str, Any], List[str]]:
        """
        Parse LVM file content to extract header information and data lines.

        Args:
            content: File content as string

        Returns:
            Tuple of header info dict and data lines list
        """
        lines = content.strip().split('\n')
        header_info = {}
        data_start_idx = 0

        # Find data section start
        for i, line in enumerate(lines):
            line = line.strip()

            # LVM files typically have headers starting with specific keywords
            if line.startswith('***End_of_Header***') or line == '***End_of_Header***':
                data_start_idx = i + 1
                break
            elif line and not line.startswith('#') and not line.startswith(';'):
                # Try to detect if this looks like data (contains numbers/tabs)
                if '\t' in line or re.search(r'\d+\.?\d*', line):
                    data_start_idx = i
                    break

            # Parse header information
            if line.startswith('Date\t') or line.startswith('Time\t'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    header_info[parts[0]] = parts[1]
            elif line.startswith('Channels\t'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        header_info['channels'] = int(parts[1])
                    except ValueError:
                        pass
            elif line.startswith('Samples\t'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        header_info['samples'] = int(parts[1])
                    except ValueError:
                        pass

        data_lines = lines[data_start_idx:] if data_start_idx < len(lines) else []

        Logger.log_message_static(f"Parser-LVM: Found {len(data_lines)} data lines", Logger.DEBUG)
        return header_info, data_lines

    def _parse_data_section(self, data_lines: List[str], header_info: Dict[str, Any]) -> Tuple[
        np.ndarray, Dict[str, np.ndarray]]:
        """
        Parse the data section of LVM file.

        Args:
            data_lines: List of data line strings
            header_info: Header information dictionary

        Returns:
            Tuple of timestamps and signals
        """
        # Filter out empty lines and comments
        clean_lines = []
        for line in data_lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith(';'):
                clean_lines.append(line)

        if not clean_lines:
            raise ValueError("No valid data lines found")

        # Try to parse as tab-delimited data
        try:
            # Create DataFrame from tab-delimited data
            data_rows = []
            for line in clean_lines:
                # Replace common decimal separators
                cleaned_line = line.replace(',', '.')
                parts = cleaned_line.split('\t')
                data_rows.append(parts)

            # Find the maximum number of columns
            max_cols = max(len(row) for row in data_rows)

            # Pad shorter rows
            for row in data_rows:
                while len(row) < max_cols:
                    row.append('')

            # Create DataFrame
            df = pd.DataFrame(data_rows)

            Logger.log_message_static(f"Parser-LVM: Created DataFrame with shape {df.shape}", Logger.DEBUG)

        except Exception as e:
            Logger.log_message_static(f"Parser-LVM: Failed to parse data section: {str(e)}", Logger.ERROR)
            raise ValueError(f"Failed to parse data section: {e}")

        # Generate timestamps
        timestamps = self._generate_timestamps(len(df), header_info)

        # Process signals
        signals = {}

        for col_idx in range(len(df.columns)):
            col_name = f"Channel_{col_idx}"
            col_data = df.iloc[:, col_idx]

            # Skip empty columns
            if col_data.isna().all() or (col_data == '').all():
                continue

            # Process signal data
            processed_signal = self._process_signal_column(col_data, col_name)
            if processed_signal is not None:
                signals[col_name] = processed_signal

        return timestamps, signals

    def _process_signal_column(self, col_data: pd.Series, col_name: str) -> np.ndarray:
        """
        Process a single signal column.

        Args:
            col_data: Column data as pandas Series
            col_name: Column name

        Returns:
            Processed signal array or None if processing fails
        """
        try:
            # Convert to string and check for boolean values
            str_data = col_data.astype(str).str.strip().str.upper()

            # Check for boolean values
            true_mask = str_data == "TRUE"
            false_mask = str_data == "FALSE"

            if true_mask.any() or false_mask.any():
                Logger.log_message_static(f"Parser-LVM: Detected boolean signal in column '{col_name}'", Logger.DEBUG)
                return str_data.to_numpy()

            # Try numeric conversion
            try:
                # Clean numeric data
                cleaned_data = col_data.astype(str).str.replace(',', '.', regex=False)
                cleaned_data = cleaned_data.replace('', np.nan)

                numeric_data = pd.to_numeric(cleaned_data, errors='coerce')

                # Check if conversion was successful for most values
                valid_ratio = (~numeric_data.isna()).sum() / len(numeric_data)
                if valid_ratio > 0.5:  # At least 50% valid numeric values
                    Logger.log_message_static(f"Parser-LVM: Successfully converted column '{col_name}' to numeric",
                                              Logger.DEBUG)
                    return numeric_data.to_numpy(dtype=np.float32)
                else:
                    Logger.log_message_static(f"Parser-LVM: Column '{col_name}' contains mostly non-numeric values",
                                              Logger.WARNING)
                    return None

            except Exception as e:
                Logger.log_message_static(f"Parser-LVM: Failed to convert column '{col_name}' to numeric: {str(e)}",
                                          Logger.WARNING)
                return None

        except Exception as e:
            Logger.log_message_static(f"Parser-LVM: Error processing column '{col_name}': {str(e)}", Logger.WARNING)
            return None

    def _generate_timestamps(self, num_samples: int, header_info: Dict[str, Any]) -> np.ndarray:
        """
        Generate timestamps for the data.

        Args:
            num_samples: Number of data samples
            header_info: Header information dictionary

        Returns:
            Array of timestamps
        """
        try:
            # Try to use header information for timestamps
            base_time = datetime.now().timestamp()

            if 'Date' in header_info and 'Time' in header_info:
                try:
                    date_str = header_info['Date']
                    time_str = header_info['Time']
                    datetime_str = f"{date_str} {time_str}"

                    # Try different datetime formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%d.%m.%Y %H:%M:%S']:
                        try:
                            dt = datetime.strptime(datetime_str, fmt)
                            base_time = dt.timestamp()
                            Logger.log_message_static(f"Parser-LVM: Using header timestamp: {datetime_str}",
                                                      Logger.DEBUG)
                            break
                        except ValueError:
                            continue
                except Exception as e:
                    Logger.log_message_static(f"Parser-LVM: Failed to parse header timestamp: {str(e)}", Logger.WARNING)

            # Generate timestamps (assume 1 Hz sampling if no other info available)
            timestamps = base_time + np.arange(num_samples, dtype=float)

            Logger.log_message_static(f"Parser-LVM: Generated {len(timestamps)} timestamps", Logger.DEBUG)
            return timestamps

        except Exception as e:
            Logger.log_message_static(f"Parser-LVM: Error generating timestamps: {str(e)}", Logger.WARNING)
            # Fallback to simple sequential timestamps
            current_time = datetime.now().timestamp()
            return current_time + np.arange(num_samples, dtype=float)