"""
Auto-detection CSV parser implementation.

This parser attempts to handle the maximum variety of CSV and text file formats
with robust detection of delimiters, headers, data types, and time formats.
It serves as a universal fallback parser that can handle almost any tabular data.
"""

import os
import csv
import io
import re
from typing import Tuple, Dict, Any, List, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd

from utils.logger import Logger


class AutoParser:
    """
    Universal parser with automatic format detection for various data formats.
    Handles CSV, TSV, space-separated, fixed-width, and other tabular formats.
    """

    def __init__(self):
        """Initialize the auto parser."""
        self.time_formats = [
            '%Y-%m-%d %H:%M:%S.%f',  # Standard format with microseconds
            '%Y-%m-%d %H:%M:%S',     # Standard format without microseconds
            '%d.%m.%Y %H:%M:%S',     # European format
            '%d/%m/%Y %H:%M:%S',     # European format with slashes
            '%m/%d/%Y %H:%M:%S',     # US format
            '%Y/%m/%d %H:%M:%S',     # ISO-like with slashes
            '%H:%M:%S.%f',           # Time only with microseconds
            '%H:%M:%S',              # Time only
            '%Y-%m-%d',              # Date only
            '%d.%m.%Y',              # European date only
            '%m/%d/%Y',              # US date only
        ]

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this parser.

        Returns:
            List of supported file extensions
        """
        return ['.csv', '.txt', '.dat', '.log', '.tsv', '.prn', '.asc', '.data']

    def parse_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Parse a file with automatic format detection.
        Returns format compatible with parser_old.py: (timestamps, signals)

        Args:
            file_path: Path to the file to parse

        Returns:
            Tuple containing:
            - np.ndarray: Array of timestamps (float, seconds since epoch)
            - Dict[str, np.ndarray]: Dictionary of signal name -> signal values

        Raises:
            ValueError: If the file cannot be parsed
        """
        Logger.log_message_static(f"Parser-Auto: Starting to parse file: {os.path.basename(file_path)}", Logger.INFO)

        try:
            # Try reading with different encodings
            content = self._read_file_robust(file_path)

            # Detect and parse format
            timestamps, signals = self._auto_detect_and_parse(content, file_path)

            if not signals:
                raise ValueError("No signals could be parsed from the file")

            Logger.log_message_static(f"Parser-Auto: Successfully parsed {len(signals)} signals", Logger.INFO)
            return timestamps, signals

        except Exception as e:
            error_msg = f"AutoParser failed to parse file {os.path.basename(file_path)}: {str(e)}"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

    def _read_file_robust(self, file_path: str) -> str:
        """
        Read file with robust encoding detection.
        """
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-16']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                Logger.log_message_static(f"Parser-Auto: Successfully read file with {encoding} encoding", Logger.DEBUG)
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                Logger.log_message_static(f"Parser-Auto: Failed to read with {encoding}: {str(e)}", Logger.DEBUG)
                continue

        raise ValueError("Could not read file with any supported encoding")

    def _auto_detect_and_parse(self, content: str, file_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Auto-detect format and parse content.
        """
        lines = content.strip().splitlines()
        if not lines:
            raise ValueError("File is empty")

        Logger.log_message_static(f"Parser-Auto: File contains {len(lines)} lines", Logger.DEBUG)

        # Try different parsing strategies in order of specificity
        strategies = [
            self._try_pandas_csv,
            self._try_fixed_width,
            self._try_regex_extraction,
            self._try_space_separated,
            self._try_aggressive_numeric
        ]

        for strategy in strategies:
            try:
                result = strategy(content, file_path)
                if result is not None:
                    timestamps, signals = result
                    if signals:  # Make sure we got some data
                        Logger.log_message_static(f"Parser-Auto: Successfully parsed using {strategy.__name__}", Logger.DEBUG)
                        return timestamps, signals
            except Exception as e:
                Logger.log_message_static(f"Parser-Auto: {strategy.__name__} failed: {str(e)}", Logger.DEBUG)
                continue

        raise ValueError("All parsing strategies failed")

    def _try_pandas_csv(self, content: str, file_path: str) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Try parsing with pandas using various delimiters and options.
        """
        Logger.log_message_static("Parser-Auto: Trying pandas CSV parsing", Logger.DEBUG)

        # Try different separators
        separators = [';', ',', '\t', '|', ' ', ':']

        for sep in separators:
            try:
                # Try with pandas
                df = pd.read_csv(io.StringIO(content), sep=sep, engine='python',
                               error_bad_lines=False, warn_bad_lines=False)

                if df.empty or len(df.columns) < 2:
                    continue

                Logger.log_message_static(f"Parser-Auto: Pandas parsed with separator '{sep}', {len(df)} rows, {len(df.columns)} columns", Logger.DEBUG)

                # Try to extract timestamps and signals
                timestamps, signals = self._extract_from_dataframe(df)
                if signals:
                    return timestamps, signals

            except Exception as e:
                Logger.log_message_static(f"Parser-Auto: Pandas failed with separator '{sep}': {str(e)}", Logger.DEBUG)
                continue

        return None

    def _extract_from_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract timestamps and signals from a pandas DataFrame.
        """
        signals = {}
        timestamps = None

        # Look for timestamp columns
        time_column = self._find_time_column(df)

        if time_column is not None:
            timestamps = self._parse_time_column(df[time_column])
            Logger.log_message_static(f"Parser-Auto: Found time column: {time_column}", Logger.DEBUG)
        else:
            # Check if we have Date and Time columns like in standard format
            if 'Date' in df.columns and 'Time' in df.columns:
                try:
                    combined_time = df['Date'].astype(str) + ' ' + df['Time'].astype(str).str.replace(',', '.')
                    timestamps = pd.to_datetime(combined_time, errors='coerce').astype(np.int64) / 1e9
                    timestamps = timestamps.to_numpy()
                    Logger.log_message_static("Parser-Auto: Combined Date and Time columns", Logger.DEBUG)
                except:
                    pass

            # If still no timestamps, create synthetic ones
            if timestamps is None:
                timestamps = np.arange(len(df), dtype=np.float64)
                Logger.log_message_static("Parser-Auto: Created synthetic timestamps", Logger.DEBUG)

        # Extract signal columns
        for col in df.columns:
            if col in ['Date', 'Time'] or (time_column is not None and col == time_column):
                continue

            try:
                # Handle boolean values
                if df[col].dtype == 'object':
                    str_values = df[col].astype(str).str.upper()
                    if str_values.isin(['TRUE', 'FALSE']).any():
                        signals[col] = str_values.to_numpy()
                        Logger.log_message_static(f"Parser-Auto: Detected boolean signal: {col}", Logger.DEBUG)
                        continue

                # Try numeric conversion
                if df[col].dtype == 'object':
                    # Handle European number format
                    cleaned = df[col].astype(str).str.replace(',', '.', regex=False)
                    numeric = pd.to_numeric(cleaned, errors='coerce')
                else:
                    numeric = pd.to_numeric(df[col], errors='coerce')

                if not numeric.isnull().all():
                    signals[col] = numeric.to_numpy(dtype=np.float32)
                    Logger.log_message_static(f"Parser-Auto: Extracted numeric signal: {col}", Logger.DEBUG)

            except Exception as e:
                Logger.log_message_static(f"Parser-Auto: Failed to process column {col}: {str(e)}", Logger.DEBUG)
                continue

        return timestamps, signals

    def _find_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find potential time column in DataFrame.
        """
        time_keywords = ['time', 'timestamp', 'datetime', 'date', 'zeit', 'hora', 'temps']

        # Check column names for time-like names
        for col in df.columns:
            if any(keyword in col.lower() for keyword in time_keywords):
                if self._is_time_like_column(df[col]):
                    return col

        # Check first column if it looks like time
        if len(df.columns) > 0:
            first_col = df.columns[0]
            if self._is_time_like_column(df[first_col]):
                return first_col

        return None

    def _is_time_like_column(self, series: pd.Series) -> bool:
        """
        Check if a pandas Series contains time-like data.
        """
        # Sample a few values
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False

        time_like_count = 0
        for value in sample:
            if self._is_time_like_value(str(value)):
                time_like_count += 1

        # Consider it time-like if at least 70% of sampled values look like time
        return time_like_count >= 0.7 * len(sample)

    def _is_time_like_value(self, value: str) -> bool:
        """
        Check if a string value looks like a timestamp.
        """
        # Try parsing with known time formats
        for fmt in self.time_formats:
            try:
                datetime.strptime(value.strip(), fmt)
                return True
            except:
                continue

        # Check for numeric timestamp (Unix time)
        try:
            num_val = float(value)
            # Reasonable range for Unix timestamps (1970-2100)
            if 0 < num_val < 4102444800:
                return True
        except:
            pass

        return False

    def _parse_time_column(self, series: pd.Series) -> np.ndarray:
        """
        Parse time column to timestamps.
        """
        # Try different approaches
        timestamps = None

        # Try pandas to_datetime first
        try:
            timestamps = pd.to_datetime(series, errors='coerce')
            if not timestamps.isnull().all():
                return timestamps.astype(np.int64) / 1e9
        except:
            pass

        # Try manual parsing with known formats
        for fmt in self.time_formats:
            try:
                timestamps = pd.to_datetime(series, format=fmt, errors='coerce')
                if not timestamps.isnull().all():
                    return timestamps.astype(np.int64) / 1e9
            except:
                continue

        # Try as numeric (Unix timestamps)
        try:
            numeric_times = pd.to_numeric(series, errors='coerce')
            if not numeric_times.isnull().all():
                return numeric_times.to_numpy(dtype=np.float64)
        except:
            pass

        # Fallback: create synthetic timestamps
        return np.arange(len(series), dtype=np.float64)

    def _try_fixed_width(self, content: str, file_path: str) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Try parsing as fixed-width format.
        """
        Logger.log_message_static("Parser-Auto: Trying fixed-width parsing", Logger.DEBUG)

        lines = content.strip().splitlines()
        if len(lines) < 2:
            return None

        # Detect column positions by looking for consistent spacing
        first_line = lines[0]
        positions = []

        # Find positions where columns consistently start
        for i, char in enumerate(first_line):
            if i == 0 or (char != ' ' and first_line[i-1] == ' '):
                positions.append(i)

        if len(positions) < 2:
            return None

        # Extract columns
        data = []
        for line in lines:
            row = []
            for i, pos in enumerate(positions):
                end_pos = positions[i+1] if i+1 < len(positions) else len(line)
                cell = line[pos:end_pos].strip()
                row.append(cell)
            data.append(row)

        if not data:
            return None

        # Convert to DataFrame and process
        df = pd.DataFrame(data[1:], columns=data[0] if data[0] else [f'Col_{i}' for i in range(len(data[1]))])
        return self._extract_from_dataframe(df)

    def _try_regex_extraction(self, content: str, file_path: str) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Try extracting data using regex patterns.
        """
        Logger.log_message_static("Parser-Auto: Trying regex extraction", Logger.DEBUG)

        # Pattern for structured log-like data: [timestamp] name: value
        log_pattern = r'\[([^\]]+)\]\s*(\w+):\s*([-+]?\d*\.?\d+)'

        matches = re.findall(log_pattern, content)
        if not matches:
            return None

        # Group by signal name
        signals = {}
        timestamps = []

        for timestamp_str, signal_name, value_str in matches:
            if signal_name not in signals:
                signals[signal_name] = []

            # Parse timestamp
            timestamp = self._parse_single_timestamp(timestamp_str)
            if timestamp is not None:
                timestamps.append(timestamp)
                signals[signal_name].append(float(value_str))

        if not signals:
            return None

        # Convert to numpy arrays and align
        unique_timestamps = sorted(set(timestamps))
        timestamp_array = np.array(unique_timestamps)

        aligned_signals = {}
        for name, values in signals.items():
            # Align values with timestamps (simplified)
            aligned_values = np.array(values[:len(timestamp_array)], dtype=np.float32)
            if len(aligned_values) > 0:
                aligned_signals[name] = aligned_values

        return timestamp_array, aligned_signals

    def _try_space_separated(self, content: str, file_path: str) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Try parsing as space-separated values with variable spacing.
        """
        Logger.log_message_static("Parser-Auto: Trying space-separated parsing", Logger.DEBUG)

        lines = content.strip().splitlines()
        if len(lines) < 2:
            return None

        # Skip comment lines
        data_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        if len(data_lines) < 2:
            return None

        # Try to parse as space-separated
        try:
            rows = []
            for line in data_lines:
                # Split on whitespace
                parts = line.split()
                if not parts:
                    continue
                rows.append(parts)

            if not rows:
                return None

            # Check if all rows have same number of columns
            col_counts = [len(row) for row in rows]
            if len(set(col_counts)) > 2:  # Allow some variation
                return None

            # Use first row as headers if it contains non-numeric data
            headers = None
            data_start = 0

            if rows and not all(self._is_numeric_string(cell) for cell in rows[0]):
                headers = rows[0]
                data_start = 1

            if not headers:
                headers = [f'Column_{i+1}' for i in range(len(rows[0]) if rows else 0)]

            # Create DataFrame
            df = pd.DataFrame(rows[data_start:], columns=headers[:len(rows[data_start]) if rows[data_start:] else 0])
            return self._extract_from_dataframe(df)

        except Exception:
            return None

    def _try_aggressive_numeric(self, content: str, file_path: str) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Aggressively extract any numeric data from the file.
        """
        Logger.log_message_static("Parser-Auto: Trying aggressive numeric extraction", Logger.DEBUG)

        lines = content.strip().splitlines()
        numeric_pattern = r'[-+]?(?:\d*\.\d+|\d+\.?\d*)'

        data_columns = {}
        row_count = 0

        for line in lines:
            if not line.strip() or line.strip().startswith('#'):
                continue

            # Find all numeric values
            matches = re.findall(numeric_pattern, line)
            if not matches:
                continue

            values = []
            for match in matches:
                try:
                    values.append(float(match))
                except ValueError:
                    continue

            if not values:
                continue

            # Add to columns
            for i, val in enumerate(values):
                col_name = f'Signal_{i+1}'
                if col_name not in data_columns:
                    data_columns[col_name] = []

                # Pad with NaN if needed
                while len(data_columns[col_name]) < row_count:
                    data_columns[col_name].append(float('nan'))

                data_columns[col_name].append(val)

            row_count += 1

        if not data_columns:
            return None

        # Create timestamps and signals
        max_len = max(len(values) for values in data_columns.values())
        timestamps = np.arange(max_len, dtype=np.float64)

        signals = {}
        for name, values in data_columns.items():
            # Pad to max length
            values.extend([float('nan')] * (max_len - len(values)))
            signals[name] = np.array(values, dtype=np.float32)

        Logger.log_message_static(f"Parser-Auto: Aggressively extracted {len(signals)} signals", Logger.DEBUG)
        return timestamps, signals

    def _parse_single_timestamp(self, timestamp_str: str) -> Optional[float]:
        """
        Parse a single timestamp string.
        """
        # Try known formats
        for fmt in self.time_formats:
            try:
                dt = datetime.strptime(timestamp_str.strip(), fmt)
                return dt.timestamp()
            except:
                continue

        # Try as float (Unix timestamp)
        try:
            return float(timestamp_str)
        except:
            pass

        return None

    def _is_numeric_string(self, s: str) -> bool:
        """
        Check if string represents a number.
        """
        try:
            float(s.replace(',', '.'))
            return True
        except ValueError:
            return False