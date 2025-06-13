"""
Base parser class with common functionality shared between different parser implementations.

This module contains the common time parsing, data extraction, and format detection
functionality that is shared between AutoParser and ExcelParser.
"""

import re
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from utils.logger import Logger


class BaseParser:
    """
    Base class containing common parsing functionality for different data formats.
    """

    def __init__(self):
        """Initialize the base parser with common time formats."""
        self.time_formats = [
            '%Y-%m-%d %H:%M:%S.%f',  # Standard format with microseconds
            '%Y-%m-%d %H:%M:%S',  # Standard format without microseconds
            '%d.%m.%Y %H:%M:%S',  # European format
            '%d/%m/%Y %H:%M:%S',  # European format with slashes
            '%m/%d/%Y %H:%M:%S',  # US format
            '%Y/%m/%d %H:%M:%S',  # ISO-like with slashes
            '%H:%M:%S.%f',  # Time only with microseconds
            '%H:%M:%S',  # Time only
            '%Y-%m-%d',  # Date only
            '%d.%m.%Y',  # European date only
            '%m/%d/%Y',  # US date only
        ]

    def extract_from_dataframe(self, df: pd.DataFrame, parser_name: str = "BaseParser") -> Tuple[
        np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract timestamps and signals from a pandas DataFrame.

        Args:
            df: The DataFrame to process
            parser_name: Name of the calling parser (for logging)

        Returns:
            Tuple of (timestamps, signals_dict)
        """
        signals = {}
        timestamps = None

        Logger.log_message_static(
            f"{parser_name}: Processing DataFrame with {len(df)} rows and {len(df.columns)} columns", Logger.DEBUG)

        # Look for timestamp columns
        time_column = self.find_time_column(df)

        if time_column is not None:
            timestamps = self.parse_time_column(df[time_column])
            Logger.log_message_static(f"{parser_name}: Found time column: {time_column}", Logger.DEBUG)
        else:
            # Check if we have Date and Time columns like in standard format
            timestamps = self._try_combine_date_time_columns(df, parser_name)

            # Check for single datetime column
            if timestamps is None:
                datetime_col = self.find_datetime_column(df)
                if datetime_col:
                    timestamps = self.parse_time_column(df[datetime_col])
                    Logger.log_message_static(f"{parser_name}: Found datetime column: {datetime_col}", Logger.DEBUG)

            # If still no timestamps, create synthetic ones
            if timestamps is None:
                timestamps = np.arange(len(df), dtype=np.float64)
                Logger.log_message_static(f"{parser_name}: Created synthetic timestamps", Logger.DEBUG)

        # Ensure timestamps match DataFrame length
        if len(timestamps) != len(df):
            min_len = min(len(timestamps), len(df))
            timestamps = timestamps[:min_len]
            df = df.iloc[:min_len]

        # Extract signal columns
        for col in df.columns:
            if col in ['Date', 'Time'] or (time_column is not None and col == time_column):
                continue

            signal_values = self._extract_signal_column(df[col], col, parser_name)
            if signal_values is not None:
                signals[col] = signal_values

        return timestamps, signals

    def find_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find potential time column in DataFrame.
        """
        time_keywords = ['time', 'timestamp', 'datetime', 'date', 'zeit', 'hora', 'temps']

        # Check column names for time-like names
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in time_keywords):
                if self.is_time_like_column(df[col]):
                    return col

        # Check for datetime columns by data type
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        # Check first column if it looks like time
        if len(df.columns) > 0:
            first_col = df.columns[0]
            if self.is_time_like_column(df[first_col]):
                return first_col

        return None

    def find_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find datetime column specifically (useful for Excel and other formats).
        """
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        return None

    def is_time_like_column(self, series: pd.Series) -> bool:
        """
        Check if a pandas Series contains time-like data.
        """
        # If it's already a datetime type, it's definitely time-like
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        # Sample a few values
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False

        time_like_count = 0
        for value in sample:
            if self.is_time_like_value(str(value)):
                time_like_count += 1

        # Consider it time-like if at least 70% of sampled values look like time
        return time_like_count >= 0.7 * len(sample)

    def is_time_like_value(self, value: str) -> bool:
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

        # Try pandas' flexible datetime parsing
        try:
            result = pd.to_datetime(value.strip(), errors='coerce')
            if pd.notna(result):
                return True
        except:
            pass

        # Check for numeric timestamp (Unix time or Excel serial date)
        try:
            num_val = float(value)
            # Excel serial dates start from 1900-01-01 (value ~= 1)
            # Unix timestamps are much larger
            if (1 <= num_val <= 100000) or (1000000000 <= num_val <= 4102444800):
                return True
        except:
            pass

        return False

    def parse_time_column(self, series: pd.Series) -> np.ndarray:
        """
        Parse time column to timestamps.
        """
        # If it's already datetime, convert directly
        if pd.api.types.is_datetime64_any_dtype(series):
            return series.astype(np.int64) / 1e9

        # Try pandas to_datetime first (handles many formats)
        try:
            timestamps = pd.to_datetime(series, errors='coerce')
            if not timestamps.isnull().all():
                return timestamps.astype(np.int64) / 1e9
        except Exception as e:
            Logger.log_message_static(f"BaseParser: Failed to parse time column with pandas: {str(e)}", Logger.DEBUG)

        # Try manual parsing with known formats
        for fmt in self.time_formats:
            try:
                timestamps = pd.to_datetime(series, format=fmt, errors='coerce')
                if not timestamps.isnull().all():
                    return timestamps.astype(np.int64) / 1e9
            except:
                continue

        # Try as numeric (Unix timestamps or Excel serial dates)
        try:
            numeric_times = pd.to_numeric(series, errors='coerce')
            if not numeric_times.isnull().all():
                return self._handle_numeric_timestamps(numeric_times)
        except Exception as e:
            Logger.log_message_static(f"BaseParser: Failed to parse numeric timestamps: {str(e)}", Logger.DEBUG)

        # Fallback: create synthetic timestamps
        Logger.log_message_static("BaseParser: Creating synthetic timestamps as fallback", Logger.DEBUG)
        return np.arange(len(series), dtype=np.float64)

    def parse_single_timestamp(self, timestamp_str: str) -> Optional[float]:
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

    def is_numeric_string(self, s: str) -> bool:
        """
        Check if string represents a number.
        """
        try:
            float(s.replace(',', '.'))
            return True
        except ValueError:
            return False

    def _try_combine_date_time_columns(self, df: pd.DataFrame, parser_name: str) -> Optional[np.ndarray]:
        """
        Try to combine separate Date and Time columns.
        """
        if 'Date' not in df.columns or 'Time' not in df.columns:
            return None

        try:
            date_series = df['Date']
            time_series = df['Time']

            # Convert both to string and combine
            if pd.api.types.is_datetime64_any_dtype(date_series):
                date_str = date_series.dt.strftime('%Y-%m-%d')
            else:
                date_str = date_series.astype(str)

            if pd.api.types.is_datetime64_any_dtype(time_series):
                time_str = time_series.dt.strftime('%H:%M:%S.%f')
            else:
                time_str = time_series.astype(str).str.replace(',', '.')

            combined_time = date_str + ' ' + time_str
            timestamps = pd.to_datetime(combined_time, errors='coerce').astype(np.int64) / 1e9
            timestamps = timestamps.to_numpy()

            # Drop rows with invalid timestamps
            valid_mask = ~np.isnan(timestamps)
            if valid_mask.any():
                timestamps = timestamps[valid_mask]
                Logger.log_message_static(f"{parser_name}: Combined Date and Time columns", Logger.DEBUG)
                return timestamps

        except Exception as e:
            Logger.log_message_static(f"{parser_name}: Failed to combine Date and Time: {str(e)}", Logger.DEBUG)

        return None

    def _extract_signal_column(self, series: pd.Series, col: str, parser_name: str) -> Optional[np.ndarray]:
        """
        Extract signal values from a DataFrame column.
        """
        try:
            # Skip completely empty columns
            if series.isnull().all():
                Logger.log_message_static(f"{parser_name}: Skipping empty column: {col}", Logger.DEBUG)
                return None

            # Handle boolean values
            if series.dtype == bool:
                Logger.log_message_static(f"{parser_name}: Detected boolean signal: {col}", Logger.DEBUG)
                return series.astype(np.float32)

            # Check for text boolean values
            if series.dtype == 'object':
                str_values = series.astype(str).str.upper()
                bool_indicators = ['TRUE', 'FALSE', '1', '0']
                if str_values.isin(bool_indicators).any():
                    bool_map = {'TRUE': 1.0, 'FALSE': 0.0, '1': 1.0, '0': 0.0, 'NAN': 0.0}
                    bool_values = str_values.map(bool_map).fillna(0.0)
                    Logger.log_message_static(f"{parser_name}: Detected text boolean signal: {col}", Logger.DEBUG)
                    return bool_values.astype(np.float32)

            # Handle numeric data
            if pd.api.types.is_numeric_dtype(series):
                # Already numeric
                numeric = series.fillna(method='ffill').fillna(0)
                Logger.log_message_static(f"{parser_name}: Extracted numeric signal: {col}", Logger.DEBUG)
                return numeric.astype(np.float32)
            else:
                # Try to convert to numeric
                if series.dtype == 'object':
                    # Handle European number format and other text formats
                    cleaned = series.astype(str).str.replace(',', '.', regex=False)
                    # Remove any non-numeric characters except for decimal points and signs
                    cleaned = cleaned.str.replace(r'[^\d\.\-\+]', '', regex=True)
                    numeric = pd.to_numeric(cleaned, errors='coerce')
                else:
                    numeric = pd.to_numeric(series, errors='coerce')

                if not numeric.isnull().all():
                    # Fill NaN values with forward fill, then with 0
                    numeric = numeric.fillna(method='ffill').fillna(0)
                    Logger.log_message_static(f"{parser_name}: Converted and extracted signal: {col}", Logger.DEBUG)
                    return numeric.astype(np.float32)
                else:
                    Logger.log_message_static(f"{parser_name}: Could not convert column to numeric: {col}",
                                              Logger.DEBUG)
                    return None

        except Exception as e:
            Logger.log_message_static(f"{parser_name}: Failed to process column {col}: {str(e)}", Logger.DEBUG)
            return None

    def _handle_numeric_timestamps(self, numeric_times: pd.Series) -> np.ndarray:
        """
        Handle numeric timestamps (Unix timestamps or Excel serial dates).
        """
        # Check if values look like Excel serial dates (1-100000) vs Unix timestamps
        sample_vals = numeric_times.dropna().head(10)
        if len(sample_vals) > 0:
            avg_val = sample_vals.mean()
            if 1 <= avg_val <= 100000:
                # Likely Excel serial dates - convert from Excel to Unix timestamp
                # Excel epoch is 1900-01-01, Unix epoch is 1970-01-01
                # Excel day 25569 = 1970-01-01
                unix_timestamps = (numeric_times - 25569) * 86400  # Convert to seconds
                return unix_timestamps.to_numpy(dtype=np.float64)
            else:
                # Likely already Unix timestamps
                return numeric_times.to_numpy(dtype=np.float64)

        # Fallback
        return numeric_times.to_numpy(dtype=np.float64)