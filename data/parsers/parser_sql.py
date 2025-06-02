"""
SQL parser implementation.

This parser handles SQLite database files and returns data
in the same format as the StandardParser for compatibility.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from datetime import datetime
import sqlite3

from utils.logger import Logger


class SQLParser:
    """
    Parser for SQLite database files.
    Returns data in the same format as StandardParser for compatibility.
    """

    def __init__(self):
        """Initialize the SQL parser."""
        pass

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this parser.

        Returns:
            List of supported file extensions
        """
        return ['.db', '.sqlite', '.sqlite3', '.sql']

    def parse_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Parse a SQLite database file and extract timestamps and signals.

        Args:
            file_path: Path to the SQLite file to parse

        Returns:
            Tuple containing:
            - np.ndarray: Array of timestamps (float, seconds since epoch)
            - Dict[str, np.ndarray]: Dictionary of signal name -> signal values

        Raises:
            ValueError: If the file cannot be parsed
        """
        Logger.log_message_static(f"Parser-SQL: Parsing SQLite file: {os.path.basename(file_path)}", Logger.DEBUG)

        if not os.path.exists(file_path):
            Logger.log_message_static(f"Parser-SQL: File not found: {file_path}", Logger.ERROR)
            raise ValueError(f"File not found: {file_path}")

        try:
            # Connect to SQLite database
            Logger.log_message_static("Parser-SQL: Connecting to SQLite database", Logger.DEBUG)
            conn = sqlite3.connect(file_path)

            # Get list of tables
            tables = self._get_table_list(conn)

            if not tables:
                Logger.log_message_static("Parser-SQL: No tables found in database", Logger.ERROR)
                conn.close()
                raise ValueError("No tables found in database")

            Logger.log_message_static(f"Parser-SQL: Found {len(tables)} table(s): {tables}", Logger.DEBUG)

            # Extract data from tables
            timestamps, signals = self._extract_data_from_tables(conn, tables)

            conn.close()

        except sqlite3.Error as e:
            Logger.log_message_static(f"Parser-SQL: SQLite error: {str(e)}", Logger.ERROR)
            raise ValueError(f"SQLite error: {e}")
        except Exception as e:
            Logger.log_message_static(f"Parser-SQL: Failed to parse SQLite file: {str(e)}", Logger.ERROR)
            raise ValueError(f"Failed to parse SQLite file: {e}")

        if not signals:
            Logger.log_message_static("Parser-SQL: No signals could be parsed from the database", Logger.ERROR)
            raise ValueError("No signals could be parsed from the database")

        Logger.log_message_static(f"Parser-SQL: Successfully parsed {len(signals)} signals from SQLite file",
                                  Logger.INFO)
        return timestamps, signals

    def _get_table_list(self, conn: sqlite3.Connection) -> List[str]:
        """
        Get list of tables in the database.

        Args:
            conn: SQLite connection

        Returns:
            List of table names
        """
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]
            return tables
        except Exception as e:
            Logger.log_message_static(f"Parser-SQL: Error getting table list: {str(e)}", Logger.WARNING)
            return []

    def _extract_data_from_tables(self, conn: sqlite3.Connection, tables: List[str]) -> Tuple[
        np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract data from database tables.

        Args:
            conn: SQLite connection
            tables: List of table names

        Returns:
            Tuple of timestamps and signals
        """
        timestamps = None
        signals = {}

        for table_name in tables:
            Logger.log_message_static(f"Parser-SQL: Processing table '{table_name}'", Logger.DEBUG)

            try:
                # Get table info
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

                if df.empty:
                    Logger.log_message_static(f"Parser-SQL: Table '{table_name}' is empty", Logger.WARNING)
                    continue

                Logger.log_message_static(
                    f"Parser-SQL: Table '{table_name}' has {len(df)} rows and {len(df.columns)} columns", Logger.DEBUG)

                # Process table data
                table_timestamps, table_signals = self._process_table_data(df, table_name)

                # Use the first valid timestamp set found
                if timestamps is None and table_timestamps is not None:
                    timestamps = table_timestamps
                    Logger.log_message_static(f"Parser-SQL: Using timestamps from table '{table_name}'", Logger.DEBUG)

                # Add signals with table prefix
                for col_name, signal_data in table_signals.items():
                    full_name = f"{table_name}.{col_name}"
                    signals[full_name] = signal_data

            except Exception as e:
                Logger.log_message_static(f"Parser-SQL: Error processing table '{table_name}': {str(e)}",
                                          Logger.WARNING)
                continue

        # If no timestamps found, generate synthetic ones
        if timestamps is None and signals:
            Logger.log_message_static("Parser-SQL: No timestamps found, generating synthetic timestamps",
                                      Logger.WARNING)
            timestamps = self._generate_synthetic_timestamps(signals)

        return timestamps, signals

    def _process_table_data(self, df: pd.DataFrame, table_name: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Process data from a single table.

        Args:
            df: DataFrame with table data
            table_name: Name of the table

        Returns:
            Tuple of timestamps and signals from this table
        """
        timestamps = None
        signals = {}

        # Look for timestamp columns (similar to StandardParser)
        timestamp_cols = self._find_timestamp_columns(df)

        if timestamp_cols:
            timestamps = self._extract_timestamps_from_columns(df, timestamp_cols)

        # Process other columns as signals
        for col in df.columns:
            if col in timestamp_cols:
                continue  # Skip timestamp columns

            Logger.log_message_static(f"Parser-SQL: Processing column '{col}' from table '{table_name}'", Logger.DEBUG)

            # Process signal data
            processed_signal = self._process_signal_column(df[col], col)
            if processed_signal is not None:
                signals[col] = processed_signal

        return timestamps, signals

    def _find_timestamp_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Find columns that could contain timestamp data.

        Args:
            df: DataFrame to analyze

        Returns:
            List of column names that could be timestamps
        """
        timestamp_cols = []

        # Check for Date/Time columns like StandardParser
        if 'Date' in df.columns and 'Time' in df.columns:
            timestamp_cols = ['Date', 'Time']
            Logger.log_message_static("Parser-SQL: Found Date and Time columns", Logger.DEBUG)
        else:
            # Look for other timestamp-like columns
            time_keywords = ['timestamp', 'datetime', 'date', 'time', 'created_at', 'updated_at']

            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in time_keywords):
                    # Check if column actually contains timestamp-like data
                    if self._column_looks_like_timestamps(df[col]):
                        timestamp_cols.append(col)
                        Logger.log_message_static(f"Parser-SQL: Found timestamp column '{col}'", Logger.DEBUG)

        return timestamp_cols

    def _column_looks_like_timestamps(self, col_data: pd.Series) -> bool:
        """
        Check if a column looks like it contains timestamps.

        Args:
            col_data: Column data

        Returns:
            True if column looks like timestamps
        """
        # Check for numeric timestamps
        if pd.api.types.is_numeric_dtype(col_data):
            min_val, max_val = col_data.min(), col_data.max()
            # Unix timestamp range check
            if 0 < min_val < 4102444800 and max_val > min_val:
                return True

        # Check for string timestamps
        if pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            # Try to parse a few values
            sample_values = col_data.dropna().head(5)
            parsed_count = 0

            for val in sample_values:
                try:
                    pd.to_datetime(val)
                    parsed_count += 1
                except:
                    pass

            # If most values can be parsed as dates, consider it a timestamp column
            if parsed_count >= len(sample_values) * 0.8:
                return True

        return False

    def _extract_timestamps_from_columns(self, df: pd.DataFrame, timestamp_cols: List[str]) -> np.ndarray:
        """
        Extract timestamps from identified timestamp columns.

        Args:
            df: DataFrame
            timestamp_cols: List of timestamp column names

        Returns:
            Array of timestamps or None if extraction fails
        """
        try:
            if 'Date' in timestamp_cols and 'Time' in timestamp_cols:
                # Handle Date/Time columns like StandardParser
                Logger.log_message_static("Parser-SQL: Parsing Date and Time columns", Logger.DEBUG)

                df_copy = df.copy()
                df_copy['Timestamp'] = pd.to_datetime(
                    df_copy['Date'].astype(str) + ' ' + df_copy['Time'].astype(str).str.replace(',', '.', regex=False),
                    format='%Y-%m-%d %H:%M:%S.%f',
                    errors='coerce'
                )

                # Try without milliseconds if parsing failed
                if df_copy['Timestamp'].isna().sum() > 0:
                    df_copy['Timestamp'] = pd.to_datetime(
                        df_copy['Date'].astype(str) + ' ' + df_copy['Time'].astype(str).str.replace(',', '.',
                                                                                                    regex=False),
                        format='%Y-%m-%d %H:%M:%S',
                        errors='coerce'
                    )

                # Try flexible parsing as last resort
                if df_copy['Timestamp'].isna().sum() > 0:
                    df_copy['Timestamp'] = pd.to_datetime(
                        df_copy['Date'].astype(str) + ' ' + df_copy['Time'].astype(str).str.replace(',', '.',
                                                                                                    regex=False),
                        errors='coerce'
                    )

                # Drop rows with invalid timestamps
                valid_timestamps = df_copy['Timestamp'].dropna()
                if len(valid_timestamps) > 0:
                    timestamps = valid_timestamps.astype(np.int64) / 1e9
                    return timestamps.to_numpy()

            else:
                # Handle single timestamp column
                timestamp_col = timestamp_cols[0]
                Logger.log_message_static(f"Parser-SQL: Parsing timestamp column '{timestamp_col}'", Logger.DEBUG)

                col_data = df[timestamp_col]

                if pd.api.types.is_numeric_dtype(col_data):
                    # Numeric timestamps
                    timestamps = col_data.dropna().astype(np.float64).to_numpy()

                    # Check if it looks like Unix timestamps
                    min_val = np.min(timestamps)
                    if min_val > 1e9:
                        return timestamps
                    else:
                        # Relative timestamps - convert to absolute
                        current_time = datetime.now().timestamp()
                        return current_time + timestamps

                else:
                    # String timestamps
                    parsed_timestamps = pd.to_datetime(col_data, errors='coerce')
                    valid_timestamps = parsed_timestamps.dropna()

                    if len(valid_timestamps) > 0:
                        timestamps = valid_timestamps.astype(np.int64) / 1e9
                        return timestamps.to_numpy()

            return None

        except Exception as e:
            Logger.log_message_static(f"Parser-SQL: Error extracting timestamps: {str(e)}", Logger.WARNING)
            return None

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
            # Skip columns with too many null values
            if col_data.isna().sum() / len(col_data) > 0.8:
                Logger.log_message_static(f"Parser-SQL: Column '{col_name}' has too many null values, skipping",
                                          Logger.WARNING)
                return None

            # Convert to string and check for boolean values
            str_data = col_data.astype(str).str.strip().str.upper()

            # Check for boolean values
            true_mask = str_data.isin(['TRUE', '1', 'YES', 'Y'])
            false_mask = str_data.isin(['FALSE', '0', 'NO', 'N'])

            if true_mask.any() or false_mask.any():
                Logger.log_message_static(f"Parser-SQL: Detected boolean signal in column '{col_name}'", Logger.DEBUG)
                # Convert to standard TRUE/FALSE format
                result = str_data.copy()
                result[true_mask] = 'TRUE'
                result[false_mask] = 'FALSE'
                result[~(true_mask | false_mask)] = 'FALSE'  # Default for other values
                return result.to_numpy()

            # Try numeric conversion
            try:
                # Handle different numeric representations
                if pd.api.types.is_numeric_dtype(col_data):
                    numeric_data = col_data.fillna(0).astype(np.float32)
                else:
                    # String to numeric conversion
                    cleaned_data = col_data.astype(str).str.replace(',', '.', regex=False)
                    cleaned_data = cleaned_data.replace(['nan', 'NaN', 'NULL', 'null', ''], np.nan)
                    numeric_data = pd.to_numeric(cleaned_data, errors='coerce')

                    # Check if conversion was successful for most values
                    valid_ratio = (~numeric_data.isna()).sum() / len(numeric_data)
                    if valid_ratio > 0.5:  # At least 50% valid numeric values
                        numeric_data = numeric_data.fillna(0).astype(np.float32)
                    else:
                        Logger.log_message_static(f"Parser-SQL: Column '{col_name}' contains mostly non-numeric values",
                                                  Logger.WARNING)
                        return None

                Logger.log_message_static(f"Parser-SQL: Successfully converted column '{col_name}' to numeric",
                                          Logger.DEBUG)
                return numeric_data.to_numpy()

            except Exception as e:
                Logger.log_message_static(f"Parser-SQL: Failed to convert column '{col_name}' to numeric: {str(e)}",
                                          Logger.WARNING)
                return None

        except Exception as e:
            Logger.log_message_static(f"Parser-SQL: Error processing column '{col_name}': {str(e)}", Logger.WARNING)
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

        Logger.log_message_static(f"Parser-SQL: Generated {len(timestamps)} synthetic timestamps", Logger.DEBUG)
        return timestamps