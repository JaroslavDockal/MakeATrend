"""
Standard CSV parser implementation.

This parser handles CSV files with different delimiters (semicolon or comma)
and supports Date/Time columns for timestamps.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

from utils.logger import Logger


class StandardParser:
    """
    Parser for standard CSV files with Date and Time columns.
    Replicates the exact behavior of parse_csv_file from parser.py.
    """

    def __init__(self):
        """Initialize the standard parser."""
        pass

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this parser.

        Returns:
            List of supported file extensions
        """
        return ['.csv', '.txt', '.dat']

    def parse_file(self, file_path: str) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
        """
        Parse a standard CSV file with Date and Time columns.

        This method replicates the exact behavior of parse_csv_file from parser.py.

        Args:
            file_path: Path to the file to parse

        Returns:
            Tuple containing:
            - Dictionary mapping signal names to tuples of (time_array, values_array)
            - Dictionary of metadata about the file

        Raises:
            ValueError: If the file cannot be parsed
        """
        Logger.log_message_static(f"Parser-Standard: Parsing CSV file: {os.path.basename(file_path)}", Logger.DEBUG)

        try:
            Logger.log_message_static("Parser-Standard: Attempting to parse with semicolon delimiter", Logger.DEBUG)
            df = pd.read_csv(file_path, sep=';', engine='python')
        except pd.errors.ParserError as e:
            Logger.log_message_static(f"Parser-Standard: CSV parsing error with semicolon delimiter: {str(e)}", Logger.WARNING)
            try:
                Logger.log_message_static("Parser-Standard: Trying with comma delimiter", Logger.DEBUG)
                df = pd.read_csv(file_path, sep=',', engine='python')
            except pd.errors.ParserError as e:
                Logger.log_message_static(f"Parser-Standard: CSV parsing error with comma delimiter: {str(e)}", Logger.ERROR)
                raise ValueError(f"CSV parsing error: {e}")
        except FileNotFoundError:
            Logger.log_message_static(f"Parser-Standard: File not found: {file_path}", Logger.ERROR)
            raise ValueError(f"File not found: {file_path}")
        except Exception as e:
            Logger.log_message_static(f"Parser-Standard: Failed to read CSV: {str(e)}", Logger.ERROR)
            raise ValueError(f"Failed to read CSV: {e}")

        # === Parse timestamp ===
        Logger.log_message_static("Parser-Standard: Checking for Date and Time columns", Logger.DEBUG)
        if 'Date' in df.columns and 'Time' in df.columns:
            Logger.log_message_static("Parser-Standard: Found Date and Time columns, parsing timestamps", Logger.DEBUG)
            try:
                df['Timestamp'] = pd.to_datetime(
                    df['Date'] + ' ' + df['Time'].str.replace(',', '.', regex=False),
                    format='%Y-%m-%d %H:%M:%S.%f',
                    errors='coerce'
                )
                Logger.log_message_static("Parser-Standard: Successfully parsed timestamps", Logger.DEBUG)
            except Exception as e:
                Logger.log_message_static(f"Parser-Standard: Error parsing timestamps: {str(e)}, trying flexible parsing", Logger.WARNING)
                try:
                    df['Timestamp'] = pd.to_datetime(
                        df['Date'] + ' ' + df['Time'].str.replace(',', '.', regex=False),
                        errors='coerce'
                    )
                    Logger.log_message_static("Parser-Standard: Successfully parsed timestamps with flexible format", Logger.DEBUG)
                except Exception as e:
                    Logger.log_message_static(f"Parser-Standard: Failed to parse timestamps: {str(e)}", Logger.ERROR)
                    raise ValueError("Failed to parse timestamps")
        else:
            Logger.log_message_static("Parser-Standard: Missing 'Date' and 'Time' columns", Logger.ERROR)
            raise ValueError("Missing 'Date' and 'Time' columns.")

        # Drop rows with invalid timestamps
        invalid_count = df['Timestamp'].isna().sum()
        if invalid_count > 0:
            Logger.log_message_static(f"Parser-Standard: Found {invalid_count} rows with invalid timestamps, dropping them", Logger.WARNING)

        df.dropna(subset=['Timestamp'], inplace=True)
        timestamps = df['Timestamp'].astype(np.int64) / 1e9
        timestamps = timestamps.to_numpy()
        Logger.log_message_static(f"Parser-Standard: Created timestamp array with {len(timestamps)} points", Logger.DEBUG)

        signals = {}
        Logger.log_message_static("Parser-Standard: Starting to parse signal columns", Logger.DEBUG)
        for col in df.columns:
            if col in ('Date', 'Time', 'Timestamp'):
                continue

            # Direct approach for boolean values - check if any values are TRUE/FALSE
            values = df[col].astype(str).str.upper()
            true_mask = values == "TRUE"
            false_mask = values == "FALSE"

            # If column contains TRUE/FALSE values, treat as boolean
            if true_mask.any() or false_mask.any():
                Logger.log_message_static(f"Parser-Standard: Detected boolean signal in column '{col}'", Logger.DEBUG)
                # Force a refresh of the plot
                signals[col] = values.to_numpy()  # Preserve TRUE/FALSE strings
            else:
                # Try numeric conversion for non-boolean columns
                try:
                    Logger.log_message_static(f"Parser-Standard: Converting column '{col}' to numeric", Logger.DEBUG)
                    cleaned = df[col].astype(str).str.replace(',', '.', regex=False)
                    numeric = pd.to_numeric(cleaned, errors='coerce')
                    if not numeric.isnull().all():
                        signals[col] = numeric.to_numpy(dtype=np.float32)
                        Logger.log_message_static(f"Parser-Standard: Successfully converted '{col}' to numeric", Logger.DEBUG)
                    else:
                        Logger.log_message_static(f"Parser-Standard: Column '{col}' contains only non-numeric values", Logger.WARNING)
                except Exception as e:
                    Logger.log_message_static(f"Parser-Standard: Failed to convert column '{col}' to numeric: {str(e)}", Logger.WARNING)
                    # Skip columns that can't be parsed

        if not signals:
            Logger.log_message_static("Parser-Standard: No signals could be parsed from the file", Logger.ERROR)
            raise ValueError("No signals could be parsed.")

        # Convert format to match parser_master expectations
        result_signals = {}
        metadata = {"source_file": file_path, "parser": "StandardParser"}

        for name, values in signals.items():
            result_signals[name] = (timestamps, values)

        Logger.log_message_static(f"Parser-Standard: Successfully parsed {len(signals)} signals from CSV file", Logger.INFO)
        return result_signals, metadata

