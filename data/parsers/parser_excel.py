"""
Excel parser implementation.

This parser handles Excel files (XLS, XLSX) and OpenDocument Spreadsheets (ODS)
with automatic sheet detection and supports Date/Time columns for timestamps.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional

from utils.logger import Logger
from data.parsers.parser_base import BaseParser


class ExcelParser(BaseParser):
    """
    Parser for Excel files (XLS, XLSX) and OpenDocument Spreadsheets (ODS).
    Supports multiple sheets and automatic format detection.
    Inherits common functionality from BaseParser.
    """

    def __init__(self):
        """Initialize the Excel parser."""
        super().__init__()

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this parser.

        Returns:
            List of supported file extensions
        """
        return ['.xlsx', '.xls', '.ods']

    def parse_file(self, file_path: str) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
        """
        Parse an Excel file with automatic sheet and format detection.

        Args:
            file_path: Path to the Excel file to parse

        Returns:
            Tuple containing:
            - Dictionary mapping signal names to tuples of (time_array, values_array)
            - Dictionary of metadata about the file

        Raises:
            ValueError: If the file cannot be parsed
        """
        Logger.log_message_static(f"Parser-Excel: Parsing Excel file: {os.path.basename(file_path)}", Logger.DEBUG)

        try:
            if file_path.lower().endswith('.ods'):
                engine = 'odf'
            else:
                engine = 'openpyxl' if file_path.lower().endswith('.xlsx') else 'xlrd'

            try:
                xl_file = pd.ExcelFile(file_path, engine=engine)
                sheet_names = xl_file.sheet_names
                Logger.log_message_static(f"Parser-Excel: Found {len(sheet_names)} sheets: {sheet_names}", Logger.DEBUG)
            except Exception as e:
                Logger.log_message_static(f"Parser-Excel: Failed to read Excel file info: {str(e)}", Logger.ERROR)
                raise ValueError(f"Failed to read Excel file: {e}")

            best_sheet_data = None
            best_sheet_name = None
            max_signals = 0

            for sheet_name in sheet_names:
                try:
                    Logger.log_message_static(f"Parser-Excel: Trying to parse sheet: {sheet_name}", Logger.DEBUG)
                    df = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine)
                    Logger.log_message_static(f"Parser-Excel: Sheet head preview:\n{df.head()}", Logger.DEBUG)
                    Logger.log_message_static(f"Parser-Excel: Sheet dtypes:\n{df.dtypes}", Logger.DEBUG)

                    if df.empty:
                        Logger.log_message_static(f"Parser-Excel: Sheet '{sheet_name}' is empty", Logger.DEBUG)
                        continue

                    # Drop completely empty columns
                    df.dropna(axis=1, how='all', inplace=True)
                    # Drop rows where all values are NaN
                    df.dropna(axis=0, how='all', inplace=True)

                    if df.empty or df.shape[1] < 2:
                        Logger.log_message_static(f"Parser-Excel: Sheet '{sheet_name}' has insufficient valid data", Logger.DEBUG)
                        continue

                    timestamps, signals = self.extract_from_dataframe(df, "ExcelParser")

                    Logger.log_message_static(f"Parser-Excel: Sheet '{sheet_name}' yielded {len(signals)} signal(s)", Logger.DEBUG)

                    if signals and len(signals) > max_signals:
                        max_signals = len(signals)
                        best_sheet_data = (timestamps, signals)
                        best_sheet_name = sheet_name
                        Logger.log_message_static(
                            f"Parser-Excel: Sheet '{sheet_name}' has {len(signals)} signals (new best)", Logger.DEBUG)

                except Exception as e:
                    Logger.log_message_static(f"Parser-Excel: Failed to parse sheet '{sheet_name}': {str(e)}", Logger.DEBUG)
                    continue

            if best_sheet_data is None:
                raise ValueError("No usable data found in any sheet")

            timestamps, signals = best_sheet_data

            result_signals = {}
            metadata = {
                "source_file": file_path,
                "parser": "ExcelParser",
                "sheet_name": best_sheet_name,
                "total_sheets": len(sheet_names)
            }

            for i, (name, values) in enumerate(signals.items()):
                if timestamps is None or len(timestamps) != len(values):
                    dt = 1.0
                    offset = i * 0.1
                    synthetic_time = np.arange(len(values), dtype=np.float64) * dt + offset
                    Logger.log_message_static(f"Parser-Excel: Using synthetic time for signal '{name}'", Logger.DEBUG)
                    result_signals[name] = (synthetic_time, values)
                else:
                    result_signals[name] = (timestamps, values)

            Logger.log_message_static(
                f"Parser-Excel: Successfully parsed {len(signals)} signals from sheet '{best_sheet_name}'", Logger.INFO)
            return result_signals, metadata

        except Exception as e:
            error_msg = f"ExcelParser failed to parse file {os.path.basename(file_path)}: {str(e)}"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)
