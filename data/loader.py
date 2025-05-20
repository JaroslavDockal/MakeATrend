"""
Data loader module for handling signal file operations.

This module provides functionality for loading signal data from various file formats,
with primary support for CSV files. It includes:
- Single and multi-file loading through file dialogs
- Signal merging for data from multiple files
- Time-series alignment and processing
- Integration with parsing modules for different file formats
- Project state persistence (save/load functionality)

The loader acts as a bridge between raw data files and the application's
data structures, handling file selection, parsing, and initial data processing.

Functions:
    load_single_file: Load a single data file with user file dialog
    load_multiple_files: Load multiple data files and merge signals with same names
    parse_csv_or_recorder: Parse data from CSV or recorder format files
    get_parse_options: Display dialog for configuring CSV parsing options
    save_project_state: Save current signals and analysis state to a file
    load_project_state: Load signals and analysis state from a saved project file
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from PySide6.QtWidgets import QFileDialog, QDialog, QMessageBox

from .parser import parse_csv_or_recorder
from .signal_utils import is_digital_signal, find_nearest_index
from .csv_dialect import ParseOptions, ParseOptionsDialog, detect_csv_dialect
from .project import save_project_state, load_project_state
from utils.logger import Logger


def load_single_file():
    """
    Load a single signal file using a file dialog.

    Returns:
        dict: Dictionary of signals {name: (time_array, value_array)}.
    """
    Logger.log_message_static("Opening file dialog to load a single file.", Logger.INFO)
    path, _ = QFileDialog.getOpenFileName(None, "Open Data File", "", "Data Files (*.csv *.txt)")
    if not path:
        Logger.log_message_static("No file selected. Operation canceled.", Logger.DEBUG)
        return {}

    try:
        time_arr, signals = parse_csv_or_recorder(path)
        Logger.log_message_static(f"Successfully loaded file '{os.path.basename(path)}' with {len(signals)} signals.", Logger.INFO)
        return {name: (time_arr, values) for name, values in signals.items()}
    except Exception as e:
        Logger.log_message_static(f"Failed to load file '{os.path.basename(path)}'. Exception: {e}", Logger.ERROR)
        return {}

def load_multiple_files(file_paths=None):
    """
    Opens a dialog to select multiple files, merges signals with same names.
    Inserts NaNs between time segments to prevent false interpolation.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: signal name -> (time, values)
    """
    if file_paths is None:
        Logger.log_message_static("Opening file dialog to load multiple files.", Logger.DEBUG)
        file_paths, _ = QFileDialog.getOpenFileNames(None, "Open Data Files", "", "Data Files (*.csv *.txt)")
        if not file_paths:
            Logger.log_message_static("No files selected. Operation canceled.", Logger.DEBUG)
            return {}

    all_signals = {}

    for path in file_paths:
        try:
            time_arr, signals = parse_csv_or_recorder(path)
            Logger.log_message_static(f"Successfully loaded file '{os.path.basename(path)}' with {len(signals)} signals.", Logger.DEBUG)
            for name, values in signals.items():
                if name not in all_signals:
                    all_signals[name] = [(time_arr, values)]
                else:
                    Logger.log_message_static(f"Signal '{name}' already exists. Appending new data.", Logger.WARNING)
                    all_signals[name].append((time_arr, values))
        except Exception as e:
            Logger.log_message_static(f"Failed to load file '{os.path.basename(path)}'. Exception: {e}", Logger.ERROR)
            continue

    result = {}
    for name, parts in all_signals.items():
        parts.sort(key=lambda x: x[0][0])
        merged_time = np.concatenate([p[0] for p in parts])
        merged_values = np.concatenate([p[1] for p in parts])
        result[name] = (merged_time, merged_values)
        Logger.log_message_static(f"Merged signal '{name}' with {len(merged_time)} points.", Logger.DEBUG)

    Logger.log_message_static(f"Successfully loaded and merged {len(result)} signals from {len(file_paths)} files.", Logger.INFO)
    return result

