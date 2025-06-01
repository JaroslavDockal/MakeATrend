"""
Data loader module for handling signal file operations.

This module provides functionality for loading signal data from various file formats:
- CSV files (standard and custom formats)
- HDM5 files
- LVM (LabVIEW Measurement) files
- MAT (MATLAB) files
- SQL database exports
- TMDS format files
- Debug format files

Key features:
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
    save_project_state: Save current signals and analysis state to a file
    load_project_state: Load signals and analysis state from a saved project file
"""

import os
import numpy as np

from PySide6.QtWidgets import QFileDialog

from data.parser_master import ParserMaster
from utils.logger import Logger


def load_single_file():
    """
    Load a single signal file using a file dialog.
    Uses ParserMaster to determine which parser to use based on file content.

    Returns:
        dict: Dictionary of signals {name: (time_array, value_array)}.
    """
    # Get supported file extensions from the parser master
    parser = ParserMaster()
    extensions = parser.get_supported_extensions()
    filter_str = f"Data Files ({' '.join(['*' + ext for ext in extensions])})"

    Logger.log_message_static("Data-Loader: Opening file dialog to load a single file.", Logger.INFO)
    path, _ = QFileDialog.getOpenFileName(None, "Open Data File", "", filter_str)
    if not path:
        Logger.log_message_static("Data-Loader: No file selected. Operation canceled.", Logger.DEBUG)
        return {}

    try:
        signals, metadata = parser.parse_file(path)
        Logger.log_message_static(
            f"Data-Loader: Successfully loaded file '{os.path.basename(path)}' with {len(signals)} signals.",
            Logger.INFO
        )
        return signals
    except Exception as e:
        Logger.log_message_static(f"Data-Loader: Failed to load file '{os.path.basename(path)}'. Exception: {e}", Logger.ERROR)
        return {}


def load_multiple_files(file_paths=None):
    """
    Opens a dialog to select multiple files, merges signals with same names.
    Inserts NaNs between time segments to prevent false interpolation.
    Supports all file formats handled by the parsers in data/parsers.

    Args:
        file_paths (list, optional): List of file paths to load. If None, a file dialog is shown.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: signal name -> (time, values)
    """
    # Get supported file extensions from the parser master
    parser = ParserMaster()
    extensions = parser.get_supported_extensions()
    filter_str = f"Data Files ({' '.join(['*' + ext for ext in extensions])})"

    if file_paths is None:
        Logger.log_message_static("Data-Loader: Opening file dialog to load multiple files.", Logger.DEBUG)
        file_paths, _ = QFileDialog.getOpenFileNames(None, "Open Data Files", "", filter_str)
        if not file_paths:
            Logger.log_message_static("Data-Loader: No files selected. Operation canceled.", Logger.DEBUG)
            return {}

    all_signals = {}

    for path in file_paths:
        try:
            signals, metadata = parser.parse_file(path)
            Logger.log_message_static(
                f"Data-Loader: Successfully loaded file '{os.path.basename(path)}' with {len(signals)} signals.",
                Logger.DEBUG
            )

            for name, signal_data in signals.items():
                if name not in all_signals:
                    all_signals[name] = [signal_data]
                else:
                    Logger.log_message_static(f"Data-Loader: Signal '{name}' already exists. Appending new data.", Logger.WARNING)
                    all_signals[name].append(signal_data)
        except Exception as e:
            Logger.log_message_static(f"Data-Loader: Failed to load file '{os.path.basename(path)}'. Exception: {e}", Logger.ERROR)
            continue

    result = {}
    for name, parts in all_signals.items():
        # Sort signal parts by start time
        parts.sort(key=lambda x: x[0][0] if len(x[0]) > 0 else float('inf'))

        # Merge signal parts
        merged_time = np.concatenate([p[0] for p in parts])
        merged_values = np.concatenate([p[1] for p in parts])
        result[name] = (merged_time, merged_values)
        Logger.log_message_static(f"Data-Loader: Merged signal '{name}' with {len(merged_time)} points.", Logger.DEBUG)

    Logger.log_message_static(f"Data-Loader: Successfully loaded and merged {len(result)} signals from {len(file_paths)} files.", Logger.INFO)
    return result

