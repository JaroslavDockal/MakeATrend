"""
combined_utils.py

A combined module that provides:
1. Utilities for parsing CSV and proprietary recorder files
2. Functions for exporting graphs to various formats (PNG, PDF, SVG)

Optional dependencies:
- PySide6.QtPrintSupport: Required for PDF export
- PySide6.QtSvg: Required for SVG export
"""

import os
import re
import pandas as pd
import numpy as np
import json
from logger import Logger

from datetime import datetime, timedelta
from PySide6.QtWidgets import QMessageBox, QFileDialog
from PySide6.QtCore import QSize, QRect
from PySide6.QtGui import QPixmap, QPainter

# ===== CSV AND DATA PARSING FUNCTIONS =====

def parse_csv_or_recorder(path: str):
    """
    Parses either a standard CSV file or a proprietary recorder TXT file.

    Args:
        path (str): Path to the file.

    Returns:
        tuple:
            - np.ndarray: Array of timestamps (float, seconds since epoch).
            - dict[str, np.ndarray]: Dictionary of signal name -> signal values.
    """
    Logger.log_message_static(f"Starting to parse file: {os.path.basename(path)}", Logger.DEBUG)

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            Logger.log_message_static(f"Successfully read file: {os.path.basename(path)}", Logger.DEBUG)
    except UnicodeDecodeError:
        Logger.log_message_static(f"UTF-8 encoding failed, trying with latin1 for file: {os.path.basename(path)}", Logger.WARNING)
        try:
            with open(path, "r", encoding="latin1") as f:
                content = f.read()
        except Exception as e:
            Logger.log_message_static(f"Failed to read file {os.path.basename(path)}: {str(e)}", Logger.ERROR)
            raise ValueError(f"Cannot read file: {str(e)}")
    except Exception as e:
        Logger.log_message_static(f"Failed to read file {os.path.basename(path)}: {str(e)}", Logger.ERROR)
        raise ValueError(f"Cannot read file: {str(e)}")

    if ("RECORDER VALUES" in content or "TREND VALUES" in content) and "Interval:" in content:
        Logger.log_message_static(f"Detected Drive Debug format for file: {os.path.basename(path)}", Logger.INFO)
        return parse_recorder_format(content)
    else:
        Logger.log_message_static(f"Treating file as standard CSV: {os.path.basename(path)}", Logger.INFO)
        return parse_csv_file(path)


def parse_csv_file(path):
    """
    Parses a standard CSV file and returns timestamp array and signal data.
    Handles both numeric values and boolean string values ('TRUE'/'FALSE').

    Args:
        path (str): Path to the CSV file.

    Returns:
        tuple:
            - np.ndarray: Array of datetime timestamps as float seconds.
            - dict[str, np.ndarray]: Dictionary of signal name to signal values.

    Raises:
        ValueError: If timestamp cannot be parsed or signals are missing.
    """
    Logger.log_message_static(f"Parsing CSV file: {os.path.basename(path)}", Logger.DEBUG)

    try:
        Logger.log_message_static("Attempting to parse with semicolon delimiter", Logger.DEBUG)
        df = pd.read_csv(path, sep=';', engine='python')
    except pd.errors.ParserError as e:
        Logger.log_message_static(f"CSV parsing error with semicolon delimiter: {str(e)}", Logger.WARNING)
        try:
            Logger.log_message_static("Trying with comma delimiter", Logger.DEBUG)
            df = pd.read_csv(path, sep=',', engine='python')
        except pd.errors.ParserError as e:
            Logger.log_message_static(f"CSV parsing error with comma delimiter: {str(e)}", Logger.ERROR)
            raise ValueError(f"CSV parsing error: {e}")
    except FileNotFoundError:
        Logger.log_message_static(f"File not found: {path}", Logger.ERROR)
        raise ValueError(f"File not found: {path}")
    except Exception as e:
        Logger.log_message_static(f"Failed to read CSV: {str(e)}", Logger.ERROR)
        raise ValueError(f"Failed to read CSV: {e}")

    # === Parse timestamp ===
    Logger.log_message_static("Checking for Date and Time columns", Logger.DEBUG)
    if 'Date' in df.columns and 'Time' in df.columns:
        Logger.log_message_static("Found Date and Time columns, parsing timestamps", Logger.DEBUG)
        try:
            df['Timestamp'] = pd.to_datetime(
                df['Date'] + ' ' + df['Time'].str.replace(',', '.', regex=False),
                format='%Y-%m-%d %H:%M:%S.%f',
                errors='coerce'
            )
            Logger.log_message_static("Successfully parsed timestamps", Logger.DEBUG)
        except Exception as e:
            Logger.log_message_static(f"Error parsing timestamps: {str(e)}, trying flexible parsing", Logger.WARNING)
            try:
                df['Timestamp'] = pd.to_datetime(
                    df['Date'] + ' ' + df['Time'].str.replace(',', '.', regex=False),
                    errors='coerce'
                )
                Logger.log_message_static("Successfully parsed timestamps with flexible format", Logger.DEBUG)
            except Exception as e:
                Logger.log_message_static(f"Failed to parse timestamps: {str(e)}", Logger.ERROR)
                raise ValueError("Failed to parse timestamps")
    else:
        Logger.log_message_static("Missing 'Date' and 'Time' columns", Logger.ERROR)
        raise ValueError("Missing 'Date' and 'Time' columns.")

    # Drop rows with invalid timestamps
    invalid_count = df['Timestamp'].isna().sum()
    if invalid_count > 0:
        Logger.log_message_static(f"Found {invalid_count} rows with invalid timestamps, dropping them", Logger.WARNING)

    df.dropna(subset=['Timestamp'], inplace=True)
    timestamps = df['Timestamp'].astype(np.int64) / 1e9
    timestamps = timestamps.to_numpy()
    Logger.log_message_static(f"Created timestamp array with {len(timestamps)} points", Logger.DEBUG)

    signals = {}
    Logger.log_message_static("Starting to parse signal columns", Logger.DEBUG)
    for col in df.columns:
        if col in ('Date', 'Time', 'Timestamp'):
            continue

        # Direct approach for boolean values - check if any values are TRUE/FALSE
        values = df[col].astype(str).str.upper()
        true_mask = values == 'TRUE'
        false_mask = values == 'FALSE'

        # If column contains TRUE/FALSE values, treat as boolean
        if true_mask.any() or false_mask.any():
            Logger.log_message_static(f"Detected boolean signal in column '{col}'", Logger.DEBUG)
            bool_array = np.zeros(len(df), dtype=np.float32)
            bool_array[true_mask] = 1.0
            signals[col] = bool_array
        else:
            # Try numeric conversion for non-boolean columns
            try:
                Logger.log_message_static(f"Converting column '{col}' to numeric", Logger.DEBUG)
                cleaned = df[col].astype(str).str.replace(',', '.', regex=False)
                numeric = pd.to_numeric(cleaned, errors='coerce')
                if not numeric.isnull().all():
                    signals[col] = numeric.to_numpy(dtype=np.float32)
                    Logger.log_message_static(f"Successfully converted '{col}' to numeric", Logger.DEBUG)
                else:
                    Logger.log_message_static(f"Column '{col}' contains only non-numeric values", Logger.WARNING)
            except Exception as e:
                Logger.log_message_static(f"Failed to convert column '{col}' to numeric: {str(e)}", Logger.WARNING)
                # Skip columns that can't be parsed

    if not signals:
        Logger.log_message_static("No signals could be parsed from the file", Logger.ERROR)
        raise ValueError("No signals could be parsed.")

    Logger.log_message_static(f"Successfully parsed {len(signals)} signals from CSV file", Logger.INFO)
    return timestamps, signals


def parse_recorder_format(text):
    """
    Parses a text file in the special "Drive Debug" format.

    Args:
        text (str): Full content of the file.

    Returns:
        tuple:
            - np.ndarray: Array of timestamps (float, seconds since UNIX epoch).
            - dict[str, np.ndarray]: Dictionary of signal name -> signal values as float32 arrays.
    """
    Logger.log_message_static("Parsing Drive Debug format file", Logger.DEBUG)

    lines = text.strip().splitlines()
    item_map = {}
    start_time_str = None
    interval_sec = None
    data_lines = []

    Logger.log_message_static(f"File contains {len(lines)} lines", Logger.DEBUG)

    for line in lines:
        line = line.strip()
        if line.startswith("Item "):
            match = re.match(r"Item\s+(\d+)\s*=\s*(.+)", line)
            if match:
                item_id = int(match.group(1))
                item_name = match.group(2).strip()
                item_map[item_id] = item_name
                Logger.log_message_static(f"Found Item {item_id} = {item_name}", Logger.DEBUG)
        elif "Time of Interval" in line:
            start_time_str = line.split(":", 1)[1].strip()
            Logger.log_message_static(f"Found start time: {start_time_str}", Logger.DEBUG)
        elif "Interval:" in line:
            m = re.search(r"([\d.]+)\s*sec", line)
            if m:
                interval_sec = float(m.group(1))
                Logger.log_message_static(f"Found interval: {interval_sec} seconds", Logger.DEBUG)
        elif re.match(r"\s*\d+\s+", line):
            parts = re.split(r'\s+', line)
            try:
                idx = int(parts[0])
                values = [float(p.replace(',', '.')) for p in parts[1:]]
                data_lines.append([idx] + values)
            except ValueError as e:
                Logger.log_message_static(f"Skipping invalid data line: {line}", Logger.WARNING)
                continue

    if not start_time_str:
        Logger.log_message_static("Missing start time in recorder file", Logger.ERROR)
        raise ValueError("Invalid recorder format: missing start time.")
    if not interval_sec:
        Logger.log_message_static("Missing interval in recorder file", Logger.ERROR)
        raise ValueError("Invalid recorder format: missing interval.")
    if not item_map:
        Logger.log_message_static("No signal items found in recorder file", Logger.ERROR)
        raise ValueError("Invalid recorder format: no signal items found.")
    if not data_lines:
        Logger.log_message_static("No data lines found in recorder file", Logger.ERROR)
        raise ValueError("Invalid recorder format: no data found.")

    try:
        start_dt = datetime.strptime(start_time_str, "%m/%d/%y %H:%M:%S")
        Logger.log_message_static(f"Parsed start time: {start_dt}", Logger.DEBUG)
    except ValueError as e:
        Logger.log_message_static(f"Failed to parse start time '{start_time_str}': {str(e)}", Logger.ERROR)
        raise ValueError(f"Invalid start time format: {start_time_str}")

    Logger.log_message_static(f"Sorting {len(data_lines)} data lines", Logger.DEBUG)
    data_lines.sort(key=lambda row: row[0])

    Logger.log_message_static("Calculating timestamps", Logger.DEBUG)
    timestamps = [(start_dt - timedelta(seconds=row[0] * interval_sec)).timestamp() for row in data_lines]

    signals = {}
    Logger.log_message_static("Extracting signal values", Logger.DEBUG)
    for i in range(len(data_lines[0]) - 1):
        name = item_map.get(i + 1, f"Signal {i + 1}")
        signals[name] = [row[i + 1] for row in data_lines]

    for name in signals:
        signals[name] = np.array(signals[name], dtype=np.float32)
        Logger.log_message_static(f"Created signal '{name}' with {len(signals[name])} points", Logger.DEBUG)

    Logger.log_message_static(f"Successfully parsed {len(signals)} signals from recorder file", Logger.INFO)
    return np.array(timestamps, dtype=np.float64), signals

def find_nearest_index(array, value):
    """
    Find the index of the closest value in an array.

    Args:
        array (np.ndarray): Array to search in.
        value (float): Target value to find.

    Returns:
        int: Index of the closest value.
    """
    if array is None:
        Logger.log_message_static("Cannot find nearest index in None array", Logger.WARNING)
        return None

    if len(array) == 0:
        Logger.log_message_static("Cannot find nearest index in empty array", Logger.WARNING)
        return None

    try:
        idx = (np.abs(array - value)).argmin()
        Logger.log_message_static(f"Found nearest index {idx} with value {array[idx]:.6f} for target {value:.6f}", Logger.DEBUG)
        return idx
    except Exception as e:
        Logger.log_message_static(f"Error finding nearest index: {str(e)}", Logger.ERROR)
        return None


def is_digital_signal(arr):
    """
    Determines whether a signal is digital (boolean-like).

    A signal is considered digital if it only contains values like:
    - 'TRUE' / 'FALSE' (case-insensitive)
    - Not numeric 0/1, to avoid misclassification of analog signals

    Args:
        arr: NumPy array or tuple of (time_array, values_array)

    Returns:
        bool: True if signal appears to be digital/binary
    """
    Logger.log_message_static("Checking if signal is digital", Logger.DEBUG)

    # Handle case where argument is a tuple (time, values)
    if isinstance(arr, tuple) and len(arr) == 2:
        Logger.log_message_static("Processing (time, values) tuple, extracting values", Logger.DEBUG)
        _, arr = arr  # Extract just the values

    if arr is None:
        Logger.log_message_static("Cannot determine if None signal is digital", Logger.WARNING)
        return False

    if not isinstance(arr, np.ndarray):
        try:
            Logger.log_message_static("Converting input to numpy array", Logger.DEBUG)
            arr = np.array(arr)
        except Exception as e:
            Logger.log_message_static(f"Failed to convert input to array: {str(e)}", Logger.ERROR)
            return False

    if len(arr) == 0:
        Logger.log_message_static("Empty signal, cannot determine if digital", Logger.WARNING)
        return False

    try:
        if np.issubdtype(arr.dtype, np.number):
            # Find unique values excluding NaN
            unique_vals = np.unique(arr[~np.isnan(arr)])
            unique_count = len(unique_vals)

            # Consider digital if 2-3 unique values within integers 0-5
            if unique_count <= 3:
                all_integers = all(val == int(val) for val in unique_vals)
                all_in_range = all(0 <= val <= 5 for val in unique_vals)

                if all_integers and all_in_range:
                    Logger.log_message_static(f"Signal classified as digital with values: {unique_vals}", Logger.INFO)
                    return True
                else:
                    if not all_integers:
                        Logger.log_message_static("Signal contains non-integer values, not digital", Logger.DEBUG)
                    if not all_in_range:
                        Logger.log_message_static("Signal values outside 0-5 range, not digital", Logger.DEBUG)
            else:
                Logger.log_message_static(f"Signal has {unique_count} unique values, too many for digital", Logger.DEBUG)

        else:
            Logger.log_message_static(f"Signal has non-numeric dtype: {arr.dtype}, not digital", Logger.DEBUG)

        Logger.log_message_static("Signal classified as analog", Logger.DEBUG)
        return False
    except Exception as e:
        Logger.log_message_static(f"Error determining if signal is digital: {str(e)}", Logger.ERROR)
        return False

# ===== GRAPH EXPORT FUNCTIONS =====

def export_graph(plot_widget, parent_widget=None):
    """
    Exports a graph to a PNG, PDF, or SVG file.

    Implements robust export with error handling and user feedback.
    Unlike PyQtGraph Exporter, it works without external library dependencies.

    Args:
        plot_widget (pg.PlotWidget): Graph widget to export.
        parent_widget (QWidget, optional): Parent widget for dialogs.

    Returns:
        bool: True if export was successful, otherwise False.
    """
    Logger.log_message_static("Starting graph export", Logger.INFO)

    try:
        # Get current graph dimensions
        width = plot_widget.width()
        height = plot_widget.height()
        Logger.log_message_static(f"Graph dimensions: {width}x{height} pixels", Logger.DEBUG)

        # Offer file selection
        file_filters = "PNG images (*.png);;PDF documents (*.pdf);;SVG vector format (*.svg)"
        Logger.log_message_static("Opening file save dialog", Logger.DEBUG)
        file_path, selected_filter = QFileDialog.getSaveFileName(
            parent_widget, "Export Graph", "graph.png", file_filters
        )

        if not file_path:
            Logger.log_message_static("Export canceled by user", Logger.DEBUG)
            return False  # User canceled the dialog

        # Determine format by selected filter
        if "PNG" in selected_filter:
            if not file_path.lower().endswith('.png'):
                file_path += '.png'
            export_format = 'png'
        elif "PDF" in selected_filter:
            if not file_path.lower().endswith('.pdf'):
                file_path += '.pdf'
            export_format = 'pdf'
        elif "SVG" in selected_filter:
            if not file_path.lower().endswith('.svg'):
                file_path += '.svg'
            export_format = 'svg'
        else:
            # Default to PNG
            if not file_path.lower().endswith(('.png', '.pdf', '.svg')):
                file_path += '.png'
            export_format = 'png'

        Logger.log_message_static(f"Exporting graph as {export_format.upper()} to {os.path.basename(file_path)}", Logger.INFO)

        # Create QPixmap for rendering the graph
        Logger.log_message_static("Creating QPixmap for rendering", Logger.DEBUG)
        pixmap = QPixmap(width, height)
        pixmap.fill()  # Fill with transparent color

        # Render the graph to QPixmap
        Logger.log_message_static("Rendering graph to QPixmap", Logger.DEBUG)
        painter = QPainter(pixmap)
        plot_widget.render(painter)
        painter.end()

        # Export according to chosen format
        success = False

        if export_format == 'png':
            Logger.log_message_static("Saving as PNG image", Logger.DEBUG)
            success = pixmap.save(file_path, "PNG")
        elif export_format == 'pdf':
            # For PDF we need to use QPrinter
            try:
                Logger.log_message_static("Attempting PDF export using QPrinter", Logger.DEBUG)
                from PySide6.QtPrintSupport import QPrinter
                printer = QPrinter()
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(file_path)

                # PySide6 has different enum values in different versions
                # Try to set page size in a compatible way
                try:
                    # Try using the page size enum constants directly
                    Logger.log_message_static("Setting page size to A4", Logger.DEBUG)
                    printer.setPageSize(QPrinter.A4)
                except AttributeError:
                    try:
                        # For newer PySide6 versions that use QPageSize
                        Logger.log_message_static("Using QPageSize for newer PySide6 version", Logger.DEBUG)
                        from PySide6.QtGui import QPageSize
                        printer.setPageSize(QPageSize(QPageSize.A4))
                    except (AttributeError, ImportError) as e:
                        Logger.log_message_static(f"Could not set page size: {str(e)}, using default", Logger.WARNING)
                        # If all else fails, just use whatever default size is available
                        pass

                # Render to printer
                Logger.log_message_static("Rendering graph to PDF", Logger.DEBUG)
                painter = QPainter()
                try:
                    if painter.begin(printer):
                        plot_widget.render(painter)
                        success = True
                        Logger.log_message_static("PDF rendering completed successfully", Logger.DEBUG)
                    else:
                        Logger.log_message_static("Failed to start PDF painter", Logger.ERROR)
                finally:
                    painter.end()
            except ImportError as e:
                Logger.log_message_static(f"QPrinter import failed: {str(e)}", Logger.ERROR)
                QMessageBox.critical(
                    parent_widget,
                    "Export Error",
                    "For PDF export, the QtPrintSupport library is required but not available."
                )
                return False
        elif export_format == 'svg':
            try:
                Logger.log_message_static("Attempting SVG export", Logger.DEBUG)
                from PySide6.QtSvg import QSvgGenerator
                generator = QSvgGenerator()
                generator.setFileName(file_path)
                generator.setSize(QSize(width, height))
                generator.setViewBox(QRect(0, 0, width, height))

                # Render to SVG with proper resource management
                Logger.log_message_static("Rendering graph to SVG", Logger.DEBUG)
                painter = QPainter()
                try:
                    if painter.begin(generator):
                        plot_widget.render(painter)
                        success = True
                        Logger.log_message_static("SVG rendering completed successfully", Logger.DEBUG)
                    else:
                        Logger.log_message_static("Failed to start SVG painter", Logger.ERROR)
                finally:
                    painter.end()  # Make sure painter is always ended
            except ImportError as e:
                Logger.log_message_static(f"QSvgGenerator import failed: {str(e)}", Logger.ERROR)
                generator = None
                QMessageBox.critical(
                    parent_widget,
                    "Export Error",
                    "For SVG export, the QtSvg library is required but not available."
                )
                return False

        # Inform user about the result
        if success:
            Logger.log_message_static(f"Graph export successful: {os.path.basename(file_path)}", Logger.INFO)
            QMessageBox.information(
                parent_widget,
                "Export Complete",
                f"Graph was successfully exported to file:\n{file_path}"
            )
            return True
        else:
            Logger.log_message_static(f"Graph export failed: {os.path.basename(file_path)}", Logger.ERROR)
            QMessageBox.critical(
                parent_widget,
                "Export Error",
                f"Export to file {file_path} failed."
            )
            return False

    except Exception as e:
        Logger.log_message_static(f"Unexpected error during graph export: {str(e)}", Logger.ERROR)
        # Catch any errors and display them to the user
        QMessageBox.critical(
            parent_widget,
            "Export Error",
            f"An error occurred during graph export:\n{str(e)}"
        )
        return False


def export_graph_fallback(plot_widget, parent_widget=None):
    """
    Fallback export method using PyQtGraph Exporter if available.

    Args:
        plot_widget (pg.PlotWidget): Graph widget to export.
        parent_widget (QWidget, optional): Parent widget for dialogs.

    Returns:
        bool: True if export was successful, otherwise False.
    """
    Logger.log_message_static("Using fallback graph export method", Logger.DEBUG)

    try:
        # Try to import the exporter from PyQtGraph
        from pyqtgraph.exporters import ImageExporter
        Logger.log_message_static("Successfully imported PyQtGraph ImageExporter", Logger.DEBUG)

        Logger.log_message_static("Opening file save dialog", Logger.DEBUG)
        file_path, _ = QFileDialog.getSaveFileName(
            parent_widget, "Export Graph", "graph.png", "PNG images (*.png)"
        )

        if file_path:
            Logger.log_message_static(f"Exporting graph to {os.path.basename(file_path)} using PyQtGraph", Logger.INFO)
            exporter = ImageExporter(plot_widget.plotItem)
            exporter.export(file_path)
            Logger.log_message_static("PyQtGraph export successful", Logger.INFO)
            QMessageBox.information(
                parent_widget,
                "Export Complete",
                f"Graph was successfully exported to file:\n{file_path}"
            )
            return True
        else:
            Logger.log_message_static("Export canceled by user", Logger.DEBUG)
            return False
    except ImportError:
        Logger.log_message_static("PyQtGraph ImageExporter not available, using built-in export", Logger.WARNING)
        # If exporter is not available, try our own export
        return export_graph(plot_widget, parent_widget)
    except Exception as e:
        Logger.log_message_static(f"Error in fallback export: {str(e)}", Logger.ERROR)
        return False


def save_project_state(file_path, state):
    """
    Saves the project state to a JSON file.

    Args:
        file_path (str): Path to save the project state.
        state (dict): The project state to save.
    """
    Logger.log_message_static(f"Saving project state to {os.path.basename(file_path)}", Logger.INFO)

    try:
        Logger.log_message_static(f"Project state contains {len(state)} entries", Logger.DEBUG)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)
        Logger.log_message_static("Project state saved successfully", Logger.INFO)
    except Exception as e:
        Logger.log_message_static(f"Failed to save project state: {str(e)}", Logger.ERROR)
        raise IOError(f"Failed to save project state: {e}")

def load_project_state(file_path):
    """
    Loads the project state from a JSON file.

    Args:
        file_path (str): Path to the project state file.

    Returns:
        dict: The loaded project state.
    """
    Logger.log_message_static(f"Loading project state from {os.path.basename(file_path)}", Logger.INFO)

    if not os.path.exists(file_path):
        Logger.log_message_static(f"Project file not found: {file_path}", Logger.ERROR)
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        Logger.log_message_static(f"Project state loaded successfully with {len(state)} entries", Logger.INFO)
        return state
    except json.JSONDecodeError as e:
        Logger.log_message_static(f"Invalid JSON in project file: {str(e)}", Logger.ERROR)
        raise IOError(f"Failed to load project state: Invalid JSON format - {e}")
    except Exception as e:
        Logger.log_message_static(f"Failed to load project state: {str(e)}", Logger.ERROR)
        raise IOError(f"Failed to load project state: {e}")