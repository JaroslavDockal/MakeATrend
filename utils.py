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
from datetime import datetime, timedelta

from PySide6.QtWidgets import QMessageBox, QFileDialog
from PySide6.QtCore import QSize, QRect
from PySide6.QtGui import QPixmap, QPainter

# Globální nastavení logování
_LOG_TO_CONSOLE = True
_LOG_TO_FILE = True
_LOG_FILE_PATH = "application.log"

def print_to_log(message, is_debug=False):
    """
    Log a message to console, log window, and optionally to a file.

    Args:
        message (str): The message to log.
        is_debug (bool): Whether the message is a debug-level message.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {'DEBUG' if is_debug else 'INFO'}: {message}"

    # Log to console
    if _LOG_TO_CONSOLE:
        print(formatted_message)

    # Log to file
    if _LOG_TO_FILE:
        with open(_LOG_FILE_PATH, "a") as log_file:
            log_file.write(formatted_message + "\n")

    # Log to GUI (pokud je okno aktivní)
    if hasattr(print_to_log, "log_window"):
        print_to_log.log_window.add_message(formatted_message, is_debug)

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
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if ("RECORDER VALUES" in content or "TREND VALUES" in content) and "Interval:" in content:
        return parse_recorder_format(content)
    else:
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
    try:
        df = pd.read_csv(path, sep=';', engine='python')
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV parsing error: {e}")
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    # === Parse timestamp ===
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Timestamp'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'].str.replace(',', '.', regex=False),
            format='%Y-%m-%d %H:%M:%S.%f',
            errors='coerce'
        )
    else:
        raise ValueError("Missing 'Date' and 'Time' columns.")

    df.dropna(subset=['Timestamp'], inplace=True)
    timestamps = df['Timestamp'].astype(np.int64) / 1e9
    timestamps = timestamps.to_numpy()

    signals = {}
    for col in df.columns:
        if col in ('Date', 'Time', 'Timestamp'):
            continue

        # Direct approach for boolean values - check if any values are TRUE/FALSE
        values = df[col].astype(str).str.upper()
        true_mask = values == 'TRUE'
        false_mask = values == 'FALSE'

        # If column contains TRUE/FALSE values, treat as boolean
        if true_mask.any() or false_mask.any():
            bool_array = np.zeros(len(df), dtype=np.float32)
            bool_array[true_mask] = 1.0
            signals[col] = bool_array
        else:
            # Try numeric conversion for non-boolean columns
            try:
                cleaned = df[col].astype(str).str.replace(',', '.', regex=False)
                numeric = pd.to_numeric(cleaned, errors='coerce')
                if not numeric.isnull().all():
                    signals[col] = numeric.to_numpy(dtype=np.float32)
            except Exception:
                # Skip columns that can't be parsed
                pass

    if not signals:
        raise ValueError("No signals could be parsed.")

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
    lines = text.strip().splitlines()
    item_map = {}
    start_time_str = None
    interval_sec = None
    data_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith("Item "):
            match = re.match(r"Item\s+(\d+)\s*=\s*(.+)", line)
            if match:
                item_map[int(match.group(1))] = match.group(2).strip()
        elif "Time of Interval" in line:
            start_time_str = line.split(":", 1)[1].strip()
        elif "Interval:" in line:
            m = re.search(r"([\d.]+)\s*sec", line)
            if m:
                interval_sec = float(m.group(1))
        elif re.match(r"\s*\d+\s+", line):
            parts = re.split(r'\s+', line)
            try:
                idx = int(parts[0])
                values = [float(p.replace(',', '.')) for p in parts[1:]]
                data_lines.append([idx] + values)
            except ValueError:
                continue

    if not (start_time_str and interval_sec and item_map and data_lines):
        raise ValueError("Invalid recorder format.")

    start_dt = datetime.strptime(start_time_str, "%m/%d/%y %H:%M:%S")
    data_lines.sort(key=lambda row: row[0])
    timestamps = [(start_dt - timedelta(seconds=row[0] * interval_sec)).timestamp() for row in data_lines]

    signals = {}
    for i in range(len(data_lines[0]) - 1):
        name = item_map.get(i + 1, f"Signal {i + 1}")
        signals[name] = [row[i + 1] for row in data_lines]

    for name in signals:
        signals[name] = np.array(signals[name], dtype=np.float32)

    return np.array(timestamps, dtype=np.float64), signals


def find_nearest_index(array, value):
    """
    Finds the index of the closest value in an array.

    Args:
        array (np.ndarray): The array to search.
        value (float): The value to find the closest match to.

    Returns:
        int: Index of the closest value in the array.
    """
    if array.size == 0:
        raise ValueError("Cannot find nearest index in an empty array")
    return (np.abs(array - value)).argmin()


def is_digital_signal(arr):
    #TODO Zajistit že signál s nulovou hodnotou nebude klasifikován jako digital
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

    # Handle case where argument is a tuple (time, values)
    if isinstance(arr, tuple) and len(arr) == 2:
        _, arr = arr  # Extract just the values

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if np.issubdtype(arr.dtype, np.number):
        # Find unique values excluding NaN
        unique_vals = np.unique(arr[~np.isnan(arr)])
        # Consider digital if 2-3 unique values within integers 0-5
        if len(unique_vals) <= 3 and all(val.is_integer() for val in unique_vals) and all(
                0 <= val <= 5 for val in unique_vals):
            return True
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
    try:
        # Get current graph dimensions
        width = plot_widget.width()
        height = plot_widget.height()

        # Offer file selection
        file_filters = "PNG images (*.png);;PDF documents (*.pdf);;SVG vector format (*.svg)"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            parent_widget, "Export Graph", "graph.png", file_filters
        )

        if not file_path:
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

        # Create QPixmap for rendering the graph
        pixmap = QPixmap(width, height)
        pixmap.fill()  # Fill with transparent color

        # Render the graph to QPixmap
        painter = QPainter(pixmap)
        plot_widget.render(painter)
        painter.end()

        # Export according to chosen format
        success = False

        if export_format == 'png':
            success = pixmap.save(file_path, "PNG")
        elif export_format == 'pdf':
            # For PDF we need to use QPrinter
            try:
                from PySide6.QtPrintSupport import QPrinter
                printer = QPrinter()
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(file_path)

                # PySide6 has different enum values in different versions
                # Try to set page size in a compatible way
                try:
                    # Try using the page size enum constants directly
                    printer.setPageSize(QPrinter.A4)
                except AttributeError:
                    try:
                        # For newer PySide6 versions that use QPageSize
                        from PySide6.QtGui import QPageSize
                        printer.setPageSize(QPageSize(QPageSize.A4))
                    except (AttributeError, ImportError):
                        # If all else fails, just use whatever default size is available
                        pass

                # Render to printer
                painter = QPainter()
                try:
                    if painter.begin(printer):
                        plot_widget.render(painter)
                        success = True
                finally:
                    painter.end()
            except ImportError:
                QMessageBox.critical(
                    parent_widget,
                    "Export Error",
                    "For PDF export, the QtPrintSupport library is required but not available."
                )
                return False
        elif export_format == 'svg':
            try:
                from PySide6.QtSvg import QSvgGenerator
                generator = QSvgGenerator()
                generator.setFileName(file_path)
                generator.setSize(QSize(width, height))
                generator.setViewBox(QRect(0, 0, width, height))

                # Render to SVG with proper resource management
                painter = QPainter()
                try:
                    if painter.begin(generator):
                        plot_widget.render(painter)
                        success = True
                finally:
                    painter.end()  # Make sure painter is always ended
            except ImportError:
                generator = None
                QMessageBox.critical(
                    parent_widget,
                    "Export Error",
                    "For SVG export, the QtSvg library is required but not available."
                )
                return False

        # Inform user about the result
        if success:
            QMessageBox.information(
                parent_widget,
                "Export Complete",
                f"Graph was successfully exported to file:\n{file_path}"
            )
            return True
        else:
            QMessageBox.critical(
                parent_widget,
                "Export Error",
                f"Export to file {file_path} failed."
            )
            return False

    except Exception as e:
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
    try:
        # Try to import the exporter from PyQtGraph
        from pyqtgraph.exporters import ImageExporter

        file_path, _ = QFileDialog.getSaveFileName(
            parent_widget, "Export Graph", "graph.png", "PNG images (*.png)"
        )

        if file_path:
            exporter = ImageExporter(plot_widget.plotItem)
            exporter.export(file_path)
            QMessageBox.information(
                parent_widget,
                "Export Complete",
                f"Graph was successfully exported to file:\n{file_path}"
            )
            return True
        return False
    except ImportError:
        # If exporter is not available, try our own export
        return export_graph(plot_widget, parent_widget)


def save_project_state(file_path, state):
    """
    Saves the project state to a JSON file.

    Args:
        file_path (str): Path to save the project state.
        state (dict): The project state to save.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        raise IOError(f"Failed to save project state: {e}")

def load_project_state(file_path):
    """
    Loads the project state from a JSON file.

    Args:
        file_path (str): Path to the project state file.

    Returns:
        dict: The loaded project state.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise IOError(f"Failed to load project state: {e}")