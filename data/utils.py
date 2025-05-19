"""
Utils module

A combined module that provides:
1. Utilities for parsing CSV and proprietary recorder files
2. Functions for exporting graphs to various formats (PNG, PDF, SVG)

Optional dependencies:
- PySide6.QtPrintSupport: Required for PDF export
- PySide6.QtSvg: Required for SVG export
"""
import os
import re
import json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from PySide6.QtWidgets import QMessageBox, QFileDialog
from PySide6.QtCore import QSize, QRect
from PySide6.QtGui import QPixmap, QPainter

from utils.logger import Logger


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
        true_mask = values == "TRUE"
        false_mask = values == "FALSE"

        # If column contains TRUE/FALSE values, treat as boolean
        if true_mask.any() or false_mask.any():
            Logger.log_message_static(f"Detected boolean signal in column '{col}'", Logger.DEBUG)
            # Force a refresh of the plot
            signals[col] = values.to_numpy()  # Preserve TRUE/FALSE strings
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
        Logger.log_message_static("Missing start time in the file", Logger.ERROR)
        raise ValueError("Invalid recorder format: missing start time.")
    if not interval_sec:
        Logger.log_message_static("Missing interval in the file", Logger.ERROR)
        raise ValueError("Invalid recorder format: missing interval.")
    if not item_map:
        Logger.log_message_static("No signal items found in the file", Logger.ERROR)
        raise ValueError("Invalid recorder format: no signal items found.")
    if not data_lines:
        Logger.log_message_static("No data lines found in the file", Logger.ERROR)
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

    Logger.log_message_static(f"Successfully parsed {len(signals)} signals from Drive Debug file", Logger.INFO)
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
        # Check for boolean-like strings
        if arr.dtype == object:
            Logger.log_message_static("Signal has object dtype, checking for TRUE/FALSE values", Logger.DEBUG)
            unique_vals = set(str(val).upper() for val in arr if val is not None)
            if unique_vals.issubset({'TRUE', 'FALSE'}):
                Logger.log_message_static(f"Signal classified as digital with values: {unique_vals}", Logger.DEBUG)
                return True
            else:
                Logger.log_message_static(f"Signal contains non-boolean values: {unique_vals}", Logger.DEBUG)

        # Check for numeric values (e.g., 0/1)
        elif np.issubdtype(arr.dtype, np.number):
            unique_vals = np.unique(arr[~np.isnan(arr)])
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                Logger.log_message_static(f"Signal classified as digital with numeric values: {unique_vals}",
                                          Logger.INFO)
                return True
            else:
                Logger.log_message_static(f"Signal contains non-binary numeric values: {unique_vals}", Logger.DEBUG)

        Logger.log_message_static("Signal classified as analog", Logger.DEBUG)
        return False
    except Exception as e:
        Logger.log_message_static(f"Error determining if signal is digital: {str(e)}", Logger.ERROR)
        return False

def export_graph(plot_widget, parent_widget=None, export_full_window=False):
    """
    Exports a graph to a PNG, PDF, or SVG file.

    Ensures the output is at least Full HD (1920x1080) resolution, and will use
    the actual window size if it's larger.

    Args:
        plot_widget (pg.PlotWidget): Graph widget to export.
        parent_widget (QWidget, optional): Parent widget for dialogs.
        export_full_window (bool): If True, exports the entire application window.

    Returns:
        bool: True if export was successful, otherwise False.
    """
    Logger.log_message_static("Starting graph export", Logger.INFO)

    try:
        # Determine what to export
        if export_full_window and parent_widget:
            export_widget = parent_widget
            Logger.log_message_static("Exporting full application window", Logger.DEBUG)
        else:
            export_widget = plot_widget
            Logger.log_message_static("Exporting only plot widget", Logger.DEBUG)

        # Get current dimensions of what we're exporting
        current_width = export_widget.width()
        current_height = export_widget.height()
        Logger.log_message_static(f"Original widget dimensions: {current_width}x{current_height} pixels", Logger.DEBUG)

        # Ensure at least Full HD resolution (1920x1080)
        MIN_WIDTH = 1920
        MIN_HEIGHT = 1080

        # Calculate export dimensions
        export_width = max(current_width, MIN_WIDTH)
        export_height = max(current_height, MIN_HEIGHT)

        # Maintain aspect ratio when scaling up
        if current_width < MIN_WIDTH or current_height < MIN_HEIGHT:
            aspect_ratio = current_width / current_height

            # If one dimension needs scaling, adjust the other to maintain aspect ratio
            if current_width < MIN_WIDTH and current_height < MIN_HEIGHT:
                # Both dimensions need scaling, use the larger scale factor
                scale_x = MIN_WIDTH / current_width
                scale_y = MIN_HEIGHT / current_height
                if scale_x > scale_y:
                    export_width = MIN_WIDTH
                    export_height = int(MIN_WIDTH / aspect_ratio)
                else:
                    export_height = MIN_HEIGHT
                    export_width = int(MIN_HEIGHT * aspect_ratio)
            elif current_width < MIN_WIDTH:
                export_width = MIN_WIDTH
                export_height = int(MIN_WIDTH / aspect_ratio)
            elif current_height < MIN_HEIGHT:
                export_height = MIN_HEIGHT
                export_width = int(MIN_HEIGHT * aspect_ratio)

        Logger.log_message_static(f"Export dimensions set to {export_width}x{export_height} pixels", Logger.DEBUG)

        # Offer file selection
        file_filters = "PNG images (*.png);;PDF documents (*.pdf);;SVG vector format (*.svg)"
        Logger.log_message_static("Opening file save dialog", Logger.DEBUG)
        file_path, selected_filter = QFileDialog.getSaveFileName(
            parent_widget,
            "Export Graph",
            os.path.expanduser("~") + "/graph",
            file_filters
        )

        if not file_path:
            Logger.log_message_static("Export cancelled by user", Logger.INFO)
            return False

        # Determine format by selected filter
        if "PNG" in selected_filter:
            export_format = 'png'
            if not file_path.lower().endswith('.png'):
                file_path += '.png'
        elif "PDF" in selected_filter:
            export_format = 'pdf'
            if not file_path.lower().endswith('.pdf'):
                file_path += '.pdf'
        elif "SVG" in selected_filter:
            export_format = 'svg'
            if not file_path.lower().endswith('.svg'):
                file_path += '.svg'
        else:
            # Default to PNG
            export_format = 'png'
            if not file_path.lower().endswith('.png'):
                file_path += '.png'

        Logger.log_message_static(f"Exporting graph as {export_format.upper()} to {file_path}", Logger.DEBUG)

        # Use PyQtGraph's exporter for PNG/SVG to ensure we get the entire view
        if export_format in ['png', 'svg']:
            try:
                # Import specific exporter based on format
                if export_format == 'png':
                    from pyqtgraph.exporters import ImageExporter
                    Logger.log_message_static("Using PyQtGraph ImageExporter", Logger.DEBUG)
                    exporter = ImageExporter(plot_widget.plotItem)
                elif export_format == 'svg':
                    from pyqtgraph.exporters import SVGExporter
                    Logger.log_message_static("Using PyQtGraph SVGExporter", Logger.DEBUG)
                    exporter = SVGExporter(plot_widget.plotItem)

                # Set export dimensions
                original_size = exporter.getTargetRect()
                scaling_factor = min(export_width / original_size.width(), export_height / original_size.height())
                exporter.parameters()['width'] = int(original_size.width() * scaling_factor)

                # Export the file
                exporter.export(file_path)

                success = os.path.exists(file_path) and os.path.getsize(file_path) > 0
                Logger.log_message_static(f"Export result: {success}", Logger.DEBUG)

                if success:
                    file_size = os.path.getsize(file_path)
                    Logger.log_message_static(f"Export successful: {file_path} ({file_size} bytes)", Logger.INFO)
                    QMessageBox.information(
                        parent_widget,
                        "Export Successful",
                        f"Graph exported as {export_format.upper()} to:\n{file_path}"
                    )
                    return True
            except ImportError as e:
                Logger.log_message_static(f"PyQtGraph exporter not available: {str(e)}, falling back to manual export",
                                          Logger.WARNING)
                # Continue with manual export if PyQtGraph exporters aren't available
            except Exception as e:
                Logger.log_message_static(f"Error with PyQtGraph exporter: {str(e)}, falling back to manual export",
                                          Logger.WARNING)
                # Continue with manual export if PyQtGraph exporter fails

        # Create QPixmap for rendering at the specified resolution
        pixmap = QPixmap(export_width, export_height)
        pixmap.fill()  # Fill with transparent background

        # Create a painter for the pixmap
        painter = QPainter(pixmap)

        # Enable high quality rendering
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Scale if dimensions changed
        if export_width != current_width or export_height != current_height:
            scale_x = export_width / current_width
            scale_y = export_height / current_height
            Logger.log_message_static(f"Scaling by factors: x={scale_x:.2f}, y={scale_y:.2f}", Logger.DEBUG)
            painter.scale(scale_x, scale_y)

        # Render the widget to the pixmap
        export_widget.render(painter)
        painter.end()

        # Export according to chosen format
        success = False

        if export_format == 'png':
            Logger.log_message_static("Saving as PNG image (manual method)", Logger.DEBUG)
            success = pixmap.save(file_path, "PNG")
            Logger.log_message_static(f"PNG save result: {success}", Logger.DEBUG)

        elif export_format == 'pdf':
            Logger.log_message_static("Saving as PDF document", Logger.DEBUG)
            try:
                from PySide6.QtPrintSupport import QPrinter
                from PySide6.QtCore import QPageLayout

                # Create printer with high resolution
                printer = QPrinter(QPrinter.HighResolution)
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(file_path)

                # Use QPageLayout.Landscape instead of QPrinter.Landscape
                printer.setPageOrientation(QPageLayout.Landscape)

                # Set custom page size to match our export dimensions
                printer.setPageSize(QPrinter.Custom)
                printer.setPaperSize(QSize(export_width, export_height), QPrinter.DevicePixel)

                # No margins
                printer.setPageMargins(0, 0, 0, 0, QPrinter.Point)

                # Create PDF painter and draw the pixmap to it
                pdf_painter = QPainter()
                if pdf_painter.begin(printer):
                    Logger.log_message_static("PDF painter started successfully", Logger.DEBUG)
                    pdf_painter.drawPixmap(0, 0, pixmap)
                    pdf_painter.end()
                    success = os.path.exists(file_path) and os.path.getsize(file_path) > 0
                    Logger.log_message_static(f"PDF exists check: {os.path.exists(file_path)}", Logger.DEBUG)
                    if success:
                        Logger.log_message_static(f"PDF size: {os.path.getsize(file_path)} bytes", Logger.DEBUG)
                else:
                    Logger.log_message_static("Failed to begin PDF painter", Logger.ERROR)

            except ImportError as e:
                Logger.log_message_static(f"Failed to import QtPrintSupport: {str(e)}", Logger.ERROR)
                QMessageBox.critical(parent_widget, "Export Error",
                                     "PDF export requires QtPrintSupport module.\nPlease use PNG format instead.")
                return False
            except Exception as e:
                Logger.log_message_static(f"PDF export error: {str(e)}", Logger.ERROR)
                import traceback
                Logger.log_message_static(f"PDF export traceback: {traceback.format_exc()}", Logger.DEBUG)
                QMessageBox.critical(parent_widget, "PDF Export Error", str(e))
                return False

        elif export_format == 'svg':
            Logger.log_message_static("Saving as SVG vector graphic (manual method)", Logger.DEBUG)
            try:
                from PySide6.QtSvg import QSvgGenerator
                generator = QSvgGenerator()
                generator.setFileName(file_path)
                generator.setSize(QSize(export_width, export_height))
                generator.setViewBox(QRect(0, 0, export_width, export_height))
                generator.setTitle("Signal Graph Export")
                generator.setDescription(f"Exported at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                svg_painter = QPainter()
                if svg_painter.begin(generator):
                    svg_painter.drawPixmap(0, 0, pixmap)
                    svg_painter.end()
                    success = os.path.exists(file_path) and os.path.getsize(file_path) > 0
                else:
                    Logger.log_message_static("Failed to begin SVG painter", Logger.ERROR)

            except ImportError as e:
                Logger.log_message_static(f"Failed to import QtSvg: {str(e)}", Logger.ERROR)
                QMessageBox.critical(parent_widget, "Export Error",
                                     "SVG export requires QtSvg module.\nPlease use PNG format instead.")
                return False

        # Check if the export was successful
        if success:
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            Logger.log_message_static(f"Export successful: {file_path} ({file_size} bytes)", Logger.INFO)
            QMessageBox.information(
                parent_widget,
                "Export Successful",
                f"Graph exported as {export_format.upper()} to:\n{file_path}\n"
                f"Resolution: {export_width}x{export_height} pixels"
            )
            return True
        else:
            Logger.log_message_static(f"Export failed: {file_path}", Logger.ERROR)
            QMessageBox.critical(
                parent_widget,
                "Export Failed",
                f"Failed to export graph to {file_path}."
            )
            return False

    except Exception as e:
        Logger.log_message_static(f"Unexpected error during export: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Export error traceback: {traceback.format_exc()}", Logger.DEBUG)
        QMessageBox.critical(
            parent_widget,
            "Export Error",
            f"An error occurred during export:\n{str(e)}"
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