"""
Graph export module for saving visualizations to various file formats.

This module provides functionality to export graph visualizations from PyQtGraph
widgets to common file formats (PNG, PDF, SVG). It supports:
- High-resolution exports with minimum Full HD (1920x1080) resolution
- Maintaining aspect ratio during scaling
- Multiple file format options with appropriate file extensions
- Both direct PyQtGraph exporters and manual rendering fallbacks
- Comprehensive error handling with user feedback

The module ensures exported graphs maintain visual quality and properly represent
all data points, labels, and styling from the original visualization.

Functions:
    export_graph: Main function to export a graph widget to PNG, PDF, or SVG
    export_graph_fallback: Alternative export method using PyQtGraph's exporters
"""

import os
from datetime import datetime
import pyqtgraph as pg

from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtCore import QSize, QRect

from utils.logger import Logger

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