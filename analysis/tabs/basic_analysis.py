"""
Basic Analysis Tab - Contains basic signal analysis tools like statistics, FFT, and time domain analysis.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFormLayout, QComboBox, QMainWindow
)

from utils.logger import Logger
from ..calculation import (
    calculate_basic_statistics,
    calculate_fft_analysis,
    calculate_time_domain_analysis
)


class BasicAnalysisTab(QWidget):
    """Tab containing basic signal analysis operations."""

    def __init__(self, parent_dialog):
        super().__init__()
        self.dialog = parent_dialog
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface for basic analysis."""
        Logger.log_message_static("Analysis-Dialog: Creating Basic Analysis tab", Logger.DEBUG)

        layout = QVBoxLayout(self)

        # Signal selector
        select_layout = QFormLayout()
        self.signal_combo = QComboBox()
        select_layout.addRow("Select Signal:", self.signal_combo)
        layout.addLayout(select_layout)

        # Basic analysis buttons
        button_layout = QHBoxLayout()

        stats_btn = QPushButton("Basic Statistics")
        stats_btn.setToolTip("View basic statistical metrics of the signal.")
        stats_btn.clicked.connect(self.show_basic_statistics)
        button_layout.addWidget(stats_btn)

        fft_btn = QPushButton("FFT Analysis")
        fft_btn.setToolTip("Analyze the frequency spectrum of the signal.")
        fft_btn.clicked.connect(self.show_fft_analysis)
        button_layout.addWidget(fft_btn)

        time_analysis_btn = QPushButton("Time Domain Analysis")
        time_analysis_btn.setToolTip("Examine time-domain characteristics of the signal.")
        time_analysis_btn.clicked.connect(self.show_time_analysis)
        button_layout.addWidget(time_analysis_btn)

        layout.addLayout(button_layout)

        Logger.log_message_static("Analysis-Dialog: Basic Analysis tab setup complete", Logger.DEBUG)

    def update_signal_list(self, signals):
        """Update the signal list in the combo box."""
        self.signal_combo.clear()
        self.signal_combo.addItems(signals)

    def get_selected_signal(self):
        """Get the currently selected signal name."""
        signal = self.signal_combo.currentText()
        if not signal:
            Logger.log_message_static("Analysis-Dialog: No signal selected in basic tab", Logger.WARNING)
            return None
        return signal

    def show_basic_statistics(self):
        """Calculate and display basic statistics for the selected signal."""
        Logger.log_message_static("Analysis-Dialog: Calculating basic statistics", Logger.INFO)

        signal = self.get_selected_signal()
        if not signal:
            return

        time_arr, values = self.dialog.get_signal_data(signal)
        if time_arr is None or values is None:
            return

        try:
            Logger.log_message_static(f"Analysis-Dialog: Computing statistics for signal '{signal}'", Logger.DEBUG)

            stats = calculate_basic_statistics(self.dialog, values)
            if stats is None:
                Logger.log_message_static("Analysis-Dialog: Basic statistics calculation was cancelled or failed",
                                          Logger.INFO)
                return

            # Show results in main panel
            self.dialog.show_analysis_results("Basic Statistics", signal, stats)

            # Create plot window
            self._create_statistics_plot(signal, time_arr, values, stats)

            Logger.log_message_static("Analysis-Dialog: Basic statistics displayed successfully", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Analysis-Dialog: Error calculating statistics: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Analysis-Dialog: Statistics traceback: {traceback.format_exc()}", Logger.DEBUG)

    def _create_statistics_plot(self, signal, time_arr, values, stats):
        """Create a plot window showing signal with statistical overlays."""
        plot_window = QMainWindow(self.dialog)
        plot_window.setWindowTitle(f"Statistics Plot: {signal}")
        plot_window.resize(800, 600)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        plot_widget = pg.GraphicsLayoutWidget()

        plot_item = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
        plot_item.setTitle("Signal Trend and Statistics")
        plot_item.setLabel('left', 'Amplitude')
        plot_item.setLabel('bottom', 'Time')

        # Plot signal
        if len(time_arr) == len(values):
            x_data = time_arr
        else:
            x_data = np.arange(len(values))
        plot_item.plot(x_data, values, pen='b')

        # Add statistical lines
        self._add_statistical_lines(plot_item, stats)

        plot_widget.addItem(plot_item, row=0, col=0)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(plot_window.close)

        layout.addWidget(plot_widget)
        layout.addWidget(close_btn)
        central_widget.setLayout(layout)
        plot_window.setCentralWidget(central_widget)
        plot_window.show()

        self.dialog.add_plot_window(plot_window)

    def _add_statistical_lines(self, plot_item, stats):
        """Add statistical reference lines to the plot."""
        # Mean line
        plot_item.addItem(pg.InfiniteLine(
            pos=stats["Mean"], angle=0,
            pen=pg.mkPen('r', width=2),
            label=f"Mean: {stats['Mean']:.2f}",
            labelOpts={"position": 0.95, "color": "r", "fill": (255, 255, 255, 50), "movable": False}
        ))

        # Standard deviation lines
        plot_item.addItem(pg.InfiniteLine(
            pos=stats["Mean"] + stats["Standard Deviation"], angle=0,
            pen=pg.mkPen('g', width=1, style=Qt.DashLine),
            label=f"Mean+σ: {stats['Mean'] + stats['Standard Deviation']:.2f}",
            labelOpts={"position": 0.95, "color": "g", "fill": (255, 255, 255, 50), "movable": False}
        ))

        plot_item.addItem(pg.InfiniteLine(
            pos=stats["Mean"] - stats["Standard Deviation"], angle=0,
            pen=pg.mkPen('g', width=1, style=Qt.DashLine),
            label=f"Mean-σ: {stats['Mean'] - stats['Standard Deviation']:.2f}",
            labelOpts={"position": 0.95, "color": "g", "fill": (255, 255, 255, 50), "movable": False}
        ))

        # Min/Max lines
        plot_item.addItem(pg.InfiniteLine(
            pos=stats["Min"], angle=0,
            pen=pg.mkPen('y', width=1, style=Qt.DotLine),
            label=f"Min: {stats['Min']:.2f}",
            labelOpts={"position": 0.95, "color": "y", "fill": (255, 255, 255, 50), "movable": False}
        ))

        plot_item.addItem(pg.InfiniteLine(
            pos=stats["Max"], angle=0,
            pen=pg.mkPen('y', width=1, style=Qt.DotLine),
            label=f"Max: {stats['Max']:.2f}",
            labelOpts={"position": 0.95, "color": "y", "fill": (255, 255, 255, 50), "movable": False}
        ))

        # RMS line
        plot_item.addItem(pg.InfiniteLine(
            pos=stats["RMS"], angle=0,
            pen=pg.mkPen('b', width=2, style=Qt.DotLine),
            label=f"RMS: {stats['RMS']:.2f}",
            labelOpts={"position": 0.95, "color": "b", "fill": (255, 255, 255, 50), "movable": False}
        ))

    def show_fft_analysis(self):
        """Perform FFT analysis on the selected signal."""
        Logger.log_message_static("Analysis-Dialog: Preparing FFT analysis", Logger.INFO)

        signal = self.get_selected_signal()
        if not signal:
            return

        time_arr, values = self.dialog.get_signal_data(signal)
        if time_arr is None or values is None:
            return

        try:
            Logger.log_message_static(f"Analysis-Dialog: Retrieved {len(values)} data points for FFT analysis",
                                      Logger.DEBUG)

            results = calculate_fft_analysis(self.dialog, time_arr, values)
            if results is None:
                Logger.log_message_static("Analysis-Dialog: FFT analysis aborted or failed", Logger.INFO)
                return

            # Create FFT plot window
            self._create_fft_plot(signal, results)

            # Display summary stats
            stats = {
                "Peak Frequency (Hz)": f"{results['Peak Frequency (Hz)']:.3f}",
                "Max Magnitude": f"{results['Max Magnitude']:.4e}",
                "Total Frequency Domain Energy": f"{results['Total Energy (Freq Domain)']:.4e}"
            }

            self.dialog.show_analysis_results("FFT Spectrum", signal, stats)

        except Exception as e:
            Logger.log_message_static(f"Analysis-Dialog: Error in FFT analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Analysis-Dialog: FFT traceback: {traceback.format_exc()}", Logger.DEBUG)

    def _create_fft_plot(self, signal, results):
        """Create FFT analysis plot window."""
        plot_window = QMainWindow(self.dialog)
        plot_window.setWindowTitle(f"FFT Analysis: {signal}")
        plot_window.resize(800, 600)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        plot_widget = pg.GraphicsLayoutWidget()

        # Time domain
        p1 = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
        p1.setTitle("Time Domain")
        p1.setLabel('left', 'Amplitude')
        p1.setLabel('bottom', 'Time (s)')
        p1.plot(results["Time Array"], results["Processed Signal"], pen='b')
        plot_widget.addItem(p1, row=0, col=0)

        # Frequency domain
        p2 = plot_widget.addPlot(row=1, col=0)
        p2.setTitle("Frequency Domain")
        p2.setLabel('left', 'Magnitude')
        p2.setLabel('bottom', 'Frequency (Hz)')
        p2.plot(results["Frequency Axis (Hz)"], results["Magnitude Spectrum"], pen='r')
        p2.setLogMode(x=True, y=False)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(plot_window.close)

        layout.addWidget(plot_widget)
        layout.addWidget(close_button)
        central_widget.setLayout(layout)
        plot_window.setCentralWidget(central_widget)
        plot_window.show()

        self.dialog.add_plot_window(plot_window)
        Logger.log_message_static("Analysis-Dialog: FFT plot window displayed successfully", Logger.INFO)

    def show_time_analysis(self):
        """Perform time-domain analysis on the selected signal."""
        Logger.log_message_static("Analysis-Dialog: Performing time-domain analysis", Logger.INFO)

        signal = self.get_selected_signal()
        if not signal:
            return

        time_arr, values = self.dialog.get_signal_data(signal)
        if time_arr is None or values is None:
            return

        try:
            Logger.log_message_static(f"Analysis-Dialog: Retrieved {len(values)} data points for time analysis",
                                      Logger.DEBUG)

            results = calculate_time_domain_analysis(self.dialog, time_arr, values)
            if results is None:
                Logger.log_message_static("Analysis-Dialog: Time-domain analysis cancelled or failed", Logger.INFO)
                return

            # Create time analysis plot
            self._create_time_analysis_plot(signal, time_arr, values, results)

            # Show results table
            self._show_time_analysis_results(signal, results)

            Logger.log_message_static("Analysis-Dialog: Time-domain analysis results displayed", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Analysis-Dialog: Error in time-domain analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Analysis-Dialog: Time analysis traceback: {traceback.format_exc()}",
                                      Logger.DEBUG)

    def _create_time_analysis_plot(self, signal, time_arr, values, results):
        """Create time domain analysis plot window."""
        plot_window = QMainWindow(self.dialog)
        plot_window.setWindowTitle(f"Time Analysis Plot: {signal}")
        plot_window.resize(800, 600)

        central = QWidget()
        layout = QVBoxLayout(central)
        plot_widget = pg.GraphicsLayoutWidget()

        # Signal with trend plot
        p1 = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
        p1.setTitle("Signal with Trendline")
        p1.setLabel('left', 'Amplitude')
        p1.setLabel('bottom', 'Time')
        p1.plot(time_arr, values, pen='b')

        if "Trend Coefficients" in results:
            trendline = np.polyval(results["Trend Coefficients"], time_arr)
            p1.plot(time_arr, trendline, pen=pg.mkPen('r', width=2), name="Trend")

        if "RMS" in results:
            p1.addItem(pg.InfiniteLine(
                pos=results["RMS"], angle=0,
                pen=pg.mkPen('g', width=2, style=Qt.DashLine),
                label=f"RMS: {results['RMS']:.2f}",
                labelOpts={"position": 0.95, "color": "g", "fill": (255, 255, 255, 50), "movable": False}
            ))

        plot_widget.addItem(p1, row=0, col=0)

        # Derivative plot
        p2 = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
        p2.setTitle("Signal Rate of Change")
        p2.setLabel('left', 'Rate')
        p2.setLabel('bottom', 'Time')

        if "First Derivative Stats" in results:
            dt = np.mean(np.diff(time_arr))
            derivative = np.gradient(values, dt)
            derivative_time = time_arr[:-1] + dt / 2
            p2.plot(derivative_time, derivative[:-1], pen='m')

            if "Mean Rate" in results:
                p2.addItem(pg.InfiniteLine(
                    pos=results["Mean Rate"], angle=0,
                    pen=pg.mkPen('y', width=2, style=Qt.DashLine),
                    label=f"Mean Rate: {results['Mean Rate']:.2f}",
                    labelOpts={"position": 0.95, "color": "y", "fill": (255, 255, 255, 50), "movable": False}
                ))

        p2.setXLink(p1)
        plot_widget.addItem(p2, row=1, col=0)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(plot_window.close)

        layout.addWidget(plot_widget)
        layout.addWidget(close_btn)
        central.setLayout(layout)
        plot_window.setCentralWidget(central)
        plot_window.show()

        self.dialog.add_plot_window(plot_window)

    def _show_time_analysis_results(self, signal, results):
        """Show time analysis results in the results panel."""
        # Format results for display
        display_results = {}

        for key, value in results.items():
            if key == "Trend Coefficients":
                display_results["Trend Order"] = f"{len(value) - 1}"
                continue
            elif isinstance(value, dict):
                display_results[key] = "See details below"
                for sub_key, sub_value in value.items():
                    display_results[f"└─ {sub_key}"] = f"{sub_value:.6g}" if isinstance(sub_value,
                                                                                        (int, float)) else str(
                        sub_value)
            elif isinstance(value, (int, float)):
                display_results[key] = f"{value:.6g}"
            else:
                display_results[key] = str(value)

        self.dialog.show_analysis_results("Time Domain Analysis", signal, display_results)