"""
Signal Analysis Module - Provides advanced signal analysis tools through a standalone dialog.

This module integrates with the CSV Signal Viewer application to add statistical analysis,
FFT processing, and time domain analysis capabilities. It works with the application's
data structure where signals are stored as (time_array, values_array) tuples.

Usage:
    from signal_analysis import show_analysis_dialog

    # In your main application:
    def open_analysis_dialog(self):
        show_analysis_dialog(self)
"""
import numpy as np
import pywt

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QPushButton, QTableWidget, QTableWidgetItem, QDialogButtonBox,
    QHeaderView, QComboBox, QWidget, QHBoxLayout, QGroupBox,
    QTabWidget, QFormLayout, QDoubleSpinBox, QTextEdit, QSplitter,
    QVBoxLayout, QScrollArea, QMainWindow, QLabel, QSpinBox, QMessageBox
)

from utils.logger import Logger
from .explanation import ExplanationTab
from .calculation import (
    calculate_basic_statistics,
    calculate_fft_analysis,
    calculate_time_domain_analysis,
    calculate_psd_analysis,
    calculate_peak_detection,
    calculate_hilbert_analysis,
    calculate_energy_analysis,
    calculate_phase_analysis,
    calculate_cepstrum_analysis,
    calculate_autocorrelation_analysis,
    calculate_cross_correlation_analysis,
    calculate_wavelet_analysis_cwt,
    calculate_wavelet_analysis_dwt,
    calculate_iir_filter,
    calculate_fir_filter,
    safe_prepare_signal,
    safe_sample_rate
)
from .helpers import calculate_bandwidth, format_array_for_display


class SignalAnalysisDialog(QDialog):
    """
    Dialog for performing various signal analysis operations.

    This dialog provides a user interface for selecting signals and applying
    different analysis methods. It displays the results in the dialog itself or
    in separate windows for visualizations like FFT.

    Attributes:
        parent (QWidget): Parent widget, typically the main application window
        signal_combo (QComboBox): Dropdown for selecting signals to analyze
        results_layout (QVBoxLayout): Layout for displaying analysis results
    """

    def __init__(self, parent=None):
        """
        Initialize the analysis dialog with reference to parent application data.

        Args:
            parent (QWidget): Parent widget, should be the main application window
                              that contains data_signals dictionary
        """
        Logger.log_message_static("Initializing SignalAnalysisDialog", Logger.DEBUG)
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Signal Analysis")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowMaximizeButtonHint)
        self.resize(800, 600)

        # Store plot windows to prevent garbage collection
        self._plot_windows = []
        Logger.log_message_static("Creating UI for signal analysis dialog", Logger.DEBUG)
        self.setup_ui()
        self.update_signal_list()
        Logger.log_message_static("SignalAnalysisDialog initialization complete", Logger.DEBUG)

    def setup_ui(self):
        """
        Create and arrange the user interface components for the dialog.
        """
        Logger.log_message_static("Setting up UI components for signal analysis dialog", Logger.DEBUG)
        layout = QVBoxLayout(self)

        # Add a QSplitter to allow resizing between top and bottom areas
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(self.splitter)

        # Create top widget for tabs
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)

        # Create tabs for better organization
        self.tab_widget = QTabWidget()
        top_layout.addWidget(self.tab_widget)

        # Add the top widget to the splitter
        self.splitter.addWidget(top_widget)

        # ===== Basic Analysis Tab =====
        Logger.log_message_static("Creating Basic Analysis tab", Logger.DEBUG)
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        self.tab_widget.addTab(basic_tab, "Basic Analysis")

        # Signal selector
        select_layout = QFormLayout()
        select_layout.addRow("Select Signal:", self.create_signal_selector())
        basic_layout.addLayout(select_layout)

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

        basic_layout.addLayout(button_layout)

        # ===== Advanced Analysis Tab =====
        Logger.log_message_static("Creating Advanced Analysis tab", Logger.DEBUG)
        adv_tab = QWidget()
        adv_layout = QVBoxLayout(adv_tab)
        self.tab_widget.addTab(adv_tab, "Advanced Analysis")

        # Signal selector for advanced tab
        adv_select_layout = QFormLayout()
        self.adv_signal_combo = QComboBox()
        adv_select_layout.addRow("Select Signal:", self.adv_signal_combo)
        adv_layout.addLayout(adv_select_layout)

        # Advanced analysis buttons
        adv_button_layout = QHBoxLayout()

        psd_btn = QPushButton("Power Spectral Density")
        psd_btn.setToolTip("Visualize the power distribution across frequencies.")
        psd_btn.clicked.connect(self.show_psd_analysis)
        adv_button_layout.addWidget(psd_btn)

        autocorr_btn = QPushButton("Autocorrelation")
        autocorr_btn.setToolTip("Measure the signal's self-similarity over time.")
        autocorr_btn.clicked.connect(self.show_autocorrelation_analysis)
        adv_button_layout.addWidget(autocorr_btn)

        peaks_btn = QPushButton("Peak Detection")
        peaks_btn.setToolTip("Identify and display peaks in the signal.")
        peaks_btn.clicked.connect(self.show_peak_detection)
        adv_button_layout.addWidget(peaks_btn)

        adv_layout.addLayout(adv_button_layout)

        adv_button_layout2 = QHBoxLayout()

        filter_btn = QPushButton("Apply Filter")
        filter_btn.setToolTip("Apply a frequency filter to the signal.")
        filter_btn.clicked.connect(self.show_filter_dialog)
        adv_button_layout2.addWidget(filter_btn)

        hilbert_btn = QPushButton("Hilbert Transform")
        hilbert_btn.setToolTip("Extract amplitude, phase, and frequency details.")
        hilbert_btn.clicked.connect(self.show_hilbert_analysis)
        adv_button_layout2.addWidget(hilbert_btn)

        energy_btn = QPushButton("Energy Analysis")
        energy_btn.setToolTip("Evaluate the energy distribution of the signal.")
        energy_btn.clicked.connect(self.show_energy_analysis)
        adv_button_layout2.addWidget(energy_btn)

        adv_layout.addLayout(adv_button_layout2)

        adv_button_layout3 = QHBoxLayout()

        phase_btn = QPushButton("Phase Analysis")
        phase_btn.setToolTip("Inspect the phase behavior of the signal.")
        phase_btn.clicked.connect(self.show_phase_analysis)
        adv_button_layout3.addWidget(phase_btn)

        cepstrum_btn = QPushButton("Cepstral Analysis")
        cepstrum_btn.setToolTip("Reveal periodic patterns in the signal's spectrum.")
        cepstrum_btn.clicked.connect(self.show_cepstrum_analysis)
        adv_button_layout3.addWidget(cepstrum_btn)

        wavelet_btn = QPushButton("Wavelet Transform")
        wavelet_btn.setToolTip("Decompose the signal into time-frequency components.")
        wavelet_btn.clicked.connect(self.show_wavelet_dialog)
        adv_button_layout3.addWidget(wavelet_btn)

        adv_layout.addLayout(adv_button_layout3)

        # ===== Cross Analysis Tab =====
        Logger.log_message_static("Creating Cross Analysis tab", Logger.DEBUG)
        cross_tab = QWidget()
        cross_layout = QVBoxLayout(cross_tab)
        self.tab_widget.addTab(cross_tab, "Cross Analysis")

        # Signal selectors for cross analysis
        cross_select_layout = QFormLayout()
        self.cross_signal1_combo = QComboBox()
        self.cross_signal2_combo = QComboBox()
        cross_select_layout.addRow("Signal 1:", self.cross_signal1_combo)
        cross_select_layout.addRow("Signal 2:", self.cross_signal2_combo)
        cross_layout.addLayout(cross_select_layout)

        # Cross analysis buttons
        cross_button_layout = QHBoxLayout()

        xcorr_btn = QPushButton("Cross Correlation")
        xcorr_btn.setToolTip("Compare and find similarities between two signals.")
        xcorr_btn.clicked.connect(self.show_cross_correlation_analysis)
        cross_button_layout.addWidget(xcorr_btn)

        cross_layout.addLayout(cross_button_layout)

        # Add the Explanations tab
        Logger.log_message_static("Creating Explanations tab", Logger.DEBUG)
        explanation_tab = ExplanationTab(self)
        self.tab_widget.addTab(explanation_tab, "Explanations")

        # Create results area and add to splitter
        results_widget = QWidget()
        self.results_layout = QVBoxLayout(results_widget)

        # Results area (initially empty) - shared across tabs
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        results_scroll.setWidget(self.results_widget)

        results_container = QWidget()
        results_container_layout = QVBoxLayout(results_container)
        results_container_layout.addWidget(results_scroll)

        self.splitter.addWidget(results_container)

        # Set initial sizes of the splitter areas (e.g., 30% top, 70% bottom)
        self.splitter.setSizes([300, 700])

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        Logger.log_message_static("UI setup complete for signal analysis dialog", Logger.DEBUG)

    def create_signal_selector(self):
        """Create a signal selection dropdown."""
        Logger.log_message_static("Creating signal selector dropdown", Logger.DEBUG)
        self.signal_combo = QComboBox()
        return self.signal_combo

    def update_signal_list(self):
        """
        Populate all signal selection dropdowns with available signals.
        """
        Logger.log_message_static("Updating signal lists in all dropdowns", Logger.DEBUG)
        if hasattr(self.parent, 'data_signals'):
            signals = list(self.parent.data_signals.keys())
            Logger.log_message_static(f"Found {len(signals)} available signals", Logger.DEBUG)

            # Update all combo boxes
            self.signal_combo.clear()
            self.signal_combo.addItems(signals)

            self.adv_signal_combo.clear()
            self.adv_signal_combo.addItems(signals)

            self.cross_signal1_combo.clear()
            self.cross_signal1_combo.addItems(signals)

            self.cross_signal2_combo.clear()
            self.cross_signal2_combo.addItems(signals)
        else:
            Logger.log_message_static("No data_signals attribute found in parent", Logger.WARNING)

    def get_selected_signal(self, combo=None):
        """
        Get the name of the currently selected signal in the dropdown.

        Args:
            combo (QComboBox, optional): The combobox to get the selection from.
                                        Defaults to the main signal_combo.

        Returns:
            str or None: Name of the selected signal, or None if no signal is selected
        """
        if combo is None:
            combo = self.signal_combo

        signal = combo.currentText()
        if not signal:
            Logger.log_message_static("No signal selected in dropdown", Logger.WARNING)
            return None

        Logger.log_message_static(f"Selected signal: '{signal}'", Logger.DEBUG)
        return signal

    def clear_results(self):
        """
        Clear all widgets from the results area to prepare for new results.
        """
        Logger.log_message_static("Clearing results area", Logger.DEBUG)
        for i in reversed(range(self.results_layout.count())):
            widget = self.results_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

    def show_basic_statistics(self):
        """
        Calculate and display basic statistics for the selected signal.
        Includes tabular results in panel and separate trend visualization in a new window.
        """
        Logger.log_message_static("Calculating basic statistics", Logger.INFO)
        signal = self.get_selected_signal()
        if not signal:
            Logger.log_message_static("Cannot calculate statistics: No signal selected", Logger.WARNING)
            return

        Logger.log_message_static(f"Computing statistics for signal '{signal}'", Logger.DEBUG)
        try:
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for statistics", Logger.DEBUG)

            stats = calculate_basic_statistics(self, values)
            if stats is None:
                Logger.log_message_static("Basic statistics calculation was cancelled or failed", Logger.INFO)
                return

            # === 1) Show tabular results in main panel (as before) ===
            self.clear_results()

            result_title = QLabel(f"Statistics Results: {signal}")
            result_title.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.results_layout.addWidget(result_title)

            table = QTableWidget()
            table.setColumnCount(2)
            table.setRowCount(len(stats))
            table.setHorizontalHeaderLabels(["Metric", "Value"])
            header = table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

            Logger.log_message_static(f"Creating results table with {len(stats)} rows", Logger.DEBUG)

            for i, (key, value) in enumerate(stats.items()):
                table.setItem(i, 0, QTableWidgetItem(key))
                if isinstance(value, (int, float)):
                    table.setItem(i, 1, QTableWidgetItem(f"{value:.6g}"))
                else:
                    table.setItem(i, 1, QTableWidgetItem(str(value)))
                Logger.log_message_static(f"Added table row: {key} = {value}", Logger.DEBUG)

            self.results_layout.addWidget(table)

            # === 2) Plot in separate window ===
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Statistics Plot: {signal}")
            plot_window.resize(800, 600)

            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            plot_widget = pg.GraphicsLayoutWidget()
            plot_item = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
            plot_item.setTitle("Signal Trend and Statistics")
            plot_item.setLabel('left', 'Amplitude')
            plot_item.setLabel('bottom', 'Time')
            if len(time_arr) == len(values):
                x_data = time_arr
            else:
                x_data = np.arange(len(values))
            plot_item.plot(x_data, values, pen='b')
            plot_widget.addItem(plot_item, row=0, col=0)

            # Optional overlays
            plot_item.addItem(pg.InfiniteLine(pos=stats["Mean"], angle=0,
                                              pen=pg.mkPen('r', width=2),
                                              label=f"Mean: {stats['Mean']:.2f}",
                                              labelOpts={"position": 0.95, "color": "r", "fill": (255, 255, 255, 50),
                                                         "movable": False}
                                              )
                              )
            plot_item.addItem(pg.InfiniteLine(pos=stats["Mean"] + stats["Standard Deviation"], angle=0,
                                              pen=pg.mkPen('g', width=1, style=Qt.DashLine),
                                              label=f"Mean+: {stats['Mean'] + stats['Standard Deviation']:.2f}",
                                              labelOpts={"position": 0.95, "color": "g", "fill": (255, 255, 255, 50),
                                                         "movable": False}
                                              )
                              )
            plot_item.addItem(pg.InfiniteLine(pos=stats["Mean"] - stats["Standard Deviation"], angle=0,
                                              pen=pg.mkPen('g', width=1, style=Qt.DashLine),
                                              label=f"Mean-: {stats['Mean'] - stats['Standard Deviation']:.2f}",
                                              labelOpts={"position": 0.95, "color": "g", "fill": (255, 255, 255, 50),
                                                         "movable": False}
                                              )
                              )
            plot_item.addItem(pg.InfiniteLine(pos=stats["Min"], angle=0,
                                              pen=pg.mkPen('y', width=1, style=Qt.DotLine),
                                              label=f"Min: {stats['Min']:.2f}",
                                              labelOpts={"position": 0.95, "color": "y", "fill": (255, 255, 255, 50),
                                                         "movable": False}
                                              )
                              )
            plot_item.addItem(pg.InfiniteLine(pos=stats["Max"], angle=0,
                                              pen=pg.mkPen('y', width=1, style=Qt.DotLine),
                                              label=f"Max: {stats['Max']:.2f}",
                                              labelOpts={"position": 0.95, "color": "y", "fill": (255, 255, 255, 50),
                                                         "movable": False}
                                              )
                              )
            plot_item.addItem(pg.InfiniteLine(pos=stats["RMS"], angle=0,
                                              pen=pg.mkPen('y', width=2, style=Qt.DotLine),
                                              label=f"RMS: {stats['RMS']:.2f}",
                                              labelOpts={"position": 0.95, "color": "b", "fill": (255, 255, 255, 50),
                                                         "movable": False}
                                              )
                              )

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(plot_window.close)

            layout.addWidget(plot_widget)
            layout.addWidget(close_btn)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)
            plot_window.show()
            self._plot_windows.append(plot_window)

            Logger.log_message_static("Basic statistics displayed with trend", Logger.INFO)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error calculating statistics: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Statistics traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_fft_analysis(self):
        """
        Perform FFT analysis on the selected signal and display results in time and frequency domains.
        Uses `calculate_fft_analysis` internally.
        """
        Logger.log_message_static("Preparing FFT analysis", Logger.INFO)
        signal = self.get_selected_signal()
        if not signal:
            Logger.log_message_static("Cannot perform FFT: No signal selected", Logger.WARNING)
            return

        try:
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for FFT analysis", Logger.DEBUG)

            results = calculate_fft_analysis(self, time_arr, values)
            if results is None:
                Logger.log_message_static("FFT analysis aborted or failed", Logger.INFO)
                return

            # Extract result data
            time_arr = results["Time Array"]
            signal_data = results["Processed Signal"]
            freqs = results["Frequency Axis (Hz)"]
            spectrum = results["Magnitude Spectrum"]

            # Create GUI
            plot_window = QMainWindow(self)
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
            p1.plot(time_arr, signal_data, pen='b')
            plot_widget.addItem(p1, row=0, col=0)

            # Frequency domain
            p2 = plot_widget.addPlot(row=1, col=0)
            p2.setTitle("Frequency Domain")
            p2.setLabel('left', 'Magnitude')
            p2.setLabel('bottom', 'Frequency (Hz)')
            p2.plot(freqs, spectrum, pen='r')
            p2.setLogMode(x=True, y=False)

            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)
            plot_window.show()

            self._plot_windows.append(plot_window)
            Logger.log_message_static("FFT plot window displayed successfully", Logger.INFO)

            # Display summary stats
            stats = {
                "Peak Frequency (Hz)": f"{results['Peak Frequency (Hz)']:.3f}",
                "Max Magnitude": f"{results['Max Magnitude']:.4e}",
                "Total Frequency Domain Energy": f"{results['Total Energy (Freq Domain)']:.4e}"
            }

            self.show_analysis_results("FFT Spectrum", signal, stats)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in FFT analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"FFT traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_time_analysis(self):
        """
        Perform time-domain analysis on the selected signal and display the results
        with both visualization and detailed metrics.
        """
        Logger.log_message_static("Performing time-domain analysis", Logger.INFO)
        signal = self.get_selected_signal()
        if not signal:
            Logger.log_message_static("Cannot perform time analysis: No signal selected", Logger.WARNING)
            return

        try:
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for time analysis", Logger.DEBUG)

            results = calculate_time_domain_analysis(self, time_arr, values)
            if results is None:
                Logger.log_message_static("Time-domain analysis cancelled or failed", Logger.INFO)
                return

            self.clear_results()

            result_title = QLabel(f"Time-Domain Analysis Results: {signal}")
            result_title.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.results_layout.addWidget(result_title)

            # === Plot window ===
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Time Analysis Plot: {signal}")
            plot_window.resize(800, 600)

            central = QWidget()
            layout = QVBoxLayout(central)
            plot_widget = pg.GraphicsLayoutWidget()

            # === Helper for labeled horizontal lines ===
            def add_hline(plot, y_val, color, label_text, style=Qt.DashLine):
                plot.addItem(pg.InfiniteLine(
                    pos=y_val,
                    angle=0,
                    pen=pg.mkPen(color=color, width=2, style=style),
                    label=label_text,
                    labelOpts={"position": 0.95, "color": color,
                               "fill": (255, 255, 255, 50),
                               "movable": False}
                ))

            # === Plot 1: signal + trend ===
            p1 = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
            p1.setTitle("Signal with Trendline")
            p1.setLabel('left', 'Amplitude')
            p1.setLabel('bottom', 'Time')
            p1.plot(time_arr, values, pen='b')

            if "Trend Coefficients" in results:
                trendline = np.polyval(results["Trend Coefficients"], time_arr)
                p1.plot(time_arr, trendline, pen=pg.mkPen('r', width=2), name="Trend")

            if "RMS" in results:
                add_hline(p1, results["RMS"], "g", f"RMS: {results['RMS']:.2f}")

            plot_widget.addItem(p1, row=0, col=0)

            # === Plot 2: derivative ===
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
                    add_hline(p2, results["Mean Rate"], "y", f"Mean Rate: {results['Mean Rate']:.2f}")

            p2.setXLink(p1)
            plot_widget.addItem(p2, row=1, col=0)

            # Finalize
            layout.addWidget(plot_widget)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(plot_window.close)
            layout.addWidget(close_btn)

            central.setLayout(layout)
            plot_window.setCentralWidget(central)
            plot_window.show()
            self._plot_windows.append(plot_window)

            # === Table ===
            table = QTableWidget()
            table.setColumnCount(2)
            table.setRowCount(len(results))
            table.setHorizontalHeaderLabels(["Metric", "Value"])
            header = table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

            row = 0
            for key, value in results.items():
                if key == "Trend Coefficients":
                    table.setItem(row, 0, QTableWidgetItem("Trend Order"))
                    table.setItem(row, 1, QTableWidgetItem(f"{len(value) - 1}"))
                    row += 1
                    continue

                table.setItem(row, 0, QTableWidgetItem(key))
                if isinstance(value, (int, float)):
                    table.setItem(row, 1, QTableWidgetItem(f"{value:.6g}"))
                elif isinstance(value, dict):
                    table.setItem(row, 1, QTableWidgetItem("See below"))
                    row += 1
                    for sub_key, sub_value in value.items():
                        table.setItem(row, 0, QTableWidgetItem(f"└─ {sub_key}"))
                        table.setItem(row, 1, QTableWidgetItem(
                            f"{sub_value:.6g}" if isinstance(sub_value, (int, float)) else str(sub_value)))
                        row += 1
                    continue
                else:
                    table.setItem(row, 1, QTableWidgetItem(str(value)))
                row += 1

            table.setRowCount(row)
            self.results_layout.addWidget(table)

            Logger.log_message_static("Time-domain analysis results displayed with visualization", Logger.INFO)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in time-domain analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Time analysis traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_psd_analysis(self):
        """
        Calculate and display Power Spectral Density for the selected signal.

        This method retrieves the selected signal, computes its PSD using Welch's method,
        displays the result in a plot window, and logs associated statistics.

        GUI-specific: uses self.parent.data_signals, self.display_results, self._plot_windows, etc.
        """
        Logger.log_message_static("Preparing PSD analysis", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("Cannot perform PSD analysis: No signal selected", Logger.WARNING)
            return

        try:
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for PSD analysis", Logger.DEBUG)

            result = calculate_psd_analysis(self, time_arr, values)
            if result is None:
                Logger.log_message_static("PSD analysis returned no result", Logger.WARNING)
                return

            freqs = result["Frequency Axis (Hz)"]
            psd = result["PSD (Power/Hz)"]
            peak_freq = result["Peak Frequency (Hz)"]

            # Plotting
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Power Spectral Density: {signal}")
            plot_window.resize(800, 600)

            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            plot_widget = pg.GraphicsLayoutWidget()

            p = plot_widget.addPlot()
            p.setTitle("Power Spectral Density")
            p.setLabel('left', 'Power/Frequency (dB/Hz)')
            p.setLabel('bottom', 'Frequency (Hz)')
            p.setLogMode(x=True, y=False)

            psd_db = 10 * np.log10(psd)
            p.plot(freqs, psd_db, pen='g')

            peak_line = pg.InfiniteLine(pos=peak_freq, angle=90, pen='r')
            p.addItem(peak_line)

            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)
            plot_window.show()

            Logger.log_message_static("PSD plot window displayed successfully", Logger.INFO)
            self._plot_windows.append(plot_window)

            # Show stats
            self.show_analysis_results("PSD Analysis", signal, {
                "Peak Frequency (Hz)": result["Peak Frequency (Hz)"],
                "Max Power (dB)": result["Peak Power (dB)"],
                "Total Power": result["Total Power"],
                "RMS Power": result["RMS Amplitude"],
                "Bandwidth (3dB)": calculate_bandwidth(freqs, psd)
            })

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in PSD analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"PSD analysis traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_peak_detection(self):
        """
        Detect and display peaks for the selected signal, including type, heights, widths, and locations.

        Automatically detects whether to analyze positive or negative peaks based on signal polarity.
        Visualizes detected peaks on the time-domain signal.
        """
        Logger.log_message_static("Preparing peak detection analysis", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("No signal selected for peak detection", Logger.WARNING)
            return

        try:
            time_arr, values = self.parent.data_signals[signal]
            result = calculate_peak_detection(self, time_arr, values)
            if result is None:
                return
            if "Result" in result:
                self.show_analysis_results("Peak Detection", signal, result)
                return

            Logger.log_message_static("Creating peak detection plot window", Logger.DEBUG)
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Peak Detection: {signal}")
            plot_window.resize(800, 600)

            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            plot_widget = pg.GraphicsLayoutWidget()

            p = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
            p.setTitle(f"{result['Peak Type']} Peak Detection")
            p.setLabel('left', 'Amplitude')
            p.setLabel('bottom', 'Time (s)')
            p.plot(time_arr, values, pen='b')
            plot_widget.addItem(p, row=0, col=0)

            scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen('r', width=2), brush=pg.mkBrush('r'))
            spots = [{'pos': (t, h), 'data': i} for t, h, i in
                     zip(result["Times"], result["Heights"], result["Indices"])]
            scatter.addPoints(spots)
            p.addItem(scatter)

            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)
            plot_window.show()

            self._plot_windows.append(plot_window)

            # Prepare display
            display_data = {
                "Peak Type": result["Peak Type"],
                "Count": result["Count"],
                "Mean Height": round(result["Mean Height"], 3),
                "Max Height": round(result["Max Height"], 3),
                "Min Height": round(result["Min Height"], 3),
                "Mean Width (s)": round(result["Mean Width"], 3),
                "Peak Times (s)": format_array_for_display(result["Times"]),
                "Peak Heights": format_array_for_display(result["Heights"]),
                "Peak Indices": format_array_for_display(result["Indices"])
            }

            self.show_analysis_results("Peak Detection", signal, display_data)
            Logger.log_message_static("Peak detection analysis complete", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Error in peak detection: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(traceback.format_exc(), Logger.DEBUG)

    def show_hilbert_analysis(self):
        """
        GUI wrapper for Hilbert analysis. Displays amplitude envelope, phase, and frequency.

        This method uses calculate_hilbert_analysis() and renders 4 linked subplots + summary.
        """
        Logger.log_message_static("Preparing Hilbert transform analysis", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("No signal selected for Hilbert transform", Logger.WARNING)
            return

        try:
            time_arr, values = self.parent.data_signals[signal]
            result = calculate_hilbert_analysis(self, time_arr, values)
            if result is None:
                return

            amplitude_envelope = result["Amplitude Envelope"]
            phase = result["Unwrapped Phase"]
            inst_freq = result["Instantaneous Frequency (Hz)"]

            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Hilbert Transform: {signal}")
            plot_window.resize(800, 800)

            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            plot_widget = pg.GraphicsLayoutWidget()

            p1 = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
            p1.setTitle("Original Signal")
            p1.setLabel('left', 'Amplitude')
            p1.setLabel('bottom', 'Time')
            p1.plot(time_arr, values, pen='b')
            plot_widget.addItem(p1, row=0, col=0)

            p2 = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
            p2.setTitle("Amplitude Envelope")
            p2.setLabel('left', 'Amplitude')
            p2.setLabel('bottom', 'Time')
            p2.plot(time_arr, amplitude_envelope, pen='r')
            p2.plot(time_arr, values, pen=pg.mkPen('b', width=1, style=Qt.PenStyle.DotLine))
            plot_widget.addItem(p2, row=1, col=0)

            p3 = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
            p3.setTitle("Instantaneous Phase")
            p3.setLabel('left', 'Phase (rad)')
            p3.setLabel('bottom', 'Time')
            p3.plot(time_arr, phase, pen='g')
            plot_widget.addItem(p3, row=2, col=0)

            p4 = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
            p4.setTitle("Instantaneous Frequency")
            p4.setLabel('left', 'Frequency (Hz)')
            p4.setLabel('bottom', 'Time')
            p4.plot(time_arr, inst_freq, pen='m')
            plot_widget.addItem(p4, row=3, col=0)

            p2.setXLink(p1)
            p3.setXLink(p1)
            p4.setXLink(p1)

            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)
            plot_window.show()

            self._plot_windows.append(plot_window)

            summary = {
                "Mean Amplitude": round(result["Mean Amplitude"], 3),
                "Max Amplitude": round(result["Max Amplitude"], 3),
                "Mean Frequency (Hz)": round(result["Mean Frequency (Hz)"], 3),
                "Median Frequency (Hz)": round(result["Median Frequency (Hz)"], 3),
                "Max Frequency (Hz)": round(result["Max Frequency (Hz)"], 3),
                "Phase Range (rad)": round(result["Phase Range (rad)"], 3)
            }

            self.show_analysis_results("Hilbert Transform", signal, summary)
            Logger.log_message_static("Hilbert transform analysis complete", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Error in Hilbert transform: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(traceback.format_exc(), Logger.DEBUG)

    def show_energy_analysis(self):

        """
        Perform and display energy analysis including time/frequency domain energy and band distribution.
        """
        Logger.log_message_static("Preparing energy analysis", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("No signal selected for energy analysis", Logger.WARNING)
            return

        try:
            time_arr, values = self.parent.data_signals[signal]
            result = calculate_energy_analysis(self, time_arr, values)
            if result is None:
                return

            freqs = result["Freqs"]
            spectrum = result["Spectrum"]
            band_edges = result["Band Edges"]
            band_percentages = result["Band Percentages"]

            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Energy Analysis: {signal}")
            plot_window.resize(800, 600)

            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            plot_widget = pg.GraphicsLayoutWidget()

            # Spectrum plot
            p1 = plot_widget.addPlot(row=0, col=0)
            p1.setTitle("Energy Density Spectrum")
            p1.setLabel('left', 'Energy Density')
            p1.setLabel('bottom', 'Frequency (Hz)')
            p1.setLogMode(x=True, y=True)
            p1.plot(freqs, spectrum, pen='r')

            # Band bar plot
            p2 = plot_widget.addPlot(row=1, col=0)
            p2.setTitle("Energy Distribution by Frequency Band")
            p2.setLabel('left', 'Energy Percentage (%)')
            p2.setLabel('bottom', 'Band')

            x = np.arange(len(band_percentages))
            bar = pg.BarGraphItem(x=x, height=band_percentages, width=0.6, brush='b')
            p2.addItem(bar)

            ticks = [(i, f"{band_edges[i]:.1f}-{band_edges[i + 1]:.1f}") for i in range(len(band_percentages))]
            p2.getAxis('bottom').setTicks([ticks])

            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)
            plot_window.show()
            self._plot_windows.append(plot_window)

            summary = {
                "Total Energy (Time Domain)": round(result["Total Energy (Time Domain)"], 3),
                "Total Energy (Frequency Domain)": round(result["Total Energy (Frequency Domain)"], 3),
                "Signal Power": round(result["Signal Power"], 3),
                "RMS Value": round(result["RMS Value"], 3),
                "Dominant Frequency Band": result["Dominant Frequency Band"],
                "Dominant Band Energy": result["Dominant Band Energy"]
            }

            # Add band breakdown
            summary.update({
                band: f"{perc:.1f}%" for band, perc in result["Energy Distribution (%)"].items()
            })

            self.show_analysis_results("Energy Analysis", signal, summary)
            Logger.log_message_static("Energy analysis complete", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Error in energy analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(traceback.format_exc(), Logger.DEBUG)

    def show_phase_analysis(self):
        """
        Perform phase analysis using the Hilbert transform and display phase and velocity plots.
        """
        Logger.log_message_static("Preparing phase analysis", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("No signal selected for phase analysis", Logger.WARNING)
            return

        try:
            time_arr, values = self.parent.data_signals[signal]
            result = calculate_phase_analysis(self, time_arr, values)
            if result is None:
                return

            phase = result["Phase"]
            velocity = result["Phase Velocity"]
            velocity_time = time_arr[1:]

            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Phase Analysis: {signal}")
            plot_window.resize(800, 600)

            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            plot_widget_phase = pg.PlotWidget(
                title="Unwrapped Phase",
                axisItems={'bottom': pg.DateAxisItem(orientation='bottom')}
            )
            plot_widget_phase.setLabel('left', 'Phase (rad)')
            plot_widget_phase.setLabel('bottom', 'Time (s)')
            plot_widget_phase.plot(time_arr, phase, pen='b')

            plot_widget_velocity = pg.PlotWidget(
                title="Phase Velocity",
                axisItems={'bottom': pg.DateAxisItem(orientation='bottom')}
            )
            plot_widget_velocity.setLabel('left', 'Phase Velocity (rad/s)')
            plot_widget_velocity.setLabel('bottom', 'Time (s)')
            plot_widget_velocity.plot(velocity_time, velocity, pen='g')

            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            layout.addWidget(plot_widget_phase)
            layout.addWidget(plot_widget_velocity)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)
            plot_window.show()

            self._plot_windows.append(plot_window)

            stats = result["Phase Stats"]
            formatted_stats = {k: round(v, 4) for k, v in stats.items()}
            self.show_analysis_results("Phase Analysis", signal, formatted_stats)
            Logger.log_message_static("Phase analysis complete", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Error in phase analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(traceback.format_exc(), Logger.DEBUG)

    def show_cepstrum_analysis(self):
        """
        GUI wrapper for cepstrum analysis. Displays original signal, log power spectrum and cepstrum.
        """
        Logger.log_message_static("Preparing cepstrum analysis", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("No signal selected for cepstrum analysis", Logger.WARNING)
            return

        try:
            time_arr, values = self.parent.data_signals[signal]
            result = calculate_cepstrum_analysis(self, time_arr, values)
            if result is None:
                return

            cepstrum = result["Cepstrum"]
            quefrency = result["Quefrency"]
            log_power = result["Log Power Spectrum"]
            fs = result["Sampling Rate"]

            n = len(cepstrum)
            freqs = np.fft.fftfreq(n, d=1 / fs)
            freqs = np.fft.fftshift(freqs)
            log_power_shift = np.fft.fftshift(log_power)

            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Cepstrum Analysis: {signal}")
            plot_window.resize(800, 600)

            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            plot_widget = pg.GraphicsLayoutWidget()

            p1 = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
            p1.setTitle("Original Signal")
            p1.setLabel('left', 'Amplitude')
            p1.setLabel('bottom', 'Time (s)')
            p1.plot(time_arr, values, pen='b')
            plot_widget.addItem(p1, row=0, col=0)

            p2 = plot_widget.addPlot(row=1, col=0)
            p2.setTitle("Log Power Spectrum")
            p2.setLabel('left', 'Log Power')
            p2.setLabel('bottom', 'Frequency (Hz)')
            mid = len(freqs) // 2
            p2.plot(freqs[mid:], log_power_shift[mid:], pen='g')

            p3 = plot_widget.addPlot(row=2, col=0)
            p3.setTitle("Cepstrum")
            p3.setLabel('left', 'Amplitude')
            p3.setLabel('bottom', 'Quefrency (s)')
            p3.plot(quefrency[:n // 2], cepstrum[:n // 2], pen='r')

            # Peak annotation
            if result["Fundamental Frequency (Hz)"] > 0:
                fq = result["Peak Quefrency (s)"]
                val = cepstrum[int(fq * fs)]
                text = pg.TextItem(text=f"{result['Fundamental Frequency (Hz)']:.2f} Hz", color='y', anchor=(0, 0))
                text.setPos(fq, val)
                p3.addItem(text)
                scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen('y', width=2), brush=pg.mkBrush('y'))
                scatter.addPoints([{'pos': (fq, val)}])
                p3.addItem(scatter)

            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)
            plot_window.show()

            self._plot_windows.append(plot_window)

            summary = {
                "Detected Fundamental Frequency": f"{result['Fundamental Frequency (Hz)']:.2f} Hz"
                if result['Fundamental Frequency (Hz)'] > 0 else "None",
                "Peak Quefrency (s)": f"{result['Peak Quefrency (s)']:.6f}",
                "Max Cepstrum Value": round(result["Max Cepstrum Value"], 4),
                "Mean Cepstrum Value": round(result["Mean Cepstrum Value"], 4)
            }

            for i, (q, val, freq) in enumerate(result["Peaks"]):
                summary[f"Peak {i + 1} Quefrency"] = f"{q:.6f} s"
                summary[f"Peak {i + 1} Frequency"] = f"{freq:.2f} Hz"

            self.show_analysis_results("Cepstrum Analysis", signal, summary)
            Logger.log_message_static("Cepstrum analysis complete", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Error in cepstrum analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Cepstrum traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_autocorrelation_analysis(self):
        """
        GUI wrapper for autocorrelation analysis. Displays ACF plot and key time statistics.
        """
        Logger.log_message_static("Preparing autocorrelation analysis", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("No signal selected for autocorrelation", Logger.WARNING)
            return

        try:
            time_arr, values = self.parent.data_signals[signal]
            result = calculate_autocorrelation_analysis(self, time_arr, values)
            if result is None:
                return

            acf = result["Autocorrelation"]
            lags = result["Lags (s)"]

            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Autocorrelation: {signal}")
            plot_window.resize(800, 600)

            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            plot_widget = pg.GraphicsLayoutWidget()

            p = plot_widget.addPlot()
            p.setTitle("Autocorrelation")
            p.setLabel('left', 'Correlation')
            p.setLabel('bottom', 'Lag (s)')
            p.plot(lags, acf, pen='b')

            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)
            plot_window.show()

            self._plot_windows.append(plot_window)

            result_display = {
                "Peak Correlation": result["Peak Correlation"],
                "First Minimum": result["First Minimum (s)"],
                "First Zero Crossing": result["First Zero Crossing (s)"],
                "Decorrelation Time": result["Decorrelation Time (s)"]
            }

            self.show_analysis_results("Autocorrelation Analysis", signal, result_display)
            Logger.log_message_static("Autocorrelation analysis complete", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Error in autocorrelation analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Autocorrelation traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_cross_correlation_analysis(self):
        """
        GUI wrapper to show cross-correlation analysis results between two selected signals.
        Computes correlation using `calculate_cross_correlation_analysis` and displays plots and statistics.
        """
        Logger.log_message_static("Preparing cross-correlation analysis", Logger.INFO)
        signal1 = self.get_selected_signal(self.cross_signal1_combo)
        signal2 = self.get_selected_signal(self.cross_signal2_combo)

        if not signal1 or not signal2:
            Logger.log_message_static("Cannot perform cross-correlation: One or both signals not selected",
                                      Logger.WARNING)
            return

        try:
            time_arr1, values1 = self.parent.data_signals[signal1]
            time_arr2, values2 = self.parent.data_signals[signal2]

            Logger.log_message_static(f"Computing cross-correlation between '{signal1}' and '{signal2}'", Logger.DEBUG)
            results = calculate_cross_correlation_analysis(self, time_arr1, values1, time_arr2, values2)
            if results is None:
                Logger.log_message_static("Cross-correlation analysis aborted", Logger.INFO)
                return

            lags = results["Lags (s)"]
            corr = results["Cross-Correlation"]
            max_corr = results["Max Correlation"]
            max_lag = results["Lag at Max Correlation (s)"]

            # Plot setup
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Cross-Correlation: {signal1} & {signal2}")
            plot_window.resize(800, 600)

            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            plot_widget = pg.GraphicsLayoutWidget()

            # Cross-correlation plot
            p1 = plot_widget.addPlot(row=0, col=0)
            p1.setTitle("Cross-Correlation")
            p1.setLabel('left', 'Correlation')
            p1.setLabel('bottom', 'Lag (s)')
            p1.plot(lags, corr, pen='b')
            p1.addLine(y=0, pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
            p1.addLine(x=max_lag, pen=pg.mkPen('g', width=2))

            # Signals plot
            p2 = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
            p2.setTitle("Original Signals")
            p2.setLabel('left', 'Amplitude')
            p2.setLabel('bottom', 'Time (s)')
            p2.plot(time_arr1, values1, pen='b', name=signal1)
            p2.plot(time_arr2, values2, pen='r', name=signal2)
            plot_widget.addItem(p2, row=1, col=0)

            legend = p2.addLegend()
            legend.addItem(pg.PlotDataItem(pen='b'), signal1)
            legend.addItem(pg.PlotDataItem(pen='r'), signal2)

            if abs(max_lag) > 0.001:
                time_arr2_shifted = time_arr2 + max_lag
                valid_mask = (time_arr2_shifted >= min(time_arr1)) & (time_arr2_shifted <= max(time_arr1))
                if np.any(valid_mask):
                    p2.plot(time_arr2_shifted[valid_mask], values2[valid_mask],
                            pen=pg.mkPen('g', width=2, style=Qt.PenStyle.DashLine),
                            name=f"{signal2} (shifted)")
                    legend.addItem(pg.PlotDataItem(pen=pg.mkPen('g', width=2, style=Qt.PenStyle.DashLine)),
                                   f"{signal2} (shifted by {max_lag:.4f}s)")

            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)
            plot_window.show()

            self._plot_windows.append(plot_window)
            Logger.log_message_static("Cross-correlation plot window displayed successfully", Logger.INFO)

            # Prepare stats for display
            display_stats = {
                "Maximum Correlation": f"{results['Max Correlation']:.4f}",
                "Lag at Maximum Correlation": f"{results['Lag at Max Correlation (s)']:.6f} s",
                "Correlation at Zero Lag": f"{results['Correlation at Zero Lag']:.4f}",
            }

            for key in sorted(results):
                if key.startswith("Correlation Width"):
                    display_stats[key] = f"{results[key]} s" if isinstance(results[key], float) else results[key]

            self.show_analysis_results("Cross-Correlation Analysis", f"{signal1} & {signal2}", display_stats)
            Logger.log_message_static("Cross-correlation analysis complete", Logger.INFO)

        except KeyError as ke:
            Logger.log_message_static(f"Signal not found in data_signals: {str(ke)}", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in cross-correlation analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_wavelet_dialog(self):
        """
        Show wavelet analysis parameter dialog, run wavelet analysis (CWT or DWT), and display results.
        """
        Logger.log_message_static("Opening wavelet analysis dialog", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("No signal selected for wavelet analysis", Logger.WARNING)
            return

        try:
            time_arr, values = self.parent.data_signals[signal]
            processed = safe_prepare_signal(values, self, "Wavelet Analysis")
            if processed is None:
                return

            # --- Dialog to choose wavelet settings ---
            dialog = QDialog(self)
            dialog.setWindowTitle("Wavelet Analysis Settings")
            layout = QVBoxLayout(dialog)
            form = QFormLayout()

            wavelet_combo = QComboBox()

            result = self.result()
            if not result:
                QMessageBox.warning(self, "Wavelet Analysis", "Analysis failed or was cancelled.")
                return

            self.show_analysis_results("Wavelet Analysis", signal, result)

            if "Coefficients" in result:
                plot_window = QMainWindow(self)
                plot_window.setWindowTitle(f"CWT Spectrogram: {signal}")
                plot_window.resize(800, 600)

                central = QWidget()
                layout = QVBoxLayout(central)

                img_widget = pg.ImageView()
                coeffs = result["Coefficients"]
                time_arr = result["Time Array"]
                scales = result["Scales"]
                freqs = result["Frequencies (Hz)"]

                img_widget.setImage(np.abs(coeffs), xvals=time_arr)
                img_widget.ui.histogram.hide()
                img_widget.ui.roiBtn.hide()
                img_widget.ui.menuBtn.hide()
                img_widget.setPredefinedGradient("viridis")
                img_widget.setMinimumHeight(500)

                layout.addWidget(img_widget)

                close_btn = QPushButton("Close")
                close_btn.clicked.connect(plot_window.close)
                layout.addWidget(close_btn)

                central.setLayout(layout)
                plot_window.setCentralWidget(central)
                plot_window.show()

                self._plot_windows.append(plot_window)
                Logger.log_message_static("CWT image plot displayed", Logger.INFO)

            Logger.log_message_static("Wavelet analysis complete", Logger.INFO)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Wavelet analysis failed: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_filter_dialog(self):
        """
        Show filter configuration dialog and display filtered signal using IIR or FIR.
        """
        Logger.log_message_static("Opening filter dialog", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("No signal selected for filtering", Logger.WARNING)
            return

        try:
            time_arr, values = self.parent.data_signals[signal]
            sample_rate = safe_sample_rate(time_arr)
            if sample_rate == 0.0:
                QMessageBox.information(self, "Filter", "Sampling rate could not be determined.")
                return

            # --- Filter dialog setup ---
            dialog = QDialog(self)
            dialog.setWindowTitle("Filter Settings")
            layout = QVBoxLayout(dialog)
            form = QFormLayout()

            filter_type_box = QComboBox()
            filter_type_box.addItems(["lowpass", "highpass", "bandpass", "bandstop"])
            form.addRow("Filter Type:", filter_type_box)

            design_box = QComboBox()
            design_box.addItems(["IIR (Butterworth)", "FIR"])
            form.addRow("Filter Design:", design_box)

            cutoff1_spin = QDoubleSpinBox()
            cutoff1_spin.setRange(0.1, sample_rate / 2)
            cutoff1_spin.setValue(10.0)
            cutoff1_spin.setSuffix(" Hz")
            form.addRow("Cutoff Frequency:", cutoff1_spin)

            cutoff2_spin = QDoubleSpinBox()
            cutoff2_spin.setRange(0.1, sample_rate / 2)
            cutoff2_spin.setValue(100.0)
            cutoff2_spin.setSuffix(" Hz")
            cutoff2_spin.setVisible(False)
            form.addRow("Second Cutoff:", cutoff2_spin)

            order_spin = QSpinBox()
            order_spin.setRange(1, 10)
            order_spin.setValue(4)
            form.addRow("Order (IIR) / Taps (FIR):", order_spin)

            fir_window_box = QComboBox()
            fir_window_box.addItems(["hamming", "hann", "blackman", "bartlett", "boxcar"])
            form.addRow("FIR Window:", fir_window_box)

            layout.addLayout(form)

            # Update cutoff2 visibility
            def update_cutoffs():
                ftype = filter_type_box.currentText()
                cutoff2_spin.setVisible(ftype in ["bandpass", "bandstop"])

            filter_type_box.currentTextChanged.connect(update_cutoffs)

            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)

            if not dialog.exec():
                Logger.log_message_static("Filter dialog cancelled by user", Logger.INFO)
                return

            # --- Retrieve settings ---
            filter_type = filter_type_box.currentText()
            design_type = design_box.currentText()
            cutoff_freq = [cutoff1_spin.value(), cutoff2_spin.value()] if filter_type in ["bandpass",
                                                                                          "bandstop"] else cutoff1_spin.value()
            order = order_spin.value()
            fir_window = fir_window_box.currentText()

            Logger.log_message_static(
                f"Filter config: {filter_type}, {design_type}, cutoff={cutoff_freq}, order={order}", Logger.DEBUG)

            # --- Perform filtering ---
            result = self.apply_filter_and_return_result(
                time_arr, values,
                filter_type=filter_type,
                cutoff_freq=cutoff_freq,
                order=order,
                filter_method="fir" if "FIR" in design_type else "iir",
                numtaps=order,  # order is used for both taps/order
                window=fir_window
            )

            if not result:
                return

            filtered = result["Filtered Signal"]

            # --- Plot results ---
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"{design_type} Filtered Signal: {signal}")
            plot_window.resize(800, 600)
            central = QWidget()
            vbox = QVBoxLayout(central)
            widget = pg.GraphicsLayoutWidget()

            p1 = pg.PlotItem(axisItems={'bottom':pg.DateAxisItem(orientation='bottom')})
            p1.setTitle("Original Signal")
            p1.setLabel("bottom", "Time (s)")
            p1.plot(time_arr, values, pen='b')
            widget.addItem(p1, row=0, col=0)

            p2 = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
            p2.setTitle(f"Filtered Signal ({filter_type})")
            p2.setLabel("bottom", "Time (s)")
            p2.plot(time_arr, filtered, pen='g')
            widget.addItem(p2, row=1, col=0)

            p2.setXLink(p1)

            vbox.addWidget(widget)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(plot_window.close)
            vbox.addWidget(close_btn)

            central.setLayout(vbox)
            plot_window.setCentralWidget(central)
            plot_window.show()
            self._plot_windows.append(plot_window)

            Logger.log_message_static("Filter result displayed", Logger.INFO)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in filter dialog: {e}", Logger.ERROR)
            import traceback
            Logger.log_message_static(traceback.format_exc(), Logger.DEBUG)

    def apply_filter_and_return_result(dialog, time_arr, values, filter_type="lowpass", cutoff_freq=1.0,
                             order=4, filter_method="iir", numtaps=101, window="hamming"):
        """
        Apply a filter to the signal and return results for displaying in the UI.

        Args:
            dialog: The parent dialog for status messages
            time_arr (np.ndarray): Time array for the signal
            values (np.ndarray): Signal amplitude values
            filter_type (str): Filter type ("lowpass", "highpass", "bandpass", "bandstop")
            cutoff_freq (float or tuple): Cutoff frequency in Hz (tuple for bandpass/bandstop)
            order (int): Filter order (for IIR filters)
            filter_method (str): "iir" or "fir"
            numtaps (int): Number of taps for FIR filter
            window (str): Window function for FIR filter

        Returns:
            dict or None: Dictionary with filtered signal and parameters or None on error
        """
        from utils.logger import Logger

        try:
            Logger.log_message_static(f"Filter config: {filter_type}, {filter_method.upper()}"
                                      f"{' (Butterworth)' if filter_method == 'iir' else ''}, "
                                      f"cutoff={cutoff_freq}, "
                                      f"{'order' if filter_method == 'iir' else 'taps'}="
                                      f"{order if filter_method == 'iir' else numtaps}",
                                      Logger.DEBUG)

            # Prepare signal
            processed_values = safe_prepare_signal(values, dialog, "Filter Application")
            if processed_values is None:
                return None

            # Apply filter based on method
            if filter_method.lower() == "iir":
                result = calculate_iir_filter(processed_values, time_arr, filter_type, cutoff_freq, order)
            elif filter_method.lower() == "fir":
                result = calculate_fir_filter(processed_values, time_arr, filter_type, cutoff_freq, numtaps, window)
            else:
                Logger.log_message_static(f"Unknown filter method: {filter_method}", Logger.ERROR)
                return None

            if result is None:
                Logger.log_message_static("Filtering failed", Logger.ERROR)
                return None

            # Extract filtered signal for the result
            filtered = result.get("Filtered Signal")

            return {
                "Original Signal": values,
                "Filtered Signal": filtered,
                "Time Array": time_arr,
                "Filter Type": filter_type,
                "Filter Method": filter_method,
                "Cutoff Frequency": cutoff_freq,
                "Parameters": {
                    "Order" if filter_method == "iir" else "Taps": order if filter_method == "iir" else numtaps,
                    "Window": window if filter_method == "fir" else "N/A"
                }
            }

        except Exception as e:
            Logger.log_message_static(f"Error in filter application: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(traceback.format_exc(), Logger.DEBUG)
            return None

    def show_analysis_results(self, title, signal_name, data_dict):
        """Display analysis results in a table in the results area."""
        Logger.log_message_static(f"Displaying {title} results for {signal_name}", Logger.DEBUG)
        self.clear_results()

        # Create title
        result_title = QLabel(f"{title} Results: {signal_name}")
        result_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.results_layout.addWidget(result_title)
        Logger.log_message_static(f"Added results title: {title} for {signal_name}", Logger.DEBUG)

        # Create table
        table = QTableWidget()
        table.setColumnCount(2)
        table.setRowCount(len(data_dict))
        table.setHorizontalHeaderLabels(["Metric", "Value"])

        # Set column widths
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        Logger.log_message_static(f"Creating results table with {len(data_dict)} rows", Logger.DEBUG)

        # Populate table
        for i, (key, value) in enumerate(data_dict.items()):
            table.setItem(i, 0, QTableWidgetItem(key))
            table.setItem(i, 1, QTableWidgetItem(str(value)))
            Logger.log_message_static(f"Added table row: {key} = {value}", Logger.DEBUG)

        self.results_layout.addWidget(table)
        Logger.log_message_static("Analysis results displayed successfully", Logger.DEBUG)

    def show_help_in_results(self, topic, content):
        """Display help information in the results area."""
        Logger.log_message_static(f"Displaying help for: {topic}", Logger.DEBUG)
        self.clear_results()

        # Create title
        result_title = QLabel(f"Help: {topic}")
        result_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.results_layout.addWidget(result_title)

        # Create text display
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        text_widget.setHtml(content)
        text_widget.setMinimumHeight(300)
        self.results_layout.addWidget(text_widget)

        Logger.log_message_static(f"Help content for '{topic}' displayed successfully", Logger.DEBUG)


def show_analysis_dialog(parent):
    """
    Shows the signal analysis dialog with all the available analysis tools.

    This function serves as the main entry point for the signal analysis features.

    Args:
        parent: The parent application that has the data_signals attribute
               containing the signal data.
    """
    Logger.log_message_static("Opening Signal Analysis Dialog", Logger.INFO)
    try:
        dialog = SignalAnalysisDialog(parent)
        dialog.exec()
        Logger.log_message_static("Signal Analysis Dialog closed", Logger.INFO)
    except Exception as e:
        Logger.log_message_static(f"Error in signal analysis dialog: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Signal analysis dialog traceback: {traceback.format_exc()}", Logger.DEBUG)

def add_explanation_group(layout, title, text):
    """
    Add an expandable group with explanation text.

    Args:
        layout (QLayout): Layout to add the group to
        title (str): Title of the explanation group
        text (str): Explanation text
    """
    Logger.log_message_static(f"Adding explanation group: {title}", Logger.DEBUG)
    group = QGroupBox(title)
    group.setCheckable(True)
    group.setChecked(False)  # Start collapsed

    group_layout = QVBoxLayout(group)
    text_edit = QTextEdit()
    text_edit.setReadOnly(True)
    text_edit.setText(text)
    text_edit.setMinimumHeight(100)
    group_layout.addWidget(text_edit)

    layout.addWidget(group)
    Logger.log_message_static(f"Added explanation group for: {title}", Logger.DEBUG)



