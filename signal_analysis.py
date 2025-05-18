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
import pyqtgraph as pg
import pywt
import scipy.signal as sc_signal
import scipy.stats as sc_stats
from scipy.signal import hilbert, butter, filtfilt, correlate
from PySide6.QtCore import Qt
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import (QDialog, QPushButton, QTableWidget, QTableWidgetItem, QDialogButtonBox,
                               QHeaderView, QComboBox, QWidget, QHBoxLayout, QGroupBox,
                               QTabWidget, QFormLayout, QDoubleSpinBox, QTextEdit, QSplitter,
                               QVBoxLayout, QGridLayout, QScrollArea, QMainWindow, QLabel)
from logger import Logger

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
        stats_btn.clicked.connect(self.show_statistics)
        button_layout.addWidget(stats_btn)

        fft_btn = QPushButton("FFT Analysis")
        fft_btn.setToolTip("Analyze the frequency spectrum of the signal.")
        fft_btn.clicked.connect(self.show_fft)
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
        autocorr_btn.clicked.connect(self.show_autocorrelation)
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
        hilbert_btn.clicked.connect(self.show_hilbert_transform)
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
        xcorr_btn.clicked.connect(self.show_cross_correlation)
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

    def show_statistics(self):
        """Calculate and display basic statistics for the selected signal."""
        Logger.log_message_static("Calculating basic statistics", Logger.INFO)
        signal = self.get_selected_signal()
        if not signal:
            Logger.log_message_static("Cannot calculate statistics: No signal selected", Logger.WARNING)
            return

        Logger.log_message_static(f"Computing statistics for signal '{signal}'", Logger.DEBUG)
        try:
            # Get the signal values directly
            _, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for statistics", Logger.DEBUG)

            # Calculate statistics directly on the original values
            stats = {
                "Mean": np.mean(values),
                "Median": np.median(values),
                "Standard Deviation": np.std(values),
                "Variance": np.var(values),
                "Min": np.min(values),
                "Max": np.max(values),
                "Range": np.max(values) - np.min(values),
                "RMS": np.sqrt(np.mean(values ** 2)),
                "Skewness": sc_stats.skew(values),
                "Kurtosis": sc_stats.kurtosis(values)
            }

            Logger.log_message_static("Successfully calculated basic statistics", Logger.DEBUG)
            self.show_analysis_results("Statistics", signal, stats)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error calculating statistics: {str(e)}", Logger.ERROR)

    def show_fft(self):
        """Perform FFT analysis on the selected signal and display in a new window."""
        Logger.log_message_static("Preparing FFT analysis", Logger.INFO)
        signal = self.get_selected_signal()
        if not signal:
            Logger.log_message_static("Cannot perform FFT: No signal selected", Logger.WARNING)
            return

        try:
            # Get the source signal data
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for FFT analysis", Logger.DEBUG)

            # Process the signal values directly
            Logger.log_message_static("Preparing signal for FFT processing", Logger.DEBUG)
            processed_values = prepare_signal_for_analysis(self, values, "FFT Input Signal")
            if processed_values is None:
                Logger.log_message_static("FFT analysis canceled by user", Logger.INFO)
                return  # User canceled the operation

            # Perform FFT analysis
            Logger.log_message_static("Calculating sampling frequency", Logger.DEBUG)
            if len(time_arr) < 2:
                Logger.log_message_static("Time array has insufficient points for frequency calculation", Logger.WARNING)
                fs = 1.0  # Default value
            else:
                fs = 1 / np.mean(np.diff(time_arr))  # Sampling frequency

            Logger.log_message_static(f"Calculated sampling frequency: {fs} Hz", Logger.DEBUG)
            n = len(processed_values)

            Logger.log_message_static("Performing FFT calculation", Logger.DEBUG)
            fft_values = np.fft.rfft(processed_values)
            freqs = np.fft.rfftfreq(n, d=1 / fs)
            magnitudes = np.abs(fft_values) / n * 2  # Scale appropriately
            Logger.log_message_static(f"FFT calculation complete, {len(freqs)} frequency points generated", Logger.DEBUG)

            # Create a proper window using QMainWindow
            Logger.log_message_static("Creating FFT plot window", Logger.DEBUG)
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"FFT Analysis: {signal}")
            plot_window.resize(800, 600)

            # Create central widget with layout
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            # Create plot widget
            plot_widget = pg.GraphicsLayoutWidget()

            # Time domain plot
            p1 = plot_widget.addPlot(row=0, col=0)
            p1.setTitle("Time Domain")
            p1.setLabel('left', 'Amplitude')
            p1.setLabel('bottom', 'Time (s)')
            p1.plot(time_arr, processed_values, pen='b')

            # Frequency domain plot
            p2 = plot_widget.addPlot(row=1, col=0)
            p2.setTitle("Frequency Domain")
            p2.setLabel('left', 'Magnitude')
            p2.setLabel('bottom', 'Frequency (Hz)')
            p2.plot(freqs, magnitudes, pen='r')
            p2.setLogMode(x=True, y=False)  # Set log scale for better visualization

            # Add close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            # Set up layout
            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)

            # Show the window
            plot_window.show()
            Logger.log_message_static("FFT plot window displayed successfully", Logger.INFO)

            # Keep reference to prevent garbage collection
            self._plot_windows.append(plot_window)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in FFT analysis: {str(e)}", Logger.ERROR)

    def show_time_analysis(self):
        """Perform time-domain analysis on the selected signal and display results."""
        Logger.log_message_static("Performing time-domain analysis", Logger.INFO)
        signal = self.get_selected_signal()
        if not signal:
            Logger.log_message_static("Cannot perform time analysis: No signal selected", Logger.WARNING)
            return

        try:
            # Get the source signal data directly
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for time analysis", Logger.DEBUG)

            # Time domain analysis
            if len(time_arr) < 2:
                Logger.log_message_static("Time array has insufficient points", Logger.WARNING)
                duration = 0
                sample_rate = 0
            else:
                duration = time_arr[-1] - time_arr[0]
                sample_rate = len(values) / duration

            Logger.log_message_static(f"Signal duration: {duration}s, Sample rate: {sample_rate}Hz", Logger.DEBUG)

            # Calculate zero crossings
            Logger.log_message_static("Calculating zero crossings", Logger.DEBUG)
            zero_crossings = np.sum(np.diff(np.signbit(values).astype(int)) != 0)

            # Calculate signal energy and power
            Logger.log_message_static("Calculating signal energy and power", Logger.DEBUG)
            energy = np.sum(values ** 2)
            power = energy / len(values)

            analysis = {
                "Duration (s)": duration,
                "Samples": len(values),
                "Sample Rate (Hz)": sample_rate,
                "Zero Crossings": zero_crossings,
                "Mean Amplitude": np.mean(values),
                "Peak Amplitude": np.max(np.abs(values)),
                "Energy": energy,
                "Power": power,
                "Crest Factor": np.max(np.abs(values)) / np.sqrt(np.mean(values ** 2))
            }

            Logger.log_message_static("Time-domain analysis complete", Logger.DEBUG)
            self.show_analysis_results("Time Analysis", signal, analysis)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in time-domain analysis: {str(e)}", Logger.ERROR)

    def show_psd_analysis(self):
        """Calculate and display Power Spectral Density for the selected signal."""
        Logger.log_message_static("Preparing PSD analysis", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("Cannot perform PSD analysis: No signal selected", Logger.WARNING)
            return

        try:
            # Get the source signal data
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for PSD analysis", Logger.DEBUG)

            # Calculate sampling frequency directly
            fs = 1 / np.mean(np.diff(time_arr)) if len(time_arr) > 1 else 1
            Logger.log_message_static(f"Using sampling frequency of {fs} Hz for PSD calculation", Logger.DEBUG)

            # Remove DC offset (mean) without using the prepare function
            signal_mean = np.mean(values)
            detrended_values = values - signal_mean
            Logger.log_message_static(f"Removed DC offset: {signal_mean}", Logger.DEBUG)

            # Apply window to reduce spectral leakage
            window = np.hanning(len(detrended_values))
            windowed_values = detrended_values * window

            # Compute PSD using Welch's method
            Logger.log_message_static("Computing Welch PSD", Logger.DEBUG)
            # Set nperseg to a power of 2 but not more than 1/4 of signal length
            nperseg = min(256, len(windowed_values) // 4)
            nperseg = 2 ** int(np.log2(nperseg))  # Ensure power of 2

            freqs, psd = sc_signal.welch(
                windowed_values,
                fs=fs,
                nperseg=nperseg,
                scaling='density'
            )

            Logger.log_message_static(f"PSD calculation complete, {len(freqs)} frequency bins generated", Logger.DEBUG)

            # Replace any zero values with small number to avoid log(0)
            psd = np.where(psd <= 0, 1e-10, psd)

            # Calculate PSD statistics
            peak_idx = np.argmax(psd)
            peak_freq = freqs[peak_idx]
            max_power_db = 10 * np.log10(np.max(psd))
            total_power = np.sum(psd)

            Logger.log_message_static(f"PSD peak frequency: {peak_freq} Hz, max power: {max_power_db} dB", Logger.DEBUG)

            psd_stats = {
                "Peak Frequency (Hz)": peak_freq,
                "Max Power (dB)": max_power_db,
                "Total Power": total_power,
                "RMS Power": np.sqrt(total_power),
                "Bandwidth (3dB)": self._calculate_bandwidth(freqs, psd)
            }

            # Create a proper window using QMainWindow
            Logger.log_message_static("Creating PSD plot window", Logger.DEBUG)
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Power Spectral Density: {signal}")
            plot_window.resize(800, 600)

            # Create central widget with layout
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            # Create plot widget
            plot_widget = pg.GraphicsLayoutWidget()

            # PSD plot
            p = plot_widget.addPlot()
            p.setTitle("Power Spectral Density")
            p.setLabel('left', 'Power/Frequency (dB/Hz)')
            p.setLabel('bottom', 'Frequency (Hz)')

            # Convert to dB for plotting
            psd_db = 10 * np.log10(psd)
            p.plot(freqs, psd_db, pen='g')

            # Set log scale for x-axis
            p.setLogMode(x=True, y=False)

            # Add reference line at peak frequency
            peak_line = pg.InfiniteLine(pos=peak_freq, angle=90, pen='r')
            p.addItem(peak_line)

            # Add close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            # Set up layout
            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)

            # Show the window
            plot_window.show()
            Logger.log_message_static("PSD plot window displayed successfully", Logger.INFO)

            # Keep reference to prevent garbage collection
            self._plot_windows.append(plot_window)

            # Show PSD statistics
            self.show_analysis_results("PSD Analysis", signal, psd_stats)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in PSD analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"PSD analysis traceback: {traceback.format_exc()}", Logger.DEBUG)

    def _calculate_bandwidth(self, freqs, psd):
        """Calculate the 3dB bandwidth of a PSD."""
        max_psd = np.max(psd)
        half_power = max_psd / 2  # -3dB point

        # Find indices where PSD is above half power
        above_half_power = psd >= half_power

        if not np.any(above_half_power):
            return 0

        # Find the first and last indices where PSD is above half power
        indices = np.where(above_half_power)[0]
        if len(indices) <= 1:
            return 0

        lower_freq = freqs[indices[0]]
        upper_freq = freqs[indices[-1]]

        return upper_freq - lower_freq

    def show_autocorrelation(self):
        """Calculate and display the autocorrelation of the selected signal."""
        Logger.log_message_static("Preparing autocorrelation analysis", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("Cannot perform autocorrelation: No signal selected", Logger.WARNING)
            return

        try:
            # Get the source signal data
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for autocorrelation", Logger.DEBUG)

            # Handle the signal directly without prepare_signal_for_analysis

            # Remove mean (DC offset)
            detrended_values = values - np.mean(values)
            Logger.log_message_static("Removed DC offset from signal", Logger.DEBUG)

            # Apply windowing to reduce edge effects
            window = np.hanning(len(detrended_values))
            windowed_values = detrended_values * window
            Logger.log_message_static("Applied Hanning window to signal", Logger.DEBUG)

            # Calculate autocorrelation directly
            Logger.log_message_static("Computing autocorrelation", Logger.DEBUG)
            autocorr = correlate(windowed_values, windowed_values, mode='full')

            # Normalize
            Logger.log_message_static("Normalizing autocorrelation", Logger.DEBUG)
            autocorr = autocorr / np.max(autocorr)

            # Keep only the second half (positive lags)
            center = len(autocorr) // 2
            autocorr = autocorr[center:]
            Logger.log_message_static(f"Extracted {len(autocorr)} positive lag points", Logger.DEBUG)

            # Create lag axis
            lags = np.arange(len(autocorr))
            if len(time_arr) > 1:
                # Convert lags to time units
                dt = np.mean(np.diff(time_arr))
                lags = lags * dt
                Logger.log_message_static(f"Converted lags to time units with dt={dt}", Logger.DEBUG)

            # Create a proper window
            Logger.log_message_static("Creating autocorrelation plot window", Logger.DEBUG)
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Autocorrelation: {signal}")
            plot_window.resize(800, 600)

            # Create central widget with layout
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            # Create plot widget
            plot_widget = pg.GraphicsLayoutWidget()

            # Autocorrelation plot
            p = plot_widget.addPlot()
            p.setTitle("Autocorrelation")
            p.setLabel('left', 'Correlation')
            p.setLabel('bottom', 'Lag' + (' (s)' if len(time_arr) > 1 else ''))
            p.plot(lags, autocorr, pen='b')

            # Add close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            # Set up layout
            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)

            # Show the window
            plot_window.show()
            Logger.log_message_static("Autocorrelation plot window displayed successfully", Logger.INFO)

            # Keep reference to prevent garbage collection
            self._plot_windows.append(plot_window)

            # Add analysis results
            first_minimum_idx = None
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] < autocorr[i - 1] and autocorr[i] < autocorr[i + 1]:
                    first_minimum_idx = i
                    break

            first_zero_idx = np.where(np.diff(np.signbit(autocorr)))[0]
            first_zero_idx = first_zero_idx[0] if len(first_zero_idx) > 0 else None

            autocorr_results = {
                "Peak Correlation": 1.0,  # Always 1.0 after normalization
                "First Minimum": lags[first_minimum_idx] if first_minimum_idx else "Not found",
                "First Zero Crossing": lags[first_zero_idx] if first_zero_idx else "Not found",
                "Decorrelation Time": lags[first_zero_idx] if first_zero_idx else "Not found"
            }

            self.show_analysis_results("Autocorrelation Analysis", signal, autocorr_results)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in autocorrelation analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Autocorrelation traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_peak_detection(self):
        """Detect and analyze peaks in the selected signal. For predominantly negative signals,
            negative peaks (valleys) are detected and reported instead of positive peaks.
        """
        Logger.log_message_static("Preparing peak detection analysis", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("Cannot perform peak detection: No signal selected", Logger.WARNING)
            return

        try:
            # Get the source signal data directly
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for peak detection", Logger.DEBUG)

            # Determine if we should look for positive or negative peaks
            signal_mean = np.mean(values)
            Logger.log_message_static(f"Signal mean value: {signal_mean}", Logger.DEBUG)

            if np.all(values < 0) or (
                    np.any(values < 0) and abs(np.min(values) - signal_mean) > abs(np.max(values) - signal_mean)):
                # For predominantly negative signals, look for negative peaks (valleys)
                # We invert the signal to use the same peak finding algorithm
                Logger.log_message_static("Signal is predominantly negative, inverting to detect valleys", Logger.DEBUG)
                processed_values = -values
                peak_type = "negative"
            else:
                # For positive or mixed signals, look for positive peaks
                Logger.log_message_static("Signal is predominantly positive or mixed, detecting peaks", Logger.DEBUG)
                processed_values = values
                peak_type = "positive"

            # Use default parameters for peak detection
            Logger.log_message_static("Finding peaks with default parameters", Logger.DEBUG)
            peaks, properties = sc_signal.find_peaks(processed_values, height=None, distance=None)
            Logger.log_message_static(f"Found {len(peaks)} peaks in signal", Logger.DEBUG)

            if len(peaks) == 0:
                Logger.log_message_static("No peaks detected in signal", Logger.WARNING)
                self.show_analysis_results("Peak Detection", signal, {"Result": "No peaks detected in this signal"})
                return

            # Calculate additional peak properties
            Logger.log_message_static("Calculating peak heights", Logger.DEBUG)
            peak_heights = processed_values[peaks]
            if peak_type == "negative":
                # Convert heights back to negative for display
                Logger.log_message_static("Converting heights back to negative values", Logger.DEBUG)
                peak_heights = -peak_heights

            Logger.log_message_static("Calculating peak widths", Logger.DEBUG)
            peak_widths = sc_signal.peak_widths(processed_values, peaks, rel_height=0.5)[0]

            peaks_data = {
                "Peak Type": "Negative" if peak_type == "negative" else "Positive",
                "Count": len(peaks),
                "Indices": peaks,
                "Times": time_arr[peaks],
                "Heights": peak_heights,
                "Widths": peak_widths,
                "Mean Height": np.mean(peak_heights) if len(peak_heights) > 0 else 0,
                "Max Height": np.max(peak_heights) if len(peak_heights) > 0 else 0,
                "Min Height": np.min(peak_heights) if len(peak_heights) > 0 else 0,
                "Mean Width": np.mean(peak_widths) if len(peak_widths) > 0 else 0,
            }
            Logger.log_message_static(f"Peak detection summary: {len(peaks)} peaks, type={peak_type}", Logger.DEBUG)

            # Create a proper window for visualization
            Logger.log_message_static("Creating peak detection plot window", Logger.DEBUG)
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Peak Detection: {signal}")
            plot_window.resize(800, 600)

            # Create central widget with layout
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            # Create plot widget
            plot_widget = pg.GraphicsLayoutWidget()

            # Signal with peak markers plot
            p = plot_widget.addPlot()
            p.setTitle(f"{'Negative' if peak_type == 'negative' else 'Positive'} Peak Detection")
            p.setLabel('left', 'Amplitude')
            p.setLabel('bottom', 'Time')

            # Plot original signal
            Logger.log_message_static("Plotting original signal", Logger.DEBUG)
            p.plot(time_arr, values, pen='b')

            # Add peak markers
            Logger.log_message_static("Adding peak markers to plot", Logger.DEBUG)
            scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen('r', width=2), brush=pg.mkBrush('r'))
            spots = [{'pos': (time_arr[idx], values[idx]), 'data': idx} for idx in peaks]
            scatter.addPoints(spots)
            p.addItem(scatter)

            # Add close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            # Set up layout
            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)

            # Show the window
            plot_window.show()
            Logger.log_message_static("Peak detection plot window displayed successfully", Logger.INFO)

            # Keep reference to prevent garbage collection
            self._plot_windows.append(plot_window)

            # Format peak data for display
            Logger.log_message_static("Formatting peak data for display", Logger.DEBUG)
            display_data = {}
            for key, value in peaks_data.items():
                if key in ["Count", "Peak Type"]:
                    display_data[key] = value
                elif key == "Indices":
                    display_data[key] = str(value[:5].tolist()) + (" ..." if len(value) > 5 else "")
                elif key == "Times":
                    display_data[key] = str([round(x, 3) for x in value[:5].tolist()]) + (
                        " ..." if len(value) > 5 else "")
                elif key == "Heights":
                    display_data[key] = str([round(x, 3) for x in value[:5].tolist()]) + (
                        " ..." if len(value) > 5 else "")
                elif key == "Widths":
                    display_data[key] = str([round(x, 3) for x in value[:5].tolist()]) + (
                        " ..." if len(value) > 5 else "")
                else:
                    display_data[key] = round(value, 3) if isinstance(value, (int, float)) else value

            self.show_analysis_results("Peak Detection", signal, display_data)
            Logger.log_message_static("Peak detection analysis complete", Logger.INFO)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in peak detection: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Peak detection traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_filter_dialog(self):
        """Show dialog for filtering signal and display the filtered result."""
        Logger.log_message_static("Opening filter dialog", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("Cannot show filter dialog: No signal selected", Logger.WARNING)
            return

        try:
            Logger.log_message_static(f"Initializing filter dialog for signal '{signal}'", Logger.DEBUG)
            # Create filter settings in a groupbox
            filter_dialog = QDialog(self)
            filter_dialog.setWindowTitle("Filter Settings")
            filter_dialog.resize(400, 300)

            layout = QVBoxLayout(filter_dialog)

            form = QFormLayout()

            # Filter type selector
            filter_type = QComboBox()
            filter_type.addItems(["lowpass", "highpass", "bandpass", "bandstop"])
            form.addRow("Filter Type:", filter_type)
            Logger.log_message_static("Added filter type selector with options", Logger.DEBUG)

            # Filter order
            order = QDoubleSpinBox()
            order.setDecimals(0)
            order.setRange(1, 10)
            order.setValue(4)
            form.addRow("Filter Order:", order)

            # Cutoff frequency
            cutoff = QDoubleSpinBox()
            cutoff.setRange(0.1, 1000)
            cutoff.setValue(10)
            cutoff.setSuffix(" Hz")
            form.addRow("Cutoff Frequency:", cutoff)
            Logger.log_message_static("Added cutoff frequency control", Logger.DEBUG)

            # For bandpass/bandstop, add second cutoff
            cutoff2 = QDoubleSpinBox()
            cutoff2.setRange(0.1, 1000)
            cutoff2.setValue(100)
            cutoff2.setSuffix(" Hz")
            cutoff2.setVisible(False)
            form.addRow("Second Cutoff:", cutoff2)

            # Connect filter type changes to showing/hiding second cutoff
            def update_cutoff_visibility():
                Logger.log_message_static(f"Filter type changed to: {filter_type.currentText()}", Logger.DEBUG)
                if filter_type.currentText() in ["bandpass", "bandstop"]:
                    cutoff2.setVisible(True)
                    cutoff.setPrefix("Lower ")
                    cutoff2.setPrefix("Upper ")
                else:
                    cutoff2.setVisible(False)
                    cutoff.setPrefix("")

            filter_type.currentTextChanged.connect(update_cutoff_visibility)

            layout.addLayout(form)

            # Buttons
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(filter_dialog.accept)
            buttons.rejected.connect(filter_dialog.reject)
            layout.addWidget(buttons)

            # Show the dialog
            Logger.log_message_static("Displaying filter dialog", Logger.DEBUG)
            if not filter_dialog.exec():
                Logger.log_message_static("Filter dialog canceled by user", Logger.INFO)
                return  # User canceled

            # Get filter parameters
            Logger.log_message_static("User accepted filter dialog, retrieving filter parameters", Logger.DEBUG)
            filter_params = {
                "type": filter_type.currentText(),
                "order": int(order.value()),
                "cutoff": cutoff.value(),
                "cutoff2": cutoff2.value() if filter_type.currentText() in ["bandpass", "bandstop"] else None
            }
            Logger.log_message_static(f"Filter parameters: {filter_params}", Logger.DEBUG)

            # Get signal data
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for filtering", Logger.DEBUG)

            # Calculate sample rate
            if len(time_arr) < 2:
                Logger.log_message_static("Signal too short to determine sample rate", Logger.WARNING)
                return
            fs = 1 / np.mean(np.diff(time_arr))
            Logger.log_message_static(f"Calculated sample rate: {fs} Hz", Logger.DEBUG)

            # Design the filter
            Logger.log_message_static("Designing Butterworth filter", Logger.DEBUG)
            if filter_params["type"] in ["bandpass", "bandstop"]:
                # For bandpass and bandstop, need array of cutoff frequencies
                Wn = [filter_params["cutoff"] / (abs(fs) / 2), filter_params["cutoff2"] / (abs(fs) / 2)]
                if Wn[0] >= Wn[1]:
                    Logger.log_message_static("Lower cutoff must be less than upper cutoff", Logger.WARNING)
                    return
                Logger.log_message_static(f"Using normalized cutoff frequencies: {Wn}", Logger.DEBUG)
            else:
                # For lowpass and highpass, single cutoff frequency
                Wn = filter_params["cutoff"] / (abs(fs) / 2)
                Logger.log_message_static(f"Using normalized cutoff frequency: {Wn}", Logger.DEBUG)

            try:
                b, a = butter(filter_params["order"], Wn, btype=filter_params["type"])
                Logger.log_message_static("Filter design successful", Logger.DEBUG)
            except Exception as e:
                Logger.log_message_static(f"Error designing filter: {str(e)}", Logger.ERROR)
                return

            # Apply the filter
            Logger.log_message_static("Applying filter to signal", Logger.DEBUG)
            try:
                filtered_values = filtfilt(b, a, values)
                Logger.log_message_static("Filter application successful", Logger.DEBUG)
            except Exception as e:
                Logger.log_message_static(f"Error applying filter: {str(e)}", Logger.ERROR)
                return

            # Create a proper window for visualization
            Logger.log_message_static("Creating filter result plot window", Logger.DEBUG)
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Filtered Signal: {signal}")
            plot_window.resize(800, 600)

            # Create central widget with layout
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            # Create plot widget
            plot_widget = pg.GraphicsLayoutWidget()

            # Original signal plot
            p1 = plot_widget.addPlot(row=0, col=0)
            p1.setTitle("Original Signal")
            p1.setLabel('left', 'Amplitude')
            p1.setLabel('bottom', 'Time')
            p1.plot(time_arr, values, pen='b')

            # Filtered signal plot
            p2 = plot_widget.addPlot(row=1, col=0)
            p2.setTitle(f"Filtered Signal ({filter_params['type']})")
            p2.setLabel('left', 'Amplitude')
            p2.setLabel('bottom', 'Time')
            p2.plot(time_arr, filtered_values, pen='g')

            # Link X axes for synchronized zooming
            p1.setXLink(p2)

            # Add close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            # Set up layout
            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)

            # Show the window
            plot_window.show()
            Logger.log_message_static("Filter result plot window displayed successfully", Logger.INFO)

            # Keep reference to prevent garbage collection
            self._plot_windows.append(plot_window)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in filter dialog: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Filter dialog traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_hilbert_transform(self):
        """Perform Hilbert transform to extract amplitude, phase, and frequency information."""
        Logger.log_message_static("Preparing Hilbert transform analysis", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("Cannot perform Hilbert transform: No signal selected", Logger.WARNING)
            return

        try:
            # Get the source signal data
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for Hilbert transform", Logger.DEBUG)

            # Process the signal values
            Logger.log_message_static("Preparing signal for Hilbert transform", Logger.DEBUG)
            processed_values = prepare_signal_for_analysis(self, values, "Hilbert Transform Input Signal")
            if processed_values is None:
                Logger.log_message_static("Hilbert transform canceled by user", Logger.INFO)
                return  # User canceled the operation

            # Perform Hilbert transform
            Logger.log_message_static("Computing Hilbert transform", Logger.DEBUG)
            analytic_signal = hilbert(processed_values)

            # Extract attributes
            Logger.log_message_static("Extracting amplitude envelope", Logger.DEBUG)
            amplitude_envelope = np.abs(analytic_signal)

            Logger.log_message_static("Extracting instantaneous phase", Logger.DEBUG)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))

            # Calculate sample rate for frequency calculation
            if len(time_arr) < 2:
                Logger.log_message_static("Signal too short to determine sample rate", Logger.WARNING)
                return

            Logger.log_message_static("Calculating sample rate for instantaneous frequency", Logger.DEBUG)
            fs = 1 / np.mean(np.diff(time_arr))
            Logger.log_message_static(f"Sample rate: {fs} Hz", Logger.DEBUG)

            Logger.log_message_static("Computing instantaneous frequency", Logger.DEBUG)
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
            # Pad with the last value to match original length
            instantaneous_frequency = np.append(instantaneous_frequency, instantaneous_frequency[-1])

            # Create a proper window
            Logger.log_message_static("Creating Hilbert transform plot window", Logger.DEBUG)
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Hilbert Transform: {signal}")
            plot_window.resize(800, 800)

            # Create central widget with layout
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            # Create plot widget with 4 subplots
            plot_widget = pg.GraphicsLayoutWidget()

            # 1. Original signal plot
            p1 = plot_widget.addPlot(row=0, col=0)
            p1.setTitle("Original Signal")
            p1.setLabel('left', 'Amplitude')
            p1.setLabel('bottom', 'Time')
            p1.plot(time_arr, processed_values, pen='b')

            # 2. Amplitude envelope
            p2 = plot_widget.addPlot(row=1, col=0)
            p2.setTitle("Amplitude Envelope")
            p2.setLabel('left', 'Amplitude')
            p2.setLabel('bottom', 'Time')
            p2.plot(time_arr, amplitude_envelope, pen='r')
            # Add original signal for comparison
            p2.plot(time_arr, processed_values, pen=pg.mkPen('b', width=1, style=Qt.PenStyle.DotLine))

            # 3. Instantaneous phase
            p3 = plot_widget.addPlot(row=2, col=0)
            p3.setTitle("Instantaneous Phase")
            p3.setLabel('left', 'Phase (rad)')
            p3.setLabel('bottom', 'Time')
            p3.plot(time_arr, instantaneous_phase, pen='g')

            # 4. Instantaneous frequency
            p4 = plot_widget.addPlot(row=3, col=0)
            p4.setTitle("Instantaneous Frequency")
            p4.setLabel('left', 'Frequency (Hz)')
            p4.setLabel('bottom', 'Time')
            p4.plot(time_arr, instantaneous_frequency, pen='m')

            # Link X axes
            p2.setXLink(p1)
            p3.setXLink(p1)
            p4.setXLink(p1)

            # Add close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            # Set up layout
            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)

            # Show the window
            plot_window.show()
            Logger.log_message_static("Hilbert transform plot window displayed successfully", Logger.INFO)

            # Keep reference to prevent garbage collection
            self._plot_windows.append(plot_window)

            # Calculate some summary statistics
            Logger.log_message_static("Calculating summary statistics for Hilbert transform", Logger.DEBUG)
            hilbert_stats = {
                "Mean Amplitude": np.mean(amplitude_envelope),
                "Max Amplitude": np.max(amplitude_envelope),
                "Mean Frequency": np.mean(instantaneous_frequency),
                "Median Frequency": np.median(instantaneous_frequency),
                "Max Frequency": np.max(instantaneous_frequency),
                "Phase Range": np.ptp(instantaneous_phase)
            }

            self.show_analysis_results("Hilbert Transform", signal, hilbert_stats)
            Logger.log_message_static("Hilbert transform analysis complete", Logger.INFO)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in Hilbert transform: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Hilbert transform traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_energy_analysis(self):
        """Analyze and visualize the energy distribution of the signal."""
        Logger.log_message_static("Preparing energy analysis", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("Cannot perform energy analysis: No signal selected", Logger.WARNING)
            return

        try:
            # Get the source signal data
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for energy analysis", Logger.DEBUG)

            # Check if we have enough data points
            if len(values) < 10:
                Logger.log_message_static("Signal too short for meaningful energy analysis", Logger.WARNING)
                self.show_analysis_results("Energy Analysis", signal, {"Result": "Signal too short for analysis"})
                return

            # Calculate sampling frequency
            fs = 1 / np.mean(np.diff(time_arr)) if len(time_arr) > 1 else 1
            Logger.log_message_static(f"Using sampling frequency of {fs} Hz for energy calculations", Logger.DEBUG)

            # Calculate time domain energy directly (squared sum)
            energy_time = np.sum(values ** 2)
            Logger.log_message_static(f"Time domain energy: {energy_time:.6e}", Logger.DEBUG)

            # For frequency domain, we'll remove the mean first (DC component)
            detrended_values = values - np.mean(values)

            # Apply window to reduce spectral leakage
            window = np.hanning(len(detrended_values))
            windowed_values = detrended_values * window

            # Calculate energy in frequency domain
            # First, compute FFT
            fft_values = np.fft.rfft(windowed_values)
            freqs = np.fft.rfftfreq(len(windowed_values), d=1 / fs)

            # Energy in frequency domain (Parseval's theorem)
            # Normalize by the window energy and length
            window_energy = np.sum(window ** 2)
            energy_freq = np.sum(np.abs(fft_values) ** 2) / (len(values) ** 2) * window_energy
            Logger.log_message_static(f"Frequency domain energy: {energy_freq:.6e}", Logger.DEBUG)

            # Calculate energy density spectrum (power spectrum)
            energy_density = np.abs(fft_values) ** 2 / len(values)

            # Create energy bands for analysis (divide spectrum into 5 bands)
            num_bands = 5
            band_edges = np.logspace(np.log10(freqs[1] if len(freqs) > 1 else 0.1),
                                     np.log10(freqs[-1]),
                                     num_bands + 1)

            # Calculate energy in each band
            band_energies = []
            for i in range(num_bands):
                lower = band_edges[i]
                upper = band_edges[i + 1]

                # Find indices within this band
                band_indices = (freqs >= lower) & (freqs <= upper)

                # Sum energy in this band
                band_energy = np.sum(energy_density[band_indices])
                band_energies.append(band_energy)

                Logger.log_message_static(
                    f"Band {i + 1} ({lower:.2f}-{upper:.2f} Hz): Energy = {band_energy:.6e}",
                    Logger.DEBUG
                )

            # Find dominant frequency band (maximum energy)
            max_band_idx = np.argmax(band_energies)
            max_band_lower = band_edges[max_band_idx]
            max_band_upper = band_edges[max_band_idx + 1]

            # Calculate total energy and percentages in each band
            total_band_energy = np.sum(band_energies)
            band_percentages = [100 * e / total_band_energy if total_band_energy > 0 else 0 for e in band_energies]

            # Create results dictionary
            energy_results = {
                "Total Energy (Time Domain)": energy_time,
                "Total Energy (Frequency Domain)": energy_freq,
                "Signal Power": energy_time / len(values),
                "RMS Value": np.sqrt(np.mean(values ** 2)),
                f"Dominant Frequency Band": f"{max_band_lower:.2f}-{max_band_upper:.2f} Hz",
                f"Dominant Band Energy": f"{band_energies[max_band_idx]:.6e} ({band_percentages[max_band_idx]:.1f}%)"
            }

            # Add band energy percentages
            for i in range(num_bands):
                energy_results[f"Band {i + 1} ({band_edges[i]:.2f}-{band_edges[i + 1]:.2f} Hz)"] = \
                    f"{band_percentages[i]:.1f}% of total"

            # Create energy plot window
            Logger.log_message_static("Creating energy analysis plot window", Logger.DEBUG)
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Energy Analysis: {signal}")
            plot_window.resize(800, 600)

            # Create central widget with layout
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            # Create plot widget
            plot_widget = pg.GraphicsLayoutWidget()

            # Energy density spectrum
            p1 = plot_widget.addPlot(row=0, col=0)
            p1.setTitle("Energy Density Spectrum")
            p1.setLabel('left', 'Energy Density')
            p1.setLabel('bottom', 'Frequency (Hz)')
            p1.plot(freqs, energy_density, pen='r')
            p1.setLogMode(x=True, y=True)  # Log-log scale

            # Band energy distribution
            p2 = plot_widget.addPlot(row=1, col=0)
            p2.setTitle("Energy Distribution by Frequency Band")
            p2.setLabel('left', 'Energy Percentage (%)')
            p2.setLabel('bottom', 'Band')

            # Create bar chart for band energies
            x = np.arange(num_bands)
            bar_graph = pg.BarGraphItem(x=x, height=band_percentages, width=0.6, brush='b')
            p2.addItem(bar_graph)

            # Set x-ticks to show frequency bands
            ticks = []
            for i in range(num_bands):
                label = f"{band_edges[i]:.1f}-{band_edges[i + 1]:.1f}"
                ticks.append((i, label))
            p2.getAxis('bottom').setTicks([ticks])

            # Add close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            # Set up layout
            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)

            # Show the window
            plot_window.show()
            Logger.log_message_static("Energy analysis plot window displayed successfully", Logger.INFO)

            # Keep reference to prevent garbage collection
            self._plot_windows.append(plot_window)

            # Show energy results in main dialog
            self.show_analysis_results("Energy Analysis", signal, energy_results)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in energy analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Energy analysis traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_phase_analysis(self):
        """Analyze and display phase information of the signal."""
        Logger.log_message_static("Preparing phase analysis", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("Cannot perform phase analysis: No signal selected", Logger.WARNING)
            return

        try:
            # Get the source signal data
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for phase analysis", Logger.DEBUG)

            # Process the signal values directly
            Logger.log_message_static("Preparing signal for phase analysis", Logger.DEBUG)
            processed_values = prepare_signal_for_analysis(self, values, "Phase Analysis Input Signal")
            if processed_values is None:
                Logger.log_message_static("Phase analysis canceled by user", Logger.INFO)
                return  # User canceled the operation

            # Calculate Hilbert transform to get phase information
            Logger.log_message_static("Computing Hilbert transform for phase extraction", Logger.DEBUG)
            analytic_signal = hilbert(processed_values)

            # Extract instantaneous phase and unwrap to avoid phase jumps
            Logger.log_message_static("Extracting and unwrapping instantaneous phase", Logger.DEBUG)
            phase = np.unwrap(np.angle(analytic_signal))

            # Calculate phase statistics
            Logger.log_message_static("Computing phase statistics", Logger.DEBUG)
            phase_stats = {
                "Mean Phase": np.mean(phase),
                "Phase Standard Deviation": np.std(phase),
                "Phase Range": np.max(phase) - np.min(phase),
                "Phase Rate of Change": np.mean(np.abs(np.diff(phase))) / np.mean(np.diff(time_arr))
            }
            Logger.log_message_static(f"Phase statistics calculated: {phase_stats}", Logger.DEBUG)

            # Create a proper window using QMainWindow
            Logger.log_message_static("Creating phase analysis plot window", Logger.DEBUG)
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Phase Analysis: {signal}")
            plot_window.resize(800, 600)

            # Create central widget with layout
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            # Create plot widget
            plot_widget = pg.PlotWidget(title=f"Phase Analysis: {signal}")
            plot_widget.setLabel('left', 'Phase (rad)')
            plot_widget.setLabel('bottom', 'Time (s)')
            plot_widget.plot(time_arr, phase, pen='b')

            # Add phase velocity plot
            Logger.log_message_static("Computing phase velocity for secondary plot", Logger.DEBUG)
            phase_velocity = np.diff(phase) / np.diff(time_arr)
            phase_velocity_time = time_arr[1:]  # Adjust time array to match velocity length

            # Create a second plot for phase velocity
            phase_velocity_widget = pg.PlotWidget(title="Phase Velocity")
            phase_velocity_widget.setLabel('left', 'Phase Velocity (rad/s)')
            phase_velocity_widget.setLabel('bottom', 'Time (s)')
            phase_velocity_widget.plot(phase_velocity_time, phase_velocity, pen='g')

            # Add close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            # Set up layout
            layout.addWidget(plot_widget)
            layout.addWidget(phase_velocity_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)

            # Show the window
            plot_window.show()
            Logger.log_message_static("Phase analysis plot window displayed successfully", Logger.INFO)

            # Keep reference to prevent garbage collection
            self._plot_windows.append(plot_window)

            # Display results in the dialog
            Logger.log_message_static("Displaying phase analysis results in main dialog", Logger.DEBUG)
            self.show_analysis_results("Phase Analysis", signal, phase_stats)
            Logger.log_message_static("Phase analysis complete", Logger.INFO)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in phase analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Phase analysis traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_cepstrum_analysis(self):
        """Analyze and display cepstrum analysis of the signal."""
        Logger.log_message_static("Preparing cepstrum analysis", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("Cannot perform cepstrum analysis: No signal selected", Logger.WARNING)
            return

        try:
            # Get the source signal data
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for cepstrum analysis", Logger.DEBUG)

            # Process the signal values
            Logger.log_message_static("Preparing signal for cepstrum analysis", Logger.DEBUG)
            processed_values = prepare_signal_for_analysis(self, values, "Cepstrum Analysis Input Signal")
            if processed_values is None:
                Logger.log_message_static("Cepstrum analysis canceled by user", Logger.INFO)
                return  # User canceled the operation

            # Calculate sample rate
            if len(time_arr) < 2:
                Logger.log_message_static("Signal too short for cepstrum analysis", Logger.WARNING)
                return

            fs = 1 / np.mean(np.diff(time_arr))
            Logger.log_message_static(f"Sample rate calculated: {fs} Hz", Logger.DEBUG)

            # Compute cepstrum
            Logger.log_message_static("Computing FFT for cepstrum analysis", Logger.DEBUG)
            n = len(processed_values)
            fft_values = np.fft.fft(processed_values)

            # Logarithm of the power spectrum
            Logger.log_message_static("Computing log power spectrum", Logger.DEBUG)
            log_power = np.log(np.abs(fft_values) ** 2 + 1e-10)  # Add small epsilon to avoid log(0)

            # Inverse FFT of the log spectrum gives the cepstrum
            Logger.log_message_static("Computing inverse FFT to obtain cepstrum", Logger.DEBUG)
            cepstrum = np.fft.ifft(log_power).real

            # Create quefrency array (time-like x-axis for cepstrum)
            quefrency = np.arange(n) / fs
            Logger.log_message_static(f"Created quefrency array with {len(quefrency)} points", Logger.DEBUG)

            # Create a proper window
            Logger.log_message_static("Creating cepstrum analysis plot window", Logger.DEBUG)
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Cepstrum Analysis: {signal}")
            plot_window.resize(800, 600)

            # Create central widget with layout
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            # Create plot widget with 3 subplots
            plot_widget = pg.GraphicsLayoutWidget()

            # 1. Original signal
            p1 = plot_widget.addPlot(row=0, col=0)
            p1.setTitle("Original Signal")
            p1.setLabel('left', 'Amplitude')
            p1.setLabel('bottom', 'Time (s)')
            p1.plot(time_arr, processed_values, pen='b')

            # 2. Log power spectrum
            p2 = plot_widget.addPlot(row=1, col=0)
            p2.setTitle("Log Power Spectrum")
            p2.setLabel('left', 'Log Power')
            p2.setLabel('bottom', 'Frequency (Hz)')

            # For logarithmic spectrum, only display the positive frequency half
            freqs = np.fft.fftfreq(n, d=1 / fs)
            # Rearrange to have 0 frequency at the start
            freqs = np.fft.fftshift(freqs)
            log_power_shift = np.fft.fftshift(log_power)

            # Plot only the positive frequency half
            mid_point = len(freqs) // 2
            p2.plot(freqs[mid_point:], log_power_shift[mid_point:], pen='g')
            Logger.log_message_static("Plotted log power spectrum (positive frequencies)", Logger.DEBUG)

            # 3. Cepstrum
            p3 = plot_widget.addPlot(row=2, col=0)
            p3.setTitle("Cepstrum")
            p3.setLabel('left', 'Amplitude')
            p3.setLabel('bottom', 'Quefrency (s)')

            # Only show first half of cepstrum (real signals are symmetric)
            n_half = n // 2
            p3.plot(quefrency[:n_half], cepstrum[:n_half], pen='r')
            Logger.log_message_static("Plotted cepstrum (first half)", Logger.DEBUG)

            # Detect peaks in the cepstrum that might indicate periodicity
            Logger.log_message_static("Detecting peaks in cepstrum", Logger.DEBUG)
            # Skip the first few points where the high peaks often occur
            skip_points = int(0.002 * fs)  # Skip first 2ms or equivalent points
            if skip_points >= n_half:
                skip_points = n_half // 10  # Safety check
                Logger.log_message_static(f"Adjusted skip points to {skip_points} (10% of data)", Logger.WARNING)

            peaks, _ = sc_signal.find_peaks(cepstrum[skip_points:n_half],
                                            height=0.1 * np.max(cepstrum[skip_points:n_half]))
            peaks = peaks + skip_points  # Adjust indices back to original scale

            if len(peaks) > 0:
                Logger.log_message_static(f"Found {len(peaks)} peaks in cepstrum", Logger.DEBUG)
                # Add peak markers
                scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen('y', width=2), brush=pg.mkBrush('y'))
                spots = [{'pos': (quefrency[idx], cepstrum[idx]), 'data': idx} for idx in peaks]
                scatter.addPoints(spots)
                p3.addItem(scatter)

                # Find the highest peak (ignoring the first few points)
                highest_peak_idx = peaks[np.argmax(cepstrum[peaks])]
                peak_quefrency = quefrency[highest_peak_idx]
                fundamental_frequency = 1.0 / peak_quefrency if peak_quefrency > 0 else 0
                Logger.log_message_static(
                    f"Highest peak at quefrency {peak_quefrency:.6f}s, corresponding to frequency {fundamental_frequency:.2f}Hz",
                    Logger.DEBUG)

                # Add annotation for the highest peak
                text = pg.TextItem(text=f"{fundamental_frequency:.1f} Hz", color='y', anchor=(0, 0))
                text.setPos(quefrency[highest_peak_idx], cepstrum[highest_peak_idx])
                p3.addItem(text)
            else:
                Logger.log_message_static("No significant peaks found in cepstrum", Logger.WARNING)
                fundamental_frequency = 0

            # Add close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            # Set up layout
            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)

            # Show the window
            plot_window.show()
            Logger.log_message_static("Cepstrum analysis plot window displayed successfully", Logger.INFO)

            # Keep reference to prevent garbage collection
            self._plot_windows.append(plot_window)

            # Prepare results
            Logger.log_message_static("Preparing cepstrum analysis results", Logger.DEBUG)
            cepstrum_stats = {
                "Maximum Cepstrum Value": np.max(cepstrum[skip_points:n_half]),
                "Mean Cepstrum Value": np.mean(cepstrum[skip_points:n_half]),
                "Detected Fundamental Frequency": f"{fundamental_frequency:.2f} Hz" if fundamental_frequency > 0 else "None detected"
            }

            # Add information about peaks if found
            if len(peaks) > 0:
                top_peaks = sorted([(quefrency[idx], cepstrum[idx], 1.0 / quefrency[idx] if quefrency[idx] > 0 else 0)
                                    for idx in peaks], key=lambda x: x[1], reverse=True)[:3]

                for i, (q, val, freq) in enumerate(top_peaks):
                    cepstrum_stats[f"Peak {i + 1} Quefrency"] = f"{q:.6f} s"
                    cepstrum_stats[f"Peak {i + 1} Frequency"] = f"{freq:.2f} Hz"

            # Display results in the dialog
            Logger.log_message_static("Displaying cepstrum analysis results in main dialog", Logger.DEBUG)
            self.show_analysis_results("Cepstrum Analysis", signal, cepstrum_stats)
            Logger.log_message_static("Cepstrum analysis complete", Logger.INFO)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in cepstrum analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Cepstrum analysis traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_wavelet_dialog(self):
        """Show dialog for wavelet analysis and display the results."""
        Logger.log_message_static("Opening wavelet analysis dialog", Logger.INFO)
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            Logger.log_message_static("Cannot perform wavelet analysis: No signal selected", Logger.WARNING)
            return

        try:
            # Get the source signal data
            time_arr, values = self.parent.data_signals[signal]
            Logger.log_message_static(f"Retrieved {len(values)} data points for wavelet analysis", Logger.DEBUG)

            # Process the signal values
            Logger.log_message_static("Preparing signal for wavelet analysis", Logger.DEBUG)
            processed_values = prepare_signal_for_analysis(self, values, "Wavelet Analysis Input Signal")
            if processed_values is None:
                Logger.log_message_static("Wavelet analysis canceled by user", Logger.INFO)
                return  # User canceled the operation

            # Create wavelet settings in a dialog
            Logger.log_message_static("Initializing wavelet dialog settings", Logger.DEBUG)
            wavelet_dialog = QDialog(self)
            wavelet_dialog.setWindowTitle("Wavelet Analysis Settings")
            wavelet_dialog.resize(400, 300)

            layout = QVBoxLayout(wavelet_dialog)

            form = QFormLayout()

            # Wavelet type selector
            wavelet_type = QComboBox()
            available_wavelets = ['haar', 'db4', 'sym4', 'coif4', 'morl', 'cmor', 'mexh']
            wavelet_type.addItems(available_wavelets)
            form.addRow("Wavelet Type:", wavelet_type)
            Logger.log_message_static(f"Added wavelet type selector with {len(available_wavelets)} options",
                                            Logger.DEBUG)

            # Number of scales
            scales_spinbox = QDoubleSpinBox()
            scales_spinbox.setDecimals(0)
            scales_spinbox.setRange(1, 128)
            scales_spinbox.setValue(64)
            form.addRow("Number of Scales:", scales_spinbox)
            Logger.log_message_static("Added scales spinbox with range 1-128", Logger.DEBUG)

            # Scale spacing - linear or logarithmic
            scale_spacing = QComboBox()
            scale_spacing.addItems(["logarithmic", "linear"])
            form.addRow("Scale Spacing:", scale_spacing)

            layout.addLayout(form)

            # Buttons
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(wavelet_dialog.accept)
            buttons.rejected.connect(wavelet_dialog.reject)
            layout.addWidget(buttons)

            # Show the dialog
            Logger.log_message_static("Displaying wavelet dialog", Logger.DEBUG)
            if not wavelet_dialog.exec():
                Logger.log_message_static("Wavelet dialog canceled by user", Logger.INFO)
                return

            # Get parameters
            Logger.log_message_static("User accepted wavelet dialog, retrieving parameters", Logger.DEBUG)
            wavelet_name = wavelet_type.currentText()
            num_scales = int(scales_spinbox.value())
            is_log_scale = scale_spacing.currentText() == "logarithmic"
            Logger.log_message_static(
                f"Wavelet parameters: type={wavelet_name}, scales={num_scales}, log_scale={is_log_scale}", Logger.DEBUG)

            # Calculate sample rate
            if len(time_arr) < 2:
                Logger.log_message_static("Signal too short for wavelet analysis", Logger.WARNING)
                return

            fs = 1 / np.mean(np.diff(time_arr))
            Logger.log_message_static(f"Sample rate calculated: {fs} Hz", Logger.DEBUG)

            # Generate scales
            if is_log_scale:
                Logger.log_message_static("Generating logarithmic scales", Logger.DEBUG)
                scales = np.logspace(0, np.log10(num_scales), num_scales)
            else:
                Logger.log_message_static("Generating linear scales", Logger.DEBUG)
                scales = np.linspace(1, num_scales, num_scales)

            # Perform wavelet transform
            Logger.log_message_static(f"Performing wavelet transform with {wavelet_name} wavelet", Logger.DEBUG)
            try:
                if wavelet_name in ['morl', 'cmor', 'mexh']:
                    # Continuous wavelet transform for these wavelet types
                    Logger.log_message_static("Using continuous wavelet transform (CWT)", Logger.DEBUG)
                    coefficients, frequencies = pywt.cwt(processed_values, scales, wavelet_name)
                    # Frequencies are in terms of scale - convert to Hz
                    frequencies = pywt.scale2frequency(wavelet_name, scales) * fs
                    Logger.log_message_static(f"CWT completed with {len(frequencies)} frequency bands", Logger.DEBUG)
                else:
                    # Discrete wavelet transform with multi-level decomposition
                    Logger.log_message_static("Using discrete wavelet transform (DWT)", Logger.DEBUG)
                    max_level = pywt.dwt_max_level(len(processed_values), pywt.Wavelet(wavelet_name).dec_len)
                    level = min(int(np.log2(num_scales)), max_level)
                    Logger.log_message_static(f"Using DWT at level {level} (max possible: {max_level})", Logger.DEBUG)

                    # Perform multilevel DWT
                    coeffs = pywt.wavedec(processed_values, wavelet_name, level=level)
                    Logger.log_message_static(f"DWT completed with {len(coeffs)} coefficient sets", Logger.DEBUG)

                    # Reshape coefficients to match expected format for CWT display
                    total_len = sum(len(c) for c in coeffs)
                    coefficients = np.zeros((len(coeffs), len(processed_values)))

                    # Calculate pseudo-frequencies for each level
                    frequencies = []
                    for i, c in enumerate(coeffs):
                        # Upsample each coefficient array to match the original signal length
                        coefficients[i, :len(c)] = c
                        # For simplicity, we'll use approximate frequency bands for each level
                        band_max = fs / (2 ** (i + 1))
                        band_min = fs / (2 ** (i + 2)) if i < len(coeffs) - 1 else 0
                        center_freq = (band_max + band_min) / 2
                        frequencies.append(center_freq)

                    # Sort by frequency (high to low, which is opposite of level order)
                    idx = np.argsort(frequencies)[::-1]
                    frequencies = np.array(frequencies)[idx]
                    coefficients = coefficients[idx]
                    Logger.log_message_static(
                        f"Prepared DWT coefficients for display with frequencies {frequencies}", Logger.DEBUG)
            except Exception as e:
                Logger.log_message_static(f"Error computing wavelet transform: {str(e)}", Logger.ERROR)
                # Consider simpler option if the first attempt fails
                if wavelet_name not in ['morl', 'cmor', 'mexh']:
                    try:
                        Logger.log_message_static("Falling back to simpler wavelet transform", Logger.WARNING)
                        wavelet_name = 'morl'
                        coefficients, frequencies = pywt.cwt(processed_values, scales, wavelet_name)
                        frequencies = pywt.scale2frequency(wavelet_name, scales) * fs
                        Logger.log_message_static("Fallback wavelet transform successful", Logger.INFO)
                    except Exception as e2:
                        Logger.log_message_static(f"Fallback wavelet transform also failed: {str(e2)}", Logger.ERROR)
                        raise Exception(f"Failed to compute wavelet transform: {str(e)}")

            # Create a proper window
            Logger.log_message_static("Creating wavelet analysis plot window", Logger.DEBUG)
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Wavelet Analysis: {signal} ({wavelet_name})")
            plot_window.resize(800, 600)

            # Create central widget with layout
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            # Create plot widget
            plot_widget = pg.GraphicsLayoutWidget()

            # 1. Original signal plot
            p1 = plot_widget.addPlot(row=0, col=0)
            p1.setTitle("Original Signal")
            p1.setLabel('left', 'Amplitude')
            p1.setLabel('bottom', 'Time (s)')
            p1.plot(time_arr, processed_values, pen='b')
            Logger.log_message_static("Added original signal plot", Logger.DEBUG)

            # 2. Scalogram plot (2D heatmap of wavelet coefficients)
            p2 = plot_widget.addPlot(row=1, col=0)
            p2.setTitle("Wavelet Scalogram")

            # Convert coefficients to magnitude for better visualization
            abs_coeffs = np.abs(coefficients)
            Logger.log_message_static(f"Coefficient matrix shape: {abs_coeffs.shape}", Logger.DEBUG)

            # Create image item for the scalogram
            img = pg.ImageItem()
            p2.addItem(img)

            # Set the data with appropriate levels for good contrast
            img.setImage(abs_coeffs)
            min_val = np.min(abs_coeffs)
            max_val = np.max(abs_coeffs)
            img.setLevels([min_val, max_val * 0.8])  # Use 80% of max for better visibility
            Logger.log_message_static(f"Set scalogram levels: [{min_val}, {max_val * 0.8}]", Logger.DEBUG)

            # Create a colorbar
            hist = pg.HistogramLUTItem()
            hist.setImageItem(img)
            hist.setLevels(min_val, max_val * 0.8)
            plot_widget.addItem(hist, row=1, col=1)

            # Set the y-axis to show frequency instead of scale
            p2.setLabel('left', 'Frequency (Hz)')
            p2.setLabel('bottom', 'Time (s)')

            # Adjust axes to match time and frequency scales
            transform = QTransform().scale(time_arr[-1] / abs_coeffs.shape[1], 1).translate(0, 0)
            img.setTransform(transform)

            # Create custom y-axis ticks for frequencies
            if len(frequencies) > 1:
                # Create frequency axis with simplified labels
                freq_ticks = []
                num_labels = min(8, len(frequencies))  # Limit number of labels to avoid overcrowding
                step = len(frequencies) // num_labels

                for i in range(0, len(frequencies), step):
                    freq_ticks.append((i, f"{frequencies[i]:.1f}"))

                ax = p2.getAxis('left')
                ax.setTicks([freq_ticks])
                Logger.log_message_static(f"Set {len(freq_ticks)} frequency axis ticks", Logger.DEBUG)

            # 3. Power plot of different scales/frequencies
            p3 = plot_widget.addPlot(row=2, col=0)
            p3.setTitle("Average Power by Frequency")
            p3.setLabel('left', 'Power')
            p3.setLabel('bottom', 'Frequency (Hz)')

            # Calculate average power at each scale/frequency
            avg_power = np.mean(abs_coeffs ** 2, axis=1)

            if wavelet_name in ['morl', 'cmor', 'mexh']:
                # For CWT, we have direct frequency mapping
                p3.plot(frequencies, avg_power, pen='g')
                p3.setLogMode(x=True, y=True)  # Log scale often works better for frequency plots
            else:
                # For DWT, we have discrete bands
                bar_graph = pg.BarGraphItem(x=range(len(frequencies)), height=avg_power, width=0.8, brush='g')
                p3.addItem(bar_graph)

                # Set appropriate x-axis ticks
                freq_ticks = [(i, f"{freq:.1f}") for i, freq in enumerate(frequencies)]
                ax = p3.getAxis('bottom')
                ax.setTicks([freq_ticks])

            Logger.log_message_static("Added power by frequency plot", Logger.DEBUG)

            # Add close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            # Set up layout
            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)

            # Show the window
            plot_window.show()
            Logger.log_message_static("Wavelet analysis plot window displayed successfully", Logger.INFO)

            # Keep reference to prevent garbage collection
            self._plot_windows.append(plot_window)

            # Calculate summary statistics for wavelet analysis
            Logger.log_message_static("Calculating wavelet analysis statistics", Logger.DEBUG)

            # Find dominant frequency (band with most energy)
            dominant_idx = np.argmax(avg_power)
            dominant_freq = frequencies[dominant_idx] if len(frequencies) > dominant_idx else 0
            Logger.log_message_static(f"Identified dominant frequency: {dominant_freq} Hz", Logger.DEBUG)

            # Energy distribution across frequency bands
            total_energy = np.sum(avg_power)
            energy_percent = (avg_power / total_energy) * 100 if total_energy > 0 else np.zeros_like(avg_power)

            # High/low frequency energy ratio
            if len(frequencies) > 2:
                mid_point = len(frequencies) // 2
                high_freq_energy = np.sum(avg_power[:mid_point])
                low_freq_energy = np.sum(avg_power[mid_point:])
                energy_ratio = high_freq_energy / low_freq_energy if low_freq_energy > 0 else float('inf')
                Logger.log_message_static(f"High/low frequency energy ratio: {energy_ratio}", Logger.DEBUG)
            else:
                energy_ratio = 1.0

            # Prepare results
            wavelet_stats = {
                "Wavelet Type": wavelet_name,
                "Number of Scales/Levels": len(frequencies),
                "Dominant Frequency": f"{dominant_freq:.2f} Hz",
                "Total Energy": total_energy,
                "High/Low Frequency Energy Ratio": f"{energy_ratio:.4f}"
            }

            # Add top frequency bands by energy
            if len(frequencies) >= 3:
                top_indices = np.argsort(avg_power)[-3:][::-1]
                for i, idx in enumerate(top_indices):
                    wavelet_stats[
                        f"Top Frequency Band {i + 1}"] = f"{frequencies[idx]:.2f} Hz ({energy_percent[idx]:.1f}% of energy)"

            # Display results in the dialog
            Logger.log_message_static("Displaying wavelet analysis results in main dialog", Logger.DEBUG)
            self.show_analysis_results("Wavelet Analysis", signal, wavelet_stats)
            Logger.log_message_static("Wavelet analysis complete", Logger.INFO)

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in wavelet analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Wavelet analysis traceback: {traceback.format_exc()}", Logger.DEBUG)

    def show_cross_correlation(self):
        """Calculate and display cross-correlation between two signals."""
        Logger.log_message_static("Preparing cross-correlation analysis", Logger.INFO)
        signal1 = self.get_selected_signal(self.cross_signal1_combo)
        signal2 = self.get_selected_signal(self.cross_signal2_combo)

        if not signal1 or not signal2:
            Logger.log_message_static(
                "Cannot perform cross-correlation: One or both signals not selected", Logger.WARNING)
            return

        try:
            Logger.log_message_static(f"Computing cross-correlation between '{signal1}' and '{signal2}'",
                                            Logger.DEBUG)
            # Get the source signal data directly
            time_arr1, values1 = self.parent.data_signals[signal1]
            time_arr2, values2 = self.parent.data_signals[signal2]

            Logger.log_message_static(
                f"Signal lengths - {signal1}: {len(values1)}, {signal2}: {len(values2)}", Logger.DEBUG)

            # Handle signals with different lengths by padding the shorter one
            if len(values1) > len(values2):
                Logger.log_message_static(f"Padding {signal2} to match length of {signal1}", Logger.DEBUG)
                values2 = np.pad(values2, (0, len(values1) - len(values2)), 'constant')
            elif len(values2) > len(values1):
                Logger.log_message_static(f"Padding {signal1} to match length of {signal2}", Logger.DEBUG)
                values1 = np.pad(values1, (0, len(values2) - len(values1)), 'constant')

            # Calculate cross-correlation
            Logger.log_message_static("Computing cross-correlation with full mode", Logger.DEBUG)
            corr = correlate(values1, values2, mode='full')

            # Normalize to [-1, 1] range
            Logger.log_message_static("Normalizing cross-correlation values", Logger.DEBUG)
            norm_factor = np.sqrt(np.sum(values1 ** 2) * np.sum(values2 ** 2))
            if norm_factor > 0:
                corr_normalized = corr / norm_factor
            else:
                Logger.log_message_static("Warning: Zero normalization factor in cross-correlation",
                                                Logger.WARNING)
                corr_normalized = corr

            # Create lag time array
            Logger.log_message_static("Creating lag time axis", Logger.DEBUG)
            sample_rate1 = 1.0 / np.mean(np.diff(time_arr1))
            lags = np.arange(-len(corr_normalized) // 2 + 1, len(corr_normalized) // 2 + 1) / sample_rate1

            # Create a proper window
            Logger.log_message_static("Creating cross-correlation plot window", Logger.DEBUG)
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Cross Correlation: {signal1} & {signal2}")
            plot_window.resize(800, 600)

            # Create central widget with layout
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            # Create plot widget
            plot_widget = pg.GraphicsLayoutWidget()

            # Add cross-correlation plot
            p1 = plot_widget.addPlot(row=0, col=0)
            p1.setTitle("Cross-Correlation")
            p1.setLabel('left', 'Correlation')
            p1.setLabel('bottom', 'Lag (s)')
            p1.plot(lags, np.roll(corr_normalized, len(corr_normalized) // 2), pen='b')

            # Draw a horizontal line at 0
            p1.addLine(y=0, pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
            Logger.log_message_static("Created cross-correlation plot", Logger.DEBUG)

            # Find the max correlation and its lag
            max_idx = np.argmax(corr_normalized)
            max_lag = lags[max_idx - len(corr_normalized) // 2]
            max_corr = corr_normalized[max_idx]

            Logger.log_message_static(f"Maximum correlation: {max_corr:.4f} at lag {max_lag:.4f}s", Logger.DEBUG)

            # Add a vertical line at max correlation
            p1.addLine(x=max_lag, pen=pg.mkPen('g', width=2))

            # Add comparison plot of the two signals
            p2 = plot_widget.addPlot(row=1, col=0)
            p2.setTitle("Signal Comparison")
            p2.setLabel('left', 'Amplitude')
            p2.setLabel('bottom', 'Time (s)')

            # Make sure we plot the original (unpadded) signals
            time_arr1_orig, values1_orig = self.parent.data_signals[signal1]
            time_arr2_orig, values2_orig = self.parent.data_signals[signal2]

            p2.plot(time_arr1_orig, values1_orig, pen='b', name=signal1)
            p2.plot(time_arr2_orig, values2_orig, pen='r', name=signal2)
            Logger.log_message_static("Added original signals comparison plot", Logger.DEBUG)

            # Add legend
            legend = p2.addLegend()
            legend.addItem(pg.PlotDataItem(pen='b'), signal1)
            legend.addItem(pg.PlotDataItem(pen='r'), signal2)

            # Add time-shifted version of signal 2 using the lag
            if abs(max_lag) > 0.001:  # Only if lag is significant
                Logger.log_message_static(f"Adding time-shifted version of {signal2} by {max_lag:.4f}s",
                                                Logger.DEBUG)
                # Create time-shifted signal
                time_arr2_shifted = time_arr2_orig + max_lag

                # Only plot the overlapping part
                valid_mask = (time_arr2_shifted >= min(time_arr1_orig)) & (
                            time_arr2_shifted <= max(time_arr1_orig))
                if np.any(valid_mask):
                    p2.plot(time_arr2_shifted[valid_mask], values2_orig[valid_mask],
                            pen=pg.mkPen('g', width=2, style=Qt.PenStyle.DashLine),
                            name=f"{signal2} (shifted)")
                    legend.addItem(pg.PlotDataItem(pen=pg.mkPen('g', width=2, style=Qt.PenStyle.DashLine)),
                                   f"{signal2} (shifted by {max_lag:.4f}s)")
                    Logger.log_message_static("Added time-shifted signal to comparison plot", Logger.DEBUG)
                else:
                    Logger.log_message_static("Shifted signal has no overlap with original time range",
                                                    Logger.WARNING)

            # Add close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(plot_window.close)

            # Set up layout
            layout.addWidget(plot_widget)
            layout.addWidget(close_button)
            central_widget.setLayout(layout)
            plot_window.setCentralWidget(central_widget)

            # Show the window
            plot_window.show()
            Logger.log_message_static("Cross-correlation plot window displayed successfully", Logger.INFO)

            # Keep reference to prevent garbage collection
            self._plot_windows.append(plot_window)

            # Calculate statistics
            Logger.log_message_static("Calculating cross-correlation statistics", Logger.DEBUG)

            # Calculate lags at different correlation thresholds
            threshold_corrs = [0.5, 0.7, 0.9]
            threshold_lags = {}

            for threshold in threshold_corrs:
                above_threshold = np.where(corr_normalized >= threshold * max_corr)[0]
                if len(above_threshold) > 0:
                    min_idx = np.min(above_threshold)
                    max_idx = np.max(above_threshold)
                    lag_range = lags[max_idx - len(corr_normalized) // 2] - lags[
                        min_idx - len(corr_normalized) // 2]
                    threshold_lags[threshold] = lag_range
                    Logger.log_message_static(
                        f"Width at {threshold * 100:.0f}% of max correlation: {lag_range:.4f}s", Logger.DEBUG)
                else:
                    threshold_lags[threshold] = None
                    Logger.log_message_static(f"No points above {threshold * 100:.0f}% threshold found",
                                                    Logger.WARNING)

            # Compute correlation at zero lag
            zero_lag_idx = len(corr_normalized) // 2
            zero_lag_correlation = corr_normalized[zero_lag_idx]
            Logger.log_message_static(f"Correlation at zero lag: {zero_lag_correlation:.4f}", Logger.DEBUG)

            # Prepare results
            cross_corr_stats = {
                "Maximum Correlation": f"{max_corr:.4f}",
                "Lag at Maximum Correlation": f"{max_lag:.6f} seconds",
                "Correlation at Zero Lag": f"{zero_lag_correlation:.4f}",
                "Correlation Peak Width (50%)": f"{threshold_lags.get(0.5, 'N/A')} seconds",
                "Correlation Peak Width (70%)": f"{threshold_lags.get(0.7, 'N/A')} seconds",
                "Correlation Peak Width (90%)": f"{threshold_lags.get(0.9, 'N/A')} seconds",
            }

            # Display results in the dialog
            Logger.log_message_static("Displaying cross-correlation results in main dialog", Logger.DEBUG)
            self.show_analysis_results("Cross-Correlation Analysis", f"{signal1} & {signal2}", cross_corr_stats)
            Logger.log_message_static("Cross-correlation analysis complete", Logger.INFO)

        except KeyError as ke:
            Logger.log_message_static(f"Signal not found in data_signals: {str(ke)}", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in cross-correlation analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Cross-correlation analysis traceback: {traceback.format_exc()}",
                                                Logger.DEBUG)

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


class ExplanationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_dialog = parent  # Store reference to parent dialog
        self.setup_ui()

    def setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)

        # Create scrollable area for buttons
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Create button groups
        self.create_button_groups(scroll_layout)

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

    def create_button_groups(self, parent_layout):
        # Basic Analysis group
        basic_group = QGroupBox("Basic Analysis")
        basic_layout = QGridLayout()

        basic_buttons = [
            "Statistics", "FFT Analysis", "Time Domain Analysis"
        ]

        self.add_buttons_to_grid(basic_layout, basic_buttons)
        basic_group.setLayout(basic_layout)
        parent_layout.addWidget(basic_group)

        # Advanced Analysis group
        advanced_group = QGroupBox("Advanced Analysis")
        advanced_layout = QGridLayout()

        advanced_buttons = [
            "Power Spectral Density", "Autocorrelation", "Peak Detection",
            "Hilbert Transform", "Energy Analysis", "Phase Analysis", "Cepstral Analysis"
        ]

        self.add_buttons_to_grid(advanced_layout, advanced_buttons)
        advanced_group.setLayout(advanced_layout)
        parent_layout.addWidget(advanced_group)

        # Cross Analysis group
        cross_group = QGroupBox("Cross Analysis")
        cross_layout = QGridLayout()

        cross_buttons = ["Cross-Correlation"]

        self.add_buttons_to_grid(cross_layout, cross_buttons)
        cross_group.setLayout(cross_layout)
        parent_layout.addWidget(cross_group)

        # Filtering group
        filter_group = QGroupBox("Filtering")
        filter_layout = QGridLayout()

        filter_buttons = ["Lowpass Filter", "Highpass Filter", "Bandpass Filter"]

        self.add_buttons_to_grid(filter_layout, filter_buttons)
        filter_group.setLayout(filter_layout)
        parent_layout.addWidget(filter_group)

        # Wavelet Analysis group
        wavelet_group = QGroupBox("Wavelet Analysis")
        wavelet_layout = QGridLayout()

        wavelet_buttons = ["Wavelet Transform", "Wavelet Types", "Wavelet Applications"]

        self.add_buttons_to_grid(wavelet_layout, wavelet_buttons)
        wavelet_group.setLayout(wavelet_layout)
        parent_layout.addWidget(wavelet_group)

    def add_buttons_to_grid(self, layout, button_texts, cols=3):
        for i, text in enumerate(button_texts):
            row = i // cols
            col = i % cols
            button = QPushButton(text)
            # Connect to new method that shows help in results area
            button.clicked.connect(lambda checked, t=text: self.show_help(t))
            layout.addWidget(button, row, col)

    def show_default_help(self):
        self.help_text.setHtml("""
        <h3>Signal Analysis Help</h3>
        <p>Click on any button above to see detailed information about that analysis method.</p>
        <p>This help section will explain:</p>
        <ul>
            <li>What the analysis method does</li>
            <li>When to use it</li>
            <li>Limitations and considerations</li>
            <li>How to interpret the results</li>
        </ul>
        """)

    def show_help(self, topic):
        """Display help content for the selected topic."""
        # Dictionary of help content for each topic
        help_content = {
            "Statistics": """
                <h3>Basic Signal Statistics</h3>
                <p>Basic statistics provide fundamental insights about the signal's characteristics:</p>
                <ul>
                    <li><b>Mean:</b> The average value of the signal, indicating central tendency.
                        <br>Formula: μ = (1/N)·∑x(i)</li>
                    <li><b>Median:</b> The middle value when signal values are ordered, less affected by outliers than mean.</li>
                    <li><b>Minimum/Maximum:</b> The smallest and largest values in the signal.</li>
                    <li><b>Range:</b> The difference between maximum and minimum values.</li>
                    <li><b>Standard Deviation:</b> Measures the amount of variation or dispersion in the signal.
                        <br>Formula: σ = √[(1/N)·∑(x(i)-μ)²]</li>
                    <li><b>Variance:</b> Square of standard deviation, measures signal power variation around the mean.</li>
                    <li><b>Root Mean Square (RMS):</b> Square root of the average of squared values, relates to signal energy.
                        <br>Formula: RMS = √[(1/N)·∑x(i)²]</li>
                    <li><b>Skewness:</b> Measures asymmetry of the signal distribution. Positive values indicate right-tailed distribution.</li>
                    <li><b>Kurtosis:</b> Measures the "tailedness" of the distribution (peakedness/flatness).
                        <br>Higher values indicate more extreme outliers.</li>
                    <li><b>Interquartile Range (IQR):</b> The range between the 25th and 75th percentiles, robust to outliers.</li>
                </ul>
                <p><b>When to use:</b> These statistics provide a basic quantitative description of your signal and can identify potential issues or characteristics.</p>
            """,

            "FFT Analysis": """
                <h3>FFT Analysis</h3>
                <p>Fast Fourier Transform converts a signal from the time domain to the frequency domain:</p>
                <ul>
                    <li><b>Purpose:</b> Identifies frequency components present in the signal.</li>
                    <li><b>Usage:</b> Detect periodic patterns, dominant frequencies, and harmonic content.</li>
                    <li><b>Mathematical basis:</b> Decomposes a signal into a sum of sinusoids of different frequencies.</li>
                </ul>
                <p><b>Interpretation:</b> 
                    <ul>
                        <li>Peaks in the frequency spectrum indicate strong periodic components at those frequencies.</li>
                        <li>Broader peaks suggest frequency variation or modulation.</li>
                        <li>Evenly spaced harmonics indicate a complex periodic signal.</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Assumes signal stationarity (frequency content doesn't change over time).</li>
                        <li>May not capture time-varying frequency content.</li>
                        <li>Subject to spectral leakage if signal period doesn't match window size.</li>
                        <li>Limited frequency resolution for short signals.</li>
                    </ul>
                </p>
            """,

            "Time Domain Analysis": """
                <h3>Time Domain Analysis</h3>
                <p>Examines signal characteristics directly in the time domain:</p>
                <ul>
                    <li><b>Duration:</b> Total time span of the signal.</li>
                    <li><b>Sample Rate:</b> Number of samples per second.</li>
                    <li><b>Zero Crossings:</b> Number of times the signal crosses the zero level, related to frequency content.</li>
                    <li><b>Signal Energy:</b> Sum of squared sample values (∑x²), represents total energy contained in the signal.</li>
                    <li><b>Signal Power:</b> Average power of the signal over time (energy/duration).</li>
                    <li><b>Crest Factor:</b> Ratio of peak value to RMS value, indicates signal impulsiveness.
                        <br>High values suggest transients or impulses.</li>
                </ul>
                <p><b>When to use:</b> For initial signal characterization, identifying abrupt changes, assessing signal quality, or determining appropriate processing methods.</p>
                <p><b>Limitations:</b> May not easily reveal frequency-related information or subtle patterns that frequency analysis would highlight.</p>
            """,

            "Power Spectral Density": """
                <h3>Power Spectral Density (PSD)</h3>
                <p>Measures how signal power is distributed across frequency:</p>
                <ul>
                    <li><b>Purpose:</b> Shows which frequencies contain the signal's power.</li>
                    <li><b>Usage:</b> Identify dominant frequencies, noise sources, or resonance.</li>
                    <li><b>Formula:</b> Squared magnitude of the Fourier transform, normalized by signal length.</li>
                    <li><b>Units:</b> Power per frequency (e.g., V²/Hz).</li>
                </ul>
                <p><b>Interpretation:</b> 
                    <ul>
                        <li>Areas with high PSD values indicate frequency bands that contribute significantly to the signal's power.</li>
                        <li>Peak width indicates stability of frequency component (narrower = more stable).</li>
                        <li>Log scale often used to visualize both strong and weak components.</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul> 
                        <li>Resolution depends on signal length and windowing.</li>
                        <li>Assumes signal is statistically stationary.</li>
                        <li>Averaging may be needed for noisy signals.</li>
                    </ul>
                </p>
            """,

            "Autocorrelation": """
                <h3>Autocorrelation</h3>
                <p>Measures similarity between a signal and a time-shifted version of itself:</p>
                <ul>
                    <li><b>Purpose:</b> Detect repeating patterns, periodicities, or signal memory.</li>
                    <li><b>Usage:</b> Find hidden periodicities, estimate fundamental frequency, detect signal redundancy.</li>
                    <li><b>Formula:</b> R(τ) = E[x(t)·x(t-τ)], where τ is the time lag.</li>
                </ul>
                <p><b>Interpretation:</b>
                    <ul>
                        <li>Peak at zero lag (always present) represents signal energy.</li>
                        <li>Secondary peaks indicate periodic components.</li>
                        <li>Distance between peaks represents period of repetitive pattern.</li>
                        <li>Decay rate indicates "memory" in the signal (how quickly it becomes uncorrelated with itself).</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>May be affected by noise or trends in the signal.</li>
                        <li>Multiple periodicities can create complex patterns that are difficult to interpret.</li>
                        <li>Requires sufficient signal length to detect long-period patterns.</li>
                    </ul>
                </p>
            """,

            "Peak Detection": """
                <h3>Peak Detection</h3>
                <p>Identifies local maxima (peaks) in the signal:</p>
                <ul>
                    <li><b>Purpose:</b> Locate significant events or features in the signal.</li>
                    <li><b>Usage:</b> Count events, measure intervals between events, identify important signal points.</li>
                    <li><b>Parameters:</b>
                        <ul>
                            <li><b>Height threshold:</b> Minimum amplitude to be considered a peak.</li>
                            <li><b>Distance:</b> Minimum separation between adjacent peaks.</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Applications:</b>
                    <ul>
                        <li>Heartbeat detection in ECG signals.</li>
                        <li>Event counting in sensor data.</li>
                        <li>Pulse detection in various signals.</li>
                        <li>Peak analysis in spectral data.</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Sensitivity to threshold settings and noise.</li>
                        <li>May miss closely spaced peaks due to distance parameter.</li>
                        <li>Difficulty with very broad or asymmetric peaks.</li>
                        <li>Baseline drift can affect detection accuracy.</li>
                    </ul>
                </p>
            """,

            "Lowpass Filter": """
                <h3>Lowpass Filter</h3>
                <p>Allows low-frequency components to pass while attenuating high frequencies:</p>
                <ul>
                    <li><b>Purpose:</b> Remove high-frequency noise or isolate low-frequency trends.</li>
                    <li><b>Parameters:</b>
                        <ul>
                            <li><b>Cutoff frequency:</b> Frequency above which signals are attenuated.</li>
                            <li><b>Filter order:</b> Controls steepness of transition (higher = steeper).</li>
                        </ul>
                    </li>
                    <li><b>Applications:</b>
                        <ul>
                            <li>Noise reduction</li>
                            <li>Smoothing signals</li>
                            <li>Extracting slow trends</li>
                            <li>Anti-aliasing before downsampling</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Interpretation:</b> After filtering, the signal will appear smoother, with rapid changes removed.</p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>May introduce phase distortion or time delays.</li>
                        <li>Higher order filters can cause ringing artifacts.</li>
                        <li>Cannot selectively preserve high-frequency features.</li>
                        <li>Choice of cutoff frequency is critical for retaining important information.</li>
                    </ul>
                </p>
            """,

            "Highpass Filter": """
                <h3>Highpass Filter</h3>
                <p>Allows high-frequency components to pass while attenuating low frequencies:</p>
                <ul>
                    <li><b>Purpose:</b> Remove baseline drift or isolate rapid changes in signals.</li>
                    <li><b>Parameters:</b>
                        <ul>
                            <li><b>Cutoff frequency:</b> Frequency below which signals are attenuated.</li>
                            <li><b>Filter order:</b> Controls steepness of transition (higher = steeper).</li>
                        </ul>
                    </li>
                    <li><b>Applications:</b>
                        <ul>
                            <li>Removing DC offset or drift</li>
                            <li>Detecting edges or rapid transitions</li>
                            <li>Isolating high-frequency events</li>
                            <li>AC coupling in electronic signals</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Interpretation:</b> After filtering, only rapid changes and high-frequency components remain; slow trends are removed.</p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>May introduce phase distortion.</li>
                        <li>Can remove important low-frequency information.</li>
                        <li>May amplify high-frequency noise.</li>
                        <li>Can create artifact "ringing" around sharp transitions.</li>
                    </ul>
                </p>
            """,

            "Bandpass Filter": """
                <h3>Bandpass Filter</h3>
                <p>Allows frequencies within a specific band to pass while attenuating others:</p>
                <ul>
                    <li><b>Purpose:</b> Isolate specific frequency components or remove noise outside a frequency band.</li>
                    <li><b>Parameters:</b>
                        <ul>
                            <li><b>Lower cutoff:</b> Lower boundary of the passband.</li>
                            <li><b>Upper cutoff:</b> Upper boundary of the passband.</li>
                            <li><b>Filter order:</b> Controls steepness of transitions.</li>
                        </ul>
                    </li>
                    <li><b>Applications:</b>
                        <ul>
                            <li>Isolating specific frequency bands (e.g., alpha waves in EEG)</li>
                            <li>Extracting signals in noisy environments</li>
                            <li>Communication channel selection</li>
                            <li>Musical instrument or voice isolation</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Interpretation:</b> After filtering, only components within the specified frequency band remain.</p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Requires accurate knowledge of the frequency band of interest.</li>
                        <li>Narrow bandpass filters can cause significant signal distortion.</li>
                        <li>May introduce phase shifts or time delays.</li>
                        <li>May create ringing artifacts, especially with high filter orders.</li>
                    </ul>
                </p>
            """,

            "Hilbert Transform": """
                <h3>Hilbert Transform</h3>
                <p>Creates the analytic signal and extracts instantaneous attributes:</p>
                <ul>
                    <li><b>Purpose:</b> Extract amplitude envelope, instantaneous phase, and frequency.</li>
                    <li><b>Usage:</b> Analyze modulated signals, extract signal envelope, frequency modulation analysis.</li>
                    <li><b>Components produced:</b>
                        <ul>
                            <li><b>Amplitude envelope:</b> Instantaneous amplitude of the signal.</li>
                            <li><b>Instantaneous phase:</b> Phase angle of the analytic signal.</li>
                            <li><b>Instantaneous frequency:</b> Rate of change of the phase.</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Applications:</b>
                    <ul>
                        <li>Demodulation of AM signals</li>
                        <li>Analysis of frequency modulation</li>
                        <li>Extracting temporal structure in complex signals</li>
                        <li>Speech and audio processing</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Best suited for narrowband signals (limited frequency range).</li>
                        <li>Instantaneous frequency may be difficult to interpret for broadband signals.</li>
                        <li>Edge effects at signal boundaries.</li>
                        <li>Phase unwrapping may introduce errors in instantaneous frequency.</li>
                    </ul>
                </p>
            """,

            "Energy Analysis": """
                <h3>Energy Analysis</h3>
                <p>Examines how signal energy is distributed over time intervals:</p>
                <ul>
                    <li><b>Purpose:</b> Identify energy variations over time, detect events or changes in signal activity.</li>
                    <li><b>Calculation:</b> Energy in interval = sum of squared values within each time window.</li>
                    <li><b>Applications:</b>
                        <ul>
                            <li>Speech segment detection</li>
                            <li>Activity monitoring in sensors</li>
                            <li>Transient detection</li>
                            <li>Signal quality assessment over time</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Interpretation:</b>
                    <ul>
                        <li>High energy intervals indicate greater signal activity or amplitude.</li>
                        <li>Sudden changes in energy can indicate events or transitions.</li>
                        <li>Energy distribution can reveal patterns in signal activity over time.</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Results depend on interval size choice (too small: noisy results, too large: temporal details lost).</li>
                        <li>May be sensitive to outliers or noise spikes.</li>
                        <li>Doesn't preserve frequency information.</li>
                        <li>May miss low-amplitude but important signal features.</li>
                    </ul>
                </p>
            """,

            "Phase Analysis": """
                <h3>Phase Analysis</h3>
                <p>Studies the phase behavior of a signal:</p>
                <ul>
                    <li><b>Purpose:</b> Understand angular position in oscillations, detect phase shifts or synchronization.</li>
                    <li><b>Calculation:</b> Extracts phase angle from analytic signal via Hilbert transform.</li>
                    <li><b>Key metrics:</b>
                        <ul>
                            <li><b>Phase consistency:</b> How stable the phase progression is.</li>
                            <li><b>Phase velocity:</b> Rate of phase change (related to frequency).</li>
                            <li><b>Phase jumps:</b> Sudden changes in phase that may indicate events.</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Applications:</b>
                    <ul>
                        <li>Brain connectivity analysis (phase synchronization)</li>
                        <li>Communication signal demodulation</li>
                        <li>Mechanical vibration analysis</li>
                        <li>Detecting coherence between signals</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Phase unwrapping may introduce artifacts in long signals.</li>
                        <li>Interpretation can be challenging for broadband signals.</li>
                        <li>Sensitive to noise, especially at low amplitudes.</li>
                        <li>Phase is only meaningful for oscillatory signals.</li>
                    </ul>
                </p>
            """,

            "Cepstral Analysis": """
                <h3>Cepstral Analysis</h3>
                <p>The "spectrum of the logarithm of the spectrum" - reveals periodic patterns in spectra:</p>
                <ul>
                    <li><b>Purpose:</b> Detect periodic structures in the spectrum, separate source and filter components.</li>
                    <li><b>Formula:</b> Inverse Fourier transform of the logarithm of the magnitude spectrum.</li>
                    <li><b>Key concepts:</b>
                        <ul>
                            <li><b>Quefrency:</b> The x-axis in cepstral domain (a form of time).</li>
                            <li><b>Rahmonics:</b> Peaks in the cepstrum (analogous to harmonics).</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Applications:</b>
                    <ul>
                        <li>Pitch detection in speech (fundamental frequency)</li>
                        <li>Echo detection and removal</li>
                        <li>Speech processing and recognition</li>
                        <li>Mechanical fault diagnosis (detecting periodicities)</li>
                    </ul>
                </p>
                <p><b>Interpretation:</b>
                    <ul>
                        <li>Peaks in the cepstrum represent periodic components in the original spectrum.</li>
                        <li>First significant peak indicates fundamental period or echo delay.</li>
                        <li>Lower quefrencies relate to spectral envelope, higher to fine structure.</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Interpretation can be complex without domain knowledge.</li>
                        <li>May require pre-processing for optimal results.</li>
                        <li>Performance degrades in noisy signals.</li>
                        <li>Less effective for signals with rapidly changing pitch.</li>
                    </ul>
                </p>
            """,

            "Cross-Correlation": """
                <h3>Cross-Correlation</h3>
                <p>Measures similarity between two different signals as a function of time lag:</p>
                <ul>
                    <li><b>Purpose:</b> Determine time delay between signals, measure similarity, detect common patterns.</li>
                    <li><b>Formula:</b> (f⋆g)(τ) = ∫f*(t)·g(t+τ)dt (or discrete equivalent)</li>
                    <li><b>Key results:</b>
                        <ul>
                            <li><b>Maximum correlation value:</b> Indicates degree of similarity (0-1 when normalized).</li>
                            <li><b>Lag at maximum:</b> Time offset that best aligns the signals.</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Applications:</b>
                    <ul>
                        <li>Finding signal delays (e.g., acoustics, radar)</li>
                        <li>Pattern detection across multiple sensors</li>
                        <li>Template matching in signal processing</li>
                        <li>Time difference of arrival (TDOA) calculations</li>
                        <li>Measuring similarity between related signals</li>
                    </ul>
                </p>
                <p><b>Interpretation:</b>
                    <ul>
                        <li>The peak in cross-correlation indicates the time lag that maximizes similarity.</li>
                        <li>Higher correlation values suggest stronger relationships between signals.</li>
                        <li>Multiple peaks may indicate repeating patterns or multiple path propagation.</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Assumes signals are related and have similar structures.</li>
                        <li>May be misleading if signals have different amplitude scales (normalization helps).</li>
                        <li>Sensitive to noise and outliers.</li>
                        <li>May detect spurious correlations in complex signals.</li>
                    </ul>
                </p>
            """,

            "Wavelet Transform": """
                <h3>Wavelet Transform</h3>
                <p>Decomposes a signal into components at different scales/frequencies with time localization:</p>
                <ul>
                    <li><b>Purpose:</b> Multi-resolution analysis providing both time and frequency information.</li>
                    <li><b>Key concepts:</b>
                        <ul>
                            <li><b>Approximation:</b> Low-frequency components of the signal.</li>
                            <li><b>Details:</b> High-frequency components at different scales.</li>
                            <li><b>Decomposition Level:</b> Number of scales analyzed (more levels = finer frequency division).</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Advantages over Fourier Transform:</b>
                    <ul>
                        <li>Provides both time and frequency information simultaneously.</li>
                        <li>Better suited for non-stationary signals with changing frequency content.</li>
                        <li>Adaptive resolution (fine time resolution at high frequencies, fine frequency resolution at low frequencies).</li>
                        <li>More effective at capturing transient events.</li>
                    </ul>
                </p>
                <p><b>Applications:</b>
                    <ul>
                        <li>Identifying transient events at different time scales</li>
                        <li>Denoising signals while preserving important features</li>
                        <li>Feature extraction for classification tasks</li>
                        <li>Image and audio compression</li>
                        <li>Biomedical signal processing (EEG, ECG analysis)</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>More complex to interpret than traditional spectral analysis.</li>
                        <li>Choice of wavelet family affects results.</li>
                        <li>Edge effects at signal boundaries.</li>
                        <li>Computational intensity increases with decomposition levels.</li>
                    </ul>
                </p>
            """,

            "Wavelet Types": """
                <h3>Wavelet Types</h3>
                <p>Different wavelet families have unique characteristics suited to specific signal types:</p>
                <ul>
                    <li><b>Haar:</b>
                        <ul>
                            <li>The simplest wavelet, resembling a step function.</li>
                            <li>Good for detecting abrupt transitions and edges.</li>
                            <li>Limited smoothness, resulting in blocky approximations.</li>
                            <li>Best for: Signals with sudden jumps or digital/binary signals.</li>
                        </ul>
                    </li>
                    <li><b>Daubechies (db4, db8):</b>
                        <ul>
                            <li>Compactly supported wavelets with maximum number of vanishing moments.</li>
                            <li>Good balance between smoothness and localization.</li>
                            <li>Higher order (db8) provides smoother representation than lower order (db4).</li>
                            <li>Best for: General-purpose analysis, signals with polynomial trends.</li>
                        </ul>
                    </li>
                    <li><b>Symlets (sym4, sym8):</b>
                        <ul>
                            <li>Modified version of Daubechies wavelets with increased symmetry.</li>
                            <li>Nearly symmetrical, reducing phase distortion.</li>
                            <li>Good time-frequency localization properties.</li>
                            <li>Best for: Applications where phase information is important.</li>
                        </ul>
                    </li>
                    <li><b>Coiflets (coif1, coif3):</b>
                        <ul>
                            <li>More symmetrical than Daubechies wavelets.</li>
                            <li>Have vanishing moments for both wavelet and scaling functions.</li>
                            <li>Good for preserving signal features during analysis/reconstruction.</li>
                            <li>Best for: Function approximation, signals requiring accurate reconstruction.</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Selection criteria:</b>
                    <ul>
                        <li>Signal characteristics (smooth vs. abrupt changes)</li>
                        <li>Analysis goals (detection, denoising, compression)</li>
                        <li>Required frequency resolution</li>
                        <li>Computational constraints</li>
                    </ul>
                </p>
            """,

            "Wavelet Applications": """
                <h3>Wavelet Applications</h3>
                <p>Common applications of wavelet analysis in signal processing:</p>
                <ul>
                    <li><b>Signal Denoising:</b>
                        <ul>
                            <li>Wavelets can separate signal from noise at different scales.</li>
                            <li>Thresholding detail coefficients removes noise while preserving signal features.</li>
                            <li>More effective than traditional filtering for preserving edges and transients.</li>
                        </ul>
                    </li>
                    <li><b>Feature Detection:</b>
                        <ul>
                            <li>Identify specific patterns or events at appropriate scales.</li>
                            <li>Useful for detecting discontinuities, spikes, or other transient events.</li>
                            <li>Can locate features that are difficult to detect in time or frequency domain alone.</li>
                        </ul>
                    </li>
                    <li><b>Compression:</b>
                        <ul>
                            <li>Many signals can be represented with few wavelet coefficients.</li>
                            <li>Discarding small coefficients enables efficient storage while preserving essential information.</li>
                            <li>Basis for JPEG2000 image compression standard.</li>
                        </ul>
                    </li>
                    <li><b>Component Separation:</b>
                        <ul>
                            <li>Isolate different physical processes operating at different scales.</li>
                            <li>Extract specific signal components by focusing on relevant decomposition levels.</li>
                            <li>Separate fast vs. slow processes in complex signals.</li>
                        </ul>
                    </li>
                    <li><b>Non-Stationary Signal Analysis:</b>
                        <ul>
                            <li>Track how frequency content changes over time.</li>
                            <li>Identify time-varying behavior that Fourier analysis would miss.</li>
                            <li>Particularly valuable for biological signals, seismic data, or financial time series.</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Implementation approach:</b>
                    <ul>
                        <li>First select appropriate wavelet family for your signal.</li>
                        <li>Choose decomposition level based on frequency resolution needs.</li>
                        <li>Examine both approximation (general trend) and details (specific scales).</li>
                        <li>Consider energy distribution across levels to identify important components.</li>
                    </ul>
                </p>
            """
        }

        if self.parent_dialog:
            if topic in help_content:
                self.parent_dialog.show_help_in_results(topic, help_content[topic])
            else:
                self.parent_dialog.show_help_in_results(topic, f"<h3>No help available for '{topic}'</h3><p>Please select another topic.</p>")


# ======================
# Helper Functions
# ======================

import numpy as np
from PySide6.QtWidgets import QMessageBox

def prepare_signal_for_analysis(dialog, values, title="Signal Processing"):
    """
    Prepares a signal for analysis by ensuring all values are positive and handling special cases.

    Args:
        dialog (QDialog): Parent dialog for showing messages.
        values (np.ndarray): Signal values.
        title (str): Title for the message box.

    Returns:
        np.ndarray: Processed signal values, or None if canceled.
    """
    Logger.log_message_static(f"Preparing signal for {title}", Logger.DEBUG)

    try:
        # Check for NaN or Inf values
        if np.any(~np.isfinite(values)):
            bad_values = np.sum(~np.isfinite(values))
            total_values = len(values)
            percent_bad = (bad_values / total_values) * 100 if total_values > 0 else 0

            Logger.log_message_static(f"Signal contains {bad_values} non-finite values ({percent_bad:.2f}%)", Logger.WARNING)

            msg = QMessageBox(dialog)
            msg.setWindowTitle(title)
            msg.setText(f"Signal contains {bad_values} non-finite values ({percent_bad:.2f}%).")
            msg.setInformativeText("How would you like to proceed?")
            replace_button = msg.addButton("Replace with zeros", QMessageBox.ButtonRole.AcceptRole)
            interpolate_button = msg.addButton("Interpolate", QMessageBox.ButtonRole.AcceptRole)
            cancel_button = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

            Logger.log_message_static("Prompting user for handling non-finite values", Logger.INFO)
            msg.exec()

            if msg.clickedButton() == cancel_button:
                Logger.log_message_static("User canceled signal preparation", Logger.DEBUG)
                return None
            elif msg.clickedButton() == replace_button:
                Logger.log_message_static("Replacing non-finite values with zeros", Logger.DEBUG)
                values[~np.isfinite(values)] = 0.0
            elif msg.clickedButton() == interpolate_button:
                Logger.log_message_static("Interpolating non-finite values", Logger.DEBUG)
                mask = np.isfinite(values)
                values = np.interp(np.arange(len(values)), np.where(mask)[0], values[mask])

        # Check if all values are positive
        if np.all(values >= 0):
            Logger.log_message_static("All values are positive, returning unchanged", Logger.INFO)
            return values

        # Check if all values are negative
        if np.all(values < 0):
            Logger.log_message_static("All values are negative, flipping to positive", Logger.WARNING)
            QMessageBox.information(dialog, title, "All values are negative. They will be flipped to positive.")
            return np.abs(values)

        # Check for a mix of positive and negative values
        total_values = len(values)
        negative_count = np.sum(values < 0)
        negative_ratio = negative_count / total_values

        if negative_ratio < 0.05:  # Negligible negative values
            Logger.log_message_static("Negligible negative values, replacing with near-zero", Logger.WARNING)
            QMessageBox.information(dialog, title, "Negligible negative values detected. Replacing them with near-zero.")
            values[values < 0] = 1e-10
            return values

        # Significant mix of positive and negative values
        Logger.log_message_static("Significant mix of positive and negative values, prompting user", Logger.WARNING)
        msg = QMessageBox(dialog)
        msg.setWindowTitle(title)
        msg.setText("The signal contains a significant mix of positive and negative values.")
        msg.setInformativeText("How would you like to proceed?")
        abs_button = msg.addButton("Use absolute values", QMessageBox.ButtonRole.AcceptRole)
        pos_button = msg.addButton("Replace negatives with near-zero", QMessageBox.ButtonRole.AcceptRole)
        neg_button = msg.addButton("Replace positives with near-zero", QMessageBox.ButtonRole.AcceptRole)
        cancel_button = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

        msg.exec()

        if msg.clickedButton() == cancel_button:
            Logger.log_message_static("User canceled signal preparation", Logger.DEBUG)
            return None
        elif msg.clickedButton() == abs_button:
            Logger.log_message_static("User chose to use absolute values", Logger.DEBUG)
            return np.abs(values)
        elif msg.clickedButton() == pos_button:
            Logger.log_message_static("User chose to replace negatives with near-zero", Logger.DEBUG)
            values[values < 0] = 1e-10
            return values
        elif msg.clickedButton() == neg_button:
            Logger.log_message_static("User chose to replace positives with near-zero", Logger.DEBUG)
            values[values > 0] = 1e-10
            return values

    except Exception as e:
        Logger.log_message_static(f"Error in signal preparation: {str(e)}", Logger.ERROR)
        return None

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