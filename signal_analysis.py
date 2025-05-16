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
from PySide6.QtWidgets import (QDialog, QPushButton, QTableWidget, QTableWidgetItem,
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
        Logger.log_message_static("Initializing SignalAnalysisDialog", Logger.INFO)
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

            # Process the signal values directly
            Logger.log_message_static("Preparing signal for PSD processing", Logger.DEBUG)
            processed_values = prepare_signal_for_analysis(self, values, "PSD Input Signal")
            if processed_values is None:
                Logger.log_message_static("PSD analysis canceled by user", Logger.INFO)
                return  # User canceled the operation

            # Calculate PSD
            Logger.log_message_static("Calculating sampling frequency for PSD", Logger.DEBUG)
            fs = 1 / np.mean(np.diff(time_arr)) if len(time_arr) > 1 else 1  # Sampling frequency
            Logger.log_message_static(f"Using sampling frequency of {fs} Hz for PSD calculation", Logger.DEBUG)

            Logger.log_message_static("Computing Welch PSD with nperseg=256", Logger.DEBUG)
            freqs, psd = sc_signal.welch(processed_values, fs=fs, nperseg=256)
            Logger.log_message_static(f"PSD calculation complete, {len(freqs)} frequency bins generated", Logger.DEBUG)

            # Calculate some PSD statistics
            peak_idx = np.argmax(psd)
            peak_freq = freqs[peak_idx]
            max_power_db = 10 * np.log10(np.max(psd))
            total_power = np.sum(psd)

            Logger.log_message_static(f"PSD peak frequency: {peak_freq} Hz, max power: {max_power_db} dB", Logger.DEBUG)

            psd_stats = {
                "Peak Frequency (Hz)": peak_freq,
                "Max Power (dB)": max_power_db,
                "Total Power": total_power
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
            Logger.log_message_static("Converting PSD to dB for visualization", Logger.DEBUG)
            psd_db = 10 * np.log10(psd)
            p.plot(freqs, psd_db, pen='g')

            # Add reference line at peak frequency
            Logger.log_message_static(f"Adding peak frequency marker at {peak_freq} Hz", Logger.DEBUG)
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

            # Process the signal values
            Logger.log_message_static("Preparing signal for autocorrelation", Logger.DEBUG)
            processed_values = prepare_signal_for_analysis(self, values, "Autocorrelation Input Signal")
            if processed_values is None:
                Logger.log_message_static("Autocorrelation analysis canceled by user", Logger.INFO)
                return  # User canceled the operation

            # Calculate autocorrelation
            Logger.log_message_static("Computing autocorrelation", Logger.DEBUG)
            autocorr = correlate(processed_values, processed_values, mode='full')

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
                dt = np.mean(np.diff(time_arr))
                lags = lags * dt
                Logger.log_message_static(f"Using time step of {dt}s for lag calculation", Logger.DEBUG)

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
            p.setLabel('bottom', 'Lag')
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

        except KeyError:
            Logger.log_message_static(f"Signal '{signal}' not found in data_signals", Logger.ERROR)
        except Exception as e:
            Logger.log_message_static(f"Error in autocorrelation analysis: {str(e)}", Logger.ERROR)

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
                Wn = [filter_params["cutoff"] / (fs / 2), filter_params["cutoff2"] / (fs / 2)]
                if Wn[0] >= Wn[1]:
                    Logger.log_message_static("Lower cutoff must be less than upper cutoff", Logger.WARNING)
                    return
                Logger.log_message_static(f"Using normalized cutoff frequencies: {Wn}", Logger.DEBUG)
            else:
                # For lowpass and highpass, single cutoff frequency
                Wn = filter_params["cutoff"] / (fs / 2)
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

            # Process the signal values
            Logger.log_message_static("Preparing signal for energy analysis", Logger.DEBUG)
            processed_values = prepare_signal_for_analysis(self, values, "Energy Analysis Input Signal")
            if processed_values is None:
                Logger.log_message_static("Energy analysis canceled by user", Logger.INFO)
                return  # User canceled the operation

            # Calculate sample rate
            if len(time_arr) < 2:
                Logger.log_message_static("Signal too short to determine sample rate", Logger.WARNING)
                return

            Logger.log_message_static("Calculating sample rate", Logger.DEBUG)
            fs = 1 / np.mean(np.diff(time_arr))
            Logger.log_message_static(f"Sample rate: {fs} Hz", Logger.DEBUG)

            # Calculate energy in time domain
            Logger.log_message_static("Computing time domain energy", Logger.DEBUG)
            energy_time = np.sum(np.abs(processed_values) ** 2)

            # Calculate energy density in frequency domain
            Logger.log_message_static("Computing FFT for frequency domain energy", Logger.DEBUG)
            n = len(processed_values)
            fft_values = np.fft.rfft(processed_values)
            freqs = np.fft.rfftfreq(n, d=1 / fs)
            energy_density = np.abs(fft_values) ** 2

            # Calculate cumulative energy
            Logger.log_message_static("Computing cumulative energy", Logger.DEBUG)
            cumulative_energy = np.cumsum(energy_density) / np.sum(energy_density)

            # Find the frequency below which X% of the energy is contained
            energy_thresholds = [0.5, 0.9, 0.95, 0.99]
            threshold_freqs = []

            Logger.log_message_static("Finding energy threshold frequencies", Logger.DEBUG)
            for threshold in energy_thresholds:
                idx = np.where(cumulative_energy >= threshold)[0]
                freq = freqs[idx[0]] if len(idx) > 0 else freqs[-1]
                threshold_freqs.append(freq)
                Logger.log_message_static(f"{threshold * 100}% energy frequency threshold: {freq} Hz", Logger.DEBUG)

            # Create a proper window
            Logger.log_message_static("Creating energy analysis plot window", Logger.DEBUG)
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle(f"Energy Analysis: {signal}")
            plot_window.resize(800, 600)

            # Create central widget with layout
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            # Create plot widget with 2 subplots
            plot_widget = pg.GraphicsLayoutWidget()

            # 1. Energy Spectral Density
            p1 = plot_widget.addPlot(row=0, col=0)
            p1.setTitle("Energy Spectral Density")
            p1.setLabel('left', 'Energy Density')
            p1.setLabel('bottom', 'Frequency (Hz)')
            p1.plot(freqs, energy_density, pen='b')
            p1.setLogMode(x=True, y=True)  # Log scale for better visualization

            # 2. Cumulative Energy Distribution
            p2 = plot_widget.addPlot(row=1, col=0)
            p2.setTitle("Cumulative Energy Distribution")
            p2.setLabel('left', 'Cumulative Energy (%)')
            p2.setLabel('bottom', 'Frequency (Hz)')
            p2.plot(freqs, cumulative_energy * 100, pen='g')  # As percentage
            p2.setLogMode(x=True, y=False)

            # Add threshold markers
            Logger.log_message_static("Adding threshold markers to cumulative energy plot", Logger.DEBUG)
            for threshold, freq in zip(energy_thresholds, threshold_freqs):
                # Horizontal line at threshold level
                h_line = pg.InfiniteLine(pos=threshold * 100, angle=0,
                                         pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
                p2.addItem(h_line)

                # Vertical line at threshold frequency
                v_line = pg.InfiniteLine(pos=freq, angle=90, pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
                p2.addItem(v_line)

                # Label for the threshold
                text = pg.TextItem(text=f"{threshold * 100}%", color='r', anchor=(0, 1))
                text.setPos(freq, threshold * 100)
                p2.addItem(text)

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

            # Calculate some summary statistics
            Logger.log_message_static("Calculating summary statistics for energy analysis", Logger.DEBUG)
            energy_stats = {
                "Total Energy": energy_time,
                "50% Energy Frequency": threshold_freqs[0],
                "90% Energy Frequency": threshold_freqs[1],
                "95% Energy Frequency": threshold_freqs[2],
                "99% Energy Frequency": threshold_freqs[3],
                "Peak Energy Frequency": freqs[np.argmax(energy_density)],
                "Peak Energy Density": np.max(energy_density)
            }

            self.show_analysis_results("Energy Analysis", signal, energy_stats)
            Logger.log_message_static("Energy analysis complete", Logger.INFO)

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
            img.scale(time_arr[-1] / abs_coeffs.shape[1], 1)
            img.translate(0, 0)  # Start at time zero

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
                                                    WARNING)

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

class ExplanationTab(QWidget):
    """
    Tab providing explanations of the various analysis methods.

    This helps users understand the different analysis techniques available,
    their applications, and how to interpret the results.
    """

    def __init__(self, parent=None):
        """
        Initialize the explanations tab with accordion-style sections.

        Args:
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        Logger.log_message_static("Initializing ExplanationTab", Logger.DEBUG)
        super().__init__(parent)

        # Main layout
        main_layout = QVBoxLayout(self)

        # Create scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        # Container widget for the scroll area
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Add explanations as expandable groups
        Logger.log_message_static("Creating explanation sections", Logger.DEBUG)
        self.add_explanation_group(scroll_layout, "Basic Statistics",
                                   """Calculates fundamental statistical properties of the signal:

 - Mean: Average value of the signal
 - Median: Middle value when sorted
 - Standard Deviation: Measure of signal variation
 - Min/Max: Extreme values in the signal
 - RMS: Root Mean Square (power-related measure)
 - Kurtosis: Measure of "tailedness" of the distribution
 - Skewness: Measure of asymmetry of the distribution
                                   """)

        self.add_explanation_group(scroll_layout, "FFT Analysis",
                                   """Fast Fourier Transform converts a signal from time domain to frequency domain.

 - Reveals the frequency components present in the signal
 - Shows the amplitude/magnitude of each frequency component
 - Helps identify dominant frequencies and harmonics
 - Useful for finding oscillations and periodic patterns
                                   """)

        self.add_explanation_group(scroll_layout, "Time Domain Analysis",
                                   """Analyzes signal characteristics directly in the time domain.

 - Rise/Fall times: How quickly signal transitions
 - Peak characteristics: Height, width, spacing of peaks
 - Signal envelopes: Upper and lower bounds of the signal
 - Crossing statistics: Zero or threshold crossing frequency
                                   """)

        self.add_explanation_group(scroll_layout, "Power Spectral Density",
                                   """Shows how power is distributed across frequencies.

 - Different from FFT by focusing on power rather than amplitude
 - Helps identify which frequencies contain most energy
 - Often presented in logarithmic scale (dB)
 - Useful for noise analysis and filter design
                                   """)

        self.add_explanation_group(scroll_layout, "Autocorrelation",
                                   """Correlation of a signal with a delayed copy of itself.

 - Reveals repeating patterns or periodicities
 - Helps identify signal periodicity and fundamental frequency
 - Peak at zero lag, with repeating peaks at the signal's period
 - Useful for detecting hidden periodicities and noise analysis
                                   """)

        self.add_explanation_group(scroll_layout, "Peak Detection",
                                   """Identifies and analyzes peaks (maxima) or valleys (minima) in a signal.

 - Detects pulse-like events or oscillation peaks
 - Provides statistics on peak heights, widths, and spacing
 - Useful for event counting and characterizing oscillatory signals
 - Can be adjusted with prominence and width thresholds
                                   """)

        self.add_explanation_group(scroll_layout, "Filtering",
                                   """Applies filters to remove unwanted frequency components.

 - Low-pass: Keeps low frequencies, removes high frequencies
 - High-pass: Keeps high frequencies, removes low frequencies
 - Band-pass: Keeps a specific frequency band
 - Band-stop: Removes a specific frequency band
 - Useful for noise reduction and signal isolation
                                   """)

        self.add_explanation_group(scroll_layout, "Hilbert Transform",
                                   """Creates an analytic signal from a real signal.

 - Extracts instantaneous amplitude (envelope)
 - Provides instantaneous phase information
 - Calculates instantaneous frequency
 - Useful for analyzing modulated signals and extracting envelopes
                                   """)

        self.add_explanation_group(scroll_layout, "Energy Analysis",
                                   """Examines how energy is distributed in the signal.

 - Total energy content of the signal
 - Energy distribution across frequency bands
 - Energy distribution over time (spectrogram)
 - Useful for characterizing signal strength and identifying energy-rich regions
                                   """)

        self.add_explanation_group(scroll_layout, "Phase Analysis",
                                   """Studies the phase components of a signal.

 - Phase spectrum reveals timing relationships
 - Phase coherence measures similarity between signals
 - Phase distortion can indicate nonlinearities
 - Useful for understanding signal composition and modulation
                                   """)

        self.add_explanation_group(scroll_layout, "Cepstral Analysis",
                                   """The spectrum of the logarithm of the spectrum.

 - Separates source and filter components
 - Reveals harmonic structure and periodicity
 - Used in speech analysis to find fundamental frequency
 - Helpful for detecting echoes and harmonics
                                   """)

        self.add_explanation_group(scroll_layout, "Wavelet Transform",
                                   """Time-frequency analysis with adaptive window size.

 - Provides both time and frequency information simultaneously
 - Better time resolution at high frequencies than STFT
 - Better frequency resolution at low frequencies
 - Useful for analyzing non-stationary signals with varying frequency content
                                   """)

        self.add_explanation_group(scroll_layout, "Cross Correlation",
                                   """Measures similarity between two signals as a function of time lag.

 - Reveals time shifts between similar signals
 - Quantifies similarity between signals
 - Peak at the lag where signals best align
 - Useful for signal alignment, time delay estimation, and pattern matching
                                   """)

        # Set the scroll widget
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
        Logger.log_message_static("ExplanationTab initialization complete", Logger.DEBUG)

    def add_explanation_group(self, layout, title, text):
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

# ======================
# Helper Functions
# ======================

def prepare_signal_for_analysis(dialog, values, title="Signal Processing"):
    """
    Prepares a signal for analysis by handling special cases.

    Args:
        dialog (QDialog): Parent dialog for showing messages
        values (np.ndarray): Signal values
        title (str): Title for the message box

    Returns:
        np.ndarray: Processed signal values, None if canceled
    """
    Logger.log_message_static(f"Preparing signal for {title}", Logger.DEBUG)
    try:
        # Check for NaN or Inf values
        if np.any(~np.isfinite(values)):
            bad_values = np.sum(~np.isfinite(values))
            total_values = len(values)
            percent_bad = (bad_values / total_values) * 100 if total_values > 0 else 0

            Logger.log_message_static(
                f"Signal contains {bad_values} non-finite values ({percent_bad:.2f}%)", Logger.WARNING)

            # Offer options to the user
            if bad_values / total_values > 0.5:  # More than 50% bad values
                Logger.log_message_static(
                    "More than 50% of values are non-finite, analysis may not be meaningful", Logger.WARNING)
                # We could return None here, but we'll let the user decide

            # Ask the user what to do
            from PySide6.QtWidgets import QMessageBox
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
                Logger.log_message_static("User canceled analysis due to non-finite values", Logger.INFO)
                return None
            elif msg.clickedButton() == replace_button:
                Logger.log_message_static("Replacing non-finite values with zeros", Logger.DEBUG)
                # Create a copy to avoid modifying the original
                values_copy = np.copy(values)
                values_copy[~np.isfinite(values_copy)] = 0.0
                return values_copy
            elif msg.clickedButton() == interpolate_button:
                Logger.log_message_static("Interpolating non-finite values", Logger.DEBUG)
                # Create mask of valid values
                mask = np.isfinite(values)
                if not np.any(mask):
                    Logger.log_message_static("No valid values to interpolate from, returning zeros",
                                                    Logger.WARNING)
                    return np.zeros_like(values)

                # Create indices array
                indices = np.arange(len(values))

                # Interpolate
                values_copy = np.copy(values)
                values_copy[~mask] = np.interp(indices[~mask], indices[mask], values[mask])
                Logger.log_message_static(f"Interpolated {bad_values} values", Logger.DEBUG)
                return values_copy

        # If we reach here, the signal is already good
        Logger.log_message_static("Signal is ready for analysis (all values are finite)", Logger.DEBUG)
        return values

    except Exception as e:
        Logger.log_message_static(f"Error preparing signal: {str(e)}", Logger.ERROR)
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