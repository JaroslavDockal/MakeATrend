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
import scipy.signal as signal
from scipy.signal import hilbert, butter, filtfilt, correlate
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QPushButton, QTableWidget,
                             QTableWidgetItem, QHeaderView, QComboBox, QLabel,
                             QWidget, QHBoxLayout, QGroupBox, QScrollArea, QTabWidget,
                             QFormLayout, QDoubleSpinBox, QRadioButton, QButtonGroup,
                               QMainWindow, QApplication)


class SignalAnalysisTools:
    """
    Basic signal analysis tools for analyzing time-series data.
    """

    @staticmethod
    def calculate_statistics(parent, signal_name):
        """
        Calculate basic statistics for a signal.

        Args:
            parent: The parent application instance containing data_signals.
            signal_name (str): Name of the signal to analyze.

        Returns:
            dict: Dictionary of statistical metrics.
        """
        _, values = parent.data_signals[signal_name]

        return {
            "Mean": np.mean(values),
            "Median": np.median(values),
            "Min": np.min(values),
            "Max": np.max(values),
            "Range": np.max(values) - np.min(values),
            "Std Dev": np.std(values),
            "Variance": np.var(values),
            "RMS": np.sqrt(np.mean(np.square(values))),
            "Samples": len(values)
        }

    @staticmethod
    def perform_fft_analysis(parent, signal_name):
        """
        Perform FFT analysis on a signal and display the results.

        Args:
            parent: The parent application instance containing data_signals.
            signal_name (str): Name of the signal to analyze.
        """
        time_arr, values = parent.data_signals[signal_name]

        # Calculate sampling frequency and adjust if irregular
        fs = 1 / np.mean(np.diff(time_arr))

        # Compute FFT
        n = len(values)
        fft_values = np.fft.rfft(values)
        freqs = np.fft.rfftfreq(n, 1 / fs)
        magnitudes = np.abs(fft_values) / n * 2  # Scale appropriately

        # Create plot window
        plot_window = pg.GraphicsLayoutWidget(title=f"FFT Analysis: {signal_name}")
        plot_window.setWindowTitle(f"FFT Analysis: {signal_name}")
        plot_window.raise_()
        plot_window.resize(800, 600)

        # Time domain plot
        p1 = plot_window.addPlot(row=0, col=0)
        p1.setTitle("Time Domain")
        p1.setLabel('left', 'Amplitude')
        p1.setLabel('bottom', 'Time (s)')
        p1.plot(time_arr, values, pen='b')

        # Frequency domain plot
        p2 = plot_window.addPlot(row=1, col=0)
        p2.setTitle("Frequency Domain")
        p2.setLabel('left', 'Magnitude')
        p2.setLabel('bottom', 'Frequency (Hz)')
        p2.plot(freqs, magnitudes, pen='r')

        # Set log scale for better visualization
        p2.setLogMode(x=True, y=False)

        plot_window.show()
        plot_window.raise_()

        # Keep a reference to prevent garbage collection
        if not hasattr(parent, '_fft_windows'):
            parent._fft_windows = []
        parent._fft_windows.append(plot_window)

    @staticmethod
    def analyze_signal(parent, signal_name):
        """
        Perform comprehensive time domain analysis of a signal.

        Args:
            parent: The parent application instance containing data_signals.
            signal_name (str): Name of the signal to analyze.

        Returns:
            dict: Dictionary of analysis results.
        """
        time_arr, values = parent.data_signals[signal_name]

        # Time domain analysis
        duration = time_arr[-1] - time_arr[0]
        sample_rate = len(values) / duration

        # Calculate zero crossings
        zero_crossings = np.sum(np.diff(np.signbit(values).astype(int)) != 0)

        # Calculate signal energy and power
        energy = np.sum(values ** 2)
        power = energy / len(values)

        return {
            "Duration (s)": duration,
            "Sample Rate (Hz)": sample_rate,
            "Zero Crossings": zero_crossings,
            "Signal Energy": energy,
            "Signal Power": power,
            "Crest Factor": np.max(np.abs(values)) / np.sqrt(np.mean(values ** 2)) if np.mean(values ** 2) > 0 else 0
        }

class AdvancedSignalAnalysisTools:
    """
    A collection of advanced signal analysis methods for processing time-series signal data.
    """

    @staticmethod
    def calculate_psd(parent, signal_name):
        """
        Calculate the Power Spectral Density (PSD) of a signal.

        Args:
            parent: The parent application instance containing data_signals.
            signal_name (str): Name of the signal to analyze.

        Returns:
            tuple: Frequencies and PSD values.
        """
        time_arr, values = parent.data_signals[signal_name]
        fs = 1 / np.mean(np.diff(time_arr))  # Sampling frequency
        freqs, psd = signal.welch(values, fs=fs, nperseg=256)
        return freqs, psd

    @staticmethod
    def calculate_autocorrelation(parent, signal_name):
        """
        Calculate the autocorrelation of a signal.

        Args:
            parent: The parent application instance containing data_signals.
            signal_name (str): Name of the signal to analyze.

        Returns:
            tuple: Lag times and autocorrelation values.
        """
        time_arr, values = parent.data_signals[signal_name]
        autocorr = np.correlate(values, values, mode='full')
        # Normalize
        autocorr = autocorr / np.max(autocorr)
        lags = np.arange(-len(values) + 1, len(values))
        return lags * np.mean(np.diff(time_arr)), autocorr

    @staticmethod
    def detect_peaks(parent, signal_name, height=None, distance=None):
        """
        Detect peaks in a signal and analyze their properties.

        Args:
            parent: The parent application instance containing data_signals.
            signal_name (str): Name of the signal to analyze.
            height (float, optional): Minimum height of peaks.
            distance (int, optional): Minimum distance between peaks.

        Returns:
            dict: Dictionary containing peak indices, times, and values.
        """
        time_arr, values = parent.data_signals[signal_name]
        peaks, properties = signal.find_peaks(values, height=height, distance=distance)
        
        # Calculate additional peak properties
        peak_heights = values[peaks]
        peak_widths = signal.peak_widths(values, peaks, rel_height=0.5)[0]
        
        return {
            "Count": len(peaks),
            "Indices": peaks,
            "Times": time_arr[peaks],
            "Heights": peak_heights,
            "Mean Height": np.mean(peak_heights) if len(peak_heights) > 0 else 0,
            "Max Height": np.max(peak_heights) if len(peak_heights) > 0 else 0,
            "Mean Width": np.mean(peak_widths) if len(peak_widths) > 0 else 0
        }

    @staticmethod
    def apply_filter(parent, signal_name, filter_type, cutoff_low, cutoff_high=None, order=4):
        """
        Apply filter to signal.

        Args:
            parent: The parent application instance containing data_signals.
            signal_name (str): Name of the signal to analyze.
            filter_type (str): Type of filter ('lowpass', 'highpass', 'bandpass').
            cutoff_low (float): Lower cutoff frequency.
            cutoff_high (float, optional): Higher cutoff frequency for bandpass filter.
            order (int): Filter order.

        Returns:
            tuple: Time array and filtered signal values.
        """
        time_arr, values = parent.data_signals[signal_name]
        fs = 1 / np.mean(np.diff(time_arr))  # Sampling frequency
        
        nyq = 0.5 * fs
        low = cutoff_low / nyq
        
        if filter_type == 'lowpass':
            b, a = butter(order, low, btype='low')
        elif filter_type == 'highpass':
            b, a = butter(order, low, btype='high')
        elif filter_type == 'bandpass':
            high = cutoff_high / nyq if cutoff_high else 0.99
            b, a = butter(order, [low, high], btype='band')
        else:
            return time_arr, values
            
        filtered = filtfilt(b, a, values)
        return time_arr, filtered

    @staticmethod
    def hilbert_transform(parent, signal_name):
        """
        Apply Hilbert transform to compute signal envelope and instantaneous frequency.

        Args:
            parent: The parent application instance containing data_signals.
            signal_name (str): Name of the signal to analyze.

        Returns:
            dict: Dictionary with envelope, instantaneous phase, and frequency.
        """
        time_arr, values = parent.data_signals[signal_name]
        analytic_signal = hilbert(values)
        
        # Envelope is the magnitude of the analytic signal
        envelope = np.abs(analytic_signal)
        
        # Instantaneous phase
        inst_phase = np.unwrap(np.angle(analytic_signal))
        
        # Instantaneous frequency is the derivative of the phase
        dt = np.mean(np.diff(time_arr))
        inst_freq = np.diff(inst_phase) / (2.0 * np.pi * dt)
        
        return {
            "envelope": (time_arr, envelope),
            "phase": (time_arr, inst_phase),
            "frequency": (time_arr[:-1], inst_freq)
        }

    @staticmethod
    def cross_correlation(parent, signal1_name, signal2_name):
        """
        Calculate cross-correlation between two signals.

        Args:
            parent: The parent application instance containing data_signals.
            signal1_name (str): Name of the first signal.
            signal2_name (str): Name of the second signal.

        Returns:
            tuple: Lag times and correlation values.
        """
        time1, values1 = parent.data_signals[signal1_name]
        time2, values2 = parent.data_signals[signal2_name]
        
        # Ensure signals have the same length by padding with zeros
        if len(values1) > len(values2):
            values2 = np.pad(values2, (0, len(values1) - len(values2)))
        elif len(values2) > len(values1):
            values1 = np.pad(values1, (0, len(values2) - len(values1)))
            
        corr = correlate(values1, values2, mode='full')
        # Normalize
        corr = corr / np.sqrt(np.sum(values1**2) * np.sum(values2**2))
        
        dt = np.mean(np.diff(time1))
        lags = np.arange(-len(values1) + 1, len(values1))
        lag_times = lags * dt
        
        # Find maximum correlation and corresponding lag
        max_corr_idx = np.argmax(np.abs(corr))
        max_lag = lag_times[max_corr_idx]
        
        return lag_times, corr, max_lag
        
    @staticmethod
    def wavelet_transform(parent, signal_name, wavelet='db4', level=5):
        """
        Perform wavelet transform on the signal.

        Args:
            parent: The parent application instance containing data_signals.
            signal_name (str): Name of the signal to analyze.
            wavelet (str): Wavelet type.
            level (int): Decomposition level.

        Returns:
            tuple: List of coefficients and approximation.
        """
        _, values = parent.data_signals[signal_name]
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(values, wavelet, level=level)
        approx = coeffs[0]  # Approximation
        details = coeffs[1:]  # Details
        
        return coeffs, approx, details

    @staticmethod
    def energy_in_intervals(parent, signal_name, num_intervals=10):
        """
        Analyze energy in time intervals.

        Args:
            parent: The parent application instance containing data_signals.
            signal_name (str): Name of the signal to analyze.
            num_intervals (int): Number of intervals to divide signal.

        Returns:
            tuple: Interval centers and energy values.
        """
        time_arr, values = parent.data_signals[signal_name]
        
        # Divide signal into intervals
        interval_size = len(values) // num_intervals
        intervals = [values[i:i+interval_size] for i in range(0, len(values), interval_size) if i + interval_size <= len(values)]
        
        # Calculate energy in each interval
        energy = [np.sum(interval**2) for interval in intervals]
        
        # Center time of each interval
        interval_centers = []
        for i in range(len(intervals)):
            start_idx = i * interval_size
            end_idx = min(start_idx + interval_size - 1, len(time_arr) - 1)
            interval_centers.append((time_arr[start_idx] + time_arr[end_idx]) / 2)
        
        return interval_centers, energy

    @staticmethod
    def phase_analysis(parent, signal_name):
        """
        Analyze phase of a signal.

        Args:
            parent: The parent application instance containing data_signals.
            signal_name (str): Name of the signal to analyze.

        Returns:
            tuple: Time array and phase values.
        """
        time_arr, values = parent.data_signals[signal_name]
        analytic_signal = hilbert(values)
        phase = np.unwrap(np.angle(analytic_signal))
        
        # Calculate phase statistics
        phase_stats = {
            "Mean Phase": np.mean(phase),
            "Phase Std Dev": np.std(phase),
            "Phase Range": np.max(phase) - np.min(phase)
        }
        
        return time_arr, phase, phase_stats

    @staticmethod
    def cepstrum_analysis(parent, signal_name):
        """
        Perform cepstral analysis of a signal.

        Args:
            parent: The parent application instance containing data_signals.
            signal_name (str): Name of the signal to analyze.

        Returns:
            tuple: Quefrency and cepstrum values.
        """
        time_arr, values = parent.data_signals[signal_name]
        
        # Calculate real cepstrum
        spectrum = np.fft.fft(values)
        log_spectrum = np.log(np.abs(spectrum) + 1e-10)  # Add small value to avoid log(0)
        cepstrum = np.fft.ifft(log_spectrum).real
        
        # Generate quefrency axis
        dt = np.mean(np.diff(time_arr))
        quefrency = np.arange(len(cepstrum)) * dt
        
        return quefrency, cepstrum


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
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Signal Analysis")
        self.resize(800, 600)
        
        # Store plot windows to prevent garbage collection
        self._plot_windows = []
        
        self.setup_ui()
        self.update_signal_list()

    def setup_ui(self):
        """
        Create and arrange the user interface components for the dialog.
        """
        layout = QVBoxLayout(self)
        
        # Create tabs for better organization
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # ===== Basic Analysis Tab =====
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
        stats_btn.clicked.connect(self.show_statistics)
        button_layout.addWidget(stats_btn)
        
        fft_btn = QPushButton("FFT Analysis")
        fft_btn.clicked.connect(self.show_fft)
        button_layout.addWidget(fft_btn)
        
        time_analysis_btn = QPushButton("Time Domain Analysis")
        time_analysis_btn.clicked.connect(self.show_time_analysis)
        button_layout.addWidget(time_analysis_btn)
        
        basic_layout.addLayout(button_layout)
        
        # ===== Advanced Analysis Tab =====
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
        psd_btn.clicked.connect(self.show_psd)
        adv_button_layout.addWidget(psd_btn)
        
        autocorr_btn = QPushButton("Autocorrelation")
        autocorr_btn.clicked.connect(self.show_autocorrelation)
        adv_button_layout.addWidget(autocorr_btn)
        
        peaks_btn = QPushButton("Peak Detection")
        peaks_btn.clicked.connect(self.show_peak_detection)
        adv_button_layout.addWidget(peaks_btn)
        
        adv_layout.addLayout(adv_button_layout)
        
        adv_button_layout2 = QHBoxLayout()
        
        filter_btn = QPushButton("Apply Filter")
        filter_btn.clicked.connect(self.show_filter_dialog)
        adv_button_layout2.addWidget(filter_btn)
        
        hilbert_btn = QPushButton("Hilbert Transform")
        hilbert_btn.clicked.connect(self.show_hilbert_transform)
        adv_button_layout2.addWidget(hilbert_btn)
        
        energy_btn = QPushButton("Energy Analysis")
        energy_btn.clicked.connect(self.show_energy_analysis)
        adv_button_layout2.addWidget(energy_btn)
        
        adv_layout.addLayout(adv_button_layout2)
        
        adv_button_layout3 = QHBoxLayout()
        
        phase_btn = QPushButton("Phase Analysis")
        phase_btn.clicked.connect(self.show_phase_analysis)
        adv_button_layout3.addWidget(phase_btn)
        
        cepstrum_btn = QPushButton("Cepstral Analysis")
        cepstrum_btn.clicked.connect(self.show_cepstrum_analysis)
        adv_button_layout3.addWidget(cepstrum_btn)
        
        wavelet_btn = QPushButton("Wavelet Transform")
        wavelet_btn.clicked.connect(self.show_wavelet_dialog)
        adv_button_layout3.addWidget(wavelet_btn)
        
        adv_layout.addLayout(adv_button_layout3)
        
        # ===== Cross Analysis Tab =====
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
        xcorr_btn.clicked.connect(self.show_cross_correlation)
        cross_button_layout.addWidget(xcorr_btn)
        
        cross_layout.addLayout(cross_button_layout)
        
        # Results area (initially empty) - shared across tabs
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        results_scroll.setWidget(self.results_widget)
        layout.addWidget(results_scroll)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def create_signal_selector(self):
        """Create a signal selection dropdown."""
        self.signal_combo = QComboBox()
        return self.signal_combo

    def update_signal_list(self):
        """
        Populate all signal selection dropdowns with available signals.
        """
        if hasattr(self.parent, 'data_signals'):
            signals = list(self.parent.data_signals.keys())
            
            # Update all combo boxes
            self.signal_combo.clear()
            self.signal_combo.addItems(signals)
            
            self.adv_signal_combo.clear()
            self.adv_signal_combo.addItems(signals)
            
            self.cross_signal1_combo.clear()
            self.cross_signal1_combo.addItems(signals)
            
            self.cross_signal2_combo.clear()
            self.cross_signal2_combo.addItems(signals)

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
            return None
        return signal

    def clear_results(self):
        """
        Clear all widgets from the results area to prepare for new results.
        """
        for i in reversed(range(self.results_layout.count())):
            widget = self.results_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

    def show_statistics(self):
        """Calculate and display basic statistics for the selected signal."""
        signal = self.get_selected_signal()
        if not signal:
            return

        stats = SignalAnalysisTools.calculate_statistics(self.parent, signal)
        self.show_analysis_results("Statistics", signal, stats)

    def show_fft(self):
        """Perform FFT analysis on the selected signal and display in a new window."""
        signal = self.get_selected_signal()
        if not signal:
            return

        SignalAnalysisTools.perform_fft_analysis(self.parent, signal)

    def show_time_analysis(self):
        """Perform time-domain analysis on the selected signal and display results."""
        signal = self.get_selected_signal()
        if not signal:
            return

        analysis = SignalAnalysisTools.analyze_signal(self.parent, signal)
        self.show_analysis_results("Time Analysis", signal, analysis)

    def show_psd(self):
        """Calculate and display Power Spectral Density for the selected signal."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        freqs, psd = AdvancedSignalAnalysisTools.calculate_psd(self.parent, signal)

        # Create plot window with parent and window flags
        plot_window = pg.PlotWidget(title=f"Power Spectral Density: {signal}", parent=self)
        plot_window.setWindowTitle(f"PSD: {signal}")
        plot_window.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        plot_window.setLabel('left', 'Power/Frequency (dB/Hz)')
        plot_window.setLabel('bottom', 'Frequency (Hz)')
        plot_window.plot(freqs, 10 * np.log10(psd), pen='b')
        plot_window.resize(800, 600)
        plot_window.show()
        plot_window.raise_()  # Bring window to front
        
        # Keep a reference to prevent garbage collection
        self._plot_windows.append(plot_window)
        
        # Show summary in results area
        psd_stats = {
            "Peak Frequency (Hz)": freqs[np.argmax(psd)],
            "Max Power (dB)": 10 * np.log10(np.max(psd)),
            "Total Power": np.sum(psd)
        }
        self.show_analysis_results("PSD Analysis", signal, psd_stats)

    def show_autocorrelation(self):
        """Calculate and display autocorrelation of the selected signal."""
        signal = self.get_selected_signal(self.auto_signal_combo)
        if not signal:
            return

        # Calculate autocorrelation
        lag_times, corr = AdvancedSignalAnalysisTools.autocorrelation(
            self.parent, signal
        )

        # Create a proper window using QMainWindow
        plot_window = QMainWindow(self)
        plot_window.setWindowTitle(f"Autocorrelation: {signal}")
        plot_window.resize(800, 600)

        # Create central widget with layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create plot widget
        plot_widget = pg.PlotWidget(title=f"Autocorrelation: {signal}")
        plot_widget.setLabel('left', 'Correlation')
        plot_widget.setLabel('bottom', 'Lag (s)')
        plot_widget.plot(lag_times, corr, pen='b')

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

        # Keep reference to prevent garbage collection
        self._plot_windows.append(plot_window)

    def show_peak_detection(self):
        """Detect and analyze peaks in the selected signal."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        time_arr, values = self.parent.data_signals[signal]
        
        # Use default parameters for peak detection
        peaks_data = AdvancedSignalAnalysisTools.detect_peaks(
            self.parent, signal, height=np.mean(values), distance=10)
        
        # Create plot to visualize the peaks
        plot_window = pg.PlotWidget(title=f"Peak Detection: {signal}")
        plot_window.setWindowTitle(f"Peak Detection: {signal}")
        plot_window.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        plot_window.setLabel('left', 'Amplitude')
        plot_window.setLabel('bottom', 'Time (s)')
        
        # Plot the signal
        plot_window.plot(time_arr, values, pen='b')
        
        # Highlight peaks
        peak_plot = pg.ScatterPlotItem(
            x=peaks_data["Times"], 
            y=peaks_data["Heights"],
            symbol='o', 
            size=10, 
            pen='r', 
            brush='r'
        )
        plot_window.addItem(peak_plot)
        plot_window.resize(800, 600)
        plot_window.show()
        plot_window.raise_()
        
        # Keep a reference to prevent garbage collection
        self._plot_windows.append(plot_window)
        
        # Remove the indices and times from display data (too verbose)
        display_data = {k: v for k, v in peaks_data.items() 
                      if k not in ["Indices", "Times", "Heights"]}
        
        self.show_analysis_results("Peak Detection", signal, display_data)

    def show_filter_dialog(self):
        """Show dialog for filter settings and then apply filter."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        # Create filter dialog
        filter_dialog = QDialog(self)
        filter_dialog.setWindowTitle("Filter Settings")
        layout = QVBoxLayout(filter_dialog)
        
        # Filter type
        filter_type_group = QGroupBox("Filter Type")
        filter_type_layout = QVBoxLayout()
        
        self.lowpass_radio = QRadioButton("Low-pass")
        self.lowpass_radio.setChecked(True)
        self.highpass_radio = QRadioButton("High-pass")
        self.bandpass_radio = QRadioButton("Band-pass")
        
        filter_type_layout.addWidget(self.lowpass_radio)
        filter_type_layout.addWidget(self.highpass_radio)
        filter_type_layout.addWidget(self.bandpass_radio)
        filter_type_group.setLayout(filter_type_layout)
        layout.addWidget(filter_type_group)
        
        # Parameters
        params_layout = QFormLayout()
        
        # Get some reasonable defaults based on signal
        time_arr, _ = self.parent.data_signals[signal]
        fs = 1 / np.mean(np.diff(time_arr))
        nyq = fs / 2
        
        self.cutoff_low = QDoubleSpinBox()
        self.cutoff_low.setRange(0, nyq)
        self.cutoff_low.setValue(nyq / 10)  # Default to 10% of Nyquist
        self.cutoff_low.setSuffix(" Hz")
        
        self.cutoff_high = QDoubleSpinBox()
        self.cutoff_high.setRange(0, nyq)
        self.cutoff_high.setValue(nyq / 2)  # Default to 50% of Nyquist
        self.cutoff_high.setSuffix(" Hz")
        
        self.filter_order = QDoubleSpinBox()
        self.filter_order.setRange(1, 10)
        self.filter_order.setValue(4)
        self.filter_order.setDecimals(0)
        
        params_layout.addRow("Low Cutoff:", self.cutoff_low)
        params_layout.addRow("High Cutoff:", self.cutoff_high)
        params_layout.addRow("Filter Order:", self.filter_order)
        layout.addLayout(params_layout)
        
        # Enable/disable high cutoff based on filter type
        def update_high_cutoff():
            self.cutoff_high.setEnabled(self.bandpass_radio.isChecked())
        
        self.lowpass_radio.toggled.connect(update_high_cutoff)
        self.highpass_radio.toggled.connect(update_high_cutoff)
        self.bandpass_radio.toggled.connect(update_high_cutoff)
        update_high_cutoff()
        
        # Apply button
        apply_btn = QPushButton("Apply Filter")
        apply_btn.clicked.connect(lambda: self.apply_filter(signal, filter_dialog))
        layout.addWidget(apply_btn)
        
        filter_dialog.exec()

    def apply_filter(self, signal, dialog):
        """Apply the configured filter to the signal."""
        if self.lowpass_radio.isChecked():
            filter_type = 'lowpass'
        elif self.highpass_radio.isChecked():
            filter_type = 'highpass'
        else:
            filter_type = 'bandpass'
        
        cutoff_low = self.cutoff_low.value()
        cutoff_high = self.cutoff_high.value() if filter_type == 'bandpass' else None
        order = int(self.filter_order.value())
        
        time_arr, filtered = AdvancedSignalAnalysisTools.apply_filter(
            self.parent, signal, filter_type, cutoff_low, cutoff_high, order)
            
        # Plot original vs filtered signal
        plot_window = pg.PlotWidget(title=f"Filtered Signal: {signal}")
        plot_window.setWindowTitle(f"Filtered Signal: {signal}")
        plot_window.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        plot_window.setLabel('left', 'Amplitude')
        plot_window.setLabel('bottom', 'Time (s)')
        
        # Original signal in light color
        orig_values = self.parent.data_signals[signal][1]
        plot_window.plot(time_arr, orig_values, pen=pg.mkPen('b', width=1, alpha=100), name="Original")
        
        # Filtered signal in dark color
        plot_window.plot(time_arr, filtered, pen=pg.mkPen('r', width=2), name="Filtered")
        plot_window.addLegend()
        plot_window.resize(800, 600)
        plot_window.show()
        plot_window.raise_()

        # Keep a reference to prevent garbage collection
        self._plot_windows.append(plot_window)

        dialog.accept()

    def show_hilbert_transform(self):
        """Calculate and display Hilbert transform results for the selected signal."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        results = AdvancedSignalAnalysisTools.hilbert_transform(self.parent, signal)

        # Create plot window with multiple plots
        plot_window = pg.GraphicsLayoutWidget(title=f"Hilbert Transform: {signal}")
        plot_window.setWindowTitle(f"Hilbert Transform: {signal}")
        plot_window.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        # Original signal plot
        p1 = plot_window.addPlot(row=0, col=0)
        p1.setTitle("Original Signal with Envelope")
        p1.setLabel('left', 'Amplitude')
        p1.setLabel('bottom', 'Time (s)')

        time_arr, values = self.parent.data_signals[signal]
        p1.plot(time_arr, values, pen='b', name="Signal")
        p1.plot(results["envelope"][0], results["envelope"][1], pen='r', name="Envelope")
        p1.addLegend()

        # Phase plot
        p2 = plot_window.addPlot(row=1, col=0)
        p2.setTitle("Instantaneous Phase")
        p2.setLabel('left', 'Phase (rad)')
        p2.setLabel('bottom', 'Time (s)')
        p2.plot(results["phase"][0], results["phase"][1], pen='g')

        # Frequency plot
        p3 = plot_window.addPlot(row=2, col=0)
        p3.setTitle("Instantaneous Frequency")
        p3.setLabel('left', 'Frequency (Hz)')
        p3.setLabel('bottom', 'Time (s)')
        p3.plot(results["frequency"][0], results["frequency"][1], pen='y')

        plot_window.resize(800, 600)
        plot_window.show()
        plot_window.raise_()
        self._plot_windows.append(plot_window)

        # Show summary in results area
        self.show_analysis_results("Hilbert Transform", signal, {
            "Mean Envelope": np.mean(results["envelope"][1]),
            "Max Envelope": np.max(results["envelope"][1]),
            "Mean Inst. Frequency": np.mean(results["frequency"][1])
        })

    def show_energy_analysis(self):
        """Calculate and display energy distribution across time intervals."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        interval_centers, energy = AdvancedSignalAnalysisTools.energy_in_intervals(self.parent, signal)

        # Create plot window
        plot_window = pg.PlotWidget(title=f"Energy Analysis: {signal}")
        plot_window.setWindowTitle(f"Energy Analysis: {signal}")
        plot_window.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        plot_window.setLabel('left', 'Energy')
        plot_window.setLabel('bottom', 'Time (s)')

        # Plot energy distribution as bar graph
        bar_graph = pg.BarGraphItem(x=interval_centers, height=energy,
                                    width=interval_centers[1] - interval_centers[0] if len(
                                        interval_centers) > 1 else 0.1, brush='b')
        plot_window.addItem(bar_graph)
        plot_window.resize(800, 600)
        plot_window.show()
        plot_window.raise_()
        self._plot_windows.append(plot_window)

        # Calculate statistics on energy distribution
        energy_stats = {
            "Total Energy": np.sum(energy),
            "Mean Energy per Interval": np.mean(energy),
            "Max Energy": np.max(energy),
            "Energy StdDev": np.std(energy),
            "Energy Variance": np.var(energy)
        }
        self.show_analysis_results("Energy Analysis", signal, energy_stats)

    def show_phase_analysis(self):
        """Analyze and display phase information of the signal."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        time_arr, phase, phase_stats = AdvancedSignalAnalysisTools.phase_analysis(self.parent, signal)

        # Create plot window
        plot_window = pg.PlotWidget(title=f"Phase Analysis: {signal}")
        plot_window.setWindowTitle(f"Phase Analysis: {signal}")
        plot_window.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        plot_window.setLabel('left', 'Phase (rad)')
        plot_window.setLabel('bottom', 'Time (s)')
        plot_window.plot(time_arr, phase, pen='b')
        plot_window.resize(800, 600)
        plot_window.show()
        plot_window.raise_()
        self._plot_windows.append(plot_window)

        self.show_analysis_results("Phase Analysis", signal, phase_stats)

    def show_cepstrum_analysis(self):
        """Perform and display cepstral analysis of the signal."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        quefrency, cepstrum = AdvancedSignalAnalysisTools.cepstrum_analysis(self.parent, signal)

        # Create plot window
        plot_window = pg.PlotWidget(title=f"Cepstrum Analysis: {signal}")
        plot_window.setWindowTitle(f"Cepstrum Analysis: {signal}")
        plot_window.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        plot_window.setLabel('left', 'Amplitude')
        plot_window.setLabel('bottom', 'Quefrency (s)')
        plot_window.plot(quefrency, cepstrum, pen='b')
        plot_window.resize(800, 600)
        plot_window.show()
        plot_window.raise_()
        self._plot_windows.append(plot_window)

        # Find maximum in cepstrum (excluding the first bin)
        max_idx = np.argmax(cepstrum[1:]) + 1
        fundamental_period = quefrency[max_idx]

        cepstrum_stats = {
            "Fundamental Period": fundamental_period,
            "Fundamental Frequency": 1.0 / fundamental_period if fundamental_period > 0 else float('inf'),
            "Peak Cepstrum Value": cepstrum[max_idx]
        }
        self.show_analysis_results("Cepstrum Analysis", signal, cepstrum_stats)

    def show_wavelet_dialog(self):
        """Show dialog for wavelet transform configuration and then perform the transform."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        # Create wavelet dialog
        wavelet_dialog = QDialog(self)
        wavelet_dialog.setWindowTitle("Wavelet Transform Settings")
        layout = QVBoxLayout(wavelet_dialog)

        # Wavelet type selection
        wavelet_form = QFormLayout()
        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(['db4', 'db8', 'sym4', 'sym8', 'coif4', 'haar'])
        wavelet_form.addRow("Wavelet Type:", self.wavelet_combo)

        # Decomposition level
        self.level_spin = QDoubleSpinBox()
        self.level_spin.setRange(1, 10)
        self.level_spin.setValue(5)
        self.level_spin.setDecimals(0)
        wavelet_form.addRow("Decomposition Level:", self.level_spin)

        layout.addLayout(wavelet_form)

        # Apply button
        apply_btn = QPushButton("Apply Transform")
        apply_btn.clicked.connect(lambda: self.apply_wavelet_transform(signal, wavelet_dialog))
        layout.addWidget(apply_btn)

        wavelet_dialog.exec()

    def apply_wavelet_transform(self, signal, dialog):
        """Apply wavelet transform with selected parameters."""
        wavelet = self.wavelet_combo.currentText()
        level = int(self.level_spin.value())

        coeffs, approx, details = AdvancedSignalAnalysisTools.wavelet_transform(
            self.parent, signal, wavelet, level)

        # Create a proper window using QMainWindow
        plot_window = QMainWindow(self)
        plot_window.setWindowTitle(f"Wavelet Transform: {signal}")
        plot_window.resize(800, 600)

        # Create central widget with layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create GraphicsLayoutWidget for plots
        plot_widget = pg.GraphicsLayoutWidget()

        # Original signal
        p1 = plot_widget.addPlot(row=0, col=0)
        p1.setTitle("Original Signal")
        time_arr, values = self.parent.data_signals[signal]
        p1.plot(time_arr, values, pen='b')

        # Approximation
        p2 = plot_widget.addPlot(row=1, col=0)
        p2.setTitle(f"Approximation (Level {level})")
        p2.plot(np.linspace(0, time_arr[-1], len(approx)), approx, pen='r')

        # Details for each level
        for i, detail in enumerate(details):
            p = plot_widget.addPlot(row=i + 2, col=0)
            p.setTitle(f"Detail Level {level - i}")
            p.plot(np.linspace(0, time_arr[-1], len(detail)), detail, pen='g')

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

        # Keep reference to prevent garbage collection
        self._plot_windows.append(plot_window)

        # Calculate energy in each component
        energy_approx = np.sum(approx ** 2)
        energy_details = [np.sum(d ** 2) for d in details]
        total_energy = energy_approx + sum(energy_details)

        wavelet_stats = {
            "Approximation Energy": energy_approx,
            "Approximation Energy %": 100 * energy_approx / total_energy,
        }

        for i, e in enumerate(energy_details):
            wavelet_stats[f"Detail {level - i} Energy"] = e
            wavelet_stats[f"Detail {level - i} Energy %"] = 100 * e / total_energy

        self.show_analysis_results("Wavelet Transform", signal, wavelet_stats)
        dialog.accept()

    def show_cross_correlation(self):
        """Calculate and display cross-correlation between two signals."""
        signal1 = self.get_selected_signal(self.cross_signal1_combo)
        signal2 = self.get_selected_signal(self.cross_signal2_combo)

        if not signal1 or not signal2:
            return

        # Calculate cross-correlation
        lag_times, corr, max_lag = AdvancedSignalAnalysisTools.cross_correlation(
            self.parent, signal1, signal2
        )

        # Create a proper window using QMainWindow
        plot_window = QMainWindow(self)
        plot_window.setWindowTitle(f"Cross-Correlation: {signal1} vs {signal2}")
        plot_window.resize(800, 600)

        # Create central widget with layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create plot widget
        plot_widget = pg.PlotWidget(title=f"Cross-Correlation: {signal1} vs {signal2}")
        plot_widget.setLabel('left', 'Correlation')
        plot_widget.setLabel('bottom', 'Lag (s)')
        plot_widget.plot(lag_times, corr, pen='b')

        # Add vertical line at max correlation
        max_idx = np.argmax(np.abs(corr))
        vline = pg.InfiniteLine(pos=lag_times[max_idx], angle=90, pen=pg.mkPen('r', width=2))
        plot_widget.addItem(vline)

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

        # Keep reference to prevent garbage collection
        self._plot_windows.append(plot_window)

        # Show correlation statistics
        corr_stats = {
            "Max Correlation": corr[max_idx],
            "Max Correlation Lag": lag_times[max_idx],
            "Time Shift": f"{lag_times[max_idx]:.4f} seconds"
        }
        self.show_analysis_results(f"Cross-Correlation: {signal1} vs {signal2}",
                                   f"{signal1} vs {signal2}",
                                   corr_stats)

    def show_analysis_results(self, analysis_type, signal_name, results):
        """
        Display analysis results in the results area.

        Args:
            analysis_type (str): Type of analysis performed
            signal_name (str): Name of signal analyzed
            results (dict): Dictionary of result values to display
        """
        self.clear_results()

        # Create group box for results
        group = QGroupBox(f"{analysis_type}: {signal_name}")
        layout = QVBoxLayout()

        # Create table for results
        table = QTableWidget(len(results), 2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        # Populate table with results
        for i, (key, value) in enumerate(results.items()):
            table.setItem(i, 0, QTableWidgetItem(str(key)))

            # Format numerical values
            if isinstance(value, (int, float)):
                if abs(value) > 1000 or abs(value) < 0.001:
                    formatted_val = f"{value:.6e}"
                else:
                    formatted_val = f"{value:.6f}"
                table.setItem(i, 1, QTableWidgetItem(formatted_val))
            else:
                table.setItem(i, 1, QTableWidgetItem(str(value)))

        layout.addWidget(table)
        group.setLayout(layout)
        self.results_layout.addWidget(group)


def show_analysis_dialog(parent):
    """
    Shows the signal analysis dialog for analyzing selected signals.

    Args:
        parent: The parent application instance that has data_signals.
    """
    # Create and show the dialog
    dialog = SignalAnalysisDialog(parent)
    dialog.exec()