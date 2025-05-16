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
                               QVBoxLayout, QGridLayout, QScrollArea, QMainWindow)


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
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowMaximizeButtonHint)
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

        # Get the signal values directly
        _, values = self.parent.data_signals[signal]

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

        self.show_analysis_results("Statistics", signal, stats)

    def show_fft(self):
        """Perform FFT analysis on the selected signal and display in a new window."""
        signal = self.get_selected_signal()
        if not signal:
            return

        # Get the source signal data
        time_arr, values = self.parent.data_signals[signal]

        # Process the signal values directly
        processed_values = prepare_signal_for_analysis(self, values, "FFT Input Signal")
        if processed_values is None:
            return  # User canceled the operation

        # Perform FFT analysis
        fs = 1 / np.mean(np.diff(time_arr))  # Sampling frequency
        n = len(processed_values)
        fft_values = np.fft.rfft(processed_values)
        freqs = np.fft.rfftfreq(n, d=1 / fs)
        magnitudes = np.abs(fft_values) / n * 2  # Scale appropriately

        # Create a proper window using QMainWindow
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

        # Keep reference to prevent garbage collection
        self._plot_windows.append(plot_window)

    def show_time_analysis(self):
        """Perform time-domain analysis on the selected signal and display results."""
        signal = self.get_selected_signal()
        if not signal:
            return

        # Get the source signal data directly
        time_arr, values = self.parent.data_signals[signal]

        # Time domain analysis
        duration = time_arr[-1] - time_arr[0]
        sample_rate = len(values) / duration

        # Calculate zero crossings
        zero_crossings = np.sum(np.diff(np.signbit(values).astype(int)) != 0)

        # Calculate signal energy and power
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

        self.show_analysis_results("Time Analysis", signal, analysis)

    def show_psd_analysis(self):
        """Calculate and display Power Spectral Density for the selected signal."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        # Get the source signal data
        time_arr, values = self.parent.data_signals[signal]

        # Process the signal values directly
        processed_values = prepare_signal_for_analysis(self, values, "PSD Input Signal")
        if processed_values is None:
            return  # User canceled the operation

        # Calculate PSD
        fs = 1 / np.mean(np.diff(time_arr)) if len(time_arr) > 1 else 1  # Sampling frequency
        freqs, psd = sc_signal.welch(processed_values, fs=fs, nperseg=256)

        # Calculate some PSD statistics

        psd_stats = {
            "Peak Frequency (Hz)": freqs[np.argmax(psd)],
            "Max Power (dB)": 10 * np.log10(np.max(psd)),
            "Total Power": np.sum(psd)
        }

        # Create a proper window using QMainWindow
        plot_window = QMainWindow(self)
        plot_window.setWindowTitle(f"Power Spectral Density: {signal}")
        plot_window.resize(800, 600)

        # Create central widget with layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create plot widget
        plot_widget = pg.PlotWidget(title=f"Power Spectral Density: {signal}")
        plot_widget.setLabel('left', 'Power/Frequency (dB/Hz)')
        plot_widget.setLabel('bottom', 'Frequency (Hz)')
        plot_widget.plot(freqs, 10 * np.log10(psd), pen='b')

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

        # Show PSD statistics
        self.show_analysis_results("PSD Analysis", signal, psd_stats)

    def show_autocorrelation(self):
        """Calculate and display autocorrelation of the selected signal."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        # Get the source signal data directly
        time_arr, values = self.parent.data_signals[signal]

        # Calculate autocorrelation directly on the original values
        autocorr = np.correlate(values, values, mode='full')

        # Normalize
        autocorr = autocorr / np.max(autocorr)

        # Create lag array
        lags = np.arange(-len(values) + 1, len(values))
        lag_times = lags * np.mean(np.diff(time_arr))

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
        plot_widget.plot(lag_times, autocorr, pen='b')

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
        """Detect and analyze peaks in the selected signal. For predominantly negative signals,
            negative peaks (valleys) are detected and reported instead of positive peaks."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        # Get the source signal data directly
        time_arr, values = self.parent.data_signals[signal]

        # Determine if we should look for positive or negative peaks
        signal_mean = np.mean(values)
        if np.all(values < 0) or (
                np.any(values < 0) and abs(np.min(values) - signal_mean) > abs(np.max(values) - signal_mean)):
            # For predominantly negative signals, look for negative peaks (valleys)
            # We invert the signal to use the same peak finding algorithm
            processed_values = -values
            peak_type = "negative"
        else:
            # For positive or mixed signals, look for positive peaks
            processed_values = values
            peak_type = "positive"

        # Use default parameters for peak detection
        peaks, properties = sc_signal.find_peaks(processed_values, height=None, distance=None)

        # Calculate additional peak properties
        peak_heights = processed_values[peaks]
        if peak_type == "negative":
            # Convert heights back to negative for display
            peak_heights = -peak_heights

        peak_widths = sc_signal.peak_widths(processed_values, peaks, rel_height=0.5)[0]

        peaks_data = {
            "Peak Type": "Negative" if peak_type == "negative" else "Positive",
            "Count": len(peaks),
            "Indices": peaks,
            "Times": time_arr[peaks],
            "Heights": peak_heights,
            "Mean Height": np.mean(peak_heights) if len(peak_heights) > 0 else 0,
            "Max Height": np.max(peak_heights) if len(peak_heights) > 0 else 0,
            "Mean Width": np.mean(peak_widths) if len(peak_widths) > 0 else 0
        }

        # Create a proper window using QMainWindow
        plot_window = QMainWindow(self)
        plot_window.setWindowTitle(f"Peak Detection: {signal}")
        plot_window.resize(800, 600)

        # Create central widget with layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create plot widget
        plot_widget = pg.PlotWidget(title=f"Peak Detection: {signal}")
        plot_widget.setLabel('left', 'Amplitude')
        plot_widget.setLabel('bottom', 'Time (s)')

        # Plot the signal
        plot_widget.plot(time_arr, values, pen='b')

        # Highlight peaks
        peak_plot = pg.ScatterPlotItem(
            x=time_arr[peaks],
            y=values[peaks],
            symbol='o',
            size=10,
            pen=pg.mkPen('r', width=2),
            brush=pg.mkBrush('r')
        )
        plot_widget.addItem(peak_plot)

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

        # Remove the indices and times from display data (too verbose)
        display_data = {k: v for k, v in peaks_data.items()
                        if k not in ["Indices", "Times", "Heights"]}

        self.show_analysis_results("Peak Detection", signal, display_data)

    def show_filtered_signal(self, original_values, filtered_values, filter_type, signal_name):
        """
        Display the original and filtered signals in a new window.

        Args:
            original_values: The original signal values
            filtered_values: The filtered signal values
            filter_type: Type of applied filter (e.g., "Lowpass")
            signal_name: Name of the signal
        """
        # Create a new window for displaying the signals
        plot_window = QMainWindow(self)
        plot_window.setWindowTitle(f"Filtered Signal: {signal_name} ({filter_type})")
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
        p1.setLabel('bottom', 'Sample Index')
        p1.plot(original_values, pen='b', name="Original")

        # Filtered signal plot
        p2 = plot_widget.addPlot(row=1, col=0)
        p2.setTitle("Filtered Signal")
        p2.setLabel('left', 'Amplitude')
        p2.setLabel('bottom', 'Sample Index')
        p2.plot(filtered_values, pen='r', name="Filtered")

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

        # Ensure we have a list to store windows
        if not hasattr(self, '_plot_windows'):
            self._plot_windows = []

        # Keep reference to prevent garbage collection
        self._plot_windows.append(plot_window)

    def show_filter_dialog(self):
        """Show dialog for filter settings and apply the filter."""
        from PySide6.QtWidgets import QSpinBox, QGroupBox, QRadioButton, QFormLayout

        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        # Create filter dialog
        filter_dialog = QDialog(self)
        filter_dialog.setWindowTitle("Filter Settings")
        layout = QVBoxLayout(filter_dialog)

        # Filter type
        filter_type_lowpass = QRadioButton("Lowpass")
        filter_type_lowpass.setToolTip("Allow frequencies below the cutoff to pass.")
        filter_type_highpass = QRadioButton("Highpass")
        filter_type_highpass.setToolTip("Allow frequencies above the cutoff to pass.")
        filter_type_bandpass = QRadioButton("Bandpass")
        filter_type_bandpass.setToolTip("Allow frequencies within a specific range to pass.")
        filter_type_lowpass.setChecked(True)

        filter_group = QGroupBox("Filter Type")
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.addWidget(filter_type_lowpass)
        filter_layout.addWidget(filter_type_highpass)
        filter_layout.addWidget(filter_type_bandpass)
        layout.addWidget(filter_group)

        # Filter parameters
        params_group = QGroupBox("Filter Parameters")
        params_layout = QFormLayout(params_group)

        cutoff_low_spin = QDoubleSpinBox()
        cutoff_low_spin.setRange(0.1, 1000)
        cutoff_low_spin.setValue(10)
        cutoff_low_spin.setSuffix(" Hz")
        cutoff_low_spin.setToolTip("Set the lower cutoff frequency for the filter.")

        cutoff_high_spin = QDoubleSpinBox()
        cutoff_high_spin.setRange(0.1, 1000)
        cutoff_high_spin.setValue(100)
        cutoff_high_spin.setSuffix(" Hz")
        cutoff_high_spin.setEnabled(filter_type_bandpass.isChecked())
        cutoff_high_spin.setToolTip("Set the upper cutoff frequency for the bandpass filter")

        filter_order_spin = QSpinBox()
        filter_order_spin.setRange(1, 10)
        filter_order_spin.setValue(4)
        filter_order_spin.setToolTip("Specify the order of the filter (higher values result in sharper transitions).")

        params_layout.addRow("Cutoff Frequency (Low):", cutoff_low_spin)
        params_layout.addRow("Cutoff Frequency (High):", cutoff_high_spin)
        params_layout.addRow("Filter Order:", filter_order_spin)

        # Enable/disable high cutoff based on filter type
        filter_type_bandpass.toggled.connect(cutoff_high_spin.setEnabled)

        layout.addWidget(params_group)

        # Buttons
        buttons_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        buttons_layout.addWidget(apply_btn)
        buttons_layout.addWidget(cancel_btn)
        layout.addLayout(buttons_layout)

        def apply_filter():
            if filter_type_lowpass.isChecked():
                filter_type = 'lowpass'
            elif filter_type_highpass.isChecked():
                filter_type = 'highpass'
            else:
                filter_type = 'bandpass'

            cutoff_low = cutoff_low_spin.value()
            cutoff_high = cutoff_high_spin.value() if filter_type == 'bandpass' else None
            order = int(filter_order_spin.value())

            try:
                # Get signal data
                time_arr, values = self.parent.data_signals[signal]

                # Calculate Nyquist frequency
                sampling_rate = 1 / np.mean(np.diff(time_arr))
                nyquist = 0.5 * sampling_rate

                # Normalize cutoff frequencies
                normalized_cutoff_low = cutoff_low / nyquist
                normalized_cutoff_high = cutoff_high / nyquist if cutoff_high else None

                # Validate cutoff frequencies
                if not (0 < normalized_cutoff_low < 1):
                    raise ValueError("Low cutoff frequency must be within (0, Nyquist).")
                if filter_type == 'bandpass' and not (0 < normalized_cutoff_high < 1):
                    raise ValueError("High cutoff frequency must be within (0, Nyquist).")

                # Apply Butterworth filter
                if filter_type == 'lowpass':
                    b, a = butter(order, normalized_cutoff_low, btype='low')
                elif filter_type == 'highpass':
                    b, a = butter(order, normalized_cutoff_low, btype='high')
                elif filter_type == 'bandpass':
                    b, a = butter(order, [normalized_cutoff_low, normalized_cutoff_high], btype='band')

                # Filter the signal
                original_values = values.copy()
                filtered_values = filtfilt(b, a, values)

                # Display results
                self.show_filtered_signal(original_values, filtered_values, filter_type, signal)
                filter_info = {
                    "Filter Type": filter_type,
                    "Cutoff Low": cutoff_low,
                    "Cutoff High": cutoff_high if filter_type == 'bandpass' else "N/A",
                    "Order": order
                }
                self.show_analysis_results("Filter Results", signal, filter_info)
                filter_dialog.accept()
            except Exception as e:
                print(f"Error applying filter: {e}")
                filter_dialog.reject()

        apply_btn.clicked.connect(apply_filter)
        cancel_btn.clicked.connect(filter_dialog.reject)

        filter_dialog.exec()

    def show_hilbert_transform(self):
        """Calculate and display Hilbert transform results for the selected signal."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        # Get the source signal data
        time_arr, values = self.parent.data_signals[signal]

        # Process the signal values directly
        processed_values = prepare_signal_for_analysis(self, values, "Hilbert Input Signal")
        if processed_values is None:
            return  # User canceled the operation

        # Calculate Hilbert transform directly
        analytic_signal = hilbert(processed_values)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        # Calculate instantaneous frequency (derivative of phase)
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * np.mean(np.diff(time_arr)))
        # Add a zero to match original length (since diff reduces length by 1)
        instantaneous_frequency = np.append(instantaneous_frequency, instantaneous_frequency[-1])

        # Create a proper window using QMainWindow
        plot_window = QMainWindow(self)
        plot_window.setWindowTitle(f"Hilbert Transform: {signal}")
        plot_window.resize(800, 600)

        # Create central widget with layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create GraphicsLayoutWidget for plots
        plot_widget = pg.GraphicsLayoutWidget()

        # Original signal plot
        p1 = plot_widget.addPlot(row=0, col=0)
        p1.setTitle("Original Signal with Envelope")
        p1.setLabel('left', 'Amplitude')
        p1.setLabel('bottom', 'Time (s)')
        p1.plot(time_arr, processed_values, pen='b', name="Signal")
        p1.plot(time_arr, amplitude_envelope, pen='r', name="Envelope")
        p1.addLegend()

        # Phase plot
        p2 = plot_widget.addPlot(row=1, col=0)
        p2.setTitle("Instantaneous Phase")
        p2.setLabel('left', 'Phase (rad)')
        p2.setLabel('bottom', 'Time (s)')
        p2.plot(time_arr, instantaneous_phase, pen='g')

        # Frequency plot
        p3 = plot_widget.addPlot(row=2, col=0)
        p3.setTitle("Instantaneous Frequency")
        p3.setLabel('left', 'Frequency (Hz)')
        p3.setLabel('bottom', 'Time (s)')
        p3.plot(time_arr, instantaneous_frequency, pen='y')

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

        # Show summary in results area
        self.show_analysis_results("Hilbert Transform", signal, {
            "Mean Envelope": np.mean(amplitude_envelope),
            "Max Envelope": np.max(amplitude_envelope),
            "Mean Inst. Frequency": np.mean(instantaneous_frequency)
        })

    def show_energy_analysis(self):
        """Calculate and display energy distribution across time intervals."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        # Get the source signal data
        time_arr, values = self.parent.data_signals[signal]

        # Process the signal values directly
        processed_values = prepare_signal_for_analysis(self, values, "Energy Analysis Input Signal")
        if processed_values is None:
            return  # User canceled the operation

        # Calculate energy distribution across time intervals
        num_intervals = 20  # Divide the signal into 20 intervals
        interval_size = len(processed_values) // num_intervals

        # Calculate energy in each interval
        energy = []
        interval_centers = []

        for i in range(num_intervals):
            start_idx = i * interval_size
            end_idx = start_idx + interval_size if i < num_intervals - 1 else len(processed_values)
            segment = processed_values[start_idx:end_idx]

            # Energy calculation (sum of squares)
            segment_energy = np.sum(segment ** 2)
            energy.append(segment_energy)

            # Calculate the center time of this interval
            center_time = time_arr[start_idx] + (
                        time_arr[min(end_idx - 1, len(time_arr) - 1)] - time_arr[start_idx]) / 2
            interval_centers.append(center_time)

        # Create a proper window using QMainWindow
        plot_window = QMainWindow(self)
        plot_window.setWindowTitle(f"Energy Analysis: {signal}")
        plot_window.resize(800, 600)

        # Create central widget with layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create plot widget
        plot_widget = pg.PlotWidget(title=f"Energy Analysis: {signal}")
        plot_widget.setLabel('left', 'Energy')
        plot_widget.setLabel('bottom', 'Time (s)')

        # Plot energy distribution as bar graph
        bar_width = interval_centers[1] - interval_centers[0] if len(interval_centers) > 1 else 0.1
        bar_graph = pg.BarGraphItem(x=interval_centers, height=energy, width=bar_width, brush='b')
        plot_widget.addItem(bar_graph)

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

        # Get the source signal data
        time_arr, values = self.parent.data_signals[signal]

        # Process the signal values directly
        processed_values = prepare_signal_for_analysis(self, values, "Phase Analysis Input Signal")
        if processed_values is None:
            return  # User canceled the operation

        # Calculate Hilbert transform to get phase information
        analytic_signal = hilbert(processed_values)

        # Extract instantaneous phase and unwrap to avoid phase jumps
        phase = np.unwrap(np.angle(analytic_signal))

        # Calculate phase statistics
        phase_stats = {
            "Mean Phase": np.mean(phase),
            "Phase Standard Deviation": np.std(phase),
            "Phase Range": np.max(phase) - np.min(phase),
            "Phase Rate of Change": np.mean(np.abs(np.diff(phase))) / np.mean(np.diff(time_arr))
        }

        # Create a proper window using QMainWindow
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

        self.show_analysis_results("Phase Analysis", signal, phase_stats)

    def show_cepstrum_analysis(self):
        """Perform and display cepstral analysis of the signal."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        # Get the source signal data
        time_arr, values = self.parent.data_signals[signal]

        # Process the signal values directly
        processed_values = prepare_signal_for_analysis(self, values, "Cepstrum Analysis Input Signal")
        if processed_values is None:
            return  # User canceled the operation

        # Calculate sampling rate
        fs = 1 / np.mean(np.diff(time_arr))

        # Compute FFT of the log of the magnitude spectrum (cepstrum)
        spectrum = np.fft.fft(processed_values)
        log_spectrum = np.log(np.abs(spectrum) + 1e-10)  # Add small value to avoid log(0)
        cepstrum = np.abs(np.fft.ifft(log_spectrum))

        # Calculate quefrency axis (time)
        n = len(cepstrum)
        quefrency = np.arange(n) / fs

        # We only need the first half of the cepstrum (symmetric)
        quefrency = quefrency[:n // 2]
        cepstrum = cepstrum[:n // 2]

        # Create a proper window using QMainWindow
        plot_window = QMainWindow(self)
        plot_window.setWindowTitle(f"Cepstrum Analysis: {signal}")
        plot_window.resize(800, 600)

        # Create central widget with layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create plot widget
        plot_widget = pg.PlotWidget(title=f"Cepstrum Analysis: {signal}")
        plot_widget.setLabel('left', 'Amplitude')
        plot_widget.setLabel('bottom', 'Quefrency (s)')
        plot_widget.plot(quefrency, cepstrum, pen='b')

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

        # Find maximum in cepstrum (excluding the first few bins which often contain DC components)
        start_bin = 5  # Skip first few bins to avoid DC and very low quefrency components
        if len(cepstrum) > start_bin:
            max_idx = np.argmax(cepstrum[start_bin:]) + start_bin
            fundamental_period = quefrency[max_idx]
            fundamental_frequency = 1.0 / fundamental_period if fundamental_period > 0 else float('inf')
        else:
            max_idx = 0
            fundamental_period = 0
            fundamental_frequency = float('inf')

        cepstrum_stats = {
            "Fundamental Period": fundamental_period,
            "Fundamental Frequency": fundamental_frequency,
            "Peak Cepstrum Value": cepstrum[max_idx] if max_idx < len(cepstrum) else 0
        }
        self.show_analysis_results("Cepstrum Analysis", signal, cepstrum_stats)

    def show_wavelet_dialog(self):
        """Show dialog for wavelet transform settings."""
        signal = self.get_selected_signal(self.adv_signal_combo)
        if not signal:
            return

        # Create wavelet dialog
        wavelet_dialog = QDialog(self)
        wavelet_dialog.setWindowTitle("Wavelet Transform Settings")
        layout = QVBoxLayout(wavelet_dialog)

        # Wavelet type
        form_layout = QFormLayout()
        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(["db4", "db8", "sym4", "sym8", "coif1", "coif3", "haar"])
        self.wavelet_combo.setToolTip("Select the type of wavelet to use for the transform.")
        form_layout.addRow("Wavelet Type:", self.wavelet_combo)

        # Decomposition level
        self.level_spin = QDoubleSpinBox()
        self.level_spin.setDecimals(0)
        self.level_spin.setMinimum(1)
        self.level_spin.setMaximum(10)
        self.level_spin.setValue(5)
        self.level_spin.setToolTip("Set the decomposition level (higher levels analyze broader time scales).")
        form_layout.addRow("Decomposition Level:", self.level_spin)

        layout.addLayout(form_layout)

        # Buttons
        buttons_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(lambda: self.apply_wavelet_transform(signal, wavelet_dialog))
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(wavelet_dialog.reject)

        buttons_layout.addWidget(apply_btn)
        buttons_layout.addWidget(cancel_btn)
        layout.addLayout(buttons_layout)

        wavelet_dialog.exec()

    def apply_wavelet_transform(self, signal, dialog):
        """Apply wavelet transform with selected parameters."""
        wavelet = self.wavelet_combo.currentText()
        level = int(self.level_spin.value())

        # Get the source signal data
        time_arr, values = self.parent.data_signals[signal]

        # Process the signal values directly
        processed_values = prepare_signal_for_analysis(self, values, "Wavelet Input Signal")
        if processed_values is None:
            return  # User canceled the operation

        # Perform wavelet decomposition
        try:
            # Use PyWavelets for the wavelet transform
            coeffs = pywt.wavedec(processed_values, wavelet, level=level)

            # Extract approximation and details
            approx = coeffs[0]  # Approximation coefficients at the final level
            details = coeffs[1:]  # Detail coefficients at all levels

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
            p1.plot(time_arr, processed_values, pen='b')

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

            # Prepare statistics for results display
            wavelet_stats = {
                "Wavelet Type": wavelet,
                "Decomposition Level": level,
                "Total Energy": total_energy,
                "Approximation Energy": energy_approx,
                "Approximation Energy %": 100 * energy_approx / total_energy
            }

            # Add each detail level's energy percentage
            for i, e in enumerate(energy_details):
                wavelet_stats[f"Detail Level {level - i} Energy %"] = 100 * e / total_energy

            self.show_analysis_results("Wavelet Transform", signal, wavelet_stats)
            dialog.accept()

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Wavelet Transform Error",
                                 f"Error performing wavelet transform: {str(e)}")
            dialog.reject()

    def show_cross_correlation(self):
        """Calculate and display cross-correlation between two signals."""
        signal1 = self.get_selected_signal(self.cross_signal1_combo)
        signal2 = self.get_selected_signal(self.cross_signal2_combo)

        if not signal1 or not signal2:
            return

        # Get the source signal data directly
        time_arr1, values1 = self.parent.data_signals[signal1]
        time_arr2, values2 = self.parent.data_signals[signal2]

        # Handle signals with different lengths by padding the shorter one
        if len(values1) > len(values2):
            values2 = np.pad(values2, (0, len(values1) - len(values2)), 'constant')
        elif len(values2) > len(values1):
            values1 = np.pad(values1, (0, len(values2) - len(values1)), 'constant')

        # Calculate cross-correlation
        corr = correlate(values1, values2, mode='full')

        # Normalize to [-1, 1] range
        corr_normalized = corr / np.sqrt(np.sum(values1 ** 2) * np.sum(values2 ** 2))

        # Create lag time array based on actual time data
        sample_period = np.mean(np.diff(time_arr1))  # Average time between samples
        lags = np.arange(-len(values1) + 1, len(values1))
        lag_times = lags * sample_period

        # Find the lag with maximum correlation
        max_idx = np.argmax(np.abs(corr_normalized))
        max_lag = lag_times[max_idx]
        max_corr = corr_normalized[max_idx]

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
        plot_widget.plot(lag_times, corr_normalized, pen='b')

        # Add vertical line at max correlation
        vline = pg.InfiniteLine(pos=max_lag, angle=90, pen=pg.mkPen('r', width=2))
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

        # Show results in results area
        correlation_stats = {
            "Maximum Correlation": max_corr,
            "Lag at Maximum (s)": max_lag,
            "Correlation at Zero Lag": corr_normalized[len(corr_normalized) // 2]
        }
        self.show_analysis_results("Cross-Correlation", f"{signal1} vs {signal2}", correlation_stats)

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

    def show_help_in_results(self, topic, content):
        """
        Display help content in the results area.

        Args:
            topic: The help topic title
            content: HTML formatted help content
        """
        self.clear_results()

        # Create group box for help content
        group = QGroupBox(f"Help: {topic}")
        group_layout = QVBoxLayout()

        # Create text display for help
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml(content)

        group_layout.addWidget(help_text)
        group.setLayout(group_layout)
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


def prepare_signal_for_analysis(parent, arr, label="Signal"):
    """
    Handles negative/mixed signals for analyses requiring positive values.

    Args:
        parent: Parent widget for displaying dialogs
        arr: Numpy array to check
        label: Name of analysis for user messages

    Returns:
        Processed array or None if user cancels
    """
    from PySide6.QtWidgets import QMessageBox

    if np.all(arr > 0):
        return arr
    elif np.all(arr < 0):
        QMessageBox.warning(parent, f"{label} Warning",
                            f"All values in {label} are negative. The signal will be flipped for analysis.")
        return -arr
    elif np.any(arr < 0):
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(f"{label} Warning")
        msg.setText(f"{label} contains both positive and negative values.\n"
                    "Do you want to continue with absolute values?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        result = msg.exec()
        if result == QMessageBox.Yes:
            return np.abs(arr)
        else:
            return None
    else:
        # Handles cases with zeros
        return arr

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