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
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QPushButton, QTableWidget,
                             QTableWidgetItem, QHeaderView, QComboBox, QLabel,
                             QWidget)


class SignalAnalysisTools:
    """
    A collection of signal analysis methods for processing time-series signal data.

    This class provides statistical, frequency domain (FFT) and time domain analysis
    for signal data. It works with the main application's data_signals dictionary
    which contains (time_array, values_array) tuples.
    """

    @staticmethod
    def calculate_statistics(parent, signal_name):
        """
        Calculate comprehensive statistical metrics for a signal.

        Args:
            parent: The parent application instance containing data_signals
            signal_name (str): Name of the signal to analyze

        Returns:
            dict: Dictionary containing statistical measures including:
                - Min: Minimum signal value
                - Max: Maximum signal value
                - Mean: Average value
                - Median: Middle value when sorted
                - Std Dev: Standard deviation (signal variability)
                - RMS: Root mean square (effective power)
        """
        _, values = parent.data_signals[signal_name]
        stats = {
            'Min': np.min(values),
            'Max': np.max(values),
            'Mean': np.mean(values),
            'Median': np.median(values),
            'Std Dev': np.std(values),
            'RMS': np.sqrt(np.mean(np.square(values)))
        }
        return stats

    @staticmethod
    def perform_fft_analysis(parent, signal_name):
        """
        Perform Fast Fourier Transform analysis on a signal and display results.

        The FFT transforms time-domain data into frequency domain, revealing the frequency
        components present in the signal. This is particularly useful for identifying
        oscillations, harmonics, and noise characteristics.

        Args:
            parent: The parent application instance containing data_signals
            signal_name (str): Name of the signal to analyze

        Returns:
            None: Creates and shows a PyQtGraph window with the FFT results
        """
        time_arr, values = parent.data_signals[signal_name]

        # Calculate time step (dt) from time array
        dt = np.mean(np.diff(time_arr)) if len(time_arr) > 1 else 1.0

        # Perform FFT
        fft_values = np.fft.rfft(values)
        fft_freqs = np.fft.rfftfreq(len(values), dt)

        # Create FFT plot window
        fft_window = pg.PlotWidget(title=f"FFT Analysis: {signal_name}")
        fft_window.setWindowTitle(f"FFT Analysis: {signal_name}")
        fft_window.setLabel('left', 'Magnitude')
        fft_window.setLabel('bottom', 'Frequency (Hz)')
        fft_window.plot(fft_freqs, np.abs(fft_values), pen='b')
        fft_window.show()

        # Keep a reference to prevent garbage collection
        parent._fft_windows = getattr(parent, '_fft_windows', [])
        parent._fft_windows.append(fft_window)

    @staticmethod
    def analyze_signal(parent, signal_name):
        """
        Analyze time-domain characteristics of a signal.

        Extracts information about signal dynamics such as zero crossings,
        dominant frequency (estimated from zero crossings), rates of change,
        and signal energy. This provides insights into signal behavior over time.

        Args:
            parent: The parent application instance containing data_signals
            signal_name (str): Name of the signal to analyze

        Returns:
            dict: Dictionary containing time-domain analysis metrics:
                - Zero Crossings: Number of times signal crosses zero
                - Estimated Frequency: Dominant frequency estimated from zero crossings
                - Max Rising Rate: Maximum positive rate of change
                - Max Falling Rate: Maximum negative rate of change (absolute value)
                - Signal Energy: Sum of squared values (indication of signal power)
        """
        time_arr, values = parent.data_signals[signal_name]

        # Calculate zero crossings
        zero_crossings = np.where(np.diff(np.signbit(values)))[0]

        # Estimate dominant period if we have enough zero crossings
        if len(zero_crossings) > 4:
            # Distance between zero crossings can estimate period
            crossings_time = time_arr[zero_crossings]
            avg_period = np.mean(np.diff(crossings_time)) * 2  # Multiply by 2 because each cycle has 2 crossings
            frequency = 1.0 / avg_period if avg_period > 0 else 0
        else:
            frequency = 0

        # Calculate rising and falling rates
        diff_values = np.diff(values)
        diff_time = np.diff(time_arr)
        rates = diff_values / diff_time

        analysis = {
            'Zero Crossings': len(zero_crossings),
            'Estimated Frequency (Hz)': frequency,
            'Max Rising Rate': np.max(rates) if len(rates) > 0 else 0,
            'Max Falling Rate': abs(np.min(rates)) if len(rates) > 0 else 0,
            'Signal Energy': np.sum(np.square(values))
        }
        return analysis


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
        self.resize(600, 400)

        self.setup_ui()
        self.update_signal_list()

    def setup_ui(self):
        """
        Create and arrange the user interface components for the dialog.

        Sets up:
        - Signal selector dropdown
        - Analysis buttons (Statistics, FFT, Time Domain)
        - Results area for displaying analysis output
        - Close button
        """
        layout = QVBoxLayout(self)

        # Signal selector
        select_label = QLabel("Select Signal:")
        layout.addWidget(select_label)

        self.signal_combo = QComboBox()
        layout.addWidget(self.signal_combo)

        # Analysis buttons
        stats_btn = QPushButton("Basic Statistics")
        stats_btn.clicked.connect(self.show_statistics)
        layout.addWidget(stats_btn)

        fft_btn = QPushButton("FFT Analysis")
        fft_btn.clicked.connect(self.show_fft)
        layout.addWidget(fft_btn)

        time_analysis_btn = QPushButton("Time Domain Analysis")
        time_analysis_btn.clicked.connect(self.show_time_analysis)
        layout.addWidget(time_analysis_btn)

        # Results area (initially empty)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        layout.addWidget(self.results_widget)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def update_signal_list(self):
        """
        Populate the signal selection dropdown with available signals.

        Gets the list of signals from the parent application's data_signals dictionary.
        """
        self.signal_combo.clear()
        if hasattr(self.parent, 'data_signals'):
            signals = list(self.parent.data_signals.keys())
            self.signal_combo.addItems(signals)

    def get_selected_signal(self):
        """
        Get the name of the currently selected signal in the dropdown.

        Returns:
            str or None: Name of the selected signal, or None if no signal is selected
        """
        signal = self.signal_combo.currentText()
        if not signal:
            return None
        return signal

    def clear_results(self):
        """
        Clear all widgets from the results area to prepare for new results.

        Removes all existing widgets from the results layout to avoid
        accumulation of previous analysis results.
        """
        for i in reversed(range(self.results_layout.count())):
            widget = self.results_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

    def show_statistics(self):
        """
        Calculate and display basic statistics for the selected signal.

        Retrieves statistical metrics for the selected signal and
        displays them in a formatted table.
        """
        signal = self.get_selected_signal()
        if not signal:
            return

        stats = SignalAnalysisTools.calculate_statistics(self.parent, signal)
        self.show_analysis_results("Statistics", signal, stats)

    def show_fft(self):
        """
        Perform FFT analysis on the selected signal and display in a new window.

        Opens a separate window showing the frequency domain representation of the signal.
        """
        signal = self.get_selected_signal()
        if not signal:
            return

        SignalAnalysisTools.perform_fft_analysis(self.parent, signal)

    def show_time_analysis(self):
        """
        Perform time-domain analysis on the selected signal and display results.

        Analyzes time-based characteristics of the signal and displays the results
        in a formatted table.
        """
        signal = self.get_selected_signal()
        if not signal:
            return

        analysis = SignalAnalysisTools.analyze_signal(self.parent, signal)
        self.show_analysis_results("Time Analysis", signal, analysis)

    def show_analysis_results(self, title, signal, data_dict):
        """
        Display analysis results in the dialog's results area.

        Creates a formatted table showing analysis parameters and their values.

        Args:
            title (str): Title for the results section
            signal (str): Name of the analyzed signal
            data_dict (dict): Dictionary of analysis results to display
        """
        self.clear_results()

        # Add title
        result_title = QLabel(f"{title}: {signal}")
        result_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.results_layout.addWidget(result_title)

        # Create table to display results
        table = QTableWidget(len(data_dict), 2)
        table.setHorizontalHeaderLabels(["Parameter", "Value"])

        for i, (param, value) in enumerate(data_dict.items()):
            table.setItem(i, 0, QTableWidgetItem(str(param)))
            table.setItem(i, 1, QTableWidgetItem(
                f"{value:.6g}" if isinstance(value, float) else str(value)))

        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        self.results_layout.addWidget(table)


def show_analysis_dialog(parent):
    """
    Create and show the signal analysis dialog with the parent application's data.

    This is the main entry point for using this module from the main application.

    Args:
        parent: The parent application (should have data_signals attribute)

    Example:
        # In the main application:
        from signal_analysis import show_analysis_dialog

        def open_analysis_dialog(self):
            show_analysis_dialog(self)
    """
    dialog = SignalAnalysisDialog(parent=parent)
    dialog.exec()