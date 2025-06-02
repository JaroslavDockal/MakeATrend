"""
Cross Analysis Tab - Contains cross-signal analysis tools like cross-correlation.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFormLayout, QComboBox, QMainWindow
)

from utils.logger import Logger
from ..calculation import calculate_cross_correlation_analysis


class CrossAnalysisTab(QWidget):
    """Tab containing cross-signal analysis operations."""

    def __init__(self, parent_dialog):
        super().__init__()
        self.dialog = parent_dialog
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface for cross analysis."""
        Logger.log_message_static("Analysis-Dialog: Creating Cross Analysis tab", Logger.DEBUG)

        layout = QVBoxLayout(self)

        # Signal selectors for cross analysis
        select_layout = QFormLayout()

        self.signal1_combo = QComboBox()
        self.signal2_combo = QComboBox()

        select_layout.addRow("Signal 1:", self.signal1_combo)
        select_layout.addRow("Signal 2:", self.signal2_combo)
        layout.addLayout(select_layout)

        # Cross analysis buttons
        button_layout = QHBoxLayout()

        xcorr_btn = QPushButton("Cross Correlation")
        xcorr_btn.setToolTip("Compare and find similarities between two signals.")
        xcorr_btn.clicked.connect(self.show_cross_correlation_analysis)
        button_layout.addWidget(xcorr_btn)

        # Placeholder for future cross-analysis methods
        coherence_btn = QPushButton("Coherence Analysis")
        coherence_btn.setToolTip("Analyze frequency-domain coherence between signals.")
        coherence_btn.setEnabled(False)  # Not implemented yet
        button_layout.addWidget(coherence_btn)

        phase_diff_btn = QPushButton("Phase Difference")
        phase_diff_btn.setToolTip("Calculate phase difference between signals.")
        phase_diff_btn.setEnabled(False)  # Not implemented yet
        button_layout.addWidget(phase_diff_btn)

        layout.addLayout(button_layout)

        Logger.log_message_static("Analysis-Dialog: Cross Analysis tab setup complete", Logger.DEBUG)

    def update_signal_list(self, signals):
        """Update the signal lists in both combo boxes."""
        self.signal1_combo.clear()
        self.signal1_combo.addItems(signals)

        self.signal2_combo.clear()
        self.signal2_combo.addItems(signals)

    def get_selected_signals(self):
        """Get the currently selected signal names."""
        signal1 = self.signal1_combo.currentText()
        signal2 = self.signal2_combo.currentText()

        if not signal1 or not signal2:
            Logger.log_message_static("Analysis-Dialog: One or both signals not selected in cross tab", Logger.WARNING)
            return None, None

        return signal1, signal2

    def show_cross_correlation_analysis(self):
        """
        GUI wrapper to show cross-correlation analysis results between two selected signals.
        """
        Logger.log_message_static("Analysis-Dialog: Preparing cross-correlation analysis", Logger.INFO)

        signal1, signal2 = self.get_selected_signals()
        if not signal1 or not signal2:
            return

        # Get signal data
        time_arr1, values1 = self.dialog.get_signal_data(signal1)
        time_arr2, values2 = self.dialog.get_signal_data(signal2)

        if time_arr1 is None or values1 is None or time_arr2 is None or values2 is None:
            return

        try:
            Logger.log_message_static(
                f"Analysis-Dialog: Computing cross-correlation between '{signal1}' and '{signal2}'", Logger.DEBUG)

            results = calculate_cross_correlation_analysis(self.dialog, time_arr1, values1, time_arr2, values2)
            if results is None:
                Logger.log_message_static("Analysis-Dialog: Cross-correlation analysis aborted", Logger.INFO)
                return

            # Create cross-correlation plot
            self._create_cross_correlation_plot(signal1, signal2, time_arr1, values1, time_arr2, values2, results)

            # Prepare stats for display
            display_stats = {
                "Maximum Correlation": f"{results['Max Correlation']:.4f}",
                "Lag at Maximum Correlation": f"{results['Lag at Max Correlation (s)']:.6f} s",
                "Correlation at Zero Lag": f"{results['Correlation at Zero Lag']:.4f}",
            }

            # Add correlation width information if available
            for key in sorted(results):
                if key.startswith("Correlation Width"):
                    display_stats[key] = f"{results[key]} s" if isinstance(results[key], float) else results[key]

            self.dialog.show_analysis_results("Cross-Correlation Analysis", f"{signal1} & {signal2}", display_stats)
            Logger.log_message_static("Analysis-Dialog: Cross-correlation analysis complete", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Analysis-Dialog: Error in cross-correlation analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Analysis-Dialog: Traceback: {traceback.format_exc()}", Logger.DEBUG)

    def _create_cross_correlation_plot(self, signal1, signal2, time_arr1, values1, time_arr2, values2, results):
        """Create cross-correlation plot window."""
        lags = results["Lags (s)"]
        corr = results["Cross-Correlation"]
        max_corr = results["Max Correlation"]
        max_lag = results["Lag at Max Correlation (s)"]

        # Plot setup
        plot_window = QMainWindow(self.dialog)
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

        # Add reference lines
        p1.addLine(y=0, pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
        p1.addLine(x=max_lag, pen=pg.mkPen('g', width=2))

        # Add max correlation point
        max_idx = np.argmax(np.abs(corr))
        scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen('r', width=2), brush=pg.mkBrush('r'))
        scatter.addPoints([{'pos': (max_lag, max_corr)}])
        p1.addItem(scatter)

        # Original signals plot
        p2 = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
        p2.setTitle("Original Signals")
        p2.setLabel('left', 'Amplitude')
        p2.setLabel('bottom', 'Time (s)')

        # Plot both signals
        curve1 = p2.plot(time_arr1, values1, pen='b', name=signal1)
        curve2 = p2.plot(time_arr2, values2, pen='r', name=signal2)

        # Add legend
        legend = p2.addLegend()
        legend.addItem(curve1, signal1)
        legend.addItem(curve2, signal2)

        # Show shifted signal if there's significant lag
        if abs(max_lag) > 0.001:
            time_arr2_shifted = time_arr2 + max_lag
            valid_mask = (time_arr2_shifted >= min(time_arr1)) & (time_arr2_shifted <= max(time_arr1))

            if np.any(valid_mask):
                shifted_curve = p2.plot(
                    time_arr2_shifted[valid_mask],
                    values2[valid_mask],
                    pen=pg.mkPen('g', width=2, style=Qt.PenStyle.DashLine),
                    name=f"{signal2} (shifted)"
                )
                legend.addItem(shifted_curve, f"{signal2} (shifted by {max_lag:.4f}s)")

        plot_widget.addItem(p2, row=1, col=0)

        # Add correlation coefficient text
        corr_text = pg.TextItem(
            text=f"Max Correlation: {max_corr:.4f}\nAt Lag: {max_lag:.4f}s",
            color='black',
            anchor=(0, 1)
        )
        corr_text.setPos(lags[len(lags) // 4], max(corr) * 0.8)
        p1.addItem(corr_text)

        close_button = QPushButton("Close")
        close_button.clicked.connect(plot_window.close)

        layout.addWidget(plot_widget)
        layout.addWidget(close_button)
        central_widget.setLayout(layout)
        plot_window.setCentralWidget(central_widget)
        plot_window.show()

        self.dialog.add_plot_window(plot_window)
        Logger.log_message_static("Analysis-Dialog: Cross-correlation plot window displayed successfully", Logger.INFO)

    def show_coherence_analysis(self):
        """
        Placeholder for coherence analysis between two signals.
        This would calculate the magnitude-squared coherence.
        """
        Logger.log_message_static("Analysis-Dialog: Coherence analysis not implemented yet", Logger.INFO)
        # TODO: Implement coherence analysis
        pass

    def show_phase_difference_analysis(self):
        """
        Placeholder for phase difference analysis between two signals.
        This would calculate the phase difference in frequency domain.
        """
        Logger.log_message_static("Analysis-Dialog: Phase difference analysis not implemented yet", Logger.INFO)
        # TODO: Implement phase difference analysis
        pass