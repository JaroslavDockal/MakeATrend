"""
Advanced Analysis Tab - Contains advanced signal analysis tools like PSD, Hilbert transform, etc.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFormLayout, QComboBox, QMainWindow, QDialog,
    QDoubleSpinBox, QSpinBox, QDialogButtonBox,
    QMessageBox
)

from utils.logger import Logger
from ..calculation import (
    calculate_psd_analysis,
    calculate_peak_detection,
    calculate_hilbert_analysis,
    calculate_energy_analysis,
    calculate_phase_analysis,
    calculate_cepstrum_analysis,
    calculate_autocorrelation_analysis,
    calculate_wavelet_analysis_cwt,
    calculate_wavelet_analysis_dwt,
    calculate_iir_filter,
    calculate_fir_filter,
    safe_prepare_signal,
    safe_sample_rate
)
from analysis.helpers import calculate_bandwidth, format_array_for_display


class AdvancedAnalysisTab(QWidget):
    """Tab containing advanced signal analysis operations."""

    def __init__(self, parent_dialog):
        super().__init__()
        self.dialog = parent_dialog
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface for advanced analysis."""
        Logger.log_message_static("Analysis-Dialog: Creating Advanced Analysis tab", Logger.DEBUG)

        layout = QVBoxLayout(self)

        # Signal selector for advanced tab
        select_layout = QFormLayout()
        self.signal_combo = QComboBox()
        select_layout.addRow("Select Signal:", self.signal_combo)
        layout.addLayout(select_layout)

        # Advanced analysis buttons - Row 1
        button_layout1 = QHBoxLayout()

        psd_btn = QPushButton("Power Spectral Density")
        psd_btn.setToolTip("Visualize the power distribution across frequencies.")
        psd_btn.clicked.connect(self.show_psd_analysis)
        button_layout1.addWidget(psd_btn)

        autocorr_btn = QPushButton("Autocorrelation")
        autocorr_btn.setToolTip("Measure the signal's self-similarity over time.")
        autocorr_btn.clicked.connect(self.show_autocorrelation_analysis)
        button_layout1.addWidget(autocorr_btn)

        peaks_btn = QPushButton("Peak Detection")
        peaks_btn.setToolTip("Identify and display peaks in the signal.")
        peaks_btn.clicked.connect(self.show_peak_detection)
        button_layout1.addWidget(peaks_btn)

        layout.addLayout(button_layout1)

        # Advanced analysis buttons - Row 2
        button_layout2 = QHBoxLayout()

        filter_btn = QPushButton("Apply Filter")
        filter_btn.setToolTip("Apply a frequency filter to the signal.")
        filter_btn.clicked.connect(self.show_filter_dialog)
        button_layout2.addWidget(filter_btn)

        hilbert_btn = QPushButton("Hilbert Transform")
        hilbert_btn.setToolTip("Extract amplitude, phase, and frequency details.")
        hilbert_btn.clicked.connect(self.show_hilbert_analysis)
        button_layout2.addWidget(hilbert_btn)

        energy_btn = QPushButton("Energy Analysis")
        energy_btn.setToolTip("Evaluate the energy distribution of the signal.")
        energy_btn.clicked.connect(self.show_energy_analysis)
        button_layout2.addWidget(energy_btn)

        layout.addLayout(button_layout2)

        # Advanced analysis buttons - Row 3
        button_layout3 = QHBoxLayout()

        phase_btn = QPushButton("Phase Analysis")
        phase_btn.setToolTip("Inspect the phase behavior of the signal.")
        phase_btn.clicked.connect(self.show_phase_analysis)
        button_layout3.addWidget(phase_btn)

        cepstrum_btn = QPushButton("Cepstral Analysis")
        cepstrum_btn.setToolTip("Reveal periodic patterns in the signal's spectrum.")
        cepstrum_btn.clicked.connect(self.show_cepstrum_analysis)
        button_layout3.addWidget(cepstrum_btn)

        wavelet_btn = QPushButton("Wavelet Transform")
        wavelet_btn.setToolTip("Decompose the signal into time-frequency components.")
        wavelet_btn.clicked.connect(self.show_wavelet_dialog)
        button_layout3.addWidget(wavelet_btn)

        layout.addLayout(button_layout3)

        Logger.log_message_static("Analysis-Dialog: Advanced Analysis tab setup complete", Logger.DEBUG)

    def update_signal_list(self, signals):
        """Update the signal list in the combo box."""
        self.signal_combo.clear()
        self.signal_combo.addItems(signals)

    def get_selected_signal(self):
        """Get the currently selected signal name."""
        signal = self.signal_combo.currentText()
        if not signal:
            Logger.log_message_static("Analysis-Dialog: No signal selected in advanced tab", Logger.WARNING)
            return None
        return signal

    def show_psd_analysis(self):
        """Calculate and display Power Spectral Density for the selected signal."""
        Logger.log_message_static("Analysis-Dialog: Preparing PSD analysis", Logger.INFO)

        signal = self.get_selected_signal()
        if not signal:
            return

        time_arr, values = self.dialog.get_signal_data(signal)
        if time_arr is None or values is None:
            return

        try:
            Logger.log_message_static(f"Analysis-Dialog: Retrieved {len(values)} data points for PSD analysis", Logger.DEBUG)

            result = calculate_psd_analysis(self.dialog, time_arr, values)
            if result is None:
                Logger.log_message_static("Analysis-Dialog: PSD analysis returned no result", Logger.WARNING)
                return

            freqs = result["Frequency Axis (Hz)"]
            psd = result["PSD (Power/Hz)"]
            peak_freq = result["Peak Frequency (Hz)"]

            # Create plot window
            self._create_psd_plot(signal, freqs, psd, peak_freq)

            # Show stats
            stats = {
                "Peak Frequency (Hz)": result["Peak Frequency (Hz)"],
                "Max Power (dB)": result["Peak Power (dB)"],
                "Total Power": result["Total Power"],
                "RMS Power": result["RMS Amplitude"],
                "Bandwidth (3dB)": calculate_bandwidth(freqs, psd)
            }

            self.dialog.show_analysis_results("PSD Analysis", signal, stats)

        except Exception as e:
            Logger.log_message_static(f"Analysis-Dialog: Error in PSD analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Analysis-Dialog: PSD analysis traceback: {traceback.format_exc()}", Logger.DEBUG)

    def _create_psd_plot(self, signal, freqs, psd, peak_freq):
        """Create PSD plot window."""
        plot_window = QMainWindow(self.dialog)
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

        self.dialog.add_plot_window(plot_window)
        Logger.log_message_static("Analysis-Dialog: PSD plot window displayed successfully", Logger.INFO)

    def show_peak_detection(self):
        """Detect and display peaks for the selected signal."""
        Logger.log_message_static("Analysis-Dialog: Preparing peak detection analysis", Logger.INFO)

        signal = self.get_selected_signal()
        if not signal:
            return

        time_arr, values = self.dialog.get_signal_data(signal)
        if time_arr is None or values is None:
            return

        try:
            result = calculate_peak_detection(self.dialog, time_arr, values)
            if result is None:
                return

            if "Result" in result:
                self.dialog.show_analysis_results("Peak Detection", signal, result)
                return

            # Create plot
            self._create_peak_plot(signal, time_arr, values, result)

            # Prepare display data
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

            self.dialog.show_analysis_results("Peak Detection", signal, display_data)
            Logger.log_message_static("Analysis-Dialog: Peak detection analysis complete", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Analysis-Dialog: Error in peak detection: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(traceback.format_exc(), Logger.DEBUG)

    def _create_peak_plot(self, signal, time_arr, values, result):
        """Create peak detection plot window."""
        Logger.log_message_static("Analysis-Dialog: Creating peak detection plot window", Logger.DEBUG)

        plot_window = QMainWindow(self.dialog)
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

        self.dialog.add_plot_window(plot_window)

    def show_hilbert_analysis(self):
        """GUI wrapper for Hilbert analysis."""
        Logger.log_message_static("Analysis-Dialog: Preparing Hilbert transform analysis", Logger.INFO)

        signal = self.get_selected_signal()
        if not signal:
            return

        time_arr, values = self.dialog.get_signal_data(signal)
        if time_arr is None or values is None:
            return

        try:
            result = calculate_hilbert_analysis(self.dialog, time_arr, values)
            if result is None:
                return

            # Create plot
            self._create_hilbert_plot(signal, time_arr, values, result)

            # Show summary
            summary = {
                "Mean Amplitude": round(result["Mean Amplitude"], 3),
                "Max Amplitude": round(result["Max Amplitude"], 3),
                "Mean Frequency (Hz)": round(result["Mean Frequency (Hz)"], 3),
                "Median Frequency (Hz)": round(result["Median Frequency (Hz)"], 3),
                "Max Frequency (Hz)": round(result["Max Frequency (Hz)"], 3),
                "Phase Range (rad)": round(result["Phase Range (rad)"], 3)
            }

            self.dialog.show_analysis_results("Hilbert Transform", signal, summary)
            Logger.log_message_static("Analysis-Dialog: Hilbert transform analysis complete", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Analysis-Dialog: Error in Hilbert transform: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(traceback.format_exc(), Logger.DEBUG)

    def _create_hilbert_plot(self, signal, time_arr, values, result):
        """Create Hilbert analysis plot window."""
        amplitude_envelope = result["Amplitude Envelope"]
        phase = result["Unwrapped Phase"]
        inst_freq = result["Instantaneous Frequency (Hz)"]

        plot_window = QMainWindow(self.dialog)
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

        self.dialog.add_plot_window(plot_window)

    def show_energy_analysis(self):
        """Perform and display energy analysis."""
        Logger.log_message_static("Analysis-Dialog: Preparing energy analysis", Logger.INFO)

        signal = self.get_selected_signal()
        if not signal:
            return

        time_arr, values = self.dialog.get_signal_data(signal)
        if time_arr is None or values is None:
            return

        try:
            result = calculate_energy_analysis(self.dialog, time_arr, values)
            if result is None:
                return

            # Create plot
            self._create_energy_plot(signal, result)

            # Show summary
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

            self.dialog.show_analysis_results("Energy Analysis", signal, summary)
            Logger.log_message_static("Analysis-Dialog: Energy analysis complete", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Analysis-Dialog: Error in energy analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(traceback.format_exc(), Logger.DEBUG)

    def _create_energy_plot(self, signal, result):
        """Create energy analysis plot window."""
        freqs = result["Freqs"]
        spectrum = result["Spectrum"]
        band_edges = result["Band Edges"]
        band_percentages = result["Band Percentages"]

        plot_window = QMainWindow(self.dialog)
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

        self.dialog.add_plot_window(plot_window)

    def show_phase_analysis(self):
        """Perform phase analysis using the Hilbert transform."""
        Logger.log_message_static("Analysis-Dialog: Preparing phase analysis", Logger.INFO)

        signal = self.get_selected_signal()
        if not signal:
            return

        time_arr, values = self.dialog.get_signal_data(signal)
        if time_arr is None or values is None:
            return

        try:
            result = calculate_phase_analysis(self.dialog, time_arr, values)
            if result is None:
                return

            # Create plot
            self._create_phase_plot(signal, time_arr, result)

            # Show stats
            stats = result["Phase Stats"]
            formatted_stats = {k: round(v, 4) for k, v in stats.items()}
            self.dialog.show_analysis_results("Phase Analysis", signal, formatted_stats)
            Logger.log_message_static("Analysis-Dialog: Phase analysis complete", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Analysis-Dialog: Error in phase analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(traceback.format_exc(), Logger.DEBUG)

    def _create_phase_plot(self, signal, time_arr, result):
        """Create phase analysis plot window."""
        phase = result["Phase"]
        velocity = result["Phase Velocity"]
        velocity_time = time_arr[1:]

        plot_window = QMainWindow(self.dialog)
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

        self.dialog.add_plot_window(plot_window)

    def show_cepstrum_analysis(self):
        """GUI wrapper for cepstrum analysis."""
        Logger.log_message_static("Analysis-Dialog: Preparing cepstrum analysis", Logger.INFO)

        signal = self.get_selected_signal()
        if not signal:
            return

        time_arr, values = self.dialog.get_signal_data(signal)
        if time_arr is None or values is None:
            return

        try:
            result = calculate_cepstrum_analysis(self.dialog, time_arr, values)
            if result is None:
                return

            # Create plot
            self._create_cepstrum_plot(signal, time_arr, values, result)

            # Show summary
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

            self.dialog.show_analysis_results("Cepstrum Analysis", signal, summary)
            Logger.log_message_static("Analysis-Dialog: Cepstrum analysis complete", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Analysis-Dialog: Error in cepstrum analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Analysis-Dialog: Cepstrum traceback: {traceback.format_exc()}", Logger.DEBUG)

    def _create_cepstrum_plot(self, signal, time_arr, values, result):
        """Create cepstrum analysis plot window."""
        cepstrum = result["Cepstrum"]
        quefrency = result["Quefrency"]
        log_power = result["Log Power Spectrum"]
        fs = result["Sampling Rate"]

        n = len(cepstrum)
        freqs = np.fft.fftfreq(n, d=1 / fs)
        freqs = np.fft.fftshift(freqs)
        log_power_shift = np.fft.fftshift(log_power)

        plot_window = QMainWindow(self.dialog)
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

        self.dialog.add_plot_window(plot_window)

    def show_autocorrelation_analysis(self):
        """GUI wrapper for autocorrelation analysis."""
        Logger.log_message_static("Analysis-Dialog: Preparing autocorrelation analysis", Logger.INFO)

        signal = self.get_selected_signal()
        if not signal:
            return

        time_arr, values = self.dialog.get_signal_data(signal)
        if time_arr is None or values is None:
            return

        try:
            result = calculate_autocorrelation_analysis(self.dialog, time_arr, values)
            if result is None:
                return

            # Create plot
            self._create_autocorrelation_plot(signal, result)

            # Show results
            result_display = {
                "Peak Correlation": result["Peak Correlation"],
                "First Minimum": result["First Minimum (s)"],
                "First Zero Crossing": result["First Zero Crossing (s)"],
                "Decorrelation Time": result["Decorrelation Time (s)"]
            }

            self.dialog.show_analysis_results("Autocorrelation Analysis", signal, result_display)
            Logger.log_message_static("Analysis-Dialog: Autocorrelation analysis complete", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Analysis-Dialog: Error in autocorrelation analysis: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Analysis-Dialog: Autocorrelation traceback: {traceback.format_exc()}", Logger.DEBUG)

    def _create_autocorrelation_plot(self, signal, result):
        """Create autocorrelation plot window."""
        acf = result["Autocorrelation"]
        lags = result["Lags (s)"]

        plot_window = QMainWindow(self.dialog)
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

        self.dialog.add_plot_window(plot_window)

    def show_wavelet_dialog(self):
        """Show wavelet analysis parameter dialog and run analysis."""
        Logger.log_message_static("Analysis-Dialog: Opening wavelet analysis dialog", Logger.INFO)

        signal = self.get_selected_signal()
        if not signal:
            return

        time_arr, values = self.dialog.get_signal_data(signal)
        if time_arr is None or values is None:
            return

        try:
            processed = safe_prepare_signal(values, self.dialog, "Wavelet Analysis")
            if processed is None:
                return

            # Create wavelet dialog
            dialog = QDialog(self.dialog)
            dialog.setWindowTitle("Wavelet Analysis Settings")
            layout = QVBoxLayout(dialog)
            form = QFormLayout()

            # Wavelet type selection
            wavelet_combo = QComboBox()
            wavelet_combo.addItems(["morlet", "mexh", "cgau1", "db1", "db4", "db8", "sym4", "sym8", "coif2", "coif4"])
            form.addRow("Wavelet Type:", wavelet_combo)

            # Analysis type selection
            analysis_combo = QComboBox()
            analysis_combo.addItems(["CWT (Continuous)", "DWT (Discrete)"])
            form.addRow("Analysis Type:", analysis_combo)

            # CWT-specific parameters
            scales_spin = QSpinBox()
            scales_spin.setRange(10, 200)
            scales_spin.setValue(50)
            form.addRow("Number of Scales (CWT):", scales_spin)

            # DWT-specific parameters
            levels_spin = QSpinBox()
            levels_spin.setRange(1, 10)
            levels_spin.setValue(5)
            form.addRow("Decomposition Levels (DWT):", levels_spin)

            layout.addLayout(form)

            # Buttons
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)

            if not dialog.exec():
                Logger.log_message_static("Analysis-Dialog: Wavelet dialog cancelled by user", Logger.INFO)
                return

            # Get parameters
            wavelet_type = wavelet_combo.currentText()
            analysis_type = analysis_combo.currentText()
            scales = scales_spin.value()
            levels = levels_spin.value()

            # Perform analysis
            if "CWT" in analysis_type:
                result = calculate_wavelet_analysis_cwt(self.dialog, time_arr, processed, wavelet_type, scales)
            else:
                result = calculate_wavelet_analysis_dwt(self.dialog, processed, wavelet_type, levels)

            if result is None:
                QMessageBox.warning(self.dialog, "Wavelet Analysis", "Analysis failed or was cancelled.")
                return

            # Show results
            self.dialog.show_analysis_results("Wavelet Analysis", signal, result)

            # Create plot if CWT
            if "Coefficients" in result:
                self._create_cwt_plot(signal, result)

            Logger.log_message_static("Analysis-Dialog: Wavelet analysis complete", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Analysis-Dialog: Wavelet analysis failed: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Analysis-Dialog: Traceback: {traceback.format_exc()}", Logger.DEBUG)

    def _create_cwt_plot(self, signal, result):
        """Create CWT spectrogram plot window."""
        plot_window = QMainWindow(self.dialog)
        plot_window.setWindowTitle(f"CWT Spectrogram: {signal}")
        plot_window.resize(800, 600)

        central = QWidget()
        layout = QVBoxLayout(central)

        img_widget = pg.ImageView()
        coeffs = result["Coefficients"]
        time_arr = result["Time Array"]

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

        self.dialog.add_plot_window(plot_window)
        Logger.log_message_static("Analysis-Dialog: CWT image plot displayed", Logger.INFO)

    def show_filter_dialog(self):
        """Show filter configuration dialog and display filtered signal."""
        Logger.log_message_static("Analysis-Dialog: Opening filter dialog", Logger.INFO)

        signal = self.get_selected_signal()
        if not signal:
            return

        time_arr, values = self.dialog.get_signal_data(signal)
        if time_arr is None or values is None:
            return

        try:
            sample_rate = safe_sample_rate(time_arr)
            if sample_rate == 0.0:
                QMessageBox.information(self.dialog, "Filter", "Sampling rate could not be determined.")
                return

            # Create filter dialog
            dialog = QDialog(self.dialog)
            dialog.setWindowTitle("Filter Settings")
            layout = QVBoxLayout(dialog)
            form = QFormLayout()

            # Filter type
            filter_type_box = QComboBox()
            filter_type_box.addItems(["lowpass", "highpass", "bandpass", "bandstop"])
            form.addRow("Filter Type:", filter_type_box)

            # Filter design
            design_box = QComboBox()
            design_box.addItems(["IIR (Butterworth)", "FIR"])
            form.addRow("Filter Design:", design_box)

            # Cutoff frequency 1
            cutoff1_spin = QDoubleSpinBox()
            cutoff1_spin.setRange(0.1, sample_rate / 2)
            cutoff1_spin.setValue(10.0)
            cutoff1_spin.setSuffix(" Hz")
            form.addRow("Cutoff Frequency:", cutoff1_spin)

            # Cutoff frequency 2 (for bandpass/bandstop)
            cutoff2_spin = QDoubleSpinBox()
            cutoff2_spin.setRange(0.1, sample_rate / 2)
            cutoff2_spin.setValue(100.0)
            cutoff2_spin.setSuffix(" Hz")
            cutoff2_spin.setVisible(False)
            form.addRow("Second Cutoff:", cutoff2_spin)

            # Filter order/taps
            order_spin = QSpinBox()
            order_spin.setRange(1, 20)
            order_spin.setValue(4)
            form.addRow("Order (IIR) / Taps (FIR):", order_spin)

            # FIR window
            fir_window_box = QComboBox()
            fir_window_box.addItems(["hamming", "hann", "blackman", "bartlett", "boxcar"])
            form.addRow("FIR Window:", fir_window_box)

            layout.addLayout(form)

            # Update cutoff2 visibility based on filter type
            def update_cutoffs():
                ftype = filter_type_box.currentText()
                cutoff2_spin.setVisible(ftype in ["bandpass", "bandstop"])

            filter_type_box.currentTextChanged.connect(update_cutoffs)

            # Buttons
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)

            if not dialog.exec():
                Logger.log_message_static("Analysis-Dialog: Filter dialog cancelled by user", Logger.INFO)
                return

            # Get settings
            filter_type = filter_type_box.currentText()
            design_type = design_box.currentText()
            cutoff_freq = [cutoff1_spin.value(), cutoff2_spin.value()] if filter_type in ["bandpass", "bandstop"] else cutoff1_spin.value()
            order = order_spin.value()
            fir_window = fir_window_box.currentText()

            Logger.log_message_static(
                f"Analysis-Dialog: Filter config: {filter_type}, {design_type}, cutoff={cutoff_freq}, order={order}", Logger.DEBUG)

            # Apply filter
            result = self._apply_filter_and_return_result(
                time_arr, values,
                filter_type=filter_type,
                cutoff_freq=cutoff_freq,
                order=order,
                filter_method="fir" if "FIR" in design_type else "iir",
                numtaps=order,
                window=fir_window
            )

            if not result:
                return

            # Create plot
            self._create_filter_plot(signal, time_arr, values, result, filter_type, design_type)

            Logger.log_message_static("Analysis-Dialog: Filter result displayed", Logger.INFO)

        except Exception as e:
            Logger.log_message_static(f"Analysis-Dialog: Error in filter dialog: {e}", Logger.ERROR)
            import traceback
            Logger.log_message_static(traceback.format_exc(), Logger.DEBUG)

    def _apply_filter_and_return_result(self, time_arr, values, filter_type="lowpass", cutoff_freq=1.0,
                                      order=4, filter_method="iir", numtaps=101, window="hamming"):
        """Apply a filter to the signal and return results."""
        try:
            Logger.log_message_static(f"Analysis-Dialog: Filter config: {filter_type}, {filter_method.upper()}"
                                      f"{' (Butterworth)' if filter_method == 'iir' else ''}, "
                                      f"cutoff={cutoff_freq}, "
                                      f"{'order' if filter_method == 'iir' else 'taps'}="
                                      f"{order if filter_method == 'iir' else numtaps}",
                                      Logger.DEBUG)

            # Prepare signal
            processed_values = safe_prepare_signal(values, self.dialog, "Filter Application")
            if processed_values is None:
                return None

            # Apply filter based on method
            if filter_method.lower() == "iir":
                result = calculate_iir_filter(processed_values, time_arr, filter_type, cutoff_freq, order)
            elif filter_method.lower() == "fir":
                result = calculate_fir_filter(processed_values, time_arr, filter_type, cutoff_freq, numtaps, window)
            else:
                Logger.log_message_static(f"Analysis-Dialog: Unknown filter method: {filter_method}", Logger.ERROR)
                return None

            if result is None:
                Logger.log_message_static("Analysis-Dialog: Filtering failed", Logger.ERROR)
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
            Logger.log_message_static(f"Analysis-Dialog: Error in filter application: {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(traceback.format_exc(), Logger.DEBUG)
            return None

    def _create_filter_plot(self, signal, time_arr, values, result, filter_type, design_type):
        """Create filter result plot window."""
        filtered = result["Filtered Signal"]

        plot_window = QMainWindow(self.dialog)
        plot_window.setWindowTitle(f"{design_type} Filtered Signal: {signal}")
        plot_window.resize(800, 600)

        central = QWidget()
        vbox = QVBoxLayout(central)
        widget = pg.GraphicsLayoutWidget()

        p1 = pg.PlotItem(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
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

        self.dialog.add_plot_window(plot_window)