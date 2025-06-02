"""
Specializované okno pro vibrační analýzy.

Toto okno poskytuje pokročilé rozhraní speciálně navržené pro vibrační analýzy:
- Multi-kanálové analýzy (DE/NDE, X/Y/Z)
- Automatická detekce vibračních signálů
- Harmonická analýza s RPM
- Envelope analýza pro ložiska
- ISO severity assessment
- Generování reportů

Optimalizováno pro workflow vibrační diagnostiky.
"""

import sys
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QTabWidget, QGroupBox, QPushButton, QLabel,
                               QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit,
                               QTextEdit, QProgressBar, QMessageBox, QCheckBox,
                               QFrame, QScrollArea, QGridLayout, QFormLayout,
                               QSplitter, QTableWidget, QTableWidgetItem,
                               QTreeWidget, QTreeWidgetItem, QHeaderView)
from PySide6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PySide6.QtGui import QFont, QIcon, QPalette, QAction

import numpy as np
from typing import Dict, List, Optional, Any, Tuple

# Import vibračních funkcí
try:
    from analysis.vibration import *
    from analysis.calculations import *

    VIBRATION_AVAILABLE = True
except ImportError:
    VIBRATION_AVAILABLE = False

from utils.logger import Logger


class VibrationAnalysisWorker(QThread):
    """Worker thread pro vibrační analýzy."""

    analysis_completed = pyqtSignal(str, dict)  # analysis_type, result
    analysis_failed = pyqtSignal(str, str)  # analysis_type, error
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)

    def __init__(self, signal_data, analysis_tasks):
        super().__init__()
        self.signal_data = signal_data
        self.analysis_tasks = analysis_tasks

    def run(self):
        """Spustí všechny naplánované analýzy."""
        total_tasks = len(self.analysis_tasks)

        for i, (analysis_type, params) in enumerate(self.analysis_tasks):
            try:
                self.status_updated.emit(f"Running {analysis_type}...")
                progress = int((i / total_tasks) * 90)  # Reserve 10% for completion
                self.progress_updated.emit(progress)

                result = self.run_single_analysis(analysis_type, params)

                if result:
                    self.analysis_completed.emit(analysis_type, result)
                else:
                    self.analysis_failed.emit(analysis_type, "No results returned")

            except Exception as e:
                Logger.log_message_static(f"VibrationAnalysisWorker: Error in {analysis_type}: {e}", Logger.ERROR)
                self.analysis_failed.emit(analysis_type, str(e))

        self.progress_updated.emit(100)
        self.status_updated.emit("Analysis completed")

    def run_single_analysis(self, analysis_type: str, params: dict):
        """Spustí jednotlivou analýzu."""
        if analysis_type == "signal_detection":
            return detect_vibration_signals(self.signal_data)

        elif analysis_type == "vibration_metrics":
            channel = params.get('channel')
            if channel and channel in self.signal_data:
                time_arr, values = self.signal_data[channel]
                return calculate_vibration_metrics(time_arr, values)

        elif analysis_type == "vibration_fft":
            channel = params.get('channel')
            rpm = params.get('rpm')
            if channel and channel in self.signal_data:
                time_arr, values = self.signal_data[channel]
                return calculate_vibration_fft(time_arr, values, rpm=rpm)

        elif analysis_type == "envelope_analysis":
            channel = params.get('channel')
            filter_type = params.get('filter_type', 'adaptive')
            if channel and channel in self.signal_data:
                time_arr, values = self.signal_data[channel]
                return calculate_envelope_analysis(time_arr, values, filter_type=filter_type)

        elif analysis_type == "bearing_frequencies":
            rpm = params.get('rpm')
            bearing_params = params.get('bearing_params')
            if rpm and bearing_params:
                return calculate_bearing_fault_frequencies(rpm, bearing_params)

        elif analysis_type == "severity_assessment":
            rms_value = params.get('rms_value')
            machine_class = params.get('machine_class', 'II')
            if rms_value:
                return assess_vibration_severity_iso10816(rms_value, machine_class)

        return None


class VibrationWindow(QMainWindow):
    """
    Specializované okno pro vibrační analýzy.

    Poskytuje kompletní workflow pro analýzu vibračních signálů
    včetně multi-kanálových analýz a reportování.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Vibration Analysis - MakeATrend")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Data a výsledky
        self.signal_data = {}  # {channel_name: (time_arr, values)}
        self.analysis_results = {}  # {analysis_type: {channel: result}}
        self.machine_info = {}
        self.current_rpm = 1800

        # UI komponenty
        self.worker_thread = None
        self.progress_bar = None
        self.status_label = None
        self.signal_tree = None
        self.results_tabs = None

        # Setup
        self.setup_ui()
        self.setup_menus()
        self.check_vibration_availability()

        Logger.log_message_static("VibrationWindow: Window initialized", Logger.DEBUG)

    def setup_ui(self):
        """Nastavení hlavního UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Hlavní layout
        main_layout = QVBoxLayout()

        # Header
        header = self.create_header()
        main_layout.addWidget(header)

        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Levý panel - signály a nastavení
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)

        # Pravý panel - výsledky
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)

        # Nastavení velikostí
        main_splitter.setSizes([400, 1000])
        main_layout.addWidget(main_splitter)

        # Footer
        footer = self.create_footer()
        main_layout.addWidget(footer)

        central_widget.setLayout(main_layout)

    def create_header(self) -> QFrame:
        """Vytvoří header s informacemi."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout()

        # Název
        title = QLabel("Vibration Analysis Center")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        layout.addStretch()

        # Rychlé akce
        auto_detect_btn = QPushButton("Auto-Detect Signals")
        auto_detect_btn.clicked.connect(self.auto_detect_signals)
        layout.addWidget(auto_detect_btn)

        run_all_btn = QPushButton("Run All Analyses")
        run_all_btn.clicked.connect(self.run_comprehensive_analysis)
        layout.addWidget(run_all_btn)

        frame.setLayout(layout)
        return frame

    def create_left_panel(self) -> QTabWidget:
        """Vytvoří levý panel s ovládáním."""
        tab_widget = QTabWidget()

        # Tab 1: Signály
        signals_tab = self.create_signals_tab()
        tab_widget.addTab(signals_tab, "Signals")

        # Tab 2: Nastavení stroje
        machine_tab = self.create_machine_tab()
        tab_widget.addTab(machine_tab, "Machine Setup")

        # Tab 3: Analýzy
        analysis_tab = self.create_analysis_tab()
        tab_widget.addTab(analysis_tab, "Analysis")

        return tab_widget

    def create_signals_tab(self) -> QWidget:
        """Vytvoří tab pro správu signálů."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Signal tree
        signals_group = QGroupBox("Available Signals")
        signals_layout = QVBoxLayout()

        self.signal_tree = QTreeWidget()
        self.signal_tree.setHeaderLabels(["Signal", "Type", "Length", "Rate"])
        self.signal_tree.itemSelectionChanged.connect(self.on_signal_selected)
        signals_layout.addWidget(self.signal_tree)

        # Tlačítka pro signály
        signal_buttons = QHBoxLayout()

        load_btn = QPushButton("Load Signals")
        load_btn.clicked.connect(self.load_signals)
        signal_buttons.addWidget(load_btn)

        classify_btn = QPushButton("Classify Signals")
        classify_btn.clicked.connect(self.classify_signals)
        signal_buttons.addWidget(classify_btn)

        signals_layout.addLayout(signal_buttons)
        signals_group.setLayout(signals_layout)
        layout.addWidget(signals_group)

        # RPM nastavení
        rpm_group = QGroupBox("RPM Settings")
        rpm_layout = QFormLayout()

        self.rpm_spinbox = QSpinBox()
        self.rpm_spinbox.setRange(0, 10000)
        self.rpm_spinbox.setValue(1800)
        self.rpm_spinbox.setSuffix(" RPM")
        self.rpm_spinbox.valueChanged.connect(self.on_rpm_changed)
        rpm_layout.addRow("Operating Speed:", self.rpm_spinbox)

        rpm_group.setLayout(rpm_layout)
        layout.addWidget(rpm_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_machine_tab(self) -> QWidget:
        """Vytvoří tab pro nastavení stroje."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Základní info o stroji
        machine_group = QGroupBox("Machine Information")
        machine_layout = QFormLayout()

        self.machine_type_combo = QComboBox()
        self.machine_type_combo.addItems(["Motor", "Pump", "Fan", "Compressor", "Generator", "Other"])
        machine_layout.addRow("Machine Type:", self.machine_type_combo)

        self.power_spinbox = QDoubleSpinBox()
        self.power_spinbox.setRange(0.1, 10000.0)
        self.power_spinbox.setValue(75.0)
        self.power_spinbox.setSuffix(" kW")
        machine_layout.addRow("Rated Power:", self.power_spinbox)

        self.machine_class_combo = QComboBox()
        self.machine_class_combo.addItems(["I", "II", "III", "IV"])
        self.machine_class_combo.setCurrentText("II")
        machine_layout.addRow("ISO Machine Class:", self.machine_class_combo)

        machine_group.setLayout(machine_layout)
        layout.addWidget(machine_group)

        # Parametry ložisek
        bearing_group = QGroupBox("Bearing Parameters")
        bearing_layout = QFormLayout()

        # DE ložisko
        self.de_bearing_combo = QComboBox()
        self.populate_bearing_combo(self.de_bearing_combo)
        bearing_layout.addRow("DE Bearing:", self.de_bearing_combo)

        # NDE ložisko
        self.nde_bearing_combo = QComboBox()
        self.populate_bearing_combo(self.nde_bearing_combo)
        bearing_layout.addRow("NDE Bearing:", self.nde_bearing_combo)

        # Custom bearing parameters
        self.custom_balls_spinbox = QSpinBox()
        self.custom_balls_spinbox.setRange(3, 30)
        self.custom_balls_spinbox.setValue(8)
        bearing_layout.addRow("Custom - Balls:", self.custom_balls_spinbox)

        self.custom_pitch_spinbox = QDoubleSpinBox()
        self.custom_pitch_spinbox.setRange(10.0, 200.0)
        self.custom_pitch_spinbox.setValue(50.0)
        self.custom_pitch_spinbox.setSuffix(" mm")
        bearing_layout.addRow("Custom - Pitch Ø:", self.custom_pitch_spinbox)

        self.custom_ball_spinbox = QDoubleSpinBox()
        self.custom_ball_spinbox.setRange(1.0, 50.0)
        self.custom_ball_spinbox.setValue(8.0)
        self.custom_ball_spinbox.setSuffix(" mm")
        bearing_layout.addRow("Custom - Ball Ø:", self.custom_ball_spinbox)

        bearing_group.setLayout(bearing_layout)
        layout.addWidget(bearing_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_analysis_tab(self) -> QWidget:
        """Vytvoří tab pro ovládání analýz."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Rychlé analýzy
        quick_group = QGroupBox("Quick Analysis")
        quick_layout = QGridLayout()

        metrics_btn = QPushButton("Vibration Metrics")
        metrics_btn.clicked.connect(lambda: self.run_analysis_for_selected("vibration_metrics"))
        quick_layout.addWidget(metrics_btn, 0, 0)

        fft_btn = QPushButton("FFT + Harmonics")
        fft_btn.clicked.connect(lambda: self.run_analysis_for_selected("vibration_fft"))
        quick_layout.addWidget(fft_btn, 0, 1)

        envelope_btn = QPushButton("Envelope Analysis")
        envelope_btn.clicked.connect(lambda: self.run_analysis_for_selected("envelope_analysis"))
        quick_layout.addWidget(envelope_btn, 1, 0)

        severity_btn = QPushButton("ISO Severity")
        severity_btn.clicked.connect(lambda: self.run_analysis_for_selected("severity_assessment"))
        quick_layout.addWidget(severity_btn, 1, 1)

        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)

        # Pokročilé analýzy
        advanced_group = QGroupBox("Advanced Analysis")
        advanced_layout = QFormLayout()

        bearing_freq_btn = QPushButton("Calculate Bearing Frequencies")
        bearing_freq_btn.clicked.connect(self.calculate_bearing_frequencies)
        advanced_layout.addRow(bearing_freq_btn)

        multi_channel_btn = QPushButton("Multi-Channel Analysis")
        multi_channel_btn.clicked.connect(self.run_multi_channel_analysis)
        advanced_layout.addRow(multi_channel_btn)

        comprehensive_btn = QPushButton("Comprehensive Assessment")
        comprehensive_btn.clicked.connect(self.run_comprehensive_analysis)
        advanced_layout.addRow(comprehensive_btn)

        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        # Nastavení analýz
        settings_group = QGroupBox("Analysis Settings")
        settings_layout = QFormLayout()

        self.envelope_filter_combo = QComboBox()
        self.envelope_filter_combo.addItems(["adaptive", "bearing", "gear", "custom"])
        settings_layout.addRow("Envelope Filter:", self.envelope_filter_combo)

        self.auto_rpm_checkbox = QCheckBox("Auto-detect RPM")
        self.auto_rpm_checkbox.setChecked(True)
        settings_layout.addRow(self.auto_rpm_checkbox)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_right_panel(self) -> QTabWidget:
        """Vytvoří pravý panel s výsledky."""
        self.results_tabs = QTabWidget()

        # Tab pro každý typ analýzy
        self.create_results_tab("Overview", "overview")
        self.create_results_tab("Vibration Metrics", "metrics")
        self.create_results_tab("FFT Analysis", "fft")
        self.create_results_tab("Envelope Analysis", "envelope")
        self.create_results_tab("ISO Assessment", "severity")
        self.create_results_tab("Reports", "reports")

        return self.results_tabs

    def create_results_tab(self, title: str, tab_id: str) -> QWidget:
        """Vytvoří tab pro výsledky."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Text area pro výsledky
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setFont(QFont("Courier", 9))
        layout.addWidget(text_area)

        # Uložení reference
        setattr(self, f"{tab_id}_text", text_area)

        widget.setLayout(layout)
        self.results_tabs.addTab(widget, title)
        return widget

    def create_footer(self) -> QFrame:
        """Vytvoří footer s progress bar."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout()

        self.status_label = QLabel("Ready for vibration analysis")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(300)
        layout.addWidget(self.progress_bar)

        frame.setLayout(layout)
        return frame

    def setup_menus(self):
        """Nastavení menu."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        load_action = QAction("Load Signals", self)
        load_action.triggered.connect(self.load_signals)
        file_menu.addAction(load_action)

        export_action = QAction("Export Results", self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)

        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")

        auto_detect_action = QAction("Auto-Detect Signals", self)
        auto_detect_action.triggered.connect(self.auto_detect_signals)
        analysis_menu.addAction(auto_detect_action)

        comprehensive_action = QAction("Comprehensive Analysis", self)
        comprehensive_action.triggered.connect(self.run_comprehensive_analysis)
        analysis_menu.addAction(comprehensive_action)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        bearing_db_action = QAction("Bearing Database", self)
        bearing_db_action.triggered.connect(self.show_bearing_database)
        tools_menu.addAction(bearing_db_action)

        iso_info_action = QAction("ISO Standards Info", self)
        iso_info_action.triggered.connect(self.show_iso_info)
        tools_menu.addAction(iso_info_action)

    def check_vibration_availability(self):
        """Kontrola dostupnosti vibračních funkcí."""
        if not VIBRATION_AVAILABLE:
            QMessageBox.warning(
                self,
                "Vibration Analysis Not Available",
                "Vibration analysis functions are not available. Please check your installation."
            )
            self.setEnabled(False)

    def populate_bearing_combo(self, combo: QComboBox):
        """Naplní combo box s ložisky."""
        try:
            bearing_db = get_typical_bearing_parameters()
            combo.addItem("Custom", None)

            for designation, params in bearing_db.items():
                display_name = f"{designation} ({params.get('type', 'Unknown')})"
                combo.addItem(display_name, params)

        except Exception as e:
            Logger.log_message_static(f"VibrationWindow: Error populating bearing combo: {e}", Logger.WARNING)
            combo.addItem("Custom", None)

    def load_signals(self):
        """Načte signály (mock implementace - normálně by byl file dialog)."""
        # Pro testování vytvoříme ukázková data
        time_data = np.linspace(0, 10, 10000)

        test_signals = {
            'DE_X_Accel': (time_data, np.random.randn(10000) * 0.1 + np.sin(2 * np.pi * 30 * time_data)),
            'DE_Y_Accel': (time_data, np.random.randn(10000) * 0.08 + np.sin(2 * np.pi * 30 * time_data) * 0.8),
            'NDE_X_Velocity': (time_data, np.random.randn(10000) * 0.05 + np.sin(2 * np.pi * 30 * time_data) * 0.6),
            'RPM_Signal': (time_data, 1800 + np.random.randn(10000) * 10),
        }

        self.signal_data = test_signals
        self.update_signal_tree()

        self.status_label.setText(f"Loaded {len(test_signals)} signals")
        Logger.log_message_static(f"VibrationWindow: Loaded {len(test_signals)} test signals", Logger.DEBUG)

    def update_signal_tree(self):
        """Aktualizuje strom signálů."""
        self.signal_tree.clear()

        for signal_name, (time_arr, values) in self.signal_data.items():
            item = QTreeWidgetItem()
            item.setText(0, signal_name)

            # Klasifikace signálu
            classification = classify_signal_type(signal_name, values)
            item.setText(1, classification.get('type', 'Unknown'))

            # Délka a sample rate
            item.setText(2, str(len(values)))

            if len(time_arr) > 1:
                sample_rate = 1.0 / np.mean(np.diff(time_arr))
                item.setText(3, f"{sample_rate:.1f} Hz")
            else:
                item.setText(3, "N/A")

            # Uložení dat
            item.setData(0, Qt.ItemDataRole.UserRole, signal_name)

            self.signal_tree.addTopLevelItem(item)

        # Autosize columns
        self.signal_tree.header().resizeSections(QHeaderView.ResizeMode.ResizeToContents)

    def auto_detect_signals(self):
        """Automatická detekce vibračních signálů."""
        if not self.signal_data:
            QMessageBox.information(self, "No Signals", "Please load signals first.")
            return

        try:
            detection_result = detect_vibration_signals(self.signal_data)

            # Zobrazení výsledků detekce
            self.overview_text.clear()
            self.overview_text.append("=== SIGNAL DETECTION RESULTS ===\n")

            self.overview_text.append(f"Detection Summary: {detection_result.get('detection_summary', 'N/A')}\n")

            vibration_signals = detection_result.get('vibration_signals', [])
            rpm_signals = detection_result.get('rpm_signals', [])

            self.overview_text.append(f"Vibration Signals ({len(vibration_signals)}):")
            for signal in vibration_signals:
                classification = detection_result['classification'].get(signal, {})
                confidence = classification.get('confidence', 0)
                self.overview_text.append(f"  • {signal} (confidence: {confidence:.0f}%)")

            self.overview_text.append(f"\nRPM Signals ({len(rpm_signals)}):")
            for signal in rpm_signals:
                self.overview_text.append(f"  • {signal}")

            self.overview_text.append(f"\nRecommendations:")
            for rec in detection_result.get('recommendations', []):
                self.overview_text.append(f"  • {rec}")

            # Aktualizace RPM pokud detekováno
            if rpm_signals and self.auto_rpm_checkbox.isChecked():
                rpm_signal_name = rpm_signals[0]
                if rpm_signal_name in self.signal_data:
                    _, rpm_values = self.signal_data[rpm_signal_name]
                    avg_rpm = np.mean(rpm_values)
                    self.rpm_spinbox.setValue(int(avg_rpm))
                    self.status_label.setText(f"Auto-detected RPM: {avg_rpm:.0f}")

        except Exception as e:
            QMessageBox.critical(self, "Detection Error", f"Signal detection failed:\n{str(e)}")
            Logger.log_message_static(f"VibrationWindow: Signal detection error: {e}", Logger.ERROR)

    def classify_signals(self):
        """Klasifikuje všechny signály."""
        if not self.signal_data:
            return

        self.overview_text.clear()
        self.overview_text.append("=== SIGNAL CLASSIFICATION ===\n")

        for signal_name, (time_arr, values) in self.signal_data.items():
            classification = classify_signal_type(signal_name, values)

            self.overview_text.append(f"Signal: {signal_name}")
            self.overview_text.append(f"  Type: {classification.get('type', 'Unknown')}")
            self.overview_text.append(f"  Confidence: {classification.get('confidence', 0):.0f}%")

            if 'axis' in classification:
                self.overview_text.append(f"  Axis: {classification['axis']}")
            if 'location' in classification:
                self.overview_text.append(f"  Location: {classification['location']}")
            if 'measurement_type' in classification:
                self.overview_text.append(f"  Measurement: {classification['measurement_type']}")

            self.overview_text.append("")

    def on_signal_selected(self):
        """Zpracování výběru signálu."""
        selected_items = self.signal_tree.selectedItems()
        if selected_items:
            item = selected_items[0]
            signal_name = item.data(0, Qt.ItemDataRole.UserRole)
            self.status_label.setText(f"Selected: {signal_name}")

    def on_rpm_changed(self, value):
        """Zpracování změny RPM."""
        self.current_rpm = value
        Logger.log_message_static(f"VibrationWindow: RPM changed to {value}", Logger.DEBUG)

    def run_analysis_for_selected(self, analysis_type: str):
        """Spustí analýzu pro vybraný signál."""
        selected_items = self.signal_tree.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Please select a signal first.")
            return

        item = selected_items[0]
        signal_name = item.data(0, Qt.ItemDataRole.UserRole)

        if signal_name not in self.signal_data:
            QMessageBox.warning(self, "Signal Error", f"Signal {signal_name} not found.")
            return

        # Spuštění analýzy
        analysis_tasks = [(analysis_type, {'channel': signal_name, 'rpm': self.current_rpm})]
        self.start_analysis_worker(analysis_tasks)

    def run_multi_channel_analysis(self):
        """Spustí multi-kanálovou analýzu."""
        if not self.signal_data:
            QMessageBox.information(self, "No Signals", "Please load signals first.")
            return

        # Identifikace vibračních kanálů
        detection_result = detect_vibration_signals(self.signal_data)
        vibration_channels = detection_result.get('vibration_signals', [])

        if len(vibration_channels) < 2:
            QMessageBox.information(self, "Insufficient Channels",
                                    "At least 2 vibration channels needed for multi-channel analysis.")
            return

        # Vytvoření úkolů pro analýzu
        analysis_tasks = []

        for channel in vibration_channels:
            # Metriky pro každý kanál
            analysis_tasks.append(('vibration_metrics', {'channel': channel}))
            # FFT pro každý kanál
            analysis_tasks.append(('vibration_fft', {'channel': channel, 'rpm': self.current_rpm}))

        self.start_analysis_worker(analysis_tasks)

    def run_comprehensive_analysis(self):
        """Spustí kompletní analýzu všech signálů."""
        if not self.signal_data:
            QMessageBox.information(self, "No Signals", "Please load signals first.")
            return

        # Detekce signálů
        analysis_tasks = [('signal_detection', {})]

        # Analýzy pro každý vibrační kanál
        detection_result = detect_vibration_signals(self.signal_data)
        vibration_channels = detection_result.get('vibration_signals', [])

        for channel in vibration_channels:
            analysis_tasks.extend([
                ('vibration_metrics', {'channel': channel}),
                ('vibration_fft', {'channel': channel, 'rpm': self.current_rpm}),
                ('envelope_analysis', {'channel': channel, 'filter_type': self.envelope_filter_combo.currentText()})
            ])

        # Bearing frequencies
        bearing_params = self.get_current_bearing_params()
        if bearing_params:
            analysis_tasks.append(('bearing_frequencies', {
                'rpm': self.current_rpm,
                'bearing_params': bearing_params
            }))

        self.start_analysis_worker(analysis_tasks)

    def start_analysis_worker(self, analysis_tasks):
        """Spustí worker thread pro analýzy."""
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.information(self, "Analysis Running", "Please wait for current analysis to complete.")
            return

        self.worker_thread = VibrationAnalysisWorker(self.signal_data, analysis_tasks)
        self.worker_thread.analysis_completed.connect(self.on_analysis_completed)
        self.worker_thread.analysis_failed.connect(self.on_analysis_failed)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.status_updated.connect(self.status_label.setText)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker_thread.start()

    def on_analysis_completed(self, analysis_type: str, result: dict):
        """Zpracování dokončené analýzy."""
        # Uložení výsledku
        if analysis_type not in self.analysis_results:
            self.analysis_results[analysis_type] = {}

        self.analysis_results[analysis_type]['result'] = result

        # Zobrazení výsledku podle typu
        self.display_analysis_result(analysis_type, result)

        Logger.log_message_static(f"VibrationWindow: {analysis_type} analysis completed", Logger.DEBUG)

    def on_analysis_failed(self, analysis_type: str, error: str):
        """Zpracování chyby analýzy."""
        self.status_label.setText(f"{analysis_type} failed: {error}")
        Logger.log_message_static(f"VibrationWindow: {analysis_type} failed: {error}", Logger.ERROR)

    def display_analysis_result(self, analysis_type: str, result: dict):
        """Zobrazí výsledek analýzy v příslušném tabu."""
        try:
            if analysis_type == "signal_detection":
                self.display_signal_detection_result(result)
            elif analysis_type == "vibration_metrics":
                self.display_vibration_metrics_result(result)
            elif analysis_type == "vibration_fft":
                self.display_fft_result(result)
            elif analysis_type == "envelope_analysis":
                self.display_envelope_result(result)
            elif analysis_type == "bearing_frequencies":
                self.display_bearing_frequencies_result(result)
            elif analysis_type == "severity_assessment":
                self.display_severity_result(result)

        except Exception as e:
            Logger.log_message_static(f"VibrationWindow: Error displaying {analysis_type}: {e}", Logger.WARNING)

    def display_signal_detection_result(self, result: dict):
        """Zobrazí výsledky detekce signálů."""
        text = self.overview_text
        text.append("\n=== SIGNAL DETECTION COMPLETED ===")
        text.append(f"Summary: {result.get('detection_summary', 'N/A')}")
        text.append(f"Vibration signals: {len(result.get('vibration_signals', []))}")
        text.append(f"RPM signals: {len(result.get('rpm_signals', []))}")

    def display_vibration_metrics_result(self, result: dict):
        """Zobrazí výsledky vibračních metrik."""
        text = self.metrics_text
        text.append("\n=== VIBRATION METRICS ===")

        for key, value in result.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    text.append(f"{key}: {value:.6g}")
                else:
                    text.append(f"{key}: {value}")

    def display_fft_result(self, result: dict):
        """Zobrazí výsledky FFT analýzy."""
        text = self.fft_text
        text.append("\n=== FFT ANALYSIS ===")

        # Základní info
        if 'Peak Frequencies' in result:
            peak_freqs = result['Peak Frequencies']
            text.append(f"Number of peaks: {len(peak_freqs)}")
            if len(peak_freqs) > 0:
                text.append(f"Top peaks: {peak_freqs[:5]}")

        # Harmonická analýza
        if 'Harmonic Analysis' in result and result['Harmonic Analysis']:
            harmonic_info = result['Harmonic Analysis']
            text.append(f"Fundamental: {harmonic_info.get('Fundamental Frequency (Hz)', 'N/A')} Hz")
            if 'Harmonics' in harmonic_info:
                text.append(f"Harmonics detected: {len(harmonic_info['Harmonics'])}")

    def display_envelope_result(self, result: dict):
        """Zobrazí výsledky envelope analýzy."""
        text = self.envelope_text
        text.append("\n=== ENVELOPE ANALYSIS ===")

        if 'Quality Metrics' in result:
            quality = result['Quality Metrics']
            text.append(f"Analysis quality: {quality.get('overall_quality', 'Unknown')}")

        if 'Fault Detection' in result:
            faults = result['Fault Detection']
            text.append(f"Fault summary: {faults.get('fault_summary', 'No faults detected')}")

    def display_bearing_frequencies_result(self, result: dict):
        """Zobrazí výsledky výpočtu frekvencí ložisek."""
        text = self.envelope_text
        text.append("\n=== BEARING FAULT FREQUENCIES ===")

        for key, value in result.items():
            if isinstance(value, (int, float)) and 'Frequency' in key:
                text.append(f"{key}: {value:.2f} Hz")

    def display_severity_result(self, result: dict):
        """Zobrazí výsledky ISO severity assessment."""
        text = self.severity_text
        text.append("\n=== ISO SEVERITY ASSESSMENT ===")

        text.append(f"Severity Zone: {result.get('severity_zone', 'Unknown')}")
        text.append(f"Description: {result.get('severity_description', 'N/A')}")
        text.append(f"Action Required: {result.get('action_required', 'N/A')}")

        if 'monitoring_recommendations' in result:
            text.append("\nMonitoring Recommendations:")
            for rec in result['monitoring_recommendations']:
                text.append(f"  • {rec}")

    def get_current_bearing_params(self) -> dict:
        """Získá aktuální parametry ložiska."""
        try:
            # Pokud je vybrané custom, použij ruční hodnoty
            bearing_data = self.de_bearing_combo.currentData()

            if bearing_data is None:  # Custom
                return {
                    'balls': self.custom_balls_spinbox.value(),
                    'pitch_diameter': self.custom_pitch_spinbox.value(),
                    'ball_diameter': self.custom_ball_spinbox.value(),
                    'contact_angle': 0
                }
            else:
                return {
                    'balls': bearing_data['balls'],
                    'pitch_diameter': bearing_data['pitch_diameter'],
                    'ball_diameter': bearing_data['ball_diameter'],
                    'contact_angle': bearing_data.get('contact_angle', 0)
                }

        except Exception as e:
            Logger.log_message_static(f"VibrationWindow: Error getting bearing params: {e}", Logger.WARNING)
            return {
                'balls': 8,
                'pitch_diameter': 50.0,
                'ball_diameter': 8.0,
                'contact_angle': 0
            }

    def calculate_bearing_frequencies(self):
        """Vypočítá frekvence chyb ložisek."""
        bearing_params = self.get_current_bearing_params()

        try:
            result = calculate_bearing_fault_frequencies(self.current_rpm, bearing_params)
            self.display_bearing_frequencies_result(result)

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Failed to calculate bearing frequencies:\n{str(e)}")

    def show_bearing_database(self):
        """Zobrazí databázi ložisek."""
        try:
            bearing_db = get_typical_bearing_parameters()

            dialog = QDialog(self)
            dialog.setWindowTitle("Bearing Database")
            dialog.setMinimumSize(600, 400)

            layout = QVBoxLayout()

            table = QTableWidget()
            table.setRowCount(len(bearing_db))
            table.setColumnCount(6)
            table.setHorizontalHeaderLabels(["Designation", "Type", "Balls", "Pitch Ø", "Ball Ø", "Manufacturer"])

            for i, (designation, params) in enumerate(bearing_db.items()):
                table.setItem(i, 0, QTableWidgetItem(designation))
                table.setItem(i, 1, QTableWidgetItem(params.get('type', 'Unknown')))
                table.setItem(i, 2, QTableWidgetItem(str(params.get('balls', 'N/A'))))
                table.setItem(i, 3, QTableWidgetItem(str(params.get('pitch_diameter', 'N/A'))))
                table.setItem(i, 4, QTableWidgetItem(str(params.get('ball_diameter', 'N/A'))))
                table.setItem(i, 5, QTableWidgetItem(params.get('manufacturer', 'Unknown')))

            table.resizeColumnsToContents()
            layout.addWidget(table)

            dialog.setLayout(layout)
            dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to show bearing database:\n{str(e)}")

    def show_iso_info(self):
        """Zobrazí informace o ISO standardech."""
        try:
            from analysis.vibration.severity import get_iso10816_thresholds

            dialog = QDialog(self)
            dialog.setWindowTitle("ISO 10816 Standards Information")
            dialog.setMinimumSize(500, 300)

            layout = QVBoxLayout()

            text = QTextEdit()
            text.setReadOnly(True)

            text.append("ISO 10816 - Mechanical vibration evaluation\n")
            text.append("Machine classification and vibration limits:\n")

            for class_name in ['I', 'II', 'III', 'IV']:
                thresholds = get_iso10816_thresholds(class_name)
                text.append(f"\nClass {class_name}: {thresholds.get('description', 'N/A')}")
                text.append(f"  Zone A (Good): ≤ {thresholds['A']} mm/s")
                text.append(f"  Zone B (Satisfactory): ≤ {thresholds['B']} mm/s")
                text.append(f"  Zone C (Unsatisfactory): ≤ {thresholds['C']} mm/s")
                text.append(f"  Zone D (Unacceptable): > {thresholds['C']} mm/s")

            layout.addWidget(text)
            dialog.setLayout(layout)
            dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "ISO Info Error", f"Failed to show ISO information:\n{str(e)}")

    def export_results(self):
        """Exportuje výsledky."""
        if not self.analysis_results:
            QMessageBox.information(self, "No Results", "No results to export.")
            return

        try:
            filename = "vibration_analysis_results.txt"

            with open(filename, 'w') as f:
                f.write("MakeATrend Vibration Analysis Results\n")
                f.write("=" * 50 + "\n\n")

                for analysis_type, data in self.analysis_results.items():
                    f.write(f"{analysis_type.upper()}\n")
                    f.write("-" * len(analysis_type) + "\n")

                    result = data.get('result', {})
                    for key, value in result.items():
                        f.write(f"{key}: {value}\n")

                    f.write("\n")

            QMessageBox.information(self, "Export Complete", f"Results exported to {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")


# Funkce pro testování
def main():
    """Hlavní funkce pro spuštění okna."""
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    window = VibrationWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()