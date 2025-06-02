"""
Main Signal Analysis Dialog - Contains the primary UI structure and tab management.
"""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QWidget, QHBoxLayout,
    QTabWidget, QVBoxLayout, QScrollArea, QLabel, QMessageBox
)

from utils.logger import Logger
from analysis.tabs.basic_analysis import BasicAnalysisTab
from analysis.tabs.advanced_analysis import AdvancedAnalysisTab
from analysis.tabs.cross_analysis import CrossAnalysisTab
from analysis.tabs.explanation import ExplanationTab


class SignalAnalysisDialog(QDialog):
    """
    Dialog for performing various signal analysis operations.

    This dialog provides a user interface for selecting signals and applying
    different analysis methods. It displays the results in the dialog itself or
    in separate windows for visualizations like FFT.

    Attributes:
        parent (QWidget): Parent widget, typically the main application window
        tab_widget (QTabWidget): Contains all analysis tabs
        results_layout (QVBoxLayout): Layout for displaying analysis results
    """

    def __init__(self, parent=None):
        """
        Initialize the analysis dialog with reference to parent application data.

        Args:
            parent (QWidget): Parent widget, should be the main application window
                              that contains data_signals dictionary
        """
        Logger.log_message_static("Analysis-Dialog: Initializing SignalAnalysisDialog", Logger.DEBUG)
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Signal Analysis")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowMaximizeButtonHint)
        self.resize(1000, 700)

        # Store plot windows to prevent garbage collection
        self._plot_windows = []

        Logger.log_message_static("Analysis-Dialog: Creating UI for signal analysis dialog", Logger.DEBUG)
        self.setup_ui()
        self.update_signal_lists()
        Logger.log_message_static("Analysis-Dialog: SignalAnalysisDialog initialization complete", Logger.DEBUG)

    def setup_ui(self):
        """
        Create and arrange the user interface components for the dialog.
        """
        Logger.log_message_static("Analysis-Dialog: Setting up UI components for signal analysis dialog", Logger.DEBUG)

        from PySide6.QtWidgets import QSplitter

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

        # Create and add tabs
        self.create_tabs()

        # Create results area and add to splitter
        self.create_results_area()

        # Set initial sizes of the splitter areas (e.g., 40% top, 60% bottom)
        self.splitter.setSizes([400, 600])

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        Logger.log_message_static("Analysis-Dialog: UI setup complete for signal analysis dialog", Logger.DEBUG)

    def create_tabs(self):
        """Create and add all analysis tabs to the tab widget."""
        Logger.log_message_static("Analysis-Dialog: Creating analysis tabs", Logger.DEBUG)

        # Basic Analysis Tab
        self.basic_tab = BasicAnalysisTab(self)
        self.tab_widget.addTab(self.basic_tab, "Basic Analysis")

        # Advanced Analysis Tab
        self.advanced_tab = AdvancedAnalysisTab(self)
        self.tab_widget.addTab(self.advanced_tab, "Advanced Analysis")

        # Cross Analysis Tab
        self.cross_tab = CrossAnalysisTab(self)
        self.tab_widget.addTab(self.cross_tab, "Cross Analysis")

        # Explanations Tab
        self.explanation_tab = ExplanationTab(self)
        self.tab_widget.addTab(self.explanation_tab, "Explanations")

        Logger.log_message_static("Analysis-Dialog: All tabs created successfully", Logger.DEBUG)

    def create_results_area(self):
        """Create the results display area."""
        results_widget = QWidget()

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

    def update_signal_lists(self):
        """
        Update signal lists in all tabs that have signal selectors.
        """
        Logger.log_message_static("Analysis-Dialog: Updating signal lists in all tabs", Logger.DEBUG)

        if hasattr(self.parent, 'data_signals'):
            signals = list(self.parent.data_signals.keys())
            Logger.log_message_static(f"Analysis-Dialog: Found {len(signals)} available signals", Logger.DEBUG)

            # Update signal lists in all tabs
            for i in range(self.tab_widget.count()):
                tab = self.tab_widget.widget(i)
                if hasattr(tab, 'update_signal_list'):
                    tab.update_signal_list(signals)
        else:
            Logger.log_message_static("Analysis-Dialog: No data_signals attribute found in parent", Logger.WARNING)

    def get_signal_data(self, signal_name):
        """
        Get signal data from parent application.

        Args:
            signal_name (str): Name of the signal to retrieve

        Returns:
            tuple: (time_array, values_array) or (None, None) if not found
        """
        if not signal_name:
            return None, None

        try:
            return self.parent.data_signals[signal_name]
        except (KeyError, AttributeError):
            Logger.log_message_static(f"Analysis-Dialog: Signal '{signal_name}' not found", Logger.ERROR)
            return None, None

    def clear_results(self):
        """
        Clear all widgets from the results area to prepare for new results.
        """
        Logger.log_message_static("Analysis-Dialog: Clearing results area", Logger.DEBUG)
        for i in reversed(range(self.results_layout.count())):
            widget = self.results_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

    def show_analysis_results(self, title, signal_name, data_dict):
        """
        Display analysis results in a table in the results area.

        Args:
            title (str): Title of the analysis
            signal_name (str): Name of the analyzed signal
            data_dict (dict): Dictionary containing analysis results
        """
        Logger.log_message_static(f"Analysis-Dialog: Displaying {title} results for {signal_name}", Logger.DEBUG)
        self.clear_results()

        # Create title
        result_title = QLabel(f"{title} Results: {signal_name}")
        result_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.results_layout.addWidget(result_title)

        # Create table
        table = QTableWidget()
        table.setColumnCount(2)
        table.setRowCount(len(data_dict))
        table.setHorizontalHeaderLabels(["Metric", "Value"])

        # Set column widths
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        # Populate table
        for i, (key, value) in enumerate(data_dict.items()):
            table.setItem(i, 0, QTableWidgetItem(key))
            table.setItem(i, 1, QTableWidgetItem(str(value)))

        self.results_layout.addWidget(table)
        Logger.log_message_static("Analysis-Dialog: Analysis results displayed successfully", Logger.DEBUG)

    def show_help_in_results(self, topic, content):
        """
        Display help information in the results area.

        Args:
            topic (str): Help topic title
            content (str): Help content (HTML format supported)
        """
        from PySide6.QtWidgets import QTextEdit

        Logger.log_message_static(f"Analysis-Dialog: Displaying help for: {topic}", Logger.DEBUG)
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

        Logger.log_message_static(f"Analysis-Dialog: Help content for '{topic}' displayed successfully", Logger.DEBUG)

    def add_plot_window(self, window):
        """
        Add a plot window to the tracking list to prevent garbage collection.

        Args:
            window: Plot window to track
        """
        self._plot_windows.append(window)