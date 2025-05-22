"""
Main GUI implementation of the CSV Signal Viewer with custom color support.
"""
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QWidget, QStatusBar
)

import pyqtgraph as pg
from pyqtgraph import setConfigOption, DateAxisItem

from .ui_components.control_panel import setup_control_panel
from .ui_components.plot_area import setup_plot_area, setup_axes
from utils.custom_viewbox import CustomViewBox
from utils.logger import Logger
from .ui_components.log_window import LogWindow
from utils.signal_colors import SignalColors

class SignalViewer(QMainWindow):
    """
    Main application window for signal plotting and interaction.

    Attributes:
        data_time (np.ndarray): Time axis values.
        data_signals (dict): Dictionary of signal arrays.
        curves (dict): Active plotted curves.
        signal_axis_map (dict): Mapping of axis -> list of signal names.
        signal_styles (dict): Mapping of signal name -> (axis, color, width).
        viewboxes (dict): Mapping of axis -> ViewBox.
        axis_labels (dict): Mapping of axis -> AxisItem.
        signal_widgets (dict): Mapping of signal name -> UI widgets.
        complex_mode (bool): Whether advanced UI controls are shown.
    """

    # Class-level constants for log levels
    DEBUG = Logger.DEBUG
    INFO = Logger.INFO
    WARNING = Logger.WARNING
    ERROR = Logger.ERROR

    # Add the instance variable
    instance = None

    def __init__(self):
        super().__init__()

        SignalViewer.instance = self
        self.logger = Logger.get_instance()
        SignalColors.initialize()

        setConfigOption('useOpenGL', False)
        setConfigOption('enableExperimental', False)

        icon_path = os.path.join(os.path.dirname(__file__), "_assets", "line-graph.ico")
        self.setWindowIcon(QIcon(icon_path))

        self.setWindowTitle("CSV Signal Viewer")
        self.resize(1400, 800)

        self.log_window = LogWindow()

        # Enable drag-and-drop
        self.setAcceptDrops(True)

        self.data_signals = {}
        self.curves = {}
        self.signal_axis_map = {}
        self.signal_styles = {}
        self.viewboxes = {}
        self.axis_labels = {}
        self.signal_widgets = {}
        self.complex_mode = False
        self.signal_filter_text = ""
        self.color_counter = 0

        self.log_message("Viewer: Application starting", Logger.INFO)

        self.init_ui()

    def init_ui(self):
        """
        Initializes the full UI of the application, including:
        - Trend chart (pyqtgraph)
        - Right control panel with signal selection and tools
        - Floating ☰ button to reopen hidden panel
        - Cursor lines and status bar
        """
        self.log_message("Viewer: Initializing main UI components", Logger.DEBUG)

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Plot Widget
        date_axis = DateAxisItem(orientation='bottom')
        self.custom_viewbox = CustomViewBox()
        self.plot_widget = pg.PlotWidget(viewBox=self.custom_viewbox, axisItems={'bottom': date_axis})
        self.plot_widget.showAxis('bottom')
        self.plot_widget.showGrid(x=True, y=True)
        self.main_view = self.custom_viewbox
        splitter.addWidget(self.plot_widget)

        # Control Panel
        self.control_panel = QWidget()
        self.control_panel.setMinimumWidth(320)
        splitter.addWidget(self.control_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        setup_plot_area(self)
        setup_axes(self)
        setup_control_panel(self)

        self.setStatusBar(QStatusBar())

    def log_message(self, message, level=Logger.INFO):
        """
        Logs a message using the Logger class.
        """
        self.logger.log_message(message, level)

    @staticmethod
    def log_message_static(message, level=Logger.INFO):
        """
        Static method to log messages globally.
        """
        Logger.log_message_static(message, level)

    # Import methods from other modules to maintain API compatibility
    from .ui_components.file_operations import (
        load_data, load_dropped_files, drag_enter_event, drop_event
    )

    from .ui_components.plot_operations import (
        toggle_signal, update_axis_labels, pick_color, clear_signals,
        build_signal_row, apply_signal_filter, downsample_signal
    )

    from .ui_components.panel_operations import (
        toggle_complex_mode, toggle_right_panel, toggle_cursor_info_mode,
        toggle_crosshair, toggle_grid, toggle_cursor, update_cursor_info,
        toggle_log_window
    )

    from .ui_components.signal_operations import (
        add_virtual_signal, export_graph_simple, export_graph_via_utils,
        open_analysis_dialog
    )