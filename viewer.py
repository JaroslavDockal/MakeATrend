"""
Main GUI implementation of the CSV Signal Viewer with custom color support.
"""
import os

import pyqtgraph as pg
import numpy as np
import datetime
from pyqtgraph import setConfigOption

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QCheckBox, QScrollArea, QSplitter, QStatusBar,
    QComboBox, QColorDialog, QSpinBox, QLineEdit, QDialog,
    QFileDialog, QGraphicsProxyWidget, QMessageBox, QTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from pyqtgraph import DateAxisItem

from utils import find_nearest_index, is_digital_signal
from cursor_info import CursorInfoDialog
from crosshair import Crosshair
from custom_viewbox import CustomViewBox
from virtual_signal_dialog import VirtualSignalDialog
from loader import load_multiple_files, load_single_file
from signal_colors import SignalColors
from signal_analysis import show_analysis_dialog
from logger import Logger


class LogWindow(QDialog):
    """
    A window to display log messages with different severity levels:
    DEBUG (0), INFO (1), WARNING (2), ERROR (3)
    """
    # Log levels
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Log Window")
        self.resize(600, 400)

        self.layout = QVBoxLayout(self)
        self.log_view = QTextEdit(self)
        self.log_view.setReadOnly(True)
        self.layout.addWidget(self.log_view)

        # Layout for checkboxes
        checkbox_layout = QHBoxLayout()

        self.debug_checkbox = QCheckBox("Show Debug", self)
        self.debug_checkbox.setChecked(False)
        self.debug_checkbox.setToolTip("Show detailed debug messages")
        checkbox_layout.addWidget(self.debug_checkbox)

        self.autoscroll_checkbox = QCheckBox("Autoscroll", self)
        self.autoscroll_checkbox.setChecked(True)
        self.autoscroll_checkbox.setToolTip("Automatically scroll to newest messages")
        checkbox_layout.addWidget(self.autoscroll_checkbox)

        self.layout.addLayout(checkbox_layout)

    def add_message(self, message, level=INFO):
        """
        Add a message to the log view.

        Args:
            message (str): The message to add.
            level (int): Message level (DEBUG=0, INFO=1, WARNING=2, ERROR=3)
        """
        # Skip debug messages if debug checkbox is not checked
        if level == LogWindow.DEBUG and not self.debug_checkbox.isChecked():
            return

        # Apply appropriate styling based on message level
        if level == LogWindow.ERROR:
            html = f'<span style="color:#ff5050;font-weight:bold;">{message}</span>'
        elif level == LogWindow.WARNING:
            html = f'<span style="color:#ffcc00;font-weight:bold;">{message}</span>'
        elif level == LogWindow.INFO:
            html = f'<span style="color:white;">{message}</span>'
        else:  # DEBUG
            html = f'<span style="color:#808080;">{message}</span>'

        self.log_view.append(html)

        if self.autoscroll_checkbox.isChecked():
            scrollbar = self.log_view.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

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

        setConfigOption('useOpenGL', False)
        setConfigOption('enableExperimental', False)

        icon_path = os.path.join(os.path.dirname(__file__), "assets", "line-graph.ico")
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

        self.log_message("Application starting", Logger.INFO)

        self.init_ui()

    def init_ui(self):
        """
        Initializes the full UI of the application, including:
        - Trend chart (pyqtgraph)
        - Right control panel with signal selection and tools
        - Floating ☰ button to reopen hidden panel
        - Cursor lines and status bar
        """
        self.log_message("Initializing main UI components", Logger.DEBUG)

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

        self.setup_floating_button()
        self.setup_axes()
        self.setup_control_panel()

    def setup_floating_button(self):
        self.show_panel_btn = QPushButton("☰")
        self.show_panel_btn.setFixedSize(30, 30)
        self.show_panel_btn.setStyleSheet("background-color: gray; color: white; font-weight: bold; border: none;")
        self.show_panel_btn.clicked.connect(lambda: self.toggle_panel_btn.setChecked(True))
        self.show_panel_btn.setToolTip("Show control panel")

        self.proxy_btn = QGraphicsProxyWidget()
        self.proxy_btn.setWidget(self.show_panel_btn)
        self.plot_widget.scene().addItem(self.proxy_btn)

        def update_button_pos():
            self.proxy_btn.setPos(self.plot_widget.width() - 40, 10)

        self.plot_widget.resizeEvent = lambda event: (
            pg.PlotWidget.resizeEvent(self.plot_widget, event), update_button_pos()
        )
        update_button_pos()

    def setup_axes(self):
        self.viewboxes = {
            'Left': self.main_view,
            'Right': pg.ViewBox(),
            'Digital': pg.ViewBox()
        }
        self.plot_widget.scene().addItem(self.viewboxes['Right'])
        self.plot_widget.scene().addItem(self.viewboxes['Digital'])
        self.viewboxes['Right'].setXLink(self.main_view)
        self.viewboxes['Digital'].setXLink(self.main_view)
        self.viewboxes['Digital'].setYRange(-0.1, 1.1, padding=0.1)

        self.signal_axis_map = {k: [] for k in self.viewboxes}
        self.axis_labels = {
            'Left': self.plot_widget.getAxis('left'),
            'Right': self.plot_widget.getAxis('right'),
            'Digital': pg.AxisItem('right')
        }

        self.plot_widget.showAxis('right')
        self.axis_labels['Right'].linkToView(self.viewboxes['Right'])
        self.plot_widget.getPlotItem().layout.addItem(self.axis_labels['Digital'], 2, 4)
        self.axis_labels['Digital'].linkToView(self.viewboxes['Digital'])

        digital_background = pg.QtWidgets.QGraphicsRectItem()
        digital_background.setPen(pg.mkPen(None))
        digital_background.setBrush(pg.mkBrush(20, 20, 20, 50))  # Subtle dark background
        self.viewboxes['Digital'].addItem(digital_background)

        def sync_views():
            geom = self.main_view.sceneBoundingRect()
            self.viewboxes['Right'].setGeometry(geom)
            self.viewboxes['Digital'].setGeometry(geom)
            self.viewboxes['Right'].linkedViewChanged(self.main_view, self.viewboxes['Right'].XAxis)
            self.viewboxes['Digital'].linkedViewChanged(self.main_view, self.viewboxes['Digital'].XAxis)

        self.main_view.sigResized.connect(sync_views)

    def setup_control_panel(self):
        layout = QVBoxLayout(self.control_panel)

        # Create a widget for buttons with a grid layout
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)

        # Create two columns
        left_column = QVBoxLayout()
        right_column = QVBoxLayout()

        load_btn = QPushButton("Load Files")
        load_btn.clicked.connect(lambda: self.load_data(multiple=True))
        load_btn.setToolTip("Open file dialog to select and load one or more files")
        left_column.addWidget(load_btn)

        self.toggle_mode_btn = QPushButton("Advanced Mode")
        self.toggle_mode_btn.setCheckable(True)
        self.toggle_mode_btn.toggled.connect(self.toggle_complex_mode)
        self.toggle_mode_btn.setToolTip("Show additional signal controls (color, width, axis selection)")
        left_column.addWidget(self.toggle_mode_btn)

        self.toggle_panel_btn = QPushButton("Hide Panel")
        self.toggle_panel_btn.clicked.connect(self.toggle_right_panel)
        self.toggle_panel_btn.setToolTip("Hide the control panel")
        left_column.addWidget(self.toggle_panel_btn)

        export_btn = QPushButton("Export Graph")
        export_btn.clicked.connect(self.export_graph)
        export_btn.setToolTip("Save current graph as image (PNG) or PDF file")
        right_column.addWidget(export_btn)

        analysis_btn = QPushButton("Signal Analysis")
        analysis_btn.clicked.connect(self.open_analysis_dialog)
        analysis_btn.setToolTip("Open signal analysis tools (FFT, statistics, etc.)")
        right_column.addWidget(analysis_btn)

        virtual_btn = QPushButton("Add Virtual Signal")
        virtual_btn.clicked.connect(self.add_virtual_signal)
        virtual_btn.setToolTip("Create calculated signals using expressions with existing signals")
        right_column.addWidget(virtual_btn)

        button_layout.addLayout(left_column)
        button_layout.addLayout(right_column)

        layout.addWidget(button_widget)

        self.setup_checkboxes(layout)
        self.setup_filter_section(layout)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll.setWidget(self.scroll_content)
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        layout.addWidget(self.scroll)

        self.cursor_a = pg.InfiniteLine(angle=90, movable=True, pen='m')
        self.cursor_b = pg.InfiniteLine(angle=90, movable=True, pen='c')
        self.cursor_a.setVisible(False)
        self.cursor_b.setVisible(False)
        self.plot_widget.addItem(self.cursor_a)
        self.plot_widget.addItem(self.cursor_b)

        self.cursor_a.sigPositionChanged.connect(self.update_cursor_info)
        self.cursor_b.sigPositionChanged.connect(self.update_cursor_info)

        self.cursor_info = CursorInfoDialog(self)
        self.crosshair = Crosshair(self.main_view)

        self.setStatusBar(QStatusBar())

    def setup_checkboxes(self, layout):
        checkbox_container = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_container)

        col1 = QVBoxLayout()
        col2 = QVBoxLayout()

        self.toggle_grid_chk = QCheckBox("Show Grid")
        self.toggle_grid_chk.setChecked(True)
        self.toggle_grid_chk.toggled.connect(self.toggle_grid)
        self.toggle_grid_chk.setToolTip("Show/hide grid lines on graph")
        col1.addWidget(self.toggle_grid_chk)

        self.cursor_a_chk = QCheckBox("Show Cursor A")
        self.cursor_a_chk.toggled.connect(lambda s: self.toggle_cursor(self.cursor_a, s))
        self.cursor_a_chk.setToolTip("Show/hide magenta vertical cursor line")
        col1.addWidget(self.cursor_a_chk)

        self.dock_cursor_info_chk = QCheckBox("Dock Cursor Info")
        self.dock_cursor_info_chk.toggled.connect(self.toggle_cursor_info_mode)
        self.dock_cursor_info_chk.setToolTip("Show cursor measurements directly in control panel")
        col1.addWidget(self.dock_cursor_info_chk)

        self.downsample_chk = QCheckBox("Downsample")
        self.downsample_chk.setChecked(False)
        self.downsample_chk.setToolTip("Reduce data points for better performance with large datasets")
        col1.addWidget(self.downsample_chk)

        self.toggle_crosshair_chk = QCheckBox("Show Crosshair")
        self.toggle_crosshair_chk.toggled.connect(self.toggle_crosshair)
        self.toggle_crosshair_chk.setToolTip("Show/hide cursor crosshair following mouse")
        col2.addWidget(self.toggle_crosshair_chk)

        self.cursor_b_chk = QCheckBox("Show Cursor B")
        self.cursor_b_chk.toggled.connect(lambda s: self.toggle_cursor(self.cursor_b, s))
        self.cursor_b_chk.setToolTip("Show/hide cyan vertical cursor line")
        col2.addWidget(self.cursor_b_chk)

        self.show_log_chk = QCheckBox("Show Log")
        self.show_log_chk.toggled.connect(self.toggle_log_window)
        self.show_log_chk.setToolTip("Show application log messages")
        col2.addWidget(self.show_log_chk)

        self.downsample_points = QSpinBox()
        self.downsample_points.setRange(100, 100000)
        self.downsample_points.setValue(5000)
        self.downsample_points.setEnabled(True)
        self.downsample_points.setToolTip("Maximum number of points to display per signal")
        col2.addWidget(self.downsample_points)

        checkbox_layout.addLayout(col1)
        checkbox_layout.addLayout(col2)
        layout.addWidget(checkbox_container)

    def setup_filter_section(self, layout):
        self.filter_box = QLineEdit()
        self.filter_box.setPlaceholderText("Filter signals...")
        self.filter_box.textChanged.connect(self.apply_signal_filter)
        self.filter_box.setToolTip("Type to filter signal list by name")
        layout.addWidget(self.filter_box)

        layout.addWidget(QLabel("Signals:"))

    def build_signal_row(self, name):
        """
        Constructs UI row for a given signal.

        Args:
            name (str): Name of the signal.

        Returns:
            QWidget: The row widget.
        """
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        cb = QCheckBox(name)
        cb.stateChanged.connect(self.toggle_signal)

        axis_cb = QComboBox()
        axis_cb.addItems(['Left', 'Right'])
        axis_cb.setVisible(False)

        # Použití SignalColors pro konzistentní barvu
        signal_color = SignalColors.get_color_for_name(name)

        color_btn = QPushButton("Color")
        color_btn.setStyleSheet(f"background-color: {signal_color}")
        color_btn.clicked.connect(lambda _, b=color_btn: self.pick_color(b))
        color_btn.setVisible(False)

        width_spin = QSpinBox()
        width_spin.setRange(1, 10)
        width_spin.setValue(2)
        width_spin.setVisible(False)

        row_layout.addWidget(cb)
        row_layout.addWidget(axis_cb)
        row_layout.addWidget(color_btn)
        row_layout.addWidget(width_spin)

        self.signal_widgets[name] = {
            'checkbox': cb,
            'axis': axis_cb,
            'color_btn': color_btn,
            'width': width_spin,
            'row': row
        }

        return row

    def clear_signals(self):
        """
        Removes all signal plots and resets widgets.
        """
        self.log_message("Clearing all signals from plot", Logger.INFO)
        for curve in self.curves.values():
            for vb in self.viewboxes.values():
                vb.removeItem(curve)

        self.curves.clear()
        self.signal_axis_map = {k: [] for k in self.viewboxes}
        self.signal_styles.clear()
        self.color_counter = 0  # Reset color counter

        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.signal_widgets.clear()
        self.update_axis_labels()

    def toggle_signal(self):
        """
        Handles toggling of a signal. Adds or removes the signal from the plot.
        Automatically assigns boolean signals to the Digital axis with step-style rendering.
        """
        cb = self.sender()
        for name, widgets in self.signal_widgets.items():
            if widgets['checkbox'] == cb:
                if cb.isChecked():
                    self.log_message(f"Displaying signal: {name}", Logger.DEBUG)
                    # Použít barvu z tlačítka, které již bylo inicializováno pomocí SignalColors
                    color = widgets['color_btn'].palette().button().color().name()
                    width = 2
                    axis = 'Left'
                    if self.complex_mode:
                        width = widgets['width'].value()
                        axis = widgets['axis'].currentText()

                    if is_digital_signal(self.data_signals[name]):
                        axis = 'Digital'
                        style = dict(stepMode=True, width=3)
                    else:
                        style = dict(width=width)

                    pen = pg.mkPen(color=color, **style)
                    time_arr, value_arr = self.data_signals[name]

                    # Downsample data before plotting
                    # Only downsample if checkbox is checked
                    if hasattr(self, 'downsample_chk') and self.downsample_chk.isChecked():
                        max_points = self.downsample_points.value()
                        time_arr, value_arr = self.downsample_signal(time_arr, value_arr, max_points)

                    curve = pg.PlotCurveItem(x=time_arr, y=value_arr, pen=pen)
                    self.viewboxes[axis].addItem(curve)
                    self.curves[name] = curve
                    self.signal_axis_map[axis].append(name)
                    self.signal_styles[name] = (axis, color, style['width'])
                else:
                    self.log_message(f"Hiding signal: {name}", Logger.DEBUG)
                    curve = self.curves.pop(name, None)
                    if curve:
                        axis = self.signal_styles[name][0]
                        self.viewboxes[axis].removeItem(curve)
                        self.signal_axis_map[axis].remove(name)
                        del self.signal_styles[name]
                self.update_axis_labels()
                break

    def update_axis_labels(self):
        """
        Updates the Y-axis labels with colored names of plotted signals in one row.
        """
        self.log_message(
            f"Updating axis labels with {len(self.signal_axis_map['Left'])} left, {len(self.signal_axis_map['Right'])} right, {len(self.signal_axis_map['Digital'])} digital signals",
            Logger.DEBUG)
        for axis, label in self.axis_labels.items():
            names = self.signal_axis_map.get(axis, [])
            html_parts = []
            for name in names:
                base_name = name.split('[')[0].strip()
                _, color, _ = self.signal_styles.get(name, (None, "#FFFFFF", 2))
                html_parts.append(f'<span style="color:{color}">{base_name}</span>')
            html_text = ", ".join(html_parts) if html_parts else axis
            label.setLabel(text=html_text)

    # Replace the current static method with an instance method
    def pick_color(self, btn):
        """
        Opens color dialog to select line color.

        Args:
            btn (QPushButton): The button to apply color to.
        """
        # Find which signal this button belongs to
        signal_name = None
        for name, widgets in self.signal_widgets.items():
            if widgets['color_btn'] == btn:
                signal_name = name
                break

        color = QColorDialog.getColor()
        if color.isValid():
            btn.setStyleSheet(f"background-color: {color.name()}")
            self.log_message(f"Changed signal color for '{signal_name}' to {color.name()}", Logger.DEBUG)

            # If signal is currently shown, update its color
            if signal_name in self.curves:
                curve = self.curves[signal_name]
                axis, _, width = self.signal_styles[signal_name]
                pen = pg.mkPen(color=color.name(), width=width)
                curve.setPen(pen)

                # Update signal style info
                self.signal_styles[signal_name] = (axis, color.name(), width)
                self.update_axis_labels()

    def toggle_complex_mode(self, state):
        """
        Enables or disables advanced plotting controls.

        Args:
            state (bool): True to enable advanced options.
        """
        self.log_message(f"Advanced mode {'enabled' if state else 'disabled'}", Logger.INFO)
        self.complex_mode = state
        for widgets in self.signal_widgets.values():
            widgets['axis'].setVisible(state)
            widgets['color_btn'].setVisible(state)
            widgets['width'].setVisible(state)

    def toggle_cursor(self, cursor, state):
        """
        Shows or hides a specific vertical cursor line.

        Args:
            cursor (pg.InfiniteLine): The cursor line.
            state (bool): Visibility flag.
        """
        self.log_message(f"{'Showing' if state else 'Hiding'} cursor {'A' if cursor == self.cursor_a else 'B'}", Logger.DEBUG)
        cursor.setVisible(state)
        if state:
            try:
                mid = None
                for name, (time_arr, _) in self.data_signals.items():
                    if time_arr is not None and len(time_arr) > 0:
                        mid = time_arr[len(time_arr) // 2]
                        break
                if mid is not None:
                    cursor.setPos(mid)
            except Exception:
                pass
        self.cursor_info.setVisible(self.cursor_a.isVisible() or self.cursor_b.isVisible())
        self.update_cursor_info()

    def update_cursor_info(self):
        """
        Refreshes the floating or docked cursor data panel.
        Now supports individual time vectors for each signal.
        """
        if not self.cursor_info.isVisible():
            return

        has_a = self.cursor_a.isVisible()
        has_b = self.cursor_b.isVisible()

        t_a = self.cursor_a.value() if has_a else None
        t_b = self.cursor_b.value() if has_b else None

        fmt = "%H:%M:%S.%f"
        try:
            s_a = datetime.datetime.fromtimestamp(t_a).strftime(fmt)[:-3] if has_a else "-"
            s_b = datetime.datetime.fromtimestamp(t_b).strftime(fmt)[:-3] if has_b else "-"
        except Exception:
            s_a, s_b = "-", "-"

        self.log_message(f"Updating cursor info: A={s_a}, B={s_b}", Logger.DEBUG)

        def get_vals(t, enabled):
            vals = {}
            if not enabled:
                return vals
            for name in self.curves:
                time_arr, value_arr = self.data_signals.get(name, (None, None))
                if time_arr is not None and value_arr is not None:
                    try:
                        idx = find_nearest_index(time_arr, t)
                        vals[name] = value_arr[idx]
                    except Exception:
                        vals[name] = np.nan
            return vals

        v_a = get_vals(t_a, has_a)
        v_b = get_vals(t_b, has_b)

        self.cursor_info.update_data(s_a, s_b, v_a, v_b, has_a, has_b)

    def toggle_right_panel(self, visible):
        """
        Show/hide the entire control panel.

        Args:
            visible (bool): Whether to show the panel.
        """
        self.log_message(f"Control panel {'shown' if visible else 'hidden'}", Logger.DEBUG)
        self.control_panel.setVisible(visible)
        self.show_panel_btn.setVisible(not visible)

    def toggle_cursor_info_mode(self, docked):
        """
        Moves cursor info to control panel or keeps it in its own window.

        Args:
            docked (bool): If True, docks the info panel.
        """
        self.log_message(f"Cursor info panel {'docked' if docked else 'undocked'}", Logger.DEBUG)

        if docked:
            for i in reversed(range(self.scroll_layout.count())):
                widget = self.scroll_layout.itemAt(i).widget()
                if widget and widget != self.cursor_info:
                    widget.setParent(None)
            self.scroll_layout.addWidget(self.cursor_info)
        else:
            self.scroll_layout.removeWidget(self.cursor_info)
            self.cursor_info.setParent(None)

            for name, widgets in self.signal_widgets.items():
                self.scroll_layout.addWidget(widgets['row'])

    def toggle_crosshair(self, state):
        """
        Toggles the crosshair visibility.

        Args:
            state (bool): True to show, False to hide.
        """
        self.log_message(f"Crosshair {'enabled' if state else 'disabled'}", Logger.DEBUG)
        self.crosshair.toggle(state)

    def toggle_grid(self, state: bool):
        """
        Enables/disables X and Y grid lines on the plot.

        Args:
            state (bool): True = show grid, False = hide.
        """
        self.log_message(f"Grid lines {'shown' if state else 'hidden'}", Logger.DEBUG)
        self.plot_widget.showGrid(x=state, y=state)

    def apply_signal_filter(self, text: str):
        """
        Filters signal rows in the panel based on user input.

        Args:
            text (str): Filter string (case-insensitive).
        """
        self.log_message(f"Filtering signals with text: '{text}'", Logger.DEBUG)
        self.signal_filter_text = text.lower().strip()
        for name, widgets in self.signal_widgets.items():
            row_widget = widgets.get('row')
            visible = self.signal_filter_text in name.lower()
            row_widget.setVisible(visible)

    def add_virtual_signal(self):
        """
        Opens a dialog for creating a new virtual signal from an expression.
        The user defines a name and an expression based on existing signals.
        If the expression is valid, the new signal is added and plotted.
        """
        self.log_message("Opening virtual signal creation dialog", Logger.DEBUG)

        signal_names = list(self.data_signals.keys())

        if not signal_names:
            QMessageBox.warning(self, "Virtual Signal", "No signals loaded. Load some signals first.")
            self.log_message("Cannot create virtual signal: No signals loaded", Logger.WARNING)
            return

        dialog = VirtualSignalDialog(signal_names, self)
        if dialog.exec():
            signal_name, expression, alias_mapping = dialog.get_result()

            if expression.strip() == "":
                self.log_message("Empty expression provided for virtual signal", Logger.WARNING)
                return

            try:
                # Use the dedicated compute_virtual_signal function from virtual_signal_dialog
                from virtual_signal_dialog import compute_virtual_signal

                # Compute the virtual signal
                self.log_message(f"Computing virtual signal '{signal_name}' with expression: {expression}", Logger.DEBUG)
                time_array, values = compute_virtual_signal(expression, alias_mapping, self.data_signals)
                self.log_message(f"Virtual signal calculation complete: {len(values)} points generated", Logger.DEBUG)

                # Add the virtual signal to the data dictionary
                self.data_signals[signal_name] = (time_array, values)

                # Create UI row for the new signal
                row = self.build_signal_row(signal_name)
                self.scroll_layout.addWidget(row)

                # Auto-select the new signal
                self.signal_widgets[signal_name]['checkbox'].setChecked(True)

                QMessageBox.information(self, "Virtual Signal",
                                        f"Virtual signal '{signal_name}' created successfully.")
                self.log_message(f"Virtual signal '{signal_name}' created successfully with expression: {expression}",
                                 INFO)
            except Exception as e:
                QMessageBox.critical(self, "Virtual Signal Error", str(e))

    def load_data(self, multiple=False):
        """
        Loads one or more data files depending on the 'multiple' flag.
        Updates data_signals as dict[name] = (time, values).
        """
        self.log_message(f"Loading {'multiple' if multiple else 'single'} data file(s)...", Logger.INFO)

        if multiple:
            signals = load_multiple_files()
        else:
            signals = load_single_file()

        if not signals:
            self.log_message("No data was loaded - user cancelled or file error", Logger.WARNING)
            return
        else:
            self.log_message(f"Successfully loaded {len(signals)} signals", Logger.INFO)

        self.data_signals = signals
        self.clear_signals()

        for name in signals:
            row = self.build_signal_row(name)
            self.scroll_layout.addWidget(row)
            self.log_message(f"Added signal: {name}", Logger.DEBUG)

    def export_graph(self):
        self.log_message("Starting graph export...", Logger.INFO)
        try:
            from pyqtgraph.exporters import ImageExporter
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Graph", "graph.png", "PNG Images (*.png);;PDF Files (*.pdf)"
            )
            if file_path:
                exporter = ImageExporter(self.plot_widget.plotItem)
                exporter.export(file_path)
                self.log_message(f"Graph exported successfully to {file_path}", Logger.INFO)
            else:
                self.log_message("Graph export cancelled by user", Logger.DEBUG)
        except ImportError:
            self.log_message("pyqtgraph.exporters not available. Cannot export graph.", Logger.ERROR)
        except Exception as e:
            self.log_message(f"Graph export failed: {str(e)}", Logger.ERROR)

    def open_analysis_dialog(self):
        self.log_message("Opening signal analysis dialog", Logger.INFO)
        show_analysis_dialog(self)

    def dragEnterEvent(self, event):
        """
        Accept drag events if they contain files.
        """
        self.log_message(f"File drag detected with {len(event.mimeData().urls())} items", Logger.DEBUG)

        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """
        Handle file drop events and load the dropped files.
        """
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        if files:
            self.log_message(
                f"Files dropped: {[os.path.basename(url.toLocalFile()) for url in event.mimeData().urls()]}", Logger.INFO)
            self.load_dropped_files(files)

    def load_dropped_files(self, files):
        """
        Load the dropped files into the application.
        """
        signals = load_multiple_files(files)
        if not signals:
            self.log_message("No valid signals found in dropped files", Logger.WARNING)
            return
        else:
            self.log_message(f"Successfully loaded {len(signals)} signals from dropped files", Logger.INFO)

        self.data_signals = signals
        self.clear_signals()

        for name in signals:
            row = self.build_signal_row(name)
            self.scroll_layout.addWidget(row)

    def downsample_signal(self, time_arr, value_arr, max_points):
        """
        Downsamples time and value arrays to have at most max_points.
        Preserves shape and important features while reducing memory usage.

        Args:
            time_arr (np.ndarray): Original time array
            value_arr (np.ndarray): Original values array
            max_points (int): Maximum number of points to keep

        Returns:
            tuple: (downsampled_time, downsampled_values)
        """
        if len(time_arr) <= max_points:
            return time_arr, value_arr

        self.log_message(f"Downsampling signal from {len(time_arr)} to ~{max_points} points", Logger.DEBUG)

        # Calculate stride for even sampling
        stride = len(time_arr) // max_points

        if stride > 2:
            self.log_message(
                f"Heavy downsampling applied - using stride of {stride} (original: {len(time_arr)}, target: {max_points})",
                Logger.WARNING)

        # Use stride-based sampling to reduce points
        ds_time = time_arr[::stride]
        ds_values = value_arr[::stride]

        # Ensure we keep the last point for proper range representation
        if len(time_arr) > 0 and ds_time[-1] != time_arr[-1]:
            ds_time = np.append(ds_time, time_arr[-1])
            ds_values = np.append(ds_values, value_arr[-1])

        self.log_message(f"Downsampled signal from {len(time_arr)} to {len(ds_time)} points", Logger.DEBUG)

        return ds_time, ds_values

    def toggle_log_window(self, state):
        """
        Show or hide the log window.

        Args:
            state (bool): True to show, False to hide.
        """
        self.log_message(f"Log window {'shown' if state else 'hidden'}", Logger.DEBUG)

        if state:
            self.log_window.show()
            # Connect log window to logger
            self.logger.set_log_window(self.log_window)
        else:
            self.log_window.hide()

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
