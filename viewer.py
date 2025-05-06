"""
Main GUI implementation of the CSV Signal Viewer.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QCheckBox, QScrollArea, QSplitter, QStatusBar,
    QComboBox, QColorDialog, QSpinBox, QFileDialog
)
from PySide6.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
from PySide6.QtWidgets import QGraphicsProxyWidget

import datetime
from utils import parse_csv_or_recorder, find_nearest_index, is_digital_signal
from cursor_info import CursorInfoDialog
from crosshair import Crosshair
import numpy as np
from custom_viewbox import CustomViewBox
from PySide6.QtWidgets import QLineEdit, QDialog
from virtual_signal_dialog import VirtualSignalDialog
import re

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

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Signal Viewer")
        self.resize(1400, 800)

        self.data_time = None
        self.data_signals = {}
        self.curves = {}
        self.signal_axis_map = {}
        self.signal_styles = {}
        self.viewboxes = {}
        self.axis_labels = {}
        self.signal_widgets = {}
        self.complex_mode = False
        self.signal_filter_text = ""

        self.init_ui()

    def init_ui(self):
        """
        Initializes the full UI of the application, including:
        - Trend chart (pyqtgraph)
        - Right control panel with signal selection and tools
        - Floating ☰ button to reopen hidden panel
        - Cursor lines and status bar
        """
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # === Plot Widget ===
        date_axis = DateAxisItem(orientation='bottom')
        self.custom_viewbox = CustomViewBox()
        self.plot_widget = pg.PlotWidget(viewBox=self.custom_viewbox, axisItems={'bottom': date_axis})
        self.plot_widget.showGrid(x=True, y=True)
        self.main_view = self.custom_viewbox
        self.main_view = self.plot_widget.getViewBox()
        splitter.addWidget(self.plot_widget)

        # === Control Panel (pravá část) ===
        self.control_panel = QWidget()
        self.control_panel.setMinimumWidth(320)
        splitter.addWidget(self.control_panel)

        # Poměr rozložení mezi grafem a panelem
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        # === Floating ☰ button ===
        self.show_panel_btn = QPushButton("☰")
        self.show_panel_btn.setFixedSize(30, 30)
        self.show_panel_btn.setStyleSheet("background-color: gray; color: white; font-weight: bold; border: none;")
        self.show_panel_btn.clicked.connect(lambda: self.toggle_panel_btn.setChecked(True))

        self.proxy_btn = QGraphicsProxyWidget()
        self.proxy_btn.setWidget(self.show_panel_btn)
        self.plot_widget.scene().addItem(self.proxy_btn)

        def update_button_pos():
            self.proxy_btn.setPos(self.plot_widget.width() - 40, 10)

        self.plot_widget.resizeEvent = lambda event: (
            pg.PlotWidget.resizeEvent(self.plot_widget, event), update_button_pos()
        )
        update_button_pos()

        # === Y-Axes setup (Left + Right + Digital) ===
        self.viewboxes = {
            'Left': self.main_view,
            'Right': pg.ViewBox(),
            'Digital': pg.ViewBox()
        }
        self.plot_widget.scene().addItem(self.viewboxes['Right'])
        self.plot_widget.scene().addItem(self.viewboxes['Digital'])
        self.viewboxes['Right'].setXLink(self.main_view)
        self.viewboxes['Digital'].setXLink(self.main_view)

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

        def sync_views():
            geom = self.main_view.sceneBoundingRect()
            self.viewboxes['Right'].setGeometry(geom)
            self.viewboxes['Digital'].setGeometry(geom)
            self.viewboxes['Right'].linkedViewChanged(self.main_view, self.viewboxes['Right'].XAxis)
            self.viewboxes['Digital'].linkedViewChanged(self.main_view, self.viewboxes['Digital'].XAxis)

        self.main_view.sigResized.connect(sync_views)

        # === Control Panel Layout ===
        layout = QVBoxLayout(self.control_panel)

        # --- Buttons ---
        load_btn = QPushButton("Load file...")
        load_btn.clicked.connect(self.load_csv)
        layout.addWidget(load_btn)

        self.toggle_mode_btn = QPushButton("Complicated Mode")
        self.toggle_mode_btn.setCheckable(True)
        self.toggle_mode_btn.toggled.connect(self.toggle_complex_mode)
        layout.addWidget(self.toggle_mode_btn)

        self.toggle_panel_btn = QPushButton("Toggle Panel")
        self.toggle_panel_btn.setCheckable(True)
        self.toggle_panel_btn.setChecked(True)
        self.toggle_panel_btn.toggled.connect(self.toggle_right_panel)
        layout.addWidget(self.toggle_panel_btn)

        self.toggle_cursor_mode = QPushButton("Dock Cursor Info")
        self.toggle_cursor_mode.setCheckable(True)
        self.toggle_cursor_mode.setChecked(False)
        self.toggle_cursor_mode.toggled.connect(self.toggle_cursor_info_mode)
        layout.addWidget(self.toggle_cursor_mode)

        # === Grid layout for checkboxes ===
        checkbox_container = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_container)

        checkbox_container = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_container)

        col1 = QVBoxLayout()
        col2 = QVBoxLayout()

        self.toggle_grid_chk = QCheckBox("Show Grid")
        self.toggle_grid_chk.setChecked(True)
        self.toggle_grid_chk.toggled.connect(self.toggle_grid)
        col1.addWidget(self.toggle_grid_chk)

        self.cursor_a_chk = QCheckBox("Show Cursor A")
        self.cursor_a_chk.toggled.connect(lambda s: self.toggle_cursor(self.cursor_a, s))
        col1.addWidget(self.cursor_a_chk)

        self.toggle_crosshair_chk = QCheckBox("Show Crosshair")
        self.toggle_crosshair_chk.toggled.connect(self.toggle_crosshair)
        col2.addWidget(self.toggle_crosshair_chk)

        self.cursor_info = CursorInfoDialog(self)  # Musí být zde před použitím
        self.crosshair = Crosshair(self.main_view)

        self.cursor_b_chk = QCheckBox("Show Cursor B")
        self.cursor_b_chk.toggled.connect(lambda s: self.toggle_cursor(self.cursor_b, s))
        col2.addWidget(self.cursor_b_chk)

        checkbox_layout.addLayout(col1)
        checkbox_layout.addLayout(col2)
        layout.addWidget(checkbox_container)

        self.filter_box = QLineEdit()
        self.filter_box.setPlaceholderText("Filter signals...")
        self.filter_box.textChanged.connect(self.apply_signal_filter)
        layout.addWidget(self.filter_box)

        btn = QPushButton("Add virtual signal")
        btn.clicked.connect(self.add_virtual_signal)
        layout.addWidget(btn)

        layout.addWidget(QLabel("Signals:"))
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll.setWidget(self.scroll_content)
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        layout.addWidget(self.scroll)

        # === Cursors and Status ===
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

    def load_csv(self):
        """
        Opens a file dialog, parses selected CSV file, and populates signals.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Open Data File", "", "Data Files (*.csv *.txt)")
        if not path:
            return

        try:
            time_arr, signals = parse_csv_or_recorder(path)
        except Exception as e:
            print(f"Failed to load CSV: {e}")
            return

        self.data_time = time_arr
        self.data_signals = signals
        self.clear_signals()

        for name in signals:
            row = self.build_signal_row(name)
            self.scroll_layout.addWidget(row)

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

        color_btn = QPushButton("Color")
        color_btn.setStyleSheet("background-color: white")
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
        for curve in self.curves.values():
            for vb in self.viewboxes.values():
                vb.removeItem(curve)

        self.curves.clear()
        self.signal_axis_map = {k: [] for k in self.viewboxes}
        self.signal_styles.clear()

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
                    if self.complex_mode:
                        color = widgets['color_btn'].palette().button().color().name()
                        width = widgets['width'].value()
                        axis = widgets['axis'].currentText()
                    else:
                        color = '#ffffff'
                        width = 2
                        axis = 'Left'

                    # === Auto-detect digital signal ===
                    if is_digital_signal(self.data_signals[name]):
                        axis = 'Digital'
                        style = dict(stepMode=True, width=3)
                    else:
                        style = dict(width=width)

                    pen = pg.mkPen(color=color, **style)
                    curve = pg.PlotCurveItem(
                        x=self.data_time,
                        y=self.data_signals[name],
                        pen=pen
                    )
                    self.viewboxes[axis].addItem(curve)
                    self.curves[name] = curve
                    self.signal_axis_map[axis].append(name)
                    self.signal_styles[name] = (axis, color, style['width'])
                else:
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
        for axis, label in self.axis_labels.items():
            names = self.signal_axis_map.get(axis, [])
            html_parts = []
            for name in names:
                base_name = name.split('[')[0].strip()
                _, color, _ = self.signal_styles.get(name, (None, "#FFFFFF", 2))
                html_parts.append(f'<span style="color:{color}">{base_name}</span>')
            html_text = ", ".join(html_parts) if html_parts else axis
            label.setLabel(text=html_text)


    def pick_color(self, btn):
        """
        Opens color dialog to select line color.

        Args:
            btn (QPushButton): The button to apply color to.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            btn.setStyleSheet(f"background-color: {color.name()}")

    def toggle_complex_mode(self, state):
        """
        Enables or disables advanced plotting controls.

        Args:
            state (bool): True to enable advanced options.
        """
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
        cursor.setVisible(state)
        if state and self.data_time is not None:
            cursor.setPos(self.data_time[len(self.data_time) // 2])
        self.cursor_info.setVisible(self.cursor_a.isVisible() or self.cursor_b.isVisible())
        self.update_cursor_info()

    def update_cursor_info(self):
        """
        Refreshes the floating or docked cursor data panel.
        """
        if not self.cursor_info.isVisible():
            return
        t_a = self.cursor_a.value()
        t_b = self.cursor_b.value()

        fmt = "%H:%M:%S.%f"
        try:
            s_a = datetime.datetime.fromtimestamp(t_a).strftime(fmt)[:-3]
            s_b = datetime.datetime.fromtimestamp(t_b).strftime(fmt)[:-3]
        except Exception:
            s_a, s_b = "-", "-"

        def get_vals(t):
            idx = find_nearest_index(self.data_time, t)
            return {k: self.data_signals[k][idx] for k in self.curves}

        v_a = get_vals(t_a)
        v_b = get_vals(t_b)
        self.cursor_info.update_data(s_a, s_b, v_a, v_b)

    def toggle_right_panel(self, visible):
        """
        Show/hide the entire control panel.

        Args:
            visible (bool): Whether to show the panel.
        """
        self.control_panel.setVisible(visible)
        self.show_panel_btn.setVisible(not visible)

    def toggle_cursor_info_mode(self, docked):
        """
        Moves cursor info to control panel or keeps it in its own window.

        Args:
            docked (bool): If True, docks the info panel.
        """
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
        self.crosshair.toggle(state)

    def toggle_grid(self, state: bool):
        """
        Enables/disables X and Y grid lines on the plot.

        Args:
            state (bool): True = show grid, False = hide.
        """
        self.plot_widget.showGrid(x=state, y=state)

    def apply_signal_filter(self, text: str):
        """
        Filters signal rows in the panel based on user input.

        Args:
            text (str): Filter string (case-insensitive).
        """
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
        dialog = VirtualSignalDialog(list(self.data_signals.keys()), parent=self)
        if dialog.exec() == QDialog.Accepted:
            name, expression, mapping = dialog.get_result()
            try:
                # Replace aliases (G1, G2...) in expression with actual signal variable names
                local_vars = {}
                for alias, real_signal in mapping.items():
                    safe_name = f"__{alias}__"
                    expression = re.sub(rf'\b{alias}\b', safe_name, expression)
                    local_vars[safe_name] = self.data_signals[real_signal]

                result = eval(expression, {}, local_vars)

                if not isinstance(result, np.ndarray):
                    raise ValueError("Expression did not return an array.")

                self.data_signals[name] = result
                row = self.build_signal_row(name)
                self.scroll_layout.addWidget(row)
                self.signal_widgets[name]['checkbox'].setChecked(True)
            except Exception as e:
                print(f"Failed to compute virtual signal: {e}")
