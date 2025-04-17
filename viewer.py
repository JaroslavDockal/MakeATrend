"""
Main GUI viewer for the CSV Signal Viewer application – with dual Y-axis support.
"""

from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox, QScrollArea, QSplitter, QStatusBar,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox
)
from PySide6.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
from utils import parse_csv_file, find_nearest_index
import datetime
import numpy as np


class CursorInfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cursor Information")
        self.resize(600, 400)
        self.layout = QVBoxLayout(self)

        self.header_label = QLabel("Cursor Info")
        self.layout.addWidget(self.header_label)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Signal", "A", "B", "Δ", "Δ/s"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.table)

    def update_data(self, time_a, time_b, values_a, values_b):
        self._time_a = time_a
        self._time_b = time_b
        self._values_a = values_a
        self._values_b = values_b

        self.header_label.setText(
            f"Cursor A: {time_a}    Cursor B: {time_b}    Δt: {self.calc_time_delta(time_a, time_b)}"
        )

        try:
            fmt = "%H:%M:%S.%f"
            t1 = datetime.datetime.strptime(time_a, fmt)
            t2 = datetime.datetime.strptime(time_b, fmt)
            delta_t = (t2 - t1).total_seconds()
        except:
            delta_t = None

        keys = sorted(set(values_a.keys()) | set(values_b.keys()))
        self.table.setRowCount(len(keys))

        for i, key in enumerate(keys):
            a_val = values_a.get(key, np.nan)
            b_val = values_b.get(key, np.nan)
            delta = b_val - a_val if not (np.isnan(a_val) or np.isnan(b_val)) else np.nan
            delta_per_sec = delta / delta_t if delta_t and not np.isnan(delta) else np.nan

            self.table.setItem(i, 0, QTableWidgetItem(str(key)))
            self.table.setItem(i, 1, QTableWidgetItem(f"{a_val:.3f}" if not np.isnan(a_val) else "-"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{b_val:.3f}" if not np.isnan(b_val) else "-"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{delta:.3f}" if not np.isnan(delta) else "-"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{delta_per_sec:.3f}" if not np.isnan(delta_per_sec) else "-"))

    def calc_time_delta(self, t1_str, t2_str):
        try:
            fmt = "%H:%M:%S.%f"
            t1 = datetime.datetime.strptime(t1_str, fmt)
            t2 = datetime.datetime.strptime(t2_str, fmt)
            delta = abs((t2 - t1).total_seconds())
            return f"{delta:.3f} s"
        except:
            return "-"

    def showEvent(self, event):
        self.move(self.pos())
        super().showEvent(event)


class SignalViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Signal Viewer")
        self.resize(1400, 800)

        self.data_time = None
        self.data_signals = {}
        self.curves = {}
        self.signal_axis_map = {}

        self.init_ui()

    def init_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # LEFT: Plot area with dual axes
        date_axis = DateAxisItem(orientation='bottom')
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': date_axis})
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time')
        self.main_view = self.plot_widget.getViewBox()
        splitter.addWidget(self.plot_widget)

        # RIGHT Y-AXIS (ViewBox)
        self.right_view = pg.ViewBox()
        self.plot_widget.scene().addItem(self.right_view)
        self.plot_widget.getAxis('right').linkToView(self.right_view)
        self.right_view.setXLink(self.main_view)
        self.plot_widget.showAxis('right')
        self.plot_widget.getAxis('right').setLabel('Right Y Axis')

        def sync_views():
            self.right_view.setGeometry(self.main_view.sceneBoundingRect())
            self.right_view.linkedViewChanged(self.main_view, self.right_view.XAxis)
        self.main_view.sigResized.connect(sync_views)

        # RIGHT: Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        splitter.addWidget(control_panel)

        load_btn = QPushButton("Load CSV...")
        load_btn.clicked.connect(self.load_csv)
        control_layout.addWidget(load_btn)

        self.cursor_a_chk = QCheckBox("Show Cursor A")
        self.cursor_b_chk = QCheckBox("Show Cursor B")
        self.cursor_a_chk.toggled.connect(lambda state: self.toggle_cursor(self.cursor_a, state))
        self.cursor_b_chk.toggled.connect(lambda state: self.toggle_cursor(self.cursor_b, state))
        control_layout.addWidget(self.cursor_a_chk)
        control_layout.addWidget(self.cursor_b_chk)

        control_layout.addWidget(QLabel("Signals:"))
        self.signal_scroll = QScrollArea()
        self.signal_scroll.setWidgetResizable(True)
        self.signal_scroll_contents = QWidget()
        self.signal_scroll_layout = QVBoxLayout(self.signal_scroll_contents)
        self.signal_scroll.setWidget(self.signal_scroll_contents)
        control_layout.addWidget(self.signal_scroll)

        self.signal_checkboxes = {}
        self.signal_axis_selectors = {}

        # Cursor lines
        self.cursor_a = pg.InfiniteLine(angle=90, movable=True, pen='m')
        self.cursor_b = pg.InfiniteLine(angle=90, movable=True, pen='c')
        self.cursor_a.setVisible(False)
        self.cursor_b.setVisible(False)
        self.plot_widget.addItem(self.cursor_a)
        self.plot_widget.addItem(self.cursor_b)

        self.cursor_a.sigPositionChanged.connect(self.update_cursor_info)
        self.cursor_b.sigPositionChanged.connect(self.update_cursor_info)

        # Cursor info window
        self.cursor_info = CursorInfoDialog(self)

        self.setStatusBar(QStatusBar())

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            time_array, signals = parse_csv_file(file_path)
        except Exception as e:
            print(f"Failed to load file: {e}")
            return

        self.data_time = time_array
        self.data_signals = signals
        self.clear_plot()
        self.populate_signals()

        if len(self.data_time) > 0:
            mid = self.data_time[len(self.data_time)//2]
            self.cursor_a.setPos(mid)
            self.cursor_b.setPos(mid)

        print(f"Loaded {len(signals)} signals.")

    def clear_plot(self):
        for curve in self.curves.values():
            self.plot_widget.removeItem(curve)
            self.right_view.removeItem(curve)
        self.curves.clear()
        for i in reversed(range(self.signal_scroll_layout.count())):
            self.signal_scroll_layout.itemAt(i).widget().deleteLater()
        self.signal_checkboxes.clear()
        self.signal_axis_selectors.clear()

    def populate_signals(self):
        for name in self.data_signals:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            cb = QCheckBox(name)
            cb.setChecked(False)
            cb.stateChanged.connect(self.toggle_signal)
            self.signal_checkboxes[name] = cb

            combo = QComboBox()
            combo.addItems(["Left", "Right"])
            self.signal_axis_selectors[name] = combo

            row_layout.addWidget(cb)
            row_layout.addWidget(combo)
            self.signal_scroll_layout.addWidget(row)

    def toggle_signal(self):
        cb = self.sender()
        name = cb.text()
        axis = self.signal_axis_selectors[name].currentText()

        if cb.isChecked():
            curve = pg.PlotCurveItem(x=self.data_time, y=self.data_signals[name], name=name)
            if axis == "Left":
                self.plot_widget.addItem(curve)
            else:
                self.right_view.addItem(curve)
            self.curves[name] = curve
            self.signal_axis_map[name] = axis
        else:
            curve = self.curves.pop(name, None)
            if curve:
                if self.signal_axis_map.get(name) == "Right":
                    self.right_view.removeItem(curve)
                else:
                    self.plot_widget.removeItem(curve)

    def toggle_cursor(self, cursor, show):
        cursor.setVisible(show)
        if show and self.data_time is not None:
            mid = self.data_time[len(self.data_time) // 2]
            cursor.setPos(mid)
        self.cursor_info.setVisible(self.cursor_a.isVisible() or self.cursor_b.isVisible())
        self.update_cursor_info()

    def update_cursor_info(self):
        if not self.cursor_info.isVisible():
            return

        fmt = "%H:%M:%S.%f"
        time_a = self.cursor_a.value()
        time_b = self.cursor_b.value()

        try:
            t_a = datetime.datetime.fromtimestamp(time_a).strftime(fmt)[:-3]
        except:
            t_a = "-"
        try:
            t_b = datetime.datetime.fromtimestamp(time_b).strftime(fmt)[:-3]
        except:
            t_b = "-"

        def get_values_at(t):
            idx = find_nearest_index(self.data_time, t)
            return {name: self.data_signals[name][idx] for name in self.curves}

        vals_a = get_values_at(time_a)
        vals_b = get_values_at(time_b)

        self.cursor_info.update_data(t_a, t_b, vals_a, vals_b)
