"""
Advanced GUI viewer for the CSV Signal Viewer – with multiple Y axes,
custom styles, signal export, and axis labels with active signal names.
"""

from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox, QScrollArea, QSplitter, QStatusBar,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QColorDialog, QSpinBox
)
from PySide6.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
from utils import parse_csv_file, find_nearest_index
import datetime
import numpy as np
import csv


class CursorInfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cursor Information")
        self.resize(700, 400)
        self.layout = QVBoxLayout(self)
        self.header_label = QLabel("Cursor Info")
        self.layout.addWidget(self.header_label)
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Signal", "A", "B", "Δ", "Δ/s"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.table)
        self.export_btn = QPushButton("Export to CSV")
        self.export_btn.clicked.connect(self.export_to_csv)
        self.layout.addWidget(self.export_btn)
        self._export_data = []

    def update_data(self, time_a, time_b, values_a, values_b):
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
        self._export_data = []

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

            self._export_data.append([
                key,
                f"{a_val:.3f}" if not np.isnan(a_val) else "",
                f"{b_val:.3f}" if not np.isnan(b_val) else "",
                f"{delta:.3f}" if not np.isnan(delta) else "",
                f"{delta_per_sec:.3f}" if not np.isnan(delta_per_sec) else ""
            ])

    def export_to_csv(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save CSV", "cursor_values.csv", "CSV files (*.csv)")
        if fname:
            with open(fname, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Signal", "A", "B", "Δ", "Δ/s"])
                writer.writerows(self._export_data)

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
        self.signal_styles = {}
        self.viewboxes = {}
        self.axis_labels = {}

        self.init_ui()

    def init_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # --- PLOT ---
        date_axis = DateAxisItem(orientation='bottom')
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': date_axis})
        self.plot_widget.showGrid(x=True, y=True)
        self.main_view = self.plot_widget.getViewBox()
        splitter.addWidget(self.plot_widget)

        self.viewboxes = {
            'Left': self.main_view,
            'Right': pg.ViewBox(),
            'Axis 3': pg.ViewBox(),
            'Axis 4': pg.ViewBox()
        }

        for name, vb in self.viewboxes.items():
            if name != 'Left':
                self.plot_widget.scene().addItem(vb)
                vb.setXLink(self.main_view)
            self.signal_axis_map[name] = []

        self.axis_labels = {
            'Left': self.plot_widget.getAxis('left'),
            'Right': self.plot_widget.getAxis('right'),
            'Axis 3': pg.AxisItem('left'),
            'Axis 4': pg.AxisItem('right')
        }

        self.plot_widget.showAxis('right')
        self.plot_widget.getAxis('right').linkToView(self.viewboxes['Right'])

        def sync_views():
            geom = self.main_view.sceneBoundingRect()
            for name, vb in self.viewboxes.items():
                if name != 'Left':
                    vb.setGeometry(geom)
                    vb.linkedViewChanged(self.main_view, vb.XAxis)

        self.main_view.sigResized.connect(sync_views)

        # --- CONTROL PANEL ---
        control_panel = QWidget()
        splitter.addWidget(control_panel)
        layout = QVBoxLayout(control_panel)

        load_btn = QPushButton("Load CSV...")
        load_btn.clicked.connect(self.load_csv)
        layout.addWidget(load_btn)

        self.cursor_a_chk = QCheckBox("Show Cursor A")
        self.cursor_b_chk = QCheckBox("Show Cursor B")
        self.cursor_a_chk.toggled.connect(lambda s: self.toggle_cursor(self.cursor_a, s))
        self.cursor_b_chk.toggled.connect(lambda s: self.toggle_cursor(self.cursor_b, s))
        layout.addWidget(self.cursor_a_chk)
        layout.addWidget(self.cursor_b_chk)

        layout.addWidget(QLabel("Signals:"))
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll.setWidget(self.scroll_content)
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        layout.addWidget(self.scroll)

        self.signal_widgets = {}
        self.cursor_a = pg.InfiniteLine(angle=90, movable=True, pen='m')
        self.cursor_b = pg.InfiniteLine(angle=90, movable=True, pen='c')
        self.cursor_a.setVisible(False)
        self.cursor_b.setVisible(False)
        self.plot_widget.addItem(self.cursor_a)
        self.plot_widget.addItem(self.cursor_b)

        self.cursor_a.sigPositionChanged.connect(self.update_cursor_info)
        self.cursor_b.sigPositionChanged.connect(self.update_cursor_info)

        self.cursor_info = CursorInfoDialog(self)
        self.setStatusBar(QStatusBar())

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            time_arr, signals = parse_csv_file(path)
        except Exception as e:
            print(f"Failed to load: {e}")
            return

        self.data_time = time_arr
        self.data_signals = signals
        self.clear_signals()

        for name in signals:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            cb = QCheckBox(name)
            cb.stateChanged.connect(self.toggle_signal)

            axis_cb = QComboBox()
            axis_cb.addItems(['Left', 'Right', 'Axis 3', 'Axis 4'])

            color_btn = QPushButton("Color")
            color_btn.setStyleSheet("background-color: black")
            color_btn.clicked.connect(lambda _, b=color_btn: self.pick_color(b))

            width_spin = QSpinBox()
            width_spin.setRange(1, 10)
            width_spin.setValue(2)

            row_layout.addWidget(cb)
            row_layout.addWidget(axis_cb)
            row_layout.addWidget(color_btn)
            row_layout.addWidget(width_spin)

            self.scroll_layout.addWidget(row)

            self.signal_widgets[name] = {
                'checkbox': cb,
                'axis': axis_cb,
                'color_btn': color_btn,
                'width': width_spin
            }

    def pick_color(self, btn):
        color = QColorDialog.getColor()
        if color.isValid():
            btn.setStyleSheet(f"background-color: {color.name()}")

    def toggle_signal(self):
        cb = self.sender()
        for name, widgets in self.signal_widgets.items():
            if widgets['checkbox'] == cb:
                axis = widgets['axis'].currentText()
                color = widgets['color_btn'].palette().button().color().name()
                width = widgets['width'].value()
                if cb.isChecked():
                    curve = pg.PlotCurveItem(
                        x=self.data_time,
                        y=self.data_signals[name],
                        pen=pg.mkPen(color=color, width=width)
                    )
                    self.viewboxes[axis].addItem(curve)
                    self.curves[name] = curve
                    self.signal_axis_map.setdefault(axis, []).append(name)
                    self.signal_styles[name] = (axis, color, width)
                else:
                    curve = self.curves.pop(name, None)
                    if curve:
                        self.viewboxes[self.signal_styles[name][0]].removeItem(curve)
                        self.signal_axis_map[self.signal_styles[name][0]].remove(name)
                        del self.signal_styles[name]
                self.update_axis_labels()
                break

    def update_axis_labels(self):
        for axis, label in self.axis_labels.items():
            names = self.signal_axis_map.get(axis, [])
            short_names = [n.split("[")[0] for n in names]
            label.setLabel(text=", ".join(short_names) if short_names else axis)

    def clear_signals(self):
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

    def toggle_cursor(self, cursor, state):
        cursor.setVisible(state)
        if state and self.data_time is not None:
            cursor.setPos(self.data_time[len(self.data_time) // 2])
        self.cursor_info.setVisible(self.cursor_a.isVisible() or self.cursor_b.isVisible())
        self.update_cursor_info()

    def update_cursor_info(self):
        if not self.cursor_info.isVisible():
            return
        t_a = self.cursor_a.value()
        t_b = self.cursor_b.value()
        fmt = "%H:%M:%S.%f"
        try:
            s_a = datetime.datetime.fromtimestamp(t_a).strftime(fmt)[:-3]
            s_b = datetime.datetime.fromtimestamp(t_b).strftime(fmt)[:-3]
        except:
            s_a, s_b = "-", "-"
        def get_vals(t):
            idx = find_nearest_index(self.data_time, t)
            return {k: self.data_signals[k][idx] for k in self.curves}
        v_a = get_vals(t_a)
        v_b = get_vals(t_b)
        self.cursor_info.update_data(s_a, s_b, v_a, v_b)
