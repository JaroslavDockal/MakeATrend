"""
Main GUI viewer for the CSV Signal Viewer application.
"""
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox, QScrollArea, QSplitter, QStatusBar,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView
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
        # Prevent resetting window position
        self.move(self.pos())
        super().showEvent(event)


class SignalViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Signal Viewer")
        self.resize(1400, 800)

        self.data_time = None
        self.data_signals = {}
        self.plotted_curves = {}

        self.cursor1 = pg.InfiniteLine(angle=90, movable=True, pen='m')
        self.cursor2 = pg.InfiniteLine(angle=90, movable=True, pen='c')
        self.cursor1.setVisible(False)
        self.cursor2.setVisible(False)

        self.cursor_info_window = CursorInfoDialog(self)

        self._init_ui()

    def _init_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        self.plot_widget = pg.PlotWidget(axisItems={'bottom': DateAxisItem()})
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addItem(self.cursor1)
        self.plot_widget.addItem(self.cursor2)
        splitter.addWidget(self.plot_widget)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        splitter.addWidget(control_panel)

        open_btn = QPushButton("Open CSV...")
        open_btn.clicked.connect(self.open_csv)
        control_layout.addWidget(open_btn)

        self.grid_cb = QCheckBox("Show Grid")
        self.grid_cb.setChecked(True)
        self.grid_cb.stateChanged.connect(lambda s: self.plot_widget.showGrid(x=s, y=s))
        control_layout.addWidget(self.grid_cb)

        self.cursor_a_btn = QCheckBox("Toggle Cursor A")
        self.cursor_b_btn = QCheckBox("Toggle Cursor B")
        self.cursor_a_btn.stateChanged.connect(lambda s: self.toggle_cursor(self.cursor1, s))
        self.cursor_b_btn.stateChanged.connect(lambda s: self.toggle_cursor(self.cursor2, s))
        control_layout.addWidget(self.cursor_a_btn)
        control_layout.addWidget(self.cursor_b_btn)

        control_layout.addWidget(QLabel("Signals:"))
        self.signal_scroll = QScrollArea()
        self.signal_scroll.setWidgetResizable(True)
        self.signal_widget = QWidget()
        self.signal_layout = QVBoxLayout(self.signal_widget)
        self.signal_scroll.setWidget(self.signal_widget)
        control_layout.addWidget(self.signal_scroll)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.cursor1.sigPositionChanged.connect(self.update_cursor_status)
        self.cursor2.sigPositionChanged.connect(self.update_cursor_status)

    def toggle_cursor(self, cursor, state):
        cursor.setVisible(bool(state))
        if state and self.data_time is not None:
            mid = self.data_time[len(self.data_time) // 2]
            cursor.setPos(mid)
        self.update_cursor_status()

    def open_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if not file_path:
            return

        print(f"Loading file: {file_path}")

        try:
            self.data_time, self.data_signals = parse_csv_file(file_path)
        except Exception as e:
            print(f"Failed to load file: {e}")
            return

        self.plot_widget.clear()
        self.plot_widget.addItem(self.cursor1)
        self.plot_widget.addItem(self.cursor2)
        self.plotted_curves.clear()

        for i in reversed(range(self.signal_layout.count())):
            self.signal_layout.itemAt(i).widget().deleteLater()

        for name in self.data_signals:
            cb = QCheckBox(name)
            cb.setChecked(False)
            cb.stateChanged.connect(self.update_signal_plot)
            self.signal_layout.addWidget(cb)

        print(f"Loaded {len(self.data_signals)} signals.")

    def update_signal_plot(self):
        sender = self.sender()
        name = sender.text()

        if sender.isChecked():
            curve = self.plot_widget.plot(self.data_time, self.data_signals[name], name=name)
            self.plotted_curves[name] = curve
            print(f"Plotted signal: {name}")
        else:
            if name in self.plotted_curves:
                self.plot_widget.removeItem(self.plotted_curves[name])
                del self.plotted_curves[name]
                print(f"Removed signal: {name}")

    def update_cursor_status(self):
        t1 = self.cursor1.value() if self.cursor1.isVisible() else None
        t2 = self.cursor2.value() if self.cursor2.isVisible() else None

        fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3] if t else "--"
        str1 = fmt(t1)
        str2 = fmt(t2)

        status = f"Cursor A: {str1}"
        if t2:
            status += f" | Cursor B: {str2}"
            status += f" | Δt: {abs(t2 - t1):.3f}s" if t1 else ""
        self.status.showMessage(status)
        print(status)

        if self.cursor1.isVisible() or self.cursor2.isVisible():
            values_a = self.get_values_at(t1) if t1 else {}
            values_b = self.get_values_at(t2) if t2 else {}
            self.cursor_info_window.update_data(str1, str2, values_a, values_b)
            self.cursor_info_window.show()
        else:
            self.cursor_info_window.hide()

    def get_values_at(self, timestamp):
        idx = find_nearest_index(self.data_time, timestamp)
        return {k: v[idx] for k, v in self.data_signals.items() if k in self.plotted_curves}