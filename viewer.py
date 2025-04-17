"""
SignalViewer is a GUI tool for visualizing time-series signals from CSV files.

Features:
- Load CSV file (first column = time, rest = signals)
- Select/deselect signals to display
- Interactive zoom, pan
- Two movable cursors with delta time & value
- Toggle grid and Y-axis scaling
"""

from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox, QScrollArea, QComboBox
)
import pyqtgraph as pg
import pandas as pd
import numpy as np
from utils import parse_csv_file, get_signal_values_at_time


class SignalViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Signal Viewer")
        self.resize(1200, 700)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Plot setup
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=False, y=False)
        self.plot_widget.addLegend()
        layout.addWidget(self.plot_widget, stretch=1)

        # Sidebar controls
        self.controls = QWidget()
        self.controls_layout = QVBoxLayout(self.controls)
        layout.addWidget(self.controls, stretch=0)

        self._init_controls()

        # Internal state
        self.data = None
        self.time = None
        self.signals = {}
        self.curves = {}

    def _init_controls(self):
        """Initialize control buttons and layout."""
        open_btn = QPushButton("Open CSV...")
        open_btn.clicked.connect(self.open_csv)
        self.controls_layout.addWidget(open_btn)

        self.grid_cb = QCheckBox("Show Grid")
        self.grid_cb.stateChanged.connect(self.toggle_grid)
        self.controls_layout.addWidget(self.grid_cb)

        self.yscale_combo = QComboBox()
        self.yscale_combo.addItems(["Auto", "Full Range"])
        self.yscale_combo.currentTextChanged.connect(self.change_yscale)
        self.controls_layout.addWidget(QLabel("Y-axis Scale"))
        self.controls_layout.addWidget(self.yscale_combo)

        self.cursor_a_btn = QPushButton("Toggle Cursor A")
        self.cursor_b_btn = QPushButton("Toggle Cursor B")
        self.cursor_a_btn.setCheckable(True)
        self.cursor_b_btn.setCheckable(True)
        self.cursor_a_btn.toggled.connect(lambda checked: self.toggle_cursor(1, checked))
        self.cursor_b_btn.toggled.connect(lambda checked: self.toggle_cursor(2, checked))
        self.controls_layout.addWidget(self.cursor_a_btn)
        self.controls_layout.addWidget(self.cursor_b_btn)

        self.controls_layout.addWidget(QLabel("Signals:"))

        self.signal_checkboxes = {}
        self.signal_scroll = QScrollArea()
        self.signal_scroll_widget = QWidget()
        self.signal_scroll_layout = QVBoxLayout(self.signal_scroll_widget)
        self.signal_scroll.setWidget(self.signal_scroll_widget)
        self.signal_scroll.setWidgetResizable(True)
        self.controls_layout.addWidget(self.signal_scroll)

        self.statusBar()

        self.cursor1 = pg.InfiniteLine(angle=90, movable=True, pen="m")
        self.cursor2 = pg.InfiniteLine(angle=90, movable=True, pen="c")
        self.cursor1.setVisible(False)
        self.cursor2.setVisible(False)
        self.plot_widget.addItem(self.cursor1)
        self.plot_widget.addItem(self.cursor2)

        self.cursor1.sigPositionChanged.connect(self.update_cursors)
        self.cursor2.sigPositionChanged.connect(self.update_cursors)

    def open_csv(self):
        """Open a CSV file and load signals."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if not file_path:
            return

        self.time, self.signals = parse_csv_file(file_path)
        self.plot_widget.clear()
        self.plot_widget.addLegend()
        self.curves = {}
        self._refresh_signal_list()

    def _refresh_signal_list(self):
        """Display checkboxes for all available signals."""
        # Clear old
        for i in reversed(range(self.signal_scroll_layout.count())):
            self.signal_scroll_layout.itemAt(i).widget().deleteLater()

        for name in self.signals:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(self.toggle_signal)
            self.signal_scroll_layout.addWidget(cb)
            self.signal_checkboxes[name] = cb
            self._plot_signal(name)

    def _plot_signal(self, name):
        """Plot a signal line."""
        curve = self.plot_widget.plot(self.time, self.signals[name], name=name)
        self.curves[name] = curve

    def toggle_signal(self):
        """Show or hide signal based on checkbox state."""
        cb = self.sender()
        name = cb.text()
        if cb.isChecked():
            if name not in self.curves:
                self._plot_signal(name)
            else:
                self.curves[name].show()
        else:
            if name in self.curves:
                self.curves[name].hide()

    def toggle_grid(self, state):
        """Toggle background grid."""
        self.plot_widget.showGrid(x=bool(state), y=bool(state))

    def change_yscale(self, mode):
        """Change Y-axis scaling mode."""
        vb = self.plot_widget.getViewBox()
        if mode == "Auto":
            vb.enableAutoRange(axis="y", enable=True)
        elif mode == "Full Range":
            ymin = min(np.min(sig) for sig in self.signals.values())
            ymax = max(np.max(sig) for sig in self.signals.values())
            vb.setYRange(ymin, ymax)

    def toggle_cursor(self, index, show):
        """Show or hide cursor line."""
        cursor = self.cursor1 if index == 1 else self.cursor2
        cursor.setVisible(show)
        if show:
            cursor.setPos(self.time[len(self.time) // 2])

    def update_cursors(self):
        """Update cursor status info."""
        if self.cursor1.isVisible():
            t1 = self.cursor1.value()
            y1 = get_signal_values_at_time(self.time, self.signals, t1)
        else:
            t1 = y1 = None

        if self.cursor2.isVisible():
            t2 = self.cursor2.value()
            y2 = get_signal_values_at_time(self.time, self.signals, t2)
        else:
            t2 = y2 = None

        status = ""
        if t1:
            status += f"Cursor A: {t1:.2f}s "
        if t2:
            status += f"| Cursor B: {t2:.2f}s "
        if t1 and t2:
            status += f"| Δt: {abs(t2 - t1):.2f}s"

        self.statusBar().showMessage(status)
