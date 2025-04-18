"""
Floating or docked panel displaying signal values under cursors.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QFileDialog
)
import datetime
import numpy as np
import csv


class CursorInfoDialog(QDialog):
    """
    Dialog window showing signal values under cursor A and B, and deltas.

    Attributes:
        _export_data (list): Data for CSV export.
    """

    def __init__(self, parent=None):
        """
        Initializes the CursorInfoDialog.

        Args:
            parent (QWidget): Optional parent widget.
        """
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
        """
        Updates the data displayed in the table.

        Args:
            time_a (str): Timestamp of cursor A.
            time_b (str): Timestamp of cursor B.
            values_a (dict): Signal values under cursor A.
            values_b (dict): Signal values under cursor B.
        """
        self.header_label.setText(
            f"Cursor A: {time_a}    Cursor B: {time_b}    Δt: {self.calc_time_delta(time_a, time_b)}"
        )

        try:
            fmt = "%H:%M:%S.%f"
            t1 = datetime.datetime.strptime(time_a, fmt)
            t2 = datetime.datetime.strptime(time_b, fmt)
            delta_t = (t2 - t1).total_seconds()
        except Exception:
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
        """
        Exports the current table data to a CSV file.
        """
        fname, _ = QFileDialog.getSaveFileName(self, "Save CSV", "cursor_values.csv", "CSV files (*.csv)")
        if fname:
            with open(fname, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Signal", "A", "B", "Δ", "Δ/s"])
                writer.writerows(self._export_data)

    def calc_time_delta(self, t1_str, t2_str):
        """
        Calculates time difference between two timestamps.

        Args:
            t1_str (str): Time A as string.
            t2_str (str): Time B as string.

        Returns:
            str: Time delta in seconds with 3 decimal places.
        """
        try:
            fmt = "%H:%M:%S.%f"
            t1 = datetime.datetime.strptime(t1_str, fmt)
            t2 = datetime.datetime.strptime(t2_str, fmt)
            delta = abs((t2 - t1).total_seconds())
            return f"{delta:.3f} s"
        except Exception:
            return "-"
