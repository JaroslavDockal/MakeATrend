"""
cursor_info.py

Dialog window for displaying cursor values from a plotted CSV signal.
Shows values at two cursors (A and B), their difference (Δ), and delta per second (Δ/s).
Always shows units if available (except [-] = no unit).
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QPushButton, QHeaderView, QFileDialog
)
from PySide6.QtCore import Qt
import datetime
import numpy as np
import csv
import re


class CursorInfoDialog(QDialog):
    """
    Dialog displaying signal values at cursors A and B, their difference (Δ), and Δ/s.
    Units are always shown if available (except [-] = no unit).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cursor Information")
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
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

        # === NEW: toggle between normal and scientific format ===
        self._scientific_mode = False
        self.toggle_format_btn = QPushButton("Switch to Scientific Notation")
        self.toggle_format_btn.clicked.connect(self.toggle_format_mode)
        self.layout.addWidget(self.toggle_format_btn)

        self._export_data = []
        self._current_table_data = []

    def toggle_format_mode(self):
        """
        Toggles the numerical display format between fixed-point and scientific notation.

        This affects how values are shown in the cursor info table for both absolute values
        and calculated differences (Δ, Δ/s). When toggled, the table is rebuilt with the
        new formatting applied.

        Example formats:
            - Fixed-point: 0.00012
            - Scientific: 1.200e-04
        """
        self._scientific_mode = not self._scientific_mode
        if self._scientific_mode:
            self.toggle_format_btn.setText("Switch to Noob Mode")
        else:
            self.toggle_format_btn.setText("Switch to Scientific Notation")
        self._rebuild_table()

    def extract_unit(self, signal_name: str) -> str:
        """
        Extracts unit from a signal name using brackets (e.g., "Temp [°C]").
        Returns an empty string if unit is '-' or missing.

        Args:
            signal_name (str): Full signal name.

        Returns:
            str: Extracted unit or empty string.
        """
        match = re.search(r"\[(.*?)\]", signal_name)
        if match:
            unit = match.group(1).strip()
            return "" if unit == "-" else unit
        return ""

    def clean_signal_name(self, signal_name: str) -> str:
        """
        Removes unit part from signal name for clean display.

        Args:
            signal_name (str): Full signal name.

        Returns:
            str: Signal name without unit.
        """
        return re.sub(r"\s*\[.*?\]", "", signal_name)

    def update_data(self, time_a: str, time_b: str, values_a: dict, values_b: dict):
        """
        Update the table with new cursor values.

        Args:
            time_a (str): Timestamp for cursor A.
            time_b (str): Timestamp for cursor B.
            values_a (dict): Signal values at cursor A.
            values_b (dict): Signal values at cursor B.
        """
        self.header_label.setText(
            f"Cursor A: {time_a}    Cursor B: {time_b}    Δt: {self.calc_time_delta(time_a, time_b)}"
        )

        try:
            t1 = datetime.datetime.strptime(time_a, "%H:%M:%S.%f")
            t2 = datetime.datetime.strptime(time_b, "%H:%M:%S.%f")
            delta_t = (t2 - t1).total_seconds()
        except Exception:
            delta_t = None

        keys = sorted(set(values_a.keys()) | set(values_b.keys()))
        table_data = []

        for key in keys:
            a_val = values_a.get(key, np.nan)
            b_val = values_b.get(key, np.nan)
            delta = b_val - a_val if not (np.isnan(a_val) or np.isnan(b_val)) else np.nan
            delta_per_sec = delta / delta_t if delta_t and not np.isnan(delta) else np.nan
            unit = self.extract_unit(key)

            table_data.append({
                "key": key,
                "clean_key": self.clean_signal_name(key),
                "unit": unit,
                "a": a_val,
                "b": b_val,
                "delta": delta,
                "dps": delta_per_sec
            })

        self._current_table_data = table_data
        self._rebuild_table()

    def _rebuild_table(self):
        """
        (Re)build the table from stored _current_table_data.
        Always shows units if available (except '[-]').
        """
        self.table.setRowCount(len(self._current_table_data))
        self._export_data = []

        for i, row in enumerate(self._current_table_data):
            def format_val(val, unit=""):
                if np.isnan(val):
                    return "-"
                if self._scientific_mode:
                    formatted = f"{val:.3e}"
                else:
                    abs_val = abs(val)
                    if 0 < abs_val < 0.001:
                        formatted = f"{val:.6f}"
                    else:
                        formatted = f"{val:.3f}"
                return f"{formatted} {unit}" if unit else formatted

            unit = row["unit"]
            unit_per_s = f"{unit}/s" if unit else ""

            self.table.setItem(i, 0, QTableWidgetItem(row["clean_key"]))
            self.table.setItem(i, 1, QTableWidgetItem(format_val(row["a"], unit)))
            self.table.setItem(i, 2, QTableWidgetItem(format_val(row["b"], unit)))
            self.table.setItem(i, 3, QTableWidgetItem(format_val(row["delta"], unit)))
            self.table.setItem(i, 4, QTableWidgetItem(format_val(row["dps"], unit_per_s)))

            self._export_data.append([
                row["clean_key"],
                format_val(row["a"], unit),
                format_val(row["b"], unit),
                format_val(row["delta"], unit),
                format_val(row["dps"], unit_per_s)
            ])

    def export_to_csv(self):
        """
        Opens a save dialog and writes the current cursor values to CSV.
        """
        fname, _ = QFileDialog.getSaveFileName(self, "Save CSV", "cursor_values.csv", "CSV files (*.csv)")
        if fname:
            with open(fname, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Signal", "A", "B", "Δ", "Δ/s"])
                writer.writerows(self._export_data)

    def calc_time_delta(self, t1_str: str, t2_str: str) -> str:
        """
        Calculates the absolute time difference between two timestamps.

        Args:
            t1_str (str): Time A.
            t2_str (str): Time B.

        Returns:
            str: Time delta in seconds as string.
        """
        try:
            fmt = "%H:%M:%S.%f"
            t1 = datetime.datetime.strptime(t1_str, fmt)
            t2 = datetime.datetime.strptime(t2_str, fmt)
            delta = abs((t2 - t1).total_seconds())
            return f"{delta:.3f} s"
        except:
            return "-"

    def showEvent(self, event):
        """
        Ensures dialog window doesn't reset position on show.
        """
        self.move(self.pos())
        super().showEvent(event)
