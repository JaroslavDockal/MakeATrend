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
    Boolean signals are displayed as "On"/"Off", and Δ/Δs are hidden for them.
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

    def update_data(self, time_a: str, time_b: str, values_a: dict, values_b: dict, has_a: bool, has_b: bool):
        """
        Update the table with new cursor values.

        Args:
            time_a (str): Timestamp for cursor A.
            time_b (str): Timestamp for cursor B.
            values_a (dict): Signal values at cursor A.
            values_b (dict): Signal values at cursor B.
            has_a (bool): Whether cursor A is active.
            has_b (bool): Whether cursor B is active.
        """
        if has_a and has_b:
            delta_t = self._calc_delta_seconds(time_a, time_b)
            delta_t_str = self.calc_time_delta(time_a, time_b)
        else:
            delta_t = None
            delta_t_str = "-"

        self.header_label.setText(f"Cursor A: {time_a}    Cursor B: {time_b}    Δt: {delta_t_str}")
        self._current_table_data = []

        keys = sorted(set(values_a.keys()) | set(values_b.keys()))

        for key in keys:
            a_val = values_a.get(key, "")
            b_val = values_b.get(key, "")
            unit = self.extract_unit(key)
            clean_name = self.clean_signal_name(key)
            is_bool = self._is_boolean(a_val, b_val)

            delta = None
            dps = None

            try:
                a = float(a_val)
                b = float(b_val)
                if has_a and has_b and not is_bool:
                    delta = b - a
                    if delta_t and delta_t > 0:
                        dps = delta / delta_t
            except:
                pass

            self._current_table_data.append({
                "key": clean_name,
                "unit": unit,
                "a": a_val,
                "b": b_val,
                "delta": delta,
                "dps": dps,
                "bool": is_bool
            })

        self._rebuild_table()

    def _is_boolean(self, a_val, b_val) -> bool:
        """
        Detects if both values are textual booleans ('TRUE'/'FALSE').

        Args:
            a_val: Value at cursor A
            b_val: Value at cursor B

        Returns:
            bool: True if both are 'TRUE' or 'FALSE'
        """
        valid = {"TRUE", "FALSE"}
        try:
            return str(a_val).strip().upper() in valid and str(b_val).strip().upper() in valid
        except Exception:
            return False

    def _format_val(self, val, unit="", is_bool=False):
        if val is None or str(val).strip() == "":
            return "-"
        if is_bool:
            return "On" if str(val).strip().upper() == "TRUE" else "FALSE"
        try:
            val = float(val)
        except Exception:
            return str(val)

        if self._scientific_mode:
            formatted = f"{val:.3e}"
        else:
            abs_val = abs(val)
            if 0 < abs_val < 0.001:
                formatted = f"{val:.6f}"
            else:
                formatted = f"{val:.3f}"
        return f"{formatted} {unit}" if unit else formatted

    def _rebuild_table(self):
        """
        (Re)build the table from stored _current_table_data.
        Always shows units if available (except '[-]').
        """
        self.table.setRowCount(len(self._current_table_data))
        self._export_data = []

        for i, row in enumerate(self._current_table_data):
            unit = row["unit"]
            unit_per_s = f"{unit}/s" if unit else ""
            is_bool = row["bool"]

            self.table.setItem(i, 0, QTableWidgetItem(row["key"]))
            self.table.setItem(i, 1, QTableWidgetItem(self._format_val(row["a"], unit, is_bool)))
            self.table.setItem(i, 2, QTableWidgetItem(self._format_val(row["b"], unit, is_bool)))
            self.table.setItem(i, 3, QTableWidgetItem(
                "-" if is_bool or row["delta"] is None else self._format_val(row["delta"], unit)))
            self.table.setItem(i, 4, QTableWidgetItem(
                "-" if is_bool or row["dps"] is None else self._format_val(row["dps"], unit_per_s)))

            self._export_data.append([
                row["key"],
                self._format_val(row["a"], unit, is_bool),
                self._format_val(row["b"], unit, is_bool),
                "-" if is_bool or row["delta"] is None else self._format_val(row["delta"], unit),
                "-" if is_bool or row["dps"] is None else self._format_val(row["dps"], unit_per_s),
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
                for row in self._current_table_data:
                    writer.writerow([
                        row["key"],
                        row["a"],
                        row["b"],
                        "-" if row["bool"] else row["delta"],
                        "-" if row["bool"] else row["dps"]
                    ])

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

    def _calc_delta_seconds(self, t1_str, t2_str) -> float:
        try:
            fmt = "%H:%M:%S.%f"
            t1 = datetime.datetime.strptime(t1_str, fmt)
            t2 = datetime.datetime.strptime(t2_str, fmt)
            return abs((t2 - t1).total_seconds())
        except:
            return None