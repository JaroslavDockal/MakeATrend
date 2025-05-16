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
import csv
import re

from logger import Logger

class CursorInfoDialog(QDialog):
    """
    Dialog displaying signal values at cursors A and B, their difference (Δ), and Δ/s.
    Units are always shown if available (except [-] = no unit).
    Boolean signals are displayed as "On"/"Off", and Δ/Δs are hidden for them.
    """

    def __init__(self, parent=None):
        Logger.log_message_static("Initializing CursorInfoDialog", Logger.DEBUG)
        super().__init__(parent)
        self.setWindowTitle("Cursor Information")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.resize(700, 400)
        self.layout = QVBoxLayout(self)

        Logger.log_message_static("Creating cursor info dialog UI components", Logger.DEBUG)
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
        Logger.log_message_static("CursorInfoDialog initialization complete", Logger.DEBUG)

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
        Logger.log_message_static(f"Toggling to {'scientific' if self._scientific_mode else 'fixed-point'} notation", Logger.INFO)

        if self._scientific_mode:
            self.toggle_format_btn.setText("Switch to Noob Mode")
        else:
            self.toggle_format_btn.setText("Switch to Scientific Notation")

        self._rebuild_table()
        Logger.log_message_static("Table rebuilt with new number format", Logger.DEBUG)

    @staticmethod
    def extract_unit(signal_name: str) -> str:
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
            Logger.log_message_static(f"Extracted unit '{unit}' from signal '{signal_name}'", Logger.DEBUG)
            return "" if unit == "-" else unit

        Logger.log_message_static(f"No unit found in signal name '{signal_name}'", Logger.DEBUG)
        return ""

    @staticmethod
    def clean_signal_name(signal_name: str) -> str:
        """
        Removes unit part from signal name for clean display.

        Args:
            signal_name (str): Full signal name.

        Returns:
            str: Signal name without unit.
        """
        cleaned = re.sub(r"\s*\[.*?\]", "", signal_name)
        Logger.log_message_static(f"Cleaned signal name from '{signal_name}' to '{cleaned}'", Logger.DEBUG)
        return cleaned

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
        Logger.log_message_static(f"Updating cursor data: A={'active' if has_a else 'inactive'}, B={'active' if has_b else 'inactive'}", Logger.INFO)
        Logger.log_message_static(f"Cursor A time: {time_a}, Cursor B time: {time_b}", Logger.DEBUG)
        Logger.log_message_static(f"Values A contains {len(values_a)} signals, Values B contains {len(values_b)} signals", Logger.DEBUG)

        if has_a and has_b:
            delta_t = self._calc_delta_seconds(time_a, time_b)
            delta_t_str = self.calc_time_delta(time_a, time_b)
            Logger.log_message_static(f"Time delta calculated: {delta_t_str} ({delta_t} seconds)", Logger.DEBUG)
        else:
            delta_t = None
            delta_t_str = "-"
            Logger.log_message_static("Cannot calculate time delta, missing cursor", Logger.DEBUG)

        self.header_label.setText(f"Cursor A: {time_a}    Cursor B: {time_b}    Δt: {delta_t_str}")
        self._current_table_data = []

        keys = sorted(set(values_a.keys()) | set(values_b.keys()))
        Logger.log_message_static(f"Processing {len(keys)} unique signals from both cursors", Logger.DEBUG)

        for key in keys:
            a_val = values_a.get(key, "")
            b_val = values_b.get(key, "")
            unit = self.extract_unit(key)
            clean_name = self.clean_signal_name(key)
            is_bool = self._is_boolean(a_val, b_val)

            Logger.log_message_static(f"Processing signal '{clean_name}' [{unit}], boolean: {is_bool}", Logger.DEBUG)
            Logger.log_message_static(f"  A value: {a_val}, B value: {b_val}", Logger.DEBUG)

            delta = None
            dps = None

            if is_bool:
                Logger.log_message_static(f"Boolean signal '{clean_name}', delta/dps not calculated", Logger.DEBUG)
                # Booleans don't have meaningful deltas
                a_val = "On" if str(a_val).strip().upper() == "TRUE" else "Off" if a_val else ""
                b_val = "On" if str(b_val).strip().upper() == "TRUE" else "Off" if b_val else ""
            else:
                try:
                    # Calculate numerical delta if both values exist
                    if a_val and b_val and a_val != "" and b_val != "":
                        a_num = float(str(a_val).replace(',', '.'))
                        b_num = float(str(b_val).replace(',', '.'))
                        delta = abs(b_num - a_num)
                        Logger.log_message_static(f"  Delta calculated: {delta}", Logger.DEBUG)

                        # Calculate delta per second if time delta available
                        if delta_t is not None and delta_t > 0:
                            dps = delta / delta_t
                            Logger.log_message_static(f"  Delta per second: {dps}", Logger.DEBUG)
                        else:
                            Logger.log_message_static("  Cannot calculate delta per second, invalid time delta", Logger.DEBUG)
                    else:
                        Logger.log_message_static("  Cannot calculate delta, missing value at cursor A or B", Logger.DEBUG)
                except (ValueError, TypeError) as e:
                    # Handle conversion errors gracefully
                    Logger.log_message_static(f"  Error calculating delta: {str(e)}", Logger.WARNING)
                    delta = None
                    dps = None

            self._current_table_data.append({
                "name": clean_name,
                "a_val": a_val,
                "b_val": b_val,
                "delta": delta,
                "dps": dps,
                "unit": unit,
                "is_bool": is_bool
            })

        Logger.log_message_static(f"Processed {len(self._current_table_data)} signals for cursor info display", Logger.DEBUG)
        self._rebuild_table()

    @staticmethod
    def _is_boolean(a_val, b_val) -> bool:
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
            a_bool = str(a_val).strip().upper() in valid
            b_bool = str(b_val).strip().upper() in valid
            result = a_bool and b_bool
            Logger.log_message_static(f"Boolean detection: a={a_bool}, b={b_bool}, result={result}", Logger.DEBUG)
            return result
        except Exception as e:
            Logger.log_message_static(f"Error in boolean detection: {str(e)}", Logger.WARNING)
            return False

    def _format_val(self, val, unit="", is_bool=False):
        """
        Formats a value according to current display mode with appropriate units.

        Args:
            val: The value to format
            unit: The unit to append (if any)
            is_bool: Whether this is a boolean value

        Returns:
            str: Formatted value string
        """
        if val is None or str(val).strip() == "":
            Logger.log_message_static("Formatting empty/None value", Logger.DEBUG)
            return "-"

        if is_bool:
            Logger.log_message_static(f"Formatting boolean value: {val}", Logger.DEBUG)
            return val  # Boolean values are already formatted as "On"/"Off"

        try:
            num_val = float(str(val).replace(',', '.'))
            Logger.log_message_static(f"Converting value {val} to float: {num_val}", Logger.DEBUG)
        except (ValueError, TypeError) as e:
            Logger.log_message_static(f"Error converting value to float: {str(e)}", Logger.WARNING)
            return str(val)

        if self._scientific_mode:
            Logger.log_message_static(f"Formatting {num_val} in scientific notation", Logger.DEBUG)
            formatted = f"{num_val:.6e}"
        else:
            Logger.log_message_static(f"Formatting {num_val} in fixed-point notation", Logger.DEBUG)
            if abs(num_val) < 0.001 or abs(num_val) >= 10000:
                formatted = f"{num_val:.6e}"
            else:
                formatted = f"{num_val:.6f}".rstrip('0').rstrip('.')

        result = f"{formatted} {unit}" if unit else formatted
        Logger.log_message_static(f"Final formatted value: {result}", Logger.DEBUG)
        return result

    def _rebuild_table(self):
        """
        Rebuilds the table with current data and formatting settings.
        Always shows units if available (except '[-]').
        """
        Logger.log_message_static(f"Rebuilding table with {len(self._current_table_data)} rows", Logger.DEBUG)
        self.table.setRowCount(len(self._current_table_data))
        self._export_data = []

        header_row = ["Signal", "A", "B", "Δ", "Δ/s"]
        self._export_data.append(header_row)

        for i, row in enumerate(self._current_table_data):
            Logger.log_message_static(f"Formatting row {i}: {row['name']}", Logger.DEBUG)

            # Create table row
            name_item = QTableWidgetItem(row["name"])
            a_item = QTableWidgetItem(self._format_val(row["a_val"], row["unit"], row["is_bool"]))
            b_item = QTableWidgetItem(self._format_val(row["b_val"], row["unit"], row["is_bool"]))

            # Handle delta and delta per second
            if row["is_bool"]:
                Logger.log_message_static(f"Boolean signal, using '-' for delta values", Logger.DEBUG)
                delta_item = QTableWidgetItem("-")
                dps_item = QTableWidgetItem("-")
            else:
                delta_item = QTableWidgetItem(self._format_val(row["delta"], row["unit"]))
                dps_item = QTableWidgetItem(self._format_val(row["dps"], f"{row['unit']}/s" if row["unit"] else "/s"))

            # Set items in table
            self.table.setItem(i, 0, name_item)
            self.table.setItem(i, 1, a_item)
            self.table.setItem(i, 2, b_item)
            self.table.setItem(i, 3, delta_item)
            self.table.setItem(i, 4, dps_item)

            # Add row to export data
            export_row = [
                row["name"],
                a_item.text(),
                b_item.text(),
                delta_item.text(),
                dps_item.text()
            ]
            self._export_data.append(export_row)

        Logger.log_message_static("Table rebuild complete", Logger.DEBUG)

    def export_to_csv(self):
        """
        Opens a save dialog and writes the current cursor values to CSV.
        """
        Logger.log_message_static("Opening file dialog for CSV export", Logger.INFO)
        fname, _ = QFileDialog.getSaveFileName(self, "Save CSV", "cursor_values.csv", "CSV files (*.csv)")

        if fname:
            Logger.log_message_static(f"Exporting to CSV file: {fname}", Logger.INFO)
            try:
                with open(fname, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    headers = ["Signal Name", "Value A", "Value B", "Delta", "Delta/s"]
                    writer.writerow(headers)

                    for row in self._export_data[1:]:
                        writer.writerow(row)
                Logger.log_message_static(f"Successfully exported {len(self._export_data)} rows to CSV", Logger.INFO)
            except Exception as e:
                Logger.log_message_static(f"Error exporting to CSV: {str(e)}", Logger.ERROR)
        else:
            Logger.log_message_static("CSV export canceled by user", Logger.DEBUG)

    @staticmethod
    def calc_time_delta(t1_str: str, t2_str: str) -> str:
        """
        Calculates the absolute time difference between two timestamps.

        Args:
            t1_str: First timestamp string in format HH:MM:SS.mmm
            t2_str: Second timestamp string in format HH:MM:SS.mmm

        Returns:
            str: Formatted time difference as HH:MM:SS.mmm
        """
        Logger.log_message_static(f"Calculating time delta between {t1_str} and {t2_str}", Logger.DEBUG)
        try:
            fmt = "%H:%M:%S.%f"
            t1 = datetime.datetime.strptime(t1_str, fmt)
            t2 = datetime.datetime.strptime(t2_str, fmt)
            delta = abs((t2 - t1).total_seconds())

            # Format delta as time string
            hours, remainder = divmod(delta, 3600)
            minutes, seconds = divmod(remainder, 60)
            result = f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"
            Logger.log_message_static(f"Time delta calculated: {result}", Logger.DEBUG)
            return result
        except Exception as e:
            Logger.log_message_static(f"Error calculating time delta: {str(e)}", Logger.WARNING)
            return "-"

    def showEvent(self, event):
        """
        Ensures dialog window doesn't reset position on show.
        """
        Logger.log_message_static("Showing cursor info dialog", Logger.DEBUG)
        self.move(self.pos())
        super().showEvent(event)

    @staticmethod
    def _calc_delta_seconds(t1_str, t2_str) -> float:
        """
        Calculates the absolute time difference between two timestamps in seconds.

        Args:
            t1_str: First timestamp string (HH:MM:SS.mmm)
            t2_str: Second timestamp string (HH:MM:SS.mmm)

        Returns:
            float: Time difference in seconds or None if calculation fails
        """
        Logger.log_message_static(f"Calculating seconds delta between {t1_str} and {t2_str}", Logger.DEBUG)
        try:
            fmt = "%H:%M:%S.%f"
            t1 = datetime.datetime.strptime(t1_str, fmt)
            t2 = datetime.datetime.strptime(t2_str, fmt)
            delta_seconds = abs((t2 - t1).total_seconds())
            Logger.log_message_static(f"Delta seconds calculated: {delta_seconds}", Logger.DEBUG)
            return delta_seconds
        except Exception as e:
            Logger.log_message_static(f"Error calculating delta seconds: {str(e)}", Logger.WARNING)
            return None