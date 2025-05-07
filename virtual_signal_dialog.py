"""
Dialog for defining virtual signals via expressions using aliases (e.g., G1 + G2).
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QDialogButtonBox, QComboBox, QMessageBox
)
from PySide6.QtCore import Qt
import re

class VirtualSignalDialog(QDialog):
    """
    Dialog allowing user to define a new virtual signal via an expression.
    Example: G1 - G2 or (G1 + G2 + G3) / 3
    User assigns real signals to each alias.
    """

    def __init__(self, signal_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Virtual Signal")
        self.setMinimumWidth(400)
        self.signal_names = signal_names
        self.alias_mapping = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Name of new signal
        layout.addWidget(QLabel("New Signal Name:"))
        self.name_edit = QLineEdit()
        layout.addWidget(self.name_edit)

        # Expression
        layout.addWidget(QLabel("Expression using aliases (e.g., G1 + G2):"))
        self.expr_edit = QLineEdit()
        layout.addWidget(self.expr_edit)

        # Area for alias assignments
        self.alias_area = QVBoxLayout()
        layout.addLayout(self.alias_area)

        # Update alias area on text change
        self.expr_edit.textChanged.connect(self._update_alias_inputs)

        # Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _update_alias_inputs(self):
        """
        Detects used aliases in expression and creates combo boxes for assignment.
        """
        # Remove previous alias input widgets
        for i in reversed(range(self.alias_area.count())):
            item = self.alias_area.itemAt(i)
            if item:
                w = item.layout()
                if w:
                    while w.count():
                        c = w.takeAt(0).widget()
                        if c:
                            c.setParent(None)

        expression = self.expr_edit.text()
        aliases = sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", expression)))

        self.alias_mapping.clear()

        for alias in aliases:
            row = QHBoxLayout()
            label = QLabel(f"{alias} =")
            combo = QComboBox()
            combo.addItems(self.signal_names)
            row.addWidget(label)
            row.addWidget(combo)
            self.alias_area.addLayout(row)
            self.alias_mapping[alias] = combo

    def get_result(self):
        """
        Returns signal name, expression and mapping of aliases to real signal names.

        Returns:
            tuple[str, str, dict[str, str]]: (signal_name, expression, alias_mapping)
        """
        name = self.name_edit.text().strip()
        expr = self.expr_edit.text().strip()
        mapping = {alias: combo.currentText() for alias, combo in self.alias_mapping.items()}
        return name, expr, mapping

    def accept(self):
        """
        Validates input and accepts dialog.
        """
        name, expr, mapping = self.get_result()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Signal name is required.")
            return
        if not expr:
            QMessageBox.warning(self, "Validation Error", "Expression is required.")
            return
        if not mapping:
            QMessageBox.warning(self, "Validation Error", "No aliases detected.")
            return
        self._result = self.get_result()
        super().accept()
