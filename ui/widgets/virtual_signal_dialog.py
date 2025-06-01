"""
Virtual Signal Dialog UI

This module provides the dialog for creating virtual signals using expressions with aliases
in the CSV Signal Viewer application.
"""
import re

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QDialogButtonBox, QComboBox, QMessageBox, QPushButton,
    QGridLayout, QCheckBox, QGroupBox, QWidget
)
from PySide6.QtCore import Qt

from utils.logger import Logger
from utils.expression_validator import validate_expression, validate_signal_name

class VirtualSignalDialog(QDialog):
    """
    Dialog allowing the user to define a new virtual signal using an expression.

    Allows creating expressions like: A - B or (A + B + C) / 3,
    where the user assigns real signals to each alias.
    Implements expression validation to prevent dangerous code.

    Attributes:
        signal_names (list): List of available signal names.
        alias_mapping (dict): Mapping of alias -> QComboBox with signal selection.
    """

    def __init__(self, signal_names, parent=None):
        """
        Initializes the dialog for creating a virtual signal.

        Args:
            signal_names (list): List of available signal names.
            parent (QWidget, optional): Parent widget.
        """
        Logger.log_message_static("Widget-VirtualSignal: Initializing VirtualSignalDialog", Logger.DEBUG)
        super().__init__(parent)
        self.setWindowTitle("Create Virtual Signal")
        self.setMinimumWidth(500)
        self.signal_names = signal_names
        Logger.log_message_static(f"Widget-VirtualSignal: Available signals: {len(signal_names)}", Logger.DEBUG)
        self.alias_mapping = {}

        # Store the dialog result for later access
        self._result = None

        # Default aliases for examples
        self._example_alias1 = "A" if signal_names else ""
        self._example_alias2 = "B" if len(signal_names) > 1 else ""
        Logger.log_message_static(f"Widget-VirtualSignal: Using example aliases: {self._example_alias1}, {self._example_alias2}", Logger.DEBUG)

        self.bit_mode = False
        self.bit_checkboxes = []
        self.bit_aliases = []

        self.init_ui()
        Logger.log_message_static("Widget-VirtualSignal: VirtualSignalDialog initialization complete", Logger.DEBUG)

    def init_ui(self):
        """
        Creates the user interface for the dialog.
        """
        Logger.log_message_static("Widget-VirtualSignal: Setting up Virtual Signal Dialog UI", Logger.DEBUG)
        layout = QVBoxLayout(self)

        # Name of the new signal with unit
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("New signal name:"))
        self.name_edit = QLineEdit()
        name_layout.addWidget(self.name_edit)

        # Add unit field
        name_layout.addWidget(QLabel("Unit:"))
        self.unit_edit = QLineEdit()
        self.unit_edit.setPlaceholderText("optional")
        self.unit_edit.setMaximumWidth(55)
        name_layout.addWidget(self.unit_edit)

        layout.addLayout(name_layout)

        # Expression
        expr_layout = QHBoxLayout()
        expr_layout.addWidget(QLabel("Expression with aliases:"))
        self.expr_edit = QLineEdit()

        # Example expressions with math functions
        self.expr_edit.setPlaceholderText(f"sin({self._example_alias1}) + {self._example_alias2} * 2")
        expr_layout.addWidget(self.expr_edit)
        layout.addLayout(expr_layout)

        # Add a note about available math functions and constants
        # math_note = QLabel("Available math functions: sin, cos, tan, sqrt, abs, log, exp | Constants: pi, e")
        # math_note.setStyleSheet("color: gray;")
        # layout.addWidget(math_note)

        # Validation button
        validate_btn = QPushButton("Check Expression")
        validate_btn.clicked.connect(self._validate_current_expression)
        layout.addWidget(validate_btn)

        # Container for bit decomposition UI
        self.bit_container = QGroupBox("Bit Decomposition")
        self.bit_container.setVisible(False)
        bit_layout = QVBoxLayout(self.bit_container)

        # Signal selection for bit decomposition
        bit_signal_layout = QHBoxLayout()
        bit_signal_layout.addWidget(QLabel("Select signal to decompose:"))
        self.bit_signal_combo = QComboBox()
        self.bit_signal_combo.addItems(self.signal_names)
        bit_signal_layout.addWidget(self.bit_signal_combo)
        bit_layout.addLayout(bit_signal_layout)

        # Bit selection grid
        bit_grid = QGridLayout()
        self.bit_checkboxes = []
        self.bit_aliases = []

        # Create two columns of bits (0-7 and 8-15)
        for i in range(16):
            row = i % 8
            col = 0 if i < 8 else 2  # Column 0 and 2 for checkboxes

            checkbox = QCheckBox(f"Bit {i}")
            checkbox.setChecked(True)
            self.bit_checkboxes.append(checkbox)
            bit_grid.addWidget(checkbox, row, col)

            # Add alias input next to each checkbox
            alias_input = QLineEdit(f"B{i}")
            self.bit_aliases.append(alias_input)
            bit_grid.addWidget(alias_input, row, col+1)

        bit_layout.addLayout(bit_grid)
        layout.addWidget(self.bit_container)

        # Create a widget container for the alias area
        self.alias_label = QLabel("Assign aliases to real signals:")
        layout.addWidget(self.alias_label)

        # Use a widget to contain the alias area
        self.alias_container = QWidget()
        self.alias_area = QVBoxLayout(self.alias_container)
        layout.addWidget(self.alias_container)

        # Update the alias area when text changes
        self.expr_edit.textChanged.connect(self._handle_expression_change)

        # Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        Logger.log_message_static("Widget-VirtualSignal: Virtual Signal Dialog UI setup complete", Logger.DEBUG)

    def _handle_expression_change(self):
        """
        Handles expression changes and switches between normal and bit decomposition modes.
        """
        expression = self.expr_edit.text().strip()

        # Check if we're in bit decomposition mode
        if expression.lower() == "bits":
            self.bit_mode = True
            self.bit_container.setVisible(True)
            self.alias_label.setVisible(False)
            self.alias_container.setVisible(False)
            Logger.log_message_static("Widget-VirtualSignal: Switched to bit decomposition mode", Logger.DEBUG)
        else:
            self.bit_mode = False
            self.bit_container.setVisible(False)
            self.alias_label.setVisible(True)
            self.alias_container.setVisible(True)
            self._update_alias_inputs()
            Logger.log_message_static("Widget-VirtualSignal: Switched to normal expression mode", Logger.DEBUG)

    def get_result(self):
        """
        Returns the signal name, expression, and mapping of aliases to real signal names.
        For bit decomposition mode, returns a list of individual bit signal definitions.
        For normal mode, returns a single tuple.

        Returns:
            tuple or list:
                Normal mode: (signal_name, unit, expression, alias_mapping)
                Bit mode: [(bit_signal_name, unit, bit_expression, bit_mapping), ...]
        """
        base_name = self.name_edit.text().strip()
        unit = self.unit_edit.text().strip()
        expr = self.expr_edit.text().strip()

        if self.bit_mode:
            # For bit mode, return list of individual bit signals
            selected_bits = [(i, cb.isChecked(), self.bit_aliases[i].text())
                           for i, cb in enumerate(self.bit_checkboxes) if cb.isChecked()]

            if not selected_bits:
                return []

            bit_signals = []
            source_signal = self.bit_signal_combo.currentText()

            for bit_index, _, bit_alias in selected_bits:
                # Create individual signal name
                bit_name = f"{base_name}_{bit_alias}" if base_name else bit_alias

                # Create individual expression for this bit
                bit_expr = f"bit({source_signal}, {bit_index})"

                # Create individual mapping
                bit_mapping = {
                    "source_signal": source_signal,
                    "bit_index": bit_index,
                    "_unit": unit,
                    "_bit_mode": True
                }

                bit_signals.append((bit_name, unit, bit_expr, bit_mapping))
                Logger.log_message_static(f"Widget-VirtualSignal: Created bit signal definition: {bit_name}", Logger.DEBUG)

            Logger.log_message_static(f"Widget-VirtualSignal: Returning {len(bit_signals)} bit signal definitions", Logger.DEBUG)
            return bit_signals
        else:
            # Normal mode: return single signal
            mapping = {alias: combo.currentText() for alias, combo in self.alias_mapping.items()}
            mapping["_unit"] = unit

            Logger.log_message_static(f"Widget-VirtualSignal: Returning single signal definition: {base_name}", Logger.DEBUG)
            return (base_name, unit, expr, mapping)

    def _validate_current_expression(self):
        """
        Checks the validity of the current expression and displays the result to the user.
        """
        Logger.log_message_static("Widget-VirtualSignal: User requested expression validation", Logger.INFO)
        expression = self.expr_edit.text().strip()

        if expression.lower() == "bits":
            Logger.log_message_static("Widget-VirtualSignal: 'bits' is a special keyword for bit decomposition", Logger.DEBUG)
            QMessageBox.information(self, "Expression Validation",
                                    "Bit decomposition mode active. The expression is valid.")
            return

        Logger.log_message_static(f"Widget-VirtualSignal: Validating expression: '{expression}'", Logger.DEBUG)

        aliases = sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", expression)))
        Logger.log_message_static(f"Widget-VirtualSignal: Found aliases in expression: {aliases}", Logger.DEBUG)

        is_valid, error_msg = validate_expression(expression, aliases)

        if is_valid:
            Logger.log_message_static("Widget-VirtualSignal: Expression validation successful, showing confirmation message", Logger.DEBUG)
            QMessageBox.information(self, "Expression Validation", "The expression is syntactically correct.")
        else:
            Logger.log_message_static(f"Widget-VirtualSignal: Expression validation failed: {error_msg}, showing warning message", Logger.WARNING)
            QMessageBox.warning(self, "Expression Validation", f"Invalid expression:\n{error_msg}")

    def _update_alias_inputs(self):
        """
        Detects aliases used in the expression and creates dropdown menus for their assignment.
        """
        Logger.log_message_static("Widget-VirtualSignal: Updating alias input fields based on expression", Logger.DEBUG)

        # Remove previous alias widgets
        for i in reversed(range(self.alias_area.count())):
            item = self.alias_area.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                layout = item.layout()
                if layout:
                    # Clear this layout first
                    for j in reversed(range(layout.count())):
                        inner_item = layout.itemAt(j)
                        if inner_item.widget():
                            inner_item.widget().deleteLater()
                    # Then remove the layout
                    self.alias_area.removeItem(item)

        expression = self.expr_edit.text()
        aliases = sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", expression)))

        # Filter out math functions and constants
        math_functions = {'sin', 'cos', 'tan', 'sqrt', 'abs', 'log', 'exp'}
        constants = {'pi', 'e'}
        aliases = [a for a in aliases if a not in math_functions and a not in constants]

        Logger.log_message_static(f"Widget-VirtualSignal: Found {len(aliases)} aliases in expression: {aliases}", Logger.DEBUG)

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
            Logger.log_message_static(f"Widget-VirtualSignal: Added dropdown for alias '{alias}'", Logger.DEBUG)

    def validate_and_accept(self):
        """
        Validates the input and accepts the dialog on success.
        Checks the signal name, expression, and use of aliases.
        """
        Logger.log_message_static("Widget-VirtualSignal: Validating dialog inputs before accepting", Logger.INFO)

        # Get raw values for validation
        name = self.name_edit.text().strip()
        unit = self.unit_edit.text().strip()
        expr = self.expr_edit.text().strip()

        # Name validation
        Logger.log_message_static(f"Widget-VirtualSignal: Validating signal name: '{name}'", Logger.DEBUG)
        is_valid_name, name_error = validate_signal_name(name, self.signal_names)
        if not is_valid_name:
            Logger.log_message_static(f"Widget-VirtualSignal: Name validation failed: {name_error}", Logger.WARNING)
            QMessageBox.warning(self, "Validation Error", name_error)
            return

        # Expression validation
        if not expr:
            Logger.log_message_static("Widget-VirtualSignal: Expression is empty", Logger.WARNING)
            QMessageBox.warning(self, "Validation Error", "Expression is required.")
            return

        if self.bit_mode:
            # For bit decomposition, check if any bits are selected
            selected_bits = [i for i, cb in enumerate(self.bit_checkboxes) if cb.isChecked()]
            if not selected_bits:
                Logger.log_message_static("Widget-VirtualSignal: No bits selected for decomposition", Logger.WARNING)
                QMessageBox.warning(self, "Validation Error", "Select at least one bit to decompose.")
                return

            # Check if selected aliases are unique
            aliases = [self.bit_aliases[i].text() for i in selected_bits]
            if len(aliases) != len(set(aliases)):
                Logger.log_message_static("Widget-VirtualSignal: Duplicate bit aliases detected", Logger.WARNING)
                QMessageBox.warning(self, "Validation Error", "Bit aliases must be unique.")
                return
        else:
            # Normal expression validation
            result = self.get_result()
            if isinstance(result, tuple):
                _, _, _, mapping = result
                # Alias validations
                aliases = [a for a in mapping.keys() if not a.startswith('_')]  # Skip special keys like _unit
                if not aliases:
                    Logger.log_message_static("Widget-VirtualSignal: No aliases detected in expression", Logger.WARNING)
                    QMessageBox.warning(self, "Validation Error", "No aliases were detected in the provided expression.")
                    return

                # Expression validity check
                Logger.log_message_static(f"Widget-VirtualSignal: Validating expression: '{expr}'", Logger.DEBUG)
                is_valid_expr, expr_error = validate_expression(expr, aliases)
                if not is_valid_expr:
                    Logger.log_message_static(f"Widget-VirtualSignal: Expression validation failed: {expr_error}",
                                              Logger.WARNING)
                    QMessageBox.warning(self, "Validation Error", f"Invalid expression: {expr_errorerror}")
                    return

                # Check if each alias has an assigned signal
                empty_assignments = [alias for alias, signal_name in mapping.items()
                                     if not signal_name and not alias.startswith('_')]
                if empty_assignments:
                    Logger.log_message_static(
                        f"Widget-VirtualSignal: Missing signal assignments for aliases: {empty_assignments}",
                        Logger.WARNING)
                    QMessageBox.warning(self, "Validation Error", "Each alias must have an assigned signal.")
                    return

        # All is well, save the result
        self._result = self.get_result()
        Logger.log_message_static("Widget-VirtualSignal: Virtual signal dialog validation successful, accepting dialog",
                                  Logger.INFO)
        super().accept()