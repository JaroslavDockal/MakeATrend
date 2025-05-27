"""
Virtual Signal Expression System

This module provides tools for creating virtual signals using expressions with aliases
in a Qt-based application. It includes a dialog for user input and validation functions
to ensure expressions are safe and valid.

Example usage:
    # Get list of existing signals
    existing_signals = ["Signal1", "Signal2", "Signal3"]

    # Create and show dialog
    dialog = VirtualSignalDialog(existing_signals)
    if dialog.exec():
        result = dialog.get_result()
        if isinstance(result, list):
            # Multiple bit signals
            for name, unit, expression, mapping in result:
                # Create each bit signal
        else:
            # Single signal
            name, unit, expression, mapping = result
            # Create the virtual signal
"""
import ast
import re

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QDialogButtonBox, QComboBox, QMessageBox, QPushButton,
    QGridLayout, QCheckBox, QGroupBox, QWidget
)
from PySide6.QtCore import Qt

from utils.logger import Logger


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
            self.alias_container.setVisible(False)  # Changed from alias_area to alias_container
            Logger.log_message_static("Widget-VirtualSignal: Switched to bit decomposition mode", Logger.DEBUG)
        else:
            self.bit_mode = False
            self.bit_container.setVisible(False)
            self.alias_label.setVisible(True)
            self.alias_container.setVisible(True)  # Changed from alias_area to alias_container
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
                    QMessageBox.warning(self, "Validation Error", f"Invalid expression: {expr_error}")
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


def validate_expression(expression: str, available_aliases: list) -> tuple[bool, str]:
    """
    Validates an expression for a virtual signal for syntactic and semantic correctness.

    Checks several levels:
    1. Python syntax correctness
    2. Use of only allowed operations (mathematical operations, comparisons)
    3. Use of only known variables (aliases)
    4. Prevention of dangerous code (function calls, imports, etc.)

    Args:
        expression (str): Expression to validate (e.g., "G1 + G2 * 2")
        available_aliases (list): List of allowed variable aliases

    Returns:
        tuple[bool, str]: (is_valid, error_message)
            First value is True if the expression is valid, otherwise False.
            Second value contains an error message in case of invalidity, otherwise an empty string.
    """
    Logger.log_message_static(f"Widget-VirtualSignal: Validating expression: '{expression}' with available aliases: {available_aliases}", Logger.DEBUG)

    # Check if the expression is empty
    if not expression.strip():
        Logger.log_message_static("Widget-VirtualSignal: Expression validation failed: Expression is empty", Logger.WARNING)
        return False, "Expression is empty"

    # Check if the expression contains any aliases
    python_keywords = {'and', 'or', 'if', 'else', 'True', 'False', 'None'}
    math_functions = {'sin', 'cos', 'tan', 'sqrt', 'abs', 'log', 'exp'}
    constants = {'pi', 'e'}
    allowed_names = python_keywords.union(math_functions).union(constants)

    found_aliases = [a for a in re.findall(r"\b[A-Za-z_]\w*\b", expression)
                     if a not in allowed_names]

    Logger.log_message_static(f"Widget-VirtualSignal: Found aliases in expression: {found_aliases}", Logger.DEBUG)

    if not found_aliases:
        Logger.log_message_static("Widget-VirtualSignal: Expression validation failed: No signal aliases found in expression", Logger.WARNING)
        return False, "Expression does not contain any signal aliases"

    # Check Python syntax
    try:
        Logger.log_message_static("Widget-VirtualSignal: Parsing expression with AST to check syntax", Logger.DEBUG)
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as e:
        error_msg = f"Syntax error: {str(e)}"
        Logger.log_message_static(f"Widget-VirtualSignal: Expression validation failed: {error_msg}", Logger.WARNING)
        return False, error_msg

    class SafetyVisitor(ast.NodeVisitor):
        """
        AST node visitor that checks for potentially unsafe operations in expressions.

        Detects and reports function calls, imports, attribute access, lambda functions,
        and references to unknown variables that could lead to security issues.
        """
        def __init__(self, available_aliases):
            self.errors = []
            self.available_aliases = available_aliases
            self.python_keywords = {'and', 'or', 'if', 'else', 'True', 'False', 'None'}
            self.allowed_functions = {'sin', 'cos', 'tan', 'sqrt', 'abs', 'log', 'exp'}
            self.allowed_constants = {'pi', 'e'}

        def visit_Call(self, node):
            func_name = getattr(node.func, 'id', None)
            if func_name not in self.allowed_functions:
                error_msg = f"Function calls are not allowed: {ast.unparse(node)}"
                Logger.log_message_static(f"Widget-VirtualSignal: SafetyVisitor found disallowed call: {error_msg}", Logger.WARNING)
                self.errors.append(error_msg)
            self.generic_visit(node)

        def visit_Name(self, node):
            if (node.id not in self.available_aliases and
                node.id not in self.python_keywords and
                node.id not in self.allowed_functions and
                node.id not in self.allowed_constants):
                error_msg = f"Unknown alias: {node.id}"
                Logger.log_message_static(f"Widget-VirtualSignal: SafetyVisitor found unknown alias: {error_msg}", Logger.WARNING)
                self.errors.append(error_msg)
            self.generic_visit(node)

        def visit_Import(self, node):
            error_msg = "Import statements are not allowed"
            Logger.log_message_static(f"Widget-VirtualSignal: SafetyVisitor found disallowed import", Logger.WARNING)
            self.errors.append(error_msg)

        def visit_ImportFrom(self, node):
            error_msg = "Import statements are not allowed"
            Logger.log_message_static(f"Widget-VirtualSignal: SafetyVisitor found disallowed import", Logger.WARNING)
            self.errors.append(error_msg)

        def visit_Attribute(self, node):
            error_msg = f"Attribute access is not allowed: {ast.unparse(node)}"
            Logger.log_message_static(f"Widget-VirtualSignal: SafetyVisitor found disallowed attribute access: {error_msg}", Logger.WARNING)
            self.errors.append(error_msg)

        def visit_Lambda(self, node):
            error_msg = "Lambda functions are not allowed"
            Logger.log_message_static(f"Widget-VirtualSignal: SafetyVisitor found disallowed lambda", Logger.WARNING)
            self.errors.append(error_msg)

    # Check for dangerous constructs
    Logger.log_message_static("Widget-VirtualSignal: Running safety checks on expression AST", Logger.DEBUG)
    safety_checker = SafetyVisitor(available_aliases)
    safety_checker.visit(tree)

    if safety_checker.errors:
        error_msg = "\n".join(safety_checker.errors)
        Logger.log_message_static(f"Widget-VirtualSignal: Expression validation failed: Safety checks: {error_msg}", Logger.WARNING)
        return False, error_msg

    Logger.log_message_static("Widget-VirtualSignal: Expression validation successful", Logger.INFO)
    return True, ""

def validate_signal_name(name: str, existing_names: list) -> tuple[bool, str]:
    """
    Checks if the signal name is valid and unique.

    Args:
        name (str): Signal name to check
        existing_names (list): List of existing signal names

    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    Logger.log_message_static(f"Widget-VirtualSignal: Validating signal name: '{name}'", Logger.DEBUG)

    if not name:
        Logger.log_message_static("Widget-VirtualSignal: Signal name validation failed: Name is empty", Logger.WARNING)
        return False, "Signal name cannot be empty"

    if not re.match(r"^[A-Za-z_][A-Za-z0-9_\- ]*$", name):
        error_msg = "Signal name can only contain letters, digits, underscores, hyphens, and spaces, and must start with a letter"
        Logger.log_message_static(f"Widget-VirtualSignal: Signal name validation failed: {error_msg}", Logger.WARNING)
        return False, error_msg

    if name in existing_names:
        error_msg = f"Signal with name '{name}' already exists"
        Logger.log_message_static(f"Widget-VirtualSignal: Signal name validation failed: {error_msg}", Logger.WARNING)
        return False, error_msg

    Logger.log_message_static(f"Widget-VirtualSignal: Signal name '{name}' validation successful", Logger.DEBUG)
    return True, ""

def compute_virtual_signal(expression, alias_mapping, data_signals, unit=None):
    """
    Computes a virtual signal from an expression and signal mapping.

    Args:
        expression (str): The expression to evaluate (e.g., "A + B * 2" or "bit(Signal1, 3)")
        alias_mapping (dict): Mapping of aliases to actual signal names or bit configuration
        data_signals (dict): Dictionary of signal data as (time_array, values_array) tuples
        unit (str, optional): Unit for the virtual signal

    Returns:
        tuple: (time_array, values_array) for the computed virtual signal
    """
    Logger.log_message_static(f"Widget-VirtualSignal: Computing virtual signal from expression: '{expression}'",
                              Logger.DEBUG)

    # Check if this is a bit extraction
    if expression.startswith("bit(") and "_bit_mode" in alias_mapping:
        return compute_single_bit_extraction(alias_mapping, data_signals)

    # Special case for bit decomposition (legacy)
    if expression.lower() == "bits":
        return compute_bit_decomposition(alias_mapping, data_signals)

    Logger.log_message_static(f"Widget-VirtualSignal: Alias mapping: {alias_mapping}", Logger.DEBUG)

    # Create a namespace with the signal values
    namespace = {}

    # Basic validation
    if not alias_mapping:
        error_msg = "No signal aliases provided"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    # Filter out special keys to get actual signal mappings
    signal_mappings = {k: v for k, v in alias_mapping.items() if not k.startswith('_')}

    if not signal_mappings:
        error_msg = "No valid signal aliases found"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    # Get the time array from the first signal
    try:
        first_signal = list(signal_mappings.values())[0]
        Logger.log_message_static(f"Widget-VirtualSignal: Using '{first_signal}' as reference for time array",
                                  Logger.DEBUG)

        if first_signal not in data_signals:
            error_msg = f"Signal '{first_signal}' not found in data"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

        time_array, _ = data_signals[first_signal]
        Logger.log_message_static(f"Widget-VirtualSignal: Reference time array length: {len(time_array)}", Logger.DEBUG)
    except (IndexError, KeyError) as e:
        error_msg = f"Error accessing first signal: {str(e)}"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    # Add each signal's values to the namespace
    for alias, signal_name in signal_mappings.items():
        if signal_name not in data_signals:
            error_msg = f"Signal '{signal_name}' not found in data"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

        try:
            time_vals, signal_vals = data_signals[signal_name]
            namespace[alias] = signal_vals
            Logger.log_message_static(
                f"Widget-VirtualSignal: Added signal '{signal_name}' to namespace as '{alias}', length={len(signal_vals)}",
                Logger.DEBUG)
        except Exception as e:
            error_msg = f"Error extracting data for signal '{signal_name}': {str(e)}"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

    # Add numpy functions and constants to namespace
    safe_functions = {
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'abs': np.abs, 'sqrt': np.sqrt, 'log': np.log,
        'exp': np.exp, 'pi': np.pi, 'e': np.e
    }
    namespace.update(safe_functions)
    Logger.log_message_static(
        f"Widget-VirtualSignal: Added safe NumPy functions to namespace: {list(safe_functions.keys())}", Logger.DEBUG)

    try:
        # Safely evaluate the expression
        Logger.log_message_static(f"Widget-VirtualSignal: Evaluating expression: '{expression}'", Logger.DEBUG)
        result = eval(expression, {"__builtins__": {}}, namespace)

        # Debug output
        Logger.log_message_static(f"Widget-VirtualSignal: Expression result type: {type(result)}", Logger.DEBUG)

        # Check if result is array-like
        if result is None:
            error_msg = "Expression returned None"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

        if not hasattr(result, '__len__'):
            Logger.log_message_static(f"Widget-VirtualSignal: Converting scalar {result} to array", Logger.DEBUG)
            result = np.full_like(time_array, result)
        elif not isinstance(result, np.ndarray):
            Logger.log_message_static(f"Widget-VirtualSignal: Converting {type(result)} to array", Logger.DEBUG)
            result = np.array(result)

        # Ensure result has same length as time_array
        if len(result) != len(time_array):
            error_msg = f"Result length ({len(result)}) doesn't match time array length ({len(time_array)})"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

        Logger.log_message_static("Widget-VirtualSignal: Virtual signal computation successful", Logger.DEBUG)
        return time_array, result

    except Exception as e:
        import traceback
        error_msg = f"Failed to compute virtual signal: {str(e)}"
        Logger.log_message_static(error_msg, Logger.ERROR)
        Logger.log_message_static(f"Widget-VirtualSignal: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        raise ValueError(error_msg)

def compute_single_bit_extraction(alias_mapping, data_signals):
    """
    Extracts a single bit from a signal based on bit mode configuration.

    Args:
        alias_mapping (dict): Contains "source_signal", "bit_index" and "_bit_mode" keys
        data_signals (dict): Dictionary of signal data as (time_array, values_array) tuples

    Returns:
        tuple: (time_array, bit_values) for the extracted bit
    """
    Logger.log_message_static("Widget-VirtualSignal: Computing single bit extraction", Logger.DEBUG)

    source_signal = alias_mapping.get("source_signal")
    bit_index = alias_mapping.get("bit_index")

    if source_signal is None or bit_index is None:
        error_msg = "Missing source_signal or bit_index in bit mode configuration"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    # Get the signal data
    if source_signal not in data_signals:
        error_msg = f"Signal '{source_signal}' not found in data"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    time_array, values = data_signals[source_signal]

    # Check if the signal contains integer values for bit operations
    if not np.all(np.equal(np.mod(values, 1), 0)):
        error_msg = f"Signal '{source_signal}' contains non-integer values and cannot be used for bit extraction"
        Logger.log_message_static(error_msg, Logger.WARNING)
        raise ValueError(error_msg)

    # Convert to integers to ensure proper bit operations
    values_int = values.astype(np.int64)

    # Extract the specified bit
    bit_mask = 1 << bit_index
    bit_values = ((values_int & bit_mask) > 0).astype(bool)

    Logger.log_message_static(f"Widget-VirtualSignal: Extracted bit {bit_index} from '{source_signal}'", Logger.DEBUG)
    return time_array, bit_values

def compute_bit_decomposition(alias_mapping, data_signals):
    """
    Decomposes a signal into its individual bits.
    This function is kept for legacy compatibility but should not be used in normal operation.

    Args:
        alias_mapping (dict): Contains "signal" (the signal to decompose) and "bits" (list of bit configurations)
        data_signals (dict): Dictionary of signal data as (time_array, values_array) tuples

    Returns:
        list: List of tuples (bit_alias, time_array, bit_values) for each selected bit
    """
    Logger.log_message_static("Widget-VirtualSignal: Computing bit decomposition (legacy mode)", Logger.DEBUG)

    signal_name = alias_mapping.get("signal")
    selected_bits = alias_mapping.get("bits", [])

    if not signal_name:
        error_msg = "No signal specified for bit decomposition"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    if not selected_bits:
        error_msg = "No bits selected for decomposition"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    # Get the signal data
    if signal_name not in data_signals:
        error_msg = f"Signal '{signal_name}' not found in data"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    time_array, values = data_signals[signal_name]

    # Check if the signal is integer (no decimal part)
    if not np.all(np.equal(np.mod(values, 1), 0)):
        error_msg = f"Signal '{signal_name}' contains non-integer values and cannot be decomposed into bits"
        Logger.log_message_static(error_msg, Logger.WARNING)
        raise ValueError(error_msg)

    # Convert to integers to ensure proper bit operations
    values_int = values.astype(np.int64)

    # Extract each selected bit and create a list of results
    bit_signals = []
    for bit_config in selected_bits:
        if len(bit_config) >= 3:
            bit_index, is_selected, bit_alias = bit_config[:3]
            if is_selected:
                # Create bit mask and extract the bit
                bit_mask = 1 << bit_index
                bit_values = ((values_int & bit_mask) > 0).astype(bool)

                # Store the result as a tuple (alias, time_array, values)
                bit_signals.append((bit_alias, time_array, bit_values))
                Logger.log_message_static(f"Widget-VirtualSignal: Extracted bit {bit_index} as '{bit_alias}'",
                                          Logger.DEBUG)

    if not bit_signals:
        error_msg = "No valid bits were extracted"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    return bit_signals