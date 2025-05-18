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
        name, expression, mapping = dialog.get_result()
        # Create the virtual signal with the provided information
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QDialogButtonBox, QComboBox, QMessageBox, QPushButton
)
import re
import ast
import numpy as np
from logger import Logger

# ===============================
# Expression Validation Functions
# ===============================

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
    Logger.log_message_static(f"Validating expression: '{expression}' with available aliases: {available_aliases}", Logger.DEBUG)

    # Check if the expression is empty
    if not expression.strip():
        Logger.log_message_static("Expression validation failed: Expression is empty", Logger.WARNING)
        return False, "Expression is empty"

    # Check if the expression contains any aliases
    python_keywords = {'and', 'or', 'if', 'else', 'True', 'False', 'None'}
    found_aliases = [a for a in re.findall(r"\b[A-Za-z_]\w*\b", expression)
                     if a not in python_keywords]

    Logger.log_message_static(f"Found aliases in expression: {found_aliases}", Logger.DEBUG)

    if not found_aliases:
        Logger.log_message_static("Expression validation failed: No signal aliases found in expression", Logger.WARNING)
        return False, "Expression does not contain any signal aliases"

    # Check Python syntax
    try:
        Logger.log_message_static("Parsing expression with AST to check syntax", Logger.DEBUG)
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as e:
        error_msg = f"Syntax error: {str(e)}"
        Logger.log_message_static(f"Expression validation failed: {error_msg}", Logger.WARNING)
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

        def visit_Call(self, node):
            error_msg = f"Function calls are not allowed: {ast.unparse(node)}"
            Logger.log_message_static(f"SafetyVisitor found disallowed call: {error_msg}", Logger.WARNING)
            self.errors.append(error_msg)
            self.generic_visit(node)

        def visit_Name(self, node):
            if node.id not in self.available_aliases and node.id not in self.python_keywords:
                error_msg = f"Unknown alias: {node.id}"
                Logger.log_message_static(f"SafetyVisitor found unknown alias: {error_msg}", Logger.WARNING)
                self.errors.append(error_msg)
            self.generic_visit(node)

        def visit_Import(self, node):
            error_msg = "Import statements are not allowed"
            Logger.log_message_static(f"SafetyVisitor found disallowed import", Logger.WARNING)
            self.errors.append(error_msg)

        def visit_ImportFrom(self, node):
            error_msg = "Import statements are not allowed"
            Logger.log_message_static(f"SafetyVisitor found disallowed import", Logger.WARNING)
            self.errors.append(error_msg)

        def visit_Attribute(self, node):
            error_msg = f"Attribute access is not allowed: {ast.unparse(node)}"
            Logger.log_message_static(f"SafetyVisitor found disallowed attribute access: {error_msg}", Logger.WARNING)
            self.errors.append(error_msg)

        def visit_Lambda(self, node):
            error_msg = "Lambda functions are not allowed"
            Logger.log_message_static(f"SafetyVisitor found disallowed lambda", Logger.WARNING)
            self.errors.append(error_msg)

    # Check for dangerous constructs
    Logger.log_message_static("Running safety checks on expression AST", Logger.DEBUG)
    safety_checker = SafetyVisitor(available_aliases)
    safety_checker.visit(tree)

    if safety_checker.errors:
        error_msg = "\n".join(safety_checker.errors)
        Logger.log_message_static(f"Expression validation failed: Safety checks: {error_msg}", Logger.WARNING)
        return False, error_msg

    Logger.log_message_static("Expression validation successful", Logger.DEBUG)
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
    Logger.log_message_static(f"Validating signal name: '{name}'", Logger.DEBUG)

    if not name:
        Logger.log_message_static("Signal name validation failed: Name is empty", Logger.WARNING)
        return False, "Signal name cannot be empty"

    if not re.match(r"^[A-Za-z_][A-Za-z0-9_\- ]*$", name):
        error_msg = "Signal name can only contain letters, digits, underscores, hyphens, and spaces, and must start with a letter"
        Logger.log_message_static(f"Signal name validation failed: {error_msg}", Logger.WARNING)
        return False, error_msg

    if name in existing_names:
        error_msg = f"Signal with name '{name}' already exists"
        Logger.log_message_static(f"Signal name validation failed: {error_msg}", Logger.WARNING)
        return False, error_msg

    Logger.log_message_static(f"Signal name '{name}' validation successful", Logger.DEBUG)
    return True, ""


# Function to compute virtual signals (moved outside of the class)
def compute_virtual_signal(expression, alias_mapping, data_signals):
    """
    Computes a virtual signal from an expression and signal mapping.

    Args:
        expression (str): The expression to evaluate (e.g., "A + B * 2")
        alias_mapping (dict): Mapping of aliases to actual signal names
        data_signals (dict): Dictionary of signal data as (time_array, values_array) tuples

    Returns:
        tuple: (time_array, values_array) for the computed virtual signal
    """
    Logger.log_message_static(f"Computing virtual signal from expression: '{expression}'", Logger.INFO)
    Logger.log_message_static(f"Alias mapping: {alias_mapping}", Logger.DEBUG)

    # Create a namespace with the signal values
    namespace = {}

    # Basic validation
    if not alias_mapping:
        error_msg = "No signal aliases provided"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    # Get the time array from the first signal
    try:
        first_signal = list(alias_mapping.values())[0]
        Logger.log_message_static(f"Using '{first_signal}' as reference for time array", Logger.DEBUG)

        if first_signal not in data_signals:
            error_msg = f"Signal '{first_signal}' not found in data"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

        time_array, _ = data_signals[first_signal]
        Logger.log_message_static(f"Reference time array length: {len(time_array)}", Logger.DEBUG)
    except (IndexError, KeyError) as e:
        error_msg = f"Error accessing first signal: {str(e)}"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    # Add each signal's values to the namespace
    for alias, signal_name in alias_mapping.items():
        if signal_name not in data_signals:
            error_msg = f"Signal '{signal_name}' not found in data"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

        try:
            time_vals, signal_vals = data_signals[signal_name]
            namespace[alias] = signal_vals
            Logger.log_message_static(f"Added signal '{signal_name}' to namespace as '{alias}', length={len(signal_vals)}", Logger.DEBUG)
        except Exception as e:
            error_msg = f"Error extracting data for signal '{signal_name}': {str(e)}"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

    # Add numpy functions to namespace
    safe_numpy = {
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'abs': np.abs, 'sqrt': np.sqrt, 'log': np.log,
        'exp': np.exp, 'pi': np.pi
    }
    Logger.log_message_static(f"Added safe NumPy functions to namespace: {list(safe_numpy.keys())}", Logger.DEBUG)

    try:
        # Safely evaluate the expression
        Logger.log_message_static(f"Evaluating expression: '{expression}'", Logger.DEBUG)
        result = eval(expression, {"__builtins__": {}, "np": safe_numpy}, namespace)

        # Debug output
        Logger.log_message_static(f"Expression result type: {type(result)}", Logger.DEBUG)

        # Check if result is array-like
        if result is None:
            error_msg = "Expression returned None"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

        if not hasattr(result, '__len__'):
            Logger.log_message_static(f"Converting scalar {result} to array", Logger.DEBUG)
            result = np.full_like(time_array, result)
        elif not isinstance(result, np.ndarray):
            Logger.log_message_static(f"Converting {type(result)} to numpy array", Logger.DEBUG)
            result = np.array(result)

        # Ensure result has same length as time_array
        if len(result) != len(time_array):
            error_msg = f"Result length ({len(result)}) doesn't match time array length ({len(time_array)})"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

        Logger.log_message_static("Virtual signal computation successful", Logger.INFO)
        return time_array, result

    except Exception as e:
        import traceback
        error_msg = f"Failed to compute virtual signal: {str(e)}"
        Logger.log_message_static(error_msg, Logger.ERROR)
        Logger.log_message_static(f"Traceback: {traceback.format_exc()}", Logger.DEBUG)
        raise ValueError(error_msg)


# ======================
# Virtual Signal Dialog
# ======================

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
        Logger.log_message_static("Initializing VirtualSignalDialog", Logger.DEBUG)
        super().__init__(parent)
        self.setWindowTitle("Create Virtual Signal")
        self.setMinimumWidth(500)
        self.signal_names = signal_names
        Logger.log_message_static(f"Available signals: {len(signal_names)}", Logger.DEBUG)
        self.alias_mapping = {}

        # Store the dialog result for later access
        self._result = None

        # Default aliases for examples
        self._example_alias1 = "A" if signal_names else ""
        self._example_alias2 = "B" if len(signal_names) > 1 else ""
        Logger.log_message_static(f"Using example aliases: {self._example_alias1}, {self._example_alias2}", Logger.DEBUG)

        self.init_ui()
        Logger.log_message_static("VirtualSignalDialog initialization complete", Logger.DEBUG)

    def init_ui(self):
        """
        Creates the user interface for the dialog.
        """
        Logger.log_message_static("Setting up Virtual Signal Dialog UI", Logger.DEBUG)
        layout = QVBoxLayout(self)

        # Name of the new signal
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("New signal name:"))
        self.name_edit = QLineEdit()
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)

        # Expression
        expr_layout = QHBoxLayout()
        expr_layout.addWidget(QLabel("Expression with aliases (e.g., A + B):"))
        self.expr_edit = QLineEdit()

        # Example expressions
        self.expr_edit.setPlaceholderText(f"{self._example_alias1} + {self._example_alias2} * 2")
        expr_layout.addWidget(self.expr_edit)
        layout.addLayout(expr_layout)

        # Validation button
        validate_btn = QPushButton("Check Expression")
        validate_btn.clicked.connect(self._validate_current_expression)
        layout.addWidget(validate_btn)

        # Area for alias assignment
        alias_label = QLabel("Assign aliases to real signals:")
        layout.addWidget(alias_label)
        self.alias_area = QVBoxLayout()
        layout.addLayout(self.alias_area)

        # Update the alias area when text changes
        self.expr_edit.textChanged.connect(self._update_alias_inputs)

        # Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        Logger.log_message_static("Virtual Signal Dialog UI setup complete", Logger.DEBUG)

    def _validate_current_expression(self):
        """
        Checks the validity of the current expression and displays the result to the user.
        """
        Logger.log_message_static("User requested expression validation", Logger.INFO)
        expression = self.expr_edit.text().strip()
        Logger.log_message_static(f"Validating expression: '{expression}'", Logger.DEBUG)

        aliases = sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", expression)))
        Logger.log_message_static(f"Found aliases in expression: {aliases}", Logger.DEBUG)

        is_valid, error_msg = validate_expression(expression, aliases)

        if is_valid:
            Logger.log_message_static("Expression validation successful, showing confirmation message", Logger.INFO)
            QMessageBox.information(self, "Expression Validation", "The expression is syntactically correct.")
        else:
            Logger.log_message_static(f"Expression validation failed: {error_msg}, showing warning message", Logger.WARNING)
            QMessageBox.warning(self, "Expression Validation", f"Invalid expression:\n{error_msg}")

    def _update_alias_inputs(self):
        """
        Detects aliases used in the expression and creates dropdown menus for their assignment.
        """
        Logger.log_message_static("Updating alias input fields based on expression", Logger.DEBUG)

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
        Logger.log_message_static(f"Found {len(aliases)} aliases in expression: {aliases}", Logger.DEBUG)

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
            Logger.log_message_static(f"Added dropdown for alias '{alias}'", Logger.DEBUG)

    def get_result(self):
        """
        Returns the signal name, expression, and mapping of aliases to real signal names.

        Returns:
            tuple[str, str, dict[str, str]]: (signal_name, expression, alias_mapping)
        """
        name = self.name_edit.text().strip()
        expr = self.expr_edit.text().strip()
        mapping = {alias: combo.currentText() for alias, combo in self.alias_mapping.items()}
        Logger.log_message_static(f"Getting dialog result - name: '{name}', expression: '{expr}', mapping: {mapping}", Logger.DEBUG)
        return name, expr, mapping

    def validate_and_accept(self):
        """
        Validates the input and accepts the dialog on success.
        Checks the signal name, expression, and use of aliases.
        """
        Logger.log_message_static("Validating dialog inputs before accepting", Logger.INFO)
        name, expr, mapping = self.get_result()

        # Name validation
        Logger.log_message_static(f"Validating signal name: '{name}'", Logger.DEBUG)
        is_valid_name, name_error = validate_signal_name(name, self.signal_names)
        if not is_valid_name:
            Logger.log_message_static(f"Name validation failed: {name_error}", Logger.WARNING)
            QMessageBox.warning(self, "Validation Error", name_error)
            return

        # Expression validation
        if not expr:
            Logger.log_message_static("Expression is empty", Logger.WARNING)
            QMessageBox.warning(self, "Validation Error", "Expression is required.")
            return

        # Alias validations
        aliases = list(mapping.keys())
        if not aliases:
            Logger.log_message_static("No aliases detected in expression", Logger.WARNING)
            QMessageBox.warning(self, "Validation Error", "No aliases were detected in the provided expression.")
            return

        # Expression validity check
        Logger.log_message_static(f"Validating expression: '{expr}'", Logger.DEBUG)
        is_valid_expr, expr_error = validate_expression(expr, aliases)
        if not is_valid_expr:
            Logger.log_message_static(f"Expression validation failed: {expr_error}", Logger.WARNING)
            QMessageBox.warning(self, "Validation Error", f"Invalid expression: {expr_error}")
            return

        # Check if each alias has an assigned signal
        empty_assignments = [alias for alias, signal_name in mapping.items() if not signal_name]
        if empty_assignments:
            Logger.log_message_static(f"Missing signal assignments for aliases: {empty_assignments}", Logger.WARNING)
            QMessageBox.warning(self, "Validation Error", "Each alias must have an assigned signal.")
            return

        # All is well, save the result
        self._result = self.get_result()
        Logger.log_message_static("Virtual signal dialog validation successful, accepting dialog", Logger.INFO)
        super().accept()