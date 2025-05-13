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
from PySide6.QtCore import Qt
import re
import ast
import numpy as np

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
    # Check if the expression is empty
    if not expression.strip():
        return False, "Expression is empty"

    # Check if the expression contains any aliases
    python_keywords = {'and', 'or', 'if', 'else', 'True', 'False', 'None'}
    found_aliases = [a for a in re.findall(r"\b[A-Za-z_]\w*\b", expression)
                     if a not in python_keywords]
    if not found_aliases:
        return False, "Expression does not contain any signal aliases"

    # Check Python syntax
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"

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
            self.errors.append(f"Function calls are not allowed: {ast.unparse(node)}")
            self.generic_visit(node)

        def visit_Name(self, node):
            if node.id not in self.available_aliases and node.id not in self.python_keywords:
                self.errors.append(f"Unknown alias: {node.id}")
            self.generic_visit(node)

        def visit_Import(self, node):
            self.errors.append("Import statements are not allowed")

        def visit_ImportFrom(self, node):
            self.errors.append("Import statements are not allowed")

        def visit_Attribute(self, node):
            self.errors.append(f"Attribute access is not allowed: {ast.unparse(node)}")

        def visit_Lambda(self, node):
            self.errors.append("Lambda functions are not allowed")

    # Check for dangerous constructs
    safety_checker = SafetyVisitor(available_aliases)
    safety_checker.visit(tree)

    if safety_checker.errors:
        return False, "\n".join(safety_checker.errors)

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
    if not name:
        return False, "Signal name cannot be empty"

    if not re.match(r"^[A-Za-z_][A-Za-z0-9_\- ]*$", name):
        return False, "Signal name can only contain letters, digits, underscores, hyphens, and spaces, and must start with a letter"

    if name in existing_names:
        return False, f"Signal with name '{name}' already exists"

    return True, ""


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
        super().__init__(parent)
        self.setWindowTitle("Create Virtual Signal")
        self.setMinimumWidth(500)
        self.signal_names = signal_names
        self.alias_mapping = {}

        # Store the dialog result for later access
        self._result = None

        # Default aliases for examples
        self._example_alias1 = "A" if signal_names else ""
        self._example_alias2 = "B" if len(signal_names) > 1 else ""

        self.init_ui()

    def init_ui(self):
        """
        Creates the user interface for the dialog.
        """
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
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _validate_current_expression(self):
        """
        Checks the validity of the current expression and displays the result to the user.
        """
        expression = self.expr_edit.text().strip()
        aliases = sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", expression)))

        is_valid, error_msg = validate_expression(expression, aliases)

        if is_valid:
            QMessageBox.information(self, "Expression Validation", "The expression is syntactically correct.")
        else:
            QMessageBox.warning(self, "Expression Validation", f"Invalid expression:\n{error_msg}")

    def _update_alias_inputs(self):
        """
        Detects aliases used in the expression and creates dropdown menus for their assignment.
        """
        # Remove previous alias widgets
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
        Returns the signal name, expression, and mapping of aliases to real signal names.

        Returns:
            tuple[str, str, dict[str, str]]: (signal_name, expression, alias_mapping)
        """
        name = self.name_edit.text().strip()
        expr = self.expr_edit.text().strip()
        mapping = {alias: combo.currentText() for alias, combo in self.alias_mapping.items()}
        return name, expr, mapping

    def validate_and_accept(self):
        """
        Validates the input and accepts the dialog on success.
        Checks the signal name, expression, and use of aliases.
        """
        name, expr, mapping = self.get_result()

        # Name validation
        is_valid_name, name_error = validate_signal_name(name, self.signal_names)
        if not is_valid_name:
            QMessageBox.warning(self, "Validation Error", name_error)
            return

        # Expression validation
        if not expr:
            QMessageBox.warning(self, "Validation Error", "Expression is required.")
            return

        # Alias validations
        aliases = list(mapping.keys())
        if not aliases:
            QMessageBox.warning(self, "Validation Error", "No aliases were detected in the provided expression.")
            return

        # Expression validity check
        is_valid_expr, expr_error = validate_expression(expr, aliases)
        if not is_valid_expr:
            QMessageBox.warning(self, "Validation Error", f"Invalid expression: {expr_error}")
            return

        # Check if each alias has an assigned signal
        if any(not signal_name for signal_name in mapping.values()):
            QMessageBox.warning(self, "Validation Error", "Each alias must have an assigned signal.")
            return

        # All is well, save the result
        self._result = self.get_result()
        super().accept()