"""
Test script for VirtualSignalDialog class
"""
import os
import sys
import time
from pathlib import Path

# Add parent directory to import path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Now we can import from the parent directory
from virtual_signal_dialog import VirtualSignalDialog
from PySide6.QtWidgets import QApplication, QLineEdit, QMessageBox, QPushButton, QDialog
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt


def close_all_message_boxes():
    """Find and close any open message boxes and return their text content"""
    closed_count = 0
    message_texts = []

    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QMessageBox):
            message_text = widget.text()
            message_texts.append(message_text)
            print(f"Auto-closing message box: {message_text}")

            for button in widget.buttons():
                if widget.buttonRole(button) in [QMessageBox.AcceptRole, QMessageBox.YesRole,
                                                 QMessageBox.RejectRole, QMessageBox.NoRole]:
                    button.click()
                    closed_count += 1
                    break

    return closed_count, message_texts


def run_automated_tests(sample_signals):
    """Run automated tests for various expressions"""
    print("\n=== Running Automated Expression Tests ===")

    test_cases = [
        {"name": "Simple Addition", "expression": "A + B", "expected_valid": True},
        {"name": "Simple Subtraction", "expression": "A - B", "expected_valid": True},
        {"name": "Simple Multiplication", "expression": "A * B", "expected_valid": True},
        {"name": "Simple Division", "expression": "A / B", "expected_valid": True},
        {"name": "Power Calculation", "expression": "A * B * 0.8", "expected_valid": True},
        {"name": "Complex Expression", "expression": "(A * B) / (C + 0.1)", "expected_valid": True},
        {"name": "Invalid Variable", "expression": "A + D", "expected_valid": True},  # Syntax is valid
        {"name": "Syntax Error", "expression": "A + * B", "expected_valid": False},
    ]

    passed = 0
    failed = 0

    # Create a test dialog first to debug UI elements
    debug_dialog = VirtualSignalDialog(sample_signals)
    print("Dialog created. Looking for line edit fields...")

    # Debug: Print all QLineEdit fields found
    line_edits = debug_dialog.findChildren(QLineEdit)
    print(f"Found {len(line_edits)} QLineEdit fields:")
    for i, field in enumerate(line_edits):
        print(f"  {i}: objectName='{field.objectName()}', text='{field.text()}'")

    # Debug: Find buttons
    buttons = debug_dialog.findChildren(QPushButton)
    print(f"Found {len(buttons)} QPushButton objects:")
    for i, button in enumerate(buttons):
        print(f"  {i}: objectName='{button.objectName()}', text='{button.text()}'")

    debug_dialog.reject()

    # Now run the actual tests
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        print(f"Expression: {test['expression']}")

        dialog = VirtualSignalDialog(sample_signals)

        try:
            # Use index-based approach if object names aren't reliable
            line_edits = dialog.findChildren(QLineEdit)

            # Assume first field is name and second is expression
            if len(line_edits) >= 2:
                name_field = line_edits[0]
                expression_field = line_edits[1]

                name_field.setText(f"Test_{test['name'].replace(' ', '_')}")
                expression_field.setText(test["expression"])

                # Process events to ensure UI updates
                QApplication.processEvents()

                # Get buttons
                buttons = dialog.findChildren(QPushButton)
                check_button = None
                ok_button = None

                for button in buttons:
                    if button.text() == "Check Expression":
                        check_button = button
                    elif button.text() == "OK":
                        ok_button = button

                # Click the Check Expression button and capture message
                message_texts = []
                if check_button:
                    QTest.mouseClick(check_button, Qt.LeftButton)

                    # Process events and collect any messages
                    for _ in range(10):
                        QApplication.processEvents()
                        _, texts = close_all_message_boxes()
                        message_texts.extend(texts)
                        if texts:
                            break
                        time.sleep(0.05)

                # Log any messages received
                if message_texts:
                    print(f"Message received: {message_texts[0][:80]}{'...' if len(message_texts[0]) > 80 else ''}")

                # Test by clicking the OK button to see if validation passes
                has_syntax_error = any("syntax error" in msg.lower() or "invalid expression" in msg.lower()
                                       for msg in message_texts)

                # Compare results with expected
                if has_syntax_error == (not test["expected_valid"]):
                    print(f"✓ PASSED: Expression validation matches expected")
                    passed += 1
                else:
                    print(f"✗ FAILED: Expected {'invalid' if not test['expected_valid'] else 'valid'} syntax")
                    failed += 1
            else:
                print(f"Not enough QLineEdit fields found (needed 2, found {len(line_edits)})")

        except Exception as e:
            print(f"✗ FAILED: Exception occurred: {e}")
            failed += 1

        finally:
            # Close dialog and any message boxes
            dialog.reject()
            close_all_message_boxes()

    print("\n=== Test Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    return passed, failed


# Module testing
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Example signal names
    sample_signals = ["Voltage", "Current", "Power Factor", "Torque"]

    # Run automated tests
    run_automated_tests(sample_signals)

    # Interactive testing
    print("\n=== Starting Interactive Test ===")
    print("Please enter an expression manually in the dialog")

    dialog = VirtualSignalDialog(sample_signals)
    if dialog.exec():
        name, expression, mapping = dialog.get_result()
        print(f"New signal: {name}")
        print(f"Expression: {expression}")
        print(f"Mapping: {mapping}")

    # Exit automatically without showing the interactive dialog
    sys.exit(0)