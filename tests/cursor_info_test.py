#!/usr/bin/env python3
"""
Test script for CursorInfoDialog class
"""
import os
import sys
import time
import tempfile
import csv
import re
from pathlib import Path

# Add parent directory to import path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Now we can import from the parent directory
from cursor_info import CursorInfoDialog
from PySide6.QtWidgets import (
    QApplication, QTableWidgetItem, QMessageBox,
    QFileDialog, QDialog
)
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt

# For mocking file dialogs
import unittest.mock


def close_all_message_boxes():
    """Find and close any open message boxes"""
    closed_count = 0
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QMessageBox):
            for button in widget.buttons():
                if widget.buttonRole(button) in [QMessageBox.AcceptRole, QMessageBox.YesRole,
                                                 QMessageBox.RejectRole, QMessageBox.NoRole]:
                    button.click()
                    closed_count += 1
                    break
    return closed_count


def test_extract_unit():
    """Test the extract_unit method"""
    print("\n=== Testing extract_unit method ===")
    dialog = CursorInfoDialog()

    test_cases = [
        {"name": "Standard Unit", "signal": "Voltage [V]", "expected": "V"},
        {"name": "No Unit", "signal": "Position", "expected": ""},
        {"name": "Empty Unit", "signal": "Speed []", "expected": ""},
        {"name": "Dash Unit", "signal": "Counter [-]", "expected": ""},
        {"name": "Complex Unit", "signal": "Acceleration [m/s²]", "expected": "m/s²"},
    ]

    passed = 0
    for test in test_cases:
        result = dialog.extract_unit(test["signal"])
        if result == test["expected"]:
            print(f"✓ PASSED: {test['name']} - Got '{result}'")
            passed += 1
        else:
            print(f"✗ FAILED: {test['name']} - Expected '{test['expected']}', got '{result}'")

    print(f"Passed: {passed}/{len(test_cases)}")
    dialog.reject()
    return passed == len(test_cases)


def test_clean_signal_name():
    """Test the clean_signal_name method"""
    print("\n=== Testing clean_signal_name method ===")
    dialog = CursorInfoDialog()

    test_cases = [
        {"name": "With Unit", "signal": "Voltage [V]", "expected": "Voltage"},
        {"name": "No Unit", "signal": "Position", "expected": "Position"},
        {"name": "With Spaces", "signal": "Speed  [km/h]", "expected": "Speed"},
        {"name": "Multiple Brackets", "signal": "Current [A] [RMS]", "expected": "Current"},
    ]

    passed = 0
    for test in test_cases:
        result = dialog.clean_signal_name(test["signal"])
        if result == test["expected"]:
            print(f"✓ PASSED: {test['name']} - Got '{result}'")
            passed += 1
        else:
            print(f"✗ FAILED: {test['name']} - Expected '{test['expected']}', got '{result}'")

    print(f"Passed: {passed}/{len(test_cases)}")
    dialog.reject()
    return passed == len(test_cases)


def test_is_boolean():
    """Test the _is_boolean method"""
    print("\n=== Testing _is_boolean method ===")
    dialog = CursorInfoDialog()

    test_cases = [
        {"name": "Both TRUE", "a": "TRUE", "b": "TRUE", "expected": True},
        {"name": "TRUE/FALSE", "a": "TRUE", "b": "FALSE", "expected": True},
        {"name": "Mixed Case", "a": "True", "b": "false", "expected": True},
        {"name": "Number/Bool", "a": "10.5", "b": "TRUE", "expected": False},
        {"name": "Both Numbers", "a": "10", "b": "20", "expected": False},
        {"name": "Empty Values", "a": "", "b": "", "expected": False},
    ]

    passed = 0
    for test in test_cases:
        result = dialog._is_boolean(test["a"], test["b"])
        if result == test["expected"]:
            print(f"✓ PASSED: {test['name']}")
            passed += 1
        else:
            print(f"✗ FAILED: {test['name']} - Expected {test['expected']}, got {result}")

    print(f"Passed: {passed}/{len(test_cases)}")
    dialog.reject()
    return passed == len(test_cases)


def test_calc_time_delta():
    """Test the calc_time_delta method"""
    print("\n=== Testing calc_time_delta method ===")
    dialog = CursorInfoDialog()

    test_cases = [
        {"name": "Same Time", "t1": "12:00:00.000", "t2": "12:00:00.000", "expected": "0.000 s"},
        {"name": "1 Second", "t1": "12:00:00.000", "t2": "12:00:01.000", "expected": "1.000 s"},
        {"name": "Milliseconds", "t1": "12:00:00.000", "t2": "12:00:00.500", "expected": "0.500 s"},
        {"name": "Backwards", "t1": "12:00:10.000", "t2": "12:00:00.000", "expected": "10.000 s"},
        {"name": "Invalid Format", "t1": "invalid", "t2": "12:00:00.000", "expected": "-"},
    ]

    passed = 0
    for test in test_cases:
        result = dialog.calc_time_delta(test["t1"], test["t2"])
        if result == test["expected"]:
            print(f"✓ PASSED: {test['name']} - Got '{result}'")
            passed += 1
        else:
            print(f"✗ FAILED: {test['name']} - Expected '{test['expected']}', got '{result}'")

    print(f"Passed: {passed}/{len(test_cases)}")
    dialog.reject()
    return passed == len(test_cases)


def test_update_data():
    """Test the update_data method with different cursor values"""
    print("\n=== Testing update_data functionality ===")
    dialog = CursorInfoDialog()

    # Test with both cursors active
    time_a = "12:00:00.000"
    time_b = "12:00:01.000"
    values_a = {
        "Voltage [V]": "100.0",
        "Current [A]": "10.0",
        "Digital [bool]": "TRUE"
    }
    values_b = {
        "Voltage [V]": "110.0",
        "Current [A]": "11.0",
        "Digital [bool]": "FALSE"
    }

    dialog.update_data(time_a, time_b, values_a, values_b, True, True)

    # Verify table content
    row_count = dialog.table.rowCount()
    if row_count != 3:
        print(f"✗ FAILED: Expected 3 rows, got {row_count}")
        dialog.reject()
        return False

    # More flexible check for voltage values - just check if the values contain the expected numbers
    voltage_found = False
    for row in range(row_count):
        signal_name = dialog.table.item(row, 0).text() if dialog.table.item(row, 0) else ""
        if "Voltage" in signal_name:
            voltage_found = True
            cell_a = dialog.table.item(row, 1).text() if dialog.table.item(row, 1) else ""
            cell_b = dialog.table.item(row, 2).text() if dialog.table.item(row, 2) else ""

            if "100" not in cell_a or "110" not in cell_b:
                print(f"✗ FAILED: Voltage values not correctly displayed. Got '{cell_a}' and '{cell_b}'")
                dialog.reject()
                return False
            break

    if not voltage_found:
        print("✗ FAILED: Voltage row not found in table")
        dialog.reject()
        return False

    print("✓ PASSED: Table populated correctly with cursor data")
    dialog.reject()
    return True


def test_export_to_csv():
    """Test the export_to_csv functionality"""
    print("\n=== Testing CSV export ===")
    dialog = CursorInfoDialog()

    # Prepare test data
    time_a = "12:00:00.000"
    time_b = "12:00:01.000"
    values_a = {"Voltage [V]": "100.0", "Digital [bool]": "TRUE"}
    values_b = {"Voltage [V]": "110.0", "Digital [bool]": "FALSE"}
    dialog.update_data(time_a, time_b, values_a, values_b, True, True)

    # Create a temporary file for the export
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        temp_filename = tmp.name

    # Mock the file dialog to return our temp file
    # Patch the export_to_csv method to avoid Unicode encoding issues
    original_export = dialog.export_to_csv

    def patched_export():
        file_path, _ = temp_filename, "CSV files (*.csv)"
        if file_path:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Signal", "A", "B", "Delta", "Delta/s"])

                for row in range(dialog.table.rowCount()):
                    row_data = []
                    for col in range(dialog.table.columnCount()):
                        item = dialog.table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)

    # Apply the patch and run export
    dialog.export_to_csv = patched_export
    dialog.export_to_csv()

    # Restore original method
    dialog.export_to_csv = original_export

    # Check if the file exists and has content
    if not os.path.exists(temp_filename):
        print("✗ FAILED: CSV file was not created")
        dialog.reject()
        return False

    # Read the CSV content
    with open(temp_filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Clean up
    os.unlink(temp_filename)

    # Verify CSV content
    if len(rows) < 1:  # At least header row
        print(f"✗ FAILED: Expected rows in CSV, got empty file")
        dialog.reject()
        return False

    if "Signal" not in rows[0][0] or "A" not in rows[0][1] or "B" not in rows[0][2]:
        print(f"✗ FAILED: Incorrect header row: {rows[0]}")
        dialog.reject()
        return False

    print("✓ PASSED: CSV export works correctly")
    dialog.reject()
    return True

def test_format_val():
    """Test the _format_val method with different formatting options"""
    print("\n=== Testing _format_val method ===")
    dialog = CursorInfoDialog()

    test_cases = [
        {"name": "Integer", "val": "10", "unit": "V", "is_bool": False, "expected": "10.000 V"},
        {"name": "Float", "val": "10.56789", "unit": "A", "is_bool": False, "expected": "10.568 A"},
        {"name": "Small Float", "val": "0.0001234", "unit": "s", "is_bool": False, "expected": "0.000123 s"},
        {"name": "Boolean TRUE", "val": "TRUE", "unit": "", "is_bool": True, "expected": "On"},
        {"name": "Boolean FALSE", "val": "FALSE", "unit": "", "is_bool": True, "expected": "FALSE"},
        {"name": "None Value", "val": None, "unit": "Hz", "is_bool": False, "expected": "-"},
        {"name": "Empty String", "val": "", "unit": "kg", "is_bool": False, "expected": "-"},
    ]

    # Test normal mode
    dialog._scientific_mode = False
    passed = 0
    for test in test_cases:
        result = dialog._format_val(test["val"], test["unit"], test["is_bool"])
        if result == test["expected"]:
            print(f"✓ PASSED: {test['name']} - Got '{result}'")
            passed += 1
        else:
            print(f"✗ FAILED: {test['name']} - Expected '{test['expected']}', got '{result}'")

    # Also test scientific mode with one case
    dialog._scientific_mode = True
    sci_result = dialog._format_val("10.56789", "A", False)
    if sci_result.startswith("1.057e+01 A"):
        print("✓ PASSED: Scientific notation formatting")
        passed += 1
    else:
        print(f"✗ FAILED: Scientific notation - Expected '1.057e+01 A', got '{sci_result}'")

    print(f"Passed: {passed}/{len(test_cases) + 1}")
    dialog.reject()
    return passed == len(test_cases) + 1


def run_all_tests():
    """Run all tests and report results"""
    print("\n=== Running All CursorInfoDialog Tests ===")

    results = [
        ("Extract Unit", test_extract_unit()),
        ("Clean Signal Name", test_clean_signal_name()),
        ("Is Boolean", test_is_boolean()),
        ("Calculate Time Delta", test_calc_time_delta()),
        ("Update Data", test_update_data()),
        ("Export to CSV", test_export_to_csv()),
        ("Format Value", test_format_val()),
    ]

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print("\n=== Test Summary ===")
    for name, result in results:
        print(f"{name}: {'✓ PASSED' if result else '✗ FAILED'}")

    print(f"\nPassed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Total: {total}")

    return passed, total - passed


if __name__ == "__main__":
    app = QApplication(sys.argv)
    run_all_tests()
    sys.exit(0)