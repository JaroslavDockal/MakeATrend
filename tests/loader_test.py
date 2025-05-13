#!/usr/bin/env python3
"""
Test script for loader module
"""
import os
import sys
import numpy as np
from pathlib import Path
from unittest import mock

# Add parent directory to import path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Now we can import from the parent directory
from loader import load_single_file, load_multiple_files
from PySide6.QtWidgets import QApplication


def test_load_single_file_success():
    """Test successful loading of a single file"""
    print("\n=== Testing load_single_file success ===")

    # Mock data to return from parse_csv_or_recorder
    time_arr = np.array([1.0, 2.0, 3.0])
    signals = {
        "Signal1": np.array([10.0, 20.0, 30.0]),
        "Signal2": np.array([100.0, 200.0, 300.0])
    }

    # Mock QFileDialog.getOpenFileName to return a "file path"
    with mock.patch('PySide6.QtWidgets.QFileDialog.getOpenFileName',
                    return_value=("test_file.csv", "")):
        # Mock parse_csv_or_recorder to return our test data
        with mock.patch('loader.parse_csv_or_recorder',
                        return_value=(time_arr, signals)):
            result = load_single_file()

    # Check if result has the expected structure
    if len(result) != 2:
        print(f"✗ FAILED: Expected 2 signals, got {len(result)}")
        return False

    if "Signal1" not in result or "Signal2" not in result:
        print("✗ FAILED: Missing expected signal names")
        return False

    # Check if time arrays are correct
    signal1_time = result["Signal1"][0]
    if not np.array_equal(signal1_time, time_arr):
        print(f"✗ FAILED: Time array not preserved correctly")
        return False

    # Check if values are correct
    signal1_values = result["Signal1"][1]
    if not np.array_equal(signal1_values, signals["Signal1"]):
        print(f"✗ FAILED: Signal values not preserved correctly")
        return False

    print("✓ PASSED: Single file loading works correctly")
    return True


def test_load_single_file_cancel():
    """Test when user cancels file dialog"""
    print("\n=== Testing load_single_file cancel ===")

    # Mock QFileDialog.getOpenFileName to return an empty path (user canceled)
    with mock.patch('PySide6.QtWidgets.QFileDialog.getOpenFileName',
                    return_value=("", "")):
        result = load_single_file()

    # Should return empty dict
    if result != {}:
        print(f"✗ FAILED: Expected empty dict, got {result}")
        return False

    print("✓ PASSED: Cancel handling works correctly")
    return True


def test_load_single_file_error():
    """Test error handling during file parsing"""
    print("\n=== Testing load_single_file error handling ===")

    # Mock QFileDialog.getOpenFileName to return a "file path"
    with mock.patch('PySide6.QtWidgets.QFileDialog.getOpenFileName',
                    return_value=("test_file.csv", "")):
        # Mock parse_csv_or_recorder to raise an exception
        with mock.patch('loader.parse_csv_or_recorder',
                        side_effect=Exception("Test error")):
            result = load_single_file()

    # Should return empty dict
    if result != {}:
        print(f"✗ FAILED: Expected empty dict on error, got {result}")
        return False

    print("✓ PASSED: Error handling works correctly")
    return True


def test_load_multiple_files_success():
    """Test successful loading of multiple files"""
    print("\n=== Testing load_multiple_files success ===")

    # Mock data for two files with different signals
    time_arr1 = np.array([1.0, 2.0, 3.0])
    signals1 = {
        "Signal1": np.array([10.0, 20.0, 30.0]),
        "Signal2": np.array([100.0, 200.0, 300.0])
    }

    time_arr2 = np.array([4.0, 5.0, 6.0])
    signals2 = {
        "Signal1": np.array([40.0, 50.0, 60.0]),
        "Signal3": np.array([400.0, 500.0, 600.0])
    }

    # Mock QFileDialog.getOpenFileNames to return "file paths"
    with mock.patch('PySide6.QtWidgets.QFileDialog.getOpenFileNames',
                    return_value=(["file1.csv", "file2.csv"], "")):
        # Mock parse_csv_or_recorder to return our test data for each file
        with mock.patch('loader.parse_csv_or_recorder',
                        side_effect=[(time_arr1, signals1), (time_arr2, signals2)]):
            result = load_multiple_files()

    # Check if result has the expected signals
    if len(result) != 3:
        print(f"✗ FAILED: Expected 3 signals, got {len(result)}")
        return False

    if "Signal1" not in result or "Signal2" not in result or "Signal3" not in result:
        print("✗ FAILED: Missing expected signal names")
        return False

    # Check merged signal length (should have values + NaN between)
    signal1_time = result["Signal1"][0]
    signal1_values = result["Signal1"][1]

    # Should be length of both arrays + 1 for NaN
    expected_len = len(time_arr1) + len(time_arr2) + 1
    if len(signal1_time) != expected_len:
        print(f"✗ FAILED: Expected time array length {expected_len}, got {len(signal1_time)}")
        return False

    # Check for presence of NaN
    if not np.isnan(signal1_values).any():
        print("✗ FAILED: No NaN found in merged signal")
        return False

    print("✓ PASSED: Multiple file loading works correctly")
    return True


def test_load_multiple_files_overlapping():
    """Test loading multiple files with overlapping signals"""
    print("\n=== Testing load_multiple_files with overlapping signals ===")

    # Mock data for two files with overlapping time ranges
    time_arr1 = np.array([1.0, 2.0, 3.0])
    signals1 = {
        "Signal1": np.array([10.0, 20.0, 30.0])
    }

    time_arr2 = np.array([2.5, 3.5, 4.0])  # Overlaps with time_arr1
    signals2 = {
        "Signal1": np.array([25.0, 35.0, 40.0])
    }

    # Mock the required functions
    with mock.patch('PySide6.QtWidgets.QFileDialog.getOpenFileNames',
                    return_value=(["file1.csv", "file2.csv"], "")):
        with mock.patch('loader.parse_csv_or_recorder',
                        side_effect=[(time_arr1, signals1), (time_arr2, signals2)]):
            # Also mock print to capture the warning
            with mock.patch('builtins.print') as mock_print:
                result = load_multiple_files()

    # Check if warning message was printed
    warning_called = False
    for call in mock_print.call_args_list:
        if 'overlaps in time' in str(call):
            warning_called = True
            break

    if not warning_called:
        print("✗ FAILED: No overlap warning was printed")
        return False

    # Check if result still contains the signal but without overlap
    if "Signal1" not in result:
        print("✗ FAILED: Signal1 not in result")
        return False

    # Check if signal has the correct length (should be just first file data)
    signal1_values = result["Signal1"][1]
    if len(signal1_values) != len(time_arr1):
        print(f"✗ FAILED: Expected signal length {len(time_arr1)}, got {len(signal1_values)}")
        return False

    print("✓ PASSED: Overlapping signal handling works correctly")
    return True


def test_load_multiple_files_cancel():
    """Test when user cancels the file dialog"""
    print("\n=== Testing load_multiple_files cancel ===")

    # Mock QFileDialog.getOpenFileNames to return empty list (user canceled)
    with mock.patch('PySide6.QtWidgets.QFileDialog.getOpenFileNames',
                    return_value=([], "")):
        result = load_multiple_files()

    # Should return empty dict
    if result != {}:
        print(f"✗ FAILED: Expected empty dict, got {result}")
        return False

    print("✓ PASSED: Cancel handling works correctly")
    return True


def test_load_multiple_files_partial_error():
    """Test handling errors in some files but not others"""
    print("\n=== Testing load_multiple_files with partial errors ===")

    # Mock data for successful file
    time_arr = np.array([1.0, 2.0, 3.0])
    signals = {
        "Signal1": np.array([10.0, 20.0, 30.0])
    }

    # Mock the required functions
    with mock.patch('PySide6.QtWidgets.QFileDialog.getOpenFileNames',
                    return_value=(["file1.csv", "file2.csv"], "")):
        with mock.patch('loader.parse_csv_or_recorder',
                        side_effect=[(time_arr, signals), Exception("Test error")]):
            # Also mock print to capture the error
            with mock.patch('builtins.print') as mock_print:
                result = load_multiple_files()

    # Check if error message was printed
    error_called = False
    for call in mock_print.call_args_list:
        if 'Failed to load' in str(call):
            error_called = True
            break

    if not error_called:
        print("✗ FAILED: No error message was printed")
        return False

    # Check if result contains the successfully loaded signal
    if "Signal1" not in result:
        print("✗ FAILED: Successfully loaded signal not in result")
        return False

    print("✓ PASSED: Partial error handling works correctly")
    return True


def run_all_tests():
    """Run all tests and report results"""
    print("\n=== Running All Loader Tests ===")

    results = [
        ("Load Single File Success", test_load_single_file_success()),
        ("Load Single File Cancel", test_load_single_file_cancel()),
        ("Load Single File Error", test_load_single_file_error()),
        ("Load Multiple Files Success", test_load_multiple_files_success()),
        ("Load Multiple Files Overlapping", test_load_multiple_files_overlapping()),
        ("Load Multiple Files Cancel", test_load_multiple_files_cancel()),
        ("Load Multiple Files Partial Error", test_load_multiple_files_partial_error()),
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