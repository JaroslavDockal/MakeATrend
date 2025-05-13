import unittest
import sys
import os
from pathlib import Path
from unittest import mock

# Add parent directory to import path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Set up global mocking for GUI dependencies before importing tests
sys.modules['PySide6'] = mock.MagicMock()
sys.modules['PySide6.QtWidgets'] = mock.MagicMock()
sys.modules['PySide6.QtCore'] = mock.MagicMock()
sys.modules['PySide6.QtGui'] = mock.MagicMock()
sys.modules['PySide6.QtTest'] = mock.MagicMock()  # Add QtTest mock
sys.modules['pyqtgraph'] = mock.MagicMock()


def run_all_tests():
    """Run all tests in the tests directory"""
    # Get the current directory (tests directory)
    test_dir = os.path.dirname(os.path.abspath(__file__))
    current_file = os.path.basename(__file__)

    # Discover and collect all test files ending with _test.py
    test_suite = unittest.defaultTestLoader.discover(test_dir, pattern='*_test.py')

    # Filter out this file from the test suite
    filtered_suite = unittest.TestSuite()
    for suite in test_suite:
        for test_case in suite:
            if current_file not in str(test_case):
                filtered_suite.addTest(test_case)

    # Create test runner with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=False)

    # Run all discovered tests (using filtered_suite instead of test_suite)
    result = runner.run(filtered_suite)

    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()

    # Exit with appropriate code (0 for success, 1 for failure)
    sys.exit(0 if success else 1)