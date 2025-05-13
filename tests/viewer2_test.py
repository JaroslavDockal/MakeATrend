import unittest
import sys
from pathlib import Path
import numpy as np
from unittest import mock

# Add parent directory to import path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Mock Qt and related modules
sys.modules['PySide6'] = mock.MagicMock()
sys.modules['PySide6.QtWidgets'] = mock.MagicMock()
sys.modules['PySide6.QtCore'] = mock.MagicMock()
sys.modules['PySide6.QtGui'] = mock.MagicMock()
sys.modules['pyqtgraph'] = mock.MagicMock()

# Import our test module with SignalViewerTest class
from viewer_test import SignalViewerTest, MockSignalViewer


class MainViewerTests(unittest.TestCase):
    """Additional integration tests to validate component interconnections"""

    def setUp(self):
        self.viewer = MockSignalViewer()
        # Configure mock data
        self.viewer.data_signals = {
            "Signal1": (np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])),
            "Signal2": (np.array([1.0, 2.0, 3.0]), np.array([100.0, 200.0, 300.0]))
        }

    def test_signal_filter_affects_visibility(self):
        """Test that filtering signals properly affects their visibility"""
        # Set up signal widgets
        self.viewer.signal_widgets = {
            "Signal1": {"row": mock.MagicMock(), "checkbox": mock.MagicMock()},
            "TestSignal": {"row": mock.MagicMock(), "checkbox": mock.MagicMock()},
            "DataValue": {"row": mock.MagicMock(), "checkbox": mock.MagicMock()}
        }

        # Apply filter and check visibility
        self.viewer.apply_signal_filter("signal")

        # Signal1 and TestSignal should be visible, DataValue should be hidden
        self.viewer.signal_widgets["Signal1"]["row"].setVisible.assert_called_with(True)
        self.viewer.signal_widgets["TestSignal"]["row"].setVisible.assert_called_with(True)
        self.viewer.signal_widgets["DataValue"]["row"].setVisible.assert_called_with(False)

    def test_toggle_signal_updates_axis_labels(self):
        """Test that toggling a signal properly updates axis labels"""
        # Mock update_axis_labels method
        self.viewer.update_axis_labels = mock.MagicMock()

        # Setup signal toggling
        name = "Signal1"
        self.viewer.signal_widgets[name] = {
            'checkbox': mock.MagicMock(),
            'axis': mock.MagicMock(),
            'color_btn': mock.MagicMock(),
            'width': mock.MagicMock(),
            'row': mock.MagicMock()
        }

        # Configure the mock to return the desired values when called
        self.viewer.signal_widgets[name]['checkbox'].isChecked.return_value = True
        self.viewer.signal_widgets[name]['color_btn'].palette().button().color().name.return_value = "#FF0000"
        self.viewer.signal_widgets[name]['width'].value.return_value = 2
        self.viewer.signal_widgets[name]['axis'].currentText.return_value = "Left"

        # Call toggle_signal method
        self.viewer.toggle_signal()

        # Verify update_axis_labels was called
        self.viewer.update_axis_labels.assert_called_once()

    def test_cursor_info_reflects_cursor_positions(self):
        """Test that cursor info is updated when cursors change position"""
        # Mock cursor info and cursor positions
        self.viewer.cursor_info = mock.MagicMock()
        self.viewer.cursor_info.isVisible.return_value = True

        self.viewer.cursor_a.isVisible.return_value = True
        self.viewer.cursor_b.isVisible.return_value = True
        self.viewer.cursor_a.value.return_value = 1.5  # Position between data points
        self.viewer.cursor_b.value.return_value = 2.5

        # Add some active curves
        self.viewer.curves = {
            "Signal1": mock.MagicMock(),
            "Signal2": mock.MagicMock()
        }

        # Call update_cursor_info
        self.viewer.update_cursor_info()

        # Verify that update_data was called with expected parameters
        self.viewer.cursor_info.update_data.assert_called_once()
        args = self.viewer.cursor_info.update_data.call_args[0]
        self.assertEqual(len(args), 6)  # Should have 6 arguments

    def test_clear_signals_removes_all_components(self):
        """Test that clear_signals properly removes all signal components"""
        # Setup test data
        curve1 = mock.MagicMock()
        curve2 = mock.MagicMock()
        self.viewer.curves = {'Signal1': curve1, 'Signal2': curve2}
        self.viewer.signal_axis_map = {'Left': ['Signal1'], 'Right': ['Signal2'], 'Digital': []}
        self.viewer.signal_styles = {'Signal1': ('Left', '#FF0000', 2), 'Signal2': ('Right', '#00FF00', 2)}

        # Execute clear_signals
        self.viewer.clear_signals()

        # Verify curves were removed from viewboxes
        for vb in self.viewer.viewboxes.values():
            self.assertEqual(len(vb.items), 0)

        # Verify data structures were cleared
        self.assertEqual(len(self.viewer.curves), 0)
        self.assertEqual(len(self.viewer.signal_styles), 0)
        for axis, signals in self.viewer.signal_axis_map.items():
            self.assertEqual(len(signals), 0)


def create_test_suite():
    """Create a test suite containing all tests"""
    suite = unittest.TestSuite()

    # Add all tests from SignalViewerTest
    suite.addTest(unittest.makeSuite(SignalViewerTest))

    # Add all tests from MainViewerTests
    suite.addTest(unittest.makeSuite(MainViewerTests))

    return suite


if __name__ == '__main__':
    # Create and run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = create_test_suite()
    runner.run(test_suite)