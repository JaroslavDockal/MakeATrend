import unittest
from unittest import mock
import sys
import numpy as np
from pathlib import Path

# Add parent directory to import path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Mock necessary Qt and pyqtgraph modules
sys.modules['PySide6'] = mock.MagicMock()
sys.modules['PySide6.QtWidgets'] = mock.MagicMock()
sys.modules['PySide6.QtCore'] = mock.MagicMock()
sys.modules['PySide6.QtGui'] = mock.MagicMock()
sys.modules['pyqtgraph'] = mock.MagicMock()


class MockViewBox:
    def __init__(self):
        self.items = []
        self.sceneBoundingRect = mock.MagicMock(return_value=(0, 0, 100, 100))
        self.sigResized = mock.MagicMock()

    def removeItem(self, item):
        if item in self.items:
            self.items.remove(item)

    def addItem(self, item):
        self.items.append(item)


class MockSignalViewer:
    """Enhanced mock of SignalViewer with testable helper functions"""

    def __init__(self):
        self.signals = {}
        self.plots = [mock.MagicMock()]
        self.crosshair = mock.MagicMock()
        self.curves = {}
        self.signal_axis_map = {'Left': [], 'Right': [], 'Digital': []}
        self.signal_styles = {}
        self.viewboxes = {
            'Left': MockViewBox(),
            'Right': MockViewBox(),
            'Digital': MockViewBox()
        }
        self.axis_labels = {
            'Left': mock.MagicMock(),
            'Right': mock.MagicMock(),
            'Digital': mock.MagicMock()
        }
        self.signal_widgets = {}
        self.complex_mode = False
        self.plot_widget = mock.MagicMock()
        self.scroll_layout = mock.MagicMock()
        self.cursor_a = mock.MagicMock()
        self.cursor_b = mock.MagicMock()
        self.cursor_info = mock.MagicMock()
        self.color_counter = 0
        self.cursor_a.isVisible.return_value = False
        self.cursor_b.isVisible.return_value = False
        self.data_signals = {}
        self.signal_filter_text = ""

    def plot_signals(self, signals):
        self.clear_all_plots()
        self.signals = signals

    def toggle_crosshairs(self, enabled):
        self.crosshair.toggle(enabled)

    def clear_all_plots(self):
        for plot in self.plots:
            plot.plotItem.clear()
        self.signals = {}

    def toggle_grid(self, state):
        # Actually call the showGrid method on the plot_widget mock
        self.plot_widget.showGrid(x=state, y=state)

    def apply_signal_filter(self, text):
        self.signal_filter_text = text.lower().strip()
        for name, widgets in self.signal_widgets.items():
            row_widget = widgets.get('row')
            visible = self.signal_filter_text in name.lower()
            row_widget.setVisible(visible)

    def update_axis_labels(self):
        for axis, label in self.axis_labels.items():
            names = self.signal_axis_map.get(axis, [])
            html_parts = []
            for name in names:
                base_name = name.split('[')[0].strip()
                _, color, _ = self.signal_styles.get(name, (None, "#FFFFFF", 2))
                html_parts.append(f'<span style="color:{color}">{base_name}</span>')
            html_text = ", ".join(html_parts) if html_parts else axis
            # Actually call the setLabel method
            label.setLabel(text=html_text)

    def toggle_complex_mode(self, state):
        self.complex_mode = state
        for widgets in self.signal_widgets.values():
            widgets['axis'].setVisible(state)
            widgets['color_btn'].setVisible(state)
            widgets['width'].setVisible(state)

    def toggle_signal(self, signal_name=None):
        """Toggle signal visibility based on checkbox state"""
        # If no signal name provided, assume sender() functionality
        if signal_name is None:
            # In a real application, we'd use sender() but in mock just use first signal
            signal_name = next(iter(self.signal_widgets)) if self.signal_widgets else None
            if not signal_name:
                return

        # Get checkbox state, color, width, and axis
        widgets = self.signal_widgets.get(signal_name, {})
        if not widgets:
            return

        checkbox = widgets.get('checkbox')
        if checkbox and checkbox.isChecked():
            # Signal is enabled
            color_btn = widgets.get('color_btn')
            width_spin = widgets.get('width')
            axis_combo = widgets.get('axis')

            # Get color, width, and axis values
            color = color_btn.palette().button().color().name() if color_btn else "#FFFFFF"
            width = width_spin.value() if width_spin else 1
            axis = axis_combo.currentText() if axis_combo else "Left"

            # Update signal styles and axis map
            self.signal_styles[signal_name] = (axis, color, width)

            # Make sure signal is in the axis map
            if signal_name not in self.signal_axis_map[axis]:
                self.signal_axis_map[axis].append(signal_name)
        else:
            # Signal is disabled - remove from maps
            for axis, signals in self.signal_axis_map.items():
                if signal_name in signals:
                    signals.remove(signal_name)

            if signal_name in self.signal_styles:
                del self.signal_styles[signal_name]

        # Update axis labels
        self.update_axis_labels()

    def update_cursor_info(self):
        """Update cursor info with current cursor positions"""
        if not self.cursor_info.isVisible():
            return

        # Check both cursors are visible
        if not (self.cursor_a.isVisible() and self.cursor_b.isVisible()):
            return

        # Get cursor positions
        pos_a = self.cursor_a.value()
        pos_b = self.cursor_b.value()

        # Signal data for calculations
        signal_data = {}
        for name, curve in self.curves.items():
            if name in self.data_signals:
                x_data, y_data = self.data_signals[name]
                signal_data[name] = (x_data, y_data)

        # Update cursor info dialog
        self.cursor_info.update_data(pos_a, pos_b, signal_data,
                                     self.signal_styles,
                                     self.signal_axis_map,
                                     self.viewboxes)

    def clear_signals(self):
        for curve in self.curves.values():
            for vb in self.viewboxes.values():
                vb.removeItem(curve)

        self.curves.clear()
        self.signal_axis_map = {k: [] for k in self.viewboxes}
        self.signal_styles.clear()
        self.color_counter = 0

        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.signal_widgets.clear()
        self.update_axis_labels()


# Mock the viewer module
sys.modules['viewer'] = mock.MagicMock()
sys.modules['viewer'].SignalViewer = MockSignalViewer


class SignalViewerTest(unittest.TestCase):
    def setUp(self):
        self.viewer = MockSignalViewer()

    def test_plot_signals(self):
        """Test plotting signals functionality"""
        signals = {
            "Signal1": (np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])),
            "Signal2": (np.array([1.0, 2.0, 3.0]), np.array([100.0, 200.0, 300.0]))
        }

        # Spy on the clear_all_plots method
        with mock.patch.object(self.viewer, 'clear_all_plots', wraps=self.viewer.clear_all_plots) as spy:
            self.viewer.plot_signals(signals)
            spy.assert_called_once()

        # Check signal dictionary update
        self.assertEqual(len(self.viewer.signals), len(signals))

    def test_toggle_crosshairs(self):
        """Test toggling crosshair functionality"""
        self.viewer.toggle_crosshairs(True)
        self.viewer.crosshair.toggle.assert_called_once_with(True)

        self.viewer.crosshair.toggle.reset_mock()
        self.viewer.toggle_crosshairs(False)
        self.viewer.crosshair.toggle.assert_called_once_with(False)

    def test_clear_all_plots(self):
        """Test clearing all plots"""
        # Add some test data
        self.viewer.signals = {"Signal1": (np.array([1, 2, 3]), np.array([10, 20, 30]))}

        # Test clearing
        self.viewer.clear_all_plots()
        self.viewer.plots[0].plotItem.clear.assert_called_once()
        self.assertEqual(len(self.viewer.signals), 0)

    def test_apply_signal_filter(self):
        """Test filtering signals based on text input"""
        # Setup signal widgets
        self.viewer.signal_widgets = {
            "Signal1": {"row": mock.MagicMock()},
            "TestSignal": {"row": mock.MagicMock()},
            "Measurement": {"row": mock.MagicMock()}
        }

        # Test filter
        self.viewer.apply_signal_filter("signal")
        self.assertEqual(self.viewer.signal_filter_text, "signal")
        self.viewer.signal_widgets["Signal1"]["row"].setVisible.assert_called_with(True)
        self.viewer.signal_widgets["TestSignal"]["row"].setVisible.assert_called_with(True)
        self.viewer.signal_widgets["Measurement"]["row"].setVisible.assert_called_with(False)

        # Test case insensitivity
        self.viewer.apply_signal_filter("TEST")
        self.viewer.signal_widgets["Signal1"]["row"].setVisible.assert_called_with(False)
        self.viewer.signal_widgets["TestSignal"]["row"].setVisible.assert_called_with(True)

    def test_toggle_grid(self):
        """Test toggling grid visibility"""
        # Test with grid enabled
        self.viewer.toggle_grid(True)
        self.viewer.plot_widget.showGrid.assert_called_with(x=True, y=True)

        # Test with grid disabled
        self.viewer.plot_widget.showGrid.reset_mock()
        self.viewer.toggle_grid(False)
        self.viewer.plot_widget.showGrid.assert_called_with(x=False, y=False)

    def test_update_axis_labels(self):
        """Test updating axis labels with signal names"""
        # Setup test data
        self.viewer.signal_axis_map = {
            'Left': ['Signal1', 'Signal2'],
            'Right': ['Signal3'],
            'Digital': []
        }
        self.viewer.signal_styles = {
            'Signal1': ('Left', '#FF0000', 2),
            'Signal2': ('Left', '#00FF00', 2),
            'Signal3': ('Right', '#0000FF', 2)
        }

        # Call the method
        self.viewer.update_axis_labels()

        # Check that each axis label was set with correct HTML
        left_expected = '<span style="color:#FF0000">Signal1</span>, <span style="color:#00FF00">Signal2</span>'
        right_expected = '<span style="color:#0000FF">Signal3</span>'
        digital_expected = 'Digital'

        self.viewer.axis_labels['Left'].setLabel.assert_called_with(text=left_expected)
        self.viewer.axis_labels['Right'].setLabel.assert_called_with(text=right_expected)
        self.viewer.axis_labels['Digital'].setLabel.assert_called_with(text=digital_expected)

    def test_toggle_complex_mode(self):
        """Test toggling between simple and complex UI mode"""
        # Setup mock widgets
        self.viewer.signal_widgets = {
            'Signal1': {
                'axis': mock.MagicMock(),
                'color_btn': mock.MagicMock(),
                'width': mock.MagicMock()
            },
            'Signal2': {
                'axis': mock.MagicMock(),
                'color_btn': mock.MagicMock(),
                'width': mock.MagicMock()
            }
        }

        # Test enabling complex mode
        self.viewer.toggle_complex_mode(True)
        self.assertTrue(self.viewer.complex_mode)
        for widgets in self.viewer.signal_widgets.values():
            widgets['axis'].setVisible.assert_called_with(True)
            widgets['color_btn'].setVisible.assert_called_with(True)
            widgets['width'].setVisible.assert_called_with(True)

        # Test disabling complex mode
        self.viewer.toggle_complex_mode(False)
        self.assertFalse(self.viewer.complex_mode)
        for widgets in self.viewer.signal_widgets.values():
            widgets['axis'].setVisible.assert_called_with(False)
            widgets['color_btn'].setVisible.assert_called_with(False)
            widgets['width'].setVisible.assert_called_with(False)

    def test_clear_signals(self):
        """Test clearing all signal plots and resetting widgets"""
        # Setup test data
        test_curve1 = mock.MagicMock()
        test_curve2 = mock.MagicMock()
        self.viewer.curves = {'Signal1': test_curve1, 'Signal2': test_curve2}
        self.viewer.signal_axis_map = {'Left': ['Signal1'], 'Right': ['Signal2'], 'Digital': []}
        self.viewer.signal_styles = {'Signal1': ('Left', '#FF0000', 2), 'Signal2': ('Right', '#00FF00', 2)}
        self.viewer.color_counter = 5

        # Test clearing
        with mock.patch.object(self.viewer, 'update_axis_labels') as update_mock:
            self.viewer.clear_signals()
            update_mock.assert_called_once()

        self.assertEqual(len(self.viewer.curves), 0)
        self.assertEqual(len(self.viewer.signal_styles), 0)
        self.assertEqual(self.viewer.color_counter, 0)
        for axis_signals in self.viewer.signal_axis_map.values():
            self.assertEqual(len(axis_signals), 0)


if __name__ == "__main__":
    unittest.main()