#!/usr/bin/env python3
"""
Test script for Crosshair class
"""
import os
import sys
from pathlib import Path
from unittest import mock
from datetime import datetime, timezone

# Add parent directory to import path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Now we can import from the parent directory
from crosshair import Crosshair
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QRectF, QPointF, Qt
import pyqtgraph as pg


class MockScene:
    """Mock scene for testing signal connections"""

    def __init__(self):
        self.sigMouseMoved = mock.MagicMock()


class MockViewBox:
    """Mock ViewBox for testing Crosshair functionality"""

    def __init__(self):
        self.items = []
        self._scene = MockScene()
        self._mouse_enabled = {"x": True, "y": True}
        self._view_range = [[0, 100], [0, 100]]
        self._bounding_rect = QRectF(0, 0, 100, 100)

    def addItem(self, item, ignoreBounds=False):
        self.items.append(item)

    def scene(self):
        return self._scene

    def setMouseEnabled(self, x=None, y=None):
        if x is not None:
            self._mouse_enabled["x"] = x
        if y is not None:
            self._mouse_enabled["y"] = y

    def mapSceneToView(self, pos):
        return QPointF(pos.x(), pos.y())

    def viewRange(self):
        return self._view_range

    def sceneBoundingRect(self):
        return self._bounding_rect


def test_initialization():
    """Test if the crosshair initializes correctly"""
    print("\n=== Testing Crosshair initialization ===")

    viewbox = MockViewBox()
    crosshair = Crosshair(viewbox)

    # Check if all elements were created and added to the viewbox
    if len(viewbox.items) != 5:  # vline, hline, 3 labels
        print(f"✗ FAILED: Expected 5 items added to viewbox, got {len(viewbox.items)}")
        return False

    # Check if proxy was set up correctly
    if not hasattr(crosshair, "proxy"):
        print("✗ FAILED: Signal proxy was not created")
        return False

    # Check initial state
    if crosshair.enabled:
        print("✗ FAILED: Crosshair should be disabled initially")
        return False

    print("✓ PASSED: Crosshair initialization works correctly")
    return True


def test_toggle():
    """Test if the toggle method works correctly"""
    print("\n=== Testing toggle functionality ===")

    viewbox = MockViewBox()
    crosshair = Crosshair(viewbox)

    # Test enabling
    crosshair.toggle(True)

    if not crosshair.enabled:
        print("✗ FAILED: Crosshair was not enabled")
        return False

    if viewbox._mouse_enabled["x"] or viewbox._mouse_enabled["y"]:
        print("✗ FAILED: Mouse should be disabled when crosshair is enabled")
        return False

    # Test disabling
    crosshair.toggle(False)

    if crosshair.enabled:
        print("✗ FAILED: Crosshair was not disabled")
        return False

    if not viewbox._mouse_enabled["x"] or not viewbox._mouse_enabled["y"]:
        print("✗ FAILED: Mouse should be enabled when crosshair is disabled")
        return False

    print("✓ PASSED: Toggle functionality works correctly")
    return True


def test_show_hide():
    """Test if show and hide methods work correctly"""
    print("\n=== Testing show/hide functionality ===")

    viewbox = MockViewBox()
    crosshair = Crosshair(viewbox)

    # Mock the show/hide methods of elements
    crosshair.vline.show = mock.MagicMock()
    crosshair.vline.hide = mock.MagicMock()
    crosshair.hline.show = mock.MagicMock()
    crosshair.hline.hide = mock.MagicMock()
    crosshair.label_x.show = mock.MagicMock()
    crosshair.label_x.hide = mock.MagicMock()
    crosshair.label_y_left.show = mock.MagicMock()
    crosshair.label_y_left.hide = mock.MagicMock()
    crosshair.label_y_right.show = mock.MagicMock()
    crosshair.label_y_right.hide = mock.MagicMock()

    # Test show
    crosshair.show()

    if not crosshair.vline.show.called:
        print("✗ FAILED: vline.show() was not called")
        return False

    if not (crosshair.hline.show.called and
            crosshair.label_x.show.called and
            crosshair.label_y_left.show.called and
            crosshair.label_y_right.show.called):
        print("✗ FAILED: Not all elements were shown")
        return False

    # Test hide
    crosshair.hide()

    if not crosshair.vline.hide.called:
        print("✗ FAILED: vline.hide() was not called")
        return False

    if not (crosshair.hline.hide.called and
            crosshair.label_x.hide.called and
            crosshair.label_y_left.hide.called and
            crosshair.label_y_right.hide.called):
        print("✗ FAILED: Not all elements were hidden")
        return False

    print("✓ PASSED: Show/hide functionality works correctly")
    return True


def test_mouse_moved():
    """Test if the mouse_moved method works correctly"""
    print("\n=== Testing mouse_moved functionality ===")

    viewbox = MockViewBox()
    crosshair = Crosshair(viewbox)

    # Mock necessary methods
    crosshair.vline.setPos = mock.MagicMock()
    crosshair.hline.setPos = mock.MagicMock()
    crosshair.label_x.setPos = mock.MagicMock()
    crosshair.label_x.setText = mock.MagicMock()
    crosshair.label_y_left.setPos = mock.MagicMock()
    crosshair.label_y_left.setText = mock.MagicMock()
    crosshair.label_y_right.setPos = mock.MagicMock()
    crosshair.label_y_right.setText = mock.MagicMock()

    # Test with crosshair disabled
    pos = QPointF(50, 50)
    crosshair.mouse_moved([pos])

    if (crosshair.vline.setPos.called or crosshair.hline.setPos.called):
        print("✗ FAILED: Lines should not move when crosshair is disabled")
        return False

    # Enable and test again
    crosshair.enabled = True
    crosshair.mouse_moved([pos])

    if not (crosshair.vline.setPos.called and crosshair.hline.setPos.called):
        print("✗ FAILED: Lines were not repositioned on mouse move")
        return False

    if not (crosshair.label_x.setText.called and
            crosshair.label_y_left.setText.called and
            crosshair.label_y_right.setText.called):
        print("✗ FAILED: Label texts were not updated")
        return False

    # Test timestamp formatting
    timestamp = datetime.now().timestamp()
    pos = QPointF(timestamp, 50)

    # Mock mapSceneToView to return timestamp
    viewbox.mapSceneToView = lambda p: QPointF(timestamp, 50)

    crosshair.mouse_moved([pos])

    args = crosshair.label_x.setText.call_args[0][0]
    if not args.startswith("X: ") or args == "X: -":
        print("✗ FAILED: Timestamp not formatted correctly")
        print(f"Got: {args}")
        return False

    print("✓ PASSED: Mouse moved functionality works correctly")
    return True


def run_all_tests():
    """Run all tests and report results"""
    print("\n=== Running All Crosshair Tests ===")

    results = [
        ("Initialization", test_initialization()),
        ("Toggle", test_toggle()),
        ("Show/Hide", test_show_hide()),
        ("Mouse Moved", test_mouse_moved()),
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