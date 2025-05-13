"""
Test script for signal_colors.py

Tests color cycling, consistency, and provides visual comparison.
"""
import os
import sys
from pathlib import Path

# Add parent directory to import path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Now we can import from the parent directory
from signal_colors import SignalColors

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                               QWidget, QLabel, QGridLayout)
from PySide6.QtGui import QColor, QPalette
from PySide6.QtCore import Qt


class ColorTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Colors Test")
        self.resize(900, 600)

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # Add color cycling test
        main_layout.addWidget(QLabel("<h2>Color Cycling Test</h2>"))
        cycling_layout = QGridLayout()

        # Show 20 colors in sequence and after wrapping
        for i in range(40):
            color = SignalColors.get_color(i)
            label = QLabel(f"Color {i}")
            label.setAlignment(Qt.AlignCenter)

            # Set background color
            palette = label.palette()
            palette.setColor(QPalette.Window, QColor(color))
            label.setAutoFillBackground(True)
            label.setPalette(palette)

            # Set text color (white or black based on brightness)
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            text_color = "black" if brightness > 128 else "white"
            label.setStyleSheet(f"color: {text_color}; padding: 10px;")

            # Add to grid
            row, col = i // 10, i % 10
            cycling_layout.addWidget(label, row, col)

        main_layout.addLayout(cycling_layout)

        # Add name consistency test
        main_layout.addWidget(QLabel("<h2>Name Consistency Test</h2>"))
        name_layout = QGridLayout()

        # Test with 10 sample signal names
        test_names = ["Voltage", "Current", "Temperature", "Pressure",
                      "Flow", "Speed", "Position", "Torque", "Power", "Frequency"]

        for i, name in enumerate(test_names):
            color1 = SignalColors.get_color_for_name(name)
            color2 = SignalColors.get_color_for_name(name)

            # The colors should be identical
            assert color1 == color2, f"Color not consistent for name {name}"

            # Create a colored label
            label = QLabel(f"{name}: {color1}")
            label.setAlignment(Qt.AlignCenter)

            palette = label.palette()
            palette.setColor(QPalette.Window, QColor(color1))
            label.setAutoFillBackground(True)
            label.setPalette(palette)

            # Set text color
            r, g, b = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            text_color = "black" if brightness > 128 else "white"
            label.setStyleSheet(f"color: {text_color}; padding: 10px;")

            # Add to grid
            row, col = i // 5, i % 5
            name_layout.addWidget(label, row, col)

        main_layout.addLayout(name_layout)

        # Add color adjacency test
        main_layout.addWidget(QLabel("<h2>Adjacent Colors Test</h2>"))
        adjacent_layout = QGridLayout()

        # Display adjacent colors side by side
        num_colors = len(SignalColors.COLORS)
        for i in range(num_colors - 1):
            color1 = SignalColors.COLORS[i]
            color2 = SignalColors.COLORS[i + 1]

            # Create a label displaying both colors
            label = QLabel()
            label.setStyleSheet(f"""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 {color1}, stop:1 {color2});
                padding: 20px;
            """)

            # Add to grid
            row, col = i // 5, i % 5
            adjacent_layout.addWidget(label, row, col)

        main_layout.addLayout(adjacent_layout)

        # Set main widget
        self.setCentralWidget(main_widget)

        # Run tests
        self.run_unit_tests()

    def run_unit_tests(self):
        """Run programmatic tests on the SignalColors class"""
        print("\n=== Running Unit Tests ===")

        # Test 1: Colors wrap around properly
        color_count = len(SignalColors.COLORS)
        for i in range(color_count):
            assert SignalColors.get_color(i) == SignalColors.get_color(i + color_count)
        print("✓ Color cycling test passed")

        # Test 2: Same name always gets same color
        test_cases = ["Signal1", "VeryLongSignalName", "s", "123signal"]
        for name in test_cases:
            c1 = SignalColors.get_color_for_name(name)
            c2 = SignalColors.get_color_for_name(name)
            assert c1 == c2, f"Color inconsistency for {name}: {c1} != {c2}"
        print("✓ Color consistency test passed")

        # Test 3: Different names should get different colors (usually)
        unique_colors = {SignalColors.get_color_for_name(f"Signal{i}") for i in range(20)}
        assert len(unique_colors) > 1, "Different names produced too few unique colors"
        print(f"✓ Different names produced {len(unique_colors)} distinct colors")

        print("All tests passed successfully!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ColorTestWindow()
    window.show()
    sys.exit(app.exec())