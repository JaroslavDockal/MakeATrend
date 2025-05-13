"""
Test script for utils module
"""
import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to import path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Import functions from your utils module
from utils import (
    find_nearest_index,
    is_digital_signal,
    parse_csv_file,
    parse_recorder_format,
    parse_csv_or_recorder,
    export_graph
)

# Qt imports
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox
)

import pyqtgraph as pg
import tempfile

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication, QMainWindow
    import sys
    import pyqtgraph as pg
    import tempfile
    import os

    app = QApplication(sys.argv)

    # Create a test window with a plot widget
    window = QMainWindow()
    window.setWindowTitle("Utils Module Test")
    window.resize(800, 600)

    # Create a plot widget with some sample data
    plot = pg.PlotWidget()
    window.setCentralWidget(plot)

    # Add some test data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, size=len(x))
    plot.plot(x, y, pen='g')
    window.show()  # Show the window to render the plot widget

    print("=== Testing functions in utils.py ===")

    # Test find_nearest_index
    test_array = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    test_value = 3.0
    nearest_idx = find_nearest_index(test_array, test_value)
    print(f"Nearest index to {test_value} in {test_array} is: {nearest_idx}")
    assert nearest_idx == 2, "find_nearest_index failed"
    print("✓ find_nearest_index test passed")

    # Test is_digital_signal
    digital_signal = np.array([0, 1, 0, 1])
    analog_signal = np.array([0.1, 0.2, 0.3, 0.4])
    string_digital = np.array(['TRUE', 'FALSE', 'true', 'false'])
    print(f"Is [0, 1, 0, 1] digital? {is_digital_signal(digital_signal)}")
    print(f"Is [0.1, 0.2, 0.3, 0.4] digital? {is_digital_signal(analog_signal)}")
    print(f"Is ['TRUE', 'FALSE', 'true', 'false'] digital? {is_digital_signal(string_digital)}")
    assert is_digital_signal(digital_signal) == True
    assert is_digital_signal(analog_signal) == False
    assert is_digital_signal(string_digital) == True
    print("✓ is_digital_signal test passed")

    # === Create test data files ===
    print("\n=== Creating test data files ===")
    # Create a test CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        test_csv_path = f.name
        f.write("Date;Time;Signal1;Signal2\n")
        f.write("2023-01-01;12:00:00,000;10.5;20.5\n")
        f.write("2023-01-01;12:00:01,000;11.2;21.2\n")
        f.write("2023-01-01;12:00:02,000;12.1;22.1\n")
    print(f"Created test CSV at {test_csv_path}")

    # Create a test recorder file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
        test_recorder_path = f.name
        f.write("RECORDER VALUES\n")
        f.write("Time of Interval: 01/15/23 10:30:00\n")
        f.write("Interval: 0.05 sec\n")
        f.write("Item 1 = Voltage\n")
        f.write("Item 2 = Current\n")
        f.write("    0   230.5   10.2\n")
        f.write("    1   231.0   10.3\n")
        f.write("    2   230.8   10.1\n")
    print(f"Created test recorder file at {test_recorder_path}")

    # Create a test boolean CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        test_bool_csv_path = f.name
        f.write("Date;Time;DigitalSignal1;DigitalSignal2\n")
        f.write("2025-05-09;19:29:53,239;TRUE;FALSE\n")
        f.write("2025-05-09;19:29:53,342;TRUE;FALSE\n")
        f.write("2025-05-09;19:29:53,443;TRUE;FALSE\n")
    print(f"Created test boolean CSV at {test_bool_csv_path}")

    # === Test parsing functions ===
    print("\n=== Testing CSV parsing ===")
    try:
        timestamps, signals = parse_csv_file(test_csv_path)
        print(f"Parsed {len(timestamps)} timestamps")
        print(f"Found signals: {list(signals.keys())}")
        print(f"Signal1 values: {signals['Signal1'][:3]}")
        assert len(timestamps) == 3
        assert 'Signal1' in signals and 'Signal2' in signals
        print("✓ CSV parsing test passed")
    except Exception as e:
        print(f"✗ Error parsing CSV: {e}")

    print("\n=== Testing recorder parsing ===")
    try:
        timestamps, signals = parse_recorder_format(open(test_recorder_path).read())
        print(f"Parsed {len(timestamps)} timestamps")
        print(f"Found signals: {list(signals.keys())}")
        print(f"Voltage values: {signals['Voltage'][:3]}")
        assert len(timestamps) == 3
        assert 'Voltage' in signals and 'Current' in signals
        print("✓ Recorder parsing test passed")
    except Exception as e:
        print(f"✗ Error parsing recorder file: {e}")

    print("\n=== Testing boolean CSV parsing ===")
    try:
        timestamps, signals = parse_csv_file(test_bool_csv_path)
        print(f"Parsed {len(timestamps)} timestamps")
        print(f"Found signals: {list(signals.keys())}")
        print(f"DigitalSignal1 values: {signals['DigitalSignal1'][:3]}")
        print(f"DigitalSignal2 values: {signals['DigitalSignal2'][:3]}")

        # Check that values are correctly converted and detected as digital
        assert len(timestamps) == 3
        assert 'DigitalSignal1' in signals
        assert 'DigitalSignal2' in signals
        assert signals['DigitalSignal1'][0] == 1.0
        assert signals['DigitalSignal2'][0] == 0.0
        assert is_digital_signal(signals['DigitalSignal1']) == True
        assert is_digital_signal(signals['DigitalSignal2']) == True
        print("✓ Boolean CSV parsing test passed")
    except Exception as e:
        print(f"✗ Error parsing boolean CSV: {e}")

    print("\n=== Testing parse_csv_or_recorder ===")
    try:
        # Test with CSV
        ts1, sig1 = parse_csv_or_recorder(test_csv_path)
        assert len(ts1) > 0 and len(sig1) > 0

        # Test with recorder
        ts2, sig2 = parse_csv_or_recorder(test_recorder_path)
        assert len(ts2) > 0 and len(sig2) > 0
        print("✓ parse_csv_or_recorder test passed")
    except Exception as e:
        print(f"✗ Error in parse_csv_or_recorder: {e}")

    # === Test export functions ===
    print("\n=== Testing automatic graph exports ===")

    # Setup temp directory for exports
    temp_dir = tempfile.mkdtemp()

    # Monkey patch QFileDialog to avoid manual file selection
    original_getSaveFileName = QFileDialog.getSaveFileName
    export_files = []


    def mock_getSaveFileName(*args, **kwargs):
        # In the first call (PNG), use PNG filter
        if not export_files:
            file_path = os.path.join(temp_dir, "test_graph.png")
            selected_filter = "PNG images (*.png)"
        # In the second call (PDF), use PDF filter
        elif len(export_files) == 1:
            file_path = os.path.join(temp_dir, "test_graph.pdf")
            selected_filter = "PDF documents (*.pdf)"
        # In the third call (SVG), use SVG filter
        else:
            file_path = os.path.join(temp_dir, "test_graph.svg")
            selected_filter = "SVG vector format (*.svg)"

        export_files.append(file_path)
        return file_path, selected_filter


    # Monkey patch QMessageBox to avoid dialog popups
    original_information = QMessageBox.information
    original_critical = QMessageBox.critical


    def mock_information(*args, **kwargs):
        print(f"INFO: {args[2] if len(args) > 2 else ''}")
        return QMessageBox.Ok


    def mock_critical(*args, **kwargs):
        print(f"ERROR: {args[2] if len(args) > 2 else ''}")
        return QMessageBox.Ok


    try:
        # Apply monkey patches
        QFileDialog.getSaveFileName = mock_getSaveFileName
        QMessageBox.information = mock_information
        QMessageBox.critical = mock_critical

        # Test PNG export
        print("\nTesting PNG export...")
        success = export_graph(plot, window)
        assert success == True, "PNG export failed"
        assert os.path.exists(export_files[-1]), f"PNG file not created at {export_files[-1]}"
        print(f"✓ PNG export test passed, file at {export_files[-1]}")

        # Test PDF export (will be skipped if QtPrintSupport not available)
        print("\nTesting PDF export...")
        try:
            from PySide6.QtPrintSupport import QPrinter

            success = export_graph(plot, window)
            assert success == True, "PDF export failed"
            assert os.path.exists(export_files[-1]), f"PDF file not created at {export_files[-1]}"
            print(f"✓ PDF export test passed, file at {export_files[-1]}")
        except ImportError:
            print("⚠ Skipping PDF export test: QtPrintSupport not available")

        # Test SVG export (will be skipped if QtSvg not available)
        print("\nTesting SVG export...")
        try:
            from PySide6.QtSvg import QSvgGenerator

            success = export_graph(plot, window)
            assert success == True, "SVG export failed"
            assert os.path.exists(export_files[-1]), f"SVG file not created at {export_files[-1]}"
            print(f"✓ SVG export test passed, file at {export_files[-1]}")
        except ImportError:
            print("⚠ Skipping SVG export test: QtSvg not available")

    finally:
        # Restore original functions
        QFileDialog.getSaveFileName = original_getSaveFileName
        QMessageBox.information = original_information
        QMessageBox.critical = original_critical

        # Clean up test files
        print("\n=== Cleaning up test files ===")
        try:
            os.unlink(test_csv_path)
            os.unlink(test_recorder_path)
            os.unlink(test_bool_csv_path)
            for file_path in export_files:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            os.rmdir(temp_dir)
            print("✓ All test files cleaned up")
        except Exception as e:
            print(f"⚠ Error during cleanup: {e}")

    print("\n=== Tests complete! ===")
    window.close()
    app.quit()
    sys.exit(0)

    sys.exit(app.exec())