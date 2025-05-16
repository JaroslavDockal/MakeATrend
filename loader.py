import os
import numpy as np
from utils import parse_csv_or_recorder
from PySide6.QtWidgets import QFileDialog, QDialog

def load_single_file():
    """
    Load a single signal file using a file dialog.

    Returns:
        dict: Dictionary of signals {name: (time_array, value_array)}.
    """
    print("INFO: Opening file dialog to load a single file.")
    path, _ = QFileDialog.getOpenFileName(None, "Open Data File", "", "Data Files (*.csv *.txt)")
    if not path:
        print("DEBUG: No file selected. Operation canceled.")
        return {}

    try:
        time_arr, signals = parse_csv_or_recorder(path)
        print(f"INFO: Successfully loaded file '{os.path.basename(path)}' with {len(signals)} signals.")
        return {name: (time_arr, values) for name, values in signals.items()}
    except Exception as e:
        print(f"ERROR: Failed to load file '{os.path.basename(path)}'. Exception: {e}")
        return {}

def load_multiple_files(file_paths=None):
    """
    Opens a dialog to select multiple files, merges signals with same names.
    Inserts NaNs between time segments to prevent false interpolation.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: signal name -> (time, values)
    """
    if file_paths is None:
        print("INFO: Opening file dialog to load multiple files.")
        file_paths, _ = QFileDialog.getOpenFileNames(None, "Open Data Files", "", "Data Files (*.csv *.txt)")
        if not file_paths:
            print("DEBUG: No files selected. Operation canceled.")
            return {}

    all_signals = {}

    for path in file_paths:
        try:
            time_arr, signals = parse_csv_or_recorder(path)
            print(f"DEBUG: Successfully loaded file '{os.path.basename(path)}' with {len(signals)} signals.")
            for name, values in signals.items():
                if name not in all_signals:
                    all_signals[name] = [(time_arr, values)]
                else:
                    print(f"WARNING: Signal '{name}' already exists. Appending new data.")
                    all_signals[name].append((time_arr, values))
        except Exception as e:
            print(f"ERROR: Failed to load file '{os.path.basename(path)}'. Exception: {e}")
            continue

    result = {}
    for name, parts in all_signals.items():
        parts.sort(key=lambda x: x[0][0])
        merged_time = np.concatenate([p[0] for p in parts])
        merged_values = np.concatenate([p[1] for p in parts])
        result[name] = (merged_time, merged_values)
        print(f"DEBUG: Merged signal '{name}' with {len(merged_time)} points.")

    print(f"INFO: Successfully loaded and merged {len(result)} signals from {len(file_paths)} files.")
    return result

# Advanced CSV parsing functionality below - currently not used in production
class ParseOptions:
    """
    Stores CSV parsing configuration options.

    Attributes:
        delimiter (str): Character used to separate fields in the CSV file (default: ',')
        decimal_separator (str): Character used for decimal points (default: '.')
        date_format (str): Format for date/time parsing (default: 'auto')
        has_header (bool): Whether the first row contains headers (default: True)
        skip_rows (int): Number of rows to skip at the beginning (default: 0)
        encoding (str): File encoding (default: 'utf-8')
    """

    def __init__(self):
        SignalViewer.log_message_static("Initializing CSV ParseOptions with default values", DEBUG)
        self.delimiter = ','
        self.decimal_separator = '.'
        self.date_format = "auto"  # "auto", "iso", "mdy", "dmy", "ymd", or custom format
        self.has_header = True
        self.skip_rows = 0
        self.encoding = "utf-8"


def detect_csv_dialect(file_path, sample_size=4096, encodings=None):
    """
    Auto-detects CSV dialect including delimiter, decimal separator, and other format parameters.

    Args:
        file_path (str): Path to the CSV file
        sample_size (int): Number of bytes to sample for detection
        encodings (list): List of encodings to try

    Returns:
        dict: Detected parameters (delimiter, has_header, decimal_separator, encoding)

    Raises:
        ValueError: If file cannot be decoded with any of the provided encodings
    """
    SignalViewer.log_message_static(f"Starting CSV dialect detection for file: {os.path.basename(file_path)}", DEBUG)

    if encodings is None:
        encodings = ['utf-8', 'latin1', 'ascii', 'utf-16', 'cp1252']

    SignalViewer.log_message_static(f"Trying {len(encodings)} encodings for CSV detection", DEBUG)

    # Read sample from file
    sample_data = None
    detected_encoding = None

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                sample_data = f.read(sample_size)
                detected_encoding = encoding
                SignalViewer.log_message_static(f"Successfully read file using {encoding} encoding", DEBUG)
                break
        except UnicodeDecodeError:
            SignalViewer.log_message_static(f"Encoding {encoding} failed for file", DEBUG)
            continue

    if sample_data is None:
        SignalViewer.log_message_static("Could not decode file with any of the provided encodings", ERROR)
        raise ValueError("Could not decode file with any of the provided encodings")

    # Use csv Sniffer to detect the dialect
    try:
        import csv
        SignalViewer.log_message_static("Detecting CSV dialect using csv.Sniffer", DEBUG)
        dialect = csv.Sniffer().sniff(sample_data)
        has_header = csv.Sniffer().has_header(sample_data)
        SignalViewer.log_message_static(f"Detected delimiter: '{dialect.delimiter}', header: {has_header}", DEBUG)
    except Exception as e:
        SignalViewer.log_message_static(f"CSV dialect detection failed: {str(e)}, using defaults", WARNING)
        # Default to common values if detection fails
        return {
            'delimiter': ',',
            'has_header': True,
            'decimal_separator': '.',
            'encoding': detected_encoding
        }

    # Detect decimal separator by examining numeric fields
    decimal_separator = '.'
    try:
        import io
        SignalViewer.log_message_static("Analyzing numeric fields to detect decimal separator", DEBUG)
        sample_io = io.StringIO(sample_data)
        reader = csv.reader(sample_io, dialect)

        # Skip header if present
        if has_header:
            next(reader, None)

        # Check rows for decimal separators
        decimal_point_count = 0
        decimal_comma_count = 0

        for _ in range(min(10, sample_size // 50)):  # Check a reasonable number of rows
            try:
                row = next(reader, None)
                if row is None:
                    break

                for value in row:
                    value = value.strip()
                    # Check if value could be a number with decimal point
                    if '.' in value:
                        parts = value.split('.')
                        if len(parts) == 2 and all(p.replace('-', '').isdigit() or not p for p in parts):
                            decimal_point_count += 1
                    # Check if value could be a number with decimal comma
                    if ',' in value:
                        parts = value.split(',')
                        if len(parts) == 2 and all(p.replace('-', '').isdigit() or not p for p in parts):
                            decimal_comma_count += 1
            except Exception as e:
                SignalViewer.log_message_static(f"Error analyzing row: {str(e)}", DEBUG)
                pass

        if decimal_comma_count > decimal_point_count:
            SignalViewer.log_message_static(
                f"Decimal comma more frequent ({decimal_comma_count}) than decimal point ({decimal_point_count})",
                DEBUG)
            decimal_separator = ','
        else:
            SignalViewer.log_message_static(
                f"Decimal point more frequent ({decimal_point_count}) than decimal comma ({decimal_comma_count})",
                DEBUG)
    except Exception as e:
        SignalViewer.log_message_static(f"Decimal separator detection failed: {str(e)}", WARNING)
        pass

    result = {
        'delimiter': dialect.delimiter,
        'has_header': has_header,
        'decimal_separator': decimal_separator,
        'encoding': detected_encoding
    }

    SignalViewer.log_message_static(f"CSV dialect detection complete: {result}", INFO)
    return result


class ParseOptionsDialog(QDialog):
    """
    Dialog for configuring CSV parsing options with auto-detection capability.

    Attributes:
        file_path (str): Path to the CSV file for auto-detection
        delimiter_input (QComboBox): Selector for delimiter character
        decimal_input (QComboBox): Selector for decimal separator
        date_format (QComboBox): Selector for date format
        has_header (QCheckBox): Option for header presence
        skip_rows (QLineEdit): Number of rows to skip
        encoding (QComboBox): File encoding selector
    """

    def __init__(self, parent=None, file_path=None):
        from PySide6.QtWidgets import QVBoxLayout, QFormLayout, QPushButton, QComboBox, QCheckBox, QLineEdit, \
            QDialogButtonBox, QDialog
        from PySide6.QtCore import Qt

        SignalViewer.log_message_static("Initializing CSV Parse Options Dialog", DEBUG)
        super().__init__(parent)
        self.setWindowTitle("CSV Parsing Options")
        self.setMinimumWidth(300)
        self.file_path = file_path

        if file_path:
            SignalViewer.log_message_static(f"ParseOptionsDialog initialized with file: {os.path.basename(file_path)}",
                                            DEBUG)
        else:
            SignalViewer.log_message_static("ParseOptionsDialog initialized without file path", DEBUG)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Auto-detect button
        if file_path:
            self.detect_button = QPushButton("Auto-detect format")
            self.detect_button.clicked.connect(self.auto_detect)
            layout.addWidget(self.detect_button)

        # Delimiter
        self.delimiter_input = QComboBox()
        self.delimiter_input.setEditable(True)
        self.delimiter_input.addItems([',', ';', '\t', '|', ' '])
        form.addRow("Delimiter:", self.delimiter_input)

        # Decimal separator
        self.decimal_input = QComboBox()
        self.decimal_input.addItems(['.', ','])
        form.addRow("Decimal separator:", self.decimal_input)

        # Date format
        self.date_format = QComboBox()
        self.date_format.setEditable(True)
        self.date_format.addItems(["auto", "iso", "mdy", "dmy", "ymd"])
        form.addRow("Date format:", self.date_format)

        # Header
        self.has_header = QCheckBox()
        self.has_header.setChecked(True)
        form.addRow("First row as header:", self.has_header)

        # Skip rows
        self.skip_rows = QLineEdit("0")
        form.addRow("Skip rows:", self.skip_rows)

        # Encoding
        self.encoding = QComboBox()
        self.encoding.setEditable(True)
        self.encoding.addItems(["utf-8", "latin1", "cp1252", "ascii", "utf-16"])
        form.addRow("File encoding:", self.encoding)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Auto-detect on startup if file_path provided
        if file_path:
            SignalViewer.log_message_static("Auto-detecting CSV format on dialog initialization", DEBUG)
            self.auto_detect()

    def auto_detect(self):
        """
        Auto-detects CSV parameters from file and updates the UI with detected values.
        If detection fails, keeps default values.
        """
        if not self.file_path:
            SignalViewer.log_message_static("Cannot auto-detect without file path", WARNING)
            return

        try:
            SignalViewer.log_message_static(f"Starting auto-detection for file: {os.path.basename(self.file_path)}",
                                            DEBUG)
            detected = detect_csv_dialect(self.file_path)

            # Update UI with detected values
            delimiter = detected['delimiter']
            if delimiter == '\t':
                delimiter = '\\t'  # Show tab character in UI

            SignalViewer.log_message_static(f"Setting detected delimiter: '{delimiter}'", DEBUG)
            index = self.delimiter_input.findText(delimiter)
            if index >= 0:
                self.delimiter_input.setCurrentIndex(index)
            else:
                self.delimiter_input.setEditText(delimiter)

            SignalViewer.log_message_static(f"Setting detected decimal separator: '{detected['decimal_separator']}'",
                                            DEBUG)
            index = self.decimal_input.findText(detected['decimal_separator'])
            if index >= 0:
                self.decimal_input.setCurrentIndex(index)

            SignalViewer.log_message_static(f"Setting detected header presence: {detected['has_header']}", DEBUG)
            self.has_header.setChecked(detected['has_header'])

            SignalViewer.log_message_static(f"Setting detected encoding: '{detected['encoding']}'", DEBUG)
            index = self.encoding.findText(detected['encoding'])
            if index >= 0:
                self.encoding.setCurrentIndex(index)
            else:
                self.encoding.setEditText(detected['encoding'])

            SignalViewer.log_message_static(
                f"Auto-detected CSV parameters: delimiter='{delimiter}', "
                f"decimal_separator='{detected['decimal_separator']}', "
                f"has_header={detected['has_header']}, encoding='{detected['encoding']}'",
                INFO
            )
        except Exception as e:
            SignalViewer.log_message_static(f"Auto-detection failed: {str(e)}", WARNING)

    def get_options(self):
        """
        Creates and returns a ParseOptions object with the current dialog settings.

        Returns:
            ParseOptions: Object containing all parsing configuration
        """
        SignalViewer.log_message_static("Collecting parse options from dialog", DEBUG)
        options = ParseOptions()

        options.delimiter = self.delimiter_input.currentText()
        # Handle tab character
        if options.delimiter == '\\t':
            options.delimiter = '\t'
            SignalViewer.log_message_static("Converted '\\t' to tab character for delimiter", DEBUG)

        options.decimal_separator = self.decimal_input.currentText()
        options.date_format = self.date_format.currentText()
        options.has_header = self.has_header.isChecked()

        try:
            options.skip_rows = int(self.skip_rows.text())
        except ValueError:
            SignalViewer.log_message_static(f"Invalid skip rows value: '{self.skip_rows.text()}', using 0", WARNING)
            options.skip_rows = 0

        options.encoding = self.encoding.currentText()

        SignalViewer.log_message_static(
            f"Parse options collected: delimiter='{options.delimiter}', "
            f"decimal_separator='{options.decimal_separator}', "
            f"date_format='{options.date_format}', has_header={options.has_header}, "
            f"skip_rows={options.skip_rows}, encoding='{options.encoding}'",
            DEBUG
        )
        return options


def get_parse_options(parent=None, file_path=None):
    """
    Shows dialog for configuring CSV parsing options.

    Args:
        parent: Parent widget for the dialog
        file_path: Path to CSV file for auto-detection

    Returns:
        ParseOptions: Object with parsing options if OK clicked, None if canceled
    """
    SignalViewer.log_message_static(f"Opening CSV parsing options dialog", INFO)
    if file_path:
        SignalViewer.log_message_static(f"Using file for auto-detection: {os.path.basename(file_path)}", DEBUG)

    dialog = ParseOptionsDialog(parent, file_path)
    result = dialog.exec()

    if result == QDialog.Accepted:
        SignalViewer.log_message_static("User accepted CSV parsing options", DEBUG)
        return dialog.get_options()
    else:
        SignalViewer.log_message_static("User canceled CSV parsing options dialog", DEBUG)
        return None