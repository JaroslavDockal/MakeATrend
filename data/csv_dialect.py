"""
CSV Dialect Detection Module

This module provides functionality for automatically detecting and configuring CSV file
format parameters, including:
- Field delimiters (comma, semicolon, tab, etc.)
- Decimal separators (point or comma)
- Header presence
- File encoding
- Date/time formatting

It offers both automatic detection capabilities through sampling and analysis,
as well as a user interface for manual configuration through the ParseOptionsDialog.
The module helps ensure correct parsing of CSV files with different regional formats
and conventions.

Classes:
    ParseOptions: Container for CSV parsing configuration
    ParseOptionsDialog: UI dialog for configuring parsing options

Functions:
    detect_csv_dialect: Auto-detects CSV format parameters from file content
    get_parse_options: Shows dialog for configuring CSV parsing options
"""

import os
import csv
import io

from PySide6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QPushButton, QComboBox
from PySide6.QtWidgets import QCheckBox, QLineEdit, QDialogButtonBox

from utils.logger import Logger

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
    Logger.log_message_static(f"Starting CSV dialect detection for file: {os.path.basename(file_path)}", Logger.DEBUG)

    if encodings is None:
        encodings = ['utf-8', 'latin1', 'ascii', 'utf-16', 'cp1252']

    Logger.log_message_static(f"Trying {len(encodings)} encodings for CSV detection", Logger.DEBUG)

    # Read sample from file
    sample_data = None
    detected_encoding = None

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                sample_data = f.read(sample_size)
                detected_encoding = encoding
                Logger.log_message_static(f"Successfully read file using {encoding} encoding", Logger.DEBUG)
                break
        except UnicodeDecodeError:
            Logger.log_message_static(f"Encoding {encoding} failed for file", Logger.DEBUG)
            continue

    if sample_data is None:
        Logger.log_message_static("Could not decode file with any of the provided encodings", Logger.ERROR)
        raise ValueError("Could not decode file with any of the provided encodings")

    # Use csv Sniffer to detect the dialect
    try:
        import csv
        Logger.log_message_static("Detecting CSV dialect using csv.Sniffer", Logger.DEBUG)
        dialect = csv.Sniffer().sniff(sample_data)
        has_header = csv.Sniffer().has_header(sample_data)
        Logger.log_message_static(f"Detected delimiter: '{dialect.delimiter}', header: {has_header}", Logger.DEBUG)
    except Exception as e:
        Logger.log_message_static(f"CSV dialect detection failed: {str(e)}, using defaults", Logger.WARNING)
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
        Logger.log_message_static("Analyzing numeric fields to detect decimal separator", Logger.DEBUG)
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
                Logger.log_message_static(f"Error analyzing row: {str(e)}", Logger.DEBUG)
                pass

        if decimal_comma_count > decimal_point_count:
            Logger.log_message_static(
                f"Decimal comma more frequent ({decimal_comma_count}) than decimal point ({decimal_point_count})",
                Logger.DEBUG)
            decimal_separator = ','
        else:
            Logger.log_message_static(
                f"Decimal point more frequent ({decimal_point_count}) than decimal comma ({decimal_comma_count})",
                Logger.DEBUG)
    except Exception as e:
        Logger.log_message_static(f"Decimal separator detection failed: {str(e)}", Logger.WARNING)
        pass

    result = {
        'delimiter': dialect.delimiter,
        'has_header': has_header,
        'decimal_separator': decimal_separator,
        'encoding': detected_encoding
    }

    Logger.log_message_static(f"CSV dialect detection complete: {result}", Logger.INFO)
    return result

def get_parse_options(parent=None, file_path=None):
    """
    Shows dialog for configuring CSV parsing options.

    Args:
        parent: Parent widget for the dialog
        file_path: Path to CSV file for auto-detection

    Returns:
        ParseOptions: Object with parsing options if OK clicked, None if canceled
    """
    Logger.log_message_static(f"Opening CSV parsing options dialog", Logger.INFO)
    if file_path:
        Logger.log_message_static(f"Using file for auto-detection: {os.path.basename(file_path)}", Logger.DEBUG)

    dialog = ParseOptionsDialog(parent, file_path)
    result = dialog.exec()

    if result == QDialog.Accepted:
        Logger.log_message_static("User accepted CSV parsing options", Logger.DEBUG)
        return dialog.get_options()
    else:
        Logger.log_message_static("User canceled CSV parsing options dialog", Logger.DEBUG)
        return None

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
        Logger.log_message_static("Initializing CSV ParseOptions with default values", Logger.DEBUG)
        self.delimiter = ','
        self.decimal_separator = '.'
        self.date_format = "auto"  # "auto", "iso", "mdy", "dmy", "ymd", or custom format
        self.has_header = True
        self.skip_rows = 0
        self.encoding = "utf-8"

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
            QDialogButtonBox

        Logger.log_message_static("Initializing CSV Parse Options Dialog", Logger.DEBUG)
        super().__init__(parent)
        self.setWindowTitle("CSV Parsing Options")
        self.setMinimumWidth(300)
        self.file_path = file_path

        if file_path:
            Logger.log_message_static(f"ParseOptionsDialog initialized with file: {os.path.basename(file_path)}",
                                            Logger.DEBUG)
        else:
            Logger.log_message_static("ParseOptionsDialog initialized without file path", Logger.DEBUG)

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
            Logger.log_message_static("Auto-detecting CSV format on dialog initialization", Logger.DEBUG)
            self.auto_detect()

    def auto_detect(self):
        """
        Auto-detects CSV parameters from file and updates the UI with detected values.
        If detection fails, keeps default values.
        """
        if not self.file_path:
            Logger.log_message_static("Cannot auto-detect without file path", Logger.WARNING)
            return

        try:
            Logger.log_message_static(f"Starting auto-detection for file: {os.path.basename(self.file_path)}",
                                            Logger.DEBUG)
            detected = detect_csv_dialect(self.file_path)

            # Update UI with detected values
            delimiter = detected['delimiter']
            if delimiter == '\t':
                delimiter = '\\t'  # Show tab character in UI

            Logger.log_message_static(f"Setting detected delimiter: '{delimiter}'", Logger.DEBUG)
            index = self.delimiter_input.findText(delimiter)
            if index >= 0:
                self.delimiter_input.setCurrentIndex(index)
            else:
                self.delimiter_input.setEditText(delimiter)

            Logger.log_message_static(f"Setting detected decimal separator: '{detected['decimal_separator']}'",
                                            Logger.DEBUG)
            index = self.decimal_input.findText(detected['decimal_separator'])
            if index >= 0:
                self.decimal_input.setCurrentIndex(index)

            Logger.log_message_static(f"Setting detected header presence: {detected['has_header']}", Logger.DEBUG)
            self.has_header.setChecked(detected['has_header'])

            Logger.log_message_static(f"Setting detected encoding: '{detected['encoding']}'", Logger.DEBUG)
            index = self.encoding.findText(detected['encoding'])
            if index >= 0:
                self.encoding.setCurrentIndex(index)
            else:
                self.encoding.setEditText(detected['encoding'])

            Logger.log_message_static(
                f"Auto-detected CSV parameters: delimiter='{delimiter}', "
                f"decimal_separator='{detected['decimal_separator']}', "
                f"has_header={detected['has_header']}, encoding='{detected['encoding']}'",
                Logger.INFO
            )
        except Exception as e:
            Logger.log_message_static(f"Auto-detection failed: {str(e)}", Logger.WARNING)

    def get_options(self):
        """
        Creates and returns a ParseOptions object with the current dialog settings.

        Returns:
            ParseOptions: Object containing all parsing configuration
        """
        Logger.log_message_static("Collecting parse options from dialog", Logger.DEBUG)
        options = ParseOptions()

        options.delimiter = self.delimiter_input.currentText()
        # Handle tab character
        if options.delimiter == '\\t':
            options.delimiter = '\t'
            Logger.log_message_static("Converted '\\t' to tab character for delimiter", Logger.DEBUG)

        options.decimal_separator = self.decimal_input.currentText()
        options.date_format = self.date_format.currentText()
        options.has_header = self.has_header.isChecked()

        try:
            options.skip_rows = int(self.skip_rows.text())
        except ValueError:
            Logger.log_message_static(f"Invalid skip rows value: '{self.skip_rows.text()}', using 0", Logger.WARNING)
            options.skip_rows = 0

        options.encoding = self.encoding.currentText()

        Logger.log_message_static(
            f"Parse options collected: delimiter='{options.delimiter}', "
            f"decimal_separator='{options.decimal_separator}', "
            f"date_format='{options.date_format}', has_header={options.has_header}, "
            f"skip_rows={options.skip_rows}, encoding='{options.encoding}'",
            Logger.DEBUG
        )
        return options
