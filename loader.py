# loader.py
from utils import parse_csv_or_recorder
import numpy as np
from PySide6.QtWidgets import QFileDialog

def load_single_file():
    """
    Opens a file dialog, parses a selected file and returns time and signals.

    Returns:
        tuple[np.ndarray, dict]: time array, signals dictionary
    """
    path, _ = QFileDialog.getOpenFileName(None, "Open Data File", "", "Data Files (*.csv *.txt)")
    if not path:
        return {}

    try:
        time_arr, signals = parse_csv_or_recorder(path)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return {}

    return {name: (time_arr, values) for name, values in signals.items()}

def load_multiple_files(file_paths=None):
    """
    Opens a dialog to select multiple files, merges signals with same names.
    Inserts NaNs between time segments to prevent false interpolation.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: signal name -> (time, values)
    """
    if file_paths is None:
        file_paths, _ = QFileDialog.getOpenFileNames(None, "Open Data Files", "", "Data Files (*.csv *.txt)")
        if not file_paths:
            return {}

    all_signals = {}

    for path in file_paths:
        try:
            time_arr, signals = parse_csv_or_recorder(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

        for name, values in signals.items():
            if name not in all_signals:
                all_signals[name] = [(time_arr, values)]
            else:
                conflict = any(
                    t1[0] <= time_arr[-1] and t1[-1] >= time_arr[0]
                    for t1, _ in all_signals[name]
                )
                if conflict:
                    print(f"Warning: Signal '{name}' in {path} overlaps in time.")
                else:
                    all_signals[name].append((time_arr, values))

    result = {}

    return result

#TODO Vsechno dole je jen nachystane, ale zatim nepouzivane
# class ParseOptions:
#     """
#     Stores CSV parsing configuration options.
#
#     Attributes:
#         delimiter (str): Character used to separate fields in the CSV file (default: ',')
#         decimal_separator (str): Character used for decimal points (default: '.')
#         date_format (str): Format for date/time parsing (default: 'auto')
#         has_header (bool): Whether the first row contains headers (default: True)
#         skip_rows (int): Number of rows to skip at the beginning (default: 0)
#         encoding (str): File encoding (default: 'utf-8')
#     """
#
#     def __init__(self):
#         self.delimiter = ','
#         self.decimal_separator = '.'
#         self.date_format = "auto"  # "auto", "iso", "mdy", "dmy", "ymd", or custom format
#         self.has_header = True
#         self.skip_rows = 0
#         self.encoding = "utf-8"
#
#
# def detect_csv_dialect(file_path, sample_size=4096, encodings=None):
#     """
#     Auto-detects CSV dialect including delimiter, decimal separator, and other format parameters.
#
#     Args:
#         file_path (str): Path to the CSV file
#         sample_size (int): Number of bytes to sample for detection
#         encodings (list): List of encodings to try
#
#     Returns:
#         dict: Detected parameters (delimiter, has_header, decimal_separator, encoding)
#
#     Raises:
#         ValueError: If file cannot be decoded with any of the provided encodings
#     """
#     if encodings is None:
#         encodings = ['utf-8', 'latin1', 'ascii', 'utf-16', 'cp1252']
#
#     # Read sample from file
#     sample_data = None
#     detected_encoding = None
#
#     for encoding in encodings:
#         try:
#             with open(file_path, 'r', encoding=encoding) as f:
#                 sample_data = f.read(sample_size)
#                 detected_encoding = encoding
#                 break
#         except UnicodeDecodeError:
#             continue
#
#     if sample_data is None:
#         raise ValueError("Could not decode file with any of the provided encodings")
#
#     # Use csv Sniffer to detect the dialect
#     try:
#         dialect = csv.Sniffer().sniff(sample_data)
#         has_header = csv.Sniffer().has_header(sample_data)
#     except:
#         # Default to common values if detection fails
#         return {
#             'delimiter': ',',
#             'has_header': True,
#             'decimal_separator': '.',
#             'encoding': detected_encoding
#         }
#
#     # Detect decimal separator by examining numeric fields
#     decimal_separator = '.'
#     try:
#         sample_io = io.StringIO(sample_data)
#         reader = csv.reader(sample_io, dialect)
#
#         # Skip header if present
#         if has_header:
#             next(reader, None)
#
#         # Check rows for decimal separators
#         decimal_point_count = 0
#         decimal_comma_count = 0
#
#         for _ in range(min(10, sample_size // 50)):  # Check a reasonable number of rows
#             try:
#                 row = next(reader, None)
#                 if row is None:
#                     break
#
#                 for value in row:
#                     value = value.strip()
#                     # Check if value could be a number with decimal point
#                     if '.' in value:
#                         parts = value.split('.')
#                         if len(parts) == 2 and all(p.replace('-', '').isdigit() or not p for p in parts):
#                             decimal_point_count += 1
#                     # Check if value could be a number with decimal comma
#                     if ',' in value:
#                         parts = value.split(',')
#                         if len(parts) == 2 and all(p.replace('-', '').isdigit() or not p for p in parts):
#                             decimal_comma_count += 1
#             except:
#                 pass
#
#         if decimal_comma_count > decimal_point_count:
#             decimal_separator = ','
#     except:
#         pass
#
#     return {
#         'delimiter': dialect.delimiter,
#         'has_header': has_header,
#         'decimal_separator': decimal_separator,
#         'encoding': detected_encoding
#     }
#
#
# class ParseOptionsDialog(QDialog):
#     """
#     Dialog for configuring CSV parsing options with auto-detection capability.
#
#     Attributes:
#         file_path (str): Path to the CSV file for auto-detection
#         delimiter_input (QComboBox): Selector for delimiter character
#         decimal_input (QComboBox): Selector for decimal separator
#         date_format (QComboBox): Selector for date format
#         has_header (QCheckBox): Option for header presence
#         skip_rows (QLineEdit): Number of rows to skip
#         encoding (QComboBox): File encoding selector
#     """
#
#     def __init__(self, parent=None, file_path=None):
#         super().__init__(parent)
#         self.setWindowTitle("CSV Parsing Options")
#         self.setMinimumWidth(300)
#         self.file_path = file_path
#
#         layout = QVBoxLayout(self)
#         form = QFormLayout()
#
#         # Auto-detect button
#         if file_path:
#             self.detect_button = QPushButton("Auto-detect format")
#             self.detect_button.clicked.connect(self.auto_detect)
#             layout.addWidget(self.detect_button)
#
#         # Delimiter
#         self.delimiter_input = QComboBox()
#         self.delimiter_input.setEditable(True)
#         self.delimiter_input.addItems([',', ';', '\t', '|', ' '])
#         form.addRow("Delimiter:", self.delimiter_input)
#
#         # Decimal separator
#         self.decimal_input = QComboBox()
#         self.decimal_input.addItems(['.', ','])
#         form.addRow("Decimal separator:", self.decimal_input)
#
#         # Date format
#         self.date_format = QComboBox()
#         self.date_format.setEditable(True)
#         self.date_format.addItems(["auto", "iso", "mdy", "dmy", "ymd"])
#         form.addRow("Date format:", self.date_format)
#
#         # Header
#         self.has_header = QCheckBox()
#         self.has_header.setChecked(True)
#         form.addRow("First row as header:", self.has_header)
#
#         # Skip rows
#         self.skip_rows = QLineEdit("0")
#         form.addRow("Skip rows:", self.skip_rows)
#
#         # Encoding
#         self.encoding = QComboBox()
#         self.encoding.setEditable(True)
#         self.encoding.addItems(["utf-8", "latin1", "cp1252", "ascii", "utf-16"])
#         form.addRow("File encoding:", self.encoding)
#
#         layout.addLayout(form)
#
#         # Buttons
#         buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
#         buttons.accepted.connect(self.accept)
#         buttons.rejected.connect(self.reject)
#         layout.addWidget(buttons)
#
#         # Auto-detect on startup if file_path provided
#         if file_path:
#             self.auto_detect()
#
#     def auto_detect(self):
#         """
#         Auto-detects CSV parameters from file and updates the UI with detected values.
#         If detection fails, keeps default values.
#         """
#         if not self.file_path:
#             return
#
#         try:
#             detected = detect_csv_dialect(self.file_path)
#
#             # Update UI with detected values
#             delimiter = detected['delimiter']
#             if delimiter == '\t':
#                 delimiter = '\\t'  # Show tab character in UI
#
#             index = self.delimiter_input.findText(delimiter)
#             if index >= 0:
#                 self.delimiter_input.setCurrentIndex(index)
#             else:
#                 self.delimiter_input.setEditText(delimiter)
#
#             index = self.decimal_input.findText(detected['decimal_separator'])
#             if index >= 0:
#                 self.decimal_input.setCurrentIndex(index)
#
#             self.has_header.setChecked(detected['has_header'])
#
#             index = self.encoding.findText(detected['encoding'])
#             if index >= 0:
#                 self.encoding.setCurrentIndex(index)
#             else:
#                 self.encoding.setEditText(detected['encoding'])
#
#             print(f"Auto-detected: delimiter='{delimiter}', decimal_separator='{detected['decimal_separator']}', "
#                   f"has_header={detected['has_header']}, encoding='{detected['encoding']}'")
#         except Exception as e:
#             print(f"Auto-detection failed: {e}")
#
#     def get_options(self):
#         """
#         Creates and returns a ParseOptions object with the current dialog settings.
#
#         Returns:
#             ParseOptions: Object containing all parsing configuration
#         """
#         options = ParseOptions()
#         options.delimiter = self.delimiter_input.currentText()
#         # Handle tab character
#         if options.delimiter == '\\t':
#             options.delimiter = '\t'
#
#         options.decimal_separator = self.decimal_input.currentText()
#         options.date_format = self.date_format.currentText()
#         options.has_header = self.has_header.isChecked()
#         options.skip_rows = int(self.skip_rows.text())
#         options.encoding = self.encoding.currentText()
#         return options
#
#
# def get_parse_options(parent=None, file_path=None):
#     """
#     Shows dialog for configuring CSV parsing options.
#
#     Args:
#         parent: Parent widget for the dialog
#         file_path: Path to CSV file for auto-detection
#
#     Returns:
#         ParseOptions: Object with parsing options if OK clicked, None if canceled
#     """
#     dialog = ParseOptionsDialog(parent, file_path)
#     result = dialog.exec()
#     if result == QDialog.Accepted:
#         return dialog.get_options()
#     return None