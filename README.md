# 📈 MakeATrend – CSV Signal Viewer

MakeATrend is a desktop application for visualizing and analyzing signal data from CSV files. It offers an interactive UI with features such as multiple Y-axes, crosshair and cursor tools, value comparisons, and export options.

## ✨ Features

- Load and plot signals from CSV and Drive Debug text files with timestamp.
- Load and merge data from multiple files
- Dual Y-axis support (left and right)
- Filter signals by name within the control panel
- Configurable signal styling in **Advanced Mode** (custom color, axis assignment, line width)
- Create **Virtual Signals** by defining mathematical expressions based on existing signals
- Expression validation for virtual signals
- Interactive Crosshair tool with timestamp and Y-value display
- Two vertical cursors (A and B) for selecting points of interest
- Dockable/Floating Cursor Info panel showing signal values, difference (Δ), and delta per second (Δ/s) at cursor positions.
- Automatic unit detection from signal names and display in cursor info
- Floating `☰` button to easily show/hide the control panel
- Signal label coloring matching line color
- CSV export of cursor values
- Toggle grid visibility
- A set of optimized signal colors for clear visualization on dark backgrounds
- Export graph view to PNG, PDF, and SVG formats
- Light/fast UI using PySide6 + PyQtGraph

## 📂 CSV Format

The CSV file must contain at least these columns:
- `Date` (in format `YYYY-MM-DD`)
- `Time` (in format `HH:MM:SS,fff`)
- Any number of signal columns with numeric values

Examples:
```
Date;Time;Voltage [V];Current [A]
2024-01-01;12:00:00,000;230.5;12.3
2024-01-01;12:00:01,000;229.7;12.1
...
```
```
Date;Time;CB Closed [-];CB Open [-]
2024-01-01;12:00:00,000;FALSE;TRUE
2024-01-01;12:00:01,000;TRUE;FALSE
...
```

*Note: The application also supports a `Drive Debug` text format (detected automatically based on content).

## 🛠 Requirements

Tested with Python 3.11.1 and the following packages:

```
altgraph==0.17.4
numpy==2.2.4
packaging==24.2
pandas==2.2.3
pefile==2023.2.7
pyinstaller==6.13.0
pyinstaller-hooks-contrib==2025.3
pyqtgraph==0.13.7
PySide6==6.9.0
PySide6_Addons==6.9.0
PySide6_Essentials==6.9.0
python-dateutil==2.9.0.post0
pytz==2025.2
pywin32-ctypes==0.2.3
shiboken6==6.9.0
six==1.17.0
tzdata==2025.2
```

To install all dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Run the App

```bash
python main.py
```

Or compile to a standalone `.exe` using PyInstaller:

```bash
pyinstaller --noconfirm main.py
```

## 🗂 File Structure

```
MakeATrend/
├── main.py                    # Application entry point
├── viewer.py                  # Main GUI and plotting logic
├── custom_viewbox.py          # Custom pg.ViewBox implementation for specific behaviors like zoom locking
├── crosshair.py               # Custom crosshair overlay
├── cursor_info.py             # Floating/docked cursor info panel
├── utils.py                   # CSV parsing and helper functions
├── loader.py                  # File loading (single/multiple) and initial data structuring logic
├── signal_colors.py           # Defines the color palette used for signals [21, 35]
├── virtual_signal_dialog.py   # Dialog and validation logic for creating virtual signals from expressions
├── assets/                    # Directory for application assets
│   └── line-graph.ico         # Window icon
├── tests/                     # Directory for modules testing
│   └── *_test.py              # Simple tests for each module
└── README.md                  # This file
```
## ⚙️ Internal Architecture

The application is built upon the **PySide6** GUI framework and leverages **pyqtgraph** for high-performance data plotting.

Key components and their roles:
*   `main.py`: The entry point of the application, initializes the QApplication and the main viewer window.
*   `viewer.py`: Implements the main `QMainWindow`, `SignalViewer`, which orchestrates the entire application. It contains the `pg.PlotWidget`, manages multiple `pg.ViewBox` instances for the different axes (Left, Right, Digital), handles the UI elements in the control panel (signal checkboxes, filter, buttons), integrates the cursor lines (`pg.InfiniteLine`), the crosshair (`Crosshair`), and the cursor info dialog (`CursorInfoDialog`). It also manages the state of loaded signals, plotted curves, and styling options.
*   `custom_viewbox.py`: A subclass of `pg.ViewBox` used by the main plot. It overrides the `wheelEvent` to allow locking the mouse wheel zoom functionality, primarily used in conjunction with the crosshair.
*   `crosshair.py`: A class (`Crosshair`) that adds an interactive crosshair and labels to a given `pg.ViewBox`. It uses `pg.SignalProxy` to track mouse movement events and updates the position and labels of the crosshair lines accordingly.
*   `cursor_info.py`: Implements the `CursorInfoDialog`, a `QDialog` (or potentially a `QWidget` when docked) that displays detailed numerical information about the signal values at the current cursor positions in a `QTableWidget`. It handles unit extraction, boolean value display, and calculation of differences and delta per second.
*   `loader.py`: Contains functions (`load_single_file`, `load_multiple_files`) responsible for opening file dialogs, calling the appropriate parsing functions from `utils.py`, and structuring the loaded data. It includes logic for merging data from multiple files.
*   `utils.py`: A utility module containing several independent functions:
    *   **Data Parsing:** `parse_csv_or_recorder` acts as a dispatcher to `parse_csv_file` or `parse_recorder_format` based on file content. `parse_csv_file` reads standard CSV, handling date/time timestamps and both numeric and boolean ('TRUE'/'FALSE') values. `parse_recorder_format` specifically parses the proprietary text format.
    *   **Data Utilities:** `find_nearest_index` finds the index closest to a given value in an array. `is_digital_signal` attempts to detect if a signal is boolean-like based on its unique values.
    *   **Graph Export:** `export_graph` attempts to render the plot widget to a `QPixmap` or directly to a `QPrinter`/`QSvgGenerator` for PNG, PDF, or SVG export. `export_graph_fallback` provides an alternative export method using `pyqtgraph.exporters.ImageExporter` if available, falling back to `export_graph` if not.
*   `signal_colors.py`: Defines the `SignalColors` class, providing a static list of visually distinct colors. It includes methods to retrieve colors cyclically by index (`get_color`) or consistently based on a signal name's hash (`get_color_for_name`). This class is used by the `SignalViewer` when plotting signals.
*   `virtual_signal_dialog.py`: Implements the `VirtualSignalDialog` and associated validation functions (`validate_expression`, `validate_signal_name`). It uses Python's `ast` module to safely parse and validate expressions before they are potentially evaluated.


## 🐞 Known Issues

- When the crosshair is enabled, moving the mouse over the chart may cause unintended zooming or panning behavior.
- Right-side Y-axis label in the crosshair currently mirrors the left-side value and does not reflect the actual signal values plotted on the right axis.

## 💡 Planned Improvement
Future improvements are planned in several areas:
- **Robust Error Handling and Reporting:** Replacing generic exception handling with specific exceptions and providing more detailed, user-friendly error messages.
- **Enhanced Data Parsing:** Implementing more flexible CSV parsing options (delimiters, decimal separators, date/time formats) and improving resilience of the proprietary format parser.
- **Safer and More Powerful Virtual Signals:** Exploring alternatives to eval for safer expression evaluation, expanding allowed mathematical functions (e.g., from NumPy), allowing multi-character aliases, and improving type checking.
- **Improved Performance for Large Datasets:** Investigating pyqtgraph features like downsampling or OpenGL acceleration, and optimizing data loading and merging processes.
- **Enhanced UI and User Experience:** Implementing functionality to save and load project states (loaded files, signal configurations, cursor positions), adding more signal styling options (line styles, markers), refining zoom/pan interactions, adding drag-and-drop file loading, and providing tooltips.
- **Sophisticated Data Handling:** Offering explicit options for handling time overlaps when merging data from multiple files and implementing more comprehensive unit handling, including potential conversions or validation.
- **More Robust Export:** Further investigating and improving the reliability of Qt-based export methods or finding alternative libraries for graph export.
- **Documentation and Testing:** Improving code documentation, adding more unit tests, and potentially creating a user manual or tutorial for new users.
- **Advanced data analysis features:** Implementing advanced data analysis features, such as statistical analysis, filtering, FFT, or other signal processing techniques.