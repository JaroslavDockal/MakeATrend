# 📈 MakeATrend – CSV Signal Viewer

MakeATrend is a desktop application for visualizing and analyzing signal data from CSV files. It offers an interactive UI with features such as multiple Y-axes, crosshair and cursor tools, value comparisons, and export options.

## ✨ Features

- Load and plot signals from CSV with timestamp (Date + Time)
- Dual Y-axis support (left and right)
- Complicated mode for customizing signal color, axis, and line width
- Crosshair tool with timestamp display
- Two vertical cursors (A and B) with value delta and delta/s comparison
- Cursor Info window (dockable or floating)
- Signal label coloring matching line color
- Automatic unit detection from signal names (e.g. `Voltage [V]`)
- CSV export of cursor values
- Floating ☰ button to reopen hidden panel
- Toggle grid visibility
- Light/fast UI using PySide6 + PyQtGraph

## 📂 CSV Format

The CSV file must contain at least these columns:
- `Date` (in format `YYYY-MM-DD`)
- `Time` (in format `HH:MM:SS,fff`)
- Any number of signal columns with numeric values

Example:
```
Date;Time;Voltage [V];Current [A]
2024-01-01;12:00:00,000;230.5;12.3
2024-01-01;12:00:01,000;229.7;12.1
...
```

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
├── main.py               # Application entry point
├── viewer.py             # Main GUI and plotting logic
├── crosshair.py          # Custom crosshair overlay
├── cursor_info.py        # Floating/docked cursor info panel
├── utils.py              # CSV parsing and helper functions
├── assets/
│   └── line-graph.ico    # Window icon
└── README.md             # This file
```

## 🐞 Known Issues

- When the crosshair is enabled, moving the mouse over the chart may cause unintended zooming or panning behavior.
- Right-side Y-axis label in the crosshair currently mirrors the left-side value and does not reflect the actual signal values plotted on the right axis.

