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

def load_multiple_files():
    """
    Opens a dialog to select multiple files, merges signals with same names.

    Returns:
        tuple[np.ndarray, dict]: time array (first available), signals dictionary
    """
    paths, _ = QFileDialog.getOpenFileNames(None, "Open Data Files", "", "Data Files (*.csv *.txt)")
    if not paths:
        return {}

    all_signals = {}

    for path in paths:
        try:
            time_arr, signals = parse_csv_or_recorder(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

        for name, values in signals.items():
            if name not in all_signals:
                all_signals[name] = [(time_arr, values)]
            else:
                conflict = any(t1[0] <= time_arr[-1] and t1[-1] >= time_arr[0] for t1, _ in all_signals[name])
                if conflict:
                    print(f"Warning: Signal '{name}' in {path} overlaps in time.")
                else:
                    all_signals[name].append((time_arr, values))

    result = {}
    for name, chunks in all_signals.items():
        times = np.concatenate([t for t, _ in chunks])
        values = np.concatenate([v for _, v in chunks])
        idx = np.argsort(times)
        result[name] = (times[idx], values[idx])

    return result
