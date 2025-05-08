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
    Inserts NaNs between time segments to prevent false interpolation.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: signal name -> (time, values)
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
                conflict = any(
                    t1[0] <= time_arr[-1] and t1[-1] >= time_arr[0]
                    for t1, _ in all_signals[name]
                )
                if conflict:
                    print(f"Warning: Signal '{name}' in {path} overlaps in time.")
                else:
                    all_signals[name].append((time_arr, values))

    result = {}

    for name, chunks in all_signals.items():
        times_with_nans = []
        values_with_nans = []

        for i, (t, v) in enumerate(chunks):
            times_with_nans.append(t)
            values_with_nans.append(v)

            # Add NaN gap unless it's the last chunk
            if i < len(chunks) - 1:
                gap_time = t[-1] + 1e-6  # tiny gap forward
                times_with_nans.append(np.array([gap_time]))
                values_with_nans.append(np.array([np.nan]))

        times = np.concatenate(times_with_nans)
        values = np.concatenate(values_with_nans)
        idx = np.argsort(times)

        result[name] = (times[idx], values[idx])

    return result
