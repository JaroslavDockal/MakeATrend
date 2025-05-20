"""
File loading and drag-drop operations for the CSV Signal Viewer.
"""
import os

from data.loader import load_multiple_files, load_single_file


def load_data(self, multiple=False):
    """
    Loads one or more data files depending on the 'multiple' flag.
    Updates data_signals as dict[name] = (time, values).
    """
    self.log_message(f"Loading {'multiple' if multiple else 'single'} data file(s)...", self.INFO)

    if multiple:
        signals = load_multiple_files()
    else:
        signals = load_single_file()

    if not signals:
        self.log_message("No data was loaded - user cancelled or file error", self.WARNING)
        return
    else:
        self.log_message(f"Successfully loaded {len(signals)} signals", self.DEBUG)

    self.data_signals = signals
    self.clear_signals()

    for name in signals:
        row = self.build_signal_row(name)
        self.scroll_layout.addWidget(row)
        self.log_message(f"Added signal: {name}", self.DEBUG)

def drag_enter_event(self, event):
    """
    Accept drag events if they contain files.
    """
    self.log_message(f"File drag detected with {len(event.mimeData().urls())} items", self.DEBUG)

    if event.mimeData().hasUrls():
        event.accept()
    else:
        event.ignore()

def drop_event(self, event):
    """
    Handle file drop events and load the dropped files.
    """
    files = [url.toLocalFile() for url in event.mimeData().urls()]
    if files:
        self.log_message(
            f"Files dropped: {[os.path.basename(url.toLocalFile()) for url in event.mimeData().urls()]}", self.INFO)
        self.load_dropped_files(files)

def load_dropped_files(self, files):
    """
    Load the dropped files into the application.
    """
    signals = load_multiple_files(files)
    if not signals:
        self.log_message("No valid signals found in dropped files", self.WARNING)
        return
    else:
        self.log_message(f"Successfully loaded {len(signals)} signals from dropped files", self.INFO)

    self.data_signals = signals
    self.clear_signals()

    for name in signals:
        row = self.build_signal_row(name)
        self.scroll_layout.addWidget(row)