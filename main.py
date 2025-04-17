"""Main entry point for the CSV Signal Viewer."""

from PySide6.QtWidgets import QApplication
from viewer import SignalViewer
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SignalViewer()
    viewer.show()
    sys.exit(app.exec())
