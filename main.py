"""
Entry point for the CSV Signal Viewer application.
"""
import os
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ui.viewer import SignalViewer


def main():
    app = QApplication(sys.argv)

    icon_path = os.path.join(os.path.dirname(__file__), "_assets", "line-graph.ico")
    app.setWindowIcon(QIcon(icon_path))

    viewer = SignalViewer()
    viewer.show()

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("Application interrupted by user, exiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()
