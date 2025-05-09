"""
Entry point for the CSV Signal Viewer application.
"""
import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from viewer import SignalViewer


def main():
    app = QApplication(sys.argv)

    icon_path = os.path.join(os.path.dirname(__file__), "assets", "line-graph.ico")
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
