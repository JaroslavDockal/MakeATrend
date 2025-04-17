"""
Entry point for the CSV Signal Viewer application.
"""
import sys
from PySide6.QtWidgets import QApplication
from viewer import SignalViewer


def main():
    app = QApplication(sys.argv)
    viewer = SignalViewer()
    viewer.show()

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("Application interrupted by user, exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
