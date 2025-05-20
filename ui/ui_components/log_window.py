"""
Log window implementation for the CSV Signal Viewer.
"""
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QCheckBox, QPushButton


class LogWindow(QDialog):
    """
    A window to display log messages with different severity levels:
    DEBUG (0), INFO (1), WARNING (2), ERROR (3)
    """
    # Log levels
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Log Window")
        self.resize(600, 400)

        self.layout = QVBoxLayout(self)
        self.log_view = QTextEdit(self)
        self.log_view.setReadOnly(True)
        self.layout.addWidget(self.log_view)

        # Layout for checkboxes
        control_layout = QHBoxLayout()

        self.debug_checkbox = QCheckBox("Show Debug", self)
        self.debug_checkbox.setChecked(True)
        self.debug_checkbox.setToolTip("Show detailed debug messages")
        control_layout.addWidget(self.debug_checkbox)

        self.autoscroll_checkbox = QCheckBox("Autoscroll", self)
        self.autoscroll_checkbox.setChecked(True)
        self.autoscroll_checkbox.setToolTip("Automatically scroll to newest messages")
        control_layout.addWidget(self.autoscroll_checkbox)

        clear_button = QPushButton("Clear", self)
        clear_button.setToolTip("Clear all log messages")
        clear_button.clicked.connect(self.clear_log)
        control_layout.addWidget(clear_button)

        self.layout.addLayout(control_layout)

    def add_message(self, message, level=INFO):
        """
        Add a message to the log view.

        Args:
            message (str): The message to add.
            level (int): Message level (DEBUG=0, INFO=1, WARNING=2, ERROR=3)
        """
        # Skip debug messages if debug checkbox is not checked
        if level == LogWindow.DEBUG and not self.debug_checkbox.isChecked():
            return

        # Apply appropriate styling based on message level
        if level == LogWindow.ERROR:
            html = f'<span style="color:#ff5050;font-weight:bold;">{message}</span>'
        elif level == LogWindow.WARNING:
            html = f'<span style="color:#ffcc00;font-weight:bold;">{message}</span>'
        elif level == LogWindow.INFO:
            html = f'<span style="color:white;">{message}</span>'
        else:  # DEBUG
            html = f'<span style="color:#808080;">{message}</span>'

        self.log_view.append(html)

        if self.autoscroll_checkbox.isChecked():
            scrollbar = self.log_view.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        """
        Clears all log messages from the log view.
        """
        self.log_view.clear()