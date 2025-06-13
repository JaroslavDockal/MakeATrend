"""
Signal Analysis Module - Main entry point for advanced signal analysis tools.

This module integrates with the CSV Signal Viewer application to add statistical analysis,
FFT processing, and time domain analysis capabilities. It works with the application's
data structure where signals are stored as (time_array, values_array) tuples.

Usage:
    from signal_analysis import show_analysis_dialog

    # In your main application:
    def open_analysis_dialog(self):
        show_analysis_dialog(self)
"""

from utils.logger import Logger
from analysis.components.dialog import SignalAnalysisDialog


def show_analysis_dialog(parent):
    """
    Shows the signal analysis dialog with all the available analysis tools.

    This function serves as the main entry point for the signal analysis features.

    Args:
        parent: The parent application that has the data_signals attribute
               containing the signal data.
    """
    Logger.log_message_static("Analysis-Dialog: Opening Signal Analysis Dialog", Logger.INFO)
    try:
        dialog = SignalAnalysisDialog(parent)
        dialog.exec()
        Logger.log_message_static("Analysis-Dialog: Signal Analysis Dialog closed", Logger.INFO)
    except Exception as e:
        Logger.log_message_static(f"Analysis-Dialog: Error in signal analysis dialog: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Analysis-Dialog: Signal analysis dialog traceback: {traceback.format_exc()}",
                                  Logger.DEBUG)


def add_explanation_group(layout, title, text):
    """
    Add an expandable group with explanation text.

    Args:
        layout (QLayout): Layout to add the group to
        title (str): Title of the explanation group
        text (str): Explanation text
    """
    from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QTextEdit

    Logger.log_message_static(f"Analysis-Dialog: Adding explanation group: {title}", Logger.DEBUG)
    group = QGroupBox(title)
    group.setCheckable(True)
    group.setChecked(False)  # Start collapsed

    group_layout = QVBoxLayout(group)
    text_edit = QTextEdit()
    text_edit.setReadOnly(True)
    text_edit.setText(text)
    text_edit.setMinimumHeight(100)
    group_layout.addWidget(text_edit)

    layout.addWidget(group)
    Logger.log_message_static(f"Analysis-Dialog: Added explanation group for: {title}", Logger.DEBUG)