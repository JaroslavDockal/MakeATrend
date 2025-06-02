"""
UI package for MakeATrend application.

This package provides the graphical user interface components for the MakeATrend
application, including the main viewer window and all supporting UI elements.
"""

# Import main components for direct package access
from .viewer import SignalViewer

# Import UI component setup functions that might be needed by external modules
from .ui_components.control_panel import setup_control_panel
from .ui_components.plot_area import setup_plot_area, setup_axes

# Import widgets for direct access
from .widgets.virtual_signal_dialog import VirtualSignalDialog
from .widgets.virtual_signal_computation import compute_virtual_signal

# Version information
__version__ = "1.1.0"


def get_main_viewer():
    """
    Returns the current instance of the SignalViewer if one exists.

    Returns:
        SignalViewer or None: The current instance of SignalViewer, or None if not initialized
    """
    return SignalViewer.instance

