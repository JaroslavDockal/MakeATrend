"""
UI package for CSV Signal Viewer.

This package provides the graphical user interface components for the CSV Signal Viewer
application, including the main viewer window and all supporting UI elements.
"""

# Import main components for direct package access
from .viewer import SignalViewer

# Import UI component setup functions that might be needed by external modules
from .ui_components.control_panel import setup_control_panel
from .ui_components.plot_area import setup_plot_area, setup_axes

# Version information
__version__ = "1.0.0"


def get_main_viewer():
    """
    Returns the current instance of the SignalViewer if one exists.

    Returns:
        SignalViewer or None: The current instance of SignalViewer, or None if not initialized
    """
    return SignalViewer.instance


# Define what's available when using "from ui import *"
__all__ = [
    # Main class
    'SignalViewer',

    # Convenience functions
    'get_main_viewer',

    # Setup functions that might be needed externally
    'setup_control_panel',
    'setup_plot_area',
    'setup_axes',

    # Version info
    '__version__'
]