"""
Utilities package for CSV Signal Viewer.

This package provides various utility functions and helper classes used throughout
the application, including custom UI components, logging, color management, and more.
"""

# Import main utility classes
from .logger import Logger
from .signal_colors import SignalColors
from .custom_viewbox import CustomViewBox


# Create convenience functions for logging
def log_debug(message):
    """
    Log a debug message using the global logger.

    Args:
        message (str): The debug message to log
    """
    Logger.log_message_static(message, Logger.DEBUG)


def log_info(message):
    """
    Log an info message using the global logger.

    Args:
        message (str): The info message to log
    """
    Logger.log_message_static(message, Logger.INFO)


def log_warning(message):
    """
    Log a warning message using the global logger.

    Args:
        message (str): The warning message to log
    """
    Logger.log_message_static(message, Logger.WARNING)


def log_error(message):
    """
    Log an error message using the global logger.

    Args:
        message (str): The error message to log
    """
    Logger.log_message_static(message, Logger.ERROR)


# Convenience color access functions
def get_color(index):
    """
    Get a color by index from the predefined color palette.

    Args:
        index (int): Color index

    Returns:
        str: Hex color code
    """
    return SignalColors.get_color(index)


def get_color_for_signal(name):
    """
    Generate a consistent color for a signal name.

    Args:
        name (str): Signal name

    Returns:
        str: Hex color code
    """
    return SignalColors.get_color_for_name(name)


def random_color():
    """
    Generate a random bright color suitable for dark backgrounds.

    Returns:
        str: Hex color code
    """
    return SignalColors.random_color()


# Version information
__version__ = "1.0.0"

# Define what's available when using "from utils import *"
__all__ = [
    # Classes
    'Logger',
    'SignalColors',
    'CustomViewBox',

    # Logging convenience functions
    'log_debug',
    'log_info',
    'log_warning',
    'log_error',

    # Color convenience functions
    'get_color',
    'get_color_for_signal',
    'random_color',

    # Version info
    '__version__'
]