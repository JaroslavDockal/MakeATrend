"""
Parsers package for MakeATrend application.
This package contains various file format parsers used to load data into the application.
"""

from .parser_auto import *
from .parser_debug import *
from .parser_hdm5 import *
from .parser_helpers import *
from .parser_lvm import *
from .parser_mat import *
from .parser_sql import *
from .parser_standard import *
from .parser_tmds import *

# Define the list of all parsers for easy access
__all__ = [
    'parser_auto',
    'parser_debug',
    'parser_hdm5',
    'parser_helpers',
    'parser_lvm',
    'parser_mat',
    'parser_sql',
    'parser_standard',
    'parser_tmds',
]

