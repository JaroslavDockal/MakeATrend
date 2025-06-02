"""
Parsers package for MakeATrend application.
This package contains various file format parsers used to load data into the application.

Available parsers:
- parser_auto: Automatic format detection and parsing
- parser_debug: Debug/testing parser
- parser_hdm5: HDM5 format parser
- parser_helpers: Helper functions for parsing
- parser_lvm: LabVIEW Measurement File parser
- parser_mat: MATLAB file format parser
- parser_sql: SQL database parser
- parser_standard: Standard format parser
- parser_tmds: TMDS format parser
"""

__version__ = "1.0.0"

# Core parsers
from .parser_auto import *
from .parser_standard import *

# Specific format parsers
from .parser_lvm import *
from .parser_mat import *
from .parser_hdf5 import *
from .parser_sql import *
from .parser_tmds import *

# Utility parsers
from .parser_helpers import *
from .parser_debug import *

# Define the list of all parsers for easy access
__all__ = [
    # Core parsers
    'parser_auto',
    'parser_standard',

    # Specific format parsers
    'parser_lvm',
    'parser_mat',
    'parser_hdf5.py',
    'parser_sql',
    'parser_tmds',

    # Utility parsers
    'parser_helpers',
    'parser_debug',
]
