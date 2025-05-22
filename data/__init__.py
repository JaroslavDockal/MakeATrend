"""
Data Package for Signal Analysis Application

This package provides comprehensive functionality for loading, processing,
and managing time-series signal data from various file formats. Core components include:

1. Data Loading
   - CSV and proprietary recorder file support
   - Single and multi-file import with automatic merging
   - Automatic format detection (delimiters, encoding)

2. Signal Processing
   - Digital/analog signal classification
   - Time synchronization and alignment
   - Interpolation and resampling

3. Project Management
   - Save/load analysis state
   - Signal metadata handling

4. Export Capabilities
   - Graph exporting to various formats
   - Data export with configurable formatting

The package serves as the data backbone for the application, handling all aspects
of signal data from initial file loading through processing to persistence.
"""

# Export submodule functionality
from .loader import load_single_file, load_multiple_files
from .signal_utils import is_digital_signal, find_nearest_index
from .parser import parse_csv_or_recorder
from .csv_dialect import ParseOptions, ParseOptionsDialog, detect_csv_dialect
from .project import save_project_state, load_project_state
from .graph_export import export_graph

# Define public API
__all__ = [
    # Loader functions
    'load_single_file',
    'load_multiple_files',

    # Signal utilities
    'parse_csv_or_recorder',
    'is_digital_signal',
    'find_nearest_index',

    # CSV parsing
    'ParseOptions',
    'ParseOptionsDialog',
    'detect_csv_dialect',

    # Project management
    'save_project_state',
    'load_project_state',

    # Export functionality
    'export_graph'
]