"""
Analysis package for CSV Signal Viewer.

This package provides signal analysis functionality including statistical analysis,
frequency domain transformations, filtering, and various signal processing techniques.
The package includes both calculation functions and GUI components for displaying results.
"""

# Version information
__version__ = "1.0.0"

# Main dialog for signal analysis
from .analysis_dialog import SignalAnalysisDialog, show_analysis_dialog

# UI Components
from .explanation import ExplanationTab

# Helper functions
from .helpers import (
    safe_sample_rate,
    safe_prepare_signal,
    extended_prepare_signal,
    calculate_bandwidth
)

# Basic statistical analysis
from .calculation import (
    calculate_basic_statistics,
    calculate_time_domain_analysis,
    calculate_peak_detection,
    calculate_energy_analysis,
)

# Frequency domain analysis
from .calculation import (
    calculate_fft_analysis,
    calculate_psd_analysis,
    calculate_cepstrum_analysis,
    calculate_phase_analysis,
)

# Advanced analysis
from .calculation import (
    calculate_hilbert_analysis,
    calculate_autocorrelation_analysis,
    calculate_cross_correlation_analysis,
)

# Wavelet analysis
from .calculation import (
    calculate_wavelet_analysis_cwt,
    calculate_wavelet_analysis_dwt,
)

# Filtering
from .calculation import (
    calculate_iir_filter,
    calculate_fir_filter,
)

# Re-export everything from calculation for backward compatibility
from .calculation import *

# Define what's available when using "from analysis import *"
__all__ = [
    # Main dialog
    'SignalAnalysisDialog',
    'show_analysis_dialog',

    # UI Components
    'ExplanationTab',

    # Helper functions
    'safe_sample_rate',
    'safe_prepare_signal',
    'extended_prepare_signal',
    'calculate_bandwidth',

    # Analysis functions (grouped functionally)
    # Basic statistics
    'calculate_basic_statistics',
    'calculate_time_domain_analysis',
    'calculate_peak_detection',
    'calculate_energy_analysis',

    # Frequency domain analysis
    'calculate_fft_analysis',
    'calculate_psd_analysis',
    'calculate_cepstrum_analysis',
    'calculate_phase_analysis',

    # Advanced analysis
    'calculate_hilbert_analysis',
    'calculate_autocorrelation_analysis',
    'calculate_cross_correlation_analysis',

    # Wavelet analysis
    'calculate_wavelet_analysis_cwt',
    'calculate_wavelet_analysis_dwt',

    # Filtering
    'calculate_iir_filter',
    'calculate_fir_filter',

    # Version info
    '__version__'
]