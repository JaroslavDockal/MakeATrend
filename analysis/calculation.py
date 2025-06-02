#!/usr/bin/env python3
"""
Backward compatibility layer for the refactored analysis calculations.

This module provides the same interface as the original calculation.py file
while using the new modularized analysis.calculations package underneath.
All existing code that imports from calculation.py should continue to work
without modification.

Usage:
    from calculation import calculate_basic_statistics  # Old way - still works
    from analysis.calculations import calculate_basic_statistics  # New way
"""

# Import all functions from the new modular structure
from analysis.calculations.basic import (
    calculate_basic_statistics,
    calculate_time_domain_analysis,
    calculate_signal_quality_metrics
)

from analysis.calculations.frequency import (
    calculate_fft_analysis,
    calculate_psd_analysis,
    calculate_spectral_features
)

from analysis.calculations.correlation import (
    calculate_autocorrelation_analysis,
    calculate_cross_correlation_analysis
)

from analysis.calculations.hilbert import (
    calculate_hilbert_analysis,
    calculate_phase_analysis,
    calculate_energy_analysis
)

from analysis.calculations.filters import (
    calculate_iir_filter,
    calculate_fir_filter,
    design_filter_parameters
)

from analysis.calculations.wavelet import (
    calculate_wavelet_analysis_cwt,
    calculate_wavelet_analysis_dwt,
    get_available_wavelets
)

from analysis.calculations.cepstrum import (
    calculate_cepstrum_analysis,
    calculate_peak_detection
)

from analysis.calculations.common import (
    safe_prepare_signal,
    safe_sample_rate,
    extended_prepare_signal,
    validate_analysis_inputs,
    format_array_for_display,
    calculate_bandwidth
)

# Import vibration-specific functions
try:
    from analysis.vibration.metrics import (
        calculate_vibration_metrics,
        calculate_vibration_severity,
        assess_machine_condition
    )

    from analysis.vibration.fft import (
        calculate_vibration_fft
    )

    from analysis.vibration.envelope import (
        calculate_envelope_analysis
    )

    VIBRATION_AVAILABLE = True
except ImportError:
    VIBRATION_AVAILABLE = False

from utils.logger import Logger

# Log the import for debugging
Logger.log_message_static("calculation.py: Backward compatibility layer loaded", Logger.INFO)

# Define what gets exported when using "from calculation import *"
__all__ = [
    # Basic analysis
    'calculate_basic_statistics',
    'calculate_time_domain_analysis',
    'calculate_signal_quality_metrics',

    # Frequency analysis
    'calculate_fft_analysis',
    'calculate_psd_analysis',
    'calculate_spectral_features',

    # Correlation analysis
    'calculate_autocorrelation_analysis',
    'calculate_cross_correlation_analysis',

    # Hilbert transform
    'calculate_hilbert_analysis',
    'calculate_phase_analysis',
    'calculate_energy_analysis',

    # Filtering
    'calculate_iir_filter',
    'calculate_fir_filter',
    'design_filter_parameters',

    # Wavelet analysis
    'calculate_wavelet_analysis_cwt',
    'calculate_wavelet_analysis_dwt',
    'get_available_wavelets',

    # Cepstral analysis and peak detection
    'calculate_cepstrum_analysis',
    'calculate_peak_detection',

    # Common utilities
    'safe_prepare_signal',
    'safe_sample_rate',
    'extended_prepare_signal',
    'validate_analysis_inputs',
    'format_array_for_display',
    'calculate_bandwidth'
]

# Add vibration functions if available
if VIBRATION_AVAILABLE:
    __all__.extend([
        'calculate_vibration_metrics',
        'calculate_vibration_severity',
        'assess_machine_condition',
        'calculate_vibration_fft',
        'calculate_envelope_analysis'
    ])


# Legacy function aliases for maximum compatibility
def calculate_statistics(values, dialog=None, title="Statistics"):
    """Legacy alias for calculate_basic_statistics."""
    Logger.log_message_static(
        "calculation.py: Using legacy alias 'calculate_statistics' -> 'calculate_basic_statistics'", Logger.DEBUG)
    return calculate_basic_statistics(values, dialog, title)


def calculate_fft(time_arr, values, dialog=None, title="FFT"):
    """Legacy alias for calculate_fft_analysis."""
    Logger.log_message_static("calculation.py: Using legacy alias 'calculate_fft' -> 'calculate_fft_analysis'",
                              Logger.DEBUG)
    return calculate_fft_analysis(time_arr, values, dialog, title)


def calculate_psd(time_arr, values, dialog=None, title="PSD"):
    """Legacy alias for calculate_psd_analysis."""
    Logger.log_message_static("calculation.py: Using legacy alias 'calculate_psd' -> 'calculate_psd_analysis'",
                              Logger.DEBUG)
    return calculate_psd_analysis(time_arr, values, dialog, title)


def calculate_autocorrelation(time_arr, values, dialog=None, title="Autocorrelation"):
    """Legacy alias for calculate_autocorrelation_analysis."""
    Logger.log_message_static(
        "calculation.py: Using legacy alias 'calculate_autocorrelation' -> 'calculate_autocorrelation_analysis'",
        Logger.DEBUG)
    return calculate_autocorrelation_analysis(time_arr, values, dialog, title)


def calculate_cross_correlation(time_arr1, values1, time_arr2, values2, dialog=None, title="Cross-Correlation"):
    """Legacy alias for calculate_cross_correlation_analysis."""
    Logger.log_message_static(
        "calculation.py: Using legacy alias 'calculate_cross_correlation' -> 'calculate_cross_correlation_analysis'",
        Logger.DEBUG)
    return calculate_cross_correlation_analysis(time_arr1, values1, time_arr2, values2, dialog, title)


# Add legacy aliases to exports
__all__.extend([
    'calculate_statistics',
    'calculate_fft',
    'calculate_psd',
    'calculate_autocorrelation',
    'calculate_cross_correlation'
])


# Utility function to list all available functions
def list_available_functions():
    """List all available calculation functions."""
    print("Available Calculation Functions (Backward Compatibility Layer):")
    print("=" * 65)

    # Group by category
    categories = {
        'Basic Analysis': [
            'calculate_basic_statistics', 'calculate_statistics',
            'calculate_time_domain_analysis', 'calculate_signal_quality_metrics'
        ],
        'Frequency Analysis': [
            'calculate_fft_analysis', 'calculate_fft',
            'calculate_psd_analysis', 'calculate_psd',
            'calculate_spectral_features'
        ],
        'Correlation Analysis': [
            'calculate_autocorrelation_analysis', 'calculate_autocorrelation',
            'calculate_cross_correlation_analysis', 'calculate_cross_correlation'
        ],
        'Advanced Analysis': [
            'calculate_hilbert_analysis', 'calculate_phase_analysis',
            'calculate_energy_analysis', 'calculate_wavelet_analysis_cwt',
            'calculate_wavelet_analysis_dwt', 'calculate_cepstrum_analysis'
        ],
        'Filtering': [
            'calculate_iir_filter', 'calculate_fir_filter',
            'design_filter_parameters'
        ],
        'Utilities': [
            'safe_prepare_signal', 'safe_sample_rate',
            'validate_analysis_inputs', 'calculate_bandwidth'
        ]
    }

    if VIBRATION_AVAILABLE:
        categories['Vibration Analysis'] = [
            'calculate_vibration_metrics', 'calculate_vibration_severity',
            'assess_machine_condition', 'calculate_vibration_fft',
            'calculate_envelope_analysis'
        ]

    for category, functions in categories.items():
        print(f"\n{category}:")
        print("-" * len(category))
        for func_name in functions:
            if func_name in __all__:
                func = globals().get(func_name)
                if func and hasattr(func, '__doc__') and func.__doc__:
                    # Extract first line of docstring
                    first_line = func.__doc__.strip().split('\n')[0]
                    print(f"  {func_name:35} - {first_line}")
                else:
                    print(f"  {func_name:35} - Function available")


def get_compatibility_info():
    """Get information about backward compatibility status."""
    return {
        'legacy_layer_active': True,
        'new_module_structure': 'analysis.calculations.*',
        'vibration_functions_available': VIBRATION_AVAILABLE,
        'total_functions': len(__all__),
        'legacy_aliases': [
            'calculate_statistics -> calculate_basic_statistics',
            'calculate_fft -> calculate_fft_analysis',
            'calculate_psd -> calculate_psd_analysis',
            'calculate_autocorrelation -> calculate_autocorrelation_analysis',
            'calculate_cross_correlation -> calculate_cross_correlation_analysis'
        ]
    }


# Add utility functions to exports
__all__.extend([
    'list_available_functions',
    'get_compatibility_info'
])

# Version information for compatibility tracking
__version__ = "2.0.0-compat"
__compatibility_version__ = "1.0.0"

if __name__ == "__main__":
    print("MakeATrend Calculation Module - Backward Compatibility Layer")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Compatible with: {__compatibility_version__}")
    print(f"Vibration functions available: {VIBRATION_AVAILABLE}")
    print(f"Total functions: {len(__all__)}")

    print("\nThis module provides backward compatibility for existing code.")
    print("New code should import from 'analysis.calculations' directly.")

    list_available_functions()