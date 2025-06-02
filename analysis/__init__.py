"""
Signal analysis calculations module.

This package provides comprehensive signal analysis functions organized by domain:
- basic: Statistical and time-domain analysis
- frequency: FFT, PSD, and spectral analysis
- correlation: Auto and cross-correlation analysis
- hilbert: Hilbert transform, envelope, and phase analysis
- filters: IIR and FIR filtering
- wavelet: Continuous and discrete wavelet transforms
- cepstrum: Cepstral analysis for periodicity detection
- common: Shared utilities and validation functions

All functions follow consistent interfaces and error handling patterns.
"""

# Import all calculation functions for easy access
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

# Version information
__version__ = "1.0.0"
__author__ = "MakeATrend Analysis Team"

# Define what gets exported when using "from analysis.calculations import *"
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

# Optional: Create convenience function groups
BASIC_FUNCTIONS = [
    'calculate_basic_statistics',
    'calculate_time_domain_analysis',
    'calculate_signal_quality_metrics'
]

FREQUENCY_FUNCTIONS = [
    'calculate_fft_analysis',
    'calculate_psd_analysis'
]

ADVANCED_FUNCTIONS = [
    'calculate_hilbert_analysis',
    'calculate_phase_analysis',
    'calculate_energy_analysis',
    'calculate_wavelet_analysis_cwt',
    'calculate_wavelet_analysis_dwt',
    'calculate_cepstrum_analysis',
    'calculate_peak_detection'
]

CORRELATION_FUNCTIONS = [
    'calculate_autocorrelation_analysis',
    'calculate_cross_correlation_analysis'
]

FILTER_FUNCTIONS = [
    'calculate_iir_filter',
    'calculate_fir_filter'
]

def get_function_info():
    """
    Get information about available analysis functions.

    Returns:
        dict: Dictionary with function categories and descriptions
    """
    return {
        'basic': {
            'functions': BASIC_FUNCTIONS,
            'description': 'Basic statistical and time-domain analysis'
        },
        'frequency': {
            'functions': FREQUENCY_FUNCTIONS,
            'description': 'Frequency domain analysis (FFT, PSD)'
        },
        'correlation': {
            'functions': CORRELATION_FUNCTIONS,
            'description': 'Auto and cross-correlation analysis'
        },
        'advanced': {
            'functions': ADVANCED_FUNCTIONS,
            'description': 'Advanced transforms and decomposition'
        },
        'filters': {
            'functions': FILTER_FUNCTIONS,
            'description': 'Signal filtering (IIR, FIR)'
        }
    }


def list_available_functions():
    """
    Print a formatted list of all available analysis functions.
    """
    info = get_function_info()

    print("Available Signal Analysis Functions:")
    print("=" * 40)

    for category, details in info.items():
        print(f"\n{category.upper()} - {details['description']}")
        print("-" * len(f"{category.upper()} - {details['description']}"))

        for func_name in details['functions']:
            # Get function from this module
            func = globals().get(func_name)
            if func and hasattr(func, '__doc__') and func.__doc__:
                # Extract first line of docstring
                first_line = func.__doc__.strip().split('\n')[0]
                print(f"  {func_name:35} - {first_line}")
            else:
                print(f"  {func_name:35} - No description available")


# Optional: Validation functions for development/testing
def validate_all_imports():
    """
    Validate that all declared functions can be imported.
    Useful for testing after refactoring.

    Returns:
        dict: Results of import validation
    """
    results = {
        'success': [],
        'failed': [],
        'total': len(__all__)
    }

    for func_name in __all__:
        try:
            func = globals().get(func_name)
            if func is not None and callable(func):
                results['success'].append(func_name)
            else:
                results['failed'].append(f"{func_name} - not callable or None")
        except Exception as e:
            results['failed'].append(f"{func_name} - {str(e)}")

    return results


# Development helper
if __name__ == "__main__":
    # When run as script, show available functions
    list_available_functions()

    # Validate imports
    validation = validate_all_imports()
    print(f"\n\nValidation Results:")
    print(f"Successfully imported: {len(validation['success'])}/{validation['total']}")

    if validation['failed']:
        print(f"Failed imports:")
        for failure in validation['failed']:
            print(f"  - {failure}")
    else:
        print("All functions imported successfully!")