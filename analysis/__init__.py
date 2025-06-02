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
- vibration: Specialized vibration analysis functions
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

    from analysis.vibration.detection import (
        detect_vibration_signals,
        detect_rpm_signals,
        classify_signal_type,
        get_signal_recommendations
    )

    from analysis.vibration.bearing import (
        calculate_bearing_fault_frequencies,
        calculate_gear_mesh_frequencies,
        get_typical_bearing_parameters
    )

    from analysis.vibration.severity import (
        assess_vibration_severity_iso10816,
        assess_vibration_severity_iso2372,
        get_severity_recommendations
    )

    VIBRATION_AVAILABLE = True

except ImportError as e:
    import warnings
    warnings.warn(f"Vibration analysis not available: {e}")
    VIBRATION_AVAILABLE = False

# Version information
__version__ = "2.0.0"
__author__ = "MakeATrend Analysis Team"

# Define what gets exported when using "from analysis import *"
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
        # Vibration metrics
        'calculate_vibration_metrics',
        'calculate_vibration_severity',
        'assess_machine_condition',

        # Vibration FFT
        'calculate_vibration_fft',

        # Envelope analysis
        'calculate_envelope_analysis',

        # Signal detection
        'detect_vibration_signals',
        'detect_rpm_signals',
        'classify_signal_type',
        'get_signal_recommendations',

        # Bearing analysis
        'calculate_bearing_fault_frequencies',
        'calculate_gear_mesh_frequencies',
        'get_typical_bearing_parameters',

        # Severity assessment
        'assess_vibration_severity_iso10816',
        'assess_vibration_severity_iso2372',
        'get_severity_recommendations'
    ])

# Optional: Create convenience function groups
BASIC_FUNCTIONS = [
    'calculate_basic_statistics',
    'calculate_time_domain_analysis',
    'calculate_signal_quality_metrics'
]

FREQUENCY_FUNCTIONS = [
    'calculate_fft_analysis',
    'calculate_psd_analysis',
    'calculate_spectral_features'
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
    'calculate_fir_filter',
    'design_filter_parameters'
]

VIBRATION_FUNCTIONS = []
if VIBRATION_AVAILABLE:
    VIBRATION_FUNCTIONS = [
        'calculate_vibration_metrics',
        'calculate_vibration_severity',
        'assess_machine_condition',
        'calculate_vibration_fft',
        'calculate_envelope_analysis',
        'detect_vibration_signals',
        'calculate_bearing_fault_frequencies',
        'assess_vibration_severity_iso10816'
    ]


def get_function_info():
    """
    Get information about available analysis functions.

    Returns:
        dict: Dictionary with function categories and descriptions
    """
    info = {
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

    if VIBRATION_AVAILABLE:
        info['vibration'] = {
            'functions': VIBRATION_FUNCTIONS,
            'description': 'Vibration analysis and machinery diagnostics'
        }

    return info


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


def get_module_status():
    """
    Get status information about the analysis module.

    Returns:
        dict: Status information
    """
    return {
        'version': __version__,
        'vibration_available': VIBRATION_AVAILABLE,
        'total_functions': len(__all__),
        'function_categories': len(get_function_info()),
        'backward_compatibility': True,
        'import_status': 'Success'
    }


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


def check_dependencies():
    """
    Check if all required dependencies are available.

    Returns:
        dict: Dependency status
    """
    dependencies = {
        'numpy': False,
        'scipy': False,
        'pywt': False,
        'PySide6': False
    }

    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass

    try:
        import scipy
        dependencies['scipy'] = True
    except ImportError:
        pass

    try:
        import pywt
        dependencies['pywt'] = True
    except ImportError:
        pass

    try:
        import PySide6
        dependencies['PySide6'] = True
    except ImportError:
        pass

    return dependencies


# Development helper
if __name__ == "__main__":
    # When run as script, show available functions
    print("MakeATrend Analysis Module")
    print("=" * 30)

    status = get_module_status()
    print(f"Version: {status['version']}")
    print(f"Vibration functions: {'✓' if status['vibration_available'] else '✗'}")
    print(f"Total functions: {status['total_functions']}")
    print(f"Categories: {status['function_categories']}")

    print("\nDependency check:")
    deps = check_dependencies()
    for dep, available in deps.items():
        print(f"  {dep}: {'✓' if available else '✗'}")

    print("\n" + "="*50)
    list_available_functions()

    # Validate imports
    print("\n" + "="*50)
    validation = validate_all_imports()
    print(f"Import Validation Results:")
    print(f"Successfully imported: {len(validation['success'])}/{validation['total']}")

    if validation['failed']:
        print(f"Failed imports:")
        for failure in validation['failed']:
            print(f"  - {failure}")
    else:
        print("All functions imported successfully!")