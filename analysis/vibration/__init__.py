"""
Vibration Analysis Module for MakeATrend.

This package provides specialized vibration analysis capabilities optimized
for mechanical systems, rotating machinery, and condition monitoring:

- Time domain vibration metrics (RMS, peak, crest factor, kurtosis)
- Frequency domain analysis with harmonic detection
- Envelope analysis for bearing fault detection
- Auto-detection of vibration and RPM signals
- Bearing fault frequency calculations
- ISO 10816 severity assessment

The vibration module builds upon the core calculations package but provides
domain-specific optimizations and interpretations for vibration analysis.

Structure:
- metrics.py: Time-domain vibration metrics and severity assessment
- fft.py: FFT analysis optimized for vibration with harmonic markers
- envelope.py: Envelope analysis for bearing fault detection
- detection.py: Auto-detection of vibration signals by name patterns
- bearing.py: Bearing fault frequency calculations
- severity.py: Vibration severity assessment per ISO standards
"""

# Import all vibration analysis functions
from .metrics import (
    calculate_vibration_metrics,
    calculate_vibration_severity,
    assess_machine_condition
)

from .fft import (
    calculate_vibration_fft,
    calculate_harmonic_analysis,
    detect_resonances
)

from .envelope import (
    calculate_envelope_analysis,
    calculate_bearing_envelope,
    detect_bearing_faults
)

from .detection import (
    detect_vibration_signals,
    detect_rpm_signals,
    classify_signal_type,
    get_signal_recommendations
)

from .bearing import (
    calculate_bearing_fault_frequencies,
    calculate_gear_mesh_frequencies,
    get_typical_bearing_parameters
)

from .severity import (
    assess_vibration_severity_iso10816,
    assess_vibration_severity_iso2372,
    get_severity_recommendations
)

# Version information
__version__ = "1.0.0"
__author__ = "MakeATrend Vibration Team"

# Main exports
__all__ = [
    # Vibration metrics
    'calculate_vibration_metrics',
    'calculate_vibration_severity',
    'assess_machine_condition',

    # FFT analysis
    'calculate_vibration_fft',
    'calculate_harmonic_analysis',
    'detect_resonances',

    # Envelope analysis
    'calculate_envelope_analysis',
    'calculate_bearing_envelope',
    'detect_bearing_faults',

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
]

# Function categories for documentation
BASIC_VIBRATION_FUNCTIONS = [
    'calculate_vibration_metrics',
    'calculate_vibration_severity',
    'assess_machine_condition'
]

FREQUENCY_VIBRATION_FUNCTIONS = [
    'calculate_vibration_fft',
    'calculate_harmonic_analysis',
    'detect_resonances'
]

FAULT_DETECTION_FUNCTIONS = [
    'calculate_envelope_analysis',
    'calculate_bearing_envelope',
    'detect_bearing_faults'
]

DIAGNOSTIC_FUNCTIONS = [
    'calculate_bearing_fault_frequencies',
    'assess_vibration_severity_iso10816',
    'get_severity_recommendations'
]


def get_vibration_analysis_info():
    """
    Get information about available vibration analysis functions.

    Returns:
        dict: Dictionary with function categories and descriptions
    """
    return {
        'basic': {
            'functions': BASIC_VIBRATION_FUNCTIONS,
            'description': 'Basic vibration metrics and condition assessment'
        },
        'frequency': {
            'functions': FREQUENCY_VIBRATION_FUNCTIONS,
            'description': 'Frequency domain analysis with harmonics and resonances'
        },
        'fault_detection': {
            'functions': FAULT_DETECTION_FUNCTIONS,
            'description': 'Envelope analysis and bearing fault detection'
        },
        'diagnostics': {
            'functions': DIAGNOSTIC_FUNCTIONS,
            'description': 'Fault frequency calculation and severity assessment'
        }
    }


def list_vibration_standards():
    """
    List supported vibration analysis standards.

    Returns:
        dict: Information about supported standards
    """
    return {
        'ISO 10816': {
            'description': 'Mechanical vibration - Evaluation of machine vibration by measurements on non-rotating parts',
            'application': 'General machinery condition monitoring',
            'frequency_range': '10 Hz - 1 kHz',
            'measurement': 'RMS velocity (mm/s)'
        },
        'ISO 2372': {
            'description': 'Mechanical vibration of machines with operating speeds from 10 to 200 rev/s',
            'application': 'Older standard, largely superseded by ISO 10816',
            'frequency_range': '10 Hz - 1 kHz',
            'measurement': 'RMS velocity (mm/s)'
        },
        'VDI 2056': {
            'description': 'German standard for vibration assessment',
            'application': 'Complementary to ISO standards',
            'frequency_range': 'Variable',
            'measurement': 'RMS velocity and displacement'
        }
    }


def get_typical_fault_frequencies():
    """
    Get information about typical vibration fault frequencies.

    Returns:
        dict: Common fault types and their frequency characteristics
    """
    return {
        'Unbalance': {
            'frequency': '1X RPM',
            'characteristics': 'Radial vibration, 90° phase difference between sensors',
            'severity': 'Proportional to square of speed'
        },
        'Misalignment': {
            'frequency': '1X, 2X RPM (angular), 1X, 2X, 3X RPM (parallel)',
            'characteristics': 'High axial vibration for angular misalignment',
            'severity': 'Increases with load and speed'
        },
        'Looseness': {
            'frequency': '1X, 2X, 3X... RPM (many harmonics)',
            'characteristics': 'Many harmonics, unstable amplitude and phase',
            'severity': 'Can affect all frequency ranges'
        },
        'Bearing Outer Race': {
            'frequency': 'BPFO (Ball Pass Frequency Outer)',
            'characteristics': 'High frequency modulation, envelope analysis effective',
            'severity': 'Increases with load'
        },
        'Bearing Inner Race': {
            'frequency': 'BPFI (Ball Pass Frequency Inner)',
            'characteristics': '1X RPM modulation, axial and radial components',
            'severity': 'Severe fault, urgent attention needed'
        },
        'Gear Mesh': {
            'frequency': 'Number of teeth × RPM',
            'characteristics': 'High frequency, sidebands at shaft frequencies',
            'severity': 'Sidebands indicate wear or misalignment'
        }
    }


# Optional development helpers
if __name__ == "__main__":
    print("MakeATrend Vibration Analysis Module")
    print("=" * 45)

    # Show module info
    info = get_vibration_analysis_info()
    print(f"Version: {__version__}")
    print(f"Total functions: {len(__all__)}")

    print("\nFunction categories:")
    for category, details in info.items():
        print(f"  {category}: {len(details['functions'])} functions")
        print(f"    {details['description']}")

    # Show standards
    print("\nSupported standards:")
    standards = list_vibration_standards()
    for std_name, std_info in standards.items():
        print(f"  {std_name}: {std_info['description']}")

    print(f"\nUsage: from analysis.vibration import calculate_vibration_metrics")