"""
Automatic detection and classification of vibration signals.

This module provides functions for automatically identifying vibration channels
and RPM signals based on signal names and data characteristics.

Key functions:
- Detection of vibration signals by names and units
- Identification of RPM/speed signals
- Classification of axes (X, Y, Z) and positions (DE, NDE)
- Analysis recommendations based on signal type
- Automatic sorting of multi-channel data
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any
from utils.logger import Logger


def detect_vibration_signals(signal_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    """
    Automatically detect vibration signals from a collection of signals.

    Analyzes signal names, units, and data characteristics to identify
    vibration measurements according to standard conventions.

    Args:
        signal_dict (dict): Dictionary of signals in format:
            {signal_name: (time_array, values_array), ...}

    Returns:
        dict: Detection results containing:
            - vibration_signals: List of identified vibration signal names
            - rpm_signals: List of identified RPM/speed signal names
            - other_signals: List of other signal names
            - classification: Detailed classification of all signals
            - recommendations: List of analysis recommendations

    Example:
        >>> signals = {
        ...     'DE_X_Accel': (time, accel_x),
        ...     'DE_Y_Accel': (time, accel_y),
        ...     'RPM_Signal': (time, rpm_data)
        ... }
        >>> result = detect_vibration_signals(signals)
        >>> print(result['vibration_signals'])
        ['DE_X_Accel', 'DE_Y_Accel']
    """
    Logger.log_message_static("Vibration-Detection: Starting automatic vibration signal detection", Logger.DEBUG)

    vibration_signals = []
    rpm_signals = []
    other_signals = []
    classification = {}

    # Keywords for vibration signals
    vibration_keywords = [
        'vibration', 'vib', 'accel', 'acceleration', 'velocity', 'vel',
        'displacement', 'disp', 'shake', 'oscillation', 'motion',
        'de_', 'nde_', 'drive_end', 'non_drive_end', 'motor_', 'bearing_',
        'x_axis', 'y_axis', 'z_axis', '_x', '_y', '_z'
    ]

    # Keywords for RPM signals
    rpm_keywords = [
        'rpm', 'speed', 'rotation', 'rev', 'frequency', 'hz', 'tacho',
        'encoder', 'ot', 'once_per_rev', 'keyphasor'
    ]

    # Units for vibration signals
    vibration_units = [
        'g', 'mg', 'm/s2', 'm/s²', 'mm/s', 'm/s', 'mm', 'um', 'μm', 'mil'
    ]

    try:
        for signal_name, (time_arr, values) in signal_dict.items():
            signal_name_lower = signal_name.lower()

            # Calculate keyword scores
            vibration_score = sum(keyword in signal_name_lower for keyword in vibration_keywords)
            rpm_score = sum(keyword in signal_name_lower for keyword in rpm_keywords)

            # Check for units in signal name
            unit_match = False
            for unit in vibration_units:
                if unit in signal_name_lower or f"({unit})" in signal_name_lower:
                    unit_match = True
                    vibration_score += 1
                    break

            # Analyze signal characteristics
            signal_characteristics = analyze_signal_characteristics(values)

            # Classification of signal type
            if rpm_score > 0 and vibration_score == 0:
                signal_type = "RPM"
                rpm_signals.append(signal_name)
                confidence = min(100, rpm_score * 20 + signal_characteristics['rpm_likelihood'] * 50)
                classification[signal_name] = {
                    'type': signal_type,
                    'confidence': confidence,
                    'rpm_likelihood': signal_characteristics['rpm_likelihood'],
                    'characteristic_values': {
                        'mean': signal_characteristics['mean'],
                        'std': signal_characteristics['std'],
                        'coefficient_of_variation': signal_characteristics['coefficient_of_variation']
                    }
                }

            elif vibration_score > 0 or signal_characteristics['vibration_likelihood'] > 0.6:
                signal_type = "Vibration"
                vibration_signals.append(signal_name)
                confidence = min(100, vibration_score * 15 +
                                  unit_match * 20 +
                                  signal_characteristics['vibration_likelihood'] * 40)

                # Detect axis, location, and measurement type
                axis = detect_axis(signal_name_lower)
                location = detect_location(signal_name_lower)
                measurement_type = detect_measurement_type(signal_name_lower)

                classification[signal_name] = {
                    'type': signal_type,
                    'confidence': confidence,
                    'axis': axis,
                    'location': location,
                    'measurement_type': measurement_type,
                    'vibration_likelihood': signal_characteristics['vibration_likelihood'],
                    'characteristic_values': {
                        'rms': signal_characteristics['rms'],
                        'peak_to_rms': signal_characteristics['peak_to_rms'],
                        'zero_mean_likelihood': signal_characteristics['zero_mean_likelihood']
                    }
                }

            else:
                signal_type = "Other"
                other_signals.append(signal_name)
                classification[signal_name] = {
                    'type': "Other",
                    'confidence': 50,
                    'notes': "Could not confidently classify as vibration or RPM"
                }

        # Generate recommendations
        recommendations = generate_analysis_recommendations(
            vibration_signals, rpm_signals, classification
        )

        # Results
        results = {
            'vibration_signals': vibration_signals,
            'rpm_signals': rpm_signals,
            'other_signals': other_signals,
            'classification': classification,
            'total_signals': len(signal_dict),
            'vibration_count': len(vibration_signals),
            'rpm_count': len(rpm_signals),
            'recommendations': recommendations,
            'detection_summary': generate_detection_summary(vibration_signals, rpm_signals, classification)
        }

        Logger.log_message_static(
            f"Vibration-Detection: Detection completed. "
            f"Found {len(vibration_signals)} vibration signals, {len(rpm_signals)} RPM signals",
            Logger.DEBUG
        )

        return results

    except Exception as e:
        Logger.log_message_static(f"Vibration-Detection: Error in signal detection: {str(e)}", Logger.ERROR)
        return {'error': str(e), 'vibration_signals': [], 'rpm_signals': []}


def detect_rpm_signals(signal_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[str]:
    """
    Specialized detection of RPM signals.

    Args:
        signal_dict (dict): Dictionary of signals

    Returns:
        list: List of RPM signal names
    """
    result = detect_vibration_signals(signal_dict)
    return result.get('rpm_signals', [])


def classify_signal_type(signal_name: str, values: np.ndarray) -> Dict[str, Any]:
    """
    Classify an individual signal by name and characteristics.

    Args:
        signal_name (str): Signal name
        values (np.ndarray): Signal data

    Returns:
        dict: Signal classification
    """
    try:
        # Use detection logic on a single signal
        single_signal_dict = {signal_name: (np.arange(len(values)), values)}
        result = detect_vibration_signals(single_signal_dict)

        return result['classification'].get(signal_name, {'type': 'Unknown', 'confidence': 0})

    except Exception as e:
        Logger.log_message_static(f"Vibration-Detection: Error classifying signal {signal_name}: {e}", Logger.WARNING)
        return {'type': 'Unknown', 'confidence': 0, 'error': str(e)}


def analyze_signal_characteristics(values: np.ndarray) -> Dict[str, float]:
    """
    Analyze signal characteristics to determine type.

    Args:
        values (np.ndarray): Signal data

    Returns:
        dict: Signal characteristics
    """
    try:
        # Basic statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        rms_val = np.sqrt(np.mean(values**2))

        # Characteristics for RPM
        # RPM signals are typically positive with less variability
        positive_ratio = np.sum(values > 0) / len(values)
        cv = std_val / abs(mean_val) if abs(mean_val) > 0 else np.inf

        # RPM likelihood - high for stable positive signals
        rpm_likelihood = 0.0
        if positive_ratio > 0.95 and 0.01 < cv < 0.3 and mean_val > 0:
            rpm_likelihood = min(1.0, (positive_ratio - 0.95) * 20 + (0.3 - cv) * 2)

        # Characteristics for vibrations
        # Vibration signals typically have mean near zero, higher variability
        zero_mean_likelihood = 1.0 - min(1.0, abs(mean_val) / (std_val + 1e-12))
        variability_score = min(1.0, cv / 2) if cv != np.inf else 0.0

        # Vibration likelihood
        vibration_likelihood = (zero_mean_likelihood * 0.6 + variability_score * 0.4)

        # Additional characteristics
        peak_to_rms = np.max(np.abs(values)) / rms_val if rms_val > 0 else 0

        return {
            'mean': float(mean_val),
            'std': float(std_val),
            'rms': float(rms_val),
            'coefficient_of_variation': float(cv),
            'positive_ratio': float(positive_ratio),
            'peak_to_rms': float(peak_to_rms),
            'zero_mean_likelihood': float(zero_mean_likelihood),
            'variability_score': float(variability_score),
            'rpm_likelihood': float(rpm_likelihood),
            'vibration_likelihood': float(vibration_likelihood)
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Detection: Error analyzing signal characteristics: {e}", Logger.WARNING)
        return {
            'rpm_likelihood': 0.0,
            'vibration_likelihood': 0.0,
            'error': str(e)
        }


def detect_axis(signal_name_lower: str) -> str:
    """Detect axis (X, Y, Z) from signal name."""
    if any(pattern in signal_name_lower for pattern in ['_x', '-x', 'x_axis', 'radial_h', 'horizontal']):
        return 'X'
    elif any(pattern in signal_name_lower for pattern in ['_y', '-y', 'y_axis', 'radial_v', 'vertical']):
        return 'Y'
    elif any(pattern in signal_name_lower for pattern in ['_z', '-z', 'z_axis', 'axial', 'thrust']):
        return 'Z'
    else:
        return 'Unknown'


def detect_location(signal_name_lower: str) -> str:
    """Detect measurement position (DE, NDE) from signal name."""
    de_patterns = ['de_', 'drive_end', 'de-', 'motor_', 'coupling_']
    nde_patterns = ['nde_', 'non_drive_end', 'nde-', 'free_end', 'fan_', 'opposite_']

    if any(pattern in signal_name_lower for pattern in de_patterns):
        return 'DE'
    elif any(pattern in signal_name_lower for pattern in nde_patterns):
        return 'NDE'
    else:
        return 'Unknown'


def detect_measurement_type(signal_name_lower: str) -> str:
    """Detect measurement type (acceleration, velocity, displacement)."""
    if any(pattern in signal_name_lower for pattern in ['accel', 'acceleration', '_g', 'mg']):
        return 'Acceleration'
    elif any(pattern in signal_name_lower for pattern in ['vel', 'velocity', 'mm/s', 'm/s']):
        return 'Velocity'
    elif any(pattern in signal_name_lower for pattern in ['disp', 'displacement', 'mm', 'μm', 'um', 'mil']):
        return 'Displacement'
    else:
        return 'Unknown'


def generate_analysis_recommendations(vibration_signals: List[str], rpm_signals: List[str],
                                    classification: Dict) -> List[str]:
    """Generate analysis recommendations based on detected signals."""
    recommendations = []

    if len(vibration_signals) == 0:
        recommendations.append("No vibration signals detected. Check signal naming conventions.")
        return recommendations

    if len(rpm_signals) == 0:
        recommendations.append("No RPM signal detected. Harmonic analysis will use estimated frequency.")
    else:
        recommendations.append(f"RPM signal detected: {rpm_signals[0]}. Harmonic analysis recommended.")

    # Analysis by number of axes
    axes_found = set()
    locations_found = set()

    for signal in vibration_signals:
        info = classification.get(signal, {})
        if info.get('axis') != 'Unknown':
            axes_found.add(info.get('axis'))
        if info.get('location') != 'Unknown':
            locations_found.add(info.get('location'))

    if len(axes_found) >= 2:
        recommendations.append("Multi-axis vibration data available. Cross-channel analysis recommended.")

    if len(locations_found) >= 2:
        recommendations.append("Multi-location data (DE/NDE) available. Bearing analysis recommended.")

    if 'Z' in axes_found:
        recommendations.append("Axial vibration detected. Check for misalignment.")

    # Specific recommendations by measurement type
    measurement_types = [classification.get(s, {}).get('measurement_type') for s in vibration_signals]

    if 'Acceleration' in measurement_types:
        recommendations.append("Acceleration data available. Envelope analysis for bearing faults recommended.")

    if 'Velocity' in measurement_types:
        recommendations.append("Velocity data available. ISO severity assessment possible.")

    if len(vibration_signals) >= 6:  # 3 axes × 2 positions
        recommendations.append("Complete 6-DOF vibration data available. Comprehensive machinery analysis possible.")

    return recommendations


def generate_detection_summary(vibration_signals: List[str], rpm_signals: List[str],
                             classification: Dict) -> str:
    """Generate text summary of detection."""
    try:
        summary_parts = []

        # Basic counts
        summary_parts.append(f"Detected {len(vibration_signals)} vibration signals")
        if len(rpm_signals) > 0:
            summary_parts.append(f"and {len(rpm_signals)} RPM signals")

        # Analysis of vibration signals
        if vibration_signals:
            axes = [classification.get(s, {}).get('axis', 'Unknown') for s in vibration_signals]
            locations = [classification.get(s, {}).get('location', 'Unknown') for s in vibration_signals]

            unique_axes = set(axes) - {'Unknown'}
            unique_locations = set(locations) - {'Unknown'}

            if unique_axes:
                axes_str = ", ".join(sorted(unique_axes))
                summary_parts.append(f"covering {axes_str} axes")

            if unique_locations:
                locations_str = ", ".join(sorted(unique_locations))
                summary_parts.append(f"at {locations_str} locations")

        # Average detection confidence
        confidences = [classification.get(s, {}).get('confidence', 0) for s in vibration_signals + rpm_signals]
        if confidences:
            avg_confidence = np.mean(confidences)
            summary_parts.append(f"(avg. confidence: {avg_confidence:.0f}%)")

        return ". ".join(summary_parts) + "."

    except Exception as e:
        return f"Detection completed with {len(vibration_signals)} vibration signals detected."


def get_signal_recommendations(signal_classification: Dict[str, Any]) -> List[str]:
    """
    Provide specific recommendations for an individual signal.

    Args:
        signal_classification (dict): Signal classification result

    Returns:
        list: List of analysis recommendations
    """
    recommendations = []

    signal_type = signal_classification.get('type', 'Unknown')
    confidence = signal_classification.get('confidence', 0)

    if confidence < 50:
        recommendations.append("Low confidence in signal classification. Manual verification recommended.")

    if signal_type == 'Vibration':
        axis = signal_classification.get('axis', 'Unknown')
        location = signal_classification.get('location', 'Unknown')
        measurement_type = signal_classification.get('measurement_type', 'Unknown')

        # Recommendations by axis
        if axis == 'X':
            recommendations.append("Horizontal radial vibration. Check for unbalance and misalignment.")
        elif axis == 'Y':
            recommendations.append("Vertical radial vibration. Check for foundation issues and unbalance.")
        elif axis == 'Z':
            recommendations.append("Axial vibration. Check for misalignment and thrust bearing condition.")

        # Recommendations by position
        if location == 'DE':
            recommendations.append("Drive end measurement. Monitor coupling and motor bearing condition.")
        elif location == 'NDE':
            recommendations.append("Non-drive end measurement. Monitor fan/load bearing condition.")

        # Recommendations by measurement type
        if measurement_type == 'Acceleration':
            recommendations.append("Use for high-frequency analysis and bearing fault detection.")
        elif measurement_type == 'Velocity':
            recommendations.append("Use for overall vibration assessment and ISO severity evaluation.")
        elif measurement_type == 'Displacement':
            recommendations.append("Use for low-frequency analysis and shaft position monitoring.")

    elif signal_type == 'RPM':
        recommendations.append("Use for harmonic analysis and order tracking.")
        recommendations.append("Essential for bearing fault frequency calculations.")

    elif signal_type == 'Other':
        recommendations.append("Signal type unclear. Consider renaming for automatic detection.")

    return recommendations


def validate_vibration_channel_setup(signal_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    """
    Validate completeness of vibration channel setup.

    Args:
        signal_dict (dict): Dictionary of signals

    Returns:
        dict: Validation results including missing channels and recommendations
    """
    Logger.log_message_static("Vibration-Detection: Validating vibration channel setup", Logger.DEBUG)

    try:
        # Detect signals
        detection_result = detect_vibration_signals(signal_dict)
        classification = detection_result['classification']
        vibration_signals = detection_result['vibration_signals']

        # Expected channels for complete analysis
        expected_channels = {
            'DE': {'X', 'Y', 'Z'},
            'NDE': {'X', 'Y', 'Z'}
        }

        # Analyze available channels
        available_channels = {'DE': set(), 'NDE': set()}
        for signal in vibration_signals:
            info = classification.get(signal, {})
            location = info.get('location', 'Unknown')
            axis = info.get('axis', 'Unknown')

            if location in available_channels and axis != 'Unknown':
                available_channels[location].add(axis)

        # Identify missing channels
        missing_channels = {}
        completeness_score = 0
        total_expected = sum(len(axes) for axes in expected_channels.values())

        for location, expected_axes in expected_channels.items():
            available_axes = available_channels[location]
            missing_axes = expected_axes - available_axes

            if missing_axes:
                missing_channels[location] = list(missing_axes)

            completeness_score += len(available_axes)

        completeness_percentage = (completeness_score / total_expected) * 100

        # Generate recommendations
        setup_recommendations = []

        if completeness_percentage < 100:
            setup_recommendations.append(f"Channel setup is {completeness_percentage:.0f}% complete.")

            for location, missing_axes in missing_channels.items():
                setup_recommendations.append(f"Missing {location} channels: {', '.join(missing_axes)}.")

        if 'RPM' not in [classification.get(s, {}).get('type') for s in signal_dict.keys()]:
            setup_recommendations.append("No RPM signal detected. Add tachometer for enhanced analysis.")

        if completeness_percentage >= 80:
            setup_recommendations.append("Good channel coverage for machinery analysis.")
        elif completeness_percentage >= 50:
            setup_recommendations.append("Adequate channels for basic vibration analysis.")
        else:
            setup_recommendations.append("Insufficient channels for comprehensive analysis.")

        # Check signal quality
        signal_quality_issues = []
        for signal in vibration_signals:
            info = classification.get(signal, {})
            if info.get('confidence', 0) < 70:
                signal_quality_issues.append(f"Low confidence ({info.get('confidence', 0):.0f}%) for {signal}.")

        if signal_quality_issues:
            setup_recommendations.extend(signal_quality_issues)

        return {
            'completeness_percentage': completeness_percentage,
            'available_channels': dict(available_channels),
            'missing_channels': missing_channels,
            'total_vibration_signals': len(vibration_signals),
            'has_rpm_signal': len(detection_result['rpm_signals']) > 0,
            'setup_recommendations': setup_recommendations,
            'analysis_capabilities': determine_analysis_capabilities(available_channels, len(detection_result['rpm_signals']) > 0)
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Detection: Error validating channel setup: {e}", Logger.ERROR)
        return {
            'completeness_percentage': 0,
            'error': str(e),
            'setup_recommendations': ['Channel validation failed. Manual check required.']
        }


def determine_analysis_capabilities(available_channels: Dict[str, set], has_rpm: bool) -> List[str]:
    """Determine analysis capabilities based on available channels."""
    capabilities = []

    total_axes = sum(len(axes) for axes in available_channels.values())

    # Basic capabilities
    if total_axes >= 1:
        capabilities.append("Basic vibration metrics")
        capabilities.append("FFT analysis")
        capabilities.append("Time domain analysis")

    # Advanced capabilities
    if total_axes >= 2:
        capabilities.append("Cross-channel correlation analysis")

    if has_rpm:
        capabilities.append("Harmonic analysis")
        capabilities.append("Order tracking")
        capabilities.append("Bearing fault frequency analysis")

    # Specific analyses by axis
    for location, axes in available_channels.items():
        if 'X' in axes and 'Y' in axes:
            capabilities.append(f"{location} radial vibration analysis")
            capabilities.append(f"{location} orbit analysis")

        if 'Z' in axes:
            capabilities.append(f"{location} axial vibration analysis")

    # Complex analyses
    if len(available_channels['DE']) >= 2 and len(available_channels['NDE']) >= 2:
        capabilities.append("Multi-location bearing analysis")
        capabilities.append("Machine condition assessment")

    if total_axes >= 6:  # Complete 3D measurements at both positions
        capabilities.append("Complete 6-DOF machinery analysis")
        capabilities.append("Advanced fault diagnostics")
        capabilities.append("Modal analysis")

    return capabilities


# Helper functions for testing and debugging
def debug_signal_detection(signal_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    """Debug function to print detailed information about detection."""
    print("=== VIBRATION SIGNAL DETECTION DEBUG ===")

    detection_result = detect_vibration_signals(signal_dict)

    print(f"Total signals: {len(signal_dict)}")
    print(f"Vibration signals: {len(detection_result['vibration_signals'])}")
    print(f"RPM signals: {len(detection_result['rpm_signals'])}")
    print(f"Other signals: {len(detection_result['other_signals'])}")

    print("\nDetailed classification:")
    for signal_name, classification in detection_result['classification'].items():
        print(f"\n{signal_name}:")
        for key, value in classification.items():
            print(f"  {key}: {value}")

    print(f"\nRecommendations:")
    for rec in detection_result['recommendations']:
        print(f"  - {rec}")

    print(f"\nSummary: {detection_result['detection_summary']}")


if __name__ == "__main__":
    # Test with sample data
    import numpy as np

    # Create test signals
    time = np.linspace(0, 10, 10000)
    test_signals = {
        'DE_X_Accel': (time, np.random.randn(10000) * 0.1),
        'DE_Y_Accel': (time, np.random.randn(10000) * 0.1),
        'NDE_X_Velocity': (time, np.random.randn(10000) * 0.05),
        'RPM_Signal': (time, 1800 + np.random.randn(10000) * 10),
        'Temperature': (time, 50 + np.random.randn(10000) * 2)
    }

    # Run detection debug
    debug_signal_detection(test_signals)

    # Test validation
    validation_result = validate_vibration_channel_setup(test_signals)
    print("\n=== CHANNEL SETUP VALIDATION ===")
    print(f"Completeness: {validation_result['completeness_percentage']:.0f}%")
    print("Available channels:")
    for location, axes in validation_result['available_channels'].items():
        print(f"  {location}: {', '.join(sorted(axes))}")
    print("Missing channels:")
    for location, axes in validation_result.get('missing_channels', {}).items():
        print(f"  {location}: {', '.join(sorted(axes))}")
    print("\nRecommendations:")
    for rec in validation_result['setup_recommendations']:
        print(f"  - {rec}")
    print("\nAnalysis capabilities:")
    for cap in validation_result['analysis_capabilities']:
        print(f"  - {cap}")