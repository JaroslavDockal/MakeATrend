"""
Vibration severity assessment according to ISO standards.

This module provides functions for evaluating machine vibration severity
according to ISO 10816 and other industry standards, including:
- Threshold values for different machine classes
- Severity zone assessment (A, B, C, D)
- Speed-dependent severity adjustments
- Monitoring recommendations based on severity
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from utils.logger import Logger


def get_iso10816_thresholds(machine_class: str) -> Dict[str, float]:
    """
    Provides vibration threshold values according to ISO 10816.

    Args:
        machine_class (str): Machine class (I, II, III, IV)

    Returns:
        dict: Threshold values for zones A, B, C [mm/s]
    """
    thresholds = {
        'I': {    # Small machines < 15 kW
            'A': 0.71,
            'B': 1.8,
            'C': 4.5,
            'description': 'Small machines (< 15 kW)'
        },
        'II': {   # Medium machines 15-75 kW
            'A': 1.12,
            'B': 2.8,
            'C': 7.1,
            'description': 'Medium machines (15-75 kW) on rigid foundations'
        },
        'III': {  # Large machines 75-300 kW on rigid foundations
            'A': 1.8,
            'B': 4.5,
            'C': 11.2,
            'description': 'Large machines (75-300 kW) on rigid foundations'
        },
        'IV': {   # Large machines > 300 kW on flexible foundations
            'A': 2.8,
            'B': 7.1,
            'C': 18.0,
            'description': 'Large machines (> 300 kW) on flexible foundations'
        }
    }

    return thresholds.get(machine_class, thresholds['II'])


def assess_vibration_severity_iso10816(rms_velocity: float, machine_class: str) -> Dict[str, Any]:
    """
    Assess vibration severity according to ISO 10816.

    Args:
        rms_velocity (float): RMS velocity value in mm/s
        machine_class (str): Machine class (I, II, III, IV)

    Returns:
        dict: Assessment results including severity zone and recommendations
    """
    thresholds = get_iso10816_thresholds(machine_class)

    # Determine severity zone
    if rms_velocity <= thresholds['A']:
        severity_zone = 'A'
        severity_description = 'Good'
        action_required = 'No action required. Continue regular monitoring.'
    elif rms_velocity <= thresholds['B']:
        severity_zone = 'B'
        severity_description = 'Acceptable'
        action_required = 'Acceptable for long-term operation. Monitor for trends.'
    elif rms_velocity <= thresholds['C']:
        severity_zone = 'C'
        severity_description = 'Restricted'
        action_required = 'Limited operation time recommended. Plan maintenance.'
    else:
        severity_zone = 'D'
        severity_description = 'Unacceptable'
        action_required = 'Unacceptable vibration levels. Risk of damage. Immediate action required.'

    # Compile results
    results = {
        'severity_zone': severity_zone,
        'severity_description': severity_description,
        'action_required': action_required,
        'machine_class': machine_class,
        'machine_description': thresholds['description'],
        'thresholds': {k: v for k, v in thresholds.items() if k in ['A', 'B', 'C']},
        'rms_velocity': rms_velocity,
        'evaluation_standard': 'ISO 10816',
        'monitoring_recommendations': get_monitoring_recommendations(severity_zone, machine_class)
    }

    return results


def assess_speed_dependent_severity(rms_velocity: float, operating_speed: float,
                                  machine_class: str) -> Dict[str, Any]:
    """
    Assess severity considering operating speed.

    Args:
        rms_velocity (float): RMS velocity [mm/s]
        operating_speed (float): Operating speed [RPM]
        machine_class (str): Machine class

    Returns:
        dict: Speed-adjusted assessment
    """
    try:
        # Standard reference speed
        reference_speed = 1500  # RPM (50 Hz, 2-pole motor)

        # Calculate correction factor
        if operating_speed < 600:      # Low speed
            speed_factor = 0.8
            speed_note = "Low speed operation - reduced vibration limits applied"
        elif operating_speed > 3600:   # High speed
            speed_factor = 1.2
            speed_note = "High speed operation - increased vibration limits applied"
        else:                          # Normal range
            speed_factor = 1.0
            speed_note = "Normal speed range - standard limits applied"

        # Adjusted thresholds
        base_thresholds = get_iso10816_thresholds(machine_class)
        adjusted_thresholds = {
            zone: value * speed_factor
            for zone, value in base_thresholds.items()
            if zone in ['A', 'B', 'C']
        }

        # Determine adjusted zone
        if rms_velocity <= adjusted_thresholds['A']:
            adjusted_zone = 'A'
        elif rms_velocity <= adjusted_thresholds['B']:
            adjusted_zone = 'B'
        elif rms_velocity <= adjusted_thresholds['C']:
            adjusted_zone = 'C'
        else:
            adjusted_zone = 'D'

        return {
            'speed_factor': speed_factor,
            'speed_note': speed_note,
            'adjusted_thresholds': adjusted_thresholds,
            'adjusted_zone': adjusted_zone,
            'reference_speed_rpm': reference_speed,
            'speed_category': 'Low' if operating_speed < 600 else
                            'High' if operating_speed > 3600 else 'Normal'
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Severity: Error in speed adjustment: {e}", Logger.WARNING)
        return {'speed_note': 'Speed adjustment calculation failed'}


def get_monitoring_recommendations(severity_zone: str, machine_class: str) -> List[str]:
    """Provides monitoring recommendations based on severity zone."""
    recommendations = []

    if severity_zone == 'A':
        recommendations.extend([
            "Continue normal monitoring schedule",
            "Annual or semi-annual vibration measurements sufficient",
            "Focus on trending analysis to detect gradual changes"
        ])

    elif severity_zone == 'B':
        recommendations.extend([
            "Increase monitoring frequency to quarterly",
            "Monitor trends carefully for any increasing patterns",
            "Consider route-based monitoring program"
        ])

    elif severity_zone == 'C':
        recommendations.extend([
            "Implement monthly monitoring minimum",
            "Consider continuous monitoring if critical equipment",
            "Investigate root cause of elevated vibration",
            "Plan corrective maintenance"
        ])

    elif severity_zone == 'D':
        recommendations.extend([
            "Immediate action required",
            "Continuous monitoring until corrective action",
            "Consider equipment shutdown if operationally feasible",
            "Emergency maintenance planning"
        ])

    # Specific recommendations based on machine class
    if machine_class in ['III', 'IV']:
        recommendations.append("Large machine - consider impact on production planning")

    return recommendations


def get_machine_class_from_power(power_kw: float) -> str:
    """
    Determine ISO 10816 machine class based on power rating.

    Args:
        power_kw (float): Machine power in kilowatts

    Returns:
        str: Machine class (I, II, III, or IV)
    """
    if power_kw < 15:
        return 'I'
    elif power_kw < 75:
        return 'II'
    elif power_kw < 300:
        return 'III'
    else:
        return 'IV'


def convert_vibration_units(value: float, from_unit: str, to_unit: str,
                           frequency_hz: Optional[float] = None) -> float:
    """
    Convert between different vibration measurement units.

    Args:
        value (float): Value to convert
        from_unit (str): Source unit (g, mm/s2, mm/s, mm, etc.)
        to_unit (str): Target unit
        frequency_hz (float, optional): Frequency for conversion between
                                        displacement, velocity and acceleration

    Returns:
        float: Converted value

    Raises:
        ValueError: If conversion requires frequency but none provided
        ValueError: If conversion between units is not supported
    """
    # Normalize units to lowercase
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    # If units are already the same, return the value
    if from_unit == to_unit:
        return value

    # Acceleration conversions
    if from_unit == 'g':
        # Convert g to m/s²
        value_m_s2 = value * 9.80665

        if to_unit == 'm/s2' or to_unit == 'm/s²':
            return value_m_s2
        elif to_unit == 'mm/s2' or to_unit == 'mm/s²':
            return value_m_s2 * 1000

    elif from_unit == 'm/s2' or from_unit == 'm/s²':
        if to_unit == 'g':
            return value / 9.80665
        elif to_unit == 'mm/s2' or to_unit == 'mm/s²':
            return value * 1000

    elif from_unit == 'mm/s2' or from_unit == 'mm/s²':
        if to_unit == 'g':
            return value / 9806.65
        elif to_unit == 'm/s2' or to_unit == 'm/s²':
            return value / 1000

    # Conversions requiring frequency
    if frequency_hz is None and (
        (from_unit in ['mm/s', 'm/s'] and to_unit in ['mm', 'um', 'μm', 'mil']) or
        (from_unit in ['mm', 'um', 'μm', 'mil'] and to_unit in ['mm/s', 'm/s']) or
        (from_unit in ['mm/s2', 'm/s2', 'g'] and to_unit in ['mm/s', 'm/s']) or
        (from_unit in ['mm/s', 'm/s'] and to_unit in ['mm/s2', 'm/s2', 'g'])
    ):
        raise ValueError("Frequency must be provided for conversion between displacement, velocity, and acceleration")

    # Convert between displacement, velocity, and acceleration
    if frequency_hz:
        omega = 2 * np.pi * frequency_hz

        # Convert to displacement (mm) as intermediate step
        if from_unit == 'mm/s':
            displacement_mm = value / omega
        elif from_unit == 'm/s':
            displacement_mm = (value * 1000) / omega
        elif from_unit == 'mm/s2' or from_unit == 'mm/s²':
            displacement_mm = value / (omega * omega)
        elif from_unit == 'm/s2' or from_unit == 'm/s²':
            displacement_mm = (value * 1000) / (omega * omega)
        elif from_unit == 'g':
            displacement_mm = (value * 9806.65) / (omega * omega)
        elif from_unit == 'mm':
            displacement_mm = value
        elif from_unit == 'um' or from_unit == 'μm':
            displacement_mm = value / 1000
        elif from_unit == 'mil':
            displacement_mm = value * 0.0254
        else:
            raise ValueError(f"Unsupported source unit: {from_unit}")

        # Convert from displacement to target unit
        if to_unit == 'mm':
            return displacement_mm
        elif to_unit == 'um' or to_unit == 'μm':
            return displacement_mm * 1000
        elif to_unit == 'mil':
            return displacement_mm / 0.0254
        elif to_unit == 'mm/s':
            return displacement_mm * omega
        elif to_unit == 'm/s':
            return displacement_mm * omega / 1000
        elif to_unit == 'mm/s2' or to_unit == 'mm/s²':
            return displacement_mm * omega * omega
        elif to_unit == 'm/s2' or to_unit == 'm/s²':
            return displacement_mm * omega * omega / 1000
        elif to_unit == 'g':
            return displacement_mm * omega * omega / 9806.65

    # If we got here, the conversion is not supported
    raise ValueError(f"Conversion from {from_unit} to {to_unit} is not supported")


def test_severity_assessment():
    """Test vibration severity assessment."""
    print("\n=== ISO 10816 VIBRATION SEVERITY TEST ===")

    test_cases = [
        {'rms': 0.5, 'class': 'I', 'expected_zone': 'A'},
        {'rms': 1.0, 'class': 'I', 'expected_zone': 'B'},
        {'rms': 3.0, 'class': 'I', 'expected_zone': 'C'},
        {'rms': 5.0, 'class': 'I', 'expected_zone': 'D'},

        {'rms': 0.5, 'class': 'II', 'expected_zone': 'A'},
        {'rms': 2.0, 'class': 'II', 'expected_zone': 'B'},
        {'rms': 5.0, 'class': 'II', 'expected_zone': 'C'},
        {'rms': 8.0, 'class': 'II', 'expected_zone': 'D'},

        {'rms': 1.0, 'class': 'III', 'expected_zone': 'A'},
        {'rms': 3.0, 'class': 'III', 'expected_zone': 'B'},
        {'rms': 7.0, 'class': 'III', 'expected_zone': 'C'},
        {'rms': 12.0, 'class': 'III', 'expected_zone': 'D'},

        {'rms': 2.0, 'class': 'IV', 'expected_zone': 'A'},
        {'rms': 5.0, 'class': 'IV', 'expected_zone': 'B'},
        {'rms': 10.0, 'class': 'IV', 'expected_zone': 'C'},
        {'rms': 20.0, 'class': 'IV', 'expected_zone': 'D'},
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"  RMS: {case['rms']} mm/s, Class: {case['class']}")

        result = assess_vibration_severity_iso10816(case['rms'], case['class'])

        print(f"  Result: Zone {result['severity_zone']} ({result['severity_description']})")
        print(f"  Expected: Zone {case['expected_zone']}")
        print(f"  Match: {'✓' if result['severity_zone'] == case['expected_zone'] else '✗'}")
        print(f"  Action: {result['action_required']}")


def test_unit_conversion():
    """Test unit conversions."""
    print("\n=== UNIT CONVERSION TEST ===")

    test_conversions = [
        {'value': 1.0, 'from': 'g', 'to': 'mm/s2', 'freq': None},
        {'value': 1.0, 'from': 'mm/s', 'to': 'mm', 'freq': 100},
        {'value': 10.0, 'from': 'mm/s2', 'to': 'mm/s', 'freq': 100},
    ]

    for conversion in test_conversions:
        try:
            result = convert_vibration_units(
                conversion['value'],
                conversion['from'],
                conversion['to'],
                conversion['freq']
            )
            print(f"  {conversion['value']} {conversion['from']} = {result:.3f} {conversion['to']}")
        except Exception as e:
            print(f"  Conversion failed: {e}")


if __name__ == "__main__":
    test_severity_assessment()
    test_unit_conversion()