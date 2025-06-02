"""
Vibration unit conversion utilities.

This module provides comprehensive functions for converting between different
vibration measurement units, including:

- Acceleration units (g, m/s², mm/s²)
- Velocity units (mm/s, m/s)
- Displacement units (mm, μm, mil)
- Conversions between domains (acceleration ↔ velocity ↔ displacement)
"""

import numpy as np
from typing import Optional, Union
from utils.logger import Logger


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


def convert_array(values: np.ndarray, from_unit: str, to_unit: str,
                  frequency_hz: Optional[float] = None) -> np.ndarray:
    """
    Convert an array of values from one unit to another.

    Args:
        values (np.ndarray): Array of values to convert
        from_unit (str): Source unit
        to_unit (str): Target unit
        frequency_hz (float, optional): Frequency for conversion

    Returns:
        np.ndarray: Converted values
    """
    try:
        # Simple scaling conversions
        scale_factor = None

        # Simple acceleration conversions
        if from_unit == 'g' and to_unit == 'm/s2':
            scale_factor = 9.80665
        elif from_unit == 'g' and to_unit == 'mm/s2':
            scale_factor = 9806.65
        elif from_unit == 'm/s2' and to_unit == 'g':
            scale_factor = 1 / 9.80665
        elif from_unit == 'm/s2' and to_unit == 'mm/s2':
            scale_factor = 1000
        elif from_unit == 'mm/s2' and to_unit == 'g':
            scale_factor = 1 / 9806.65
        elif from_unit == 'mm/s2' and to_unit == 'm/s2':
            scale_factor = 0.001

        # Simple velocity conversions
        elif from_unit == 'm/s' and to_unit == 'mm/s':
            scale_factor = 1000
        elif from_unit == 'mm/s' and to_unit == 'm/s':
            scale_factor = 0.001

        # Simple displacement conversions
        elif from_unit == 'mm' and to_unit == 'um':
            scale_factor = 1000
        elif from_unit == 'mm' and to_unit == 'mil':
            scale_factor = 1 / 0.0254
        elif from_unit == 'um' and to_unit == 'mm':
            scale_factor = 0.001
        elif from_unit == 'um' and to_unit == 'mil':
            scale_factor = 0.03937
        elif from_unit == 'mil' and to_unit == 'mm':
            scale_factor = 0.0254
        elif from_unit == 'mil' and to_unit == 'um':
            scale_factor = 25.4

        # Apply simple scaling if possible
        if scale_factor is not None:
            return values * scale_factor

        # For conversions requiring frequency, process each value
        if frequency_hz is not None:
            converted = np.zeros_like(values)
            for i, val in enumerate(values):
                converted[i] = convert_vibration_units(val, from_unit, to_unit, frequency_hz)
            return converted
        else:
            raise ValueError(f"Frequency required for conversion from {from_unit} to {to_unit}")

    except Exception as e:
        Logger.log_message_static(f"Vibration-Units: Error converting array: {e}", Logger.ERROR)
        return values  # Return original if conversion fails


def detect_unit_from_signal_name(signal_name: str) -> str:
    """
    Attempt to detect measurement unit from signal name.

    Args:
        signal_name (str): Signal name

    Returns:
        str: Detected unit or 'Unknown'
    """
    name_lower = signal_name.lower()

    # Check for acceleration units
    if any(s in name_lower for s in ['_g', '(g)', ' g ', 'accel']):
        return 'g'
    elif any(s in name_lower for s in ['m/s2', 'm/s²']):
        return 'm/s2'
    elif any(s in name_lower for s in ['mm/s2', 'mm/s²']):
        return 'mm/s2'

    # Check for velocity units
    elif any(s in name_lower for s in ['mm/s', 'mmps']):
        return 'mm/s'
    elif any(s in name_lower for s in ['m/s', 'mps']) and not any(s in name_lower for s in ['mm/s', 'mmps']):
        return 'm/s'

    # Check for displacement units
    elif any(s in name_lower for s in ['mm ', '(mm)', '_mm']):
        return 'mm'
    elif any(s in name_lower for s in ['um', 'μm', 'micron']):
        return 'um'
    elif any(s in name_lower for s in ['mil', 'thou']):
        return 'mil'

    return 'Unknown'


def get_base_unit_type(unit: str) -> str:
    """
    Get the base unit type (acceleration, velocity, displacement).

    Args:
        unit (str): Unit string

    Returns:
        str: 'acceleration', 'velocity', 'displacement', or 'unknown'
    """
    unit = unit.lower()

    if unit in ['g', 'm/s2', 'm/s²', 'mm/s2', 'mm/s²']:
        return 'acceleration'
    elif unit in ['m/s', 'mm/s']:
        return 'velocity'
    elif unit in ['m', 'mm', 'um', 'μm', 'mil']:
        return 'displacement'
    else:
        return 'unknown'


def list_available_conversions(from_unit: str) -> list:
    """
    List all possible unit conversions from a given unit.

    Args:
        from_unit (str): Source unit

    Returns:
        list: List of possible target units
    """
    from_unit = from_unit.lower()
    base_type = get_base_unit_type(from_unit)

    direct_conversions = {
        'g': ['m/s2', 'mm/s2'],
        'm/s2': ['g', 'mm/s2'],
        'mm/s2': ['g', 'm/s2'],
        'm/s': ['mm/s'],
        'mm/s': ['m/s'],
        'mm': ['um', 'mil'],
        'um': ['mm', 'mil'],
        'mil': ['mm', 'um']
    }

    frequency_conversions = {
        'acceleration': ['m/s', 'mm/s', 'm', 'mm', 'um', 'mil'],
        'velocity': ['g', 'm/s2', 'mm/s2', 'm', 'mm', 'um', 'mil'],
        'displacement': ['g', 'm/s2', 'mm/s2', 'm/s', 'mm/s']
    }

    result = []

    # Add direct conversions
    if from_unit in direct_conversions:
        result.extend(direct_conversions[from_unit])

    # Add conversions requiring frequency
    if base_type in frequency_conversions:
        for unit in frequency_conversions[base_type]:
            if unit not in result and unit != from_unit:
                result.append(f"{unit} (requires frequency)")

    return result


if __name__ == "__main__":
    # Test unit conversions
    print("=== VIBRATION UNIT CONVERSION TESTS ===")

    test_cases = [
        (1.0, 'g', 'm/s2', None),
        (9.81, 'm/s2', 'g', None),
        (10.0, 'mm/s', 'mm', 100),  # 100 Hz
        (10.0, 'mm/s2', 'mm/s', 100),  # 100 Hz
    ]

    for value, from_unit, to_unit, freq in test_cases:
        try:
            result = convert_vibration_units(value, from_unit, to_unit, freq)
            print(f"{value} {from_unit} = {result:.6f} {to_unit}" +
                  (f" at {freq} Hz" if freq else ""))
        except Exception as e:
            print(f"Error converting {value} {from_unit} to {to_unit}: {e}")

    # Test unit detection
    test_signals = [
        "Motor_DE_X_Accel(g)",
        "Bearing_Velocity_mm/s",
        "Shaft_Displacement_um"
    ]

    print("\n=== UNIT DETECTION TESTS ===")
    for signal in test_signals:
        unit = detect_unit_from_signal_name(signal)
        print(f"Signal: {signal} → Unit: {unit}")