"""
Bearing analysis module for vibration diagnostics.

This module provides functions for analyzing bearing-related vibration signatures,
fault frequency calculations, and bearing health assessment.
"""
import numpy as np


def calculate_fault_frequencies(bearing_specs, rpm):
    """
    Calculate theoretical bearing fault frequencies.

    Args:
        bearing_specs (dict): Dictionary containing bearing specifications:
            - ball_diameter: Ball/roller diameter in mm
            - pitch_diameter: Pitch circle diameter in mm
            - contact_angle: Contact angle in degrees (0 for radial bearings)
            - num_balls: Number of rolling elements
        rpm (float): Shaft rotation speed in RPM

    Returns:
        dict: Dictionary with calculated fault frequencies (BPFO, BPFI, BSF, FTF)
    """
    # Extract bearing parameters
    bd = bearing_specs['ball_diameter']
    pd = bearing_specs['pitch_diameter']
    angle = bearing_specs['contact_angle']
    n = bearing_specs['num_balls']

    # Convert RPM to Hz
    shaft_freq = rpm / 60.0

    # Calculate contact angle factor
    contact_factor = np.cos(np.radians(angle))

    # Calculate fault frequencies
    bpfo = (n / 2) * shaft_freq * (1 - bd * contact_factor / pd)  # Ball Pass Frequency Outer race
    bpfi = (n / 2) * shaft_freq * (1 + bd * contact_factor / pd)  # Ball Pass Frequency Inner race
    bsf = (pd / bd) * shaft_freq * (1 - (bd * contact_factor / pd) ** 2)  # Ball Spin Frequency
    ftf = shaft_freq / 2 * (1 - bd * contact_factor / pd)  # Fundamental Train Frequency

    return {
        'BPFO': bpfo,
        'BPFI': bpfi,
        'BSF': bsf,
        'FTF': ftf,
        'shaft': shaft_freq
    }


def bearing_health_assessment(envelope_spectrum, fault_frequencies, tolerance=0.02):
    """
    Assess bearing health by comparing envelope spectrum peaks with fault frequencies.

    Args:
        envelope_spectrum (dict): Dictionary with 'frequencies' and 'amplitudes'
        fault_frequencies (dict): Dictionary with theoretical fault frequencies
            (BPFO, BPFI, BSF, FTF)
        tolerance (float): Frequency matching tolerance as a percentage

    Returns:
        dict: Assessment results with matching peaks and health indicators
    """
    import numpy as np
    from scipy.signal import find_peaks

    # Extract data from envelope spectrum
    frequencies = envelope_spectrum['frequencies']
    amplitudes = envelope_spectrum['amplitudes']

    # Find significant peaks in the spectrum
    peaks, _ = find_peaks(amplitudes, height=np.mean(amplitudes) + np.std(amplitudes))
    peak_freqs = frequencies[peaks]
    peak_amps = amplitudes[peaks]

    # Initialize results
    results = {
        'matches': {},
        'health_score': 100,
        'fault_indicators': []
    }

    # Check each fault frequency
    for fault_type, freq in fault_frequencies.items():
        # Look for matches in the first 3 harmonics
        harmonics = [freq * i for i in range(1, 4)]
        matches = []

        for harmonic in harmonics:
            # Find peaks near this harmonic
            for i, peak_freq in enumerate(peak_freqs):
                if abs(peak_freq - harmonic) <= harmonic * tolerance:
                    matches.append({
                        'frequency': peak_freq,
                        'amplitude': peak_amps[i],
                        'harmonic': harmonic,
                        'order': round(peak_freq / freq, 1)
                    })

        results['matches'][fault_type] = matches

        # If we found significant matches, reduce health score
        if matches:
            # Higher reduction for inner race and roller defects
            if fault_type == 'BPFI':
                results['health_score'] -= 25 * len(matches)
                results['fault_indicators'].append(f"Inner race defect (confidence: {len(matches) * 33}%)")
            elif fault_type == 'BPFO':
                results['health_score'] -= 20 * len(matches)
                results['fault_indicators'].append(f"Outer race defect (confidence: {len(matches) * 33}%)")
            elif fault_type == 'BSF':
                results['health_score'] -= 15 * len(matches)
                results['fault_indicators'].append(f"Ball/roller defect (confidence: {len(matches) * 33}%)")
            elif fault_type == 'FTF':
                results['health_score'] -= 10 * len(matches)
                results['fault_indicators'].append(f"Cage defect (confidence: {len(matches) * 33}%)")

    # Ensure health score stays in valid range
    results['health_score'] = max(0, min(100, results['health_score']))

    return results