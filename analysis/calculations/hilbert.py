"""
Hilbert transform analysis: envelope, phase, and instantaneous frequency.

This module provides Hilbert transform-based analysis functions:
- Amplitude envelope extraction
- Instantaneous phase analysis
- Instantaneous frequency calculation
- Energy analysis in time domain
- Phase-based signal characteristics

The Hilbert transform creates an analytic signal that provides insights into
signal modulation, envelope detection, and time-varying frequency content.
"""

import numpy as np
from scipy.signal import hilbert
from PySide6.QtWidgets import QMessageBox

from .common import safe_prepare_signal, extended_prepare_signal, safe_sample_rate, validate_analysis_inputs
from utils.logger import Logger


def calculate_hilbert_analysis(time_arr, values, dialog=None, title="Hilbert Analysis"):
    """
    Perform Hilbert transform to extract amplitude envelope, phase, and instantaneous frequency.

    The Hilbert transform creates an analytic signal from a real-valued signal,
    enabling extraction of instantaneous amplitude, phase, and frequency.
    This is particularly useful for analyzing modulated signals and detecting
    amplitude/frequency variations over time.

    Args:
        time_arr (np.ndarray): Time values corresponding to signal samples.
        values (np.ndarray): Signal amplitude values to analyze.
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "Hilbert Analysis".

    Returns:
        dict or None: Dictionary containing Hilbert analysis results:
            - Amplitude Envelope: Instantaneous amplitude of the signal
            - Unwrapped Phase: Continuous phase progression
            - Instantaneous Frequency (Hz): Rate of phase change
            - Mean Amplitude: Average envelope value
            - Max Amplitude: Maximum envelope value
            - Mean Frequency (Hz): Average instantaneous frequency
            - Median Frequency (Hz): Median instantaneous frequency
            - Max Frequency (Hz): Maximum instantaneous frequency
            - Phase Range (rad): Total phase variation
            - Frequency Statistics: Detailed frequency domain stats

        Returns None if validation fails or user cancels.

    Example:
        >>> t = np.linspace(0, 1, 1000)
        >>> # AM modulated signal: carrier modulated by envelope
        >>> carrier = np.sin(2*np.pi*50*t)
        >>> envelope = 1 + 0.5*np.sin(2*np.pi*5*t)
        >>> signal = envelope * carrier
        >>> result = calculate_hilbert_analysis(t, signal)
        >>> print(f"Mean frequency: {result['Mean Frequency (Hz)']:.1f} Hz")
        Mean frequency: 50.0 Hz
    """
    Logger.log_message_static(f"Calculations-Hilbert: Starting Hilbert transform analysis", Logger.DEBUG)

    # Validate inputs
    is_valid, error_msg, sample_rate = validate_analysis_inputs(time_arr, values, min_length=4,
                                                                require_positive_sample_rate=True)
    if not is_valid:
        Logger.log_message_static(f"Calculations-Hilbert: Input validation failed: {error_msg}", Logger.ERROR)
        if dialog:
            QMessageBox.warning(dialog, title, f"Hilbert Analysis Error:\n{error_msg}")
        return None

    # Prepare signal with extended validation for Hilbert transform
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Calculations-Hilbert: Signal validation failed", Logger.WARNING)
        return None

    # Extended preparation for analyses requiring positive values
    extended_values = extended_prepare_signal(processed_values, dialog, title)
    if extended_values is None:
        Logger.log_message_static("Calculations-Hilbert: Extended signal preparation failed", Logger.WARNING)
        return None

    try:
        # Compute Hilbert transform to create analytic signal
        analytic_signal = hilbert(extended_values)

        # Extract amplitude envelope (magnitude of analytic signal)
        amplitude_envelope = np.abs(analytic_signal)

        # Extract instantaneous phase
        instantaneous_phase = np.angle(analytic_signal)

        # Unwrap phase to get continuous phase progression
        unwrapped_phase = np.unwrap(instantaneous_phase)

        # Calculate instantaneous frequency as derivative of phase
        dt = 1.0 / sample_rate
        instantaneous_freq = np.diff(unwrapped_phase) / (2.0 * np.pi * dt)

        # Extend instantaneous frequency to match original length
        # (derivative reduces length by 1)
        instantaneous_freq = np.append(instantaneous_freq, instantaneous_freq[-1])

        # Calculate envelope statistics
        mean_amplitude = np.mean(amplitude_envelope)
        max_amplitude = np.max(amplitude_envelope)
        min_amplitude = np.min(amplitude_envelope)
        amplitude_std = np.std(amplitude_envelope)

        # Calculate phase statistics
        phase_range = np.max(unwrapped_phase) - np.min(unwrapped_phase)
        phase_std = np.std(unwrapped_phase)

        # Calculate frequency statistics (remove outliers for better statistics)
        # Remove extreme frequency values that might be artifacts
        freq_median = np.median(instantaneous_freq)
        freq_mad = np.median(np.abs(instantaneous_freq - freq_median))  # Median Absolute Deviation

        # Define outliers as values more than 5*MAD from median
        outlier_threshold = 5 * freq_mad
        freq_mask = np.abs(instantaneous_freq - freq_median) <= outlier_threshold
        filtered_freq = instantaneous_freq[freq_mask]

        if len(filtered_freq) > 0:
            mean_frequency = np.mean(filtered_freq)
            median_frequency = np.median(filtered_freq)
            max_frequency = np.max(filtered_freq)
            min_frequency = np.min(filtered_freq)
            freq_std = np.std(filtered_freq)
        else:
            # Fallback if all frequencies are considered outliers
            mean_frequency = np.mean(instantaneous_freq)
            median_frequency = np.median(instantaneous_freq)
            max_frequency = np.max(instantaneous_freq)
            min_frequency = np.min(instantaneous_freq)
            freq_std = np.std(instantaneous_freq)

        # Calculate modulation characteristics
        # Amplitude modulation index (if signal has modulation)
        if mean_amplitude > 0:
            am_index = (max_amplitude - min_amplitude) / (max_amplitude + min_amplitude)
        else:
            am_index = 0.0

        # Frequency modulation characteristics
        freq_deviation = max_frequency - min_frequency
        freq_modulation_index = freq_deviation / mean_frequency if mean_frequency > 0 else 0.0

        # Phase coherence (measure of phase stability)
        # Calculate phase difference variations
        phase_diff = np.diff(unwrapped_phase)
        phase_coherence = 1.0 - (np.std(phase_diff) / np.mean(np.abs(phase_diff))) if np.mean(
            np.abs(phase_diff)) > 0 else 0.0
        phase_coherence = max(0.0, min(1.0, phase_coherence))  # Clamp to [0,1]

        # Energy in envelope vs. carrier
        envelope_energy = np.sum(amplitude_envelope ** 2)
        carrier_energy = np.sum(extended_values ** 2)
        energy_ratio = envelope_energy / carrier_energy if carrier_energy > 0 else 0.0

        # Advanced envelope characteristics
        # Envelope spectral centroid (if envelope has frequency content)
        try:
            envelope_fft = np.fft.rfft(amplitude_envelope - np.mean(amplitude_envelope))
            envelope_freqs = np.fft.rfftfreq(len(amplitude_envelope), dt)
            envelope_magnitude = np.abs(envelope_fft)

            if np.sum(envelope_magnitude) > 0:
                envelope_centroid = np.sum(envelope_freqs * envelope_magnitude) / np.sum(envelope_magnitude)
            else:
                envelope_centroid = 0.0
        except:
            envelope_centroid = 0.0

        # Build comprehensive results dictionary
        results = {
            # Core Hilbert transform outputs
            "Amplitude Envelope": amplitude_envelope,
            "Unwrapped Phase": unwrapped_phase,
            "Instantaneous Frequency (Hz)": instantaneous_freq,
            "Instantaneous Phase": instantaneous_phase,

            # Amplitude envelope statistics
            "Mean Amplitude": float(mean_amplitude),
            "Max Amplitude": float(max_amplitude),
            "Min Amplitude": float(min_amplitude),
            "Amplitude Standard Deviation": float(amplitude_std),
            "Amplitude Modulation Index": float(am_index),

            # Phase statistics
            "Phase Range (rad)": float(phase_range),
            "Phase Standard Deviation": float(phase_std),
            "Phase Coherence": float(phase_coherence),

            # Frequency statistics
            "Mean Frequency (Hz)": float(mean_frequency),
            "Median Frequency (Hz)": float(median_frequency),
            "Max Frequency (Hz)": float(max_frequency),
            "Min Frequency (Hz)": float(min_frequency),
            "Frequency Standard Deviation (Hz)": float(freq_std),
            "Frequency Deviation (Hz)": float(freq_deviation),
            "Frequency Modulation Index": float(freq_modulation_index),

            # Advanced characteristics
            "Envelope Spectral Centroid (Hz)": float(envelope_centroid),
            "Energy Ratio (Envelope/Carrier)": float(energy_ratio),
            "Outlier Frequency Samples": int(len(instantaneous_freq) - len(filtered_freq)),
            "Outlier Percentage": float((len(instantaneous_freq) - len(filtered_freq)) / len(instantaneous_freq) * 100),

            # Analysis metadata
            "Sample Rate (Hz)": float(sample_rate),
            "Signal Length": len(extended_values),
            "Analysis Duration (s)": float(len(extended_values) / sample_rate)
        }

        # Add frequency band analysis
        # Divide frequency range into bands and calculate energy in each
        freq_range = max_frequency - min_frequency
        if freq_range > 0:
            num_bands = 5
            band_edges = np.linspace(min_frequency, max_frequency, num_bands + 1)
            band_energies = []

            for i in range(num_bands):
                band_mask = (instantaneous_freq >= band_edges[i]) & (instantaneous_freq < band_edges[i + 1])
                band_energy = np.sum(amplitude_envelope[band_mask] ** 2) if np.any(band_mask) else 0.0
                band_energies.append(band_energy)

                # Add to results
                results[f"Band {i + 1} Energy ({band_edges[i]:.1f}-{band_edges[i + 1]:.1f} Hz)"] = float(band_energy)

            # Dominant frequency band
            if len(band_energies) > 0:
                dominant_band = np.argmax(band_energies)
                results[
                    "Dominant Frequency Band"] = f"Band {dominant_band + 1} ({band_edges[dominant_band]:.1f}-{band_edges[dominant_band + 1]:.1f} Hz)"
                results["Dominant Band Energy Percentage"] = float(
                    band_energies[dominant_band] / np.sum(band_energies) * 100) if np.sum(band_energies) > 0 else 0.0

        Logger.log_message_static(
            f"Calculations-Hilbert: Analysis completed. "
            f"Mean_amp={mean_amplitude:.4f}, Mean_freq={mean_frequency:.2f}Hz, "
            f"Phase_range={phase_range:.2f}rad, AM_index={am_index:.3f}",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Hilbert: Error in Hilbert transform analysis: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Calculations-Hilbert: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None


def calculate_phase_analysis(time_arr, values, dialog=None, title="Phase Analysis"):
    """
    Perform detailed phase analysis using the Hilbert transform.

    Focuses specifically on phase characteristics including phase velocity,
    phase unwrapping, and phase-based signal quality metrics.

    Args:
        time_arr (np.ndarray): Time values corresponding to signal samples.
        values (np.ndarray): Signal amplitude values to analyze.
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "Phase Analysis".

    Returns:
        dict or None: Dictionary containing phase analysis results:
            - Phase: Unwrapped phase progression
            - Phase Velocity: Rate of phase change
            - Phase Statistics: Comprehensive phase metrics
            - Phase Quality: Phase stability indicators

        Returns None if validation fails or user cancels.
    """
    Logger.log_message_static(f"Calculations-Hilbert: Starting phase analysis", Logger.DEBUG)

    # Use the main Hilbert analysis and extract phase-specific results
    hilbert_results = calculate_hilbert_analysis(time_arr, values, dialog, title)
    if hilbert_results is None:
        return None

    try:
        # Extract phase data
        unwrapped_phase = hilbert_results["Unwrapped Phase"]

        # Calculate phase velocity (derivative of phase)
        dt = np.diff(time_arr) if len(time_arr) > 1 else np.array([1.0 / hilbert_results["Sample Rate (Hz)"]])
        phase_velocity = np.diff(unwrapped_phase) / dt

        # Calculate phase acceleration (second derivative)
        if len(phase_velocity) > 1:
            phase_acceleration = np.diff(phase_velocity) / dt[:-1]
        else:
            phase_acceleration = np.array([0.0])

        # Phase statistics
        phase_stats = {
            "Mean Phase": float(np.mean(unwrapped_phase)),
            "Phase Standard Deviation": float(np.std(unwrapped_phase)),
            "Phase Range": float(np.ptp(unwrapped_phase)),
            "Phase Median": float(np.median(unwrapped_phase)),

            # Phase velocity statistics
            "Mean Phase Velocity (rad/s)": float(np.mean(phase_velocity)),
            "Phase Velocity Standard Deviation": float(np.std(phase_velocity)),
            "Max Phase Velocity": float(np.max(np.abs(phase_velocity))),

            # Phase acceleration statistics
            "Mean Phase Acceleration": float(np.mean(phase_acceleration)) if len(phase_acceleration) > 0 else 0.0,
            "Phase Acceleration Standard Deviation": float(np.std(phase_acceleration)) if len(
                phase_acceleration) > 0 else 0.0,
        }

        # Phase quality metrics
        phase_smoothness = 1.0 / (1.0 + np.std(phase_velocity)) if len(phase_velocity) > 0 else 1.0
        phase_linearity = np.corrcoef(time_arr, unwrapped_phase)[0, 1] if len(time_arr) > 1 else 1.0

        results = {
            # Core phase data
            "Time": time_arr,
            "Phase": unwrapped_phase,
            "Phase Velocity": phase_velocity,
            "Phase Acceleration": phase_acceleration,

            # Statistics
            "Phase Stats": phase_stats,

            # Quality metrics
            "Phase Smoothness": float(phase_smoothness),
            "Phase Linearity": float(phase_linearity),
            "Phase Coherence": hilbert_results["Phase Coherence"]
        }

        Logger.log_message_static("Calculations-Hilbert: Phase analysis completed", Logger.DEBUG)
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Hilbert: Error in phase analysis: {str(e)}", Logger.ERROR)
        return None


def calculate_energy_analysis(time_arr, values, dialog=None, title="Energy Analysis"):
    """
    Compute time-domain and frequency-domain energy distribution analysis.

    Analyzes how signal energy is distributed across time and frequency domains,
    including energy in frequency bands and temporal energy variations.

    Args:
        time_arr (np.ndarray): Time values for temporal analysis.
        values (np.ndarray): Signal amplitude values.
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "Energy Analysis".

    Returns:
        dict or None: Dictionary containing energy analysis results:
            - Total Energy (Time Domain): Sum of squared signal values
            - Total Energy (Frequency Domain): Energy from FFT analysis
            - Signal Power: Average power over time
            - RMS Value: Root mean square value
            - Energy Distribution (%): Energy in frequency bands
            - Dominant Frequency Band: Band with highest energy
            - Temporal Energy: Energy variation over time

        Returns None if validation fails or user cancels.
    """
    Logger.log_message_static(f"Calculations-Hilbert: Starting energy analysis", Logger.DEBUG)

    # Validate inputs
    is_valid, error_msg, sample_rate = validate_analysis_inputs(time_arr, values, min_length=4,
                                                                require_positive_sample_rate=True)
    if not is_valid:
        Logger.log_message_static(f"Calculations-Hilbert: Input validation failed: {error_msg}", Logger.ERROR)
        if dialog:
            QMessageBox.warning(dialog, title, f"Energy Analysis Error:\n{error_msg}")
        return None

    # Prepare signal
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Calculations-Hilbert: Signal validation failed", Logger.WARNING)
        return None

    # Extended preparation for frequency domain analysis
    extended_values = extended_prepare_signal(processed_values, dialog, title)
    if extended_values is None:
        Logger.log_message_static("Calculations-Hilbert: Extended signal preparation failed", Logger.WARNING)
        return None

    try:
        # Time domain energy analysis
        total_energy_time = np.sum(extended_values ** 2)
        signal_power = total_energy_time / len(extended_values)
        rms_value = np.sqrt(signal_power)

        # Frequency domain energy analysis
        n = len(extended_values)
        fft_vals = np.fft.rfft(extended_values)
        magnitude_squared = np.abs(fft_vals) ** 2 / n

        # Scale for single-sided spectrum
        if n % 2 == 0:
            magnitude_squared[1:-1] *= 2  # Don't scale DC and Nyquist
        else:
            magnitude_squared[1:] *= 2  # Don't scale DC

        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        total_energy_freq = np.sum(magnitude_squared)

        # Avoid log(0) issues for later calculations
        magnitude_squared[magnitude_squared <= 0] = 1e-12

        # Energy distribution across logarithmic frequency bands
        num_bands = 8
        if len(freqs) < 2:
            band_edges = np.array([0.0, sample_rate / 2.0])
            num_bands = 1
        else:
            # Create logarithmic band edges (skip DC for log scale)
            min_freq = max(freqs[1], 0.1)  # Avoid DC and very low frequencies
            max_freq = freqs[-1]
            band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), num=num_bands + 1)

        energy_per_band = {}
        band_percentages = []

        for i in range(len(band_edges) - 1):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            energy_band = np.sum(magnitude_squared[mask])
            percentage = (energy_band / total_energy_freq) * 100 if total_energy_freq > 0 else 0

            band_label = f"Band {i + 1} ({band_edges[i]:.1f}-{band_edges[i + 1]:.1f} Hz)"
            energy_per_band[band_label] = percentage
            band_percentages.append(percentage)

        # Find dominant frequency band
        if len(band_percentages) > 0:
            dominant_idx = np.argmax(band_percentages)
            dominant_band = list(energy_per_band.keys())[dominant_idx]
            dominant_value = band_percentages[dominant_idx]
        else:
            dominant_band = "N/A"
            dominant_value = 0.0

        # Temporal energy analysis (energy in time windows)
        window_size = max(1, len(extended_values) // 10)  # 10 windows
        temporal_energy = []

        for i in range(0, len(extended_values), window_size):
            window = extended_values[i:i + window_size]
            window_energy = np.sum(window ** 2)
            temporal_energy.append(window_energy)

        temporal_energy = np.array(temporal_energy)

        # Energy variation metrics
        energy_mean = np.mean(temporal_energy)
        energy_std = np.std(temporal_energy)
        energy_cv = energy_std / energy_mean if energy_mean > 0 else 0  # Coefficient of variation

        results = {
            # Core energy data for plotting
            "Freqs": freqs,
            "Spectrum": magnitude_squared,
            "Band Edges": band_edges,
            "Band Percentages": band_percentages,
            "Temporal Energy": temporal_energy,

            # Energy measures
            "Total Energy (Time Domain)": float(total_energy_time),
            "Total Energy (Frequency Domain)": float(total_energy_freq),
            "Signal Power": float(signal_power),
            "RMS Value": float(rms_value),

            # Frequency distribution
            "Energy Distribution (%)": energy_per_band,
            "Dominant Frequency Band": dominant_band,
            "Dominant Band Energy": f"{dominant_value:.1f}%",

            # Temporal characteristics
            "Temporal Energy Mean": float(energy_mean),
            "Temporal Energy Standard Deviation": float(energy_std),
            "Temporal Energy Coefficient of Variation": float(energy_cv),
            "Energy Uniformity": float(1.0 / (1.0 + energy_cv)),  # Higher = more uniform

            # Analysis parameters
            "Sample Rate (Hz)": float(sample_rate),
            "Signal Length": len(extended_values),
            "Number of Frequency Bands": len(band_edges) - 1,
            "Temporal Window Size": window_size,
            "Number of Temporal Windows": len(temporal_energy)
        }

        Logger.log_message_static(
            f"Calculations-Hilbert: Energy analysis completed. "
            f"Total_energy={total_energy_time:.6e}, RMS={rms_value:.6f}, "
            f"Dominant_band={dominant_band}, Energy_CV={energy_cv:.3f}",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Hilbert: Error in energy analysis: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Calculations-Hilbert: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None