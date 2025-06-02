"""
Frequency domain signal analysis: FFT, PSD, and spectral characteristics.

This module provides frequency-domain analysis functions:
- Fast Fourier Transform (FFT) analysis
- Power Spectral Density (PSD) using Welch's method
- Spectral peak detection and characterization
- Bandwidth calculations and spectral statistics

These analyses reveal the frequency content and spectral characteristics of signals.
"""

import numpy as np
import scipy.signal as sc_signal
from PySide6.QtWidgets import QMessageBox

from .common import safe_prepare_signal, safe_sample_rate, validate_analysis_inputs, calculate_bandwidth
from utils.logger import Logger


def calculate_fft_analysis(time_arr, values, dialog=None, title="FFT Analysis"):
    """
    Compute FFT magnitude spectrum and key frequency-domain statistics.

    Performs Fast Fourier Transform analysis to decompose the signal into its
    frequency components. Provides both the full spectrum and key statistical measures.

    Args:
        time_arr (np.ndarray): Time values for sampling rate calculation.
        values (np.ndarray): Signal amplitude values to analyze.
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "FFT Analysis".

    Returns:
        dict or None: Dictionary containing FFT results:
            - Time Array: Original time values
            - Processed Signal: Validated signal values
            - Frequency Axis (Hz): Frequency bins
            - Magnitude Spectrum: Single-sided magnitude spectrum
            - Phase Spectrum: Phase information
            - Peak Frequency (Hz): Frequency of maximum magnitude
            - Max Magnitude: Maximum magnitude value
            - Total Energy (Freq Domain): Total spectral energy
            - Spectral Centroid (Hz): "Center of mass" of spectrum
            - Spectral Bandwidth (Hz): Weighted frequency spread
            - Spectral Rolloff (Hz): Frequency below which X% of energy lies

        Returns None if validation fails or user cancels.

    Example:
        >>> t = np.linspace(0, 1, 1000)
        >>> signal = np.sin(2 * np.pi * 50 * t)  # 50 Hz sine wave
        >>> result = calculate_fft_analysis(t, signal)
        >>> print(f"Peak frequency: {result['Peak Frequency (Hz)']:.1f} Hz")
        Peak frequency: 50.0 Hz
    """
    Logger.log_message_static(f"Calculations-Frequency: Starting FFT analysis", Logger.DEBUG)

    # Validate inputs
    is_valid, error_msg, sample_rate = validate_analysis_inputs(time_arr, values, min_length=4,
                                                                require_positive_sample_rate=True)
    if not is_valid:
        Logger.log_message_static(f"Calculations-Frequency: Input validation failed: {error_msg}", Logger.ERROR)
        if dialog:
            QMessageBox.warning(dialog, title, f"FFT Analysis Error:\n{error_msg}")
        return None

    # Prepare signal
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Calculations-Frequency: Signal validation failed", Logger.WARNING)
        return None

    try:
        n = len(processed_values)

        # Apply window to reduce spectral leakage
        window = np.hanning(n)
        windowed_signal = processed_values * window

        # Compute FFT (real input, so use rfft for efficiency)
        fft_vals = np.fft.rfft(windowed_signal)
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

        # Calculate magnitude spectrum (single-sided)
        magnitude = np.abs(fft_vals) / n

        # Scale for single-sided spectrum (except DC and Nyquist)
        if n % 2 == 0:  # Even length
            magnitude[1:-1] *= 2  # Don't scale DC (index 0) and Nyquist (index -1)
        else:  # Odd length
            magnitude[1:] *= 2  # Don't scale DC (index 0), no Nyquist for odd length

        # Avoid log(0) issues
        magnitude[magnitude <= 0] = 1e-12

        # Calculate phase spectrum
        phase = np.angle(fft_vals)

        # Find peak frequency
        peak_index = np.argmax(magnitude)
        peak_freq = fft_freqs[peak_index]
        peak_mag = magnitude[peak_index]

        # Calculate spectral energy
        spectral_energy = np.sum(magnitude ** 2)

        # Calculate spectral centroid (center of mass of spectrum)
        spectral_centroid = np.sum(fft_freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0.0

        # Calculate spectral bandwidth (weighted standard deviation around centroid)
        spectral_variance = np.sum(((fft_freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude) if np.sum(
            magnitude) > 0 else 0.0
        spectral_bandwidth = np.sqrt(spectral_variance)

        # Calculate spectral rolloff (frequency below which 85% of energy lies)
        cumulative_energy = np.cumsum(magnitude ** 2)
        total_energy = cumulative_energy[-1]
        rolloff_threshold = 0.85 * total_energy
        rolloff_index = np.where(cumulative_energy >= rolloff_threshold)[0]
        spectral_rolloff = fft_freqs[rolloff_index[0]] if len(rolloff_index) > 0 else fft_freqs[-1]

        # Find spectral peaks for detailed analysis
        # Use prominence-based peak detection
        min_prominence = np.max(magnitude) * 0.05  # 5% of max magnitude
        peaks, properties = sc_signal.find_peaks(magnitude, prominence=min_prominence,
                                                 distance=max(1, len(magnitude) // 100))

        # Get peak information
        peak_frequencies = fft_freqs[peaks] if len(peaks) > 0 else np.array([])
        peak_magnitudes = magnitude[peaks] if len(peaks) > 0 else np.array([])
        peak_prominences = properties['prominences'] if len(peaks) > 0 else np.array([])

        # Sort peaks by magnitude (descending)
        if len(peaks) > 0:
            sort_indices = np.argsort(peak_magnitudes)[::-1]
            peak_frequencies = peak_frequencies[sort_indices]
            peak_magnitudes = peak_magnitudes[sort_indices]
            peak_prominences = peak_prominences[sort_indices]

        # Build results dictionary
        results = {
            # Input data
            "Time Array": time_arr,
            "Processed Signal": processed_values,

            # Frequency domain data
            "Frequency Axis (Hz)": fft_freqs,
            "Magnitude Spectrum": magnitude,
            "Phase Spectrum": phase,
            "Magnitude Spectrum (dB)": 20 * np.log10(magnitude),

            # Peak information
            "Peak Frequency (Hz)": float(peak_freq),
            "Max Magnitude": float(peak_mag),
            "Peak Phase (rad)": float(phase[peak_index]),

            # Spectral characteristics
            "Total Energy (Freq Domain)": float(spectral_energy),
            "Spectral Centroid (Hz)": float(spectral_centroid),
            "Spectral Bandwidth (Hz)": float(spectral_bandwidth),
            "Spectral Rolloff (Hz)": float(spectral_rolloff),

            # Analysis parameters
            "Sample Rate (Hz)": float(sample_rate),
            "Frequency Resolution (Hz)": float(sample_rate / n),
            "Nyquist Frequency (Hz)": float(sample_rate / 2),
            "Window Type": "Hanning",

            # Peak detection results
            "Number of Peaks": len(peaks),
            "Peak Frequencies (Hz)": peak_frequencies.tolist() if len(peak_frequencies) > 0 else [],
            "Peak Magnitudes": peak_magnitudes.tolist() if len(peak_magnitudes) > 0 else [],
            "Peak Prominences": peak_prominences.tolist() if len(peak_prominences) > 0 else []
        }

        # Add harmonic analysis if we have a clear fundamental
        if peak_mag > np.mean(magnitude) * 10:  # Strong fundamental peak
            harmonics = []
            harmonic_magnitudes = []
            for h in range(2, 11):  # Check up to 10th harmonic
                harmonic_freq = peak_freq * h
                if harmonic_freq < sample_rate / 2:  # Within Nyquist limit
                    # Find closest frequency bin
                    harmonic_index = np.argmin(np.abs(fft_freqs - harmonic_freq))
                    if np.abs(fft_freqs[harmonic_index] - harmonic_freq) < sample_rate / n * 2:  # Within 2 bins
                        harmonics.append(float(fft_freqs[harmonic_index]))
                        harmonic_magnitudes.append(float(magnitude[harmonic_index]))

            results["Harmonic Frequencies (Hz)"] = harmonics
            results["Harmonic Magnitudes"] = harmonic_magnitudes
            results["Total Harmonic Distortion (THD)"] = float(
                np.sqrt(np.sum(np.array(harmonic_magnitudes) ** 2)) / peak_mag if len(harmonic_magnitudes) > 0 else 0.0
            )

        Logger.log_message_static(
            f"Calculations-Frequency: FFT analysis completed. "
            f"Sample_rate={sample_rate:.1f}Hz, Peak_freq={peak_freq:.2f}Hz, "
            f"Peak_mag={peak_mag:.6e}, Spectral_centroid={spectral_centroid:.2f}Hz",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Frequency: Error in FFT analysis: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Calculations-Frequency: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None


def calculate_psd_analysis(time_arr, values, dialog=None, title="PSD Analysis"):
    """
    Perform Welch Power Spectral Density analysis.

    Uses Welch's method to estimate the power spectral density, which provides
    a smoothed estimate of how signal power is distributed across frequencies.
    Better for noisy signals compared to raw FFT.

    Args:
        time_arr (np.ndarray): Time values for sampling rate calculation.
        values (np.ndarray): Signal amplitude values to analyze.
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "PSD Analysis".

    Returns:
        dict or None: Dictionary containing PSD results:
            - Frequency Axis (Hz): Frequency bins for PSD
            - PSD (Power/Hz): Power spectral density values
            - PSD (dB/Hz): PSD in decibel scale
            - Peak Frequency (Hz): Frequency of maximum power
            - Peak Power (dB): Maximum power in dB
            - Total Power: Integrated power across all frequencies
            - RMS Amplitude: RMS value from time domain
            - Bandwidth: Spectral bandwidth calculations
            - Noise Floor: Estimated noise floor level

        Returns None if validation fails or user cancels.

    Example:
        >>> t = np.linspace(0, 2, 2000)
        >>> signal = np.sin(2*np.pi*10*t) + 0.1*np.random.randn(2000)  # 10Hz + noise
        >>> result = calculate_psd_analysis(t, signal)
        >>> print(f"Peak at {result['Peak Frequency (Hz)']:.1f} Hz")
        Peak at 10.0 Hz
    """
    Logger.log_message_static(f"Calculations-Frequency: Starting PSD analysis", Logger.DEBUG)

    # Validate inputs
    is_valid, error_msg, sample_rate = validate_analysis_inputs(time_arr, values, min_length=8,
                                                                require_positive_sample_rate=True)
    if not is_valid:
        Logger.log_message_static(f"Calculations-Frequency: Input validation failed: {error_msg}", Logger.ERROR)
        if dialog:
            QMessageBox.warning(dialog, title, f"PSD Analysis Error:\n{error_msg}")
        return None

    # Prepare signal
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Calculations-Frequency: Signal validation failed", Logger.WARNING)
        return None

    if len(processed_values) < 8:
        Logger.log_message_static("Calculations-Frequency: Signal too short for PSD analysis", Logger.ERROR)
        if dialog:
            QMessageBox.warning(dialog, title, "Signal too short for PSD analysis (minimum 8 samples required).")
        return None

    try:
        # Remove DC component and apply window
        detrended = processed_values - np.mean(processed_values)

        # Determine optimal segment length for Welch's method
        # Use segments that are long enough for good frequency resolution
        # but short enough to provide good averaging
        n_samples = len(detrended)
        min_segment_length = 32
        max_segment_length = min(1024, n_samples // 4)

        # Choose segment length as power of 2 for FFT efficiency
        segment_length = 2 ** int(np.log2(max(min_segment_length, min(max_segment_length, n_samples // 8))))
        segment_length = min(segment_length, n_samples)

        # Overlap for better statistics (50% is common)
        overlap = segment_length // 2

        Logger.log_message_static(
            f"Calculations-Frequency: PSD parameters - "
            f"N_samples={n_samples}, Segment_length={segment_length}, Overlap={overlap}",
            Logger.DEBUG
        )

        # Compute PSD using Welch's method
        freqs, psd = sc_signal.welch(
            detrended,
            fs=sample_rate,
            window='hann',
            nperseg=segment_length,
            noverlap=overlap,
            scaling='density',
            detrend='linear'
        )

        # Avoid log(0) issues
        psd[psd <= 0] = 1e-12

        # Find peak in PSD
        peak_idx = np.argmax(psd)
        peak_frequency = freqs[peak_idx]
        peak_power_linear = psd[peak_idx]
        peak_power_db = 10 * np.log10(peak_power_linear)

        # Calculate total power by integrating PSD
        total_power = np.trapz(psd, freqs)

        # RMS from time domain (for comparison)
        rms_time_domain = np.sqrt(np.mean(processed_values ** 2))

        # RMS from frequency domain (should match time domain)
        rms_freq_domain = np.sqrt(total_power)

        # Estimate noise floor (median of PSD in dB)
        psd_db = 10 * np.log10(psd)
        noise_floor_db = np.median(psd_db)

        # Calculate bandwidth using different methods
        bandwidth_info = calculate_bandwidth(freqs, psd, db_level=-3)

        # Spectral characteristics
        # Spectral centroid
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0.0

        # Spectral spread (bandwidth around centroid)
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd)) if np.sum(
            psd) > 0 else 0.0

        # Peak-to-noise ratio
        noise_power = np.median(psd)  # Estimate noise as median power
        peak_to_noise_ratio = 10 * np.log10(peak_power_linear / noise_power) if noise_power > 0 else np.inf

        # Frequency domain statistics
        mean_freq = np.average(freqs, weights=psd)

        # Find significant peaks in PSD
        # Use prominence-based detection with adaptive threshold
        min_prominence_db = 3  # 3 dB above local background
        psd_normalized = psd / np.max(psd)
        prominence_threshold = 10 ** (min_prominence_db / 10) * np.median(psd_normalized)

        peaks, properties = sc_signal.find_peaks(psd, prominence=prominence_threshold, distance=max(1, len(psd) // 50))

        peak_frequencies = freqs[peaks] if len(peaks) > 0 else np.array([])
        peak_powers = psd[peaks] if len(peaks) > 0 else np.array([])
        peak_powers_db = 10 * np.log10(peak_powers) if len(peak_powers) > 0 else np.array([])

        # Sort peaks by power (descending)
        if len(peaks) > 0:
            sort_indices = np.argsort(peak_powers)[::-1]
            peak_frequencies = peak_frequencies[sort_indices]
            peak_powers = peak_powers[sort_indices]
            peak_powers_db = peak_powers_db[sort_indices]

        # Build results dictionary
        results = {
            # Frequency domain data
            "Frequency Axis (Hz)": freqs,
            "PSD (Power/Hz)": psd,
            "PSD (dB/Hz)": psd_db,

            # Peak information
            "Peak Frequency (Hz)": float(peak_frequency),
            "Peak Power (Linear)": float(peak_power_linear),
            "Peak Power (dB)": float(peak_power_db),

            # Power measurements
            "Total Power": float(total_power),
            "RMS Amplitude (Time Domain)": float(rms_time_domain),
            "RMS Amplitude (Freq Domain)": float(rms_freq_domain),

            # Spectral characteristics
            "Spectral Centroid (Hz)": float(spectral_centroid),
            "Spectral Spread (Hz)": float(spectral_spread),
            "Mean Frequency (Hz)": float(mean_freq),

            # Noise characteristics
            "Noise Floor (dB)": float(noise_floor_db),
            "Peak-to-Noise Ratio (dB)": float(peak_to_noise_ratio),

            # Bandwidth information
            "3dB Bandwidth (Hz)": bandwidth_info.get("bandwidth_hz", 0.0),
            "Lower 3dB Frequency (Hz)": bandwidth_info.get("lower_freq_hz", 0.0),
            "Upper 3dB Frequency (Hz)": bandwidth_info.get("upper_freq_hz", 0.0),

            # Analysis parameters
            "Sample Rate (Hz)": float(sample_rate),
            "Segment Length": segment_length,
            "Overlap": overlap,
            "Frequency Resolution (Hz)": float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0,
            "Number of Averages": int(np.ceil((n_samples - overlap) / (segment_length - overlap))),

            # Peak detection results
            "Number of Significant Peaks": len(peaks),
            "Significant Peak Frequencies (Hz)": peak_frequencies.tolist() if len(peak_frequencies) > 0 else [],
            "Significant Peak Powers (dB)": peak_powers_db.tolist() if len(peak_powers_db) > 0 else [],

            # Quality indicators
            "RMS Consistency": float(
                abs(rms_time_domain - rms_freq_domain) / rms_time_domain) if rms_time_domain > 0 else 0.0
        }

        Logger.log_message_static(
            f"Calculations-Frequency: PSD analysis completed. "
            f"Peak_freq={peak_frequency:.2f}Hz, Peak_power={peak_power_db:.1f}dB, "
            f"Total_power={total_power:.6e}, Bandwidth={bandwidth_info.get('bandwidth_hz', 0):.2f}Hz",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Frequency: Error in PSD analysis: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Calculations-Frequency: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None


def calculate_spectral_features(freqs, magnitude, sample_rate):
    """
    Calculate advanced spectral features from magnitude spectrum.

    Helper function for extracting detailed spectral characteristics.

    Args:
        freqs (np.ndarray): Frequency bins
        magnitude (np.ndarray): Magnitude spectrum
        sample_rate (float): Sampling rate

    Returns:
        dict: Dictionary of spectral features
    """
    try:
        # Normalize spectrum for feature calculation
        magnitude_norm = magnitude / np.sum(magnitude) if np.sum(magnitude) > 0 else magnitude

        # Spectral centroid (center of mass)
        centroid = np.sum(freqs * magnitude_norm)

        # Spectral rolloff (frequency below which X% of energy lies)
        cumulative_energy = np.cumsum(magnitude_norm)
        rolloff_85 = freqs[np.where(cumulative_energy >= 0.85)[0][0]] if np.any(cumulative_energy >= 0.85) else freqs[
            -1]
        rolloff_95 = freqs[np.where(cumulative_energy >= 0.95)[0][0]] if np.any(cumulative_energy >= 0.95) else freqs[
            -1]

        # Spectral flux (measure of how quickly the spectrum changes)
        # This would need previous frame for real implementation, here we use gradient
        spectral_flux = np.mean(np.abs(np.gradient(magnitude)))

        # Spectral flatness (measure of how noise-like vs tonal the spectrum is)
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-12)))
        arithmetic_mean = np.mean(magnitude)
        spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0

        # Spectral crest factor (peak-to-average ratio in frequency domain)
        spectral_crest = np.max(magnitude) / np.mean(magnitude) if np.mean(magnitude) > 0 else 0.0

        return {
            "Spectral Centroid (Hz)": float(centroid),
            "Spectral Rolloff 85% (Hz)": float(rolloff_85),
            "Spectral Rolloff 95% (Hz)": float(rolloff_95),
            "Spectral Flux": float(spectral_flux),
            "Spectral Flatness": float(spectral_flatness),
            "Spectral Crest Factor": float(spectral_crest)
        }

    except Exception as e:
        Logger.log_message_static(f"Calculations-Frequency: Error calculating spectral features: {str(e)}",
                                  Logger.ERROR)
        return {}