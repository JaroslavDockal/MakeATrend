"""
Cepstral analysis and peak detection.

This module provides cepstral analysis and peak detection capabilities:
- Cepstral analysis for revealing periodic patterns in frequency spectra
- Peak detection with various algorithms and parameters
- Fundamental frequency estimation
- Harmonic analysis through cepstrum
- Echo and reverberation detection

The cepstrum is the "spectrum of the logarithm of the spectrum" and is
particularly useful for detecting periodicities and separating source
and filter characteristics in signals.
"""

import numpy as np
import scipy.signal as sc_signal
from PySide6.QtWidgets import QMessageBox

from .common import safe_prepare_signal, extended_prepare_signal, safe_sample_rate, validate_analysis_inputs
from utils.logger import Logger


def calculate_cepstrum_analysis(time_arr, values, dialog=None, title="Cepstral Analysis"):
    """
    Perform cepstral analysis to reveal periodic patterns in the frequency spectrum.

    The cepstrum is computed as the inverse Fourier transform of the logarithm
    of the magnitude spectrum. It's particularly useful for:
    - Pitch detection in speech and music
    - Echo detection and removal
    - Harmonic analysis
    - Separating source and filter characteristics

    Args:
        time_arr (np.ndarray): Time values for sampling rate calculation.
        values (np.ndarray): Signal values to analyze.
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "Cepstral Analysis".

    Returns:
        dict or None: Dictionary containing cepstral analysis results:
            - Cepstrum: Real cepstrum values
            - Quefrency: Time-like axis for cepstrum (in seconds)
            - Log Power Spectrum: Log magnitude spectrum used for cepstrum
            - Sampling Rate: Sampling frequency
            - Fundamental Frequency (Hz): Detected fundamental frequency
            - Peak Quefrency (s): Quefrency of strongest cepstral peak
            - Peaks: List of significant cepstral peaks
            - Max Cepstrum Value: Maximum value in cepstrum
            - Mean Cepstrum Value: Average cepstrum value
            - Harmonic Structure: Information about detected harmonics

        Returns None if validation fails or user cancels.

    Example:
        >>> t = np.linspace(0, 1, 8000)
        >>> # Harmonic signal with fundamental at 100 Hz
        >>> signal = (np.sin(2*np.pi*100*t) + 0.5*np.sin(2*np.pi*200*t) +
        ...          0.25*np.sin(2*np.pi*300*t))
        >>> result = calculate_cepstrum_analysis(t, signal)
        >>> print(f"Fundamental frequency: {result['Fundamental Frequency (Hz)']:.1f} Hz")
        Fundamental frequency: 100.0 Hz
    """
    Logger.log_message_static(f"Calculations-Cepstrum: Starting cepstral analysis", Logger.DEBUG)

    # Validate inputs
    is_valid, error_msg, sample_rate = validate_analysis_inputs(time_arr, values, min_length=8,
                                                                require_positive_sample_rate=True)
    if not is_valid:
        Logger.log_message_static(f"Calculations-Cepstrum: Input validation failed: {error_msg}", Logger.ERROR)
        if dialog:
            QMessageBox.warning(dialog, title, f"Cepstral Analysis Error:\n{error_msg}")
        return None

    # Prepare signal
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Calculations-Cepstrum: Signal validation failed", Logger.WARNING)
        return None

    # Extended preparation for better cepstral analysis
    extended_values = extended_prepare_signal(processed_values, dialog, title)
    if extended_values is None:
        Logger.log_message_static("Calculations-Cepstrum: Extended signal preparation failed", Logger.WARNING)
        return None

    try:
        n = len(extended_values)

        if n < 8:
            Logger.log_message_static("Calculations-Cepstrum: Signal too short for cepstral analysis", Logger.ERROR)
            if dialog:
                QMessageBox.warning(dialog, title,
                                    "Signal is too short for meaningful cepstral analysis (minimum 8 samples required).")
            return None

        # Apply window to reduce spectral leakage
        windowed_signal = extended_values * np.hanning(n)

        # Compute FFT
        fft_vals = np.fft.fft(windowed_signal)

        # Compute log power spectrum (avoid log(0))
        magnitude = np.abs(fft_vals)
        magnitude[magnitude <= 0] = 1e-12  # Prevent log(0)
        log_power_spectrum = np.log(magnitude ** 2)

        # Compute real cepstrum (inverse FFT of log power spectrum)
        cepstrum = np.fft.ifft(log_power_spectrum).real

        # Create quefrency axis (time-like axis for cepstrum)
        quefrency = np.arange(n) / sample_rate

        # Analyze only positive quefrencies (first half)
        # Skip the first few samples to avoid low-quefrency artifacts
        min_quefrency_samples = max(1, int(0.002 * sample_rate))  # Skip first 2ms
        max_quefrency_samples = n // 2  # Analyze only first half

        if min_quefrency_samples >= max_quefrency_samples:
            min_quefrency_samples = 1
            max_quefrency_samples = min(n // 2, min_quefrency_samples + 10)

        search_cepstrum = cepstrum[min_quefrency_samples:max_quefrency_samples]
        search_quefrency = quefrency[min_quefrency_samples:max_quefrency_samples]

        if len(search_cepstrum) == 0:
            Logger.log_message_static("Calculations-Cepstrum: No valid quefrency range for analysis", Logger.ERROR)
            if dialog:
                QMessageBox.warning(dialog, title, "No valid quefrency range for cepstral analysis.")
            return None

        # Find peaks in cepstrum
        # Use adaptive threshold based on cepstrum statistics
        cepstrum_std = np.std(search_cepstrum)
        cepstrum_mean = np.mean(search_cepstrum)
        peak_threshold = cepstrum_mean + 2 * cepstrum_std

        # Minimum peak height (relative to maximum)
        max_cepstrum_val = np.max(search_cepstrum)
        relative_threshold = max_cepstrum_val * 0.1  # 10% of maximum

        # Use the higher of the two thresholds
        final_threshold = max(peak_threshold, relative_threshold)

        # Find peaks with minimum distance constraint
        min_peak_distance = max(1, int(0.005 * sample_rate))  # Minimum 5ms between peaks

        try:
            peaks, properties = sc_signal.find_peaks(
                search_cepstrum,
                height=final_threshold,
                distance=min_peak_distance,
                prominence=cepstrum_std
            )
        except Exception as peak_error:
            Logger.log_message_static(f"Calculations-Cepstrum: Peak detection failed: {peak_error}", Logger.WARNING)
            peaks = []
            properties = {}

        # Convert peak indices back to original cepstrum indices
        peaks_original = peaks + min_quefrency_samples

        # Extract peak information
        peak_data = []
        fundamental_frequency = 0.0
        peak_quefrency = 0.0

        if len(peaks) > 0:
            # Sort peaks by amplitude (descending)
            peak_amplitudes = search_cepstrum[peaks]
            sorted_indices = np.argsort(peak_amplitudes)[::-1]

            # Process top peaks
            for i, peak_idx in enumerate(sorted_indices[:5]):  # Top 5 peaks
                original_idx = peaks_original[peak_idx]
                q = quefrency[original_idx]
                amplitude = cepstrum[original_idx]
                frequency = 1.0 / q if q > 0 else 0.0

                peak_data.append({
                    'quefrency': q,
                    'amplitude': amplitude,
                    'frequency': frequency,
                    'peak_index': original_idx,
                    'rank': i + 1
                })

            # The strongest peak likely corresponds to the fundamental period
            if peak_data:
                strongest_peak = peak_data[0]
                peak_quefrency = strongest_peak['quefrency']
                fundamental_frequency = strongest_peak['frequency']

        # Calculate cepstrum statistics
        max_cepstrum_value = np.max(search_cepstrum)
        mean_cepstrum_value = np.mean(search_cepstrum)
        cepstrum_energy = np.sum(search_cepstrum ** 2)

        # Harmonic analysis
        harmonic_structure = analyze_harmonic_structure(peak_data, fundamental_frequency)

        # Quality metrics
        peak_to_noise_ratio = max_cepstrum_value / cepstrum_std if cepstrum_std > 0 else 0
        cepstral_clarity = max_cepstrum_value / mean_cepstrum_value if mean_cepstrum_value != 0 else 0

        # Spectral characteristics from log spectrum
        spectrum_stats = analyze_log_spectrum(log_power_spectrum, sample_rate)

        # Build comprehensive results dictionary
        results = {
            # Core cepstral data
            "Cepstrum": cepstrum,
            "Quefrency": quefrency,
            "Log Power Spectrum": log_power_spectrum,
            "Sampling Rate": float(sample_rate),

            # Peak analysis results
            "Fundamental Frequency (Hz)": float(fundamental_frequency),
            "Peak Quefrency (s)": float(peak_quefrency),
            "Peaks": peak_data,
            "Number of Peaks": len(peak_data),

            # Cepstrum statistics
            "Max Cepstrum Value": float(max_cepstrum_value),
            "Mean Cepstrum Value": float(mean_cepstrum_value),
            "Cepstrum Standard Deviation": float(cepstrum_std),
            "Cepstrum Energy": float(cepstrum_energy),

            # Quality metrics
            "Peak-to-Noise Ratio": float(peak_to_noise_ratio),
            "Cepstral Clarity": float(cepstral_clarity),

            # Harmonic analysis
            "Harmonic Structure": harmonic_structure,

            # Spectral characteristics
            "Spectrum Statistics": spectrum_stats,

            # Analysis parameters
            "Signal Length": n,
            "Quefrency Resolution (s)": float(1.0 / sample_rate),
            "Search Range (s)": f"{search_quefrency[0]:.6f} - {search_quefrency[-1]:.6f}",
            "Peak Threshold": float(final_threshold),
            "Window Applied": "Hanning"
        }

        # Add interpretation and recommendations
        interpretation = interpret_cepstral_results(results)
        results["Interpretation"] = interpretation

        Logger.log_message_static(
            f"Calculations-Cepstrum: Analysis completed. "
            f"Fundamental={fundamental_frequency:.2f}Hz, Peaks={len(peak_data)}, "
            f"Max_cepstrum={max_cepstrum_value:.6f}, Clarity={cepstral_clarity:.2f}",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Cepstrum: Error in cepstral analysis: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Calculations-Cepstrum: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None


def calculate_peak_detection(time_arr, values, method="scipy", **kwargs):
    """
    Detect peaks in a signal using various algorithms.

    Provides multiple peak detection algorithms with customizable parameters
    for different types of signals and detection requirements.

    Args:
        time_arr (np.ndarray): Time values corresponding to signal samples.
        values (np.ndarray): Signal values to analyze for peaks.
        method (str, optional): Peak detection method. Options:
            - "scipy": SciPy's find_peaks (default, most versatile)
            - "threshold": Simple threshold-based detection
            - "derivative": Zero-crossing of derivative
            - "adaptive": Adaptive threshold based on local statistics
            - "prominence": Prominence-based detection
        **kwargs: Method-specific parameters:
            For "scipy": height, distance, prominence, width, etc.
            For "threshold": threshold, min_distance
            For "adaptive": window_size, threshold_factor
            For "prominence": min_prominence, min_distance

    Returns:
        dict or None: Dictionary containing peak detection results:
            - Peak Type: Type of peaks detected (positive/negative)
            - Count: Number of peaks found
            - Indices: Array indices of peak locations
            - Times: Time values of peak locations
            - Heights: Amplitude values at peak locations
            - Widths: Peak widths (if calculable)
            - Prominences: Peak prominences (if calculable)
            - Method: Detection method used
            - Parameters: Parameters used for detection

        Returns None if validation fails.

    Example:
        >>> t = np.linspace(0, 2, 1000)
        >>> signal = np.sin(2*np.pi*5*t) + 0.1*np.random.randn(1000)
        >>> result = calculate_peak_detection(t, signal, method="scipy",
        ...                                  height=0.5, distance=50)
        >>> print(f"Found {result['Count']} peaks")
    """
    Logger.log_message_static(f"Calculations-Cepstrum: Starting peak detection using '{method}' method", Logger.DEBUG)

    # Validate inputs
    is_valid, error_msg, sample_rate = validate_analysis_inputs(time_arr, values, min_length=3)
    if not is_valid:
        Logger.log_message_static(f"Calculations-Cepstrum: Input validation failed: {error_msg}", Logger.ERROR)
        return None

    # Prepare signal
    processed_values = safe_prepare_signal(values, None, "Peak Detection")
    if processed_values is None:
        Logger.log_message_static("Calculations-Cepstrum: Signal validation failed", Logger.WARNING)
        return None

    try:
        # Determine if we should look for positive or negative peaks
        signal_mean = np.mean(processed_values)
        abs_min = abs(np.min(processed_values) - signal_mean)
        abs_max = abs(np.max(processed_values) - signal_mean)

        if abs_min > abs_max:
            # More negative deviation - look for negative peaks
            peak_type = "Negative"
            search_signal = -processed_values
            Logger.log_message_static("Calculations-Cepstrum: Detecting negative peaks (inverted signal)", Logger.DEBUG)
        else:
            # More positive deviation - look for positive peaks
            peak_type = "Positive"
            search_signal = processed_values
            Logger.log_message_static("Calculations-Cepstrum: Detecting positive peaks", Logger.DEBUG)

        # Apply peak detection method
        if method == "scipy":
            peaks, properties = detect_peaks_scipy(search_signal, **kwargs)

        elif method == "threshold":
            peaks, properties = detect_peaks_threshold(search_signal, **kwargs)

        elif method == "derivative":
            peaks, properties = detect_peaks_derivative(search_signal, sample_rate, **kwargs)

        elif method == "adaptive":
            peaks, properties = detect_peaks_adaptive(search_signal, **kwargs)

        elif method == "prominence":
            peaks, properties = detect_peaks_prominence(search_signal, **kwargs)

        else:
            Logger.log_message_static(f"Calculations-Cepstrum: Unknown peak detection method: {method}", Logger.ERROR)
            return {"Result": f"Unknown peak detection method: {method}"}

        if len(peaks) == 0:
            Logger.log_message_static("Calculations-Cepstrum: No peaks detected", Logger.INFO)
            return {"Result": "No peaks detected in this signal"}

        # Extract peak information
        peak_times = time_arr[peaks] if len(time_arr) == len(processed_values) else peaks / sample_rate
        peak_heights = processed_values[peaks]

        # If we inverted the signal for negative peak detection, restore original heights
        if peak_type == "Negative":
            peak_heights = -peak_heights

        # Calculate peak widths if possible
        try:
            if method in ["scipy", "prominence"] and len(peaks) > 0:
                widths = sc_signal.peak_widths(search_signal, peaks, rel_height=0.5)[0]
                width_times = widths / sample_rate
            else:
                width_times = estimate_peak_widths(search_signal, peaks, sample_rate)
        except Exception as width_error:
            Logger.log_message_static(f"Calculations-Cepstrum: Peak width calculation failed: {width_error}",
                                      Logger.DEBUG)
            width_times = np.zeros(len(peaks))

        # Calculate additional peak statistics
        if len(peaks) > 0:
            mean_height = np.mean(peak_heights)
            max_height = np.max(peak_heights)
            min_height = np.min(peak_heights)
            height_std = np.std(peak_heights)

            # Peak intervals (time between consecutive peaks)
            if len(peak_times) > 1:
                intervals = np.diff(peak_times)
                mean_interval = np.mean(intervals)
                interval_std = np.std(intervals)
                estimated_frequency = 1.0 / mean_interval if mean_interval > 0 else 0.0
            else:
                intervals = np.array([])
                mean_interval = 0.0
                interval_std = 0.0
                estimated_frequency = 0.0
        else:
            mean_height = max_height = min_height = height_std = 0.0
            intervals = np.array([])
            mean_interval = interval_std = estimated_frequency = 0.0

        # Build results dictionary
        results = {
            # Peak identification
            "Peak Type": peak_type,
            "Count": len(peaks),
            "Indices": peaks,
            "Times": peak_times,
            "Heights": peak_heights,
            "Widths": width_times,

            # Statistical measures
            "Mean Height": float(mean_height),
            "Max Height": float(max_height),
            "Min Height": float(min_height),
            "Height Standard Deviation": float(height_std),
            "Mean Width": float(np.mean(width_times)) if len(width_times) > 0 else 0.0,

            # Temporal characteristics
            "Intervals": intervals,
            "Mean Interval (s)": float(mean_interval),
            "Interval Standard Deviation (s)": float(interval_std),
            "Estimated Frequency (Hz)": float(estimated_frequency),

            # Method information
            "Method": method,
            "Parameters": kwargs,
            "Sample Rate (Hz)": float(sample_rate),
            "Signal Length": len(processed_values)
        }

        # Add method-specific results
        if 'prominences' in properties:
            results["Prominences"] = properties['prominences']
            results["Mean Prominence"] = float(np.mean(properties['prominences']))

        if 'peak_widths' in properties:
            results["Peak Widths (samples)"] = properties['peak_widths']

        Logger.log_message_static(
            f"Calculations-Cepstrum: Peak detection completed. "
            f"Method={method}, Count={len(peaks)}, Type={peak_type}, "
            f"Mean_height={mean_height:.4f}, Estimated_freq={estimated_frequency:.2f}Hz",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Cepstrum: Error in peak detection: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Calculations-Cepstrum: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None


# Helper functions for peak detection methods

def detect_peaks_scipy(signal, **kwargs):
    """Peak detection using SciPy's find_peaks."""
    try:
        peaks, properties = sc_signal.find_peaks(signal, **kwargs)
        return peaks, properties
    except Exception as e:
        Logger.log_message_static(f"Calculations-Cepstrum: SciPy peak detection failed: {e}", Logger.WARNING)
        return np.array([]), {}


def detect_peaks_threshold(signal, threshold=None, min_distance=1, **kwargs):
    """Simple threshold-based peak detection."""
    if threshold is None:
        threshold = np.mean(signal) + np.std(signal)

    # Find points above threshold
    above_threshold = signal > threshold

    # Find rising edges (start of peaks)
    rising_edges = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1

    # Apply minimum distance constraint
    if len(rising_edges) > 1 and min_distance > 1:
        filtered_peaks = [rising_edges[0]]
        for peak in rising_edges[1:]:
            if peak - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak)
        peaks = np.array(filtered_peaks)
    else:
        peaks = rising_edges

    return peaks, {'threshold_used': threshold}


def detect_peaks_derivative(signal, sample_rate, smoothing=True, **kwargs):
    """Peak detection using zero-crossings of the derivative."""
    # Calculate derivative
    if smoothing:
        # Apply simple smoothing before differentiation
        kernel = np.ones(3) / 3
        smoothed = np.convolve(signal, kernel, mode='same')
        derivative = np.gradient(smoothed)
    else:
        derivative = np.gradient(signal)

    # Find zero crossings (positive to negative for peaks)
    zero_crossings = []
    for i in range(1, len(derivative)):
        if derivative[i - 1] > 0 and derivative[i] <= 0:
            zero_crossings.append(i)

    peaks = np.array(zero_crossings)
    return peaks, {'smoothing_applied': smoothing}


def detect_peaks_adaptive(signal, window_size=50, threshold_factor=2.0, **kwargs):
    """Adaptive threshold peak detection using local statistics."""
    peaks = []
    half_window = window_size // 2

    for i in range(half_window, len(signal) - half_window):
        # Local window
        window = signal[i - half_window:i + half_window + 1]

        # Local statistics
        local_mean = np.mean(window)
        local_std = np.std(window)

        # Adaptive threshold
        threshold = local_mean + threshold_factor * local_std

        # Check if current point is a peak
        if (signal[i] > threshold and
                signal[i] > signal[i - 1] and
                signal[i] > signal[i + 1]):
            peaks.append(i)

    return np.array(peaks), {'window_size': window_size, 'threshold_factor': threshold_factor}


def detect_peaks_prominence(signal, min_prominence=None, min_distance=1, **kwargs):
    """Prominence-based peak detection."""
    if min_prominence is None:
        min_prominence = np.std(signal) * 0.5

    try:
        peaks, properties = sc_signal.find_peaks(
            signal,
            prominence=min_prominence,
            distance=min_distance,
            **kwargs
        )
        return peaks, properties
    except Exception as e:
        Logger.log_message_static(f"Calculations-Cepstrum: Prominence peak detection failed: {e}", Logger.WARNING)
        return np.array([]), {}


def estimate_peak_widths(signal, peaks, sample_rate):
    """Estimate peak widths using half-maximum method."""
    widths = []

    for peak_idx in peaks:
        try:
            peak_value = signal[peak_idx]
            half_max = peak_value / 2

            # Search left
            left_idx = peak_idx
            while left_idx > 0 and signal[left_idx] > half_max:
                left_idx -= 1

            # Search right
            right_idx = peak_idx
            while right_idx < len(signal) - 1 and signal[right_idx] > half_max:
                right_idx += 1

            # Calculate width in time units
            width_samples = right_idx - left_idx
            width_time = width_samples / sample_rate
            widths.append(width_time)

        except Exception:
            widths.append(0.0)

    return np.array(widths)


def analyze_harmonic_structure(peak_data, fundamental_freq):
    """Analyze harmonic structure from cepstral peaks."""
    if not peak_data or fundamental_freq <= 0:
        return {
            "Has Harmonic Structure": False,
            "Harmonic Peaks": [],
            "Harmonic Ratios": []
        }

    harmonic_peaks = []
    harmonic_ratios = []

    # Check if peaks correspond to harmonics of the fundamental
    for peak in peak_data:
        freq = peak['frequency']
        if freq > 0:
            ratio = freq / fundamental_freq

            # Check if ratio is close to an integer (within 5%)
            nearest_int = round(ratio)
            if abs(ratio - nearest_int) / nearest_int < 0.05 and nearest_int > 0:
                harmonic_peaks.append({
                    'harmonic_number': nearest_int,
                    'frequency': freq,
                    'amplitude': peak['amplitude'],
                    'quefrency': peak['quefrency']
                })
                harmonic_ratios.append(ratio)

    return {
        "Has Harmonic Structure": len(harmonic_peaks) > 1,
        "Harmonic Peaks": harmonic_peaks,
        "Harmonic Ratios": harmonic_ratios,
        "Number of Harmonics": len(harmonic_peaks)
    }


def analyze_log_spectrum(log_spectrum, sample_rate):
    """Analyze characteristics of the log power spectrum."""
    try:
        # Calculate spectral statistics
        spectrum_mean = np.mean(log_spectrum)
        spectrum_std = np.std(log_spectrum)
        spectrum_max = np.max(log_spectrum)
        spectrum_min = np.min(log_spectrum)

        # Spectral tilt (slope of spectrum)
        freqs = np.linspace(0, sample_rate / 2, len(log_spectrum) // 2)
        spectrum_half = log_spectrum[:len(log_spectrum) // 2]

        if len(freqs) > 1:
            # Linear fit to estimate spectral tilt
            coeffs = np.polyfit(freqs, spectrum_half, 1)
            spectral_tilt = coeffs[0]  # Slope
        else:
            spectral_tilt = 0.0

        return {
            "Mean Log Power": float(spectrum_mean),
            "Log Power Standard Deviation": float(spectrum_std),
            "Max Log Power": float(spectrum_max),
            "Min Log Power": float(spectrum_min),
            "Dynamic Range (dB)": float(spectrum_max - spectrum_min),
            "Spectral Tilt (dB/Hz)": float(spectral_tilt)
        }

    except Exception as e:
        Logger.log_message_static(f"Calculations-Cepstrum: Log spectrum analysis failed: {e}", Logger.DEBUG)
        return {"Error": str(e)}


def interpret_cepstral_results(results):
    """Provide interpretation of cepstral analysis results."""
    interpretation = []

    # Check fundamental frequency
    fundamental = results.get("Fundamental Frequency (Hz)", 0)
    if fundamental > 0:
        if fundamental < 80:
            interpretation.append(f"Low fundamental frequency ({fundamental:.1f} Hz) - possibly bass/low-pitched sound")
        elif fundamental > 1000:
            interpretation.append(f"High fundamental frequency ({fundamental:.1f} Hz) - possibly high-pitched sound")
        else:
            interpretation.append(f"Moderate fundamental frequency ({fundamental:.1f} Hz) - typical speech/music range")
    else:
        interpretation.append("No clear fundamental frequency detected - signal may be aperiodic or noisy")

    # Check harmonic structure
    harmonic_info = results.get("Harmonic Structure", {})
    if harmonic_info.get("Has Harmonic Structure", False):
        num_harmonics = harmonic_info.get("Number of Harmonics", 0)
        interpretation.append(f"Clear harmonic structure detected with {num_harmonics} harmonics")
    else:
        interpretation.append("No clear harmonic structure - signal may be noise-like or inharmonic")

    # Check cepstral clarity
    clarity = results.get("Cepstral Clarity", 0)
    if clarity > 5:
        interpretation.append("High cepstral clarity - strong periodic structure")
    elif clarity > 2:
        interpretation.append("Moderate cepstral clarity - some periodic structure")
    else:
        interpretation.append("Low cepstral clarity - weak or no periodic structure")

    # Check number of peaks
    num_peaks = results.get("Number of Peaks", 0)
    if num_peaks == 0:
        interpretation.append("No significant cepstral peaks found")
    elif num_peaks == 1:
        interpretation.append("Single dominant cepstral peak - simple periodic structure")
    else:
        interpretation.append(f"Multiple cepstral peaks ({num_peaks}) - complex periodic structure")

    return interpretation