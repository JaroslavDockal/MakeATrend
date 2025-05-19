import numpy as np
import pywt
from scipy import stats
from scipy import signal as sc_signal
from scipy.signal import correlate

from PySide6.QtWidgets import QMessageBox

from utils.logger import Logger
from .helpers import safe_sample_rate, safe_prepare_signal, extended_prepare_signal


def calculate_basic_statistics(dialog, values):
    """
    Compute fundamental statistical descriptors for a signal.

    Args:
        dialog (QWidget): Parent dialog for error messages or user interaction.
        values (np.ndarray): Signal values.

    Returns:
        dict or None: Dictionary of statistical metrics, or None if user cancels or data is invalid.
    """
    processed_values = safe_prepare_signal(values, dialog, "Basic Statistics")
    if processed_values is None:
        return None

    results = {}
    try:
        std = np.std(processed_values)

        results["Mean"] = float(np.mean(processed_values))
        results["Median"] = float(np.median(processed_values))
        results["Standard Deviation"] = float(std)
        results["Variance"] = float(np.var(processed_values))
        results["Min"] = float(np.min(processed_values))
        results["Max"] = float(np.max(processed_values))
        results["Range"] = float(np.max(processed_values) - np.min(processed_values))
        results["RMS"] = float(np.sqrt(np.mean(processed_values ** 2)))
        results["Skewness"] = float(stats.skew(processed_values)) if std > 0 else 0.0
        results["Kurtosis"] = float(stats.kurtosis(processed_values)) if std > 0 else 0.0

        Logger.log_message_static("Basic statistics computed successfully", Logger.DEBUG)
        return results

    except Exception as e:
        Logger.log_message_static(f"Error computing statistics: {str(e)}", Logger.ERROR)
        return None

def calculate_fft_analysis(dialog, time_arr, values):
    """
    Compute FFT magnitude spectrum and key statistics from a given signal.

    Args:
        dialog (QWidget): Parent widget for optional user messages.
        time_arr (np.ndarray): Time axis of the signal.
        values (np.ndarray): Signal values.

    Returns:
        dict or None: Dictionary containing FFT data and statistics, or None on error/cancel.
    """
    processed_values = safe_prepare_signal(values, dialog, "FFT Spectrum")
    if processed_values is None:
        return None

    sample_rate = safe_sample_rate(time_arr)
    if sample_rate == 0.0:
        QMessageBox.information(dialog, "FFT Spectrum", "Sampling rate could not be determined.")
        return None

    n = len(processed_values)
    fft_vals = np.fft.rfft(processed_values)
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    magnitude = np.abs(fft_vals) / n
    if n % 2 == 0:
        magnitude[1:-1] *= 2
    else:
        magnitude[1:] *= 2

    magnitude[magnitude <= 0] = 1e-12  # Prevent log-scale issues

    peak_index = np.argmax(magnitude)
    peak_freq = fft_freqs[peak_index]
    peak_mag = magnitude[peak_index]

    results = {
        "Time Array": time_arr,
        "Processed Signal": processed_values,
        "Frequency Axis (Hz)": fft_freqs,
        "Magnitude Spectrum": magnitude,
        "Peak Frequency (Hz)": float(peak_freq),
        "Max Magnitude": float(peak_mag),
        "Total Energy (Freq Domain)": float(np.sum(magnitude**2))
    }

    Logger.log_message_static("FFT analysis completed", Logger.DEBUG)
    return results

def calculate_time_domain_analysis(dialog, time_arr, values):
    """
    Compute time-domain statistics for a signal.

    Args:
        dialog (QWidget): Parent dialog for user interaction/logging.
        time_arr (np.ndarray): Time array of the signal.
        values (np.ndarray): Signal values.

    Returns:
        dict or None: Dictionary of statistical metrics, or None if cancelled.
    """
    processed_values = safe_prepare_signal(values, dialog, "Time Domain Analysis")
    if processed_values is None:
        return None

    results = {}
    duration = time_arr[-1] - time_arr[0] if len(time_arr) > 1 else 0.0
    sample_rate = safe_sample_rate(time_arr) if duration > 0 else 0.0

    results["Duration (s)"] = float(duration)
    results["Sample Count"] = len(processed_values)
    results["Sample Rate (Hz)"] = float(sample_rate)

    zero_crossings = ((processed_values[:-1] * processed_values[1:]) < 0).sum()
    results["Zero Crossings"] = int(zero_crossings)

    mean = np.mean(processed_values)
    results["Mean"] = float(mean)
    results["Median"] = float(np.median(processed_values))
    std = np.std(processed_values)
    results["Standard Deviation"] = float(std)
    results["Variance"] = float(np.var(processed_values))
    results["Skewness"] = float(stats.skew(processed_values)) if std > 0 else 0.0
    results["Kurtosis"] = float(stats.kurtosis(processed_values)) if std > 0 else 0.0

    peak_amplitude = np.max(np.abs(processed_values))
    results["Peak Amplitude"] = float(peak_amplitude)
    results["Min"] = float(np.min(processed_values))
    results["Max"] = float(np.max(processed_values))
    energy = np.sum(processed_values ** 2)
    results["Energy"] = float(energy)
    power = energy / len(processed_values) if len(processed_values) > 0 else 0.0
    results["Power"] = float(power)
    rms = np.sqrt(power)
    results["RMS"] = float(rms)
    results["Crest Factor"] = float(peak_amplitude / rms) if rms > 0 else np.inf

    Logger.log_message_static("Time-domain analysis completed", Logger.DEBUG)
    return results

def calculate_psd_analysis(dialog, time_arr, values):
    """
    Perform Welch Power Spectral Density analysis.

    Args:
        dialog (QWidget): Parent widget for potential UI interaction or error message.
        time_arr (np.ndarray): Time values (must be evenly sampled).
        values (np.ndarray): Signal values.

    Returns:
        dict or None: Dictionary containing frequency axis, PSD, and power metrics.
    """
    Logger.log_message_static("Starting PSD calculation (modular)", Logger.INFO)
    processed_values = safe_prepare_signal(values, dialog, "PSD Analysis")
    if processed_values is None:
        return None

    sample_rate = safe_sample_rate(time_arr)
    if sample_rate == 0.0:
        QMessageBox.information(dialog, "PSD Analysis", "Sampling rate could not be determined.")
        return None

    if len(processed_values) < 8:
        QMessageBox.information(dialog, "PSD Analysis", "Signal too short for PSD analysis.")
        return None

    detrended = processed_values - np.mean(processed_values)
    window = sc_signal.windows.hann(len(detrended))

    segment_length = max(8, min(256, len(detrended) // 4))
    segment_length = 2 ** int(np.log2(segment_length))

    freqs, psd = sc_signal.welch(
        detrended * window,
        fs=sample_rate,
        window='hann',
        nperseg=segment_length,
        scaling='density'
    )

    psd[psd <= 0] = 1e-12  # prevent log(0)

    peak_idx = np.argmax(psd)
    total_power = np.trapz(psd, freqs)
    peak_power_db = 10 * np.log10(psd[peak_idx])
    rms_power = np.sqrt(np.mean(processed_values**2))

    results = {
        "Frequency Axis (Hz)": freqs,
        "PSD (Power/Hz)": psd,
        "Peak Frequency (Hz)": freqs[peak_idx],
        "Peak Power (dB)": peak_power_db,
        "Total Power": total_power,
        "RMS Amplitude": rms_power
    }

    Logger.log_message_static("PSD analysis completed", Logger.DEBUG)
    return results

def calculate_peak_detection(dialog, time_arr, values):
    """
    Detect peaks in the input signal and extract basic statistics.

    Args:
        dialog (QWidget): Parent widget for message boxes (e.g., QMessageBox).
        time_arr (np.ndarray): Array of time values.
        values (np.ndarray): Array of signal values.

    Returns:
        dict: Dictionary containing peak analysis results.
              If no peaks are found, returns {"Result": "No peaks detected in this signal"}.
    """
    processed_values = safe_prepare_signal(values, dialog, "Peak Detection")
    if processed_values is None:
        return None

    sample_rate = safe_sample_rate(time_arr)
    if sample_rate == 0.0:
        QMessageBox.information(dialog, "Peak Detection", "Sampling rate could not be determined.")
        return None

    signal_mean = np.mean(processed_values)
    if np.all(processed_values < 0) or (
        np.any(processed_values < 0) and abs(np.min(processed_values) - signal_mean) > abs(np.max(processed_values) - signal_mean)):
        peak_type = "Negative"
        Logger.log_message_static("Signal is predominantly negative, inverting to detect valleys", Logger.DEBUG)
        processed = -processed_values
    else:
        peak_type = "Positive"
        Logger.log_message_static("Signal is predominantly positive or mixed, detecting peaks", Logger.DEBUG)
        processed = processed_values

    peak_indices, _ = sc_signal.find_peaks(processed)
    if len(peak_indices) == 0:
        return {"Result": "No peaks detected in this signal"}

    heights = processed[peak_indices]
    if peak_type == "Negative":
        heights = -heights

    peak_times = time_arr[peak_indices] if len(time_arr) == len(values) else np.arange(len(peak_indices)) / sample_rate
    widths = sc_signal.peak_widths(processed, peak_indices, rel_height=0.5)[0] / sample_rate

    results = {
        "Peak Type": peak_type,
        "Count": len(peak_indices),
        "Indices": peak_indices,
        "Times": peak_times,
        "Heights": heights,
        "Widths": widths,
        "Mean Height": float(np.mean(heights)) if len(heights) > 0 else 0,
        "Max Height": float(np.max(heights)) if len(heights) > 0 else 0,
        "Min Height": float(np.min(heights)) if len(heights) > 0 else 0,
        "Mean Width": float(np.mean(widths)) if len(widths) > 0 else 0,
    }
    Logger.log_message_static(f"Peak detection finished: {results['Count']} peaks found", Logger.DEBUG)
    return results

def calculate_hilbert_analysis(dialog, time_arr, values):
    """
    Perform Hilbert transform to extract amplitude envelope, phase and instantaneous frequency.

    Args:
        dialog (QWidget): Parent widget for displaying message boxes if needed.
        time_arr (np.ndarray): Time array of the signal.
        values (np.ndarray): Signal values.

    Returns:
        dict or None: Dictionary with Hilbert-related results, or None if cancelled or invalid input.
    """
    processed_values = safe_prepare_signal(values, dialog, "Hilbert Analysis")
    if processed_values is None:
        return None

    sample_rate = safe_sample_rate(time_arr)
    if sample_rate == 0.0:
        QMessageBox.information(dialog, "Hilbert Analysis", "Sampling rate could not be determined.")
        return None

    processed_values = extended_prepare_signal(values, dialog, "Hilbert Analysis")
    if processed_values is None:
        return None

    from scipy.signal import hilbert

    analytic = hilbert(processed_values)
    amplitude_envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    dt = 1.0 / sample_rate
    inst_freq = np.diff(phase) / (2.0 * np.pi * dt)
    inst_freq = np.append(inst_freq, inst_freq[-1])  # match length

    results = {
        "Amplitude Envelope": amplitude_envelope,
        "Unwrapped Phase": phase,
        "Instantaneous Frequency (Hz)": inst_freq,
        "Mean Frequency (Hz)": float(np.mean(inst_freq)),
        "Median Frequency (Hz)": float(np.median(inst_freq)),
        "Max Frequency (Hz)": float(np.max(inst_freq)),
        "Mean Amplitude": float(np.mean(amplitude_envelope)),
        "Max Amplitude": float(np.max(amplitude_envelope)),
        "Phase Range (rad)": float(np.max(phase) - np.min(phase))
    }

    Logger.log_message_static("Hilbert analysis completed", Logger.DEBUG)
    return results

def calculate_energy_analysis(dialog, time_arr, values):
    """
    Compute time-domain and frequency-domain energy, and energy distribution across logarithmic bands.

    Args:
        dialog (QWidget): Parent widget for optional warnings.
        time_arr (np.ndarray): Time axis.
        values (np.ndarray): Signal values.

    Returns:
        dict or None: Dictionary with energy metrics and per-band distribution, or None if cancelled/invalid.
    """
    processed_values = safe_prepare_signal(values, dialog, "Energy Analysis")
    if processed_values is None:
        return None

    sample_rate = safe_sample_rate(time_arr)
    if sample_rate == 0.0:
        QMessageBox.information(dialog, "Energy Analysis", "Sampling rate could not be determined.")
        return None

    processed_values = extended_prepare_signal(values, dialog, "Energy Analysis")
    if processed_values is None:
        return None

    n = len(processed_values)
    fft_vals = np.fft.rfft(processed_values)
    magnitude_squared = np.abs(fft_vals)**2 / n

    if n % 2 == 0:
        magnitude_squared[1:-1] *= 2
    else:
        magnitude_squared[1:] *= 2

    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    magnitude_squared[magnitude_squared <= 0] = 1e-12

    total_energy_time = np.sum(processed_values ** 2)
    total_energy_freq = np.sum(magnitude_squared)
    power = total_energy_time / n
    rms = np.sqrt(power)

    num_bands = 9
    if len(freqs) < 2:
        band_edges = [0.0, sample_rate / 2.0]
        num_bands = 1
    else:
        band_edges = np.logspace(np.log10(freqs[1]), np.log10(freqs[-1]), num=num_bands + 1)

    energy_per_band = {}
    band_percentages = []
    for i in range(num_bands):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        energy_band = np.sum(magnitude_squared[mask])
        percentage = (energy_band / total_energy_freq) * 100 if total_energy_freq > 0 else 0
        label = f"Band {i+1} ({band_edges[i]:.2f}-{band_edges[i+1]:.2f} Hz)"
        energy_per_band[label] = percentage
        band_percentages.append(percentage)

    dominant_idx = np.argmax(band_percentages)
    dominant_band = list(energy_per_band.keys())[dominant_idx]
    dominant_value = band_percentages[dominant_idx]

    results = {
        "Freqs": freqs,
        "Spectrum": magnitude_squared,
        "Total Energy (Time Domain)": total_energy_time,
        "Total Energy (Frequency Domain)": total_energy_freq,
        "Signal Power": power,
        "RMS Value": rms,
        "Energy Distribution (%)": energy_per_band,
        "Dominant Frequency Band": dominant_band,
        "Dominant Band Energy": f"{dominant_value:.1f}%",
        "Band Edges": band_edges,
        "Band Percentages": band_percentages
    }

    Logger.log_message_static("Energy analysis completed", Logger.DEBUG)
    return results

def calculate_phase_analysis(dialog, time_arr, values):
    """
    Compute unwrapped phase and phase change statistics from the Hilbert transform.

    Args:
        dialog (QWidget): Parent widget for displaying messages if needed.
        time_arr (np.ndarray): Time values corresponding to the signal.
        values (np.ndarray): Signal values.

    Returns:
        dict or None: Dictionary with phase, phase change, and statistics, or None if failed.
    """
    processed_values = safe_prepare_signal(values, dialog, "Phase Analysis")
    if processed_values is None:
        return None

    processed_values = extended_prepare_signal(values, dialog, "Phase Analysis")
    if processed_values is None:
        return None

    from scipy.signal import hilbert

    analytic = hilbert(processed_values)
    phase = np.unwrap(np.angle(analytic))
    dt = np.diff(time_arr)
    dphi = np.diff(phase)

    if len(dphi) == 0 or not np.all(np.isfinite(dphi)):
        QMessageBox.information(dialog, "Phase Analysis", "Phase data is not valid or too short.")
        return None

    phase_velocity = dphi / dt
    mean_phase_velocity = float(np.mean(np.abs(phase_velocity))) if len(phase_velocity) else 0.0

    result = {
        "Time": time_arr,
        "Phase": phase,
        "Phase Velocity": phase_velocity,
        "Phase Stats": {
            "Mean Phase": float(np.mean(phase)),
            "Phase Standard Deviation": float(np.std(phase)),
            "Phase Range": float(np.ptp(phase)),
            "Phase Rate of Change (rad/s)": mean_phase_velocity
        }
    }

    Logger.log_message_static("Phase analysis completed", Logger.DEBUG)
    return result

def calculate_cepstrum_analysis(dialog, time_arr, values):
    """
    Perform cepstrum analysis using log power spectrum and inverse FFT.

    Args:
        dialog (QWidget): Parent widget for displaying messages.
        time_arr (np.ndarray): Time axis of the signal.
        values (np.ndarray): Input signal.

    Returns:
        dict or None: Dictionary with cepstrum results and detected peaks, or None on failure.
    """
    processed_values = safe_prepare_signal(values, dialog, "Cepstrum Analysis")
    if processed_values is None:
        return None

    processed_values = extended_prepare_signal(values, dialog, "Cepstrum Analysis")
    if processed_values is None:
        return None

    if len(time_arr) < 2:
        QMessageBox.warning(dialog, "Cepstrum Analysis", "Signal is too short for analysis.")
        return None

    fs = abs(1 / np.mean(np.diff(time_arr)))
    n = len(processed_values)
    fft_vals = np.fft.fft(processed_values)

    log_power = np.log(np.abs(fft_vals) ** 2 + 1e-10)
    cepstrum = np.fft.ifft(log_power).real
    quefrency = np.arange(n) / fs

    skip_samples = int(0.002 * fs)
    if skip_samples >= n // 2:
        skip_samples = n // 10

    search_range = cepstrum[skip_samples:n // 2]
    peaks, _ = sc_signal.find_peaks(search_range, height=0.1 * np.max(search_range))
    peaks = peaks + skip_samples

    fundamental_freq = 0.0
    peak_quefrency = 0.0
    if len(peaks) > 0:
        best_idx = peaks[np.argmax(cepstrum[peaks])]
        peak_quefrency = quefrency[best_idx]
        fundamental_freq = 1.0 / peak_quefrency if peak_quefrency > 0 else 0.0

    peak_data = []
    for idx in peaks:
        q = quefrency[idx]
        val = cepstrum[idx]
        freq = 1.0 / q if q > 0 else 0.0
        peak_data.append((q, val, freq))

    result = {
        "Cepstrum": cepstrum,
        "Quefrency": quefrency,
        "Log Power Spectrum": log_power,
        "Sampling Rate": fs,
        "Fundamental Frequency (Hz)": float(fundamental_freq),
        "Peak Quefrency (s)": float(peak_quefrency),
        "Peaks": peak_data[:3],  # limit to top 3 for display
        "Max Cepstrum Value": float(np.max(search_range)),
        "Mean Cepstrum Value": float(np.mean(search_range))
    }

    Logger.log_message_static("Cepstrum analysis completed", Logger.DEBUG)
    return result

def calculate_autocorrelation_analysis(dialog, time_arr, values):
    """
    Compute autocorrelation of a signal and detect key lags (zero crossing, minimum, decorrelation).

    Args:
        dialog (QWidget): Parent widget for optional messages.
        time_arr (np.ndarray): Time vector of the signal.
        values (np.ndarray): Signal values.

    Returns:
        dict or None: Dictionary with ACF and lag info, or None if processing fails.
    """
    processed_values = safe_prepare_signal(values, dialog, "Autocorrelation Analysis")
    if processed_values is None:
        return None

    sample_rate = safe_sample_rate(time_arr)
    if sample_rate == 0.0:
        QMessageBox.information(dialog, "Autocorrelation Analysis", "Sampling rate could not be determined.")
        return None

    detrended = processed_values - np.mean(processed_values)
    if np.max(np.abs(detrended)) == 0:
        QMessageBox.information(dialog, "Autocorrelation Analysis", "Signal has no variation.")
        return None

    windowed = detrended * np.hanning(len(detrended))
    acf = correlate(windowed, windowed, mode='full')
    max_val = np.max(np.abs(acf))
    if max_val > 0:
        acf = acf / max_val
    else:
        acf[:] = 0

    center = len(acf) // 2
    pos_acf = acf[center:]
    lags = np.arange(len(pos_acf)) / sample_rate

    first_min_idx = None
    for i in range(1, len(pos_acf) - 1):
        if pos_acf[i] < pos_acf[i - 1] and pos_acf[i] < pos_acf[i + 1]:
            first_min_idx = i
            break

    zero_cross_idx = np.where(np.diff(np.signbit(pos_acf)))[0]
    first_zero_idx = zero_cross_idx[0] if len(zero_cross_idx) > 0 else None

    result = {
        "Autocorrelation": pos_acf,
        "Lags (s)": lags,
        "First Minimum (s)": lags[first_min_idx] if first_min_idx is not None else "Not found",
        "First Zero Crossing (s)": lags[first_zero_idx] if first_zero_idx is not None else "Not found",
        "Decorrelation Time (s)": lags[first_zero_idx] if first_zero_idx is not None else "Not found",
        "Peak Correlation": 1.0
    }

    Logger.log_message_static("Autocorrelation analysis calculated", Logger.DEBUG)
    return result

def calculate_cross_correlation_analysis(dialog, time_arr1, values1, time_arr2, values2):
    """
    Compute normalized cross-correlation between two signals with extended statistics.

    Args:
        dialog (QWidget): Parent widget for optional user messages.
        time_arr1 (np.ndarray): Time axis of first signal.
        values1 (np.ndarray): First signal values.
        time_arr2 (np.ndarray): Time axis of second signal.
        values2 (np.ndarray): Second signal values.

    Returns:
        dict or None: Dictionary with correlation data and detailed statistics, or None on error.
    """
    values1 = safe_prepare_signal(values1, dialog, "Cross-Correlation Analysis (Signal 1)")
    values2 = safe_prepare_signal(values2, dialog, "Cross-Correlation Analysis (Signal 2)")
    if values1 is None or values2 is None:
        return None

    sample_rate1 = safe_sample_rate(time_arr1)
    sample_rate2 = safe_sample_rate(time_arr2)
    if sample_rate1 == 0.0 or sample_rate2 == 0.0:
        QMessageBox.information(dialog, "Cross-Correlation Analysis", "Sampling rate could not be determined.")
        return None

    if abs(sample_rate1 - sample_rate2) > 1e-3:
        QMessageBox.information(dialog, "Cross-Correlation Analysis", "Sampling rates are not compatible.")
        return None

    length = min(len(values1), len(values2))
    x1 = values1[:length] - np.mean(values1[:length])
    x2 = values2[:length] - np.mean(values2[:length])

    corr = np.correlate(x1, x2, mode='full')
    lags = np.arange(-length + 1, length) / sample_rate1

    norm = np.sqrt(np.sum(x1**2) * np.sum(x2**2))
    corr_norm = corr / norm if norm > 1e-12 else np.zeros_like(corr)

    max_idx = np.argmax(corr_norm)
    max_corr = float(corr_norm[max_idx])
    max_lag = float(lags[max_idx])
    zero_lag_corr = float(corr_norm[len(corr_norm) // 2])

    thresholds = [0.5, 0.7, 0.9]
    threshold_lags = {}
    for th in thresholds:
        indices = np.where(corr_norm >= th * max_corr)[0]
        if len(indices) > 0:
            lag_range = lags[indices[-1]] - lags[indices[0]]
            threshold_lags[f"Correlation Width {int(th * 100)}% (s)"] = float(lag_range)
        else:
            threshold_lags[f"Correlation Width {int(th * 100)}% (s)"] = "N/A"

    results = {
        "Cross-Correlation": corr_norm,
        "Lags (s)": lags,
        "Max Correlation": max_corr,
        "Lag at Max Correlation (s)": max_lag,
        "Correlation at Zero Lag": zero_lag_corr,
        **threshold_lags
    }

    Logger.log_message_static("Extended cross-correlation analysis completed", Logger.DEBUG)
    return results

def calculate_wavelet_analysis_cwt(dialog, time_arr, values, wavelet_name="cmor1.5-1.0", num_scales=32):
    """
    Perform Continuous Wavelet Transform (CWT) analysis on a signal.

    Args:
        dialog (QWidget): Parent widget for messages.
        time_arr (np.ndarray): Time array.
        values (np.ndarray): Signal values.
        wavelet_name (str): Fully qualified wavelet name (e.g., "cmor1.5-1.0", "morl", "mexh").
        num_scales (int): Number of scales.

    Returns:
        dict or None: Analysis results including spectrogram-ready data.
    """
    processed = safe_prepare_signal(values, dialog, "Wavelet Analysis (CWT)")
    if processed is None:
        return None

    sample_rate = safe_sample_rate(time_arr)
    if sample_rate == 0.0:
        QMessageBox.information(dialog, "Wavelet Analysis", "Sampling rate could not be determined.")
        return None

    dt = 1.0 / sample_rate
    try:
        scales = np.arange(1, num_scales + 1)
        coeffs, freqs = pywt.cwt(processed, scales, wavelet_name, sampling_period=dt)
        power = np.abs(coeffs) ** 2
        avg_power = np.mean(power, axis=1)
        total_energy = float(np.sum(power))

        dominant_idx = int(np.argmax(avg_power))
        dominant_freq = float(freqs[dominant_idx])
        high_energy_ratio = float(
            np.sum(avg_power[:len(avg_power) // 2]) / np.sum(avg_power[len(avg_power) // 2:])
        ) if np.sum(avg_power[len(avg_power) // 2:]) > 0 else np.inf

        return {
            "Frequencies (Hz)": freqs,
            "Power per Scale": avg_power,
            "Dominant Frequency (Hz)": dominant_freq,
            "Total Energy": total_energy,
            "High/Low Energy Ratio": high_energy_ratio,
            "Top Frequency Bands": [
                {
                    "Frequency (Hz)": float(freqs[i]),
                    "Energy (%)": float(100 * avg_power[i] / total_energy)
                }
                for i in np.argsort(avg_power)[-3:][::-1]
            ],
            "Coefficients": coeffs,
            "Time Array": time_arr,
            "Scales": scales
        }

    except Exception as e:
        Logger.log_message_static(f"Wavelet CWT analysis failed: {e}", Logger.ERROR)
        return None

def calculate_wavelet_analysis_dwt(dialog, time_arr, values, wavelet_name="db4", num_levels=5):
    """
    Perform Discrete Wavelet Transform (DWT) analysis on a signal.

    Args:
        dialog (QWidget): Parent widget for messages.
        time_arr (np.ndarray): Time array (used only for logging/sample check).
        values (np.ndarray): Signal values.
        wavelet_name (str): Wavelet name (e.g., "db4", "sym4").
        num_levels (int): Desired decomposition levels (log2-based).

    Returns:
        dict or None: Dictionary with energy per level, total energy, etc.
    """
    processed = safe_prepare_signal(values, dialog, "Wavelet Analysis (DWT)")
    if processed is None:
        return None

    try:
        wavelet = pywt.Wavelet(wavelet_name)
        max_level = pywt.dwt_max_level(len(processed), wavelet.dec_len)
        level = min(num_levels, max_level)

        coeffs = pywt.wavedec(processed, wavelet, level=level)
        energies = [np.sum(np.square(c)) for c in coeffs]
        total_energy = float(np.sum(energies))

        results = {
            f"Level {i} Energy": float(energies[i]) for i in range(len(energies))
        }
        results["Total Energy"] = total_energy
        results["Wavelet Type"] = wavelet_name
        results["Decomposition Level"] = level

        Logger.log_message_static("Wavelet DWT analysis completed", Logger.DEBUG)
        return results

    except Exception as e:
        Logger.log_message_static(f"Wavelet DWT analysis failed: {e}", Logger.ERROR)
        return None

def calculate_iir_filter(values, time_arr, filter_type="lowpass", cutoff_freq=1.0, order=4):
    """
    Apply an IIR (Butterworth) filter to the input signal.

    Args:
        values (np.ndarray): Signal data.
        time_arr (np.ndarray): Corresponding time vector.
        filter_type (str): Type of filter ("lowpass", "highpass", "bandpass", "bandstop").
        cutoff_freq (float or tuple): Cutoff frequency (Hz), tuple for band filters.
        order (int): Filter order.

    Returns:
        dict or None: Dictionary with filtered signal and parameters or None on error.
    """
    sample_rate = safe_sample_rate(time_arr)
    if sample_rate == 0.0:
        Logger.log_message_static("Invalid sampling rate", Logger.ERROR)
        return None

    nyquist = 0.5 * sample_rate
    try:
        if filter_type in ["lowpass", "highpass"]:
            norm_cutoff = cutoff_freq / nyquist
            b, a = sc_signal.butter(order, norm_cutoff, btype=filter_type)
        elif filter_type in ["bandpass", "bandstop"]:
            if not isinstance(cutoff_freq, (list, tuple)) or len(cutoff_freq) != 2:
                Logger.log_message_static("Band filters require two cutoff frequencies", Logger.ERROR)
                return None
            norm_cutoff = [f / nyquist for f in cutoff_freq]
            b, a = sc_signal.butter(order, norm_cutoff, btype=filter_type)
        else:
            Logger.log_message_static(f"Unsupported filter type: {filter_type}", Logger.ERROR)
            return None

        filtered = sc_signal.filtfilt(b, a, values)
        return {
            "Filtered Signal": filtered,
            "Filter Type": filter_type,
            "Cutoff Frequency (Hz)": cutoff_freq,
            "Filter Order": order,
            "Sample Rate (Hz)": sample_rate
        }

    except Exception as e:
        Logger.log_message_static(f"IIR filtering error: {e}", Logger.ERROR)
        return None

def calculate_fir_filter(values, time_arr, filter_type="lowpass", cutoff_freq=1.0,
                         numtaps=101, window="hamming"):
    """
    Apply an FIR filter (using firwin) to the input signal.

    Args:
        values (np.ndarray): Signal data.
        time_arr (np.ndarray): Corresponding time vector.
        filter_type (str): Type of filter ("lowpass", "highpass", "bandpass", "bandstop").
        cutoff_freq (float or tuple): Cutoff frequency (Hz), tuple for band filters.
        numtaps (int): Number of filter taps (FIR order).
        window (str): FIR window function (e.g., "hamming", "hann", "blackman").

    Returns:
        dict or None: Dictionary with filtered signal and parameters or None on error.
    """
    sample_rate = safe_sample_rate(time_arr)
    if sample_rate == 0.0:
        Logger.log_message_static("Invalid sampling rate", Logger.ERROR)
        return None

    nyquist = 0.5 * sample_rate
    try:
        if filter_type in ["lowpass", "highpass"]:
            norm_cutoff = cutoff_freq / nyquist
            fir_coeffs = sc_signal.firwin(numtaps, norm_cutoff,
                                       window=window,
                                       pass_zero=(filter_type == "lowpass"))
        elif filter_type in ["bandpass", "bandstop"]:
            if not isinstance(cutoff_freq, (list, tuple)) or len(cutoff_freq) != 2:
                Logger.log_message_static("Band filters require two cutoff frequencies", Logger.ERROR)
                return None
            norm_cutoff = [f / nyquist for f in cutoff_freq]
            fir_coeffs = sc_signal.firwin(numtaps, norm_cutoff,
                                       window=window,
                                       pass_zero=(filter_type == "bandstop"))
        else:
            Logger.log_message_static(f"Unsupported filter type: {filter_type}", Logger.ERROR)
            return None

        filtered = sc_signal.filtfilt(fir_coeffs, [1.0], values)
        return {
            "Filtered Signal": filtered,
            "Filter Type": filter_type,
            "Cutoff Frequency (Hz)": cutoff_freq,
            "Taps": numtaps,
            "Window": window,
            "Sample Rate (Hz)": sample_rate
        }

    except Exception as e:
        Logger.log_message_static(f"FIR filtering error: {e}", Logger.ERROR)
        return None




