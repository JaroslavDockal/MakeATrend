"""
Vibration envelope analysis for bearing and gear fault detection.

This module provides advanced envelope analysis capabilities specifically designed
for detecting bearing faults, gear tooth damage, and other impulsive mechanical
faults in rotating machinery. Uses Hilbert transform and advanced filtering
techniques optimized for fault detection.

Key features:
- Adaptive bandpass filtering for envelope extraction
- Multiple envelope techniques (Hilbert, peak detection, RMS)
- Bearing fault frequency detection in envelope spectrum
- Gear fault detection and sidebands analysis
- Multi-band envelope analysis for broadband faults
- Statistical envelope analysis for trending
"""

import numpy as np
from scipy import signal as sc_signal
from scipy.fft import rfft, rfftfreq
from analysis.calculations.common import safe_prepare_signal, safe_sample_rate
from utils.logger import Logger


def calculate_envelope_analysis(time_arr, values, dialog=None, title="Envelope Analysis",
                                filter_type="adaptive", filter_range=None, envelope_method="hilbert"):
    """
    Perform comprehensive envelope analysis for bearing and gear fault detection.

    Uses advanced filtering and envelope extraction techniques to isolate and analyze
    impulsive fault signatures. Automatically adapts filter parameters based on
    signal characteristics and provides comprehensive fault detection capabilities.

    Args:
        time_arr (np.ndarray): Time values array.
        values (np.ndarray): Vibration signal values (typically acceleration).
        dialog (QWidget, optional): Parent dialog for user interaction.
        title (str, optional): Title for user dialogs.
        filter_type (str, optional): Filtering approach:
            - "adaptive": Automatically determine optimal filter band
            - "bearing": Optimized for bearing fault detection (500Hz-10kHz)
            - "gear": Optimized for gear fault detection (1kHz-15kHz)
            - "custom": Use specified filter_range
            - "multiband": Analyze multiple frequency bands
        filter_range (tuple, optional): Custom filter range (low_freq, high_freq) in Hz.
        envelope_method (str, optional): Envelope extraction method:
            - "hilbert": Hilbert transform (most common)
            - "rms": RMS envelope
            - "peak": Peak envelope
            - "combined": Combination of methods

    Returns:
        dict or None: Dictionary containing envelope analysis results:
            - Time Array: Time values
            - Original Signal: Input signal
            - Filtered Signal: Bandpass filtered signal
            - Envelope: Extracted envelope
            - Envelope Spectrum: FFT of envelope
            - Envelope Frequencies: Frequency axis for envelope spectrum
            - Peak Detection: Peaks in envelope spectrum
            - Fault Detection: Detected fault frequencies
            - Statistical Analysis: Envelope statistics
            - Filter Parameters: Used filter settings
            - Quality Metrics: Analysis quality indicators

        Returns None if validation fails.

    Example:
        >>> import numpy as np
        >>> t = np.linspace(0, 5, 50000)  # 5 seconds at 10kHz
        >>> # Simulate bearing fault with impulses at BPFO frequency
        >>> carrier = np.random.randn(50000) * 0.1  # Background noise
        >>> impulse_times = np.arange(0, 5, 1/85)  # 85 Hz BPFO
        >>> for imp_time in impulse_times:
        >>>     idx = int(imp_time * 10000)
        >>>     if idx < len(carrier):
        >>>         carrier[idx:idx+10] += np.exp(-np.arange(10)*0.5) * 2
        >>> result = calculate_envelope_analysis(t, carrier, filter_type="bearing")
        >>> print(f"Envelope peaks at: {result['Peak Detection']['Frequencies'][:3]} Hz")
    """
    Logger.log_message_static(f"Vibration-Envelope: Starting envelope analysis ({filter_type})", Logger.DEBUG)

    # Validate and prepare signal
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Vibration-Envelope: Signal validation failed", Logger.WARNING)
        return None

    try:
        # Get sampling parameters
        sample_rate = safe_sample_rate(time_arr)
        if sample_rate <= 0:
            Logger.log_message_static("Vibration-Envelope: Invalid sampling rate", Logger.ERROR)
            return None

        nyquist = sample_rate / 2

        # Determine optimal filter parameters
        filter_params = determine_filter_parameters(processed_values, sample_rate, filter_type, filter_range)
        if filter_params is None:
            Logger.log_message_static("Vibration-Envelope: Failed to determine filter parameters", Logger.ERROR)
            return None

        # Apply bandpass filtering
        filtered_signal = apply_envelope_filter(processed_values, sample_rate, filter_params)
        if filtered_signal is None:
            Logger.log_message_static("Vibration-Envelope: Filtering failed", Logger.ERROR)
            return None

        # Extract envelope using specified method
        envelope = extract_envelope(filtered_signal, envelope_method, sample_rate)
        if envelope is None:
            Logger.log_message_static("Vibration-Envelope: Envelope extraction failed", Logger.ERROR)
            return None

        # Analyze envelope spectrum
        envelope_spectrum_results = analyze_envelope_spectrum(envelope, sample_rate)

        # Detect peaks in envelope spectrum
        peak_detection = detect_envelope_peaks(
            envelope_spectrum_results['frequencies'],
            envelope_spectrum_results['magnitude'],
            sample_rate
        )

        # Perform fault detection analysis
        fault_detection = detect_envelope_faults(peak_detection, filter_params)

        # Calculate envelope statistics
        envelope_stats = calculate_envelope_statistics(envelope, filtered_signal)

        # Assess analysis quality
        quality_metrics = assess_envelope_quality(
            processed_values, filtered_signal, envelope,
            envelope_spectrum_results, filter_params
        )

        # Multi-band analysis if requested
        multiband_results = None
        if filter_type == "multiband":
            multiband_results = perform_multiband_envelope_analysis(
                processed_values, time_arr, sample_rate
            )

        # Build comprehensive results
        results = {
            # Core data
            "Time Array": time_arr,
            "Original Signal": processed_values,
            "Filtered Signal": filtered_signal,
            "Envelope": envelope,

            # Spectrum analysis
            "Envelope Frequencies": envelope_spectrum_results['frequencies'],
            "Envelope Spectrum": envelope_spectrum_results['magnitude'],
            "Envelope Spectrum dB": envelope_spectrum_results['magnitude_db'],
            "Envelope Phase": envelope_spectrum_results['phase'],

            # Detection results
            "Peak Detection": peak_detection,
            "Fault Detection": fault_detection,
            "Envelope Statistics": envelope_stats,
            "Quality Metrics": quality_metrics,

            # Analysis parameters
            "Filter Parameters": filter_params,
            "Envelope Method": envelope_method,
            "Sample Rate (Hz)": float(sample_rate),
            "Filter Type": filter_type,

            # Advanced analysis
            "Multiband Results": multiband_results,
            "Spectral Kurtosis": calculate_spectral_kurtosis(processed_values, sample_rate),
            "Envelope Harmonics": detect_envelope_harmonics(envelope_spectrum_results)
        }

        Logger.log_message_static(
            f"Vibration-Envelope: Analysis completed. "
            f"Filter: {filter_params['low_freq']:.0f}-{filter_params['high_freq']:.0f} Hz, "
            f"Envelope peaks: {len(peak_detection['frequencies'])}, "
            f"Quality: {quality_metrics['overall_quality']}",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Error in envelope analysis: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Vibration-Envelope: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None


def determine_filter_parameters(signal, sample_rate, filter_type, custom_range=None):
    """
    Automatically determine optimal filter parameters for envelope analysis.

    Uses signal characteristics and fault detection requirements to select
    the most appropriate frequency band for envelope extraction.
    """
    try:
        nyquist = sample_rate / 2

        if filter_type == "custom" and custom_range:
            low_freq, high_freq = custom_range
            return {
                'low_freq': max(10, min(low_freq, nyquist * 0.8)),
                'high_freq': max(low_freq + 100, min(high_freq, nyquist * 0.95)),
                'filter_order': 4,
                'filter_design': 'butterworth'
            }

        elif filter_type == "bearing":
            # Optimized for bearing fault detection
            # Bearing impacts typically generate energy in 500Hz-10kHz range
            low_freq = max(500, sample_rate * 0.02)  # At least 2% of sample rate
            high_freq = min(10000, nyquist * 0.8)  # Max 80% of Nyquist

        elif filter_type == "gear":
            # Optimized for gear fault detection
            # Gear mesh and tooth damage in higher frequency range
            low_freq = max(1000, sample_rate * 0.05)  # At least 5% of sample rate
            high_freq = min(15000, nyquist * 0.85)  # Max 85% of Nyquist

        elif filter_type == "adaptive":
            # Analyze signal spectrum to find optimal band
            optimal_band = find_optimal_envelope_band(signal, sample_rate)
            low_freq = optimal_band['low_freq']
            high_freq = optimal_band['high_freq']

        else:
            # Default bearing-optimized parameters
            low_freq = max(500, sample_rate * 0.02)
            high_freq = min(8000, nyquist * 0.8)

        # Ensure valid frequency range
        if high_freq <= low_freq:
            low_freq = max(500, sample_rate * 0.02)
            high_freq = min(8000, nyquist * 0.8)

        return {
            'low_freq': float(low_freq),
            'high_freq': float(high_freq),
            'filter_order': 4,
            'filter_design': 'butterworth',
            'filter_type_used': filter_type
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Error determining filter parameters: {e}", Logger.WARNING)
        return None


def find_optimal_envelope_band(signal, sample_rate):
    """
    Find optimal frequency band for envelope analysis using spectral kurtosis.

    Spectral kurtosis identifies frequency bands with the highest impulsive content,
    which are optimal for envelope analysis of bearing and gear faults.
    """
    try:
        # Calculate FFT
        n = len(signal)
        freqs = rfftfreq(n, 1 / sample_rate)
        fft_vals = rfft(signal)

        # Define candidate frequency bands
        nyquist = sample_rate / 2
        band_width = 2000  # 2 kHz bands

        bands = []
        f_start = 500  # Start at 500 Hz

        while f_start + band_width < nyquist * 0.9:
            f_end = f_start + band_width
            bands.append((f_start, f_end))
            f_start += band_width // 2  # 50% overlap

        # Calculate kurtosis for each band
        best_kurtosis = -10
        best_band = (500, min(8000, nyquist * 0.8))

        for low_f, high_f in bands:
            # Extract band
            band_mask = (freqs >= low_f) & (freqs <= high_f)
            if np.sum(band_mask) < 10:  # Need sufficient samples
                continue

            # Filter signal in this band
            try:
                sos = sc_signal.butter(4, [low_f, high_f], btype='band',
                                       fs=sample_rate, output='sos')
                filtered = sc_signal.sosfilt(sos, signal)

                # Calculate envelope
                analytic = sc_signal.hilbert(filtered)
                envelope = np.abs(analytic)

                # Calculate kurtosis of envelope
                from scipy import stats
                env_kurtosis = stats.kurtosis(envelope)

                if env_kurtosis > best_kurtosis:
                    best_kurtosis = env_kurtosis
                    best_band = (low_f, high_f)

            except Exception:
                continue

        return {
            'low_freq': best_band[0],
            'high_freq': best_band[1],
            'kurtosis': best_kurtosis
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Error finding optimal band: {e}", Logger.WARNING)
        return {'low_freq': 500, 'high_freq': 8000, 'kurtosis': 0}


def apply_envelope_filter(signal, sample_rate, filter_params):
    """Apply bandpass filter optimized for envelope analysis."""
    try:
        low_freq = filter_params['low_freq']
        high_freq = filter_params['high_freq']
        order = filter_params.get('filter_order', 4)

        # Ensure valid frequency range
        nyquist = sample_rate / 2
        low_freq = max(1, min(low_freq, nyquist * 0.95))
        high_freq = max(low_freq + 10, min(high_freq, nyquist * 0.95))

        # Design Butterworth bandpass filter
        sos = sc_signal.butter(order, [low_freq, high_freq],
                               btype='band', fs=sample_rate, output='sos')

        # Apply zero-phase filtering
        filtered = sc_signal.sosfiltfilt(sos, signal)

        return filtered

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Filter application error: {e}", Logger.WARNING)
        return None


def extract_envelope(signal, method="hilbert", sample_rate=None):
    """Extract envelope using specified method."""
    try:
        if method == "hilbert":
            # Hilbert transform - most common method
            analytic = sc_signal.hilbert(signal)
            envelope = np.abs(analytic)

        elif method == "rms":
            # RMS envelope - good for trending
            window_size = max(10, int(sample_rate * 0.001)) if sample_rate else 50  # 1ms window
            envelope = np.array([
                np.sqrt(np.mean(signal[max(0, i - window_size // 2):i + window_size // 2] ** 2))
                for i in range(len(signal))
            ])

        elif method == "peak":
            # Peak envelope - preserves transients
            window_size = max(5, int(sample_rate * 0.0005)) if sample_rate else 25  # 0.5ms window
            envelope = np.array([
                np.max(np.abs(signal[max(0, i - window_size // 2):i + window_size // 2]))
                for i in range(len(signal))
            ])

        elif method == "combined":
            # Combination of methods
            hilbert_env = np.abs(sc_signal.hilbert(signal))

            window_size = max(10, int(sample_rate * 0.001)) if sample_rate else 50
            rms_env = np.array([
                np.sqrt(np.mean(signal[max(0, i - window_size // 2):i + window_size // 2] ** 2))
                for i in range(len(signal))
            ])

            # Weighted combination (favor Hilbert)
            envelope = 0.7 * hilbert_env + 0.3 * rms_env

        else:
            # Default to Hilbert
            analytic = sc_signal.hilbert(signal)
            envelope = np.abs(analytic)

        return envelope

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Envelope extraction error: {e}", Logger.WARNING)
        return None


def analyze_envelope_spectrum(envelope, sample_rate):
    """Analyze the frequency content of the envelope signal."""
    try:
        # Remove DC component
        envelope_centered = envelope - np.mean(envelope)

        # Apply window to reduce spectral leakage
        window = np.hanning(len(envelope_centered))
        windowed_envelope = envelope_centered * window

        # Calculate FFT
        n = len(windowed_envelope)
        yf = rfft(windowed_envelope)
        xf = rfftfreq(n, 1 / sample_rate)

        # Calculate magnitude and phase
        magnitude = 2.0 * np.abs(yf) / n  # Single-sided spectrum
        magnitude[0] /= 2  # Correct DC component
        phase = np.angle(yf)

        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-12)

        return {
            'frequencies': xf,
            'magnitude': magnitude,
            'magnitude_db': magnitude_db,
            'phase': phase
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Spectrum analysis error: {e}", Logger.WARNING)
        return {
            'frequencies': np.array([]),
            'magnitude': np.array([]),
            'magnitude_db': np.array([]),
            'phase': np.array([])
        }


def detect_envelope_peaks(frequencies, magnitude, sample_rate):
    """Detect significant peaks in the envelope spectrum."""
    try:
        if len(frequencies) == 0 or len(magnitude) == 0:
            return {
                'frequencies': np.array([]),
                'magnitudes': np.array([]),
                'indices': np.array([])
            }

        # Calculate noise floor
        noise_floor = np.percentile(magnitude, 80)
        threshold = max(noise_floor * 2, np.max(magnitude) * 0.05)

        # Find peaks with minimum separation
        min_distance = max(1, int(2 / (frequencies[1] - frequencies[0])))  # 2 Hz minimum separation

        peaks, properties = sc_signal.find_peaks(
            magnitude,
            height=threshold,
            distance=min_distance,
            prominence=threshold * 0.3
        )

        if len(peaks) == 0:
            return {
                'frequencies': np.array([]),
                'magnitudes': np.array([]),
                'indices': np.array([])
            }

        # Sort by magnitude
        peak_magnitudes = magnitude[peaks]
        sort_indices = np.argsort(peak_magnitudes)[::-1]

        # Limit to top 30 peaks
        max_peaks = 30
        if len(sort_indices) > max_peaks:
            sort_indices = sort_indices[:max_peaks]

        sorted_peaks = peaks[sort_indices]
        sorted_frequencies = frequencies[sorted_peaks]
        sorted_magnitudes = peak_magnitudes[sort_indices]

        return {
            'frequencies': sorted_frequencies,
            'magnitudes': sorted_magnitudes,
            'indices': sorted_peaks,
            'threshold_used': threshold,
            'noise_floor': noise_floor
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Peak detection error: {e}", Logger.WARNING)
        return {
            'frequencies': np.array([]),
            'magnitudes': np.array([]),
            'indices': np.array([])
        }


def detect_envelope_faults(peak_detection, filter_params):
    """Detect potential faults based on envelope spectrum peaks."""
    try:
        frequencies = peak_detection['frequencies']
        magnitudes = peak_detection['magnitudes']

        if len(frequencies) == 0:
            return {
                'bearing_faults': [],
                'gear_faults': [],
                'electrical_faults': [],
                'other_faults': [],
                'fault_summary': 'No significant peaks detected'
            }

        bearing_faults = []
        gear_faults = []
        electrical_faults = []
        other_faults = []

        # Typical fault frequency ranges
        for i, (freq, mag) in enumerate(zip(frequencies, magnitudes)):
            # Bearing fault frequency ranges (typical)
            if 20 <= freq <= 500:
                bearing_faults.append({
                    'frequency': float(freq),
                    'magnitude': float(mag),
                    'type': 'Bearing outer race (typical range)',
                    'confidence': 'medium'
                })
            elif 100 <= freq <= 1000:
                bearing_faults.append({
                    'frequency': float(freq),
                    'magnitude': float(mag),
                    'type': 'Bearing inner race (typical range)',
                    'confidence': 'medium'
                })

            # Electrical fault indicators
            if abs(freq - 50) < 2 or abs(freq - 60) < 2:  # Line frequency
                electrical_faults.append({
                    'frequency': float(freq),
                    'magnitude': float(mag),
                    'type': 'Line frequency (50/60 Hz)',
                    'confidence': 'high'
                })
            elif abs(freq - 100) < 2 or abs(freq - 120) < 2:  # 2x line frequency
                electrical_faults.append({
                    'frequency': float(freq),
                    'magnitude': float(mag),
                    'type': '2x Line frequency',
                    'confidence': 'high'
                })

            # Low frequency faults (unbalance, misalignment)
            elif freq < 20:
                other_faults.append({
                    'frequency': float(freq),
                    'magnitude': float(mag),
                    'type': 'Low frequency (unbalance/misalignment)',
                    'confidence': 'medium'
                })

        # Generate summary
        total_faults = len(bearing_faults) + len(gear_faults) + len(electrical_faults) + len(other_faults)

        if total_faults == 0:
            fault_summary = f"No recognized fault patterns. {len(frequencies)} peaks detected."
        else:
            fault_summary = f"Detected {total_faults} potential fault indicators: "
            fault_summary += f"{len(bearing_faults)} bearing, {len(gear_faults)} gear, "
            fault_summary += f"{len(electrical_faults)} electrical, {len(other_faults)} other"

        return {
            'bearing_faults': bearing_faults,
            'gear_faults': gear_faults,
            'electrical_faults': electrical_faults,
            'other_faults': other_faults,
            'fault_summary': fault_summary,
            'total_peaks': len(frequencies),
            'total_faults': total_faults
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Fault detection error: {e}", Logger.WARNING)
        return {'fault_summary': 'Fault detection failed', 'total_faults': 0}


def calculate_envelope_statistics(envelope, filtered_signal):
    """Calculate statistical measures of the envelope."""
    try:
        from scipy import stats

        # Basic statistics
        env_mean = np.mean(envelope)
        env_std = np.std(envelope)
        env_rms = np.sqrt(np.mean(envelope ** 2))
        env_peak = np.max(envelope)
        env_p2p = np.max(envelope) - np.min(envelope)

        # Shape factors
        env_crest = env_peak / env_rms if env_rms > 0 else 0
        env_form = env_rms / env_mean if env_mean > 0 else 0

        # Higher order statistics
        env_skew = stats.skew(envelope)
        env_kurt = stats.kurtosis(envelope)

        # Envelope regularity measures
        env_diff = np.diff(envelope)
        env_smoothness = np.std(env_diff) / env_std if env_std > 0 else 0

        # Compare to filtered signal
        sig_rms = np.sqrt(np.mean(filtered_signal ** 2))
        envelope_factor = env_rms / sig_rms if sig_rms > 0 else 0

        return {
            'Mean': float(env_mean),
            'Standard Deviation': float(env_std),
            'RMS': float(env_rms),
            'Peak': float(env_peak),
            'Peak-to-Peak': float(env_p2p),
            'Crest Factor': float(env_crest),
            'Form Factor': float(env_form),
            'Skewness': float(env_skew),
            'Kurtosis': float(env_kurt),
            'Smoothness': float(env_smoothness),
            'Envelope Factor': float(envelope_factor)
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Statistics calculation error: {e}", Logger.WARNING)
        return {}


def assess_envelope_quality(original, filtered, envelope, spectrum_results, filter_params):
    """Assess the quality of the envelope analysis."""
    try:
        # Signal-to-noise ratio assessment
        signal_power = np.var(filtered)
        noise_estimate = np.var(original - filtered)
        snr = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 0 else np.inf

        # Envelope quality indicators
        envelope_snr = np.var(envelope) / np.var(np.diff(envelope)) if np.var(np.diff(envelope)) > 0 else np.inf
        envelope_snr_db = 10 * np.log10(envelope_snr) if envelope_snr != np.inf else 100

        # Spectral quality
        spectrum_peaks = len(spectrum_results['frequencies']) if 'frequencies' in spectrum_results else 0
        spectral_concentration = np.max(spectrum_results['magnitude']) / np.mean(spectrum_results['magnitude']) if len(
            spectrum_results['magnitude']) > 0 else 1

        # Filter effectiveness
        filter_bw = filter_params['high_freq'] - filter_params['low_freq']
        filter_effectiveness = min(1.0, filter_bw / 1000)  # Normalized to 1 kHz

        # Overall quality score (0-100)
        quality_score = 0

        # SNR contribution (30%)
        snr_score = min(30, max(0, (snr - 10) * 2))  # 10-25 dB range
        quality_score += snr_score

        # Envelope SNR contribution (25%)
        env_snr_score = min(25, max(0, (envelope_snr_db - 20) * 1.25))  # 20-40 dB range
        quality_score += env_snr_score

        # Spectral characteristics (25%)
        spectral_score = min(25, spectrum_peaks * 2.5)  # Up to 10 peaks
        quality_score += spectral_score

        # Filter effectiveness (20%)
        filter_score = filter_effectiveness * 20
        quality_score += filter_score

        # Determine quality level
        if quality_score >= 80:
            quality_level = "Excellent"
        elif quality_score >= 60:
            quality_level = "Good"
        elif quality_score >= 40:
            quality_level = "Fair"
        else:
            quality_level = "Poor"

        return {
            'overall_quality': quality_level,
            'quality_score': float(quality_score),
            'snr_db': float(snr),
            'envelope_snr_db': float(envelope_snr_db),
            'spectral_peaks': spectrum_peaks,
            'spectral_concentration': float(spectral_concentration),
            'filter_bandwidth': float(filter_bw),
            'recommendations': generate_quality_recommendations(quality_score, snr, envelope_snr_db)
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Quality assessment error: {e}", Logger.WARNING)
        return {'overall_quality': 'Unknown', 'quality_score': 0}


def generate_quality_recommendations(quality_score, snr, envelope_snr):
    """Generate recommendations for improving envelope analysis quality."""
    recommendations = []

    if quality_score < 40:
        recommendations.append("Consider using different filter parameters")

    if snr < 15:
        recommendations.append("Signal-to-noise ratio is low - check sensor mounting and cables")

    if envelope_snr < 25:
        recommendations.append("Envelope quality is poor - try different envelope extraction method")

    if quality_score < 60:
        recommendations.append("Increase sampling rate if possible for better frequency resolution")
        recommendations.append("Consider longer measurement time for better statistical reliability")

    if not recommendations:
        recommendations.append("Analysis quality is good - results are reliable")

    return recommendations


def calculate_spectral_kurtosis(signal, sample_rate, window_length=None):
    """
    Calculate spectral kurtosis to identify optimal frequency bands for envelope analysis.

    Spectral kurtosis measures the impulsive content at each frequency bin,
    helping to identify frequency bands most suitable for envelope analysis.
    """
    try:
        if window_length is None:
            window_length = min(2048, len(signal) // 4)

        # Perform short-time FFT
        freqs, times, stft = sc_signal.stft(signal, fs=sample_rate,
                                            nperseg=window_length,
                                            noverlap=window_length // 2)

        # Calculate magnitude spectrum for each time slice
        mag_stft = np.abs(stft)

        # Calculate spectral kurtosis for each frequency bin
        spectral_kurt = np.zeros(len(freqs))

        for i in range(len(freqs)):
            freq_data = mag_stft[i, :]
            if len(freq_data) > 3 and np.std(freq_data) > 0:
                from scipy import stats
                spectral_kurt[i] = stats.kurtosis(freq_data)

        # Find frequency bands with highest kurtosis
        high_kurt_mask = spectral_kurt > np.percentile(spectral_kurt, 90)
        optimal_freqs = freqs[high_kurt_mask]

        return {
            'frequencies': freqs,
            'spectral_kurtosis': spectral_kurt,
            'optimal_frequencies': optimal_freqs,
            'max_kurtosis': float(np.max(spectral_kurt)) if len(spectral_kurt) > 0 else 0,
            'mean_kurtosis': float(np.mean(spectral_kurt)) if len(spectral_kurt) > 0 else 0
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Spectral kurtosis calculation error: {e}", Logger.WARNING)
        return None


def detect_envelope_harmonics(spectrum_results):
    """Detect harmonic relationships in envelope spectrum."""
    try:
        frequencies = spectrum_results['frequencies']
        magnitudes = spectrum_results['magnitude']

        if len(frequencies) < 2:
            return None

        # Find the fundamental frequency (largest peak below 200 Hz)
        low_freq_mask = frequencies < 200
        if not np.any(low_freq_mask):
            return None

        low_freq_mags = magnitudes[low_freq_mask]
        low_freq_freqs = frequencies[low_freq_mask]

        fundamental_idx = np.argmax(low_freq_mags)
        fundamental_freq = low_freq_freqs[fundamental_idx]

        # Look for harmonics
        harmonics = []
        for h in range(2, 11):  # Check up to 10th harmonic
            harmonic_freq = h * fundamental_freq
            if harmonic_freq >= frequencies[-1]:
                break

            # Find closest frequency
            idx = np.argmin(np.abs(frequencies - harmonic_freq))
            freq_error = abs(frequencies[idx] - harmonic_freq)

            # Accept if within 5% of fundamental frequency
            if freq_error <= fundamental_freq * 0.05:
                harmonics.append({
                    'harmonic': h,
                    'frequency': float(frequencies[idx]),
                    'magnitude': float(magnitudes[idx]),
                    'frequency_error': float(freq_error)
                })

        return {
            'fundamental_frequency': float(fundamental_freq),
            'fundamental_magnitude': float(low_freq_mags[fundamental_idx]),
            'harmonics': harmonics,
            'harmonic_count': len(harmonics)
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Harmonic detection error: {e}", Logger.WARNING)
        return None


def perform_multiband_envelope_analysis(signal, time_arr, sample_rate):
    """
    Perform envelope analysis across multiple frequency bands.

    Analyzes the signal in different frequency bands to identify
    the most informative bands for fault detection.
    """
    try:
        # Define frequency bands for analysis
        nyquist = sample_rate / 2
        bands = [
            (500, 2000, "Low"),
            (2000, 5000, "Medium"),
            (5000, 10000, "High"),
            (10000, min(20000, nyquist * 0.9), "Very High")
        ]

        band_results = {}

        for low, high, name in bands:
            if high > nyquist * 0.9:
                continue

            try:
                # Filter signal for this band
                sos = sc_signal.butter(4, [low, high], btype='band',
                                       fs=sample_rate, output='sos')
                filtered = sc_signal.sosfiltfilt(sos, signal)

                # Extract envelope
                envelope = np.abs(sc_signal.hilbert(filtered))

                # Analyze envelope spectrum
                env_spectrum = analyze_envelope_spectrum(envelope, sample_rate)

                # Detect peaks
                peaks = detect_envelope_peaks(env_spectrum['frequencies'],
                                              env_spectrum['magnitude'], sample_rate)

                # Calculate envelope statistics
                stats = calculate_envelope_statistics(envelope, filtered)

                band_results[name] = {
                    'frequency_band': (low, high),
                    'envelope_rms': stats.get('RMS', 0),
                    'envelope_crest': stats.get('Crest Factor', 0),
                    'envelope_kurtosis': stats.get('Kurtosis', 0),
                    'peak_count': len(peaks['frequencies']),
                    'dominant_envelope_freq': float(peaks['frequencies'][0]) if len(peaks['frequencies']) > 0 else 0,
                    'envelope_spectrum': env_spectrum,
                    'envelope_peaks': peaks
                }

            except Exception as band_error:
                Logger.log_message_static(f"Vibration-Envelope: Error in band {name}: {band_error}", Logger.WARNING)
                continue

        # Find best band for envelope analysis
        best_band = None
        best_score = 0

        for band_name, results in band_results.items():
            # Score based on kurtosis and peak count
            score = results['envelope_kurtosis'] * 0.7 + results['peak_count'] * 0.3
            if score > best_score:
                best_score = score
                best_band = band_name

        return {
            'band_results': band_results,
            'best_band': best_band,
            'best_score': float(best_score),
            'band_comparison': compare_envelope_bands(band_results)
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Multiband analysis error: {e}", Logger.WARNING)
        return None


def compare_envelope_bands(band_results):
    """Compare envelope analysis results across frequency bands."""
    try:
        comparison = {
            'band_ranking': [],
            'summary': {}
        }

        # Rank bands by envelope quality metrics
        band_scores = {}
        for band_name, results in band_results.items():
            # Composite score based on multiple factors
            kurtosis = results.get('envelope_kurtosis', 0)
            crest = results.get('envelope_crest', 0)
            peak_count = results.get('peak_count', 0)

            # Weighted score
            score = (kurtosis * 0.4 + crest * 0.3 + peak_count * 0.3)
            band_scores[band_name] = score

        # Sort by score
        sorted_bands = sorted(band_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['band_ranking'] = [
            {'band': band, 'score': float(score)}
            for band, score in sorted_bands
        ]

        # Generate summary
        if sorted_bands:
            best_band = sorted_bands[0][0]
            comparison['summary'] = {
                'recommended_band': best_band,
                'reason': f"Highest envelope quality score ({sorted_bands[0][1]:.2f})",
                'frequency_range': band_results[best_band]['frequency_band']
            }

        return comparison

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Band comparison error: {e}", Logger.WARNING)
        return {}


def calculate_bearing_envelope_analysis(time_arr, values, rpm, bearing_params, dialog=None):
    """
    Specialized envelope analysis for bearing fault detection.

    Optimizes envelope analysis specifically for bearing fault detection
    using known bearing geometry and operating speed.

    Args:
        time_arr (np.ndarray): Time values array.
        values (np.ndarray): Vibration signal values.
        rpm (float): Rotational speed in RPM.
        bearing_params (dict): Bearing parameters:
            - balls: Number of rolling elements
            - pitch_diameter: Pitch diameter (mm)
            - ball_diameter: Ball diameter (mm)
            - contact_angle: Contact angle (degrees, default 0)
        dialog (QWidget, optional): Parent dialog.

    Returns:
        dict: Specialized bearing envelope analysis results.
    """
    Logger.log_message_static("Vibration-Envelope: Starting bearing-specific envelope analysis", Logger.DEBUG)

    try:
        # Calculate theoretical bearing fault frequencies
        from vibration_fft import calculate_fault_frequencies
        fault_freqs = calculate_fault_frequencies(rpm, bearing_params)

        # Perform standard envelope analysis with bearing-optimized filter
        envelope_results = calculate_envelope_analysis(
            time_arr, values, dialog,
            title="Bearing Envelope Analysis",
            filter_type="bearing"
        )

        if envelope_results is None:
            return None

        # Extract envelope spectrum
        env_freqs = envelope_results['Envelope Frequencies']
        env_spectrum = envelope_results['Envelope Spectrum']

        # Search for bearing fault frequencies in envelope spectrum
        bearing_detections = {}

        for fault_name, fault_freq in fault_freqs.items():
            if isinstance(fault_freq, (int, float)) and fault_freq > 0:
                # Find closest frequency in envelope spectrum
                if len(env_freqs) > 0 and fault_freq <= env_freqs[-1]:
                    closest_idx = np.argmin(np.abs(env_freqs - fault_freq))
                    closest_freq = env_freqs[closest_idx]
                    freq_error = abs(closest_freq - fault_freq)

                    # Accept if within 5% tolerance
                    if freq_error <= fault_freq * 0.05:
                        magnitude = env_spectrum[closest_idx]
                        bearing_detections[fault_name] = {
                            'theoretical_freq': float(fault_freq),
                            'detected_freq': float(closest_freq),
                            'magnitude': float(magnitude),
                            'frequency_error': float(freq_error),
                            'error_percent': float(freq_error / fault_freq * 100)
                        }

        # Assess bearing condition based on detections
        bearing_condition = assess_bearing_condition(bearing_detections, fault_freqs)

        # Enhanced results with bearing-specific analysis
        enhanced_results = envelope_results.copy()
        enhanced_results.update({
            'Bearing Fault Frequencies': fault_freqs,
            'Bearing Detections': bearing_detections,
            'Bearing Condition Assessment': bearing_condition,
            'RPM': rpm,
            'Bearing Parameters': bearing_params
        })

        Logger.log_message_static(
            f"Vibration-Envelope: Bearing analysis completed. "
            f"Detections: {len(bearing_detections)}, "
            f"Condition: {bearing_condition.get('overall_condition', 'Unknown')}",
            Logger.DEBUG
        )

        return enhanced_results

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Bearing envelope analysis error: {str(e)}", Logger.ERROR)
        return None


def assess_bearing_condition(detections, fault_frequencies):
    """Assess bearing condition based on envelope spectrum detections."""
    try:
        if not detections:
            return {
                'overall_condition': 'Good',
                'condition_score': 90,
                'detected_faults': [],
                'recommendations': ['Continue normal monitoring'],
                'confidence': 'High'
            }

        detected_faults = []
        condition_score = 100

        # Assess each detected fault
        for fault_name, detection in detections.items():
            magnitude = detection['magnitude']
            error_percent = detection['error_percent']

            # Severity assessment based on magnitude and frequency accuracy
            if magnitude > 0.1:  # High magnitude
                severity = 'High'
                condition_score -= 30
            elif magnitude > 0.05:  # Medium magnitude
                severity = 'Medium'
                condition_score -= 15
            else:  # Low magnitude
                severity = 'Low'
                condition_score -= 5

            # Adjust confidence based on frequency accuracy
            if error_percent < 2:
                confidence = 'High'
            elif error_percent < 5:
                confidence = 'Medium'
            else:
                confidence = 'Low'
                condition_score += 5  # Less penalty for uncertain detections

            detected_faults.append({
                'fault_type': fault_name,
                'severity': severity,
                'confidence': confidence,
                'frequency': detection['detected_freq'],
                'magnitude': magnitude
            })

        # Determine overall condition
        if condition_score >= 85:
            overall_condition = 'Good'
            recommendations = ['Continue normal monitoring']
        elif condition_score >= 70:
            overall_condition = 'Fair'
            recommendations = ['Increase monitoring frequency', 'Plan inspection']
        elif condition_score >= 50:
            overall_condition = 'Poor'
            recommendations = ['Schedule bearing replacement', 'Increase monitoring']
        else:
            overall_condition = 'Critical'
            recommendations = ['Immediate bearing inspection required', 'Consider shutdown']

        # Overall confidence
        high_conf_count = sum(1 for fault in detected_faults if fault['confidence'] == 'High')
        if high_conf_count >= len(detected_faults) * 0.7:
            overall_confidence = 'High'
        elif high_conf_count >= len(detected_faults) * 0.3:
            overall_confidence = 'Medium'
        else:
            overall_confidence = 'Low'

        return {
            'overall_condition': overall_condition,
            'condition_score': max(0, condition_score),
            'detected_faults': detected_faults,
            'recommendations': recommendations,
            'confidence': overall_confidence,
            'total_detections': len(detected_faults)
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Envelope: Bearing condition assessment error: {e}", Logger.WARNING)
        return {
            'overall_condition': 'Unknown',
            'condition_score': 0,
            'detected_faults': [],
            'recommendations': ['Manual assessment required'],
            'confidence': 'Low'
        }


def calculate_bearing_envelope(signal_data, sampling_rate, low_cut=1000, high_cut=10000,
                               envelope_smoothing=100, freq_resolution=1.0):
    """
    Calculate bearing envelope analysis from vibration data.

    This function performs envelope analysis commonly used for bearing fault detection:
    1. Band-pass filters the signal around the resonance frequency range
    2. Calculates the Hilbert transform to extract the envelope
    3. Computes the spectrum of the envelope to identify fault frequencies

    Args:
        signal_data (numpy.ndarray): The time domain vibration signal
        sampling_rate (float): Sampling rate of the signal in Hz
        low_cut (float): Lower frequency of the band-pass filter in Hz
        high_cut (float): Upper frequency of the band-pass filter in Hz
        envelope_smoothing (int): Window size for envelope smoothing
        freq_resolution (float): Desired frequency resolution in Hz for the spectrum

    Returns:
        dict: Dictionary containing envelope data and detected fault frequencies
    """
    import numpy as np
    from scipy import signal
    from scipy.signal import hilbert, find_peaks

    # Normalize the signal
    normalized_signal = signal_data - np.mean(signal_data)

    # Design band-pass filter (typically around bearing resonance frequency)
    nyquist = sampling_rate / 2
    low = low_cut / nyquist
    high = high_cut / nyquist

    # Create and apply bandpass filter
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, normalized_signal)

    # Calculate envelope using Hilbert transform
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)

    # Smooth the envelope if requested
    if envelope_smoothing > 0:
        smoothing_window = np.ones(envelope_smoothing) / envelope_smoothing
        envelope = np.convolve(envelope, smoothing_window, mode='same')

    # Calculate envelope spectrum
    n_fft = int(sampling_rate / freq_resolution)
    envelope_spectrum = np.abs(np.fft.rfft(envelope, n_fft)) / len(envelope)
    frequencies = np.fft.rfftfreq(n_fft, d=1 / sampling_rate)

    # Find peaks in the envelope spectrum
    peaks, properties = find_peaks(envelope_spectrum, height=np.std(envelope_spectrum), distance=5)
    peak_data = {
        'indices': peaks,
        'frequencies': frequencies[peaks],
        'amplitudes': envelope_spectrum[peaks]
    }

    return {
        'envelope': envelope,
        'envelope_spectrum': envelope_spectrum,
        'frequencies': frequencies,
        'peaks': peak_data
    }


def detect_bearing_faults(signal_data, sampling_rate, bearing_specs=None,
                          rpm=None, tolerance=0.02):
    """
    Detect bearing faults from vibration data.

    Args:
        signal_data (numpy.ndarray): The time domain vibration signal
        sampling_rate (float): Sampling rate of the signal in Hz
        bearing_specs (dict, optional): Dictionary with bearing specifications
            including ball_diameter, pitch_diameter, contact_angle, num_balls
        rpm (float, optional): Shaft rotation speed in RPM
        tolerance (float): Tolerance for frequency matching (as percentage)

    Returns:
        dict: Dictionary with detected faults and confidence levels
    """
    import numpy as np

    # Get envelope spectrum
    envelope_data = calculate_bearing_envelope(signal_data, sampling_rate)
    frequencies = envelope_data['frequencies']
    spectrum = envelope_data['envelope_spectrum']
    peaks = envelope_data['peaks']

    results = {
        'envelope_data': envelope_data,
        'detected_faults': [],
        'fault_frequencies': {},
        'confidence': {}
    }

    # If bearing specs and RPM are provided, calculate theoretical fault frequencies
    if bearing_specs and rpm:
        # Calculate fundamental frequencies
        shaft_freq = rpm / 60.0  # Hz

        # Extract bearing parameters
        bd = bearing_specs.get('ball_diameter', 0)
        pd = bearing_specs.get('pitch_diameter', 0)
        angle = bearing_specs.get('contact_angle', 0)
        n = bearing_specs.get('num_balls', 0)

        if bd and pd and n:
            # Calculate contact angle factor
            contact_factor = np.cos(np.radians(angle))

            # Calculate ball pass frequency outer race (BPFO)
            bpfo = (n / 2) * shaft_freq * (1 - bd * contact_factor / pd)

            # Calculate ball pass frequency inner race (BPFI)
            bpfi = (n / 2) * shaft_freq * (1 + bd * contact_factor / pd)

            # Calculate ball spin frequency (BSF)
            bsf = (pd / bd) * shaft_freq * (1 - (bd * contact_factor / pd) ** 2)

            # Calculate fundamental train frequency (FTF)
            ftf = shaft_freq / 2 * (1 - bd * contact_factor / pd)

            # Store calculated frequencies
            fault_freqs = {
                'BPFO': bpfo,
                'BPFI': bpfi,
                'BSF': bsf,
                'FTF': ftf,
                'shaft': shaft_freq
            }
            results['fault_frequencies'] = fault_freqs

            # Look for matches with peaks
            for fault_type, freq in fault_freqs.items():
                harmonics = [freq * i for i in range(1, 4)]  # Check first 3 harmonics

                matches = []
                for harmonic in harmonics:
                    # Find closest peak
                    peak_matches = [
                        (i, f, a) for i, (f, a) in
                        enumerate(zip(peaks['frequencies'], peaks['amplitudes']))
                        if abs(f - harmonic) <= tolerance * harmonic
                    ]

                    if peak_matches:
                        matches.append((harmonic, peak_matches))

                if matches:
                    confidence = len(matches) / len(harmonics) * 100
                    results['detected_faults'].append({
                        'type': fault_type,
                        'matches': matches,
                        'confidence': confidence
                    })
                    results['confidence'][fault_type] = confidence

    return results