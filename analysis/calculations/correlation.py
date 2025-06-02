"""
Signal correlation analysis: autocorrelation and cross-correlation.

This module provides correlation-based analysis functions:
- Autocorrelation analysis for detecting periodicities and self-similarity
- Cross-correlation analysis for finding delays and similarities between signals
- Statistical measures of correlation strength and significance

These analyses are fundamental for signal matching, delay estimation, and
periodicity detection in time series data.
"""

import numpy as np
from scipy.signal import correlate
from PySide6.QtWidgets import QMessageBox

from .common import safe_prepare_signal, safe_sample_rate, validate_analysis_inputs
from utils.logger import Logger


def calculate_autocorrelation_analysis(time_arr, values, dialog=None, title="Autocorrelation Analysis"):
    """
    Compute autocorrelation of a signal and detect key temporal characteristics.

    Autocorrelation measures the correlation of a signal with a delayed copy of itself.
    It's useful for detecting periodic patterns, estimating fundamental periods,
    and analyzing signal memory/persistence.

    Args:
        time_arr (np.ndarray): Time values for lag calculation.
        values (np.ndarray): Signal values to analyze.
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "Autocorrelation Analysis".

    Returns:
        dict or None: Dictionary containing autocorrelation results:
            - Lags (s): Time lag values in seconds
            - Autocorrelation: Normalized autocorrelation values
            - Peak Correlation: Maximum correlation (always 1.0 at zero lag)
            - First Minimum (s): First local minimum in autocorrelation
            - First Zero Crossing (s): First zero crossing point
            - Decorrelation Time (s): Time where correlation drops to 1/e
            - Periodic Components: Detected periodic patterns
            - Correlation Width: Width of main correlation peak

        Returns None if validation fails or user cancels.

    Example:
        >>> t = np.linspace(0, 2, 1000)
        >>> signal = np.sin(2*np.pi*5*t) + 0.5*np.random.randn(1000)  # 5Hz + noise
        >>> result = calculate_autocorrelation_analysis(t, signal)
        >>> print(f"First minimum at: {result['First Minimum (s)']} s")
        First minimum at: 0.1 s  # Should be around 1/(2*5Hz) = 0.1s
    """
    Logger.log_message_static(f"Calculations-Correlation: Starting autocorrelation analysis", Logger.DEBUG)

    # Validate inputs
    is_valid, error_msg, sample_rate = validate_analysis_inputs(time_arr, values, min_length=4,
                                                                require_positive_sample_rate=True)
    if not is_valid:
        Logger.log_message_static(f"Calculations-Correlation: Input validation failed: {error_msg}", Logger.ERROR)
        if dialog:
            QMessageBox.warning(dialog, title, f"Autocorrelation Error:\n{error_msg}")
        return None

    # Prepare signal
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Calculations-Correlation: Signal validation failed", Logger.WARNING)
        return None

    try:
        # Remove DC component and apply windowing to reduce edge effects
        detrended = processed_values - np.mean(processed_values)
        windowed = detrended * np.hanning(len(detrended))

        # Check if signal has sufficient variation
        if np.max(np.abs(windowed)) < 1e-10:
            Logger.log_message_static("Calculations-Correlation: Signal has no variation after detrending",
                                      Logger.WARNING)
            if dialog:
                QMessageBox.warning(dialog, title, "Signal has no variation. Autocorrelation cannot be computed.")
            return None

        # Compute full autocorrelation
        acf_full = correlate(windowed, windowed, mode='full')

        # Normalize by maximum value (should be at zero lag)
        max_val = np.max(np.abs(acf_full))
        if max_val > 0:
            acf_normalized = acf_full / max_val
        else:
            Logger.log_message_static("Calculations-Correlation: Autocorrelation is zero", Logger.ERROR)
            return None

        # Extract positive lags only (including zero lag)
        center = len(acf_normalized) // 2
        acf_positive = acf_normalized[center:]

        # Create lag array in seconds
        lags_samples = np.arange(len(acf_positive))
        lags_seconds = lags_samples / sample_rate

        # Find key features in autocorrelation

        # 1. First minimum (excluding zero lag)
        first_min_idx = None
        if len(acf_positive) > 3:  # Need at least a few points
            for i in range(1, len(acf_positive) - 1):
                if (acf_positive[i] < acf_positive[i - 1] and
                        acf_positive[i] < acf_positive[i + 1]):
                    first_min_idx = i
                    break

        first_min_time = lags_seconds[first_min_idx] if first_min_idx is not None else "Not found"
        first_min_value = acf_positive[first_min_idx] if first_min_idx is not None else 0.0

        # 2. First zero crossing
        zero_cross_indices = np.where(np.diff(np.signbit(acf_positive)))[0]
        first_zero_idx = zero_cross_indices[0] if len(zero_cross_indices) > 0 else None
        first_zero_time = lags_seconds[first_zero_idx] if first_zero_idx is not None else "Not found"

        # 3. Decorrelation time (1/e point)
        decorr_threshold = 1.0 / np.e  # ≈ 0.368
        decorr_indices = np.where(acf_positive <= decorr_threshold)[0]
        decorr_idx = decorr_indices[0] if len(decorr_indices) > 0 else None
        decorr_time = lags_seconds[decorr_idx] if decorr_idx is not None else "Not found"

        # 4. Correlation width (FWHM - Full Width at Half Maximum)
        half_max = 0.5
        half_max_indices = np.where(acf_positive <= half_max)[0]
        correlation_width = lags_seconds[half_max_indices[0]] if len(half_max_indices) > 0 else "Not found"

        # 5. Detect periodic components by finding secondary peaks
        from scipy.signal import find_peaks

        # Find peaks with minimum height and distance constraints
        min_peak_height = 0.1  # 10% of maximum
        min_peak_distance = max(1, int(0.01 * sample_rate))  # At least 10ms apart

        peaks, properties = find_peaks(
            acf_positive[1:],  # Exclude zero lag peak
            height=min_peak_height,
            distance=min_peak_distance
        )

        # Adjust peak indices (add 1 because we excluded zero lag)
        peaks = peaks + 1

        periodic_components = []
        if len(peaks) > 0:
            peak_times = lags_seconds[peaks]
            peak_values = acf_positive[peaks]

            # Sort by correlation strength
            sort_indices = np.argsort(peak_values)[::-1]

            for i, (peak_time, peak_value) in enumerate(zip(peak_times[sort_indices], peak_values[sort_indices])):
                if i < 5:  # Limit to top 5 periodic components
                    period = peak_time * 2  # Full period is twice the time to first peak
                    frequency = 1.0 / period if period > 0 else 0.0
                    periodic_components.append({
                        "Period (s)": float(period),
                        "Frequency (Hz)": float(frequency),
                        "Correlation Strength": float(peak_value),
                        "Peak Time (s)": float(peak_time)
                    })

        # Calculate additional statistics

        # Effective correlation length (where ACF drops below 5%)
        threshold_5pct = 0.05
        effective_length_indices = np.where(acf_positive <= threshold_5pct)[0]
        effective_length = lags_seconds[effective_length_indices[0]] if len(effective_length_indices) > 0 else \
        lags_seconds[-1]

        # Mean correlation (excluding zero lag)
        mean_correlation = np.mean(acf_positive[1:]) if len(acf_positive) > 1 else 0.0

        # Correlation decay rate (exponential fit to envelope)
        try:
            # Fit exponential decay: y = A * exp(-lambda * t)
            # Take log: log(y) = log(A) - lambda * t

            # Use first part of ACF for fitting (up to first zero crossing or 1/4 of signal)
            fit_end_idx = min(
                first_zero_idx if first_zero_idx is not None else len(acf_positive),
                len(acf_positive) // 4,
                int(0.1 * sample_rate)  # Max 0.1 seconds
            )

            if fit_end_idx > 2:
                acf_fit = acf_positive[1:fit_end_idx]  # Exclude zero lag
                lags_fit = lags_seconds[1:fit_end_idx]

                # Only fit positive values
                positive_mask = acf_fit > 0.01  # Avoid log(0)
                if np.sum(positive_mask) > 2:
                    log_acf = np.log(acf_fit[positive_mask])
                    lags_log = lags_fit[positive_mask]

                    # Linear fit to log data
                    decay_coeffs = np.polyfit(lags_log, log_acf, 1)
                    decay_rate = -decay_coeffs[0]  # Negative slope gives decay rate
                else:
                    decay_rate = 0.0
            else:
                decay_rate = 0.0

        except Exception as decay_error:
            Logger.log_message_static(f"Calculations-Correlation: Decay rate estimation failed: {decay_error}",
                                      Logger.DEBUG)
            decay_rate = 0.0

        # Build results dictionary
        results = {
            # Core autocorrelation data
            "Lags (s)": lags_seconds,
            "Autocorrelation": acf_positive,
            "Peak Correlation": 1.0,  # Always 1.0 at zero lag

            # Key temporal features
            "First Minimum (s)": first_min_time,
            "First Minimum Value": float(first_min_value),
            "First Zero Crossing (s)": first_zero_time,
            "Decorrelation Time (s)": decorr_time,
            "Correlation Width (FWHM) (s)": correlation_width,
            "Effective Correlation Length (s)": float(effective_length),

            # Statistical measures
            "Mean Correlation": float(mean_correlation),
            "Decay Rate (1/s)": float(decay_rate),
            "Decay Time Constant (s)": float(1.0 / decay_rate) if decay_rate > 0 else np.inf,

            # Periodic components
            "Number of Periodic Components": len(periodic_components),
            "Periodic Components": periodic_components,

            # Analysis parameters
            "Sample Rate (Hz)": float(sample_rate),
            "Signal Length (s)": float(len(processed_values) / sample_rate),
            "Max Lag Analyzed (s)": float(lags_seconds[-1]),
            "Lag Resolution (s)": float(1.0 / sample_rate)
        }

        Logger.log_message_static(
            f"Calculations-Correlation: Autocorrelation analysis completed. "
            f"First_min={first_min_time}, Zero_cross={first_zero_time}, "
            f"Decorr_time={decorr_time}, Periodic_components={len(periodic_components)}",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Correlation: Error in autocorrelation analysis: {str(e)}",
                                  Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Calculations-Correlation: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None


def calculate_cross_correlation_analysis(time_arr1, values1, time_arr2, values2, dialog=None,
                                         title="Cross-Correlation Analysis"):
    """
    Compute normalized cross-correlation between two signals with comprehensive statistics.

    Cross-correlation measures the similarity between two signals as a function of
    the time delay between them. It's essential for delay estimation, signal matching,
    and finding common patterns between different signals.

    Args:
        time_arr1 (np.ndarray): Time values for first signal.
        values1 (np.ndarray): First signal values.
        time_arr2 (np.ndarray): Time values for second signal.
        values2 (np.ndarray): Second signal values.
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "Cross-Correlation Analysis".

    Returns:
        dict or None: Dictionary containing cross-correlation results:
            - Lags (s): Time lag values in seconds
            - Cross-Correlation: Normalized cross-correlation values
            - Max Correlation: Maximum correlation value
            - Lag at Max Correlation (s): Time delay for maximum correlation
            - Correlation at Zero Lag: Correlation with no delay
            - Correlation Width: Width of correlation peak at various thresholds
            - Delay Estimation: Statistical measures of delay
            - Similarity Metrics: Various similarity measures

        Returns None if validation fails or user cancels.

    Example:
        >>> t1 = np.linspace(0, 1, 1000)
        >>> t2 = np.linspace(0, 1, 1000)
        >>> signal1 = np.sin(2*np.pi*10*t1)
        >>> signal2 = np.sin(2*np.pi*10*(t2 - 0.05))  # Delayed by 0.05s
        >>> result = calculate_cross_correlation_analysis(t1, signal1, t2, signal2)
        >>> print(f"Estimated delay: {result['Lag at Max Correlation (s)']} s")
        Estimated delay: 0.05 s
    """
    Logger.log_message_static(f"Calculations-Correlation: Starting cross-correlation analysis", Logger.DEBUG)

    # Validate both signals
    is_valid1, error_msg1, sample_rate1 = validate_analysis_inputs(time_arr1, values1, min_length=2)
    is_valid2, error_msg2, sample_rate2 = validate_analysis_inputs(time_arr2, values2, min_length=2)

    if not is_valid1:
        Logger.log_message_static(f"Calculations-Correlation: Signal 1 validation failed: {error_msg1}", Logger.ERROR)
        if dialog:
            QMessageBox.warning(dialog, title, f"Signal 1 Error:\n{error_msg1}")
        return None

    if not is_valid2:
        Logger.log_message_static(f"Calculations-Correlation: Signal 2 validation failed: {error_msg2}", Logger.ERROR)
        if dialog:
            QMessageBox.warning(dialog, title, f"Signal 2 Error:\n{error_msg2}")
        return None

    # Check sampling rate compatibility
    if abs(sample_rate1 - sample_rate2) > 1e-3:
        Logger.log_message_static(
            f"Calculations-Correlation: Sampling rate mismatch: {sample_rate1:.3f} vs {sample_rate2:.3f}",
            Logger.WARNING
        )
        if dialog:
            QMessageBox.warning(
                dialog, title,
                f"Sampling Rate Mismatch\n\n"
                f"Signal 1 sample rate: {sample_rate1:.2f} Hz\n"
                f"Signal 2 sample rate: {sample_rate2:.2f} Hz\n\n"
                f"Cross-correlation assumes matching sample rates.\n"
                f"Results may be inaccurate."
            )
        # Use average sample rate
        sample_rate = (sample_rate1 + sample_rate2) / 2
    else:
        sample_rate = sample_rate1

    # Prepare signals
    processed_values1 = safe_prepare_signal(values1, dialog, f"{title} (Signal 1)")
    if processed_values1 is None:
        Logger.log_message_static("Calculations-Correlation: Signal 1 validation failed", Logger.WARNING)
        return None

    processed_values2 = safe_prepare_signal(values2, dialog, f"{title} (Signal 2)")
    if processed_values2 is None:
        Logger.log_message_static("Calculations-Correlation: Signal 2 validation failed", Logger.WARNING)
        return None

    try:
        # Determine common length for cross-correlation
        length = min(len(processed_values1), len(processed_values2))

        if length < 4:
            Logger.log_message_static("Calculations-Correlation: Signals too short for meaningful cross-correlation",
                                      Logger.ERROR)
            if dialog:
                QMessageBox.warning(dialog, title, "Signals are too short for cross-correlation analysis.")
            return None

        # Truncate signals to common length and remove DC
        x1 = processed_values1[:length] - np.mean(processed_values1[:length])
        x2 = processed_values2[:length] - np.mean(processed_values2[:length])

        # Apply windowing to reduce edge effects
        window = np.hanning(length)
        x1_windowed = x1 * window
        x2_windowed = x2 * window

        # Compute cross-correlation
        cross_corr = correlate(x1_windowed, x2_windowed, mode='full')

        # Create lag array
        lags_samples = np.arange(-length + 1, length)
        lags_seconds = lags_samples / sample_rate

        # Normalize cross-correlation
        # Method: Normalize by the geometric mean of the autocorrelations at zero lag
        norm_factor = np.sqrt(np.sum(x1_windowed ** 2) * np.sum(x2_windowed ** 2))

        if norm_factor > 1e-12:
            cross_corr_normalized = cross_corr / norm_factor
        else:
            Logger.log_message_static("Calculations-Correlation: Cannot normalize cross-correlation (zero energy)",
                                      Logger.ERROR)
            if dialog:
                QMessageBox.warning(dialog, title,
                                    "One or both signals have zero energy. Cannot compute cross-correlation.")
            return None

        # Find maximum correlation and its lag
        max_idx = np.argmax(np.abs(cross_corr_normalized))
        max_corr = cross_corr_normalized[max_idx]
        max_lag = lags_seconds[max_idx]

        # Get correlation at zero lag
        zero_lag_idx = len(cross_corr_normalized) // 2
        zero_lag_corr = cross_corr_normalized[zero_lag_idx]

        # Calculate correlation width at different thresholds
        correlation_widths = {}
        thresholds = [0.9, 0.7, 0.5, 0.3]

        for threshold in thresholds:
            threshold_level = abs(max_corr) * threshold
            indices_above_threshold = np.where(np.abs(cross_corr_normalized) >= threshold_level)[0]

            if len(indices_above_threshold) > 0:
                width_samples = indices_above_threshold[-1] - indices_above_threshold[0] + 1
                width_seconds = width_samples / sample_rate
                correlation_widths[f"Width {int(threshold * 100)}%"] = float(width_seconds)
            else:
                correlation_widths[f"Width {int(threshold * 100)}%"] = 0.0

        # Delay estimation statistics
        # Find all peaks above a certain threshold for multiple delay estimates
        from scipy.signal import find_peaks

        peak_threshold = abs(max_corr) * 0.5  # 50% of max correlation
        peaks, properties = find_peaks(
            np.abs(cross_corr_normalized),
            height=peak_threshold,
            distance=max(1, int(0.01 * sample_rate))  # Minimum 10ms between peaks
        )

        # Calculate delay confidence based on peak sharpness
        if len(peaks) > 0:
            peak_lags = lags_seconds[peaks]
            peak_values = cross_corr_normalized[peaks]

            # Sort by absolute correlation value
            sort_indices = np.argsort(np.abs(peak_values))[::-1]
            peak_lags_sorted = peak_lags[sort_indices]
            peak_values_sorted = peak_values[sort_indices]

            # Primary delay is the strongest peak
            primary_delay = peak_lags_sorted[0] if len(peak_lags_sorted) > 0 else max_lag

            # Secondary delays
            secondary_delays = peak_lags_sorted[1:min(4, len(peak_lags_sorted))]  # Up to 3 additional delays
        else:
            primary_delay = max_lag
            secondary_delays = []

        # Similarity metrics
        # Pearson correlation coefficient (should match zero-lag cross-correlation)
        pearson_corr = np.corrcoef(x1, x2)[0, 1] if len(x1) > 1 and len(x2) > 1 else 0.0

        # Coherence estimate (simplified)
        coherence_estimate = abs(max_corr) ** 2

        # Signal-to-noise ratio estimate based on peak-to-background ratio
        background_level = np.median(np.abs(cross_corr_normalized))
        peak_to_background = abs(max_corr) / background_level if background_level > 0 else np.inf

        # Time reversal test (correlation with time-reversed signal)
        x2_reversed = x2[::-1]
        cross_corr_reversed = correlate(x1_windowed, x2_reversed * window, mode='full')
        cross_corr_reversed_norm = cross_corr_reversed / norm_factor if norm_factor > 0 else cross_corr_reversed
        max_corr_reversed = np.max(np.abs(cross_corr_reversed_norm))

        # Asymmetry measure
        left_half = cross_corr_normalized[:zero_lag_idx]
        right_half = cross_corr_normalized[zero_lag_idx + 1:]
        # Pad to same length
        min_len = min(len(left_half), len(right_half))
        if min_len > 0:
            asymmetry = np.mean(right_half[:min_len]) - np.mean(left_half[-min_len:])
        else:
            asymmetry = 0.0

        # Build results dictionary
        results = {
            # Core cross-correlation data
            "Lags (s)": lags_seconds,
            "Cross-Correlation": cross_corr_normalized,

            # Primary results
            "Max Correlation": float(max_corr),
            "Lag at Max Correlation (s)": float(max_lag),
            "Correlation at Zero Lag": float(zero_lag_corr),

            # Delay estimation
            "Primary Delay Estimate (s)": float(primary_delay),
            "Secondary Delay Estimates (s)": [float(d) for d in secondary_delays],
            "Number of Significant Peaks": len(peaks),

            # Correlation characteristics
            **{f"Correlation {k}": v for k, v in correlation_widths.items()},

            # Similarity metrics
            "Pearson Correlation": float(pearson_corr),
            "Coherence Estimate": float(coherence_estimate),
            "Peak-to-Background Ratio": float(peak_to_background),
            "Time Reversal Correlation": float(max_corr_reversed),
            "Asymmetry Measure": float(asymmetry),

            # Signal information
            "Signal 1 Length": len(processed_values1),
            "Signal 2 Length": len(processed_values2),
            "Analysis Length": length,
            "Sample Rate (Hz)": float(sample_rate),
            "Max Lag Analyzed (s)": float(max(np.abs(lags_seconds))),
            "Lag Resolution (s)": float(1.0 / sample_rate)
        }

        # Add confidence assessment
        confidence_score = 0.0
        confidence_factors = []

        # Factor 1: Correlation strength
        if abs(max_corr) > 0.8:
            confidence_score += 0.3
            confidence_factors.append("High correlation strength")
        elif abs(max_corr) > 0.5:
            confidence_score += 0.2
            confidence_factors.append("Moderate correlation strength")
        elif abs(max_corr) > 0.3:
            confidence_score += 0.1
            confidence_factors.append("Low correlation strength")

        # Factor 2: Peak sharpness
        if peak_to_background > 5:
            confidence_score += 0.3
            confidence_factors.append("Sharp correlation peak")
        elif peak_to_background > 2:
            confidence_score += 0.2
            confidence_factors.append("Moderate correlation peak")

        # Factor 3: Consistency with Pearson correlation
        if abs(abs(max_corr) - abs(pearson_corr)) < 0.1:
            confidence_score += 0.2
            confidence_factors.append("Consistent with Pearson correlation")

        # Factor 4: Single clear peak vs multiple peaks
        if len(peaks) == 1:
            confidence_score += 0.2
            confidence_factors.append("Single clear peak")
        elif len(peaks) <= 3:
            confidence_score += 0.1
            confidence_factors.append("Few competing peaks")

        results["Confidence Score"] = min(1.0, confidence_score)
        results["Confidence Factors"] = confidence_factors

        Logger.log_message_static(
            f"Calculations-Correlation: Cross-correlation analysis completed. "
            f"Max_corr={max_corr:.4f}, Lag={max_lag:.6f}s, "
            f"Zero_lag_corr={zero_lag_corr:.4f}, Confidence={confidence_score:.2f}",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Correlation: Error in cross-correlation analysis: {str(e)}",
                                  Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Calculations-Correlation: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None