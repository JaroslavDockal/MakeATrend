"""
Basic signal analysis: statistics and time-domain calculations.

This module provides fundamental signal analysis functions:
- Basic statistical descriptors (mean, std, min, max, etc.)
- Time-domain characteristics (duration, zero crossings, energy, etc.)
- Signal quality metrics (RMS, crest factor, etc.)

These analyses form the foundation for more advanced signal processing techniques.
"""

import numpy as np
from scipy import stats

from .common import safe_prepare_signal, safe_sample_rate, validate_analysis_inputs
from utils.logger import Logger


def calculate_basic_statistics(values, dialog=None, title="Basic Statistics"):
    """
    Compute fundamental statistical descriptors for a signal.

    Calculates comprehensive statistical metrics that describe the signal's
    central tendency, dispersion, and distribution shape.

    Args:
        values (np.ndarray): Signal values to analyze.
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "Basic Statistics".

    Returns:
        dict or None: Dictionary containing statistical metrics:
            - Mean: Average value of the signal
            - Median: Middle value when sorted
            - Mode: Most frequently occurring value (if applicable)
            - Standard Deviation: Measure of signal variability
            - Variance: Square of standard deviation
            - Min/Max: Extreme values
            - Range: Difference between max and min
            - RMS: Root mean square (energy-related measure)
            - Skewness: Asymmetry of distribution
            - Kurtosis: Measure of distribution "tailedness"
            - Percentiles: 25th, 75th percentiles and IQR

        Returns None if validation fails or user cancels.

    Example:
        >>> signal = np.random.normal(0, 1, 1000)  # Normal distribution
        >>> stats = calculate_basic_statistics(signal)
        >>> print(f"Mean: {stats['Mean']:.3f}, Std: {stats['Standard Deviation']:.3f}")
        Mean: 0.023, Std: 0.987
    """
    Logger.log_message_static(f"Calculations-Basic: Starting basic statistics calculation", Logger.DEBUG)

    # Validate and prepare signal
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Calculations-Basic: Signal validation failed", Logger.WARNING)
        return None

    try:
        # Basic descriptive statistics
        mean_val = np.mean(processed_values)
        median_val = np.median(processed_values)
        std_val = np.std(processed_values, ddof=1)  # Sample standard deviation
        var_val = np.var(processed_values, ddof=1)  # Sample variance
        min_val = np.min(processed_values)
        max_val = np.max(processed_values)
        range_val = max_val - min_val

        # Energy-related measures
        rms_val = np.sqrt(np.mean(processed_values ** 2))

        # Percentiles and quartiles
        q25 = np.percentile(processed_values, 25)
        q75 = np.percentile(processed_values, 75)
        iqr = q75 - q25

        # Shape statistics (only if std > 0 to avoid division by zero)
        if std_val > 1e-10:
            skewness_val = stats.skew(processed_values)
            kurtosis_val = stats.kurtosis(processed_values, fisher=True)  # Excess kurtosis
        else:
            skewness_val = 0.0
            kurtosis_val = 0.0
            Logger.log_message_static("Calculations-Basic: Signal has zero variance, skewness and kurtosis set to 0",
                                      Logger.INFO)

        # Peak-to-RMS ratio (Crest Factor)
        peak_val = np.max(np.abs(processed_values))
        crest_factor = peak_val / rms_val if rms_val > 1e-10 else np.inf

        # Form factor (RMS to mean of absolute values ratio)
        mean_abs = np.mean(np.abs(processed_values))
        form_factor = rms_val / mean_abs if mean_abs > 1e-10 else np.inf

        results = {
            # Central tendency
            "Mean": float(mean_val),
            "Median": float(median_val),

            # Dispersion
            "Standard Deviation": float(std_val),
            "Variance": float(var_val),
            "Range": float(range_val),
            "Interquartile Range (IQR)": float(iqr),

            # Extremes
            "Min": float(min_val),
            "Max": float(max_val),
            "Peak (|max|)": float(peak_val),

            # Quartiles
            "25th Percentile (Q1)": float(q25),
            "75th Percentile (Q3)": float(q75),

            # Energy measures
            "RMS": float(rms_val),
            "Mean Absolute Value": float(mean_abs),

            # Quality factors
            "Crest Factor": float(crest_factor),
            "Form Factor": float(form_factor),

            # Shape statistics
            "Skewness": float(skewness_val),
            "Kurtosis (Excess)": float(kurtosis_val),

            # Sample information
            "Sample Count": len(processed_values),
            "Non-zero Samples": int(np.count_nonzero(processed_values))
        }

        Logger.log_message_static(
            f"Calculations-Basic: Statistics completed for {len(processed_values)} samples. "
            f"Mean={mean_val:.6f}, Std={std_val:.6f}, Range=[{min_val:.6f}, {max_val:.6f}]",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Basic: Error computing statistics: {str(e)}", Logger.ERROR)
        return None


def calculate_time_domain_analysis(time_arr, values, dialog=None, title="Time Domain Analysis"):
    """
    Compute comprehensive time-domain characteristics of a signal.

    Analyzes temporal properties including timing, trends, transitions, and
    energy distribution over time.

    Args:
        time_arr (np.ndarray): Time values corresponding to signal samples.
        values (np.ndarray): Signal amplitude values.
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "Time Domain Analysis".

    Returns:
        dict or None: Dictionary containing time-domain metrics:
            - Duration: Total time span of the signal
            - Sample Rate: Calculated sampling frequency
            - Zero Crossings: Number of times signal crosses zero
            - Trend Analysis: Linear trend coefficients and statistics
            - Energy/Power: Signal energy and average power
            - Derivative Statistics: Rate of change characteristics
            - Temporal Statistics: Time-weighted averages

        Returns None if validation fails or user cancels.

    Example:
        >>> t = np.linspace(0, 1, 1000)
        >>> signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
        >>> analysis = calculate_time_domain_analysis(t, signal)
        >>> print(f"Duration: {analysis['Duration (s)']:.3f}s")
        >>> print(f"Zero crossings: {analysis['Zero Crossings']}")
        Duration: 1.000s
        Zero crossings: 10
    """
    Logger.log_message_static(f"Calculations-Basic: Starting time-domain analysis", Logger.DEBUG)

    # Validate inputs
    is_valid, error_msg, sample_rate = validate_analysis_inputs(time_arr, values, min_length=2)
    if not is_valid:
        Logger.log_message_static(f"Calculations-Basic: Input validation failed: {error_msg}", Logger.ERROR)
        if dialog:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(dialog, title, f"Input validation failed:\n{error_msg}")
        return None

    # Prepare signal
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Calculations-Basic: Signal validation failed", Logger.WARNING)
        return None

    try:
        results = {}

        # Basic timing information
        duration = float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 0.0
        results["Duration (s)"] = duration
        results["Sample Count"] = len(processed_values)
        results["Sample Rate (Hz)"] = float(sample_rate)
        results["Start Time (s)"] = float(time_arr[0]) if len(time_arr) > 0 else 0.0
        results["End Time (s)"] = float(time_arr[-1]) if len(time_arr) > 0 else 0.0

        # Zero crossings analysis
        zero_crossings = ((processed_values[:-1] * processed_values[1:]) < 0).sum()
        results["Zero Crossings"] = int(zero_crossings)

        # Zero crossing rate (per second)
        if duration > 0:
            results["Zero Crossing Rate (Hz)"] = float(zero_crossings / duration)
        else:
            results["Zero Crossing Rate (Hz)"] = 0.0

        # Basic statistics (redundant with basic_statistics but useful here)
        mean_val = np.mean(processed_values)
        results["Mean"] = float(mean_val)
        results["Median"] = float(np.median(processed_values))
        std_val = np.std(processed_values)
        results["Standard Deviation"] = float(std_val)
        results["Variance"] = float(np.var(processed_values))

        # Shape statistics
        if std_val > 1e-10:
            results["Skewness"] = float(stats.skew(processed_values))
            results["Kurtosis"] = float(stats.kurtosis(processed_values))
        else:
            results["Skewness"] = 0.0
            results["Kurtosis"] = 0.0

        # Amplitude characteristics
        peak_amplitude = np.max(np.abs(processed_values))
        results["Peak Amplitude"] = float(peak_amplitude)
        results["Min"] = float(np.min(processed_values))
        results["Max"] = float(np.max(processed_values))

        # Energy and power analysis
        energy = np.sum(processed_values ** 2)
        results["Energy"] = float(energy)

        power = energy / len(processed_values) if len(processed_values) > 0 else 0.0
        results["Power"] = float(power)

        rms = np.sqrt(power)
        results["RMS"] = float(rms)

        # Crest factor
        results["Crest Factor"] = float(peak_amplitude / rms) if rms > 1e-10 else np.inf

        # Trend analysis using linear regression
        if len(time_arr) > 2:
            try:
                # Fit linear trend: y = a*t + b
                trend_coeffs = np.polyfit(time_arr, processed_values, 1)
                trend_slope = trend_coeffs[0]
                trend_intercept = trend_coeffs[1]

                # Calculate trend line
                trend_line = np.polyval(trend_coeffs, time_arr)

                # Detrended signal
                detrended = processed_values - trend_line

                # Trend statistics
                results["Trend Coefficients"] = trend_coeffs.tolist()
                results["Trend Slope"] = float(trend_slope)
                results["Trend Intercept"] = float(trend_intercept)
                results["Detrended Variance"] = float(np.var(detrended))

                # R-squared for trend fit quality
                ss_res = np.sum((processed_values - trend_line) ** 2)
                ss_tot = np.sum((processed_values - mean_val) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
                results["Trend R-squared"] = float(r_squared)

            except Exception as trend_error:
                Logger.log_message_static(f"Calculations-Basic: Trend analysis failed: {trend_error}", Logger.WARNING)
                results["Trend Analysis"] = "Failed"

        # Derivative analysis (rate of change)
        if len(time_arr) > 1 and sample_rate > 0:
            try:
                # Calculate first derivative (rate of change)
                dt = 1.0 / sample_rate
                derivative = np.gradient(processed_values, dt)

                # Derivative statistics
                results["First Derivative Stats"] = {
                    "Mean Rate": float(np.mean(derivative)),
                    "RMS Rate": float(np.sqrt(np.mean(derivative ** 2))),
                    "Max Rate": float(np.max(np.abs(derivative))),
                    "Rate Variance": float(np.var(derivative))
                }

                # Add flattened versions for main results
                results["Mean Rate"] = float(np.mean(derivative))
                results["Max Rate"] = float(np.max(np.abs(derivative)))

            except Exception as deriv_error:
                Logger.log_message_static(f"Calculations-Basic: Derivative analysis failed: {deriv_error}",
                                          Logger.WARNING)
                results["Derivative Analysis"] = "Failed"

        # Activity analysis (segments above/below threshold)
        threshold = rms  # Use RMS as activity threshold
        above_threshold = np.sum(np.abs(processed_values) > threshold)
        results["Samples Above RMS Threshold"] = int(above_threshold)
        results["Activity Ratio"] = float(above_threshold / len(processed_values))

        Logger.log_message_static(
            f"Calculations-Basic: Time-domain analysis completed. "
            f"Duration={duration:.3f}s, Sample_Rate={sample_rate:.1f}Hz, "
            f"Zero_Crossings={zero_crossings}, Energy={energy:.6e}",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Basic: Error in time-domain analysis: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Calculations-Basic: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None


def calculate_signal_quality_metrics(values, dialog=None, title="Signal Quality"):
    """
    Calculate signal quality and integrity metrics.

    Useful for assessing data acquisition quality and identifying potential issues.

    Args:
        values (np.ndarray): Signal values to analyze.
        dialog (QWidget, optional): Parent dialog for user interaction.
        title (str, optional): Title for user dialogs.

    Returns:
        dict or None: Dictionary containing quality metrics:
            - SNR estimates
            - Dynamic range
            - Clipping detection
            - Quantization noise estimates
            - Outlier detection
    """
    Logger.log_message_static(f"Calculations-Basic: Starting signal quality assessment", Logger.DEBUG)

    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        return None

    try:
        results = {}

        # Dynamic range
        min_val = np.min(processed_values)
        max_val = np.max(processed_values)
        dynamic_range = max_val - min_val
        results["Dynamic Range"] = float(dynamic_range)

        # Clipping detection (values at exactly min or max for multiple consecutive samples)
        min_clipping = np.sum(processed_values == min_val)
        max_clipping = np.sum(processed_values == max_val)
        results["Potential Min Clipping"] = int(min_clipping)
        results["Potential Max Clipping"] = int(max_clipping)

        # Simple SNR estimate using signal power vs. high-frequency noise
        # (This is a rough estimate, not true SNR)
        signal_power = np.var(processed_values)
        # Estimate noise as high-frequency content (difference between adjacent samples)
        noise_estimate = np.var(np.diff(processed_values)) / 2  # Divide by 2 for differencing gain
        snr_estimate = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 1e-12 else np.inf
        results["SNR Estimate (dB)"] = float(snr_estimate)

        # Outlier detection using IQR method
        q25 = np.percentile(processed_values, 25)
        q75 = np.percentile(processed_values, 75)
        iqr = q75 - q25
        outlier_threshold = 1.5 * iqr
        outliers = np.sum((processed_values < (q25 - outlier_threshold)) |
                          (processed_values > (q75 + outlier_threshold)))
        results["Outliers (IQR method)"] = int(outliers)
        results["Outlier Percentage"] = float(outliers / len(processed_values) * 100)

        # Quantization assessment
        unique_values = len(np.unique(processed_values))
        results["Unique Values"] = unique_values
        results["Effective Bits"] = float(np.log2(unique_values)) if unique_values > 1 else 0.0

        Logger.log_message_static(f"Calculations-Basic: Signal quality assessment completed", Logger.DEBUG)
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Basic: Error in signal quality assessment: {str(e)}", Logger.ERROR)
        return None