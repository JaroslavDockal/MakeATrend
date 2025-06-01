"""
Vibration metrics calculation and severity assessment.

This module provides specialized functions for calculating vibration-specific
time-domain metrics and assessing machine condition based on industry standards.

Key metrics include:
- RMS (Root Mean Square) - Energy content indicator
- Peak value - Maximum amplitude
- Crest Factor - Ratio of peak to RMS (impulsiveness indicator)
- Kurtosis - Statistical measure of signal peakedness
- Skewness - Asymmetry measure
- Form Factor - Shape indicator
- Impulse Factor - Transient content indicator

These metrics are essential for condition monitoring and fault detection
in rotating machinery.
"""

import numpy as np
from scipy import stats
from analysis.calculations.common import safe_prepare_signal, safe_sample_rate
from utils.logger import Logger


def calculate_vibration_metrics(time_arr, values, dialog=None, title="Vibration Metrics"):
    """
    Calculate comprehensive vibration metrics for condition monitoring.

    Computes industry-standard vibration metrics used for machinery health
    assessment and fault detection. These metrics are particularly useful
    for detecting bearing faults, unbalance, misalignment, and other
    mechanical issues.

    Args:
        time_arr (np.ndarray): Time values corresponding to signal samples.
        values (np.ndarray): Vibration signal values (typically acceleration, velocity, or displacement).
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "Vibration Metrics".

    Returns:
        dict or None: Dictionary containing vibration metrics:
            - RMS: Root mean square value (energy indicator)
            - Peak: Maximum absolute amplitude
            - Peak-to-Peak: Full amplitude range
            - Crest Factor: Peak/RMS ratio (impulsiveness indicator)
            - Kurtosis: Statistical peakedness measure
            - Skewness: Asymmetry measure
            - Form Factor: RMS/Mean ratio
            - Impulse Factor: Peak/Mean ratio
            - Clearance Factor: Peak/RMS² ratio
            - Shape Factor: RMS/absolute mean ratio
            - Mean Absolute Value: Average absolute amplitude
            - Standard Deviation: Signal variability
            - Variance: Square of standard deviation
            - Energy: Total signal energy
            - Power: Average power

        Returns None if validation fails or user cancels.

    Example:
        >>> t = np.linspace(0, 1, 10000)
        >>> # Simulate bearing fault with impulses
        >>> vib_signal = np.random.normal(0, 1, 10000)
        >>> vib_signal[::1000] += 10  # Add periodic impulses
        >>> metrics = calculate_vibration_metrics(t, vib_signal)
        >>> print(f"Crest Factor: {metrics['Crest Factor']:.2f}")
        >>> print(f"Kurtosis: {metrics['Kurtosis']:.2f}")
        Crest Factor: 5.23  # High value indicates impulsive content
        Kurtosis: 8.45     # High value indicates presence of transients
    """
    Logger.log_message_static(f"Vibration-Metrics: Starting vibration metrics calculation", Logger.DEBUG)

    # Validate and prepare signal
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Vibration-Metrics: Signal validation failed", Logger.WARNING)
        return None

    # Get sampling rate for additional calculations
    sample_rate = safe_sample_rate(time_arr)

    try:
        # Ensure we have valid data
        if len(processed_values) == 0:
            Logger.log_message_static("Vibration-Metrics: Empty signal data", Logger.ERROR)
            return None

        # Basic amplitude metrics
        rms = np.sqrt(np.mean(np.square(processed_values)))
        peak = np.max(np.abs(processed_values))
        peak_to_peak = np.max(processed_values) - np.min(processed_values)
        mean_abs = np.mean(np.abs(processed_values))
        mean_val = np.mean(processed_values)

        # Statistical metrics
        std_dev = np.std(processed_values, ddof=1)  # Sample standard deviation
        variance = np.var(processed_values, ddof=1)  # Sample variance

        # Calculate higher-order statistics with error handling
        if std_dev > 1e-10:  # Avoid division by zero
            skewness = stats.skew(processed_values)
            kurtosis = stats.kurtosis(processed_values, fisher=True)  # Excess kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0
            Logger.log_message_static("Vibration-Metrics: Signal has zero variance, skewness and kurtosis set to 0",
                                      Logger.INFO)

        # Vibration-specific shape factors
        # Crest Factor (Peak/RMS) - indicates impulsiveness
        crest_factor = peak / rms if rms > 1e-10 else np.inf

        # Form Factor (RMS/Mean of absolute values)
        form_factor = rms / mean_abs if mean_abs > 1e-10 else np.inf

        # Impulse Factor (Peak/Mean of absolute values)
        impulse_factor = peak / mean_abs if mean_abs > 1e-10 else np.inf

        # Clearance Factor (Peak/Square of RMS)
        clearance_factor = peak / (rms ** 2) if rms > 1e-10 else np.inf

        # Shape Factor (alternative definition: RMS/Mean absolute)
        shape_factor = rms / mean_abs if mean_abs > 1e-10 else np.inf

        # Energy and power calculations
        energy = np.sum(processed_values ** 2)
        power = energy / len(processed_values) if len(processed_values) > 0 else 0.0

        # Frequency domain characteristics (basic)
        if sample_rate > 0:
            # Estimate dominant frequency using zero crossings
            zero_crossings = np.sum(np.diff(np.signbit(processed_values)))
            estimated_freq = zero_crossings / (2 * (time_arr[-1] - time_arr[0])) if len(time_arr) > 1 else 0.0
        else:
            estimated_freq = 0.0

        # Advanced vibration indicators

        # Peak Hold Ratio (useful for detecting intermittent faults)
        sorted_abs_values = np.sort(np.abs(processed_values))[::-1]
        top_1_percent = int(len(sorted_abs_values) * 0.01)
        if top_1_percent > 0:
            peak_hold_ratio = np.mean(sorted_abs_values[:top_1_percent]) / rms if rms > 0 else 0
        else:
            peak_hold_ratio = peak / rms if rms > 0 else 0

        # Signal-to-Noise Ratio estimation
        # Estimate noise as high-frequency content
        if len(processed_values) > 2:
            diff_signal = np.diff(processed_values)
            noise_estimate = np.std(diff_signal)
            snr_estimate = 20 * np.log10(rms / noise_estimate) if noise_estimate > 1e-12 else np.inf
        else:
            snr_estimate = np.inf

        # Regularity indicators
        # Coefficient of Variation
        cv = std_dev / abs(mean_val) if abs(mean_val) > 1e-10 else np.inf

        # Interquartile Range
        q75, q25 = np.percentile(processed_values, [75, 25])
        iqr = q75 - q25

        # Build comprehensive results dictionary
        results = {
            # Core vibration metrics
            "RMS": float(rms),
            "Peak": float(peak),
            "Peak-to-Peak": float(peak_to_peak),
            "Mean Absolute Value": float(mean_abs),
            "Mean": float(mean_val),

            # Statistical measures
            "Standard Deviation": float(std_dev),
            "Variance": float(variance),
            "Skewness": float(skewness),
            "Kurtosis": float(kurtosis),

            # Shape factors (dimensionless indicators)
            "Crest Factor": float(crest_factor),
            "Form Factor": float(form_factor),
            "Impulse Factor": float(impulse_factor),
            "Clearance Factor": float(clearance_factor),
            "Shape Factor": float(shape_factor),

            # Energy measures
            "Energy": float(energy),
            "Power": float(power),

            # Advanced indicators
            "Peak Hold Ratio": float(peak_hold_ratio),
            "Signal-to-Noise Ratio (dB)": float(snr_estimate),
            "Coefficient of Variation": float(cv),
            "Interquartile Range": float(iqr),

            # Frequency estimate
            "Estimated Frequency (Hz)": float(estimated_freq),

            # Data quality indicators
            "Sample Count": len(processed_values),
            "Sample Rate (Hz)": float(sample_rate),
            "Duration (s)": float(len(processed_values) / sample_rate) if sample_rate > 0 else 0.0,
            "Non-zero Samples": int(np.count_nonzero(processed_values)),
            "Zero Percentage": float(
                (len(processed_values) - np.count_nonzero(processed_values)) / len(processed_values) * 100)
        }

        # Add interpretation flags for common vibration conditions
        interpretation = interpret_vibration_metrics(results)
        results["Interpretation"] = interpretation

        Logger.log_message_static(
            f"Vibration-Metrics: Metrics calculated successfully. "
            f"RMS={rms:.6f}, Peak={peak:.6f}, Crest_Factor={crest_factor:.3f}, "
            f"Kurtosis={kurtosis:.3f}, SNR={snr_estimate:.1f}dB",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Vibration-Metrics: Error calculating vibration metrics: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Vibration-Metrics: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None


def calculate_vibration_severity(rms_value, units="mm/s", machine_class="II", operating_speed=None):
    """
    Assess vibration severity according to ISO 10816 standard.

    Evaluates the severity of vibration based on RMS velocity measurements
    and machine classification according to international standards.

    Args:
        rms_value (float): RMS vibration value.
        units (str, optional): Units of measurement. Options:
            - "mm/s": Millimeters per second (velocity) - most common
            - "m/s": Meters per second (velocity)
            - "mm": Millimeters (displacement)
            - "g": Acceleration in g units
            Defaults to "mm/s".
        machine_class (str, optional): Machine class according to ISO 10816. Options:
            - "I": Small machines (< 15 kW)
            - "II": Medium machines (15-75 kW)
            - "III": Large machines on rigid foundations (75-300 kW)
            - "IV": Large machines on flexible foundations (> 300 kW)
            Defaults to "II".
        operating_speed (float, optional): Operating speed in RPM for speed-dependent assessment.

    Returns:
        dict: Dictionary containing severity assessment:
            - Severity Level: A, B, C, or D
            - Severity Description: Text description
            - Recommendation: Action recommendation
            - Threshold Values: Limits for each zone
            - Assessment Standard: Standard used for evaluation
            - Machine Class: Machine classification used

    Example:
        >>> severity = calculate_vibration_severity(2.5, "mm/s", "II")
        >>> print(f"Severity: {severity['Severity Level']} - {severity['Severity Description']}")
        Severity: B - Acceptable for long-term operation
    """
    Logger.log_message_static(
        f"Vibration-Metrics: Assessing vibration severity: {rms_value} {units}, Class {machine_class}", Logger.DEBUG)

    try:
        # Convert to standard units (mm/s) for ISO 10816
        rms_mm_s = convert_to_velocity_mm_s(rms_value, units, operating_speed)

        # Get threshold values for machine class
        thresholds = get_iso10816_thresholds(machine_class)

        # Determine severity level
        if rms_mm_s <= thresholds['A']:
            level = 'A'
            description = 'Good - Newly commissioned machines'
            recommendation = 'Normal operation - Continue monitoring'
            color = 'green'
        elif rms_mm_s <= thresholds['B']:
            level = 'B'
            description = 'Satisfactory - Acceptable for long-term operation'
            recommendation = 'Normal operation - Regular monitoring'
            color = 'yellow'
        elif rms_mm_s <= thresholds['C']:
            level = 'C'
            description = 'Unsatisfactory - Limited operation time acceptable'
            recommendation = 'Increased monitoring - Plan maintenance'
            color = 'orange'
        else:
            level = 'D'
            description = 'Unacceptable - Immediate action required'
            recommendation = 'Stop operation - Immediate maintenance required'
            color = 'red'

        # Additional speed-dependent assessment if available
        speed_assessment = None
        if operating_speed and operating_speed > 0:
            speed_assessment = assess_speed_dependent_limits(rms_mm_s, operating_speed, machine_class)

        results = {
            # Core assessment
            "RMS Value (mm/s)": float(rms_mm_s),
            "Original Value": float(rms_value),
            "Original Units": units,
            "Severity Level": level,
            "Severity Description": description,
            "Recommendation": recommendation,
            "Severity Color": color,

            # Standard information
            "Assessment Standard": "ISO 10816",
            "Machine Class": machine_class,
            "Threshold Values (mm/s)": thresholds,

            # Speed-dependent assessment
            "Speed Assessment": speed_assessment,
            "Operating Speed (RPM)": operating_speed,

            # Additional context
            "Assessment Date": "Current",
            "Confidence": "High" if units == "mm/s" else "Medium"  # Higher confidence for direct velocity measurements
        }

        Logger.log_message_static(
            f"Vibration-Metrics: Severity assessment completed. "
            f"Level={level}, RMS={rms_mm_s:.2f}mm/s, Class={machine_class}",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Vibration-Metrics: Error in severity assessment: {str(e)}", Logger.ERROR)
        return {
            "Error": str(e),
            "Severity Level": "Unknown",
            "Severity Description": "Assessment failed",
            "Recommendation": "Manual evaluation required"
        }


def assess_machine_condition(vibration_metrics, machine_info=None):
    """
    Provide comprehensive machine condition assessment based on multiple vibration metrics.

    Combines multiple vibration indicators to provide an overall condition assessment
    and identify potential fault patterns.

    Args:
        vibration_metrics (dict): Results from calculate_vibration_metrics()
        machine_info (dict, optional): Machine information including:
            - machine_type: Type of machine (pump, motor, fan, etc.)
            - operating_speed: Operating speed in RPM
            - power_rating: Power rating in kW
            - bearing_type: Type of bearings used
            - foundation_type: Foundation type (rigid/flexible)

    Returns:
        dict: Comprehensive condition assessment including:
            - Overall Condition: Good, Fair, Poor, Critical
            - Condition Score: Numerical score (0-100)
            - Primary Indicators: Key metrics driving the assessment
            - Fault Indicators: Suspected fault types
            - Recommendations: Specific maintenance recommendations
            - Trending Required: Whether trending analysis is needed

    Example:
        >>> metrics = calculate_vibration_metrics(time_arr, vib_signal)
        >>> machine_info = {'machine_type': 'motor', 'operating_speed': 1800}
        >>> condition = assess_machine_condition(metrics, machine_info)
        >>> print(f"Condition: {condition['Overall Condition']}")
    """
    Logger.log_message_static("Vibration-Metrics: Starting machine condition assessment", Logger.DEBUG)

    try:
        # Extract key metrics
        rms = vibration_metrics.get('RMS', 0)
        crest_factor = vibration_metrics.get('Crest Factor', 0)
        kurtosis = vibration_metrics.get('Kurtosis', 0)
        skewness = vibration_metrics.get('Skewness', 0)
        peak = vibration_metrics.get('Peak', 0)

        # Initialize condition scoring
        condition_score = 100  # Start with perfect score
        fault_indicators = []
        primary_indicators = []
        recommendations = []

        # Assess RMS level (primary indicator)
        if rms > 0:
            # Rough severity assessment without machine class
            if rms > 10:  # Very high
                condition_score -= 40
                primary_indicators.append(f"Very high RMS ({rms:.2f})")
                recommendations.append("Immediate inspection required")
            elif rms > 4.5:  # High
                condition_score -= 25
                primary_indicators.append(f"High RMS ({rms:.2f})")
                recommendations.append("Increased monitoring recommended")
            elif rms > 1.8:  # Moderate
                condition_score -= 10
                primary_indicators.append(f"Moderate RMS ({rms:.2f})")

        # Assess Crest Factor (impulsiveness)
        if crest_factor > 6:  # Very high - indicates strong impulsive content
            condition_score -= 20
            fault_indicators.append("High impulsive content (possible bearing fault)")
            recommendations.append("Envelope analysis recommended")
        elif crest_factor > 4:  # Moderately high
            condition_score -= 10
            fault_indicators.append("Moderate impulsive content")
        elif crest_factor < 1.5:  # Very low - indicates clipping or saturation
            condition_score -= 15
            fault_indicators.append("Abnormally low crest factor (possible sensor issue)")

        # Assess Kurtosis (statistical indicator of transients)
        if abs(kurtosis) > 5:  # High kurtosis
            condition_score -= 15
            fault_indicators.append("High kurtosis (transient activity)")
            recommendations.append("Time-frequency analysis recommended")
        elif abs(kurtosis) > 3:
            condition_score -= 5
            fault_indicators.append("Moderate kurtosis")

        # Assess Skewness (asymmetry)
        if abs(skewness) > 2:
            condition_score -= 10
            fault_indicators.append("High skewness (asymmetric vibration)")

        # Combined fault pattern recognition
        # Bearing fault pattern: High crest factor + high kurtosis
        if crest_factor > 4 and kurtosis > 3:
            fault_indicators.append("Possible bearing fault (high CF + kurtosis)")
            recommendations.append("Bearing inspection recommended")

        # Unbalance pattern: High RMS + low crest factor
        if rms > 2 and crest_factor < 3:
            fault_indicators.append("Possible unbalance (high RMS, low CF)")
            recommendations.append("Balance check recommended")

        # Looseness pattern: Multiple harmonics (high kurtosis + moderate CF)
        if kurtosis > 2 and 3 < crest_factor < 5:
            fault_indicators.append("Possible looseness (multiple harmonics)")
            recommendations.append("Mechanical integrity check")

        # Machine-specific assessments
        if machine_info:
            machine_specific = assess_machine_specific_condition(vibration_metrics, machine_info)
            if machine_specific:
                condition_score -= machine_specific.get('penalty', 0)
                fault_indicators.extend(machine_specific.get('indicators', []))
                recommendations.extend(machine_specific.get('recommendations', []))

        # Determine overall condition
        condition_score = max(0, min(100, condition_score))  # Clamp to 0-100

        if condition_score >= 85:
            overall_condition = "Good"
            condition_color = "green"
        elif condition_score >= 70:
            overall_condition = "Fair"
            condition_color = "yellow"
        elif condition_score >= 50:
            overall_condition = "Poor"
            condition_color = "orange"
        else:
            overall_condition = "Critical"
            condition_color = "red"

        # Determine if trending is required
        trending_required = (
                condition_score < 85 or
                len(fault_indicators) > 0 or
                rms > 1.8
        )

        # Generate summary
        summary = generate_condition_summary(overall_condition, condition_score, fault_indicators)

        results = {
            # Overall assessment
            "Overall Condition": overall_condition,
            "Condition Score": int(condition_score),
            "Condition Color": condition_color,
            "Summary": summary,

            # Detailed analysis
            "Primary Indicators": primary_indicators,
            "Fault Indicators": fault_indicators,
            "Recommendations": list(set(recommendations)),  # Remove duplicates

            # Analysis metadata
            "Trending Required": trending_required,
            "Assessment Confidence": assess_confidence_level(vibration_metrics),
            "Key Metrics Used": ["RMS", "Crest Factor", "Kurtosis", "Skewness"],

            # Machine context
            "Machine Information": machine_info if machine_info else "Not provided"
        }

        Logger.log_message_static(
            f"Vibration-Metrics: Condition assessment completed. "
            f"Condition={overall_condition}, Score={condition_score}, "
            f"Fault_indicators={len(fault_indicators)}",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Vibration-Metrics: Error in condition assessment: {str(e)}", Logger.ERROR)
        return {
            "Overall Condition": "Unknown",
            "Condition Score": 0,
            "Error": str(e),
            "Recommendations": ["Manual assessment required due to analysis error"]
        }


# Helper functions for vibration metrics

def interpret_vibration_metrics(metrics):
    """Interpret vibration metrics and provide diagnostic insights."""
    interpretation = []

    # Crest Factor interpretation
    cf = metrics.get('Crest Factor', 0)
    if cf > 6:
        interpretation.append("Very high crest factor - strong impulsive content, possible bearing fault")
    elif cf > 4:
        interpretation.append("High crest factor - moderate impulsive content")
    elif cf < 1.5:
        interpretation.append("Low crest factor - possible signal clipping or measurement issue")
    elif 2 <= cf <= 4:
        interpretation.append("Normal crest factor - typical for healthy machinery")

    # Kurtosis interpretation
    kurtosis = metrics.get('Kurtosis', 0)
    if kurtosis > 5:
        interpretation.append("High kurtosis - significant transient activity, investigate for faults")
    elif kurtosis > 3:
        interpretation.append("Moderate kurtosis - some transient activity present")
    elif kurtosis < -1:
        interpretation.append("Negative kurtosis - signal more uniform than normal distribution")

    # RMS level interpretation (general guidelines)
    rms = metrics.get('RMS', 0)
    if rms > 10:
        interpretation.append("Very high RMS - immediate attention required")
    elif rms > 4.5:
        interpretation.append("High RMS - increased monitoring recommended")
    elif rms < 0.5:
        interpretation.append("Low RMS - good condition or possible sensor issue")

    # Combined pattern recognition
    if cf > 4 and kurtosis > 3:
        interpretation.append("Pattern suggests possible bearing fault - envelope analysis recommended")
    elif rms > 2 and cf < 3:
        interpretation.append("Pattern suggests possible unbalance - balance check recommended")

    return interpretation


def convert_to_velocity_mm_s(value, units, speed_rpm=None):
    """Convert various vibration units to velocity in mm/s."""
    try:
        if units.lower() in ['mm/s', 'mms', 'mm_s']:
            return value
        elif units.lower() in ['m/s', 'ms', 'm_s']:
            return value * 1000  # m/s to mm/s
        elif units.lower() in ['mm', 'millimeter', 'displacement']:
            # Convert displacement to velocity: v = ω * d, assuming sinusoidal
            if speed_rpm and speed_rpm > 0:
                omega = 2 * np.pi * speed_rpm / 60  # rad/s
                return value * omega
            else:
                # Use typical frequency for estimation (50 Hz)
                omega = 2 * np.pi * 50
                return value * omega
        elif units.lower() in ['g', 'acceleration', 'g_accel']:
            # Convert acceleration to velocity: v = a / (2πf)
            if speed_rpm and speed_rpm > 0:
                freq = speed_rpm / 60  # Hz
                return (value * 9.81) / (2 * np.pi * freq)  # g to m/s² then to velocity
            else:
                # Use typical frequency (50 Hz)
                return (value * 9.81) / (2 * np.pi * 50)
        else:
            Logger.log_message_static(f"Vibration-Metrics: Unknown units '{units}', assuming mm/s", Logger.WARNING)
            return value
    except Exception as e:
        Logger.log_message_static(f"Vibration-Metrics: Unit conversion error: {e}", Logger.WARNING)
        return value


def get_iso10816_thresholds(machine_class):
    """Get ISO 10816 vibration threshold values for different machine classes."""
    thresholds = {
        'I': {  # Small machines (< 15 kW)
            'A': 0.71,
            'B': 1.8,
            'C': 4.5,
            'D': float('inf')
        },
        'II': {  # Medium machines (15-75 kW)
            'A': 1.12,
            'B': 2.8,
            'C': 7.1,
            'D': float('inf')
        },
        'III': {  # Large machines on rigid foundations (75-300 kW)
            'A': 1.8,
            'B': 4.5,
            'C': 11.2,
            'D': float('inf')
        },
        'IV': {  # Large machines on flexible foundations (> 300 kW)
            'A': 2.8,
            'B': 7.1,
            'C': 18.0,
            'D': float('inf')
        }
    }

    return thresholds.get(machine_class, thresholds['II'])  # Default to class II


def assess_speed_dependent_limits(rms_value, speed_rpm, machine_class):
    """Assess vibration limits that depend on operating speed."""
    try:
        # Speed-dependent assessment according to ISO standards
        base_thresholds = get_iso10816_thresholds(machine_class)

        # Adjust thresholds based on speed (simplified approach)
        if speed_rpm < 600:  # Low speed
            speed_factor = 0.8
            speed_note = "Low speed operation - reduced limits applied"
        elif speed_rpm > 3600:  # High speed
            speed_factor = 1.2
            speed_note = "High speed operation - increased limits applied"
        else:  # Normal speed range
            speed_factor = 1.0
            speed_note = "Normal speed range"

        adjusted_thresholds = {
            level: value * speed_factor
            for level, value in base_thresholds.items()
        }

        # Determine speed-adjusted severity
        if rms_value <= adjusted_thresholds['A']:
            speed_level = 'A'
        elif rms_value <= adjusted_thresholds['B']:
            speed_level = 'B'
        elif rms_value <= adjusted_thresholds['C']:
            speed_level = 'C'
        else:
            speed_level = 'D'

        return {
            'Speed Factor': speed_factor,
            'Speed Note': speed_note,
            'Adjusted Thresholds': adjusted_thresholds,
            'Speed-Adjusted Level': speed_level
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Metrics: Speed assessment error: {e}", Logger.WARNING)
        return None


def assess_machine_specific_condition(metrics, machine_info):
    """Provide machine-specific condition assessment."""
    try:
        machine_type = machine_info.get('machine_type', '').lower()
        speed = machine_info.get('operating_speed', 0)

        penalty = 0
        indicators = []
        recommendations = []

        # Machine-specific assessments
        if 'pump' in machine_type:
            # Pumps are sensitive to cavitation and hydraulic issues
            cf = metrics.get('Crest Factor', 0)
            if cf > 5:
                penalty += 10
                indicators.append("High crest factor in pump - possible cavitation")
                recommendations.append("Check suction conditions and NPSH")

        elif 'motor' in machine_type:
            # Motors can have electrical issues affecting vibration
            rms = metrics.get('RMS', 0)
            if speed and 1700 <= speed <= 1800:  # Typical motor speeds
                if rms > 3:
                    penalty += 15
                    indicators.append("High vibration in motor - check electrical and mechanical")

        elif 'fan' in machine_type:
            # Fans are sensitive to aerodynamic issues
            skewness = metrics.get('Skewness', 0)
            if abs(skewness) > 1.5:
                penalty += 8
                indicators.append("Asymmetric vibration in fan - check blade condition")
                recommendations.append("Inspect fan blades for damage or fouling")

        elif 'compressor' in machine_type:
            # Compressors have specific fault modes
            kurtosis = metrics.get('Kurtosis', 0)
            if kurtosis > 4:
                penalty += 12
                indicators.append("High kurtosis in compressor - check valve operation")

        return {
            'penalty': penalty,
            'indicators': indicators,
            'recommendations': recommendations
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Metrics: Machine-specific assessment error: {e}", Logger.WARNING)
        return None


def assess_confidence_level(metrics):
    """Assess the confidence level of the vibration analysis."""
    try:
        # Factors affecting confidence
        sample_count = metrics.get('Sample Count', 0)
        snr = metrics.get('Signal-to-Noise Ratio (dB)', 0)
        zero_percentage = metrics.get('Zero Percentage', 0)

        confidence_score = 100

        # Sample count assessment
        if sample_count < 1000:
            confidence_score -= 20
        elif sample_count < 5000:
            confidence_score -= 10

        # SNR assessment
        if snr < 20:
            confidence_score -= 15
        elif snr < 40:
            confidence_score -= 5

        # Zero percentage assessment (indicates data quality issues)
        if zero_percentage > 10:
            confidence_score -= 25
        elif zero_percentage > 5:
            confidence_score -= 10

        # Determine confidence level
        if confidence_score >= 85:
            return "High"
        elif confidence_score >= 70:
            return "Medium"
        else:
            return "Low"

    except Exception:
        return "Medium"


def generate_condition_summary(condition, score, fault_indicators):
    """Generate a human-readable summary of machine condition."""
    try:
        if condition == "Good":
            summary = f"Machine condition is good (score: {score}). Normal operation can continue with routine monitoring."
        elif condition == "Fair":
            summary = f"Machine condition is fair (score: {score}). Increased monitoring recommended."
        elif condition == "Poor":
            summary = f"Machine condition is poor (score: {score}). Maintenance planning should be initiated."
        else:  # Critical
            summary = f"Machine condition is critical (score: {score}). Immediate attention required."

        if fault_indicators:
            summary += f" Potential issues detected: {len(fault_indicators)} fault indicators found."

        return summary

    except Exception:
        return f"Condition assessment: {condition} with score {score}."