"""
Comprehensive vibration assessment and multi-channel analysis.

This module provides comprehensive machine condition assessment based on
multi-channel vibration data (DE/NDE, X/Y/Z axes) and integrates all
vibration analysis techniques for overall machine health evaluation.

Key features:
- Multi-channel vibration analysis (DE/NDE, X/Y/Z axes)
- Overall machine condition assessment
- Fault pattern recognition across channels
- Trending and historical comparison
- ISO standards compliance assessment
- Automated reporting and recommendations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from analysis.calculations.common import safe_prepare_signal, safe_sample_rate
from utils.logger import Logger


def perform_comprehensive_vibration_assessment(
        signal_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        machine_info: Optional[Dict] = None,
        dialog=None
) -> Dict[str, Any]:
    """
    Perform comprehensive vibration assessment on multi-channel data.

    Analyzes all available vibration channels (DE/NDE X/Y/Z) and provides
    integrated machine condition assessment with fault identification,
    severity evaluation, and maintenance recommendations.

    Args:
        signal_data (dict): Dictionary of signal data:
            Keys should follow naming convention:
            - 'DE_X', 'DE_Y', 'DE_Z' for Drive End accelerometers
            - 'NDE_X', 'NDE_Y', 'NDE_Z' for Non-Drive End accelerometers
            - 'RPM' for rotational speed signal (optional)
            Values are tuples of (time_array, values_array)
        machine_info (dict, optional): Machine parameters:
            - machine_type: Type of machine (motor, pump, fan, etc.)
            - rated_power: Power rating in kW
            - operating_speed: Nominal operating speed in RPM
            - bearing_info: Bearing specifications for each location
            - foundation_type: rigid/flexible
            - installation_date: For trending analysis
        dialog (QWidget, optional): Parent dialog for user interaction.

    Returns:
        dict: Comprehensive assessment results:
            - Overall Condition: Machine health status
            - Channel Analysis: Individual channel assessments
            - Fault Detection: Identified faults across channels
            - Severity Assessment: ISO compliance and severity levels
            - Recommendations: Maintenance recommendations
            - Trending Data: Data for historical trending
            - Quality Assessment: Data quality indicators

    Example:
        >>> signals = {
        ...     'DE_X': (time_array, de_x_values),
        ...     'DE_Y': (time_array, de_y_values),
        ...     'NDE_X': (time_array, nde_x_values),
        ...     'RPM': (time_array, rpm_values)
        ... }
        >>> machine_info = {
        ...     'machine_type': 'motor',
        ...     'rated_power': 75,
        ...     'operating_speed': 1800
        ... }
        >>> assessment = perform_comprehensive_vibration_assessment(signals, machine_info)
        >>> print(f"Overall condition: {assessment['Overall Condition']['status']}")
    """
    Logger.log_message_static("Vibration-Assessment: Starting comprehensive vibration assessment", Logger.INFO)

    try:
        # Validate input data
        if not signal_data:
            Logger.log_message_static("Vibration-Assessment: No signal data provided", Logger.ERROR)
            return {"error": "No signal data provided"}

        # Extract channel information
        channel_info = classify_vibration_channels(signal_data)

        # Extract RPM if available
        rpm_value = extract_rpm_value(signal_data, channel_info)

        # Perform individual channel analysis
        channel_results = analyze_individual_channels(signal_data, channel_info, rpm_value, machine_info, dialog)

        # Perform multi-channel correlation analysis
        correlation_analysis = analyze_channel_correlations(signal_data, channel_info)

        # Detect fault patterns across channels
        fault_patterns = detect_multi_channel_fault_patterns(channel_results, machine_info)

        # Assess overall machine condition
        overall_condition = assess_overall_machine_condition(channel_results, fault_patterns, machine_info)

        # Generate ISO compliance assessment
        iso_assessment = perform_iso_compliance_assessment(channel_results, machine_info)

        # Calculate quality metrics
        quality_assessment = assess_measurement_quality(signal_data, channel_results)

        # Generate maintenance recommendations
        recommendations = generate_maintenance_recommendations(overall_condition, fault_patterns, iso_assessment)

        # Prepare trending data
        trending_data = prepare_trending_data(channel_results, overall_condition)

        # Generate summary report
        summary_report = generate_assessment_summary(
            overall_condition, fault_patterns, iso_assessment, recommendations
        )

        # Compile comprehensive results
        assessment_results = {
            # Core assessment
            "Overall Condition": overall_condition,
            "Summary Report": summary_report,
            "ISO Compliance": iso_assessment,

            # Detailed analysis
            "Channel Analysis": channel_results,
            "Multi-Channel Correlation": correlation_analysis,
            "Fault Patterns": fault_patterns,
            "Recommendations": recommendations,

            # Supporting data
            "Quality Assessment": quality_assessment,
            "Trending Data": trending_data,
            "Channel Information": channel_info,

            # Metadata
            "Assessment Timestamp": "Current",
            "Machine Information": machine_info or {},
            "RPM Value": rpm_value,
            "Analysis Software": "MakeATrend Vibration Analysis",
            "Standards Applied": ["ISO 10816", "ISO 20816"]
        }

        Logger.log_message_static(
            f"Vibration-Assessment: Assessment completed. "
            f"Overall condition: {overall_condition.get('status', 'Unknown')}, "
            f"Channels analyzed: {len(channel_results)}, "
            f"Faults detected: {len(fault_patterns.get('detected_faults', []))}",
            Logger.INFO
        )

        return assessment_results

    except Exception as e:
        Logger.log_message_static(f"Vibration-Assessment: Error in comprehensive assessment: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Vibration-Assessment: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return {
            "error": str(e),
            "Overall Condition": {"status": "Error", "message": "Assessment failed"}
        }


def classify_vibration_channels(signal_data: Dict[str, Tuple]) -> Dict[str, Any]:
    """
    Classify and organize vibration channels based on naming conventions.

    Identifies Drive End (DE) and Non-Drive End (NDE) channels,
    axis orientations (X, Y, Z), and other signal types.
    """
    try:
        channel_classification = {
            'DE': {'X': None, 'Y': None, 'Z': None},
            'NDE': {'X': None, 'Y': None, 'Z': None},
            'RPM': None,
            'Other': {},
            'Channel_Count': 0,
            'Available_Axes': set(),
            'Available_Locations': set()
        }

        # Common naming patterns
        de_patterns = ['de_', 'drive_end_', 'de-', 'drive-end-', 'motor_']
        nde_patterns = ['nde_', 'non_drive_end_', 'nde-', 'non-drive-end-', 'free_end_', 'fan_']
        rpm_patterns = ['rpm', 'speed', 'rotation', 'tacho', 'ot']

        for signal_name, signal_tuple in signal_data.items():
            name_lower = signal_name.lower()

            # Check for RPM signals
            if any(pattern in name_lower for pattern in rpm_patterns):
                channel_classification['RPM'] = signal_name
                continue

            # Determine location (DE/NDE)
            location = None
            if any(pattern in name_lower for pattern in de_patterns):
                location = 'DE'
            elif any(pattern in name_lower for pattern in nde_patterns):
                location = 'NDE'

            # Determine axis
            axis = None
            if '_x' in name_lower or '-x' in name_lower or name_lower.endswith('x'):
                axis = 'X'
            elif '_y' in name_lower or '-y' in name_lower or name_lower.endswith('y'):
                axis = 'Y'
            elif '_z' in name_lower or '-z' in name_lower or name_lower.endswith('z'):
                axis = 'Z'

            # Assign to appropriate category
            if location and axis:
                channel_classification[location][axis] = signal_name
                channel_classification['Available_Axes'].add(axis)
                channel_classification['Available_Locations'].add(location)
                channel_classification['Channel_Count'] += 1
            else:
                # Store in 'Other' category
                channel_classification['Other'][signal_name] = {
                    'detected_location': location,
                    'detected_axis': axis
                }

        # Convert sets to lists for JSON serialization
        channel_classification['Available_Axes'] = list(channel_classification['Available_Axes'])
        channel_classification['Available_Locations'] = list(channel_classification['Available_Locations'])

        Logger.log_message_static(
            f"Vibration-Assessment: Channel classification completed. "
            f"Channels: {channel_classification['Channel_Count']}, "
            f"Locations: {channel_classification['Available_Locations']}, "
            f"Axes: {channel_classification['Available_Axes']}",
            Logger.DEBUG
        )

        return channel_classification

    except Exception as e:
        Logger.log_message_static(f"Vibration-Assessment: Channel classification error: {e}", Logger.WARNING)
        return {'Channel_Count': 0, 'Available_Axes': [], 'Available_Locations': []}


def extract_rpm_value(signal_data: Dict, channel_info: Dict) -> Optional[float]:
    """Extract RPM value from available signals."""
    try:
        rpm_signal_name = channel_info.get('RPM')
        if not rpm_signal_name:
            Logger.log_message_static("Vibration-Assessment: No RPM signal found", Logger.DEBUG)
            return None

        if rpm_signal_name not in signal_data:
            Logger.log_message_static(f"Vibration-Assessment: RPM signal '{rpm_signal_name}' not found in data",
                                      Logger.WARNING)
            return None

        time_arr, rpm_values = signal_data[rpm_signal_name]

        # Calculate mean RPM, excluding outliers
        rpm_median = np.median(rpm_values)
        rpm_std = np.std(rpm_values)

        # Filter outliers (within 3 standard deviations)
        valid_rpm = rpm_values[np.abs(rpm_values - rpm_median) <= 3 * rpm_std]

        if len(valid_rpm) == 0:
            Logger.log_message_static("Vibration-Assessment: No valid RPM values after outlier removal", Logger.WARNING)
            return None

        mean_rpm = np.mean(valid_rpm)

        # Validate RPM range (typical industrial machinery)
        if not (10 <= mean_rpm <= 10000):
            Logger.log_message_static(f"Vibration-Assessment: RPM value {mean_rpm:.1f} outside typical range",
                                      Logger.WARNING)

        Logger.log_message_static(f"Vibration-Assessment: Extracted RPM: {mean_rpm:.1f}", Logger.DEBUG)
        return float(mean_rpm)

    except Exception as e:
        Logger.log_message_static(f"Vibration-Assessment: RPM extraction error: {e}", Logger.WARNING)
        return None


def analyze_individual_channels(
        signal_data: Dict,
        channel_info: Dict,
        rpm_value: Optional[float],
        machine_info: Optional[Dict],
        dialog
) -> Dict[str, Dict]:
    """Perform comprehensive analysis on each individual vibration channel."""
    try:
        channel_results = {}

        # Import analysis functions
        from vibration_metrics import calculate_vibration_metrics, calculate_vibration_severity
        from vibration_fft import calculate_vibration_fft
        from vibration_envelope import calculate_envelope_analysis

        # Analyze each channel
        for location in ['DE', 'NDE']:
            for axis in ['X', 'Y', 'Z']:
                signal_name = channel_info.get(location, {}).get(axis)
                if not signal_name or signal_name not in signal_data:
                    continue

                try:
                    time_arr, values = signal_data[signal_name]
                    channel_key = f"{location}_{axis}"

                    Logger.log_message_static(f"Vibration-Assessment: Analyzing channel {channel_key}", Logger.DEBUG)

                    # Basic vibration metrics
                    metrics = calculate_vibration_metrics(time_arr, values, dialog, f"Channel {channel_key}")
                    if metrics is None:
                        Logger.log_message_static(
                            f"Vibration-Assessment: Failed to calculate metrics for {channel_key}", Logger.WARNING)
                        continue

                    # Severity assessment (assuming velocity measurements in mm/s)
                    rms_value = metrics.get('RMS', 0)
                    severity = calculate_vibration_severity(rms_value, "mm/s", "II")  # Default to Class II

                    # FFT analysis
                    fft_results = calculate_vibration_fft(
                        time_arr, values, dialog, f"FFT {channel_key}",
                        rpm=rpm_value, machine_info=machine_info
                    )

                    # Envelope analysis
                    envelope_results = calculate_envelope_analysis(
                        time_arr, values, dialog, f"Envelope {channel_key}",
                        filter_type="adaptive"
                    )

                    # Compile channel results
                    channel_results[channel_key] = {
                        'Signal_Name': signal_name,
                        'Location': location,
                        'Axis': axis,
                        'Metrics': metrics,
                        'Severity': severity,
                        'FFT_Analysis': fft_results,
                        'Envelope_Analysis': envelope_results,
                        'Data_Quality': assess_channel_data_quality(time_arr, values),
                        'Channel_Status': determine_channel_status(metrics, severity, fft_results)
                    }

                except Exception as channel_error:
                    Logger.log_message_static(f"Vibration-Assessment: Error analyzing {channel_key}: {channel_error}",
                                              Logger.WARNING)
                    channel_results[f"{location}_{axis}"] = {
                        'Signal_Name': signal_name,
                        'Location': location,
                        'Axis': axis,
                        'Error': str(channel_error),
                        'Channel_Status': 'Error'
                    }

        # Analyze other signals
        for signal_name in channel_info.get('Other', {}):
            if signal_name in signal_data:
                try:
                    time_arr, values = signal_data[signal_name]

                    metrics = calculate_vibration_metrics(time_arr, values, dialog, f"Signal {signal_name}")
                    if metrics:
                        channel_results[f"Other_{signal_name}"] = {
                            'Signal_Name': signal_name,
                            'Location': 'Unknown',
                            'Axis': 'Unknown',
                            'Metrics': metrics,
                            'Data_Quality': assess_channel_data_quality(time_arr, values),
                            'Channel_Status': 'Analyzed'
                        }

                except Exception as other_error:
                    Logger.log_message_static(
                        f"Vibration-Assessment: Error analyzing other signal {signal_name}: {other_error}",
                        Logger.WARNING)

        Logger.log_message_static(
            f"Vibration-Assessment: Individual channel analysis completed. Channels analyzed: {len(channel_results)}",
            Logger.DEBUG)
        return channel_results

    except Exception as e:
        Logger.log_message_static(f"Vibration-Assessment: Error in individual channel analysis: {e}", Logger.ERROR)
        return {}


def assess_channel_data_quality(time_arr: np.ndarray, values: np.ndarray) -> Dict[str, Any]:
    """Assess the quality of vibration data for a single channel."""
    try:
        # Basic data characteristics
        sample_count = len(values)
        duration = time_arr[-1] - time_arr[0] if len(time_arr) > 1 else 0
        sample_rate = safe_sample_rate(time_arr)

        # Data completeness
        non_zero_count = np.count_nonzero(values)
        zero_percentage = (sample_count - non_zero_count) / sample_count * 100

        # Signal-to-noise estimation
        signal_power = np.var(values)
        noise_estimate = np.var(np.diff(values))
        snr_estimate = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 0 else np.inf

        # Clipping detection
        max_val = np.max(np.abs(values))
        near_max_count = np.sum(np.abs(values) > 0.95 * max_val)
        clipping_percentage = near_max_count / sample_count * 100

        # Determine overall quality
        quality_score = 100

        if zero_percentage > 10:
            quality_score -= 30
        elif zero_percentage > 5:
            quality_score -= 15

        if snr_estimate < 20:
            quality_score -= 20
        elif snr_estimate < 40:
            quality_score -= 10

        if clipping_percentage > 1:
            quality_score -= 25
        elif clipping_percentage > 0.1:
            quality_score -= 10

        if sample_count < 5000:
            quality_score -= 15
        elif sample_count < 10000:
            quality_score -= 5

        # Quality level
        if quality_score >= 85:
            quality_level = "Excellent"
        elif quality_score >= 70:
            quality_level = "Good"
        elif quality_score >= 50:
            quality_level = "Fair"
        else:
            quality_level = "Poor"

        return {
            'Quality_Level': quality_level,
            'Quality_Score': max(0, quality_score),
            'Sample_Count': sample_count,
            'Duration_s': float(duration),
            'Sample_Rate_Hz': float(sample_rate),
            'Zero_Percentage': float(zero_percentage),
            'SNR_Estimate_dB': float(snr_estimate),
            'Clipping_Percentage': float(clipping_percentage),
            'Data_Range': f"{np.min(values):.6f} to {np.max(values):.6f}"
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-Assessment: Data quality assessment error: {e}", Logger.WARNING)
        return {'Quality_Level': 'Unknown', 'Quality_Score': 0}


def determine_channel_status(metrics: Dict, severity: Dict, fft_results: Optional[Dict]) -> str:
    """Determine overall status of individual vibration channel."""
    try:
        # Check for errors or missing data
        if not metrics or not severity:
            return "Error"

        # Get severity level
        severity_level = severity.get('Severity Level', 'Unknown')

        # Get key metrics
        rms = metrics.get('RMS', 0)
        crest_factor = metrics.get('Crest Factor', 0)
        kurtosis = metrics.get('Kurtosis', 0)

        # Assess based on multiple factors
        if severity_level == 'A':
            status = "Good"
        elif severity_level == 'B':
            status = "Acceptable"
        elif severity_level == 'C':
            status = "Unsatisfactory"
        elif severity_level == 'D':
            status = "Unacceptable"
            else:
            # Fallback assessment based on metrics
            if rms < 1.8:
                status = "Good"
            elif rms < 4.5:
                status = "Acceptable"
            elif rms < 11.2:
                status = "Unsatisfactory"
            else:
                status = "Unacceptable"

            # Check for fault indicators
        fault_indicators = []
        if crest_factor > 6:
            fault_indicators.append("High Crest Factor")
        if kurtosis > 5:
            fault_indicators.append("High Kurtosis")
        if rms > 10:
            fault_indicators.append("Very High RMS")

        # Modify status if fault indicators present
        if fault_indicators and status in ["Good", "Acceptable"]:
            status = "Investigate"

        return status

    except Exception as e:
        Logger.log_message_static(f"Vibration-Assessment: Channel status determination error: {e}", Logger.WARNING)
        return "Unknown"

    def analyze_channel_correlations(signal_data: Dict, channel_info: Dict) -> Dict[str, Any]:
        """Analyze correlations between different vibration channels."""
        try:
            correlation_results = {
                'Cross_Correlations': {},
                'Phase_Relationships': {},
                'Coherence_Analysis': {},
                'Summary': {}
            }

            # Get all vibration channels (exclude RPM)
            vibration_channels = []
            for location in ['DE', 'NDE']:
                for axis in ['X', 'Y', 'Z']:
                    signal_name = channel_info.get(location, {}).get(axis)
                    if signal_name and signal_name in signal_data:
                        vibration_channels.append((f"{location}_{axis}", signal_name))

            if len(vibration_channels) < 2:
                Logger.log_message_static("Vibration-Assessment: Not enough channels for correlation analysis",
                                          Logger.DEBUG)
                return correlation_results

            # Calculate cross-correlations between all channel pairs
            from scipy.signal import correlate
            from scipy import stats

            for i, (ch1_key, ch1_name) in enumerate(vibration_channels):
                for j, (ch2_key, ch2_name) in enumerate(vibration_channels):
                    if i >= j:  # Only calculate upper triangle
                        continue

                    try:
                        time1, values1 = signal_data[ch1_name]
                        time2, values2 = signal_data[ch2_name]

                        # Ensure same length (use shorter signal)
                        min_length = min(len(values1), len(values2))
                        v1 = values1[:min_length]
                        v2 = values2[:min_length]

                        # Calculate correlation coefficient
                        correlation_coef = stats.pearsonr(v1, v2)[0]

                        # Calculate cross-correlation for time delay
                        cross_corr = correlate(v1 - np.mean(v1), v2 - np.mean(v2), mode='full')
                        cross_corr = cross_corr / (np.std(v1) * np.std(v2) * len(v1))

                        # Find maximum correlation and corresponding lag
                        max_corr_idx = np.argmax(np.abs(cross_corr))
                        max_correlation = cross_corr[max_corr_idx]
                        lag_samples = max_corr_idx - len(v1) + 1

                        # Convert lag to time
                        sample_rate = safe_sample_rate(time1)
                        lag_time = lag_samples / sample_rate if sample_rate > 0 else 0

                        correlation_results['Cross_Correlations'][f"{ch1_key}_vs_{ch2_key}"] = {
                            'Correlation_Coefficient': float(correlation_coef),
                            'Max_Cross_Correlation': float(max_correlation),
                            'Time_Lag_s': float(lag_time),
                            'Lag_Samples': int(lag_samples)
                        }

                    except Exception as pair_error:
                        Logger.log_message_static(
                            f"Vibration-Assessment: Correlation error for {ch1_key} vs {ch2_key}: {pair_error}",
                            Logger.WARNING)

            # Analyze same-location cross-axis correlations
            location_correlations = analyze_location_correlations(signal_data, channel_info)
            correlation_results['Location_Analysis'] = location_correlations

            # Generate correlation summary
            correlations = correlation_results['Cross_Correlations']
            if correlations:
                corr_values = [result['Correlation_Coefficient'] for result in correlations.values()]
                correlation_results['Summary'] = {
                    'Average_Correlation': float(np.mean(np.abs(corr_values))),
                    'Max_Correlation': float(np.max(np.abs(corr_values))),
                    'Min_Correlation': float(np.min(np.abs(corr_values))),
                    'High_Correlation_Pairs': len([c for c in corr_values if abs(c) > 0.7]),
                    'Total_Pairs': len(corr_values)
                }

            Logger.log_message_static(f"Vibration-Assessment: Channel correlation analysis completed", Logger.DEBUG)
            return correlation_results

        except Exception as e:
            Logger.log_message_static(f"Vibration-Assessment: Correlation analysis error: {e}", Logger.WARNING)
            return {'Cross_Correlations': {}, 'Summary': {}}

    def analyze_location_correlations(signal_data: Dict, channel_info: Dict) -> Dict[str, Any]:
        """Analyze correlations within each bearing location (DE/NDE)."""
        try:
            location_analysis = {}

            for location in ['DE', 'NDE']:
                axes_data = {}

                # Collect all axes for this location
                for axis in ['X', 'Y', 'Z']:
                    signal_name = channel_info.get(location, {}).get(axis)
                    if signal_name and signal_name in signal_data:
                        axes_data[axis] = signal_data[signal_name][1]  # values only

                if len(axes_data) < 2:
                    continue

                # Calculate RMS values for each axis
                axis_rms = {axis: np.sqrt(np.mean(values ** 2)) for axis, values in axes_data.items()}

                # Find dominant axis
                dominant_axis = max(axis_rms.keys(), key=lambda k: axis_rms[k])

                # Calculate axis ratios
                axis_ratios = {}
                for axis, rms in axis_rms.items():
                    if axis != dominant_axis:
                        axis_ratios[f"{axis}/{dominant_axis}"] = rms / axis_rms[dominant_axis]

                location_analysis[location] = {
                    'Axis_RMS': {axis: float(rms) for axis, rms in axis_rms.items()},
                    'Dominant_Axis': dominant_axis,
                    'Axis_Ratios': {ratio: float(value) for ratio, value in axis_ratios.items()},
                    'Total_RMS': float(np.sqrt(sum(rms ** 2 for rms in axis_rms.values()))),
                    'Available_Axes': list(axes_data.keys())
                }

            return location_analysis

        except Exception as e:
            Logger.log_message_static(f"Vibration-Assessment: Location correlation analysis error: {e}", Logger.WARNING)
            return {}

    def detect_multi_channel_fault_patterns(channel_results: Dict, machine_info: Optional[Dict]) -> Dict[str, Any]:
        """Detect fault patterns that manifest across multiple channels."""
        try:
            fault_patterns = {
                'Detected_Faults': [],
                'Fault_Confidence': {},
                'Pattern_Analysis': {},
                'Cross_Channel_Indicators': {}
            }

            # Extract key metrics from all channels
            channel_metrics = {}
            for channel_key, results in channel_results.items():
                if 'Metrics' in results and results['Metrics']:
                    metrics = results['Metrics']
                    channel_metrics[channel_key] = {
                        'RMS': metrics.get('RMS', 0),
                        'Peak': metrics.get('Peak', 0),
                        'Crest_Factor': metrics.get('Crest Factor', 0),
                        'Kurtosis': metrics.get('Kurtosis', 0),
                        'Location': results.get('Location', 'Unknown'),
                        'Axis': results.get('Axis', 'Unknown')
                    }

            if not channel_metrics:
                return fault_patterns

            # Pattern 1: Unbalance Detection
            unbalance_indicators = detect_unbalance_pattern(channel_metrics)
            if unbalance_indicators['detected']:
                fault_patterns['Detected_Faults'].append({
                    'Fault_Type': 'Unbalance',
                    'Confidence': unbalance_indicators['confidence'],
                    'Affected_Channels': unbalance_indicators['channels'],
                    'Indicators': unbalance_indicators['indicators'],
                    'Severity': unbalance_indicators['severity']
                })

            # Pattern 2: Misalignment Detection
            misalignment_indicators = detect_misalignment_pattern(channel_metrics)
            if misalignment_indicators['detected']:
                fault_patterns['Detected_Faults'].append({
                    'Fault_Type': 'Misalignment',
                    'Confidence': misalignment_indicators['confidence'],
                    'Affected_Channels': misalignment_indicators['channels'],
                    'Indicators': misalignment_indicators['indicators'],
                    'Severity': misalignment_indicators['severity']
                })

            # Pattern 3: Bearing Fault Detection
            bearing_indicators = detect_bearing_fault_pattern(channel_metrics, channel_results)
            if bearing_indicators['detected']:
                fault_patterns['Detected_Faults'].append({
                    'Fault_Type': 'Bearing Fault',
                    'Confidence': bearing_indicators['confidence'],
                    'Affected_Channels': bearing_indicators['channels'],
                    'Indicators': bearing_indicators['indicators'],
                    'Severity': bearing_indicators['severity'],
                    'Location': bearing_indicators.get('bearing_location', 'Unknown')
                })

            # Pattern 4: Looseness Detection
            looseness_indicators = detect_looseness_pattern(channel_metrics, channel_results)
            if looseness_indicators['detected']:
                fault_patterns['Detected_Faults'].append({
                    'Fault_Type': 'Mechanical Looseness',
                    'Confidence': looseness_indicators['confidence'],
                    'Affected_Channels': looseness_indicators['channels'],
                    'Indicators': looseness_indicators['indicators'],
                    'Severity': looseness_indicators['severity']
                })

            # Pattern 5: Foundation Issues
            foundation_indicators = detect_foundation_issues(channel_metrics)
            if foundation_indicators['detected']:
                fault_patterns['Detected_Faults'].append({
                    'Fault_Type': 'Foundation/Mounting Issues',
                    'Confidence': foundation_indicators['confidence'],
                    'Affected_Channels': foundation_indicators['channels'],
                    'Indicators': foundation_indicators['indicators'],
                    'Severity': foundation_indicators['severity']
                })

            # Calculate overall fault confidence
            if fault_patterns['Detected_Faults']:
                confidences = [fault['Confidence'] for fault in fault_patterns['Detected_Faults']]
                fault_patterns['Overall_Confidence'] = np.mean(confidences)
                fault_patterns['Highest_Confidence_Fault'] = max(fault_patterns['Detected_Faults'],
                                                                 key=lambda x: x['Confidence'])

            Logger.log_message_static(
                f"Vibration-Assessment: Multi-channel fault detection completed. "
                f"Faults detected: {len(fault_patterns['Detected_Faults'])}",
                Logger.DEBUG
            )

            return fault_patterns

        except Exception as e:
            Logger.log_message_static(f"Vibration-Assessment: Multi-channel fault detection error: {e}", Logger.WARNING)
            return {'Detected_Faults': [], 'Pattern_Analysis': {}}

    def detect_unbalance_pattern(channel_metrics: Dict) -> Dict[str, Any]:
        """Detect unbalance fault pattern across channels."""
        try:
            # Unbalance characteristics:
            # - High radial vibration (X, Y axes)
            # - Higher at drive end
            # - Low crest factor
            # - Primarily 1X frequency component

            indicators = []
            affected_channels = []
            confidence = 0

            # Get radial channels (X, Y axes)
            radial_channels = {k: v for k, v in channel_metrics.items()
                               if v['Axis'] in ['X', 'Y']}

            if not radial_channels:
                return {'detected': False, 'confidence': 0}

            # Check for high radial vibration
            radial_rms_values = [metrics['RMS'] for metrics in radial_channels.values()]
            max_radial_rms = max(radial_rms_values)
            avg_radial_rms = np.mean(radial_rms_values)

            if max_radial_rms > 2.0:  # mm/s threshold
                indicators.append(f"High radial vibration: {max_radial_rms:.2f} mm/s")
                confidence += 30
                affected_channels.extend(radial_channels.keys())

            # Check crest factor (should be relatively low for unbalance)
            low_crest_count = 0
            for channel, metrics in radial_channels.items():
                if metrics['Crest_Factor'] < 4:
                    low_crest_count += 1

            if low_crest_count >= len(radial_channels) * 0.5:
                indicators.append("Low crest factor consistent with unbalance")
                confidence += 20

            # Check for DE vs NDE pattern
            de_rms = np.mean([m['RMS'] for k, m in radial_channels.items() if m['Location'] == 'DE'])
            nde_rms = np.mean([m['RMS'] for k, m in radial_channels.items() if m['Location'] == 'NDE'])

            if de_rms > 0 and nde_rms > 0:
                if de_rms > nde_rms * 1.5:
                    indicators.append("Higher vibration at drive end")
                    confidence += 25
                elif nde_rms > de_rms * 1.5:
                    indicators.append("Higher vibration at non-drive end")
                    confidence += 20

            # Determine severity
            if max_radial_rms > 7:
                severity = "High"
            elif max_radial_rms > 3:
                severity = "Medium"
            else:
                severity = "Low"

            detected = confidence > 40 and max_radial_rms > 1.5

            return {
                'detected': detected,
                'confidence': confidence,
                'channels': affected_channels,
                'indicators': indicators,
                'severity': severity,
                'max_radial_rms': max_radial_rms
            }

        except Exception as e:
            Logger.log_message_static(f"Vibration-Assessment: Unbalance detection error: {e}", Logger.WARNING)
            return {'detected': False, 'confidence': 0}

    def detect_misalignment_pattern(channel_metrics: Dict) -> Dict[str, Any]:
        """Detect misalignment fault pattern across channels."""
        try:
            # Misalignment characteristics:
            # - High axial vibration (Z axis)
            # - High vibration at both bearings
            # - Often 2X frequency dominant
            # - Higher crest factor than unbalance

            indicators = []
            affected_channels = []
            confidence = 0

            # Get axial channels (Z axis)
            axial_channels = {k: v for k, v in channel_metrics.items() if v['Axis'] == 'Z'}
            radial_channels = {k: v for k, v in channel_metrics.items() if v['Axis'] in ['X', 'Y']}

            # Check axial vibration levels
            if axial_channels:
                axial_rms_values = [metrics['RMS'] for metrics in axial_channels.values()]
                max_axial_rms = max(axial_rms_values)

                if max_axial_rms > 1.5:  # mm/s threshold for axial
                    indicators.append(f"High axial vibration: {max_axial_rms:.2f} mm/s")
                    confidence += 35
                    affected_channels.extend(axial_channels.keys())

            # Check for high vibration at both bearing locations
            if radial_channels:
                de_radial_rms = np.mean([m['RMS'] for k, m in radial_channels.items() if m['Location'] == 'DE'])
                nde_radial_rms = np.mean([m['RMS'] for k, m in radial_channels.items() if m['Location'] == 'NDE'])

                if de_radial_rms > 2.0 and nde_radial_rms > 2.0:
                    indicators.append("High vibration at both bearing locations")
                    confidence += 30
                    affected_channels.extend([k for k in radial_channels.keys()])

            # Check crest factor pattern
            moderate_crest_count = 0
            for channel, metrics in channel_metrics.items():
                if 3 < metrics['Crest_Factor'] < 6:
                    moderate_crest_count += 1

            if moderate_crest_count >= len(channel_metrics) * 0.5:
                indicators.append("Moderate crest factor pattern")
                confidence += 15

            # Determine severity
            if axial_channels:
                max_axial = max([m['RMS'] for m in axial_channels.values()])
                if max_axial > 4:
                    severity = "High"
                elif max_axial > 2:
                    severity = "Medium"
                else:
                    severity = "Low"
            else:
                severity = "Low"

            detected = confidence > 50 and len(indicators) >= 2

            return {
                'detected': detected,
                'confidence': confidence,
                'channels': affected_channels,
                'indicators': indicators,
                'severity': severity
            }

        except Exception as e:
            Logger.log_message_static(f"Vibration-Assessment: Misalignment detection error: {e}", Logger.WARNING)
            return {'detected': False, 'confidence': 0}

    def detect_bearing_fault_pattern(channel_metrics: Dict, channel_results: Dict) -> Dict[str, Any]:
        """Detect bearing fault patterns using metrics and envelope analysis."""
        try:
            indicators = []
            affected_channels = []
            confidence = 0
            bearing_location = "Unknown"

            # Bearing fault characteristics:
            # - High crest factor and kurtosis
            # - Impulsive signals
            # - Envelope analysis shows bearing frequencies

            # Check for high crest factor and kurtosis
            high_cf_channels = []
            high_kurt_channels = []

            for channel, metrics in channel_metrics.items():
                if metrics['Crest_Factor'] > 6:
                    high_cf_channels.append(channel)
                if metrics['Kurtosis'] > 5:
                    high_kurt_channels.append(channel)

            if high_cf_channels:
                indicators.append(f"High crest factor in channels: {', '.join(high_cf_channels)}")
                confidence += 30
                affected_channels.extend(high_cf_channels)

            if high_kurt_channels:
                indicators.append(f"High kurtosis in channels: {', '.join(high_kurt_channels)}")
                confidence += 25
                affected_channels.extend(high_kurt_channels)

            # Check envelope analysis results
            envelope_detections = 0
            for channel_key, results in channel_results.items():
                if 'Envelope_Analysis' in results and results['Envelope_Analysis']:
                    env_analysis = results['Envelope_Analysis']
                    if 'Fault Detection' in env_analysis:
                        fault_detection = env_analysis['Fault Detection']
                        bearing_faults = fault_detection.get('bearing_faults', [])
                        if bearing_faults:
                            envelope_detections += len(bearing_faults)
                            if channel_key not in affected_channels:
                                affected_channels.append(channel_key)

                            # Try to determine bearing location
                            if channel_metrics.get(channel_key, {}).get('Location') in ['DE', 'NDE']:
                                bearing_location = channel_metrics[channel_key]['Location']

            if envelope_detections > 0:
                indicators.append(f"Bearing fault frequencies detected in envelope analysis")
                confidence += 35

            # Determine severity based on highest metrics
            max_crest = max([m['Crest_Factor'] for m in channel_metrics.values()], default=0)
            max_kurt = max([m['Kurtosis'] for m in channel_metrics.values()], default=0)

            if max_crest > 10 or max_kurt > 10:
                severity = "High"
            elif max_crest > 6 or max_kurt > 5:
                severity = "Medium"
            else:
                severity = "Low"

            detected = confidence > 40 and (len(high_cf_channels) > 0 or envelope_detections > 0)

            return {
                'detected': detected,
                'confidence': confidence,
                'channels': list(set(affected_channels)),
                'indicators': indicators,
                'severity': severity,
                'bearing_location': bearing_location,
                'envelope_detections': envelope_detections
            }

        except Exception as e:
            Logger.log_message_static(f"Vibration-Assessment: Bearing fault detection error: {e}", Logger.WARNING)
            return {'detected': False, 'confidence': 0}

    def detect_looseness_pattern(channel_metrics: Dict, channel_results: Dict) -> Dict[str, Any]:
        """Detect mechanical looseness patterns."""
        try:
            indicators = []
            affected_channels = []
            confidence = 0

            # Looseness characteristics:
            # - Multiple harmonics in spectrum
            # - Moderate to high kurtosis
            # - Variable vibration levels
            # - Affects multiple axes

            # Check for moderate kurtosis across multiple channels
            moderate_kurt_channels = [k for k, m in channel_metrics.items() if 3 < m['Kurtosis'] < 8]

            if len(moderate_kurt_channels) >= 2:
                indicators.append("Moderate kurtosis across multiple channels")
                confidence += 25
                affected_channels.extend(moderate_kurt_channels)

            # Check harmonic content in FFT results
            harmonic_channels = []
            for channel_key, results in channel_results.items():
                if 'FFT_Analysis' in results and results['FFT_Analysis']:
                    fft_analysis = results['FFT_Analysis']
                    if 'Harmonic Analysis' in fft_analysis and fft_analysis['Harmonic Analysis']:
                        harmonic_ratio = fft_analysis['Harmonic Analysis'].get('Harmonic Ratio', 0)
                        if harmonic_ratio > 0.5:  # Multiple harmonics present
                            harmonic_channels.append(channel_key)

            if harmonic_channels:
                indicators.append(f"Multiple harmonics detected in: {', '.join(harmonic_channels)}")
                confidence += 30
                affected_channels.extend(harmonic_channels)

            # Check for variable RMS levels (indication of looseness)
            rms_values = [m['RMS'] for m in channel_metrics.values()]
            if len(rms_values) > 1:
                rms_cv = np.std(rms_values) / np.mean(rms_values)  # Coefficient of variation
                if rms_cv > 0.5:
                    indicators.append("High variability in vibration levels between channels")
                    confidence += 20

            # Determine severity
            max_rms = max([m['RMS'] for m in channel_metrics.values()], default=0)
            if max_rms > 5:
                severity = "High"
            elif max_rms > 2.5:
                severity = "Medium"
            else:
                severity = "Low"

            detected = confidence > 45 and len(indicators) >= 2

            return {
                'detected': detected,
                'confidence': confidence,
                'channels': list(set(affected_channels)),
                'indicators': indicators,
                'severity': severity
            }

        except Exception as e:
            Logger.log_message_static(f"Vibration-Assessment: Looseness detection error: {e}", Logger.WARNING)
            return {'detected': False, 'confidence': 0}

    def detect_foundation_issues(channel_metrics: Dict) -> Dict[str, Any]:
        """Detect foundation or mounting issues."""
        try:
            indicators = []
            affected_channels = []
            confidence = 0

            # Foundation issues characteristics:
            # - High vibration in vertical direction (often Y or Z)
            # - Similar levels at both bearings
            # - Low frequency content

            # Check for high vertical vibration
            vertical_channels = {k: v for k, v in channel_metrics.items() if v['Axis'] in ['Y', 'Z']}

            if vertical_channels:
                vertical_rms = [m['RMS'] for m in vertical_channels.values()]
                max_vertical = max(vertical_rms)

                if max_vertical > 3.0:
                    indicators.append(f"High vertical vibration: {max_vertical:.2f} mm/s")
                    confidence += 35
                    affected_channels.extend(vertical_channels.keys())

            # Check for similar levels at both bearings (foundation resonance)
            de_channels = {k: v for k, v in channel_metrics.items() if v['Location'] == 'DE'}
            nde_channels = {k: v for k, v in channel_metrics.items() if v['Location'] == 'NDE'}

            if de_channels and nde_channels:
                de_avg_rms = np.mean([m['RMS'] for m in de_channels.values()])
                nde_avg_rms = np.mean([m['RMS'] for m in nde_channels.values()])

                if abs(de_avg_rms - nde_avg_rms) / max(de_avg_rms, nde_avg_rms) < 0.3:
                    indicators.append("Similar vibration levels at both bearing locations")
                    confidence += 20

            # Check for overall high vibration levels
            all_rms = [m['RMS'] for m in channel_metrics.values()]
            avg_rms = np.mean(all_rms)

            if avg_rms > 4.0:
                indicators.append("Overall high vibration levels across all channels")
                confidence += 25
                affected_channels.extend(channel_metrics.keys())

            # Determine severity
            max_rms = max(all_rms, default=0)
            if max_rms > 8:
                severity = "High"
            elif max_rms > 4:
                severity = "Medium"
            else:
                severity = "Low"

            detected = confidence > 40

            return {
                'detected': detected,
                'confidence': confidence,
                'channels': list(set(affected_channels)),
                'indicators': indicators,
                'severity': severity
            }

        except Exception as e:
            Logger.log_message_static(f"Vibration-Assessment: Foundation issues detection error: {e}", Logger.WARNING)
            return {'detected': False, 'confidence': 0}

    def assess_overall_machine_condition(
            channel_results: Dict,
            fault_patterns: Dict,
            machine_info: Optional[Dict]
    ) -> Dict[str, Any]:
        """Assess overall machine condition based on all analysis results."""
        try:
            # Initialize condition assessment
            condition_assessment = {
                'status': 'Unknown',
                'condition_score': 0,
                'primary_concerns': [],
                'secondary_concerns': [],
                'overall_severity': 'Unknown',
                'confidence_level': 'Medium',
                'assessment_basis': []
            }

            # Get all channel statuses and severities
            channel_statuses = []
            severity_levels = []

            for channel_key, results in channel_results.items():
                if 'Channel_Status' in results:
                    channel_statuses.append(results['Channel_Status'])

                if 'Severity' in results and 'Severity Level' in results['Severity']:
                    severity_levels.append(results['Severity']['Severity Level'])

            # Calculate base condition score from channel results
            base_score = 100

            # Penalty based on worst channel status
            if 'Unacceptable' in channel_statuses:
                base_score -= 60
                condition_assessment['primary_concerns'].append("Unacceptable vibration levels detected")
            elif 'Unsatisfactory' in channel_statuses:
                base_score -= 40
                condition_assessment['primary_concerns'].append("Unsatisfactory vibration levels detected")
            elif 'Investigate' in channel_statuses:
                base_score -= 25
                condition_assessment['secondary_concerns'].append("Channels require investigation")
            elif 'Acceptable' in channel_statuses:
                base_score -= 10

            # Penalty based on ISO severity levels
            if 'D' in severity_levels:
                base_score -= 50
            elif 'C' in severity_levels:
                base_score -= 30
            elif 'B' in severity_levels:
                base_score -= 15

            # Penalty for detected faults
            detected_faults = fault_patterns.get('Detected_Faults', [])
            for fault in detected_faults:
                fault_severity = fault.get('Severity', 'Low')
                fault_confidence = fault.get('Confidence', 0)

                if fault_severity == 'High':
                    penalty = 30
                elif fault_severity == 'Medium':
                    penalty = 20
                else:
                    penalty = 10

                # Adjust penalty by confidence
                penalty = penalty * (fault_confidence / 100)
                base_score -= penalty

                condition_assessment['primary_concerns'].append(
                    f"{fault['Fault_Type']} detected ({fault_severity} severity)"
                )

            # Ensure score is within bounds
            condition_score = max(0, min(100, base_score))

            # Determine overall status
            if condition_score >= 85:
                status = 'Excellent'
                overall_severity = 'None'
            elif condition_score >= 70:
                status = 'Good'
                overall_severity = 'Low'
            elif condition_score >= 55:
                status = 'Fair'
                overall_severity = 'Medium'
            """
    Comprehensive vibration assessment and multi-channel analysis.

    This module provides comprehensive machine condition assessment based on
    multi-channel vibration data (DE/NDE, X/Y/Z axes) and integrates all
    vibration analysis techniques for overall machine health evaluation.

    Key features:
    - Multi-channel vibration analysis (DE/NDE, X/Y/Z axes)
    - Overall machine condition assessment
    - Fault pattern recognition across channels
    - Trending and historical comparison
    - ISO standards compliance assessment
    - Automated reporting and recommendations
    """

    import numpy as np
    from typing import Dict, List, Tuple, Optional, Any
    from analysis.calculations.common import safe_prepare_signal, safe_sample_rate
    from utils.logger import Logger

    def perform_comprehensive_vibration_assessment(
            signal_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
            machine_info: Optional[Dict] = None,
            dialog=None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive vibration assessment on multi-channel data.

        Analyzes all available vibration channels (DE/NDE X/Y/Z) and provides
        integrated machine condition assessment with fault identification,
        severity evaluation, and maintenance recommendations.

        Args:
            signal_data (dict): Dictionary of signal data:
                Keys should follow naming convention:
                - 'DE_X', 'DE_Y', 'DE_Z' for Drive End accelerometers
                - 'NDE_X', 'NDE_Y', 'NDE_Z' for Non-Drive End accelerometers
                - 'RPM' for rotational speed signal (optional)
                Values are tuples of (time_array, values_array)
            machine_info (dict, optional): Machine parameters:
                - machine_type: Type of machine (motor, pump, fan, etc.)
                - rated_power: Power rating in kW
                - operating_speed: Nominal operating speed in RPM
                - bearing_info: Bearing specifications for each location
                - foundation_type: rigid/flexible
                - installation_date: For trending analysis
            dialog (QWidget, optional): Parent dialog for user interaction.

        Returns:
            dict: Comprehensive assessment results:
                - Overall Condition: Machine health status
                - Channel Analysis: Individual channel assessments
                - Fault Detection: Identified faults across channels
                - Severity Assessment: ISO compliance and severity levels
                - Recommendations: Maintenance recommendations
                - Trending Data: Data for historical trending
                - Quality Assessment: Data quality indicators

        Example:
            >>> signals = {
            ...     'DE_X': (time_array, de_x_values),
            ...     'DE_Y': (time_array, de_y_values),
            ...     'NDE_X': (time_array, nde_x_values),
            ...     'RPM': (time_array, rpm_values)
            ... }
            >>> machine_info = {
            ...     'machine_type': 'motor',
            ...     'rated_power': 75,
            ...     'operating_speed': 1800
            ... }
            >>> assessment = perform_comprehensive_vibration_assessment(signals, machine_info)
            >>> print(f"Overall condition: {assessment['Overall Condition']['status']}")
        """
        Logger.log_message_static("Vibration-Assessment: Starting comprehensive vibration assessment", Logger.INFO)

        try:
            # Validate input data
            if not signal_data:
                Logger.log_message_static("Vibration-Assessment: No signal data provided", Logger.ERROR)
                return {"error": "No signal data provided"}

            # Extract channel information
            channel_info = classify_vibration_channels(signal_data)

            # Extract RPM if available
            rpm_value = extract_rpm_value(signal_data, channel_info)

            # Perform individual channel analysis
            channel_results = analyze_individual_channels(signal_data, channel_info, rpm_value, machine_info, dialog)

            # Perform multi-channel correlation analysis
            correlation_analysis = analyze_channel_correlations(signal_data, channel_info)

            # Detect fault patterns across channels
            fault_patterns = detect_multi_channel_fault_patterns(channel_results, machine_info)

            # Assess overall machine condition
            overall_condition = assess_overall_machine_condition(channel_results, fault_patterns, machine_info)

            # Generate ISO compliance assessment
            iso_assessment = perform_iso_compliance_assessment(channel_results, machine_info)

            # Calculate quality metrics
            quality_assessment = assess_measurement_quality(signal_data, channel_results)

            # Generate maintenance recommendations
            recommendations = generate_maintenance_recommendations(overall_condition, fault_patterns, iso_assessment)

            # Prepare trending data
            trending_data = prepare_trending_data(channel_results, overall_condition)

            # Generate summary report
            summary_report = generate_assessment_summary(
                overall_condition, fault_patterns, iso_assessment, recommendations
            )

            # Compile comprehensive results
            assessment_results = {
                # Core assessment
                "Overall Condition": overall_condition,
                "Summary Report": summary_report,
                "ISO Compliance": iso_assessment,

                # Detailed analysis
                "Channel Analysis": channel_results,
                "Multi-Channel Correlation": correlation_analysis,
                "Fault Patterns": fault_patterns,
                "Recommendations": recommendations,

                # Supporting data
                "Quality Assessment": quality_assessment,
                "Trending Data": trending_data,
                "Channel Information": channel_info,

                # Metadata
                "Assessment Timestamp": "Current",
                "Machine Information": machine_info or {},
                "RPM Value": rpm_value,
                "Analysis Software": "MakeATrend Vibration Analysis",
                "Standards Applied": ["ISO 10816", "ISO 20816"]
            }

            Logger.log_message_static(
                f"Vibration-Assessment: Assessment completed. "
                f"Overall condition: {overall_condition.get('status', 'Unknown')}, "
                f"Channels analyzed: {len(channel_results)}, "
                f"Faults detected: {len(fault_patterns.get('detected_faults', []))}",
                Logger.INFO
            )

            return assessment_results

        except Exception as e:
            Logger.log_message_static(f"Vibration-Assessment: Error in comprehensive assessment: {str(e)}",
                                      Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Vibration-Assessment: Traceback: {traceback.format_exc()}", Logger.DEBUG)
            return {
                "error": str(e),
                "Overall Condition": {"status": "Error", "message": "Assessment failed"}
            }

    def classify_vibration_channels(signal_data: Dict[str, Tuple]) -> Dict[str, Any]:
        """
        Classify and organize vibration channels based on naming conventions.

        Identifies Drive End (DE) and Non-Drive End (NDE) channels,
        axis orientations (X, Y, Z), and other signal types.
        """
        try:
            channel_classification = {
                'DE': {'X': None, 'Y': None, 'Z': None},
                'NDE': {'X': None, 'Y': None, 'Z': None},
                'RPM': None,
                'Other': {},
                'Channel_Count': 0,
                'Available_Axes': set(),
                'Available_Locations': set()
            }

            # Common naming patterns
            de_patterns = ['de_', 'drive_end_', 'de-', 'drive-end-', 'motor_']
            nde_patterns = ['nde_', 'non_drive_end_', 'nde-', 'non-drive-end-', 'free_end_', 'fan_']
            rpm_patterns = ['rpm', 'speed', 'rotation', 'tacho', 'ot']

            for signal_name, signal_tuple in signal_data.items():
                name_lower = signal_name.lower()

                # Check for RPM signals
                if any(pattern in name_lower for pattern in rpm_patterns):
                    channel_classification['RPM'] = signal_name
                    continue

                # Determine location (DE/NDE)
                location = None
                if any(pattern in name_lower for pattern in de_patterns):
                    location = 'DE'
                elif any(pattern in name_lower for pattern in nde_patterns):
                    location = 'NDE'

                # Determine axis
                axis = None
                if '_x' in name_lower or '-x' in name_lower or name_lower.endswith('x'):
                    axis = 'X'
                elif '_y' in name_lower or '-y' in name_lower or name_lower.endswith('y'):
                    axis = 'Y'
                elif '_z' in name_lower or '-z' in name_lower or name_lower.endswith('z'):
                    axis = 'Z'

                # Assign to appropriate category
                if location and axis:
                    channel_classification[location][axis] = signal_name
                    channel_classification['Available_Axes'].add(axis)
                    channel_classification['Available_Locations'].add(location)
                    channel_classification['Channel_Count'] += 1
                else:
                    # Store in 'Other' category
                    channel_classification['Other'][signal_name] = {
                        'detected_location': location,
                        'detected_axis': axis
                    }

            # Convert sets to lists for JSON serialization
            channel_classification['Available_Axes'] = list(channel_classification['Available_Axes'])
            channel_classification['Available_Locations'] = list(channel_classification['Available_Locations'])

            Logger.log_message_static(
                f"Vibration-Assessment: Channel classification completed. "
                f"Channels: {channel_classification['Channel_Count']}, "
                f"Locations: {channel_classification['Available_Locations']}, "
                f"Axes: {channel_classification['Available_Axes']}",
                Logger.DEBUG
            )

            return channel_classification

        except Exception as e:
            Logger.log_message_static(f"Vibration-Assessment: Channel classification error: {e}", Logger.WARNING)
            return {'Channel_Count': 0, 'Available_Axes': [], 'Available_Locations': []}

    def extract_rpm_value(signal_data: Dict, channel_info: Dict) -> Optional[float]:
        """Extract RPM value from available signals."""
        try:
            rpm_signal_name = channel_info.get('RPM')
            if not rpm_signal_name:
                Logger.log_message_static("Vibration-Assessment: No RPM signal found", Logger.DEBUG)
                return None

            if rpm_signal_name not in signal_data:
                Logger.log_message_static(f"Vibration-Assessment: RPM signal '{rpm_signal_name}' not found in data",
                                          Logger.WARNING)
                return None

            time_arr, rpm_values = signal_data[rpm_signal_name]

            # Calculate mean RPM, excluding outliers
            rpm_median = np.median(rpm_values)
            rpm_std = np.std(rpm_values)

            # Filter outliers (within 3 standard deviations)
            valid_rpm = rpm_values[np.abs(rpm_values - rpm_median) <= 3 * rpm_std]

            if len(valid_rpm) == 0:
                Logger.log_message_static("Vibration-Assessment: No valid RPM values after outlier removal",
                                          Logger.WARNING)
                return None

            mean_rpm = np.mean(valid_rpm)

            # Validate RPM range (typical industrial machinery)
            if not (10 <= mean_rpm <= 10000):
                Logger.log_message_static(f"Vibration-Assessment: RPM value {mean_rpm:.1f} outside typical range",
                                          Logger.WARNING)

            Logger.log_message_static(f"Vibration-Assessment: Extracted RPM: {mean_rpm:.1f}", Logger.DEBUG)
            return float(mean_rpm)

        except Exception as e:
            Logger.log_message_static(f"Vibration-Assessment: RPM extraction error: {e}", Logger.WARNING)
            return None

    def analyze_individual_channels(
            signal_data: Dict,
            channel_info: Dict,
            rpm_value: Optional[float],
            machine_info: Optional[Dict],
            dialog
    ) -> Dict[str, Dict]:
        """Perform comprehensive analysis on each individual vibration channel."""
        try:
            channel_results = {}

            # Import analysis functions
            from vibration_metrics import calculate_vibration_metrics, calculate_vibration_severity
            from vibration_fft import calculate_vibration_fft
            from vibration_envelope import calculate_envelope_analysis

            # Analyze each channel
            for location in ['DE', 'NDE']:
                for axis in ['X', 'Y', 'Z']:
                    signal_name = channel_info.get(location, {}).get(axis)
                    if not signal_name or signal_name not in signal_data:
                        continue

                    try:
                        time_arr, values = signal_data[signal_name]
                        channel_key = f"{location}_{axis}"

                        Logger.log_message_static(f"Vibration-Assessment: Analyzing channel {channel_key}",
                                                  Logger.DEBUG)

                        # Basic vibration metrics
                        metrics = calculate_vibration_metrics(time_arr, values, dialog, f"Channel {channel_key}")
                        if metrics is None:
                            Logger.log_message_static(
                                f"Vibration-Assessment: Failed to calculate metrics for {channel_key}", Logger.WARNING)
                            continue

                        # Severity assessment (assuming velocity measurements in mm/s)
                        rms_value = metrics.get('RMS', 0)
                        severity = calculate_vibration_severity(rms_value, "mm/s", "II")  # Default to Class II

                        # FFT analysis
                        fft_results = calculate_vibration_fft(
                            time_arr, values, dialog, f"FFT {channel_key}",
                            rpm=rpm_value, machine_info=machine_info
                        )

                        # Envelope analysis
                        envelope_results = calculate_envelope_analysis(
                            time_arr, values, dialog, f"Envelope {channel_key}",
                            filter_type="adaptive"
                        )

                        # Compile channel results
                        channel_results[channel_key] = {
                            'Signal_Name': signal_name,
                            'Location': location,
                            'Axis': axis,
                            'Metrics': metrics,
                            'Severity': severity,
                            'FFT_Analysis': fft_results,
                            'Envelope_Analysis': envelope_results,
                            'Data_Quality': assess_channel_data_quality(time_arr, values),
                            'Channel_Status': determine_channel_status(metrics, severity, fft_results)
                        }

                    except Exception as channel_error:
                        Logger.log_message_static(
                            f"Vibration-Assessment: Error analyzing {channel_key}: {channel_error}", Logger.WARNING)
                        channel_results[f"{location}_{axis}"] = {
                            'Signal_Name': signal_name,
                            'Location': location,
                            'Axis': axis,
                            'Error': str(channel_error),
                            'Channel_Status': 'Error'
                        }

            # Analyze other signals
            for signal_name in channel_info.get('Other', {}):
                if signal_name in signal_data:
                    try:
                        time_arr, values = signal_data[signal_name]

                        metrics = calculate_vibration_metrics(time_arr, values, dialog, f"Signal {signal_name}")
                        if metrics:
                            channel_results[f"Other_{signal_name}"] = {
                                'Signal_Name': signal_name,
                                'Location': 'Unknown',
                                'Axis': 'Unknown',
                                'Metrics': metrics,
                                'Data_Quality': assess_channel_data_quality(time_arr, values),
                                'Channel_Status': 'Analyzed'
                            }

                    except Exception as other_error:
                        Logger.log_message_static(
                            f"Vibration-Assessment: Error analyzing other signal {signal_name}: {other_error}",
                            Logger.WARNING)

            Logger.log_message_static(
                f"Vibration-Assessment: Individual channel analysis completed. Channels analyzed: {len(channel_results)}",
                Logger.DEBUG)
            return channel_results

        except Exception as e:
            Logger.log_message_static(f"Vibration-Assessment: Error in individual channel analysis: {e}", Logger.ERROR)
            return {}

    def assess_channel_data_quality(time_arr: np.ndarray, values: np.ndarray) -> Dict[str, Any]:
        """Assess the quality of vibration data for a single channel."""
        try:
            # Basic data characteristics
            sample_count = len(values)
            duration = time_arr[-1] - time_arr[0] if len(time_arr) > 1 else 0
            sample_rate = safe_sample_rate(time_arr)

            # Data completeness
            non_zero_count = np.count_nonzero(values)
            zero_percentage = (sample_count - non_zero_count) / sample_count * 100

            # Signal-to-noise estimation
            signal_power = np.var(values)
            noise_estimate = np.var(np.diff(values))
            snr_estimate = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 0 else np.inf

            # Clipping detection
            max_val = np.max(np.abs(values))
            near_max_count = np.sum(np.abs(values) > 0.95 * max_val)
            clipping_percentage = near_max_count / sample_count * 100

            # Determine overall quality
            quality_score = 100

            if zero_percentage > 10:
                quality_score -= 30
            elif zero_percentage > 5:
                quality_score -= 15

            if snr_estimate < 20:
                quality_score -= 20
            elif snr_estimate < 40:
                quality_score -= 10

            if clipping_percentage > 1:
                quality_score -= 25
            elif clipping_percentage > 0.1:
                quality_score -= 10

            if sample_count < 5000:
                quality_score -= 15
            elif sample_count < 10000:
                quality_score -= 5

            # Quality level
            if quality_score >= 85:
                quality_level = "Excellent"
            elif quality_score >= 70:
                quality_level = "Good"
            elif quality_score >= 50:
                quality_level = "Fair"
            else:
                quality_level = "Poor"

            return {
                'Quality_Level': quality_level,
                'Quality_Score': max(0, quality_score),
                'Sample_Count': sample_count,
                'Duration_s': float(duration),
                'Sample_Rate_Hz': float(sample_rate),
                'Zero_Percentage': float(zero_percentage),
                'SNR_Estimate_dB': float(snr_estimate),
                'Clipping_Percentage': float(clipping_percentage),
                'Data_Range': f"{np.min(values):.6f} to {np.max(values):.6f}"
            }

        except Exception as e:
            Logger.log_message_static(f"Vibration-Assessment: Data quality assessment error: {e}", Logger.WARNING)
            return {'Quality_Level': 'Unknown', 'Quality_Score': 0}

    def determine_channel_status(metrics: Dict, severity: Dict, fft_results: Optional[Dict]) -> str:
        """Determine overall status of individual vibration channel."""
        try:
            # Check for errors or missing data
            if not metrics or not severity:
                return "Error"

            # Get severity level
            severity_level = severity.get('Severity Level', 'Unknown')

            # Get key metrics
            rms = metrics.get('RMS', 0)
            crest_factor = metrics.get('Crest Factor', 0)
            kurtosis = metrics.get('Kurtosis', 0)

            # Assess based on multiple factors
            if severity_level == 'A':
                status = "Good"
            elif severity_level == 'B':
                status = "Acceptable"
            elif severity_level == 'C':
                status = "Unsatisfactory"
            elif severity_level == 'D':
                status = "Unacceptable"
            else:
                # Fallback assessment based on metrics
                if rms < 1.8:
                    status =