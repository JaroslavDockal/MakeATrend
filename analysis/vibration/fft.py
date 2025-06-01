"""
Vibration FFT analysis for rotating machinery condition monitoring.

This module provides specialized FFT analysis functions for vibration signals,
including harmonic analysis, sidebands detection, and fault frequency identification.
Designed for multi-channel analysis of DE/NDE accelerometer measurements.

Key features:
- Multi-channel FFT analysis (X, Y, Z axes for DE/NDE)
- Harmonic analysis with RPM synchronization
- Sideband detection for gear/bearing faults
- Peak tracking and trending
- Order analysis capabilities
- Automatic fault frequency calculation
"""

import numpy as np
from scipy import signal as sc_signal
from scipy.fft import rfft, rfftfreq
from analysis.calculations.common import safe_prepare_signal, safe_sample_rate
from utils.logger import Logger


def calculate_vibration_fft(time_arr, values, dialog=None, title="Vibration FFT",
                            rpm=None, machine_info=None):
    """
    Perform comprehensive FFT analysis optimized for vibration signals.

    Analyzes frequency content with emphasis on rotating machinery characteristics
    including harmonics, sidebands, and fault frequencies. Automatically calculates
    theoretical fault frequencies if machine parameters are provided.

    Args:
        time_arr (np.ndarray): Time values array.
        values (np.ndarray): Vibration signal values (acceleration, velocity, or displacement).
        dialog (QWidget, optional): Parent dialog for user interaction.
        title (str, optional): Title for user dialogs.
        rpm (float, optional): Rotational speed in RPM for harmonic analysis.
        machine_info (dict, optional): Machine parameters for fault frequency calculation:
            - bearing_balls: Number of rolling elements
            - bearing_pitch_dia: Pitch diameter (mm)
            - bearing_ball_dia: Ball diameter (mm)
            - bearing_contact_angle: Contact angle (degrees, default 0)
            - gear_teeth: Number of gear teeth
            - gear_ratio: Gear ratio if applicable

    Returns:
        dict or None: Dictionary containing FFT analysis results:
            - Frequency Axis (Hz): Frequency values
            - Magnitude Spectrum: Single-sided magnitude spectrum
            - Magnitude dB: Magnitude in decibels
            - Phase Spectrum: Phase information
            - Peak Frequencies: Dominant frequency peaks
            - Peak Magnitudes: Corresponding peak amplitudes
            - Harmonic Analysis: Analysis of speed-related harmonics
            - Fault Frequencies: Theoretical fault frequencies
            - Sideband Analysis: Detection of modulation sidebands
            - Spectral Statistics: Statistical measures of spectrum

        Returns None if validation fails.

    Example:
        >>> import numpy as np
        >>> t = np.linspace(0, 10, 10000)
        >>> # Simulate motor with unbalance (1X) and bearing fault
        >>> vib = np.sin(2*np.pi*30*t) + 0.3*np.sin(2*np.pi*60*t) + 0.1*np.random.randn(10000)
        >>> machine_info = {'bearing_balls': 8, 'bearing_pitch_dia': 50, 'bearing_ball_dia': 8}
        >>> result = calculate_vibration_fft(t, vib, rpm=1800, machine_info=machine_info)
        >>> print(f"Dominant frequency: {result['Peak Frequencies'][0]:.1f} Hz")
        >>> print(f"BPFO frequency: {result['Fault Frequencies']['BPFO']:.1f} Hz")
    """
    Logger.log_message_static(f"Vibration-FFT: Starting FFT analysis for {title}", Logger.DEBUG)

    # Validate and prepare signal
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Vibration-FFT: Signal validation failed", Logger.WARNING)
        return None

    try:
        # Get sampling parameters
        sample_rate = safe_sample_rate(time_arr)
        if sample_rate <= 0:
            Logger.log_message_static("Vibration-FFT: Invalid sampling rate", Logger.ERROR)
            return None

        # Apply window function to reduce spectral leakage
        # Hanning window is standard for vibration analysis
        window = np.hanning(len(processed_values))
        windowed_signal = processed_values * window

        # Calculate window correction factor
        window_correction = len(window) / np.sum(window)

        # Perform FFT
        n = len(windowed_signal)
        yf = rfft(windowed_signal)
        xf = rfftfreq(n, 1 / sample_rate)

        # Calculate single-sided magnitude spectrum with proper scaling
        magnitude = np.abs(yf) * window_correction / n
        magnitude[1:-1] *= 2  # Double for single-sided (except DC and Nyquist)

        # Calculate phase spectrum
        phase = np.angle(yf)

        # Convert magnitude to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-12)  # Add small value to avoid log(0)

        # Find spectral peaks
        peak_analysis = find_spectral_peaks(xf, magnitude, min_peak_height=0.01)

        # Perform harmonic analysis if RPM provided
        harmonic_analysis = None
        if rpm and rpm > 0:
            harmonic_analysis = analyze_harmonics(xf, magnitude, rpm, max_harmonics=20)

        # Calculate theoretical fault frequencies
        fault_frequencies = None
        if rpm and machine_info:
            fault_frequencies = calculate_fault_frequencies(rpm, machine_info)

        # Analyze sidebands (modulation detection)
        sideband_analysis = analyze_sidebands(xf, magnitude, peak_analysis['frequencies'],
                                              sample_rate, rpm)

        # Calculate spectral statistics
        spectral_stats = calculate_spectral_statistics(xf, magnitude, magnitude_db)

        # Perform order analysis if RPM available
        order_analysis = None
        if rpm and rpm > 0:
            order_analysis = perform_order_analysis(xf, magnitude, rpm)

        # Detect resonances and anti-resonances
        resonance_analysis = detect_resonances(xf, magnitude, sample_rate)

        # Build comprehensive results
        results = {
            # Core FFT data
            "Frequency Axis (Hz)": xf,
            "Magnitude Spectrum": magnitude,
            "Magnitude dB": magnitude_db,
            "Phase Spectrum": phase,

            # Peak analysis
            "Peak Frequencies": peak_analysis['frequencies'],
            "Peak Magnitudes": peak_analysis['magnitudes'],
            "Peak Indices": peak_analysis['indices'],
            "Number of Peaks": len(peak_analysis['frequencies']),

            # Advanced analysis
            "Harmonic Analysis": harmonic_analysis,
            "Fault Frequencies": fault_frequencies,
            "Sideband Analysis": sideband_analysis,
            "Order Analysis": order_analysis,
            "Resonance Analysis": resonance_analysis,

            # Spectral characteristics
            "Spectral Statistics": spectral_stats,
            "Frequency Resolution (Hz)": float(sample_rate / n),
            "Maximum Frequency (Hz)": float(np.max(xf)),
            "RMS from Spectrum": float(np.sqrt(np.sum(magnitude ** 2))),

            # Analysis parameters
            "Sample Rate (Hz)": float(sample_rate),
            "Sample Count": n,
            "Window Type": "Hanning",
            "Window Correction Factor": float(window_correction),
            "RPM": rpm,
            "Machine Info": machine_info
        }

        Logger.log_message_static(
            f"Vibration-FFT: FFT analysis completed. "
            f"Peaks found: {len(peak_analysis['frequencies'])}, "
            f"Freq resolution: {sample_rate / n:.2f} Hz, "
            f"Max magnitude: {np.max(magnitude):.6f}",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Vibration-FFT: Error in FFT analysis: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Vibration-FFT: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None


def find_spectral_peaks(frequencies, magnitude, min_peak_height=0.01, min_distance_hz=1.0):
    """
    Find significant peaks in the magnitude spectrum.

    Uses adaptive thresholding and distance requirements optimized for vibration analysis.
    """
    try:
        # Calculate adaptive threshold
        noise_floor = np.percentile(magnitude, 75)  # Estimate noise floor
        threshold = max(min_peak_height, noise_floor * 3)  # At least 3x noise floor

        # Convert minimum distance to samples
        df = frequencies[1] - frequencies[0]  # Frequency resolution
        min_distance_samples = max(1, int(min_distance_hz / df))

        # Find peaks
        peaks, properties = sc_signal.find_peaks(
            magnitude,
            height=threshold,
            distance=min_distance_samples,
            prominence=threshold * 0.5  # Minimum prominence
        )

        if len(peaks) == 0:
            return {
                'frequencies': np.array([]),
                'magnitudes': np.array([]),
                'indices': np.array([])
            }

        # Sort peaks by magnitude (descending)
        peak_magnitudes = magnitude[peaks]
        sort_indices = np.argsort(peak_magnitudes)[::-1]

        # Limit to top 50 peaks to avoid clutter
        max_peaks = 50
        if len(sort_indices) > max_peaks:
            sort_indices = sort_indices[:max_peaks]

        sorted_peaks = peaks[sort_indices]
        sorted_magnitudes = peak_magnitudes[sort_indices]
        sorted_frequencies = frequencies[sorted_peaks]

        return {
            'frequencies': sorted_frequencies,
            'magnitudes': sorted_magnitudes,
            'indices': sorted_peaks
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-FFT: Error finding peaks: {e}", Logger.WARNING)
        return {
            'frequencies': np.array([]),
            'magnitudes': np.array([]),
            'indices': np.array([])
        }


def analyze_harmonics(frequencies, magnitude, rpm, max_harmonics=20):
    """
    Analyze harmonic content relative to fundamental frequency (1X RPM).

    Identifies and quantifies harmonics of the fundamental frequency,
    which are critical for diagnosing specific machine faults.
    """
    try:
        fundamental_freq = rpm / 60.0  # Convert RPM to Hz
        df = frequencies[1] - frequencies[0]  # Frequency resolution

        harmonic_analysis = {
            'Fundamental Frequency (Hz)': fundamental_freq,
            'Harmonics': {},
            'Total Harmonic Content': 0.0,
            'Dominant Harmonic': 1,
            'Harmonic Ratio': 0.0
        }

        harmonic_magnitudes = []
        harmonic_frequencies = []

        for h in range(1, max_harmonics + 1):
            harmonic_freq = h * fundamental_freq

            # Skip if harmonic exceeds Nyquist frequency
            if harmonic_freq >= frequencies[-1]:
                break

            # Find closest frequency bin
            freq_idx = np.argmin(np.abs(frequencies - harmonic_freq))
            freq_error = abs(frequencies[freq_idx] - harmonic_freq)

            # Only consider if frequency error is within resolution
            if freq_error <= df * 2:  # Allow 2 bins tolerance
                harmonic_mag = magnitude[freq_idx]
                harmonic_magnitudes.append(harmonic_mag)
                harmonic_frequencies.append(harmonic_freq)

                harmonic_analysis['Harmonics'][f'{h}X'] = {
                    'Frequency (Hz)': float(harmonic_freq),
                    'Magnitude': float(harmonic_mag),
                    'Magnitude dB': float(20 * np.log10(harmonic_mag + 1e-12)),
                    'Frequency Error (Hz)': float(freq_error)
                }

        if harmonic_magnitudes:
            # Calculate total harmonic content
            fundamental_mag = harmonic_magnitudes[0] if harmonic_magnitudes else 0
            total_harmonic = np.sum(harmonic_magnitudes[1:]) if len(harmonic_magnitudes) > 1 else 0

            harmonic_analysis['Total Harmonic Content'] = float(total_harmonic)
            harmonic_analysis['Fundamental Magnitude'] = float(fundamental_mag)

            # Find dominant harmonic
            if len(harmonic_magnitudes) > 1:
                dominant_idx = np.argmax(harmonic_magnitudes)
                harmonic_analysis['Dominant Harmonic'] = dominant_idx + 1

                # Calculate harmonic ratio (harmonics/fundamental)
                if fundamental_mag > 0:
                    harmonic_analysis['Harmonic Ratio'] = float(total_harmonic / fundamental_mag)

        return harmonic_analysis

    except Exception as e:
        Logger.log_message_static(f"Vibration-FFT: Error in harmonic analysis: {e}", Logger.WARNING)
        return None


def calculate_fault_frequencies(rpm, machine_info):
    """
    Calculate theoretical fault frequencies for common machine elements.

    Computes bearing fault frequencies (BPFO, BPFI, BSF, FTF) and gear mesh
    frequencies based on machine geometry and operating speed.
    """
    try:
        shaft_freq = rpm / 60.0  # Hz

        fault_freqs = {
            'Shaft Frequency (1X)': shaft_freq,
            'Shaft 2X': 2 * shaft_freq,
            'Shaft 3X': 3 * shaft_freq
        }

        # Bearing fault frequencies
        if all(key in machine_info for key in ['bearing_balls', 'bearing_pitch_dia', 'bearing_ball_dia']):
            nb = machine_info['bearing_balls']
            pd = machine_info['bearing_pitch_dia']
            bd = machine_info['bearing_ball_dia']
            alpha = np.radians(machine_info.get('bearing_contact_angle', 0))

            # Standard bearing fault frequency calculations
            bpfo = (nb / 2) * shaft_freq * (1 - (bd / pd) * np.cos(alpha))
            bpfi = (nb / 2) * shaft_freq * (1 + (bd / pd) * np.cos(alpha))
            bsf = (pd / (2 * bd)) * shaft_freq * (1 - (bd / pd) ** 2 * np.cos(alpha) ** 2)
            ftf = (1 / 2) * shaft_freq * (1 - (bd / pd) * np.cos(alpha))

            fault_freqs.update({
                'BPFO (Ball Pass Freq Outer)': bpfo,
                'BPFI (Ball Pass Freq Inner)': bpfi,
                'BSF (Ball Spin Frequency)': bsf,
                'FTF (Fundamental Train Freq)': ftf,
                'BPFO 2X': 2 * bpfo,
                'BPFI 2X': 2 * bpfi
            })

        # Gear mesh frequency
        if 'gear_teeth' in machine_info:
            teeth = machine_info['gear_teeth']
            gmf = shaft_freq * teeth
            fault_freqs.update({
                'GMF (Gear Mesh Frequency)': gmf,
                'GMF ± 1X Sidebands': [gmf - shaft_freq, gmf + shaft_freq]
            })

        return fault_freqs

    except Exception as e:
        Logger.log_message_static(f"Vibration-FFT: Error calculating fault frequencies: {e}", Logger.WARNING)
        return None


def analyze_sidebands(frequencies, magnitude, peak_frequencies, sample_rate, rpm=None):
    """
    Detect and analyze sidebands around major peaks.

    Sidebands indicate modulation phenomena often associated with
    gear faults, bearing faults, and electrical issues.
    """
    try:
        if len(peak_frequencies) == 0:
            return None

        df = frequencies[1] - frequencies[0]
        sideband_analysis = {
            'Detected Sidebands': [],
            'Modulation Summary': {}
        }

        # Analysis parameters
        max_sidebands = 5  # Maximum sidebands to search each side
        min_sideband_ratio = 0.1  # Minimum ratio to carrier

        for i, carrier_freq in enumerate(peak_frequencies[:10]):  # Analyze top 10 peaks
            carrier_idx = np.argmin(np.abs(frequencies - carrier_freq))
            carrier_mag = magnitude[carrier_idx]

            sidebands_found = []

            # Search for sidebands around carrier
            if rpm and rpm > 0:
                modulation_freq = rpm / 60.0  # 1X modulation

                for sb in range(1, max_sidebands + 1):
                    # Lower sideband
                    lower_freq = carrier_freq - sb * modulation_freq
                    if lower_freq > 0:
                        lower_idx = np.argmin(np.abs(frequencies - lower_freq))
                        if abs(frequencies[lower_idx] - lower_freq) <= df * 2:
                            lower_mag = magnitude[lower_idx]
                            if lower_mag > carrier_mag * min_sideband_ratio:
                                sidebands_found.append({
                                    'Type': f'Lower {sb}X',
                                    'Frequency': float(lower_freq),
                                    'Magnitude': float(lower_mag),
                                    'Ratio to Carrier': float(lower_mag / carrier_mag)
                                })

                    # Upper sideband
                    upper_freq = carrier_freq + sb * modulation_freq
                    if upper_freq < frequencies[-1]:
                        upper_idx = np.argmin(np.abs(frequencies - upper_freq))
                        if abs(frequencies[upper_idx] - upper_freq) <= df * 2:
                            upper_mag = magnitude[upper_idx]
                            if upper_mag > carrier_mag * min_sideband_ratio:
                                sidebands_found.append({
                                    'Type': f'Upper {sb}X',
                                    'Frequency': float(upper_freq),
                                    'Magnitude': float(upper_mag),
                                    'Ratio to Carrier': float(upper_mag / carrier_mag)
                                })

            if sidebands_found:
                sideband_analysis['Detected Sidebands'].append({
                    'Carrier Frequency': float(carrier_freq),
                    'Carrier Magnitude': float(carrier_mag),
                    'Sidebands': sidebands_found
                })

        # Generate modulation summary
        total_sidebands = sum(len(sb['Sidebands']) for sb in sideband_analysis['Detected Sidebands'])
        sideband_analysis['Modulation Summary'] = {
            'Total Carriers with Sidebands': len(sideband_analysis['Detected Sidebands']),
            'Total Sidebands Detected': total_sidebands,
            'Modulation Present': total_sidebands > 0
        }

        return sideband_analysis

    except Exception as e:
        Logger.log_message_static(f"Vibration-FFT: Error in sideband analysis: {e}", Logger.WARNING)
        return None


def calculate_spectral_statistics(frequencies, magnitude, magnitude_db):
    """Calculate statistical measures of the frequency spectrum."""
    try:
        # Basic statistics
        mean_freq = np.average(frequencies, weights=magnitude)
        rms_freq = np.sqrt(np.average(frequencies ** 2, weights=magnitude))

        # Spectral moments
        m0 = np.trapz(magnitude, frequencies)  # 0th moment (total power)
        m1 = np.trapz(frequencies * magnitude, frequencies)  # 1st moment
        m2 = np.trapz(frequencies ** 2 * magnitude, frequencies)  # 2nd moment

        # Derived parameters
        centroid_freq = m1 / m0 if m0 > 0 else 0
        bandwidth = np.sqrt(m2 / m0 - centroid_freq ** 2) if m0 > 0 else 0

        # Frequency range analysis
        above_noise = magnitude > np.percentile(magnitude, 90)
        active_freq_range = frequencies[above_noise]
        freq_span = np.max(active_freq_range) - np.min(active_freq_range) if len(active_freq_range) > 0 else 0

        return {
            'Mean Frequency (Hz)': float(mean_freq),
            'RMS Frequency (Hz)': float(rms_freq),
            'Centroid Frequency (Hz)': float(centroid_freq),
            'Spectral Bandwidth (Hz)': float(bandwidth),
            'Active Frequency Span (Hz)': float(freq_span),
            'Peak Magnitude': float(np.max(magnitude)),
            'Peak Magnitude dB': float(np.max(magnitude_db)),
            'Average Magnitude': float(np.mean(magnitude)),
            'Spectral Flatness': float(np.exp(np.mean(np.log(magnitude + 1e-12))) / np.mean(magnitude)),
            'Spectral Rolloff (95%)': float(
                frequencies[np.where(np.cumsum(magnitude) >= 0.95 * np.sum(magnitude))[0][0]]) if len(
                frequencies) > 0 else 0
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-FFT: Error calculating spectral statistics: {e}", Logger.WARNING)
        return {}


def perform_order_analysis(frequencies, magnitude, rpm):
    """
    Convert frequency spectrum to order spectrum.

    Order analysis normalizes frequencies by running speed, making it easier
    to identify speed-related components across different operating conditions.
    """
    try:
        fundamental_freq = rpm / 60.0
        orders = frequencies / fundamental_freq

        # Find integer orders (within tolerance)
        integer_orders = []
        order_magnitudes = []

        for order in range(1, 21):  # Analyze first 20 orders
            # Find closest frequency bin to this order
            target_freq = order * fundamental_freq
            if target_freq <= frequencies[-1]:
                idx = np.argmin(np.abs(frequencies - target_freq))
                freq_error = abs(frequencies[idx] - target_freq)

                # Allow small frequency error
                if freq_error <= fundamental_freq * 0.1:  # 10% tolerance
                    integer_orders.append(order)
                    order_magnitudes.append(magnitude[idx])

        return {
            'Order Axis': orders,
            'Order Spectrum': magnitude,
            'Integer Orders': integer_orders,
            'Integer Order Magnitudes': order_magnitudes,
            'Fundamental Order': 1,
            'Running Speed (RPM)': rpm
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-FFT: Error in order analysis: {e}", Logger.WARNING)
        return None


def detect_resonances(frequencies, magnitude, sample_rate):
    """
    Detect potential structural resonances in the spectrum.

    Identifies sharp peaks that could indicate natural frequencies
    of the machine structure.
    """
    try:
        # Look for sharp, isolated peaks that could be resonances
        # These typically have high Q-factor (narrow bandwidth)

        # Find peaks with high prominence
        peaks, properties = sc_signal.find_peaks(
            magnitude,
            prominence=np.max(magnitude) * 0.05,  # At least 5% of max
            width=1  # Narrow peaks
        )

        if len(peaks) == 0:
            return None

        # Calculate Q-factor for each peak (approximation)
        resonances = []
        for peak_idx in peaks:
            peak_freq = frequencies[peak_idx]
            peak_mag = magnitude[peak_idx]

            # Find -3dB points (half power)
            half_power = peak_mag / np.sqrt(2)

            # Search left and right for half-power points
            left_idx = peak_idx
            while left_idx > 0 and magnitude[left_idx] > half_power:
                left_idx -= 1

            right_idx = peak_idx
            while right_idx < len(magnitude) - 1 and magnitude[right_idx] > half_power:
                right_idx += 1

            # Calculate bandwidth and Q-factor
            bandwidth = frequencies[right_idx] - frequencies[left_idx]
            q_factor = peak_freq / bandwidth if bandwidth > 0 else np.inf

            # Consider as resonance if Q > 5 (somewhat sharp)
            if q_factor > 5 and peak_freq > 10:  # Above 10 Hz
                resonances.append({
                    'Frequency (Hz)': float(peak_freq),
                    'Magnitude': float(peak_mag),
                    'Q-Factor': float(q_factor),
                    'Bandwidth (Hz)': float(bandwidth)
                })

        # Sort by Q-factor (highest first)
        resonances.sort(key=lambda x: x['Q-Factor'], reverse=True)

        return {
            'Detected Resonances': resonances[:10],  # Top 10
            'Number of Resonances': len(resonances)
        }

    except Exception as e:
        Logger.log_message_static(f"Vibration-FFT: Error detecting resonances: {e}", Logger.WARNING)
        return None