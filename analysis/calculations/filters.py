"""
Signal filtering using IIR and FIR filters.

This module provides comprehensive signal filtering capabilities:
- IIR (Infinite Impulse Response) filters using Butterworth design
- FIR (Finite Impulse Response) filters using windowed design
- Multiple filter types: lowpass, highpass, bandpass, bandstop
- Filter parameter optimization and validation
- Filter response analysis

Both IIR and FIR filters have different characteristics:
- IIR: More efficient, sharp transitions, potential phase distortion
- FIR: Linear phase, stable, requires more computation
"""

import numpy as np
import scipy.signal as sc_signal
from PySide6.QtWidgets import QMessageBox

from .common import safe_prepare_signal, safe_sample_rate, validate_analysis_inputs
from utils.logger import Logger


def calculate_iir_filter(time_arr, values, filter_type="lowpass", cutoff_freq=1.0, order=4, dialog=None,
                         title="IIR Filter"):
    """
    Apply an IIR (Infinite Impulse Response) Butterworth filter to the signal.

    IIR filters provide efficient filtering with sharp frequency transitions
    but may introduce phase distortion. Butterworth filters provide maximally
    flat passband response.

    Args:
        time_arr (np.ndarray): Time values for sampling rate calculation.
        values (np.ndarray): Signal values to filter.
        filter_type (str, optional): Type of filter. Options:
            - "lowpass": Passes low frequencies, attenuates high frequencies
            - "highpass": Passes high frequencies, attenuates low frequencies
            - "bandpass": Passes frequencies in specified band
            - "bandstop": Attenuates frequencies in specified band
            Defaults to "lowpass".
        cutoff_freq (float or tuple, optional): Cutoff frequency in Hz.
            For lowpass/highpass: single frequency value
            For bandpass/bandstop: tuple of (low_freq, high_freq)
            Defaults to 1.0.
        order (int, optional): Filter order (higher = sharper transition). Defaults to 4.
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "IIR Filter".

    Returns:
        dict or None: Dictionary containing filtering results:
            - Filtered Signal: The filtered signal values
            - Original Signal: Original input signal
            - Filter Coefficients: (numerator, denominator) coefficients
            - Filter Type: Type of filter applied
            - Cutoff Frequency (Hz): Cutoff frequency used
            - Filter Order: Order of the filter
            - Sample Rate (Hz): Sampling rate
            - Filter Response: Frequency response information
            - Phase Response: Phase response information

        Returns None if validation fails or filtering error occurs.

    Example:
        >>> t = np.linspace(0, 1, 1000)
        >>> signal = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*50*t)  # 10Hz + 50Hz
        >>> result = calculate_iir_filter(t, signal, "lowpass", cutoff_freq=30, order=4)
        >>> filtered = result['Filtered Signal']  # Should contain mostly 10Hz component
    """
    Logger.log_message_static(f"Calculations-Filters: Starting IIR {filter_type} filter design", Logger.DEBUG)

    # Validate inputs
    is_valid, error_msg, sample_rate = validate_analysis_inputs(time_arr, values, min_length=4,
                                                                require_positive_sample_rate=True)
    if not is_valid:
        Logger.log_message_static(f"Calculations-Filters: Input validation failed: {error_msg}", Logger.ERROR)
        if dialog:
            QMessageBox.warning(dialog, title, f"IIR Filter Error:\n{error_msg}")
        return None

    # Prepare signal
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Calculations-Filters: Signal validation failed", Logger.WARNING)
        return None

    try:
        nyquist = 0.5 * sample_rate

        # Validate and normalize cutoff frequencies
        if filter_type in ["lowpass", "highpass"]:
            if isinstance(cutoff_freq, (list, tuple)):
                Logger.log_message_static(
                    "Calculations-Filters: Single cutoff expected for lowpass/highpass, using first value",
                    Logger.WARNING)
                cutoff_freq = cutoff_freq[0]

            if cutoff_freq >= nyquist:
                cutoff_freq = nyquist * 0.95
                Logger.log_message_static(
                    f"Calculations-Filters: Cutoff frequency adjusted to {cutoff_freq:.2f} Hz (below Nyquist)",
                    Logger.WARNING)
            elif cutoff_freq <= 0:
                cutoff_freq = nyquist * 0.01
                Logger.log_message_static(
                    f"Calculations-Filters: Cutoff frequency adjusted to {cutoff_freq:.2f} Hz (above zero)",
                    Logger.WARNING)

            normalized_cutoff = cutoff_freq / nyquist

        elif filter_type in ["bandpass", "bandstop"]:
            if not isinstance(cutoff_freq, (list, tuple)) or len(cutoff_freq) != 2:
                Logger.log_message_static("Calculations-Filters: Two cutoff frequencies required for bandpass/bandstop",
                                          Logger.ERROR)
                if dialog:
                    QMessageBox.warning(dialog, title,
                                        "Bandpass and bandstop filters require two cutoff frequencies [low, high].")
                return None

            low_freq, high_freq = cutoff_freq

            # Validate frequency order
            if low_freq >= high_freq:
                Logger.log_message_static("Calculations-Filters: Low frequency >= high frequency, swapping",
                                          Logger.WARNING)
                low_freq, high_freq = high_freq, low_freq

            # Validate frequency limits
            if high_freq >= nyquist:
                high_freq = nyquist * 0.95
                Logger.log_message_static(f"Calculations-Filters: High frequency adjusted to {high_freq:.2f} Hz",
                                          Logger.WARNING)
            if low_freq <= 0:
                low_freq = nyquist * 0.01
                Logger.log_message_static(f"Calculations-Filters: Low frequency adjusted to {low_freq:.2f} Hz",
                                          Logger.WARNING)

            # Ensure minimum bandwidth
            # Ensure minimum bandwidth
            min_bandwidth = nyquist * 0.01  # 1% of Nyquist frequency
            if (high_freq - low_freq) < min_bandwidth:
                center = (low_freq + high_freq) / 2
                low_freq = center - min_bandwidth / 2
                high_freq = center + min_bandwidth / 2
                Logger.log_message_static(
                    f"Calculations-Filters: Bandwidth adjusted to [{low_freq:.2f}, {high_freq:.2f}] Hz", Logger.WARNING)

            normalized_cutoff = [low_freq / nyquist, high_freq / nyquist]
            cutoff_freq = (low_freq, high_freq)  # Update for results

        else:
            Logger.log_message_static(f"Calculations-Filters: Unsupported filter type: {filter_type}", Logger.ERROR)
            if dialog:
                QMessageBox.warning(dialog, title, f"Unsupported filter type: {filter_type}")
            return None

        # Validate filter order
        if order < 1:
            order = 1
            Logger.log_message_static("Calculations-Filters: Filter order adjusted to minimum value of 1",
                                      Logger.WARNING)
        elif order > 20:
            order = 20
            Logger.log_message_static("Calculations-Filters: Filter order limited to maximum value of 20",
                                      Logger.WARNING)

        Logger.log_message_static(
            f"Calculations-Filters: IIR filter parameters - "
            f"Type={filter_type}, Order={order}, Cutoff={cutoff_freq}, Fs={sample_rate:.1f}Hz",
            Logger.DEBUG
        )

        # Design Butterworth filter
        try:
            b, a = sc_signal.butter(order, normalized_cutoff, btype=filter_type, analog=False)
        except Exception as design_error:
            Logger.log_message_static(f"Calculations-Filters: Filter design failed: {design_error}", Logger.ERROR)
            if dialog:
                QMessageBox.critical(dialog, title, f"Filter design failed:\n{str(design_error)}")
            return None

        # Apply filter using zero-phase filtering (filtfilt) to avoid phase distortion
        try:
            filtered_signal = sc_signal.filtfilt(b, a, processed_values)
        except Exception as filter_error:
            Logger.log_message_static(f"Calculations-Filters: Filter application failed: {filter_error}", Logger.ERROR)
            if dialog:
                QMessageBox.critical(dialog, title, f"Filter application failed:\n{str(filter_error)}")
            return None

        # Calculate filter frequency response for analysis
        try:
            w, h = sc_signal.freqz(b, a, worN=2048, fs=sample_rate)
            magnitude_response = np.abs(h)
            phase_response = np.angle(h)
            magnitude_db = 20 * np.log10(magnitude_response + 1e-12)

            # Find -3dB bandwidth
            max_magnitude_db = np.max(magnitude_db)
            cutoff_3db_indices = np.where(magnitude_db >= (max_magnitude_db - 3))[0]

            if len(cutoff_3db_indices) > 0:
                bandwidth_3db = w[cutoff_3db_indices[-1]] - w[cutoff_3db_indices[0]]
            else:
                bandwidth_3db = 0.0

        except Exception as response_error:
            Logger.log_message_static(f"Calculations-Filters: Filter response calculation failed: {response_error}",
                                      Logger.WARNING)
            w = np.array([0, sample_rate / 2])
            magnitude_response = np.array([1, 1])
            phase_response = np.array([0, 0])
            magnitude_db = np.array([0, 0])
            bandwidth_3db = 0.0

        # Calculate filter performance metrics
        original_energy = np.sum(processed_values ** 2)
        filtered_energy = np.sum(filtered_signal ** 2)
        energy_ratio = filtered_energy / original_energy if original_energy > 0 else 0.0

        # RMS comparison
        original_rms = np.sqrt(np.mean(processed_values ** 2))
        filtered_rms = np.sqrt(np.mean(filtered_signal ** 2))
        rms_ratio = filtered_rms / original_rms if original_rms > 0 else 0.0

        # Build comprehensive results dictionary
        results = {
            # Filtered data
            "Filtered Signal": filtered_signal,
            "Original Signal": processed_values,

            # Filter design parameters
            "Filter Coefficients": (b, a),
            "Filter Type": filter_type,
            "Cutoff Frequency (Hz)": cutoff_freq,
            "Filter Order": order,
            "Sample Rate (Hz)": float(sample_rate),
            "Nyquist Frequency (Hz)": float(nyquist),

            # Filter response data
            "Frequency Response (Hz)": w,
            "Magnitude Response": magnitude_response,
            "Magnitude Response (dB)": magnitude_db,
            "Phase Response (rad)": phase_response,
            "Phase Response (deg)": np.degrees(phase_response),

            # Performance metrics
            "3dB Bandwidth (Hz)": float(bandwidth_3db),
            "Energy Ratio (Filtered/Original)": float(energy_ratio),
            "RMS Ratio (Filtered/Original)": float(rms_ratio),
            "Original RMS": float(original_rms),
            "Filtered RMS": float(filtered_rms),

            # Filter characteristics
            "Filter Family": "Butterworth",
            "Zero-Phase Applied": True,
            "Passband Ripple (dB)": 0.0,  # Butterworth has no ripple
            "Stopband Attenuation (dB)": float(6.02 * order)  # Theoretical for Butterworth
        }

        Logger.log_message_static(
            f"Calculations-Filters: IIR filtering completed. "
            f"Energy_ratio={energy_ratio:.3f}, RMS_ratio={rms_ratio:.3f}, "
            f"3dB_BW={bandwidth_3db:.2f}Hz",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Filters: Error in IIR filtering: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Calculations-Filters: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None


def calculate_fir_filter(time_arr, values, filter_type="lowpass", cutoff_freq=1.0, numtaps=101, window="hamming",
                         dialog=None, title="FIR Filter"):
    """
    Apply an FIR (Finite Impulse Response) filter to the signal using windowed design.

    FIR filters provide linear phase response (no phase distortion) and are always stable,
    but require more computation than IIR filters. The windowed design method provides
    good control over filter characteristics.

    Args:
        time_arr (np.ndarray): Time values for sampling rate calculation.
        values (np.ndarray): Signal values to filter.
        filter_type (str, optional): Type of filter. Options:
            - "lowpass": Passes low frequencies, attenuates high frequencies
            - "highpass": Passes high frequencies, attenuates low frequencies
            - "bandpass": Passes frequencies in specified band
            - "bandstop": Attenuates frequencies in specified band
            Defaults to "lowpass".
        cutoff_freq (float or tuple, optional): Cutoff frequency in Hz.
            For lowpass/highpass: single frequency value
            For bandpass/bandstop: tuple of (low_freq, high_freq)
            Defaults to 1.0.
        numtaps (int, optional): Number of filter taps (filter length).
            Higher values give sharper transitions but more computation.
            Must be odd for some filter types. Defaults to 101.
        window (str, optional): Window function for filter design. Options:
            - "hamming": Good general purpose (default)
            - "hann": Similar to Hamming with better stopband
            - "blackman": Excellent stopband attenuation
            - "bartlett": Triangular window
            - "boxcar": Rectangular window (not recommended)
            - "kaiser": Adjustable window (requires beta parameter)
            Defaults to "hamming".
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "FIR Filter".

    Returns:
        dict or None: Dictionary containing filtering results:
            - Filtered Signal: The filtered signal values
            - Original Signal: Original input signal
            - Filter Taps: FIR filter coefficients
            - Filter Type: Type of filter applied
            - Cutoff Frequency (Hz): Cutoff frequency used
            - Number of Taps: Filter length
            - Window Function: Window used in design
            - Sample Rate (Hz): Sampling rate
            - Filter Response: Frequency response information
            - Group Delay: Constant group delay of FIR filter

        Returns None if validation fails or filtering error occurs.

    Example:
        >>> t = np.linspace(0, 2, 2000)
        >>> noise = 0.1 * np.random.randn(2000)
        >>> signal = np.sin(2*np.pi*5*t) + noise  # 5Hz signal + noise
        >>> result = calculate_fir_filter(t, signal, "lowpass", cutoff_freq=10, numtaps=51)
        >>> filtered = result['Filtered Signal']  # Should have reduced noise
    """
    Logger.log_message_static(f"Calculations-Filters: Starting FIR {filter_type} filter design", Logger.DEBUG)

    # Validate inputs
    is_valid, error_msg, sample_rate = validate_analysis_inputs(time_arr, values, min_length=4,
                                                                require_positive_sample_rate=True)
    if not is_valid:
        Logger.log_message_static(f"Calculations-Filters: Input validation failed: {error_msg}", Logger.ERROR)
        if dialog:
            QMessageBox.warning(dialog, title, f"FIR Filter Error:\n{error_msg}")
        return None

    # Prepare signal
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Calculations-Filters: Signal validation failed", Logger.WARNING)
        return None

    try:
        nyquist = 0.5 * sample_rate

        # Validate number of taps
        if numtaps < 3:
            numtaps = 3
            Logger.log_message_static("Calculations-Filters: Number of taps adjusted to minimum value of 3",
                                      Logger.WARNING)
        elif numtaps > len(processed_values) // 2:
            numtaps = len(processed_values) // 2
            Logger.log_message_static(f"Calculations-Filters: Number of taps limited to {numtaps} (half signal length)",
                                      Logger.WARNING)

        # Ensure odd number of taps for certain filter types
        if filter_type in ["highpass", "bandstop"] and numtaps % 2 == 0:
            numtaps += 1
            Logger.log_message_static(
                f"Calculations-Filters: Adjusted to odd number of taps ({numtaps}) for {filter_type} filter",
                Logger.INFO)

        # Validate and normalize cutoff frequencies
        if filter_type in ["lowpass", "highpass"]:
            if isinstance(cutoff_freq, (list, tuple)):
                Logger.log_message_static(
                    "Calculations-Filters: Single cutoff expected for lowpass/highpass, using first value",
                    Logger.WARNING)
                cutoff_freq = cutoff_freq[0]

            if cutoff_freq >= nyquist:
                cutoff_freq = nyquist * 0.95
                Logger.log_message_static(f"Calculations-Filters: Cutoff frequency adjusted to {cutoff_freq:.2f} Hz",
                                          Logger.WARNING)
            elif cutoff_freq <= 0:
                cutoff_freq = nyquist * 0.01
                Logger.log_message_static(f"Calculations-Filters: Cutoff frequency adjusted to {cutoff_freq:.2f} Hz",
                                          Logger.WARNING)

            normalized_cutoff = cutoff_freq / nyquist
            pass_zero = (filter_type == "lowpass")  # For firwin

        elif filter_type in ["bandpass", "bandstop"]:
            if not isinstance(cutoff_freq, (list, tuple)) or len(cutoff_freq) != 2:
                Logger.log_message_static("Calculations-Filters: Two cutoff frequencies required for bandpass/bandstop",
                                          Logger.ERROR)
                if dialog:
                    QMessageBox.warning(dialog, title,
                                        "Bandpass and bandstop filters require two cutoff frequencies [low, high].")
                return None

            low_freq, high_freq = cutoff_freq

            # Validate frequency order
            if low_freq >= high_freq:
                Logger.log_message_static("Calculations-Filters: Low frequency >= high frequency, swapping",
                                          Logger.WARNING)
                low_freq, high_freq = high_freq, low_freq

            # Validate frequency limits
            if high_freq >= nyquist:
                high_freq = nyquist * 0.95
                Logger.log_message_static(f"Calculations-Filters: High frequency adjusted to {high_freq:.2f} Hz",
                                          Logger.WARNING)
            if low_freq <= 0:
                low_freq = nyquist * 0.01
                Logger.log_message_static(f"Calculations-Filters: Low frequency adjusted to {low_freq:.2f} Hz",
                                          Logger.WARNING)

            # Ensure minimum bandwidth
            min_bandwidth = nyquist * 0.01
            if (high_freq - low_freq) < min_bandwidth:
                center = (low_freq + high_freq) / 2
                low_freq = center - min_bandwidth / 2
                high_freq = center + min_bandwidth / 2
                Logger.log_message_static(
                    f"Calculations-Filters: Bandwidth adjusted to [{low_freq:.2f}, {high_freq:.2f}] Hz", Logger.WARNING)

            normalized_cutoff = [low_freq / nyquist, high_freq / nyquist]
            pass_zero = (filter_type == "bandstop")  # For firwin
            cutoff_freq = (low_freq, high_freq)  # Update for results

        else:
            Logger.log_message_static(f"Calculations-Filters: Unsupported filter type: {filter_type}", Logger.ERROR)
            if dialog:
                QMessageBox.warning(dialog, title, f"Unsupported filter type: {filter_type}")
            return None

        # Validate window function
        available_windows = ["hamming", "hann", "blackman", "bartlett", "boxcar", "kaiser"]
        if window not in available_windows:
            Logger.log_message_static(f"Calculations-Filters: Unknown window '{window}', using 'hamming'",
                                      Logger.WARNING)
            window = "hamming"

        Logger.log_message_static(
            f"Calculations-Filters: FIR filter parameters - "
            f"Type={filter_type}, Taps={numtaps}, Cutoff={cutoff_freq}, "
            f"Window={window}, Fs={sample_rate:.1f}Hz",
            Logger.DEBUG
        )

        # Design FIR filter using windowed method
        try:
            if window == "kaiser":
                # Kaiser window requires beta parameter - use default for good performance
                beta = 8.6  # Good compromise between main lobe width and side lobe level
                fir_coeffs = sc_signal.firwin(
                    numtaps, normalized_cutoff,
                    window=('kaiser', beta),
                    pass_zero=pass_zero,
                    fs=sample_rate
                )
            else:
                fir_coeffs = sc_signal.firwin(
                    numtaps, normalized_cutoff,
                    window=window,
                    pass_zero=pass_zero,
                    fs=sample_rate
                )
        except Exception as design_error:
            Logger.log_message_static(f"Calculations-Filters: FIR filter design failed: {design_error}", Logger.ERROR)
            if dialog:
                QMessageBox.critical(dialog, title, f"FIR filter design failed:\n{str(design_error)}")
            return None

        # Apply FIR filter using zero-phase filtering
        try:
            filtered_signal = sc_signal.filtfilt(fir_coeffs, [1.0], processed_values)
        except Exception as filter_error:
            Logger.log_message_static(f"Calculations-Filters: FIR filter application failed: {filter_error}",
                                      Logger.ERROR)
            if dialog:
                QMessageBox.critical(dialog, title, f"FIR filter application failed:\n{str(filter_error)}")
            return None

        # Calculate filter frequency response
        try:
            w, h = sc_signal.freqz(fir_coeffs, worN=2048, fs=sample_rate)
            magnitude_response = np.abs(h)
            phase_response = np.angle(h)
            magnitude_db = 20 * np.log10(magnitude_response + 1e-12)

            # Group delay for FIR filter (constant)
            group_delay = (numtaps - 1) / 2 / sample_rate

            # Find -3dB bandwidth
            max_magnitude_db = np.max(magnitude_db)
            cutoff_3db_indices = np.where(magnitude_db >= (max_magnitude_db - 3))[0]

            if len(cutoff_3db_indices) > 0:
                bandwidth_3db = w[cutoff_3db_indices[-1]] - w[cutoff_3db_indices[0]]
            else:
                bandwidth_3db = 0.0

        except Exception as response_error:
            Logger.log_message_static(f"Calculations-Filters: Filter response calculation failed: {response_error}",
                                      Logger.WARNING)
            w = np.array([0, sample_rate / 2])
            magnitude_response = np.array([1, 1])
            phase_response = np.array([0, 0])
            magnitude_db = np.array([0, 0])
            bandwidth_3db = 0.0
            group_delay = 0.0

        # Calculate filter performance metrics
        original_energy = np.sum(processed_values ** 2)
        filtered_energy = np.sum(filtered_signal ** 2)
        energy_ratio = filtered_energy / original_energy if original_energy > 0 else 0.0

        # RMS comparison
        original_rms = np.sqrt(np.mean(processed_values ** 2))
        filtered_rms = np.sqrt(np.mean(filtered_signal ** 2))
        rms_ratio = filtered_rms / original_rms if original_rms > 0 else 0.0

        # Build comprehensive results dictionary
        results = {
            # Filtered data
            "Filtered Signal": filtered_signal,
            "Original Signal": processed_values,

            # Filter design parameters
            "Filter Taps": fir_coeffs,
            "Filter Type": filter_type,
            "Cutoff Frequency (Hz)": cutoff_freq,
            "Number of Taps": numtaps,
            "Window Function": window,
            "Sample Rate (Hz)": float(sample_rate),
            "Nyquist Frequency (Hz)": float(nyquist),

            # Filter response data
            "Frequency Response (Hz)": w,
            "Magnitude Response": magnitude_response,
            "Magnitude Response (dB)": magnitude_db,
            "Phase Response (rad)": phase_response,
            "Phase Response (deg)": np.degrees(phase_response),

            # Performance metrics
            "3dB Bandwidth (Hz)": float(bandwidth_3db),
            "Energy Ratio (Filtered/Original)": float(energy_ratio),
            "RMS Ratio (Filtered/Original)": float(rms_ratio),
            "Original RMS": float(original_rms),
            "Filtered RMS": float(filtered_rms),

            # FIR-specific characteristics
            "Group Delay (s)": float(group_delay),
            "Linear Phase": True,
            "Filter Stability": "Always Stable",
            "Zero-Phase Applied": True
        }

        Logger.log_message_static(
            f"Calculations-Filters: FIR filtering completed. "
            f"Energy_ratio={energy_ratio:.3f}, RMS_ratio={rms_ratio:.3f}, "
            f"Group_delay={group_delay * 1000:.1f}ms",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Filters: Error in FIR filtering: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Calculations-Filters: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None


def design_filter_parameters(sample_rate, filter_type, passband_freq, stopband_freq, passband_ripple_db=1,
                             stopband_attenuation_db=60):
    """
    Design optimal filter parameters based on specification requirements.

    Helper function to determine appropriate filter order and cutoff frequencies
    to meet given specifications.

    Args:
        sample_rate (float): Sampling frequency in Hz
        filter_type (str): Type of filter ("lowpass", "highpass", "bandpass", "bandstop")
        passband_freq (float or tuple): Passband edge frequency(ies) in Hz
        stopband_freq (float or tuple): Stopband edge frequency(ies) in Hz
        passband_ripple_db (float, optional): Maximum allowed ripple in passband. Defaults to 1.
        stopband_attenuation_db (float, optional): Minimum required attenuation in stopband. Defaults to 60.

    Returns:
        dict: Dictionary containing recommended filter parameters:
            - IIR Order: Recommended order for Butterworth IIR filter
            - FIR Taps: Recommended number of taps for FIR filter
            - Cutoff Frequency: Recommended cutoff frequency
            - Transition Width: Width of transition band
            - Filter Recommendations: Suggestions for filter choice
    """
    try:
        nyquist = sample_rate / 2

        # Calculate transition bandwidth
        if filter_type in ["lowpass", "highpass"]:
            if isinstance(passband_freq, (list, tuple)):
                passband_freq = passband_freq[0]
            if isinstance(stopband_freq, (list, tuple)):
                stopband_freq = stopband_freq[0]

            transition_bw = abs(stopband_freq - passband_freq)
            cutoff_freq = (passband_freq + stopband_freq) / 2  # Geometric mean might be better

        elif filter_type in ["bandpass", "bandstop"]:
            if not isinstance(passband_freq, (list, tuple)) or len(passband_freq) != 2:
                raise ValueError("Bandpass/bandstop filters require two passband frequencies")
            if not isinstance(stopband_freq, (list, tuple)) or len(stopband_freq) != 2:
                raise ValueError("Bandpass/bandstop filters require two stopband frequencies")

            # For bandpass: stopband is outside passband
            # For bandstop: stopband is inside passband
            if filter_type == "bandpass":
                transition_bw = min(
                    abs(passband_freq[0] - stopband_freq[0]),
                    abs(passband_freq[1] - stopband_freq[1])
                )
            else:  # bandstop
                transition_bw = min(
                    abs(stopband_freq[0] - passband_freq[0]),
                    abs(stopband_freq[1] - passband_freq[1])
                )

            cutoff_freq = passband_freq
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        # Estimate IIR filter order (Butterworth)
        # Using approximation: N ≈ log10(10^(As/10) - 1) / (2 * log10(Ωs/Ωp))
        omega_ratio = transition_bw / min(
            passband_freq if isinstance(passband_freq, (int, float)) else min(passband_freq))
        omega_ratio = max(omega_ratio, 0.01)  # Avoid log(0)

        numerator = np.log10(10 ** (stopband_attenuation_db / 10) - 1)
        denominator = 2 * np.log10(omega_ratio)
        iir_order = max(1, int(np.ceil(numerator / denominator)))
        iir_order = min(iir_order, 20)  # Practical limit

        # Estimate FIR filter length
        # Using Kaiser window method estimation
        delta_f = transition_bw / sample_rate  # Normalized transition width
        if delta_f > 0:
            # Empirical formula for Kaiser window
            if stopband_attenuation_db > 50:
                beta = 0.1102 * (stopband_attenuation_db - 8.7)
            elif stopband_attenuation_db > 21:
                beta = 0.5842 * (stopband_attenuation_db - 21) ** 0.4 + 0.07886 * (stopband_attenuation_db - 21)
            else:
                beta = 0

            fir_taps = int(np.ceil((stopband_attenuation_db - 8) / (2.285 * delta_f)))
            fir_taps = max(3, fir_taps)

            # Make odd for certain filter types
            if filter_type in ["highpass", "bandstop"] and fir_taps % 2 == 0:
                fir_taps += 1
        else:
            fir_taps = 101  # Default

        # Generate recommendations
        recommendations = []

        if iir_order <= 8:
            recommendations.append("IIR filter recommended: efficient and sharp transition")
        else:
            recommendations.append("IIR filter may have high order; consider relaxing specifications")

        if fir_taps <= 200:
            recommendations.append("FIR filter feasible: linear phase, always stable")
        else:
            recommendations.append("FIR filter may be computationally expensive")

        if transition_bw / nyquist < 0.01:
            recommendations.append("Very narrow transition band: requires high-order filter")
        elif transition_bw / nyquist > 0.1:
            recommendations.append("Wide transition band: low-order filter sufficient")

        return {
            "IIR Order (Butterworth)": iir_order,
            "FIR Taps (Kaiser)": fir_taps,
            "Cutoff Frequency (Hz)": cutoff_freq,
            "Transition Width (Hz)": transition_bw,
            "Normalized Transition Width": transition_bw / nyquist,
            "Filter Recommendations": recommendations,
            "Estimated FIR Beta (Kaiser)": beta if 'beta' in locals() else 0,
        }

    except Exception as e:
        Logger.log_message_static(f"Calculations-Filters: Error in filter parameter design: {str(e)}", Logger.ERROR)
        return {
            "Error": str(e),
            "IIR Order (Butterworth)": 4,  # Default fallback
            "FIR Taps (Kaiser)": 101,
            "Filter Recommendations": ["Error in parameter calculation, using defaults"]
        }