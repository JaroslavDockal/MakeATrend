"""
Wavelet-based signal analysis: Continuous and Discrete Wavelet Transform.

This module provides comprehensive wavelet analysis capabilities:
- Continuous Wavelet Transform (CWT) for time-frequency analysis
- Discrete Wavelet Transform (DWT) for multiresolution decomposition
- Multiple wavelet families (Morlet, Mexican Hat, Daubechies, etc.)
- Wavelet-based denoising and feature extraction
- Scalogram generation and analysis

Wavelets provide both time and frequency localization, making them ideal
for analyzing non-stationary signals and detecting transient events.
"""

import numpy as np
import pywt
from PySide6.QtWidgets import QMessageBox

from .common import safe_prepare_signal, safe_sample_rate, validate_analysis_inputs
from utils.logger import Logger


def calculate_wavelet_analysis_cwt(time_arr, values, wavelet='cmor1.5-1.0', scales=None, dialog=None,
                                   title="CWT Analysis"):
    """
    Perform Continuous Wavelet Transform (CWT) analysis on a signal.

    CWT provides a time-frequency representation by convolving the signal with
    scaled and shifted versions of a mother wavelet. This reveals how frequency
    content varies over time, making it ideal for non-stationary signal analysis.

    Args:
        time_arr (np.ndarray): Time values corresponding to signal samples.
        values (np.ndarray): Signal values to analyze.
        wavelet (str, optional): Mother wavelet for analysis. Options include:
            - 'cmor1.5-1.0': Complex Morlet wavelet (good for oscillatory signals)
            - 'morl': Morlet wavelet (real-valued version)
            - 'mexh': Mexican Hat wavelet (good for spike detection)
            - 'cgau8': Complex Gaussian wavelet
            - 'shan': Shannon wavelet
            Defaults to 'cmor1.5-1.0'.
        scales (np.ndarray, optional): Array of scales for analysis.
            If None, automatically generates logarithmic scale distribution.
            Defaults to None.
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "CWT Analysis".

    Returns:
        dict or None: Dictionary containing CWT analysis results:
            - Coefficients: Complex CWT coefficients matrix (scales x time)
            - Frequencies (Hz): Frequency values corresponding to scales
            - Scales: Scale values used in analysis
            - Time Array: Time values for coefficient matrix
            - Power: Power (|coefficients|²) matrix
            - Wavelet: Mother wavelet used
            - Dominant Frequency (Hz): Frequency with highest average power
            - Total Energy: Total energy in time-frequency domain
            - Frequency vs Time: Peak frequency at each time point
            - Ridge Analysis: Ridge detection and extraction

        Returns None if validation fails or user cancels.

    Example:
        >>> t = np.linspace(0, 1, 1000)
        >>> # Chirp signal: frequency increases linearly with time
        >>> signal = np.sin(2*np.pi*(10 + 40*t)*t)
        >>> result = calculate_wavelet_analysis_cwt(t, signal)
        >>> coeffs = result['Coefficients']  # Time-frequency representation
        >>> freqs = result['Frequencies (Hz)']
        >>> print(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
    """
    Logger.log_message_static(f"Calculations-Wavelet: Starting CWT analysis with wavelet '{wavelet}'", Logger.DEBUG)

    # Validate inputs
    is_valid, error_msg, sample_rate = validate_analysis_inputs(time_arr, values, min_length=4,
                                                                require_positive_sample_rate=True)
    if not is_valid:
        Logger.log_message_static(f"Calculations-Wavelet: Input validation failed: {error_msg}", Logger.ERROR)
        if dialog:
            QMessageBox.warning(dialog, title, f"CWT Analysis Error:\n{error_msg}")
        return None

    # Prepare signal
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Calculations-Wavelet: Signal validation failed", Logger.WARNING)
        return None

    try:
        # Validate wavelet
        try:
            pywt.ContinuousWavelet(wavelet)
        except ValueError:
            Logger.log_message_static(f"Calculations-Wavelet: Invalid wavelet '{wavelet}', using 'cmor1.5-1.0'",
                                      Logger.WARNING)
            wavelet = 'cmor1.5-1.0'
            if dialog:
                QMessageBox.warning(dialog, title, f"Invalid wavelet specified. Using 'cmor1.5-1.0' instead.")

        # Generate scales if not provided
        if scales is None:
            # Create logarithmic scale distribution
            min_scale = 1
            max_scale = min(128, len(processed_values) // 4)  # Reasonable upper limit
            num_scales = min(64, max_scale)  # Limit number of scales for performance

            if max_scale <= min_scale:
                max_scale = min_scale + 10

            scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)

        # Validate scales
        scales = np.asarray(scales)
        if len(scales) == 0:
            Logger.log_message_static("Calculations-Wavelet: No valid scales provided", Logger.ERROR)
            if dialog:
                QMessageBox.warning(dialog, title, "No valid scales for CWT analysis.")
            return None

        # Ensure scales are positive and reasonable
        scales = scales[scales > 0]
        max_reasonable_scale = len(processed_values) // 2
        scales = scales[scales <= max_reasonable_scale]

        if len(scales) == 0:
            Logger.log_message_static("Calculations-Wavelet: No scales within reasonable range", Logger.ERROR)
            if dialog:
                QMessageBox.warning(dialog, title, "No scales within reasonable range for analysis.")
            return None

        Logger.log_message_static(
            f"Calculations-Wavelet: CWT parameters - "
            f"Wavelet={wavelet}, Scales={len(scales)} ({scales[0]:.1f} to {scales[-1]:.1f}), "
            f"Sample_rate={sample_rate:.1f}Hz",
            Logger.DEBUG
        )

        # Perform CWT
        dt = 1.0 / sample_rate
        coefficients, frequencies = pywt.cwt(processed_values, scales, wavelet, sampling_period=dt)

        # Calculate power (magnitude squared for complex wavelets)
        power = np.abs(coefficients) ** 2

        # Calculate average power per scale/frequency
        avg_power_per_freq = np.mean(power, axis=1)

        # Find dominant frequency
        dominant_freq_idx = np.argmax(avg_power_per_freq)
        dominant_frequency = frequencies[dominant_freq_idx]

        # Calculate total energy in time-frequency domain
        total_energy = np.sum(power)

        # Energy distribution across frequency bands
        num_bands = min(8, len(frequencies))
        if num_bands > 1:
            band_indices = np.linspace(0, len(frequencies) - 1, num_bands + 1, dtype=int)
            energy_per_band = {}

            for i in range(num_bands):
                start_idx = band_indices[i]
                end_idx = band_indices[i + 1]
                band_energy = np.sum(power[start_idx:end_idx])
                band_percentage = (band_energy / total_energy) * 100 if total_energy > 0 else 0

                freq_start = frequencies[start_idx]
                freq_end = frequencies[end_idx - 1] if end_idx > start_idx else frequencies[start_idx]
                band_label = f"Band {i + 1} ({freq_start:.1f}-{freq_end:.1f} Hz)"
                energy_per_band[band_label] = band_percentage
        else:
            energy_per_band = {"Single Band": 100.0}

        # Peak frequency at each time point (ridge analysis)
        peak_freq_indices = np.argmax(power, axis=0)
        peak_frequencies_time = frequencies[peak_freq_indices]

        # Advanced ridge analysis - find continuous ridges
        try:
            # Simple ridge detection: look for local maxima along frequency axis
            ridge_points = []
            for t_idx in range(power.shape[1]):
                freq_column = power[:, t_idx]
                # Find local maxima
                local_maxima = []
                for f_idx in range(1, len(freq_column) - 1):
                    if (freq_column[f_idx] > freq_column[f_idx - 1] and
                            freq_column[f_idx] > freq_column[f_idx + 1] and
                            freq_column[f_idx] > np.max(freq_column) * 0.1):  # Above 10% of max
                        local_maxima.append((f_idx, freq_column[f_idx]))

                # Take strongest local maximum
                if local_maxima:
                    strongest = max(local_maxima, key=lambda x: x[1])
                    ridge_points.append({
                        'time_idx': t_idx,
                        'freq_idx': strongest[0],
                        'frequency': frequencies[strongest[0]],
                        'power': strongest[1]
                    })

            ridge_frequencies = [p['frequency'] for p in ridge_points] if ridge_points else []
            ridge_powers = [p['power'] for p in ridge_points] if ridge_points else []

        except Exception as ridge_error:
            Logger.log_message_static(f"Calculations-Wavelet: Ridge analysis failed: {ridge_error}", Logger.DEBUG)
            ridge_frequencies = []
            ridge_powers = []

        # Frequency statistics
        freq_statistics = {
            "Mean Frequency (Hz)": float(np.average(frequencies, weights=avg_power_per_freq)),
            "Frequency Standard Deviation (Hz)": float(np.sqrt(
                np.average((frequencies - np.average(frequencies, weights=avg_power_per_freq)) ** 2,
                           weights=avg_power_per_freq))),
            "Min Frequency (Hz)": float(np.min(frequencies)),
            "Max Frequency (Hz)": float(np.max(frequencies)),
            "Frequency Range (Hz)": float(np.max(frequencies) - np.min(frequencies))
        }

        # Time-frequency localization metrics
        # Calculate instantaneous bandwidth (frequency spread at each time)
        instantaneous_bandwidth = []
        for t_idx in range(power.shape[1]):
            freq_power = power[:, t_idx]
            if np.sum(freq_power) > 0:
                freq_power_norm = freq_power / np.sum(freq_power)
                mean_freq = np.sum(frequencies * freq_power_norm)
                freq_variance = np.sum(((frequencies - mean_freq) ** 2) * freq_power_norm)
                bandwidth = np.sqrt(freq_variance)
                instantaneous_bandwidth.append(bandwidth)
            else:
                instantaneous_bandwidth.append(0.0)

        instantaneous_bandwidth = np.array(instantaneous_bandwidth)

        # Build comprehensive results dictionary
        results = {
            # Core CWT data
            "Coefficients": coefficients,
            "Frequencies (Hz)": frequencies,
            "Scales": scales,
            "Time Array": time_arr,
            "Power": power,
            "Wavelet": wavelet,

            # Power and energy analysis
            "Power per Scale": avg_power_per_freq,
            "Dominant Frequency (Hz)": float(dominant_frequency),
            "Total Energy": float(total_energy),
            "Energy Distribution (%)": energy_per_band,

            # Time-frequency characteristics
            "Peak Frequency vs Time (Hz)": peak_frequencies_time,
            "Instantaneous Bandwidth (Hz)": instantaneous_bandwidth,
            "Mean Instantaneous Bandwidth (Hz)": float(np.mean(instantaneous_bandwidth)),

            # Ridge analysis
            "Ridge Frequencies (Hz)": ridge_frequencies,
            "Ridge Powers": ridge_powers,
            "Number of Ridge Points": len(ridge_points) if 'ridge_points' in locals() else 0,

            # Frequency statistics
            "Frequency Statistics": freq_statistics,

            # Analysis parameters
            "Sample Rate (Hz)": float(sample_rate),
            "Signal Length": len(processed_values),
            "Number of Scales": len(scales),
            "Scale Range": f"{scales[0]:.2f} - {scales[-1]:.2f}",
            "Frequency Resolution (Hz)": float((frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0),
            "Time Resolution (s)": float(dt)
        }

        # Add wavelet-specific characteristics
        if 'cmor' in wavelet:
            results["Wavelet Type"] = "Complex Morlet"
            results["Wavelet Characteristics"] = "Good for oscillatory signals, complex-valued"
        elif 'morl' in wavelet:
            results["Wavelet Type"] = "Morlet"
            results["Wavelet Characteristics"] = "Good for oscillatory signals, real-valued"
        elif 'mexh' in wavelet:
            results["Wavelet Type"] = "Mexican Hat"
            results["Wavelet Characteristics"] = "Good for spike/transient detection"
        elif 'cgau' in wavelet:
            results["Wavelet Type"] = "Complex Gaussian"
            results["Wavelet Characteristics"] = "Smooth, good time-frequency localization"
        else:
            results["Wavelet Type"] = "Other"
            results["Wavelet Characteristics"] = "Custom wavelet properties"

        Logger.log_message_static(
            f"Calculations-Wavelet: CWT analysis completed. "
            f"Dominant_freq={dominant_frequency:.2f}Hz, Total_energy={total_energy:.6e}, "
            f"Ridge_points={len(ridge_points) if 'ridge_points' in locals() else 0}",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Wavelet: Error in CWT analysis: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Calculations-Wavelet: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None


def calculate_wavelet_analysis_dwt(time_arr, values, wavelet='db4', level=None, dialog=None, title="DWT Analysis"):
    """
    Perform Discrete Wavelet Transform (DWT) analysis on a signal.

    DWT decomposes a signal into approximation and detail coefficients at multiple
    resolution levels. This provides a multiresolution analysis useful for
    denoising, compression, and feature extraction.

    Args:
        time_arr (np.ndarray): Time values (used for metadata, not decomposition).
        values (np.ndarray): Signal values to decompose.
        wavelet (str, optional): Wavelet family for decomposition. Options include:
            - 'db4': Daubechies 4 (good general purpose)
            - 'db8': Daubechies 8 (smoother, more oscillations)
            - 'haar': Haar wavelet (simplest, discontinuous)
            - 'sym4': Symlet 4 (nearly symmetric)
            - 'coif2': Coiflet 2 (symmetric, good for smooth signals)
            - 'bior2.2': Biorthogonal (symmetric, good for image processing)
            Defaults to 'db4'.
        level (int, optional): Decomposition level. If None, automatically
            calculates maximum useful level. Higher levels provide coarser
            time resolution but finer frequency resolution. Defaults to None.
        dialog (QWidget, optional): Parent dialog for user interaction. Defaults to None.
        title (str, optional): Title for user dialogs. Defaults to "DWT Analysis".

    Returns:
        dict or None: Dictionary containing DWT analysis results:
            - Coefficients: List of coefficient arrays [cA_n, cD_n, cD_n-1, ..., cD_1]
            - Approximation: Final approximation coefficients (low frequencies)
            - Details: Detail coefficients for each level (high frequencies)
            - Energy per Level: Energy distribution across decomposition levels
            - Reconstruction Error: Error in perfect reconstruction
            - Wavelet: Wavelet family used
            - Decomposition Level: Number of levels used
            - Level Frequencies: Frequency bands for each level

        Returns None if validation fails or user cancels.

    Example:
        >>> t = np.linspace(0, 1, 1024)
        >>> signal = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*50*t) + 0.1*np.random.randn(1024)
        >>> result = calculate_wavelet_analysis_dwt(t, signal, 'db4', level=5)
        >>> coeffs = result['Coefficients']
        >>> energies = result['Energy per Level']
    """
    Logger.log_message_static(f"Calculations-Wavelet: Starting DWT analysis with wavelet '{wavelet}'", Logger.DEBUG)

    # Validate inputs
    is_valid, error_msg, sample_rate = validate_analysis_inputs(time_arr, values, min_length=4)
    if not is_valid:
        Logger.log_message_static(f"Calculations-Wavelet: Input validation failed: {error_msg}", Logger.ERROR)
        if dialog:
            QMessageBox.warning(dialog, title, f"DWT Analysis Error:\n{error_msg}")
        return None

    # Prepare signal
    processed_values = safe_prepare_signal(values, dialog, title)
    if processed_values is None:
        Logger.log_message_static("Calculations-Wavelet: Signal validation failed", Logger.WARNING)
        return None

    try:
        # Validate wavelet
        try:
            wavelet_obj = pywt.Wavelet(wavelet)
        except ValueError:
            Logger.log_message_static(f"Calculations-Wavelet: Invalid wavelet '{wavelet}', using 'db4'", Logger.WARNING)
            wavelet = 'db4'
            wavelet_obj = pywt.Wavelet('db4')
            if dialog:
                QMessageBox.warning(dialog, title, f"Invalid wavelet specified. Using 'db4' instead.")

        # Determine decomposition level
        if level is None:
            max_level = pywt.dwt_max_level(len(processed_values), wavelet_obj.dec_len)
            # Use a reasonable level (not too deep to avoid over-decomposition)
            level = min(max_level, int(np.log2(len(processed_values))) - 2)
            level = max(1, level)  # At least 1 level
        else:
            max_level = pywt.dwt_max_level(len(processed_values), wavelet_obj.dec_len)
            if level > max_level:
                level = max_level
                Logger.log_message_static(f"Calculations-Wavelet: Level adjusted to maximum possible: {level}",
                                          Logger.WARNING)
            elif level < 1:
                level = 1
                Logger.log_message_static("Calculations-Wavelet: Level adjusted to minimum value: 1", Logger.WARNING)

        Logger.log_message_static(
            f"Calculations-Wavelet: DWT parameters - "
            f"Wavelet={wavelet}, Level={level}, Max_level={max_level}, "
            f"Signal_length={len(processed_values)}",
            Logger.DEBUG
        )

        # Perform DWT decomposition
        coefficients = pywt.wavedec(processed_values, wavelet, level=level)

        # Extract approximation and details
        approximation = coefficients[0]  # Lowest frequency component
        details = coefficients[1:]  # Higher frequency components (from coarse to fine)

        # Calculate energy per level
        energies = []
        total_energy = 0

        for i, coeff in enumerate(coefficients):
            energy = np.sum(coeff ** 2)
            energies.append(energy)
            total_energy += energy

        # Convert to percentages
        energy_percentages = [(e / total_energy) * 100 if total_energy > 0 else 0 for e in energies]

        # Calculate frequency bands for each level
        # Note: This is approximate since DWT doesn't have exact frequency bands
        if sample_rate > 0:
            nyquist = sample_rate / 2
            freq_bands = []

            # Approximation (level 0) contains frequencies [0, Fs/(2^(level+1))]
            approx_high_freq = nyquist / (2 ** level)
            freq_bands.append((0, approx_high_freq))

            # Details contain frequencies [Fs/(2^(i+1)), Fs/(2^i)] where i goes from level to 1
            for i in range(level, 0, -1):
                low_freq = nyquist / (2 ** i)
                high_freq = nyquist / (2 ** (i - 1))
                freq_bands.append((low_freq, high_freq))
        else:
            freq_bands = [("N/A", "N/A")] * (level + 1)

        # Test reconstruction to verify perfect reconstruction
        try:
            reconstructed = pywt.waverec(coefficients, wavelet)

            # Handle length mismatch due to boundary conditions
            min_length = min(len(processed_values), len(reconstructed))
            reconstruction_error = np.mean((processed_values[:min_length] - reconstructed[:min_length]) ** 2)
            reconstruction_snr = 10 * np.log10(
                np.var(processed_values[:min_length]) / reconstruction_error) if reconstruction_error > 0 else np.inf

        except Exception as recon_error:
            Logger.log_message_static(f"Calculations-Wavelet: Reconstruction test failed: {recon_error}",
                                      Logger.WARNING)
            reconstruction_error = np.nan
            reconstruction_snr = np.nan

        # Calculate coefficient statistics for each level
        coeff_stats = []
        for i, coeff in enumerate(coefficients):
            level_name = "Approximation" if i == 0 else f"Detail Level {level - i + 1}"
            stats = {
                "Level": level_name,
                "Coefficients": len(coeff),
                "Energy": energies[i],
                "Energy (%)": energy_percentages[i],
                "RMS": float(np.sqrt(np.mean(coeff ** 2))),
                "Max Coefficient": float(np.max(np.abs(coeff))),
                "Mean Coefficient": float(np.mean(coeff)),
                "Std Coefficient": float(np.std(coeff)),
                "Frequency Band (Hz)": f"{freq_bands[i][0]:.1f} - {freq_bands[i][1]:.1f}" if freq_bands[i][
                                                                                                 0] != "N/A" else "N/A"
            }
            coeff_stats.append(stats)

        # Wavelet properties
        wavelet_props = {
            "Family": wavelet_obj.family_name,
            "Short Name": wavelet_obj.short_name,
            "Orthogonal": wavelet_obj.orthogonal,
            "Biorthogonal": wavelet_obj.biorthogonal,
            "Symmetry": wavelet_obj.symmetry,
            "Vanishing Moments": getattr(wavelet_obj, 'vanishing_moments_psi', 'N/A'),
            "Filter Length": wavelet_obj.dec_len
        }

        # Build comprehensive results dictionary
        results = {
            # Core DWT data
            "Coefficients": coefficients,
            "Approximation": approximation,
            "Details": details,
            "Wavelet": wavelet,
            "Decomposition Level": level,

            # Energy analysis
            "Energy per Level": energies,
            "Energy Percentages": energy_percentages,
            "Total Energy": total_energy,
            "Dominant Level": int(np.argmax(energies)),
            "Dominant Level Energy (%)": max(energy_percentages),

            # Frequency information
            "Level Frequency Bands (Hz)": freq_bands,
            "Nyquist Frequency (Hz)": nyquist if sample_rate > 0 else "N/A",

            # Reconstruction quality
            "Reconstruction Error (MSE)": float(reconstruction_error) if not np.isnan(
                reconstruction_error) else "Failed",
            "Reconstruction SNR (dB)": float(reconstruction_snr) if not np.isnan(reconstruction_snr) else "N/A",

            # Detailed statistics
            "Coefficient Statistics": coeff_stats,
            "Wavelet Properties": wavelet_props,

            # Analysis metadata
            "Sample Rate (Hz)": float(sample_rate) if sample_rate > 0 else "N/A",
            "Signal Length": len(processed_values),
            "Max Possible Level": max_level,
            "Approximation Length": len(approximation),
            "Detail Lengths": [len(d) for d in details]
        }

        # Add recommendations based on energy distribution
        recommendations = []

        # Check if most energy is in approximation
        if energy_percentages[0] > 70:
            recommendations.append("Most energy in approximation - signal is predominantly low-frequency")

        # Check if energy is distributed across multiple levels
        significant_levels = sum(1 for e in energy_percentages if e > 10)
        if significant_levels > 3:
            recommendations.append("Energy distributed across multiple levels - complex signal structure")
        elif significant_levels == 1:
            recommendations.append("Energy concentrated in single level - simple signal structure")

        # Check for high-frequency content
        high_freq_energy = sum(energy_percentages[-2:])  # Last two detail levels
        if high_freq_energy > 30:
            recommendations.append("Significant high-frequency content detected")

        results["Analysis Recommendations"] = recommendations

        Logger.log_message_static(
            f"Calculations-Wavelet: DWT analysis completed. "
            f"Level={level}, Total_energy={total_energy:.6e}, "
            f"Dominant_level={np.argmax(energies)}, Recon_error={reconstruction_error:.6e}",
            Logger.DEBUG
        )
        return results

    except Exception as e:
        Logger.log_message_static(f"Calculations-Wavelet: Error in DWT analysis: {str(e)}", Logger.ERROR)
        import traceback
        Logger.log_message_static(f"Calculations-Wavelet: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        return None


def get_available_wavelets():
    """
    Get information about available wavelet families and their characteristics.

    Returns:
        dict: Dictionary containing information about wavelet families:
            - Continuous Wavelets: Wavelets suitable for CWT
            - Discrete Wavelets: Wavelets suitable for DWT
            - Wavelet Properties: Characteristics of each family
    """
    try:
        # Get all available wavelets
        continuous_wavelets = []
        discrete_wavelets = []

        # Continuous wavelets (for CWT)
        cwt_families = ['cmor', 'mexh', 'morl', 'cgau', 'shan', 'fbsp']
        for family in cwt_families:
            try:
                if family == 'cmor':
                    # Complex Morlet has parameters
                    continuous_wavelets.extend(['cmor1.0-1.0', 'cmor1.5-1.0', 'cmor2.0-1.0'])
                elif family == 'cgau':
                    # Complex Gaussian has order parameter
                    continuous_wavelets.extend(['cgau1', 'cgau2', 'cgau4', 'cgau8'])
                elif family == 'fbsp':
                    # Frequency B-Spline has parameters
                    continuous_wavelets.extend(['fbsp1-1.5-1.0', 'fbsp2-1-0.5'])
                else:
                    continuous_wavelets.append(family)
            except:
                pass

        # Discrete wavelets (for DWT)
        dwt_families = pywt.families(short=False)
        for family in dwt_families:
            try:
                wavelets = pywt.wavelist(family)
                discrete_wavelets.extend(wavelets)
            except:
                pass

        # Wavelet characteristics
        wavelet_properties = {
            'Daubechies (db)': {
                'description': 'Compactly supported, orthogonal, good for general analysis',
                'examples': ['db1', 'db4', 'db8'],
                'best_for': 'General purpose, signal compression'
            },
            'Haar (haar)': {
                'description': 'Simplest wavelet, discontinuous, good for edges',
                'examples': ['haar'],
                'best_for': 'Edge detection, simple signals'
            },
            'Symlets (sym)': {
                'description': 'Nearly symmetric version of Daubechies',
                'examples': ['sym4', 'sym8'],
                'best_for': 'Signals requiring phase information'
            },
            'Coiflets (coif)': {
                'description': 'More symmetric than Daubechies, good reconstruction',
                'examples': ['coif2', 'coif4'],
                'best_for': 'Function approximation, smooth signals'
            },
            'Biorthogonal (bior)': {
                'description': 'Symmetric, perfect reconstruction, good for images',
                'examples': ['bior2.2', 'bior4.4'],
                'best_for': 'Image processing, symmetric signals'
            },
            'Complex Morlet (cmor)': {
                'description': 'Complex-valued, good time-frequency localization',
                'examples': ['cmor1.5-1.0'],
                'best_for': 'Oscillatory signals, time-frequency analysis'
            },
            'Morlet (morl)': {
                'description': 'Real-valued version of complex Morlet',
                'examples': ['morl'],
                'best_for': 'Oscillatory signals, real-valued analysis'
            },
            'Mexican Hat (mexh)': {
                'description': 'Good for spike detection and transients',
                'examples': ['mexh'],
                'best_for': 'Spike detection, transient analysis'
            }
        }

        return {
            'Continuous Wavelets': continuous_wavelets,
            'Discrete Wavelets': discrete_wavelets[:20],  # Limit for readability
            'Wavelet Properties': wavelet_properties,
            'Total CWT Wavelets': len(continuous_wavelets),
            'Total DWT Wavelets': len(discrete_wavelets)
        }

    except Exception as e:
        Logger.log_message_static(f"Calculations-Wavelet: Error getting wavelet information: {str(e)}", Logger.ERROR)
        return {
            'Continuous Wavelets': ['cmor1.5-1.0', 'morl', 'mexh'],
            'Discrete Wavelets': ['db4', 'haar', 'sym4'],
            'Error': str(e)
        }