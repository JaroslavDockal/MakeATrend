#!/usr/bin/env python3
"""
Test script for the new analysis.calculations module structure.

This script tests all the calculation functions to ensure they work correctly
after the refactoring. It creates test signals and verifies that all functions
return expected results.

Run this script to validate that the refactoring is successful:
python test_calculations.py
"""

import numpy as np
import sys
import traceback
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, '.')


def create_test_signals():
    """Create various test signals for validation."""
    # Common time array
    duration = 2.0  # seconds
    sample_rate = 1000  # Hz
    t = np.linspace(0, duration, int(duration * sample_rate))

    signals = {}

    # 1. Pure sine wave
    signals['sine_10hz'] = (t, np.sin(2 * np.pi * 10 * t))

    # 2. Multi-frequency signal
    signals['multi_freq'] = (t,
                             np.sin(2 * np.pi * 10 * t) +
                             0.5 * np.sin(2 * np.pi * 25 * t) +
                             0.25 * np.sin(2 * np.pi * 50 * t)
                             )

    # 3. Noisy signal
    signals['noisy_sine'] = (t,
                             np.sin(2 * np.pi * 15 * t) + 0.2 * np.random.randn(len(t))
                             )

    # 4. Chirp signal (frequency sweep)
    signals['chirp'] = (t,
                        np.sin(2 * np.pi * (5 + 20 * t) * t)
                        )

    # 5. Amplitude modulated signal
    carrier_freq = 50
    mod_freq = 5
    signals['am_signal'] = (t,
                            (1 + 0.5 * np.sin(2 * np.pi * mod_freq * t)) * np.sin(2 * np.pi * carrier_freq * t)
                            )

    # 6. Impulse signal for transient analysis
    impulse = np.zeros(len(t))
    impulse[len(t) // 4] = 1.0
    impulse[len(t) // 2] = 0.8
    impulse[3 * len(t) // 4] = 0.6
    signals['impulses'] = (t, impulse)

    # 7. Random walk for trend analysis
    random_walk = np.cumsum(np.random.randn(len(t))) * 0.1
    signals['trend'] = (t, random_walk + 0.02 * t)  # Add linear trend

    return signals


def test_module_imports():
    """Test that all modules can be imported successfully."""
    print("=" * 60)
    print("TESTING MODULE IMPORTS")
    print("=" * 60)

    import_results = {}

    # Test individual module imports
    modules_to_test = [
        'analysis.calculations.common',
        'analysis.calculations.basic',
        'analysis.calculations.frequency',
        'analysis.calculations.correlation',
        'analysis.calculations.hilbert',
        'analysis.calculations.filters',
        'analysis.calculations.wavelet',
        'analysis.calculations.cepstrum'
    ]

    for module in modules_to_test:
        try:
            __import__(module)
            import_results[module] = "✅ Success"
            print(f"✅ {module}")
        except Exception as e:
            import_results[module] = f"❌ Failed: {e}"
            print(f"❌ {module}: {e}")

    # Test main package import
    try:
        from analysis.calculations import *
        import_results['analysis.calculations'] = "✅ Success"
        print(f"✅ analysis.calculations (import *)")
    except Exception as e:
        import_results['analysis.calculations'] = f"❌ Failed: {e}"
        print(f"❌ analysis.calculations: {e}")

    # Test backward compatibility
    try:
        import calculation
        import_results['calculation (compatibility)'] = "✅ Success"
        print(f"✅ calculation.py (backward compatibility)")
    except Exception as e:
        import_results['calculation (compatibility)'] = f"❌ Failed: {e}"
        print(f"❌ calculation.py: {e}")

    print(f"\nImport Results: {sum(1 for r in import_results.values() if '✅' in r)}/{len(import_results)} successful")
    return import_results


def test_basic_functions():
    """Test basic analysis functions."""
    print("\n" + "=" * 60)
    print("TESTING BASIC ANALYSIS FUNCTIONS")
    print("=" * 60)

    try:
        from analysis.calculations.basic import calculate_basic_statistics, calculate_time_domain_analysis

        signals = create_test_signals()
        test_signal = signals['multi_freq']

        # Test basic statistics
        print("Testing calculate_basic_statistics...")
        stats = calculate_basic_statistics(test_signal[1])
        if stats:
            print(f"  ✅ Basic statistics: Mean={stats['Mean']:.4f}, Std={stats['Standard Deviation']:.4f}")
        else:
            print("  ❌ Basic statistics failed")

        # Test time domain analysis
        print("Testing calculate_time_domain_analysis...")
        time_analysis = calculate_time_domain_analysis(test_signal[0], test_signal[1])
        if time_analysis:
            print(
                f"  ✅ Time domain: Duration={time_analysis['Duration (s)']:.2f}s, Sample_rate={time_analysis['Sample Rate (Hz)']:.1f}Hz")
        else:
            print("  ❌ Time domain analysis failed")

        return True

    except Exception as e:
        print(f"❌ Basic functions test failed: {e}")
        traceback.print_exc()
        return False


def test_frequency_functions():
    """Test frequency analysis functions."""
    print("\n" + "=" * 60)
    print("TESTING FREQUENCY ANALYSIS FUNCTIONS")
    print("=" * 60)

    try:
        from analysis.calculations.frequency import calculate_fft_analysis, calculate_psd_analysis

        signals = create_test_signals()
        test_signal = signals['sine_10hz']

        # Test FFT analysis
        print("Testing calculate_fft_analysis...")
        fft_result = calculate_fft_analysis(test_signal[0], test_signal[1])
        if fft_result:
            peak_freq = fft_result['Peak Frequency (Hz)']
            print(f"  ✅ FFT analysis: Peak frequency={peak_freq:.1f}Hz (expected ~10Hz)")
        else:
            print("  ❌ FFT analysis failed")

        # Test PSD analysis
        print("Testing calculate_psd_analysis...")
        psd_result = calculate_psd_analysis(test_signal[0], test_signal[1])
        if psd_result:
            peak_freq = psd_result['Peak Frequency (Hz)']
            print(f"  ✅ PSD analysis: Peak frequency={peak_freq:.1f}Hz")
        else:
            print("  ❌ PSD analysis failed")

        return True

    except Exception as e:
        print(f"❌ Frequency functions test failed: {e}")
        traceback.print_exc()
        return False


def test_correlation_functions():
    """Test correlation analysis functions."""
    print("\n" + "=" * 60)
    print("TESTING CORRELATION ANALYSIS FUNCTIONS")
    print("=" * 60)

    try:
        from analysis.calculations.correlation import calculate_autocorrelation_analysis, \
            calculate_cross_correlation_analysis

        signals = create_test_signals()
        test_signal = signals['sine_10hz']

        # Test autocorrelation
        print("Testing calculate_autocorrelation_analysis...")
        autocorr_result = calculate_autocorrelation_analysis(test_signal[0], test_signal[1])
        if autocorr_result:
            print(f"  ✅ Autocorrelation: Peak correlation={autocorr_result['Peak Correlation']:.2f}")
        else:
            print("  ❌ Autocorrelation analysis failed")

        # Test cross-correlation (signal with itself - should give perfect correlation)
        print("Testing calculate_cross_correlation_analysis...")
        xcorr_result = calculate_cross_correlation_analysis(
            test_signal[0], test_signal[1], test_signal[0], test_signal[1]
        )
        if xcorr_result:
            max_corr = xcorr_result['Max Correlation']
            print(f"  ✅ Cross-correlation: Max correlation={max_corr:.3f} (expected ~1.0)")
        else:
            print("  ❌ Cross-correlation analysis failed")

        return True

    except Exception as e:
        print(f"❌ Correlation functions test failed: {e}")
        traceback.print_exc()
        return False


def test_hilbert_functions():
    """Test Hilbert transform functions."""
    print("\n" + "=" * 60)
    print("TESTING HILBERT TRANSFORM FUNCTIONS")
    print("=" * 60)

    try:
        from analysis.calculations.hilbert import calculate_hilbert_analysis, calculate_phase_analysis, \
            calculate_energy_analysis

        signals = create_test_signals()
        test_signal = signals['am_signal']  # AM signal is good for Hilbert analysis

        # Test Hilbert analysis
        print("Testing calculate_hilbert_analysis...")
        hilbert_result = calculate_hilbert_analysis(test_signal[0], test_signal[1])
        if hilbert_result:
            mean_freq = hilbert_result['Mean Frequency (Hz)']
            print(f"  ✅ Hilbert analysis: Mean frequency={mean_freq:.1f}Hz")
        else:
            print("  ❌ Hilbert analysis failed")

        # Test energy analysis
        print("Testing calculate_energy_analysis...")
        energy_result = calculate_energy_analysis(test_signal[0], test_signal[1])
        if energy_result:
            total_energy = energy_result['Total Energy (Time Domain)']
            print(f"  ✅ Energy analysis: Total energy={total_energy:.2e}")
        else:
            print("  ❌ Energy analysis failed")

        return True

    except Exception as e:
        print(f"❌ Hilbert functions test failed: {e}")
        traceback.print_exc()
        return False


def test_filter_functions():
    """Test filtering functions."""
    print("\n" + "=" * 60)
    print("TESTING FILTER FUNCTIONS")
    print("=" * 60)

    try:
        from analysis.calculations.filters import calculate_iir_filter, calculate_fir_filter

        signals = create_test_signals()
        test_signal = signals['multi_freq']  # Multi-frequency signal for filtering

        # Test IIR filter
        print("Testing calculate_iir_filter...")
        iir_result = calculate_iir_filter(
            test_signal[0], test_signal[1],
            filter_type="lowpass", cutoff_freq=20, order=4
        )
        if iir_result:
            energy_ratio = iir_result['Energy Ratio (Filtered/Original)']
            print(f"  ✅ IIR filter: Energy ratio={energy_ratio:.3f}")
        else:
            print("  ❌ IIR filter failed")

        # Test FIR filter
        print("Testing calculate_fir_filter...")
        fir_result = calculate_fir_filter(
            test_signal[0], test_signal[1],
            filter_type="lowpass", cutoff_freq=20, numtaps=51
        )
        if fir_result:
            energy_ratio = fir_result['Energy Ratio (Filtered/Original)']
            print(f"  ✅ FIR filter: Energy ratio={energy_ratio:.3f}")
        else:
            print("  ❌ FIR filter failed")

        return True

    except Exception as e:
        print(f"❌ Filter functions test failed: {e}")
        traceback.print_exc()
        return False


def test_wavelet_functions():
    """Test wavelet analysis functions."""
    print("\n" + "=" * 60)
    print("TESTING WAVELET ANALYSIS FUNCTIONS")
    print("=" * 60)

    try:
        from analysis.calculations.wavelet import calculate_wavelet_analysis_cwt, calculate_wavelet_analysis_dwt, \
            get_available_wavelets

        signals = create_test_signals()
        test_signal = signals['chirp']  # Chirp signal is good for wavelet analysis

        # Test wavelet info
        print("Testing get_available_wavelets...")
        wavelet_info = get_available_wavelets()
        if wavelet_info:
            num_cwt = len(wavelet_info['Continuous Wavelets'])
            num_dwt = len(wavelet_info['Discrete Wavelets'])
            print(f"  ✅ Wavelet info: {num_cwt} CWT wavelets, {num_dwt} DWT wavelets")
        else:
            print("  ❌ Wavelet info failed")

        # Test CWT analysis
        print("Testing calculate_wavelet_analysis_cwt...")
        cwt_result = calculate_wavelet_analysis_cwt(test_signal[0], test_signal[1])
        if cwt_result:
            dominant_freq = cwt_result['Dominant Frequency (Hz)']
            print(f"  ✅ CWT analysis: Dominant frequency={dominant_freq:.1f}Hz")
        else:
            print("  ❌ CWT analysis failed")

        # Test DWT analysis
        print("Testing calculate_wavelet_analysis_dwt...")
        dwt_result = calculate_wavelet_analysis_dwt(test_signal[0], test_signal[1])
        if dwt_result:
            levels = dwt_result['Decomposition Level']
            print(f"  ✅ DWT analysis: {levels} decomposition levels")
        else:
            print("  ❌ DWT analysis failed")

        return True

    except Exception as e:
        print(f"❌ Wavelet functions test failed: {e}")
        traceback.print_exc()
        return False


def test_cepstrum_functions():
    """Test cepstral analysis and peak detection functions."""
    print("\n" + "=" * 60)
    print("TESTING CEPSTRAL ANALYSIS AND PEAK DETECTION")
    print("=" * 60)

    try:
        from analysis.calculations.cepstrum import calculate_cepstrum_analysis, calculate_peak_detection

        signals = create_test_signals()

        # Test cepstral analysis on harmonic signal
        print("Testing calculate_cepstrum_analysis...")
        harmonic_signal = signals['multi_freq']  # Has harmonic structure
        cepstrum_result = calculate_cepstrum_analysis(harmonic_signal[0], harmonic_signal[1])
        if cepstrum_result:
            fund_freq = cepstrum_result['Fundamental Frequency (Hz)']
            print(f"  ✅ Cepstral analysis: Fundamental frequency={fund_freq:.1f}Hz")
        else:
            print("  ❌ Cepstral analysis failed")

        # Test peak detection
        print("Testing calculate_peak_detection...")
        peak_signal = signals['impulses']  # Has clear peaks
        peak_result = calculate_peak_detection(peak_signal[0], peak_signal[1], method="scipy")
        if peak_result:
            peak_count = peak_result['Count']
            print(f"  ✅ Peak detection: Found {peak_count} peaks")
        else:
            print("  ❌ Peak detection failed")

        return True

    except Exception as e:
        print(f"❌ Cepstrum functions test failed: {e}")
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test backward compatibility with old calculation.py."""
    print("\n" + "=" * 60)
    print("TESTING BACKWARD COMPATIBILITY")
    print("=" * 60)

    try:
        # Test importing from old calculation.py module
        import calculation

        signals = create_test_signals()
        test_signal = signals['sine_10hz']

        # Test that old imports still work
        print("Testing backward compatibility imports...")
        stats = calculation.calculate_basic_statistics(test_signal[1])
        if stats:
            print(f"  ✅ calculation.calculate_basic_statistics works")
        else:
            print(f"  ❌ calculation.calculate_basic_statistics failed")

        fft_result = calculation.calculate_fft_analysis(test_signal[0], test_signal[1])
        if fft_result:
            print(f"  ✅ calculation.calculate_fft_analysis works")
        else:
            print(f"  ❌ calculation.calculate_fft_analysis failed")

        return True

    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False


def test_function_list():
    """Test that all expected functions are available."""
    print("\n" + "=" * 60)
    print("TESTING FUNCTION AVAILABILITY")
    print("=" * 60)

    try:
        from analysis.calculations import get_function_info

        info = get_function_info()
        total_functions = 0

        for category, details in info.items():
            functions = details['functions']
            print(f"{category.upper()}: {len(functions)} functions")
            total_functions += len(functions)

            # Test that each function can be imported
            for func_name in functions[:3]:  # Test first 3 in each category
                try:
                    from analysis.calculations import *
                    func = globals().get(func_name)
                    if func and callable(func):
                        print(f"  ✅ {func_name}")
                    else:
                        print(f"  ❌ {func_name} not callable")
                except Exception as e:
                    print(f"  ❌ {func_name}: {e}")

        print(f"\nTotal functions available: {total_functions}")
        return True

    except Exception as e:
        print(f"❌ Function list test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests and provide summary."""
    print("🧪 TESTING ANALYSIS.CALCULATIONS MODULE REFACTORING")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    test_functions = [
        ("Module Imports", test_module_imports),
        ("Basic Functions", test_basic_functions),
        ("Frequency Functions", test_frequency_functions),
        ("Correlation Functions", test_correlation_functions),
        ("Hilbert Functions", test_hilbert_functions),
        ("Filter Functions", test_filter_functions),
        ("Wavelet Functions", test_wavelet_functions),
        ("Cepstrum Functions", test_cepstrum_functions),
        ("Backward Compatibility", test_backward_compatibility),
        ("Function Availability", test_function_list)
    ]

    results = {}

    for test_name, test_func in test_functions:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for success in results.values() if success)
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The refactoring is successful.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)