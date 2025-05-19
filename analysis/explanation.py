from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QGroupBox,
    QGridLayout, QPushButton, QLabel
)


class ExplanationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_dialog = parent  # Store reference to parent dialog
        self.setup_ui()

    def setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        self.create_button_groups(scroll_layout)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

    def create_button_groups(self, parent_layout):
        # Basic Analysis group
        basic_group = QGroupBox("Basic Analysis")
        basic_layout = QGridLayout()
        basic_buttons = [
            "Statistics", "FFT Analysis", "Time Domain Analysis"
        ]
        self.add_buttons_to_grid(basic_layout, basic_buttons)
        basic_group.setLayout(basic_layout)
        parent_layout.addWidget(basic_group)

        # Advanced Analysis group
        advanced_group = QGroupBox("Advanced Analysis")
        advanced_layout = QGridLayout()
        advanced_buttons = [
            "Power Spectral Density", "Autocorrelation", "Peak Detection",
            "Hilbert Transform", "Energy Analysis", "Phase Analysis", "Cepstral Analysis"
        ]
        self.add_buttons_to_grid(advanced_layout, advanced_buttons)
        advanced_group.setLayout(advanced_layout)
        parent_layout.addWidget(advanced_group)

        # Cross Analysis group
        cross_group = QGroupBox("Cross Analysis")
        cross_layout = QGridLayout()
        cross_buttons = ["Cross-Correlation Analysis"]
        self.add_buttons_to_grid(cross_layout, cross_buttons)
        cross_group.setLayout(cross_layout)
        parent_layout.addWidget(cross_group)

        # Filtering group
        filter_group = QGroupBox("Filtering")
        filter_layout = QGridLayout()
        filter_buttons = ["Lowpass Filter (IIR)", "Highpass Filter (IIR)", "Bandpass Filter (IIR)",
                          "Lowpass Filter (FIR)", "Highpass Filter (FIR)", "Bandpass Filter (FIR)"]
        self.add_buttons_to_grid(filter_layout, filter_buttons)
        filter_group.setLayout(filter_layout)
        parent_layout.addWidget(filter_group)

        # Wavelet Analysis group
        wavelet_group = QGroupBox("Wavelet Analysis")
        wavelet_layout = QGridLayout()
        wavelet_buttons = ["CWT (Continuous Wavelet)", "DWT (Discrete Wavelet)",
                           "Wavelet Types", "Wavelet Applications"]
        self.add_buttons_to_grid(wavelet_layout, wavelet_buttons)
        wavelet_group.setLayout(wavelet_layout)
        parent_layout.addWidget(wavelet_group)

    def add_buttons_to_grid(self, layout, button_texts, cols=3):
        for i, text in enumerate(button_texts):
            row = i // cols
            col = i % cols
            button = QPushButton(text)
            # Connect to new method that shows help in results area
            button.clicked.connect(lambda checked, t=text: self.show_help(t))
            layout.addWidget(button, row, col)

    def show_default_help(self):
        self.help_text.setHtml("""
        <h3>Signal Analysis Help</h3>
        <p>Click on any button above to see detailed information about that analysis method.</p>
        <p>This help section will explain:</p>
        <ul>
            <li>What the analysis method does</li>
            <li>When to use it</li>
            <li>Limitations and considerations</li>
            <li>How to interpret the results</li>
        </ul>
        """)

    def show_help(self, topic):
        """Display help content for the selected topic."""
        # Dictionary of help content for each topic
        help_content = {
            "Statistics": """
                <h3>Basic Signal Statistics</h3>
                <p>Basic statistics provide fundamental insights about the signal's characteristics:</p>
                <ul>
                    <li><b>Mean:</b> The average value of the signal, indicating central tendency.
                        <br>Formula: μ = (1/N)·∑x(i)</li>
                    <li><b>Median:</b> The middle value when signal values are ordered, less affected by outliers than mean.</li>
                    <li><b>Minimum/Maximum:</b> The smallest and largest values in the signal.</li>
                    <li><b>Range:</b> The difference between maximum and minimum values.</li>
                    <li><b>Standard Deviation:</b> Measures the amount of variation or dispersion in the signal.
                        <br>Formula: σ = √[(1/N)·∑(x(i)-μ)²]</li>
                    <li><b>Variance:</b> Square of standard deviation, measures signal power variation around the mean.</li>
                    <li><b>Root Mean Square (RMS):</b> Square root of the average of squared values, relates to signal energy.
                        <br>Formula: RMS = √[(1/N)·∑x(i)²]</li>
                    <li><b>Skewness:</b> Measures asymmetry of the signal distribution. Positive values indicate right-tailed distribution.</li>
                    <li><b>Kurtosis:</b> Measures the "tailedness" of the distribution (peakedness/flatness).
                        <br>Higher values indicate more extreme outliers.</li>
                    <li><b>Interquartile Range (IQR):</b> The range between the 25th and 75th percentiles, robust to outliers.</li>
                </ul>
                <p><b>When to use:</b> These statistics provide a basic quantitative description of your signal and can identify potential issues or characteristics.</p>
            """,

            "FFT Analysis": """
                <h3>FFT Analysis</h3>
                <p>Fast Fourier Transform converts a signal from the time domain to the frequency domain:</p>
                <ul>
                    <li><b>Purpose:</b> Identifies frequency components present in the signal.</li>
                    <li><b>Usage:</b> Detect periodic patterns, dominant frequencies, and harmonic content.</li>
                    <li><b>Mathematical basis:</b> Decomposes a signal into a sum of sinusoids of different frequencies.</li>
                </ul>
                <p><b>Interpretation:</b> 
                    <ul>
                        <li>Peaks in the frequency spectrum indicate strong periodic components at those frequencies.</li>
                        <li>Broader peaks suggest frequency variation or modulation.</li>
                        <li>Evenly spaced harmonics indicate a complex periodic signal.</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Assumes signal stationarity (frequency content doesn't change over time).</li>
                        <li>May not capture time-varying frequency content.</li>
                        <li>Subject to spectral leakage if signal period doesn't match window size.</li>
                        <li>Limited frequency resolution for short signals.</li>
                    </ul>
                </p>
            """,

            "Time Domain Analysis": """
                <h3>Time Domain Analysis</h3>
                <p>Examines signal characteristics directly in the time domain:</p>
                <ul>
                    <li><b>Duration:</b> Total time span of the signal.</li>
                    <li><b>Sample Rate:</b> Number of samples per second.</li>
                    <li><b>Zero Crossings:</b> Number of times the signal crosses the zero level, related to frequency content.</li>
                    <li><b>Signal Energy:</b> Sum of squared sample values (∑x²), represents total energy contained in the signal.</li>
                    <li><b>Signal Power:</b> Average power of the signal over time (energy/duration).</li>
                    <li><b>Crest Factor:</b> Ratio of peak value to RMS value, indicates signal impulsiveness.
                        <br>High values suggest transients or impulses.</li>
                </ul>
                <p><b>When to use:</b> For initial signal characterization, identifying abrupt changes, assessing signal quality, or determining appropriate processing methods.</p>
                <p><b>Limitations:</b> May not easily reveal frequency-related information or subtle patterns that frequency analysis would highlight.</p>
            """,

            "Power Spectral Density": """
                <h3>Power Spectral Density (PSD)</h3>
                <p>Measures how signal power is distributed across frequency:</p>
                <ul>
                    <li><b>Purpose:</b> Shows which frequencies contain the signal's power.</li>
                    <li><b>Usage:</b> Identify dominant frequencies, noise sources, or resonance.</li>
                    <li><b>Formula:</b> Squared magnitude of the Fourier transform, normalized by signal length.</li>
                    <li><b>Units:</b> Power per frequency (e.g., V²/Hz).</li>
                </ul>
                <p><b>Interpretation:</b> 
                    <ul>
                        <li>Areas with high PSD values indicate frequency bands that contribute significantly to the signal's power.</li>
                        <li>Peak width indicates stability of frequency component (narrower = more stable).</li>
                        <li>Log scale often used to visualize both strong and weak components.</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul> 
                        <li>Resolution depends on signal length and windowing.</li>
                        <li>Assumes signal is statistically stationary.</li>
                        <li>Averaging may be needed for noisy signals.</li>
                    </ul>
                </p>
            """,

            "Autocorrelation": """
                <h3>Autocorrelation</h3>
                <p>Measures similarity between a signal and a time-shifted version of itself:</p>
                <ul>
                    <li><b>Purpose:</b> Detect repeating patterns, periodicities, or signal memory.</li>
                    <li><b>Usage:</b> Find hidden periodicities, estimate fundamental frequency, detect signal redundancy.</li>
                    <li><b>Formula:</b> R(τ) = E[x(t)·x(t-τ)], where τ is the time lag.</li>
                </ul>
                <p><b>Interpretation:</b>
                    <ul>
                        <li>Peak at zero lag (always present) represents signal energy.</li>
                        <li>Secondary peaks indicate periodic components.</li>
                        <li>Distance between peaks represents period of repetitive pattern.</li>
                        <li>Decay rate indicates "memory" in the signal (how quickly it becomes uncorrelated with itself).</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>May be affected by noise or trends in the signal.</li>
                        <li>Multiple periodicities can create complex patterns that are difficult to interpret.</li>
                        <li>Requires sufficient signal length to detect long-period patterns.</li>
                    </ul>
                </p>
            """,

            "Peak Detection": """
                <h3>Peak Detection</h3>
                <p>Identifies local maxima (peaks) in the signal:</p>
                <ul>
                    <li><b>Purpose:</b> Locate significant events or features in the signal.</li>
                    <li><b>Usage:</b> Count events, measure intervals between events, identify important signal points.</li>
                    <li><b>Parameters:</b>
                        <ul>
                            <li><b>Height threshold:</b> Minimum amplitude to be considered a peak.</li>
                            <li><b>Distance:</b> Minimum separation between adjacent peaks.</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Applications:</b>
                    <ul>
                        <li>Heartbeat detection in ECG signals.</li>
                        <li>Event counting in sensor data.</li>
                        <li>Pulse detection in various signals.</li>
                        <li>Peak analysis in spectral data.</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Sensitivity to threshold settings and noise.</li>
                        <li>May miss closely spaced peaks due to distance parameter.</li>
                        <li>Difficulty with very broad or asymmetric peaks.</li>
                        <li>Baseline drift can affect detection accuracy.</li>
                    </ul>
                </p>
            """,

            "Lowpass Filter (IIR)": """
                <h3>Lowpass Filter (IIR)</h3>
                <p>Allows low-frequency components to pass while attenuating higher frequencies using an IIR (Infinite Impulse Response) Butterworth filter.</p>
                <ul>
                    <li><b>Purpose:</b> Remove high-frequency noise or smooth the signal.</li>
                    <li><b>Parameters:</b>
                        <ul>
                            <li><b>Cutoff frequency:</b> Frequency above which components are attenuated.</li>
                            <li><b>Filter order:</b> Determines the steepness of the filter transition.</li>
                        </ul>
                    </li>
                    <li><b>Advantages:</b> Efficient for real-time filtering due to recursive nature.</li>
                    <li><b>Limitations:</b> May introduce phase distortion; requires careful design near Nyquist frequency.</li>
                </ul>
            """,

            "Highpass Filter (IIR)": """
                <h3>Highpass Filter (IIR)</h3>
                <p>Allows high-frequency components to pass while attenuating lower frequencies using an IIR Butterworth filter.</p>
                <ul>
                    <li><b>Purpose:</b> Remove baseline drift, DC offset, or slow trends.</li>
                    <li><b>Parameters:</b>
                        <ul>
                            <li><b>Cutoff frequency:</b> Frequency below which components are attenuated.</li>
                            <li><b>Filter order:</b> Controls the steepness of the transition.</li>
                        </ul>
                    </li>
                    <li><b>Use cases:</b> Useful for highlighting transients or quick events.</li>
                    <li><b>Limitations:</b> Can amplify high-frequency noise; phase distortion possible.</li>
                </ul>
            """,

            "Bandpass Filter (IIR)": """
                <h3>Bandpass Filter (IIR)</h3>
                <p>Passes only a specific frequency band using an IIR Butterworth filter.</p>
                <ul>
                    <li><b>Purpose:</b> Isolate components between two cutoff frequencies.</li>
                    <li><b>Parameters:</b>
                        <ul>
                            <li><b>Lower cutoff:</b> Start of passband.</li>
                            <li><b>Upper cutoff:</b> End of passband.</li>
                            <li><b>Order:</b> Steepness of transition zones.</li>
                        </ul>
                    </li>
                    <li><b>Use cases:</b> Extract specific modes (e.g., vibrations, EEG bands).</li>
                    <li><b>Limitations:</b> Band edges must be within Nyquist limit and correctly ordered.</li>
                </ul>
            """,

            "Lowpass Filter (FIR)": """
                <h3>Lowpass Filter (FIR)</h3>
                <p>Finite Impulse Response filter that allows low frequencies and blocks high frequencies.</p>
                <ul>
                    <li><b>Window-based FIR design:</b> Uses firwin method with customizable window (e.g., Hamming).</li>
                    <li><b>Cutoff frequency:</b> Frequency above which attenuation begins.</li>
                    <li><b>Taps:</b> Number of coefficients (higher = better selectivity, more computation).</li>
                    <li><b>Advantages:</b> Linear phase response — no phase distortion.</li>
                    <li><b>Limitations:</b> Requires longer filters than IIR to achieve sharp transitions.</li>
                </ul>
            """,

            "Highpass Filter (FIR)": """
                <h3>Highpass Filter (FIR)</h3>
                <p>Removes low-frequency components using a window-based FIR filter.</p>
                <ul>
                    <li><b>Ideal for:</b> DC removal, edge detection, and emphasizing rapid transitions.</li>
                    <li><b>Parameters:</b>
                        <ul>
                            <li><b>Cutoff frequency</b> (start of passband)</li>
                            <li><b>Taps:</b> Number of FIR coefficients</li>
                            <li><b>Window type:</b> e.g., Hamming, Blackman</li>
                        </ul>
                    </li>
                    <li><b>Note:</b> FIR filters maintain linear phase across the spectrum.</li>
                </ul>
            """,

            "Bandpass Filter (FIR)": """
                <h3>Bandpass Filter (FIR)</h3>
                <p>Passes only frequencies between two boundaries using a finite impulse response filter.</p>
                <ul>
                    <li><b>Parameters:</b>
                        <ul>
                            <li><b>Lower cutoff</b> and <b>upper cutoff</b> in Hz</li>
                            <li><b>Taps:</b> Higher values = narrower band but longer response</li>
                            <li><b>Window:</b> Affects stopband attenuation (Hamming, Kaiser, etc.)</li>
                        </ul>
                    </li>
                    <li><b>Advantages:</b> No phase distortion, stable.</li>
                    <li><b>Limitations:</b> Requires more taps (longer filters) for sharp bands.</li>
                </ul>
            """,

            "Hilbert Transform": """
                <h3>Hilbert Transform</h3>
                <p>Creates the analytic signal and extracts instantaneous attributes:</p>
                <ul>
                    <li><b>Purpose:</b> Extract amplitude envelope, instantaneous phase, and frequency.</li>
                    <li><b>Usage:</b> Analyze modulated signals, extract signal envelope, frequency modulation analysis.</li>
                    <li><b>Components produced:</b>
                        <ul>
                            <li><b>Amplitude envelope:</b> Instantaneous amplitude of the signal.</li>
                            <li><b>Instantaneous phase:</b> Phase angle of the analytic signal.</li>
                            <li><b>Instantaneous frequency:</b> Rate of change of the phase.</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Applications:</b>
                    <ul>
                        <li>Demodulation of AM signals</li>
                        <li>Analysis of frequency modulation</li>
                        <li>Extracting temporal structure in complex signals</li>
                        <li>Speech and audio processing</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Best suited for narrowband signals (limited frequency range).</li>
                        <li>Instantaneous frequency may be difficult to interpret for broadband signals.</li>
                        <li>Edge effects at signal boundaries.</li>
                        <li>Phase unwrapping may introduce errors in instantaneous frequency.</li>
                    </ul>
                </p>
            """,

            "Energy Analysis": """
                <h3>Energy Analysis</h3>
                <p>Examines how signal energy is distributed over time intervals:</p>
                <ul>
                    <li><b>Purpose:</b> Identify energy variations over time, detect events or changes in signal activity.</li>
                    <li><b>Calculation:</b> Energy in interval = sum of squared values within each time window.</li>
                    <li><b>Applications:</b>
                        <ul>
                            <li>Speech segment detection</li>
                            <li>Activity monitoring in sensors</li>
                            <li>Transient detection</li>
                            <li>Signal quality assessment over time</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Interpretation:</b>
                    <ul>
                        <li>High energy intervals indicate greater signal activity or amplitude.</li>
                        <li>Sudden changes in energy can indicate events or transitions.</li>
                        <li>Energy distribution can reveal patterns in signal activity over time.</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Results depend on interval size choice (too small: noisy results, too large: temporal details lost).</li>
                        <li>May be sensitive to outliers or noise spikes.</li>
                        <li>Doesn't preserve frequency information.</li>
                        <li>May miss low-amplitude but important signal features.</li>
                    </ul>
                </p>
            """,

            "Phase Analysis": """
                <h3>Phase Analysis</h3>
                <p>Studies the phase behavior of a signal:</p>
                <ul>
                    <li><b>Purpose:</b> Understand angular position in oscillations, detect phase shifts or synchronization.</li>
                    <li><b>Calculation:</b> Extracts phase angle from analytic signal via Hilbert transform.</li>
                    <li><b>Key metrics:</b>
                        <ul>
                            <li><b>Phase consistency:</b> How stable the phase progression is.</li>
                            <li><b>Phase velocity:</b> Rate of phase change (related to frequency).</li>
                            <li><b>Phase jumps:</b> Sudden changes in phase that may indicate events.</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Applications:</b>
                    <ul>
                        <li>Brain connectivity analysis (phase synchronization)</li>
                        <li>Communication signal demodulation</li>
                        <li>Mechanical vibration analysis</li>
                        <li>Detecting coherence between signals</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Phase unwrapping may introduce artifacts in long signals.</li>
                        <li>Interpretation can be challenging for broadband signals.</li>
                        <li>Sensitive to noise, especially at low amplitudes.</li>
                        <li>Phase is only meaningful for oscillatory signals.</li>
                    </ul>
                </p>
            """,

            "Cepstral Analysis": """
                <h3>Cepstral Analysis</h3>
                <p>The "spectrum of the logarithm of the spectrum" - reveals periodic patterns in spectra:</p>
                <ul>
                    <li><b>Purpose:</b> Detect periodic structures in the spectrum, separate source and filter components.</li>
                    <li><b>Formula:</b> Inverse Fourier transform of the logarithm of the magnitude spectrum.</li>
                    <li><b>Key concepts:</b>
                        <ul>
                            <li><b>Quefrency:</b> The x-axis in cepstral domain (a form of time).</li>
                            <li><b>Rahmonics:</b> Peaks in the cepstrum (analogous to harmonics).</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Applications:</b>
                    <ul>
                        <li>Pitch detection in speech (fundamental frequency)</li>
                        <li>Echo detection and removal</li>
                        <li>Speech processing and recognition</li>
                        <li>Mechanical fault diagnosis (detecting periodicities)</li>
                    </ul>
                </p>
                <p><b>Interpretation:</b>
                    <ul>
                        <li>Peaks in the cepstrum represent periodic components in the original spectrum.</li>
                        <li>First significant peak indicates fundamental period or echo delay.</li>
                        <li>Lower quefrencies relate to spectral envelope, higher to fine structure.</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Interpretation can be complex without domain knowledge.</li>
                        <li>May require pre-processing for optimal results.</li>
                        <li>Performance degrades in noisy signals.</li>
                        <li>Less effective for signals with rapidly changing pitch.</li>
                    </ul>
                </p>
            """,

            "Cross-Correlation Analysis": """
                <h3>Cross-Correlation Analysis</h3>
                <p>Measures similarity between two different signals as a function of time lag:</p>
                <ul>
                    <li><b>Purpose:</b> Determine time delay between signals, measure similarity, detect common patterns.</li>
                    <li><b>Formula:</b> (f⋆g)(τ) = ∫f*(t)·g(t+τ)dt (or discrete equivalent)</li>
                    <li><b>Key results:</b>
                        <ul>
                            <li><b>Maximum correlation value:</b> Indicates degree of similarity (0-1 when normalized).</li>
                            <li><b>Lag at maximum:</b> Time offset that best aligns the signals.</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Applications:</b>
                    <ul>
                        <li>Finding signal delays (e.g., acoustics, radar)</li>
                        <li>Pattern detection across multiple sensors</li>
                        <li>Template matching in signal processing</li>
                        <li>Time difference of arrival (TDOA) calculations</li>
                        <li>Measuring similarity between related signals</li>
                    </ul>
                </p>
                <p><b>Interpretation:</b>
                    <ul>
                        <li>The peak in cross-correlation indicates the time lag that maximizes similarity.</li>
                        <li>Higher correlation values suggest stronger relationships between signals.</li>
                        <li>Multiple peaks may indicate repeating patterns or multiple path propagation.</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>Assumes signals are related and have similar structures.</li>
                        <li>May be misleading if signals have different amplitude scales (normalization helps).</li>
                        <li>Sensitive to noise and outliers.</li>
                        <li>May detect spurious correlations in complex signals.</li>
                    </ul>
                </p>
            """,

            "Wavelet Transform": """
                <h3>Wavelet Transform</h3>
                <p>Decomposes a signal into components at different scales/frequencies with time localization:</p>
                <ul>
                    <li><b>Purpose:</b> Multi-resolution analysis providing both time and frequency information.</li>
                    <li><b>Key concepts:</b>
                        <ul>
                            <li><b>Approximation:</b> Low-frequency components of the signal.</li>
                            <li><b>Details:</b> High-frequency components at different scales.</li>
                            <li><b>Decomposition Level:</b> Number of scales analyzed (more levels = finer frequency division).</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Advantages over Fourier Transform:</b>
                    <ul>
                        <li>Provides both time and frequency information simultaneously.</li>
                        <li>Better suited for non-stationary signals with changing frequency content.</li>
                        <li>Adaptive resolution (fine time resolution at high frequencies, fine frequency resolution at low frequencies).</li>
                        <li>More effective at capturing transient events.</li>
                    </ul>
                </p>
                <p><b>Applications:</b>
                    <ul>
                        <li>Identifying transient events at different time scales</li>
                        <li>Denoising signals while preserving important features</li>
                        <li>Feature extraction for classification tasks</li>
                        <li>Image and audio compression</li>
                        <li>Biomedical signal processing (EEG, ECG analysis)</li>
                    </ul>
                </p>
                <p><b>Limitations:</b>
                    <ul>
                        <li>More complex to interpret than traditional spectral analysis.</li>
                        <li>Choice of wavelet family affects results.</li>
                        <li>Edge effects at signal boundaries.</li>
                        <li>Computational intensity increases with decomposition levels.</li>
                    </ul>
                </p>
            """,

            "CWT (Continuous Wavelet)": """
                <h3>Continuous Wavelet Transform (CWT)</h3>
                <p>Provides a time-frequency representation by convolving the signal with scaled and shifted wavelets.</p>
                <ul>
                    <li><b>Wavelet types:</b> Morlet, Mexican Hat, Complex Morlet.</li>
                    <li><b>Scales:</b> Determine resolution and frequency range.</li>
                    <li><b>Output:</b> Coefficients matrix, frequency vector, scalogram (heatmap).</li>
                    <li><b>Advantages:</b> Good time localization; handles non-stationary signals well.</li>
                    <li><b>Limitations:</b> Higher computation; less intuitive than FFT.</li>
                </ul>
            """,

            "DWT (Discrete Wavelet)": """
                <h3>Discrete Wavelet Transform (DWT)</h3>
                <p>Decomposes a signal into approximation and detail coefficients at different levels using orthogonal wavelets.</p>
                <ul>
                    <li><b>Levels:</b> Control number of scales in decomposition.</li>
                    <li><b>Wavelet types:</b> Daubechies, Symlets, Coiflets.</li>
                    <li><b>Output:</b> Set of coefficient arrays (per level), useful for denoising and feature extraction.</li>
                    <li><b>Advantages:</b> Efficient, multiscale representation; suitable for compression.</li>
                    <li><b>Limitations:</b> No redundancy (unlike CWT); limited frequency resolution.</li>
                </ul>
            """,

            "Wavelet Types": """
                <h3>Wavelet Types</h3>
                <p>Different wavelet families have unique characteristics suited to specific signal types:</p>
                <ul>
                    <li><b>Haar:</b>
                        <ul>
                            <li>The simplest wavelet, resembling a step function.</li>
                            <li>Good for detecting abrupt transitions and edges.</li>
                            <li>Limited smoothness, resulting in blocky approximations.</li>
                            <li>Best for: Signals with sudden jumps or digital/binary signals.</li>
                        </ul>
                    </li>
                    <li><b>Daubechies (db4, db8):</b>
                        <ul>
                            <li>Compactly supported wavelets with maximum number of vanishing moments.</li>
                            <li>Good balance between smoothness and localization.</li>
                            <li>Higher order (db8) provides smoother representation than lower order (db4).</li>
                            <li>Best for: General-purpose analysis, signals with polynomial trends.</li>
                        </ul>
                    </li>
                    <li><b>Symlets (sym4, sym8):</b>
                        <ul>
                            <li>Modified version of Daubechies wavelets with increased symmetry.</li>
                            <li>Nearly symmetrical, reducing phase distortion.</li>
                            <li>Good time-frequency localization properties.</li>
                            <li>Best for: Applications where phase information is important.</li>
                        </ul>
                    </li>
                    <li><b>Coiflets (coif1, coif3):</b>
                        <ul>
                            <li>More symmetrical than Daubechies wavelets.</li>
                            <li>Have vanishing moments for both wavelet and scaling functions.</li>
                            <li>Good for preserving signal features during analysis/reconstruction.</li>
                            <li>Best for: Function approximation, signals requiring accurate reconstruction.</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Selection criteria:</b>
                    <ul>
                        <li>Signal characteristics (smooth vs. abrupt changes)</li>
                        <li>Analysis goals (detection, denoising, compression)</li>
                        <li>Required frequency resolution</li>
                        <li>Computational constraints</li>
                    </ul>
                </p>
            """,

            "Wavelet Applications": """
                <h3>Wavelet Applications</h3>
                <p>Common applications of wavelet analysis in signal processing:</p>
                <ul>
                    <li><b>Signal Denoising:</b>
                        <ul>
                            <li>Wavelets can separate signal from noise at different scales.</li>
                            <li>Thresholding detail coefficients removes noise while preserving signal features.</li>
                            <li>More effective than traditional filtering for preserving edges and transients.</li>
                        </ul>
                    </li>
                    <li><b>Feature Detection:</b>
                        <ul>
                            <li>Identify specific patterns or events at appropriate scales.</li>
                            <li>Useful for detecting discontinuities, spikes, or other transient events.</li>
                            <li>Can locate features that are difficult to detect in time or frequency domain alone.</li>
                        </ul>
                    </li>
                    <li><b>Compression:</b>
                        <ul>
                            <li>Many signals can be represented with few wavelet coefficients.</li>
                            <li>Discarding small coefficients enables efficient storage while preserving essential information.</li>
                            <li>Basis for JPEG2000 image compression standard.</li>
                        </ul>
                    </li>
                    <li><b>Component Separation:</b>
                        <ul>
                            <li>Isolate different physical processes operating at different scales.</li>
                            <li>Extract specific signal components by focusing on relevant decomposition levels.</li>
                            <li>Separate fast vs. slow processes in complex signals.</li>
                        </ul>
                    </li>
                    <li><b>Non-Stationary Signal Analysis:</b>
                        <ul>
                            <li>Track how frequency content changes over time.</li>
                            <li>Identify time-varying behavior that Fourier analysis would miss.</li>
                            <li>Particularly valuable for biological signals, seismic data, or financial time series.</li>
                        </ul>
                    </li>
                </ul>
                <p><b>Implementation approach:</b>
                    <ul>
                        <li>First select appropriate wavelet family for your signal.</li>
                        <li>Choose decomposition level based on frequency resolution needs.</li>
                        <li>Examine both approximation (general trend) and details (specific scales).</li>
                        <li>Consider energy distribution across levels to identify important components.</li>
                    </ul>
                </p>
            """
        }

        if self.parent_dialog:
            if topic in help_content:
                self.parent_dialog.show_help_in_results(topic, help_content[topic])
            else:
                self.parent_dialog.show_help_in_results(topic, f"<h3>No help available for '{topic}'</h3><p>Please select another topic.</p>")