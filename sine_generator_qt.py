#!/usr/bin/env python3
"""
Triple Oscillator Synth with ADSR Envelope and Filter
Features:
- Three independent oscillators with waveform selection
- ADSR envelope generator
- Low-pass filter with cutoff and resonance
- Mixer with gain control per oscillator
- Professional synth-style interface
"""

import sys
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QDial, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class EnvelopeGenerator:
    """ADSR Envelope Generator"""
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.attack = 0.01   # seconds
        self.decay = 0.1     # seconds
        self.sustain = 0.7   # level (0-1)
        self.release = 0.3   # seconds

        self.phase = 'idle'  # idle, attack, decay, sustain, release
        self.level = 0.0
        self.samples_in_phase = 0

    def trigger(self):
        """Trigger note on (start attack phase)"""
        self.phase = 'attack'
        self.samples_in_phase = 0

    def release_note(self):
        """Trigger note off (start release phase)"""
        if self.phase != 'idle':
            self.phase = 'release'
            self.samples_in_phase = 0

    def process(self, num_samples):
        """Generate envelope for num_samples"""
        output = np.zeros(num_samples)

        for i in range(num_samples):
            if self.phase == 'idle':
                self.level = 0.0

            elif self.phase == 'attack':
                attack_samples = max(1, int(self.attack * self.sample_rate))
                self.level = self.samples_in_phase / attack_samples
                self.samples_in_phase += 1

                if self.level >= 1.0:
                    self.level = 1.0
                    self.phase = 'decay'
                    self.samples_in_phase = 0

            elif self.phase == 'decay':
                decay_samples = max(1, int(self.decay * self.sample_rate))
                progress = self.samples_in_phase / decay_samples
                self.level = 1.0 - progress * (1.0 - self.sustain)
                self.samples_in_phase += 1

                if self.samples_in_phase >= decay_samples:
                    self.level = self.sustain
                    self.phase = 'sustain'
                    self.samples_in_phase = 0

            elif self.phase == 'sustain':
                self.level = self.sustain

            elif self.phase == 'release':
                release_samples = max(1, int(self.release * self.sample_rate))
                # Start from current level, not from sustain
                start_level = self.level if self.samples_in_phase == 0 else self.level
                progress = self.samples_in_phase / release_samples
                self.level = start_level * (1.0 - progress)
                self.samples_in_phase += 1

                if self.samples_in_phase >= release_samples:
                    self.level = 0.0
                    self.phase = 'idle'
                    self.samples_in_phase = 0

            output[i] = self.level

        return output


class LowPassFilter:
    """Simple low-pass filter with resonance"""
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.cutoff = 5000.0  # Hz
        self.resonance = 0.0  # 0-1

        # Filter state
        self.y1 = 0.0
        self.y2 = 0.0
        self.x1 = 0.0
        self.x2 = 0.0

    def process(self, input_signal):
        """Apply low-pass filter to input signal"""
        output = np.zeros_like(input_signal)

        # Calculate filter coefficients
        freq = self.cutoff / self.sample_rate
        q = 1.0 + self.resonance * 10.0  # Map 0-1 to 1-11

        omega = 2.0 * np.pi * freq
        sn = np.sin(omega)
        cs = np.cos(omega)
        alpha = sn / (2.0 * q)

        a0 = 1.0 + alpha
        a1 = -2.0 * cs
        a2 = 1.0 - alpha
        b0 = (1.0 - cs) / 2.0
        b1 = 1.0 - cs
        b2 = (1.0 - cs) / 2.0

        # Normalize
        a1 /= a0
        a2 /= a0
        b0 /= a0
        b1 /= a0
        b2 /= a0

        # Apply filter
        for i in range(len(input_signal)):
            x = input_signal[i]
            y = b0 * x + b1 * self.x1 + b2 * self.x2 - a1 * self.y1 - a2 * self.y2

            self.x2 = self.x1
            self.x1 = x
            self.y2 = self.y1
            self.y1 = y

            output[i] = y

        return output


class SineWaveGenerator(QMainWindow):
    def __init__(self):
        super().__init__()

        # Audio parameters
        self.sample_rate = 44100
        self.stream = None

        # Oscillator 1 parameters
        self.freq1 = 440.0
        self.phase1 = 0
        self.osc1_on = False
        self.waveform1 = "Sine"

        # Oscillator 2 parameters
        self.freq2 = 440.0
        self.phase2 = 0
        self.osc2_on = False
        self.waveform2 = "Sine"

        # Oscillator 3 parameters
        self.freq3 = 440.0
        self.phase3 = 0
        self.osc3_on = False
        self.waveform3 = "Sine"

        # Mixer parameters (0.0 to 1.0)
        self.gain1 = 0.33
        self.gain2 = 0.33
        self.gain3 = 0.33

        # Envelope generators (one per oscillator)
        self.env1 = EnvelopeGenerator(self.sample_rate)
        self.env2 = EnvelopeGenerator(self.sample_rate)
        self.env3 = EnvelopeGenerator(self.sample_rate)

        # Filter
        self.filter = LowPassFilter(self.sample_rate)

        # Logarithmic scale parameters
        self.min_freq = 20.0
        self.max_freq = 5000.0
        self.min_log = np.log10(self.min_freq)
        self.max_log = np.log10(self.max_freq)

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Triple Oscillator Synth")
        self.setFixedSize(900, 650)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # TOP SECTION: Four column layout (Oscillators + Mixer)
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(10)

        # OSCILLATOR 1 COLUMN
        osc1_widget = self.create_oscillator_column("Oscillator 1", 1)
        columns_layout.addWidget(osc1_widget, 1)

        # OSCILLATOR 2 COLUMN
        osc2_widget = self.create_oscillator_column("Oscillator 2", 2)
        columns_layout.addWidget(osc2_widget, 1)

        # OSCILLATOR 3 COLUMN
        osc3_widget = self.create_oscillator_column("Oscillator 3", 3)
        columns_layout.addWidget(osc3_widget, 1)

        # MIXER COLUMN
        mixer_widget = self.create_mixer_column()
        columns_layout.addWidget(mixer_widget, 1)

        main_layout.addLayout(columns_layout)

        # BOTTOM SECTION: ADSR and Filter
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(20)

        # ADSR Section
        adsr_widget = self.create_adsr_section()
        bottom_layout.addWidget(adsr_widget, 2)

        # Filter Section
        filter_widget = self.create_filter_section()
        bottom_layout.addWidget(filter_widget, 1)

        main_layout.addLayout(bottom_layout)

        # Apply dark theme styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QDial {
                background-color: #3c3c3c;
            }
            QComboBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #ffffff;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                color: #ffffff;
                selection-background-color: #555555;
            }
        """)

    def create_oscillator_column(self, title, osc_num):
        """Create an oscillator column with waveform selector and frequency knob"""
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setSpacing(8)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)

        # Waveform selector
        waveform_combo = QComboBox()
        waveform_combo.addItems(["Sine", "Sawtooth", "Square"])
        waveform_combo.setFont(QFont("Arial", 9))

        if osc_num == 1:
            self.waveform1_combo = waveform_combo
            waveform_combo.currentTextChanged.connect(lambda w: self.update_waveform(1, w))
        elif osc_num == 2:
            self.waveform2_combo = waveform_combo
            waveform_combo.currentTextChanged.connect(lambda w: self.update_waveform(2, w))
        else:
            self.waveform3_combo = waveform_combo
            waveform_combo.currentTextChanged.connect(lambda w: self.update_waveform(3, w))

        layout.addWidget(waveform_combo)

        layout.addStretch(1)

        # Frequency knob
        freq_knob = QDial()
        freq_knob.setMinimum(0)
        freq_knob.setMaximum(1000)
        freq_knob.setNotchesVisible(True)
        freq_knob.setWrapping(False)
        freq_knob.setFixedSize(100, 100)

        # Set initial position for 440 Hz
        initial_position = int(1000 * (np.log10(440) - self.min_log) / (self.max_log - self.min_log))
        freq_knob.setValue(initial_position)

        # Connect to appropriate oscillator
        if osc_num == 1:
            self.freq1_knob = freq_knob
            freq_knob.valueChanged.connect(lambda v: self.update_frequency(1, v))
        elif osc_num == 2:
            self.freq2_knob = freq_knob
            freq_knob.valueChanged.connect(lambda v: self.update_frequency(2, v))
        else:
            self.freq3_knob = freq_knob
            freq_knob.valueChanged.connect(lambda v: self.update_frequency(3, v))

        # Center the knob
        knob_layout = QHBoxLayout()
        knob_layout.addStretch(1)
        knob_layout.addWidget(freq_knob)
        knob_layout.addStretch(1)
        layout.addLayout(knob_layout)

        # Frequency display
        freq_label = QLabel("440.0 Hz")
        freq_label.setAlignment(Qt.AlignCenter)
        freq_label.setFont(QFont("Arial", 10, QFont.Bold))

        if osc_num == 1:
            self.freq1_label = freq_label
        elif osc_num == 2:
            self.freq2_label = freq_label
        else:
            self.freq3_label = freq_label

        layout.addWidget(freq_label)

        # Range labels
        range_layout = QHBoxLayout()
        min_label = QLabel("20Hz")
        min_label.setFont(QFont("Arial", 7))
        max_label = QLabel("5kHz")
        max_label.setFont(QFont("Arial", 7))
        range_layout.addWidget(min_label)
        range_layout.addStretch(1)
        range_layout.addWidget(max_label)
        layout.addLayout(range_layout)

        layout.addStretch(1)

        # On/Off button
        osc_button = QPushButton("OFF")
        osc_button.setFont(QFont("Arial", 9, QFont.Bold))
        osc_button.setFixedSize(50, 50)
        osc_button.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                color: #888888;
                border: none;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
            }
        """)

        if osc_num == 1:
            self.osc1_button = osc_button
            osc_button.clicked.connect(lambda: self.toggle_oscillator(1))
        elif osc_num == 2:
            self.osc2_button = osc_button
            osc_button.clicked.connect(lambda: self.toggle_oscillator(2))
        else:
            self.osc3_button = osc_button
            osc_button.clicked.connect(lambda: self.toggle_oscillator(3))

        # Center the button
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(osc_button)
        button_layout.addStretch(1)
        layout.addLayout(button_layout)

        return column

    def create_mixer_column(self):
        """Create mixer column with three gain knobs"""
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setSpacing(8)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title_label = QLabel("Mixer")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)

        layout.addStretch(1)

        # Gain 1 knob
        gain1_label = QLabel("Osc 1")
        gain1_label.setAlignment(Qt.AlignCenter)
        gain1_label.setFont(QFont("Arial", 9))
        layout.addWidget(gain1_label)

        self.gain1_knob = QDial()
        self.gain1_knob.setMinimum(0)
        self.gain1_knob.setMaximum(100)
        self.gain1_knob.setNotchesVisible(True)
        self.gain1_knob.setWrapping(False)
        self.gain1_knob.setFixedSize(70, 70)
        self.gain1_knob.setValue(33)
        self.gain1_knob.valueChanged.connect(lambda v: self.update_gain(1, v))

        knob1_layout = QHBoxLayout()
        knob1_layout.addStretch(1)
        knob1_layout.addWidget(self.gain1_knob)
        knob1_layout.addStretch(1)
        layout.addLayout(knob1_layout)

        self.gain1_label = QLabel("33%")
        self.gain1_label.setAlignment(Qt.AlignCenter)
        self.gain1_label.setFont(QFont("Arial", 9, QFont.Bold))
        layout.addWidget(self.gain1_label)

        layout.addStretch(1)

        # Gain 2 knob
        gain2_label = QLabel("Osc 2")
        gain2_label.setAlignment(Qt.AlignCenter)
        gain2_label.setFont(QFont("Arial", 9))
        layout.addWidget(gain2_label)

        self.gain2_knob = QDial()
        self.gain2_knob.setMinimum(0)
        self.gain2_knob.setMaximum(100)
        self.gain2_knob.setNotchesVisible(True)
        self.gain2_knob.setWrapping(False)
        self.gain2_knob.setFixedSize(70, 70)
        self.gain2_knob.setValue(33)
        self.gain2_knob.valueChanged.connect(lambda v: self.update_gain(2, v))

        knob2_layout = QHBoxLayout()
        knob2_layout.addStretch(1)
        knob2_layout.addWidget(self.gain2_knob)
        knob2_layout.addStretch(1)
        layout.addLayout(knob2_layout)

        self.gain2_label = QLabel("33%")
        self.gain2_label.setAlignment(Qt.AlignCenter)
        self.gain2_label.setFont(QFont("Arial", 9, QFont.Bold))
        layout.addWidget(self.gain2_label)

        layout.addStretch(1)

        # Gain 3 knob
        gain3_label = QLabel("Osc 3")
        gain3_label.setAlignment(Qt.AlignCenter)
        gain3_label.setFont(QFont("Arial", 9))
        layout.addWidget(gain3_label)

        self.gain3_knob = QDial()
        self.gain3_knob.setMinimum(0)
        self.gain3_knob.setMaximum(100)
        self.gain3_knob.setNotchesVisible(True)
        self.gain3_knob.setWrapping(False)
        self.gain3_knob.setFixedSize(70, 70)
        self.gain3_knob.setValue(33)
        self.gain3_knob.valueChanged.connect(lambda v: self.update_gain(3, v))

        knob3_layout = QHBoxLayout()
        knob3_layout.addStretch(1)
        knob3_layout.addWidget(self.gain3_knob)
        knob3_layout.addStretch(1)
        layout.addLayout(knob3_layout)

        self.gain3_label = QLabel("33%")
        self.gain3_label.setAlignment(Qt.AlignCenter)
        self.gain3_label.setFont(QFont("Arial", 9, QFont.Bold))
        layout.addWidget(self.gain3_label)

        layout.addStretch(1)

        return column

    def create_adsr_section(self):
        """Create ADSR envelope section"""
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title_label = QLabel("ADSR Envelope")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title_label)

        # Knobs layout
        knobs_layout = QHBoxLayout()
        knobs_layout.setSpacing(15)

        # Attack
        attack_container = self.create_knob_with_label("Attack", 0, 2000, 10,
                                                        lambda v: self.update_adsr('attack', v))
        knobs_layout.addWidget(attack_container)
        self.attack_knob = attack_container.findChild(QDial)
        self.attack_label_value = attack_container.findChild(QLabel, "value_label")

        # Decay
        decay_container = self.create_knob_with_label("Decay", 0, 2000, 100,
                                                       lambda v: self.update_adsr('decay', v))
        knobs_layout.addWidget(decay_container)
        self.decay_knob = decay_container.findChild(QDial)
        self.decay_label_value = decay_container.findChild(QLabel, "value_label")

        # Sustain
        sustain_container = self.create_knob_with_label("Sustain", 0, 100, 70,
                                                         lambda v: self.update_adsr('sustain', v))
        knobs_layout.addWidget(sustain_container)
        self.sustain_knob = sustain_container.findChild(QDial)
        self.sustain_label_value = sustain_container.findChild(QLabel, "value_label")

        # Release
        release_container = self.create_knob_with_label("Release", 0, 5000, 300,
                                                         lambda v: self.update_adsr('release', v))
        knobs_layout.addWidget(release_container)
        self.release_knob = release_container.findChild(QDial)
        self.release_label_value = release_container.findChild(QLabel, "value_label")

        layout.addLayout(knobs_layout)

        return section

    def create_filter_section(self):
        """Create filter section"""
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title_label = QLabel("Low-Pass Filter")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title_label)

        # Knobs layout
        knobs_layout = QHBoxLayout()
        knobs_layout.setSpacing(15)

        # Cutoff
        cutoff_container = self.create_knob_with_label("Cutoff", 20, 5000, 5000,
                                                        lambda v: self.update_filter('cutoff', v))
        knobs_layout.addWidget(cutoff_container)
        self.cutoff_knob = cutoff_container.findChild(QDial)
        self.cutoff_label_value = cutoff_container.findChild(QLabel, "value_label")

        # Resonance
        resonance_container = self.create_knob_with_label("Resonance", 0, 100, 0,
                                                           lambda v: self.update_filter('resonance', v))
        knobs_layout.addWidget(resonance_container)
        self.resonance_knob = resonance_container.findChild(QDial)
        self.resonance_label_value = resonance_container.findChild(QLabel, "value_label")

        layout.addLayout(knobs_layout)

        return section

    def create_knob_with_label(self, name, min_val, max_val, initial_val, callback):
        """Helper to create a knob with label and value display"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(5)

        # Name label
        name_label = QLabel(name)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setFont(QFont("Arial", 10))
        layout.addWidget(name_label)

        # Knob
        knob = QDial()
        knob.setMinimum(min_val)
        knob.setMaximum(max_val)
        knob.setValue(initial_val)
        knob.setNotchesVisible(True)
        knob.setWrapping(False)
        knob.setFixedSize(80, 80)
        knob.valueChanged.connect(callback)

        knob_layout = QHBoxLayout()
        knob_layout.addStretch(1)
        knob_layout.addWidget(knob)
        knob_layout.addStretch(1)
        layout.addLayout(knob_layout)

        # Value label
        value_label = QLabel(self.format_knob_value(name, initial_val))
        value_label.setObjectName("value_label")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(value_label)

        return container

    def format_knob_value(self, knob_name, value):
        """Format knob value for display"""
        if knob_name == "Attack" or knob_name == "Decay":
            return f"{value}ms"
        elif knob_name == "Release":
            return f"{value}ms"
        elif knob_name == "Sustain":
            return f"{value}%"
        elif knob_name == "Cutoff":
            return f"{value}Hz"
        elif knob_name == "Resonance":
            return f"{value}%"
        else:
            return str(value)

    def update_frequency(self, osc_num, value):
        """Update frequency from knob using logarithmic scale"""
        slider_position = float(value) / 1000.0
        log_freq = self.min_log + slider_position * (self.max_log - self.min_log)
        frequency = 10 ** log_freq

        if osc_num == 1:
            self.freq1 = frequency
            self.freq1_label.setText(f"{self.freq1:.1f} Hz")
        elif osc_num == 2:
            self.freq2 = frequency
            self.freq2_label.setText(f"{self.freq2:.1f} Hz")
        else:
            self.freq3 = frequency
            self.freq3_label.setText(f"{self.freq3:.1f} Hz")

    def update_waveform(self, osc_num, waveform):
        """Update waveform selection"""
        if osc_num == 1:
            self.waveform1 = waveform
        elif osc_num == 2:
            self.waveform2 = waveform
        else:
            self.waveform3 = waveform

    def update_gain(self, osc_num, value):
        """Update gain from knob (0-100%)"""
        gain = value / 100.0

        if osc_num == 1:
            self.gain1 = gain
            self.gain1_label.setText(f"{value}%")
        elif osc_num == 2:
            self.gain2 = gain
            self.gain2_label.setText(f"{value}%")
        else:
            self.gain3 = gain
            self.gain3_label.setText(f"{value}%")

    def update_adsr(self, param, value):
        """Update ADSR parameters"""
        if param == 'attack':
            # Convert ms to seconds
            self.env1.attack = value / 1000.0
            self.env2.attack = value / 1000.0
            self.env3.attack = value / 1000.0
            self.attack_label_value.setText(f"{value}ms")
        elif param == 'decay':
            self.env1.decay = value / 1000.0
            self.env2.decay = value / 1000.0
            self.env3.decay = value / 1000.0
            self.decay_label_value.setText(f"{value}ms")
        elif param == 'sustain':
            self.env1.sustain = value / 100.0
            self.env2.sustain = value / 100.0
            self.env3.sustain = value / 100.0
            self.sustain_label_value.setText(f"{value}%")
        elif param == 'release':
            self.env1.release = value / 1000.0
            self.env2.release = value / 1000.0
            self.env3.release = value / 1000.0
            self.release_label_value.setText(f"{value}ms")

    def update_filter(self, param, value):
        """Update filter parameters"""
        if param == 'cutoff':
            self.filter.cutoff = float(value)
            self.cutoff_label_value.setText(f"{value}Hz")
        elif param == 'resonance':
            self.filter.resonance = value / 100.0
            self.resonance_label_value.setText(f"{value}%")

    def generate_waveform(self, waveform_type, phase, phase_increment, frames):
        """Generate a waveform based on type"""
        phases = phase + np.arange(frames) * phase_increment

        if waveform_type == "Sine":
            return np.sin(phases)
        elif waveform_type == "Sawtooth":
            return 2 * ((phases % (2 * np.pi)) / (2 * np.pi)) - 1
        elif waveform_type == "Square":
            return np.where(np.sin(phases) >= 0, 1.0, -1.0)
        else:
            return np.sin(phases)

    def audio_callback(self, outdata, frames, time, status):
        """Generate and mix three oscillators with envelopes and filter"""
        if status:
            print(status)

        mixed = np.zeros(frames)

        # Generate oscillator 1 (only if on)
        if self.osc1_on:
            phase_increment1 = 2 * np.pi * self.freq1 / self.sample_rate
            wave1 = self.generate_waveform(self.waveform1, self.phase1, phase_increment1, frames)
            env1 = self.env1.process(frames)
            mixed += self.gain1 * wave1 * env1
            self.phase1 = (self.phase1 + frames * phase_increment1) % (2 * np.pi)

        # Generate oscillator 2 (only if on)
        if self.osc2_on:
            phase_increment2 = 2 * np.pi * self.freq2 / self.sample_rate
            wave2 = self.generate_waveform(self.waveform2, self.phase2, phase_increment2, frames)
            env2 = self.env2.process(frames)
            mixed += self.gain2 * wave2 * env2
            self.phase2 = (self.phase2 + frames * phase_increment2) % (2 * np.pi)

        # Generate oscillator 3 (only if on)
        if self.osc3_on:
            phase_increment3 = 2 * np.pi * self.freq3 / self.sample_rate
            wave3 = self.generate_waveform(self.waveform3, self.phase3, phase_increment3, frames)
            env3 = self.env3.process(frames)
            mixed += self.gain3 * wave3 * env3
            self.phase3 = (self.phase3 + frames * phase_increment3) % (2 * np.pi)

        # Apply filter
        filtered = self.filter.process(mixed)

        # Apply master volume
        filtered = 0.3 * filtered

        # Output stereo
        outdata[:, 0] = filtered
        outdata[:, 1] = filtered

    def toggle_oscillator(self, osc_num):
        """Toggle oscillator on/off and trigger/release envelope"""
        if osc_num == 1:
            self.osc1_on = not self.osc1_on
            button = self.osc1_button
            if self.osc1_on:
                self.phase1 = 0
                self.env1.trigger()
            else:
                self.env1.release_note()
        elif osc_num == 2:
            self.osc2_on = not self.osc2_on
            button = self.osc2_button
            if self.osc2_on:
                self.phase2 = 0
                self.env2.trigger()
            else:
                self.env2.release_note()
        else:
            self.osc3_on = not self.osc3_on
            button = self.osc3_button
            if self.osc3_on:
                self.phase3 = 0
                self.env3.trigger()
            else:
                self.env3.release_note()

        # Update button appearance
        is_on = self.osc1_on if osc_num == 1 else (self.osc2_on if osc_num == 2 else self.osc3_on)

        if is_on:
            button.setText("ON")
            button.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    border-radius: 25px;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
            """)
        else:
            button.setText("OFF")
            button.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c;
                    color: #888888;
                    border: none;
                    border-radius: 25px;
                }
                QPushButton:hover {
                    background-color: #4c4c4c;
                }
            """)

        # Manage audio stream
        self.manage_audio_stream()

    def manage_audio_stream(self):
        """Start or stop audio stream based on oscillator states"""
        any_osc_on = self.osc1_on or self.osc2_on or self.osc3_on

        if any_osc_on and self.stream is None:
            # Start audio stream
            try:
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=2,
                    callback=self.audio_callback
                )
                self.stream.start()
            except Exception as e:
                print(f"Error starting audio: {e}")
        elif not any_osc_on and self.stream is not None:
            # Check if all envelopes are done
            all_idle = (self.env1.phase == 'idle' and
                       self.env2.phase == 'idle' and
                       self.env3.phase == 'idle')
            if all_idle:
                # Stop audio stream
                self.stream.stop()
                self.stream.close()
                self.stream = None

    def closeEvent(self, event):
        """Clean up when window is closed"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = SineWaveGenerator()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
