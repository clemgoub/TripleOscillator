#!/usr/bin/env python3
"""
Triple Oscillator Synth with ADSR Envelope and Filter
Features:
- Three independent oscillators with waveform selection
- ADSR envelope generator
- Low-pass filter with cutoff and resonance
- Mixer with gain control per oscillator
- Professional synth-style interface
- Preset save/load functionality
"""

import sys
import numpy as np
import sounddevice as sd
import mido
import threading
import time
import json
import os
from scipy import signal as scipy_signal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QDial, QComboBox,
                             QFileDialog, QMessageBox, QSlider, QGridLayout)
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QFont


class MIDIHandler(QObject):
    """Handles MIDI input in a separate thread"""
    note_on = pyqtSignal(int, int)  # note, velocity
    note_off = pyqtSignal(int)      # note

    def __init__(self):
        super().__init__()
        self.port = None
        self.running = False
        self.thread = None

    def start(self, port_name):
        """Start MIDI input from specified port"""
        if self.running:
            self.stop()

        try:
            self.port = mido.open_input(port_name)
            self.running = True
            self.thread = threading.Thread(target=self._midi_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"Error opening MIDI port: {e}")
            return False

    def stop(self):
        """Stop MIDI input"""
        self.running = False
        if self.port:
            self.port.close()
            self.port = None

    def _midi_loop(self):
        """MIDI message processing loop"""
        while self.running and self.port:
            try:
                for msg in self.port.iter_pending():
                    if msg.type == 'note_on' and msg.velocity > 0:
                        self.note_on.emit(msg.note, msg.velocity)
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        self.note_off.emit(msg.note)
                # Sleep briefly to avoid hogging CPU and causing audio dropouts
                time.sleep(0.001)  # 1ms - plenty fast for MIDI input
            except Exception as e:
                print(f"MIDI error: {e}")
                break


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
        self.attack_start_level = 0.0  # Level at which attack started
        self.release_start_level = 0.0  # Level at which release started

    def trigger(self):
        """Trigger note on (start attack phase)"""
        self.phase = 'attack'
        self.samples_in_phase = 0
        self.level = 0.0  # Reset to 0 for standard monophonic synth behavior
        self.attack_start_level = 0.0  # Always start attack from 0

    def release_note(self):
        """Trigger note off (start release phase)"""
        if self.phase != 'idle':
            self.phase = 'release'
            self.samples_in_phase = 0
            self.release_start_level = self.level  # Remember current level for smooth release

    def force_reset(self):
        """Force envelope to idle state (used when oscillator is turned off)"""
        self.phase = 'idle'
        self.level = 0.0
        self.samples_in_phase = 0

    def process(self, num_samples):
        """Generate envelope for num_samples (vectorized for performance)"""
        output = np.zeros(num_samples)
        samples_processed = 0

        while samples_processed < num_samples:
            remaining = num_samples - samples_processed

            if self.phase == 'idle':
                # Fill remaining with zeros
                output[samples_processed:] = 0.0
                self.level = 0.0
                break

            elif self.phase == 'attack':
                attack_samples = max(1, int(self.attack * self.sample_rate))
                samples_left_in_phase = attack_samples - self.samples_in_phase
                samples_to_process = min(remaining, samples_left_in_phase)

                # Safety check: ensure we have samples to process
                if samples_to_process <= 0:
                    self.level = 1.0
                    self.phase = 'decay'
                    self.samples_in_phase = 0
                    continue

                # Vectorized attack calculation
                sample_indices = np.arange(samples_to_process)
                progress = (self.samples_in_phase + sample_indices) / attack_samples
                output[samples_processed:samples_processed + samples_to_process] = \
                    self.attack_start_level + progress * (1.0 - self.attack_start_level)

                self.samples_in_phase += samples_to_process
                samples_processed += samples_to_process
                self.level = output[samples_processed - 1]

                if self.samples_in_phase >= attack_samples:
                    self.level = 1.0
                    self.phase = 'decay'
                    self.samples_in_phase = 0

            elif self.phase == 'decay':
                decay_samples = max(1, int(self.decay * self.sample_rate))
                samples_left_in_phase = decay_samples - self.samples_in_phase
                samples_to_process = min(remaining, samples_left_in_phase)

                # Safety check: ensure we have samples to process
                if samples_to_process <= 0:
                    self.level = self.sustain
                    self.phase = 'sustain'
                    self.samples_in_phase = 0
                    continue

                # Vectorized decay calculation
                sample_indices = np.arange(samples_to_process)
                progress = (self.samples_in_phase + sample_indices) / decay_samples
                output[samples_processed:samples_processed + samples_to_process] = \
                    1.0 - progress * (1.0 - self.sustain)

                self.samples_in_phase += samples_to_process
                samples_processed += samples_to_process
                self.level = output[samples_processed - 1]

                if self.samples_in_phase >= decay_samples:
                    self.level = self.sustain
                    self.phase = 'sustain'
                    self.samples_in_phase = 0

            elif self.phase == 'sustain':
                # Fill remaining with sustain level
                output[samples_processed:] = self.sustain
                self.level = self.sustain
                samples_processed = num_samples

            elif self.phase == 'release':
                release_samples = max(1, int(self.release * self.sample_rate))
                samples_left_in_phase = release_samples - self.samples_in_phase
                samples_to_process = min(remaining, samples_left_in_phase)

                # Safety check: ensure we have samples to process
                if samples_to_process <= 0:
                    self.level = 0.0
                    self.phase = 'idle'
                    self.samples_in_phase = 0
                    continue

                # Vectorized release calculation
                sample_indices = np.arange(samples_to_process)
                progress = (self.samples_in_phase + sample_indices) / release_samples
                output[samples_processed:samples_processed + samples_to_process] = \
                    self.release_start_level * (1.0 - progress)

                self.samples_in_phase += samples_to_process
                samples_processed += samples_to_process
                self.level = output[samples_processed - 1] if samples_processed > 0 else 0.0

                if self.samples_in_phase >= release_samples:
                    self.level = 0.0
                    self.phase = 'idle'
                    self.samples_in_phase = 0

        return output


class LowPassFilter:
    """Simple low-pass filter with resonance"""
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.cutoff = 5000.0  # Hz
        self.resonance = 0.0  # 0-1

        # Filter state (zi format for scipy)
        self.zi = None
        self.b = None
        self.a = None
        self.last_cutoff = None
        self.last_resonance = None

    def reset(self):
        """Reset filter state to prevent artifacts"""
        self.zi = None

    def process(self, input_signal):
        """Apply low-pass filter with proper state preservation"""
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

        # Build coefficient arrays
        b = [b0, b1, b2]
        a = [1.0, a1, a2]

        # Initialize state only once, keep it even when parameters change
        # This prevents clicks when adjusting filter in real-time
        if self.zi is None:
            self.zi = scipy_signal.lfilter_zi(b, a)

        # Apply filter with state preservation
        output, self.zi = scipy_signal.lfilter(b, a, input_signal, zi=self.zi)

        return output


class LFOGenerator:
    """Low-Frequency Oscillator for modulation"""
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.waveform = "Sine"  # Sine, Triangle, Square, Sawtooth, Random
        self.rate_mode = "Free"  # Free (Hz) or Sync (MIDI divisions)
        self.rate_hz = 2.0  # Frequency in Hz (0.1 - 20 Hz)
        self.sync_division = "1/4"  # MIDI sync: 1/16, 1/8, 1/4, 1/2, 1/1, 2/1, 4/1
        self.bpm = 120.0  # Tempo for MIDI sync
        self.phase = 0.0  # Current phase (0 to 2*pi)

        # Random waveform state
        self.last_random_value = 0.0
        self.random_samples_until_change = 0

    def get_effective_rate(self):
        """Calculate effective rate in Hz based on mode"""
        if self.rate_mode == "Free":
            return self.rate_hz
        else:
            # MIDI sync mode - convert division to Hz based on BPM
            divisions = {
                "1/16": 4.0,
                "1/8": 2.0,
                "1/4": 1.0,
                "1/2": 0.5,
                "1/1": 0.25,
                "2/1": 0.125,
                "4/1": 0.0625
            }
            beats_per_cycle = divisions.get(self.sync_division, 1.0)
            return (self.bpm / 60.0) / beats_per_cycle

    def process(self, num_samples):
        """Generate LFO waveform for num_samples (returns values from -1 to 1)"""
        output = np.zeros(num_samples)
        rate = self.get_effective_rate()
        phase_increment = 2.0 * np.pi * rate / self.sample_rate

        if self.waveform == "Sine":
            phases = self.phase + np.arange(num_samples) * phase_increment
            output = np.sin(phases)

        elif self.waveform == "Triangle":
            phases = self.phase + np.arange(num_samples) * phase_increment
            normalized_phase = (phases % (2 * np.pi)) / (2 * np.pi)
            # Triangle: rises from -1 to 1, then falls back to -1
            output = np.where(normalized_phase < 0.5,
                            -1.0 + 4.0 * normalized_phase,
                            3.0 - 4.0 * normalized_phase)

        elif self.waveform == "Square":
            phases = self.phase + np.arange(num_samples) * phase_increment
            output = np.where(np.sin(phases) >= 0, 1.0, -1.0)

        elif self.waveform == "Sawtooth":
            phases = self.phase + np.arange(num_samples) * phase_increment
            normalized_phase = (phases % (2 * np.pi)) / (2 * np.pi)
            # Sawtooth: rises from -1 to 1
            output = -1.0 + 2.0 * normalized_phase

        elif self.waveform == "Random":
            # Sample & hold: random value held for duration based on rate
            samples_per_step = int(self.sample_rate / (rate * 2))  # 2 steps per cycle
            for i in range(num_samples):
                if self.random_samples_until_change <= 0:
                    self.last_random_value = np.random.uniform(-1.0, 1.0)
                    self.random_samples_until_change = samples_per_step
                output[i] = self.last_random_value
                self.random_samples_until_change -= 1

        # Update phase for next call
        self.phase += num_samples * phase_increment
        # Keep phase in range to prevent overflow
        self.phase = self.phase % (2 * np.pi)

        return output


class SineWaveGenerator(QMainWindow):
    def __init__(self):
        super().__init__()

        # Audio parameters
        self.sample_rate = 44100
        self.stream = None
        self.power_on = True  # Master power switch

        # Oscillator 1 parameters
        self.freq1 = 440.0
        self.phase1 = 0
        self.osc1_on = False
        self.waveform1 = "Sine"
        self.detune1 = 0.0  # Detune in cents (-100 to +100)
        self.octave1 = 0  # Octave offset (-3 to +3)
        self.pulse_width1 = 0.5  # Pulse width for square wave (0.0 to 1.0, default 50%)

        # Oscillator 2 parameters (phase offset to reduce constructive interference)
        self.freq2 = 440.0
        self.phase2 = 2 * np.pi / 3
        self.osc2_on = False
        self.waveform2 = "Sine"
        self.detune2 = 0.0  # Detune in cents (-100 to +100)
        self.octave2 = 0  # Octave offset (-3 to +3)
        self.pulse_width2 = 0.5  # Pulse width for square wave (0.0 to 1.0, default 50%)

        # Oscillator 3 parameters (phase offset to reduce constructive interference)
        self.freq3 = 440.0
        self.phase3 = 4 * np.pi / 3
        self.osc3_on = False
        self.waveform3 = "Sine"
        self.detune3 = 0.0  # Detune in cents (-100 to +100)
        self.octave3 = 0  # Octave offset (-3 to +3)
        self.pulse_width3 = 0.5  # Pulse width for square wave (0.0 to 1.0, default 50%)

        # Mixer parameters (0.0 to 1.0)
        self.gain1 = 0.33
        self.gain2 = 0.33
        self.gain3 = 0.33
        self.master_volume = 0.5  # Master volume (0.0 to 1.0)

        # Envelope generators (one per oscillator)
        self.env1 = EnvelopeGenerator(self.sample_rate)
        self.env2 = EnvelopeGenerator(self.sample_rate)
        self.env3 = EnvelopeGenerator(self.sample_rate)

        # Filter
        self.filter = LowPassFilter(self.sample_rate)

        # LFO
        self.lfo = LFOGenerator(self.sample_rate)

        # Modulation matrix (depth 0-1 for each destination)
        self.lfo_to_osc1_pitch = 0.0
        self.lfo_to_osc2_pitch = 0.0
        self.lfo_to_osc3_pitch = 0.0
        self.lfo_to_osc1_pw = 0.0
        self.lfo_to_osc2_pw = 0.0
        self.lfo_to_osc3_pw = 0.0
        self.lfo_to_filter_cutoff = 0.0
        self.lfo_to_osc1_volume = 0.0
        self.lfo_to_osc2_volume = 0.0
        self.lfo_to_osc3_volume = 0.0

        # MIDI
        self.midi_handler = MIDIHandler()
        self.midi_handler.note_on.connect(self.handle_midi_note_on)
        self.midi_handler.note_off.connect(self.handle_midi_note_off)
        self.current_note = None

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
        self.setFixedSize(900, 850)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # MIDI SECTION
        midi_layout = QHBoxLayout()
        midi_label = QLabel("MIDI Input:")
        midi_label.setFont(QFont("Arial", 10))
        midi_layout.addWidget(midi_label)

        self.midi_selector = QComboBox()
        self.midi_selector.setFont(QFont("Arial", 10))
        self.refresh_midi_ports()
        self.midi_selector.currentTextChanged.connect(self.on_midi_port_changed)
        midi_layout.addWidget(self.midi_selector)

        refresh_button = QPushButton("Refresh")
        refresh_button.setFont(QFont("Arial", 10))
        refresh_button.clicked.connect(self.refresh_midi_ports)
        midi_layout.addWidget(refresh_button)

        midi_layout.addStretch(1)

        # Power button
        self.power_button = QPushButton("POWER ON")
        self.power_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.power_button.setFixedSize(100, 40)
        self.power_button.setStyleSheet("""
            QPushButton {
                background-color: #2d5016;
                color: #90ee90;
                border: 2px solid #4a7c29;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3a6620;
            }
            QPushButton:pressed {
                background-color: #1f3810;
            }
        """)
        self.power_button.clicked.connect(self.toggle_power)
        midi_layout.addWidget(self.power_button)

        midi_layout.addStretch(1)

        # Preset management buttons
        save_preset_button = QPushButton("Save Preset")
        save_preset_button.setFont(QFont("Arial", 10))
        save_preset_button.setFixedSize(100, 40)
        save_preset_button.setStyleSheet("""
            QPushButton {
                background-color: #1e3a8a;
                color: white;
                border: 2px solid #3b82f6;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
        """)
        save_preset_button.clicked.connect(self.save_preset)
        midi_layout.addWidget(save_preset_button)

        load_preset_button = QPushButton("Load Preset")
        load_preset_button.setFont(QFont("Arial", 10))
        load_preset_button.setFixedSize(100, 40)
        load_preset_button.setStyleSheet("""
            QPushButton {
                background-color: #166534;
                color: white;
                border: 2px solid #22c55e;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #16a34a;
            }
            QPushButton:pressed {
                background-color: #15803d;
            }
        """)
        load_preset_button.clicked.connect(self.load_preset)
        midi_layout.addWidget(load_preset_button)

        midi_layout.addStretch(1)
        main_layout.addLayout(midi_layout)

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

        # LFO Section (full width below filter)
        lfo_widget = self.create_lfo_section()
        main_layout.addWidget(lfo_widget)

        # Initialize ADSR values (trigger callbacks manually since setValue doesn't trigger them)
        self.update_adsr('attack', 10)      # 10ms
        self.update_adsr('decay', 100)      # 100ms
        self.update_adsr('sustain', 70)     # 70%
        self.update_adsr('release', 300)    # 300ms

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

        # Center the waveform selector
        waveform_layout = QHBoxLayout()
        waveform_layout.addStretch(1)
        waveform_layout.addWidget(waveform_combo)
        waveform_layout.addStretch(1)
        layout.addLayout(waveform_layout)

        layout.addStretch(1)

        # On/Off button (on top of knob, 33% smaller)
        osc_button = QPushButton("OFF")
        osc_button.setFont(QFont("Arial", 8, QFont.Bold))
        osc_button.setFixedSize(33, 33)
        osc_button.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                color: #888888;
                border: none;
                border-radius: 16px;
                min-width: 33px;
                max-width: 33px;
                min-height: 33px;
                max-height: 33px;
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

        # Frequency knob (Large: 100x100, use 0-100 range for clean appearance)
        freq_knob = QDial()
        freq_knob.setMinimum(0)
        freq_knob.setMaximum(100)
        freq_knob.setNotchesVisible(True)
        freq_knob.setWrapping(False)
        freq_knob.setFixedSize(100, 100)
        # Disable native styling to get clean knob appearance
        freq_knob.setAttribute(Qt.WA_MacShowFocusRect, False)
        freq_knob.setStyleSheet("""
            QDial {
                background: none;
                border: none;
            }
        """)

        # Set initial position for 440 Hz (map 0-100 to log range)
        initial_position = int(100 * (np.log10(440) - self.min_log) / (self.max_log - self.min_log))
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

        # Octave controls
        octave_layout = QHBoxLayout()
        octave_layout.addStretch(1)

        # Octave down button
        octave_down_btn = QPushButton("-")
        octave_down_btn.setFixedSize(25, 25)
        octave_down_btn.setFont(QFont("Arial", 12, QFont.Bold))
        octave_down_btn.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
            }
            QPushButton:pressed {
                background-color: #555555;
            }
        """)

        # Octave label
        octave_label = QLabel("0")
        octave_label.setAlignment(Qt.AlignCenter)
        octave_label.setFont(QFont("Arial", 9))
        octave_label.setFixedWidth(30)

        # Octave up button
        octave_up_btn = QPushButton("+")
        octave_up_btn.setFixedSize(25, 25)
        octave_up_btn.setFont(QFont("Arial", 12, QFont.Bold))
        octave_up_btn.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
            }
            QPushButton:pressed {
                background-color: #555555;
            }
        """)

        # Connect buttons and store references
        if osc_num == 1:
            self.octave1_label = octave_label
            octave_down_btn.clicked.connect(lambda: self.change_octave(1, -1))
            octave_up_btn.clicked.connect(lambda: self.change_octave(1, 1))
        elif osc_num == 2:
            self.octave2_label = octave_label
            octave_down_btn.clicked.connect(lambda: self.change_octave(2, -1))
            octave_up_btn.clicked.connect(lambda: self.change_octave(2, 1))
        else:
            self.octave3_label = octave_label
            octave_down_btn.clicked.connect(lambda: self.change_octave(3, -1))
            octave_up_btn.clicked.connect(lambda: self.change_octave(3, 1))

        octave_layout.addWidget(octave_down_btn)
        octave_layout.addWidget(octave_label)
        octave_layout.addWidget(octave_up_btn)
        octave_layout.addStretch(1)
        layout.addLayout(octave_layout)

        layout.addStretch(1)

        # Pulse Width knob (Small: 50x50, only affects Square wave)
        pw_label = QLabel("Pulse Width")
        pw_label.setAlignment(Qt.AlignCenter)
        pw_label.setFont(QFont("Arial", 9, QFont.Bold))
        layout.addWidget(pw_label)

        pw_knob = QDial()
        pw_knob.setMinimum(1)  # 1% minimum to avoid silence
        pw_knob.setMaximum(99)  # 99% maximum
        pw_knob.setValue(50)  # 50% default (square wave)
        pw_knob.setNotchesVisible(True)
        pw_knob.setWrapping(False)
        pw_knob.setFixedSize(50, 50)
        pw_knob.setAttribute(Qt.WA_MacShowFocusRect, False)
        pw_knob.setStyleSheet("""
            QDial {
                background: none;
                border: none;
            }
        """)

        # Connect to appropriate oscillator
        if osc_num == 1:
            self.pw1_knob = pw_knob
            pw_knob.valueChanged.connect(lambda v: self.update_pulse_width(1, v))
        elif osc_num == 2:
            self.pw2_knob = pw_knob
            pw_knob.valueChanged.connect(lambda v: self.update_pulse_width(2, v))
        else:
            self.pw3_knob = pw_knob
            pw_knob.valueChanged.connect(lambda v: self.update_pulse_width(3, v))

        # Center the knob
        pw_knob_layout = QHBoxLayout()
        pw_knob_layout.addStretch(1)
        pw_knob_layout.addWidget(pw_knob)
        pw_knob_layout.addStretch(1)
        layout.addLayout(pw_knob_layout)

        # Pulse width value display
        pw_value_label = QLabel("50%")
        pw_value_label.setAlignment(Qt.AlignCenter)
        pw_value_label.setFont(QFont("Arial", 9))

        if osc_num == 1:
            self.pw1_label = pw_value_label
        elif osc_num == 2:
            self.pw2_label = pw_value_label
        else:
            self.pw3_label = pw_value_label

        layout.addWidget(pw_value_label)

        layout.addStretch(1)

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

        # Gain 1 knob (Small: 50x50)
        gain1_label = QLabel("Osc 1")
        gain1_label.setAlignment(Qt.AlignCenter)
        gain1_label.setFont(QFont("Arial", 9))
        layout.addWidget(gain1_label)

        self.gain1_knob = QDial()
        self.gain1_knob.setMinimum(0)
        self.gain1_knob.setMaximum(100)
        self.gain1_knob.setNotchesVisible(True)
        self.gain1_knob.setWrapping(False)
        self.gain1_knob.setFixedSize(50, 50)
        self.gain1_knob.setStyleSheet("""
            QDial {
                background: transparent;
            }
        """)
        self.gain1_knob.setValue(33)
        self.gain1_knob.valueChanged.connect(lambda v: self.update_gain(1, v))

        knob1_layout = QHBoxLayout()
        knob1_layout.addStretch(1)
        knob1_layout.addWidget(self.gain1_knob)
        knob1_layout.addStretch(1)
        layout.addLayout(knob1_layout)

        layout.addStretch(1)

        # Gain 2 knob (Small: 50x50)
        gain2_label = QLabel("Osc 2")
        gain2_label.setAlignment(Qt.AlignCenter)
        gain2_label.setFont(QFont("Arial", 9))
        layout.addWidget(gain2_label)

        self.gain2_knob = QDial()
        self.gain2_knob.setMinimum(0)
        self.gain2_knob.setMaximum(100)
        self.gain2_knob.setNotchesVisible(True)
        self.gain2_knob.setWrapping(False)
        self.gain2_knob.setFixedSize(50, 50)
        self.gain2_knob.setStyleSheet("""
            QDial {
                background: transparent;
            }
        """)
        self.gain2_knob.setValue(33)
        self.gain2_knob.valueChanged.connect(lambda v: self.update_gain(2, v))

        knob2_layout = QHBoxLayout()
        knob2_layout.addStretch(1)
        knob2_layout.addWidget(self.gain2_knob)
        knob2_layout.addStretch(1)
        layout.addLayout(knob2_layout)

        layout.addStretch(1)

        # Gain 3 knob (Small: 50x50)
        gain3_label = QLabel("Osc 3")
        gain3_label.setAlignment(Qt.AlignCenter)
        gain3_label.setFont(QFont("Arial", 9))
        layout.addWidget(gain3_label)

        self.gain3_knob = QDial()
        self.gain3_knob.setMinimum(0)
        self.gain3_knob.setMaximum(100)
        self.gain3_knob.setNotchesVisible(True)
        self.gain3_knob.setWrapping(False)
        self.gain3_knob.setFixedSize(50, 50)
        self.gain3_knob.setStyleSheet("""
            QDial {
                background: transparent;
            }
        """)
        self.gain3_knob.setValue(33)
        self.gain3_knob.valueChanged.connect(lambda v: self.update_gain(3, v))

        knob3_layout = QHBoxLayout()
        knob3_layout.addStretch(1)
        knob3_layout.addWidget(self.gain3_knob)
        knob3_layout.addStretch(1)
        layout.addLayout(knob3_layout)

        layout.addStretch(1)

        # Master Volume knob (Medium: 70x70)
        master_label = QLabel("Master")
        master_label.setAlignment(Qt.AlignCenter)
        master_label.setFont(QFont("Arial", 9))
        layout.addWidget(master_label)

        self.master_volume_knob = QDial()
        self.master_volume_knob.setMinimum(0)
        self.master_volume_knob.setMaximum(100)
        self.master_volume_knob.setNotchesVisible(True)
        self.master_volume_knob.setWrapping(False)
        self.master_volume_knob.setFixedSize(70, 70)
        self.master_volume_knob.setStyleSheet("""
            QDial {
                background: transparent;
            }
        """)
        self.master_volume_knob.setValue(50)
        self.master_volume_knob.valueChanged.connect(self.update_master_volume)

        master_knob_layout = QHBoxLayout()
        master_knob_layout.addStretch(1)
        master_knob_layout.addWidget(self.master_volume_knob)
        master_knob_layout.addStretch(1)
        layout.addLayout(master_knob_layout)

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

        # Attack (use 0-100 range internally, scale to 0-2000 in callback)
        attack_container = self.create_knob_with_label("Attack", 0, 100, 1,
                                                        lambda v: self.update_adsr('attack', v * 20))
        knobs_layout.addWidget(attack_container)
        self.attack_knob = attack_container.findChild(QDial)
        self.attack_label_value = attack_container.findChild(QLabel, "value_label")

        # Decay (use 0-100 range internally, scale to 0-2000 in callback)
        decay_container = self.create_knob_with_label("Decay", 0, 100, 5,
                                                       lambda v: self.update_adsr('decay', v * 20))
        knobs_layout.addWidget(decay_container)
        self.decay_knob = decay_container.findChild(QDial)
        self.decay_label_value = decay_container.findChild(QLabel, "value_label")

        # Sustain
        sustain_container = self.create_knob_with_label("Sustain", 0, 100, 70,
                                                         lambda v: self.update_adsr('sustain', v))
        knobs_layout.addWidget(sustain_container)
        self.sustain_knob = sustain_container.findChild(QDial)
        self.sustain_label_value = sustain_container.findChild(QLabel, "value_label")

        # Release (use 0-100 range internally, scale to 0-5000 in callback)
        release_container = self.create_knob_with_label("Release", 0, 100, 6,
                                                         lambda v: self.update_adsr('release', v * 50))
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

        # Cutoff (Large: 100x100, use 0-100 range internally, scale to 20-5000 logarithmically)
        cutoff_container = self.create_knob_with_label("Cutoff", 0, 100, 100,
                                                        lambda v: self.update_filter('cutoff', int(20 * (250 ** (v/100)))), size=100)
        knobs_layout.addWidget(cutoff_container)
        self.cutoff_knob = cutoff_container.findChild(QDial)
        self.cutoff_label_value = cutoff_container.findChild(QLabel, "value_label")

        # Resonance (Small: 50x50)
        resonance_container = self.create_knob_with_label("Resonance", 0, 100, 0,
                                                           lambda v: self.update_filter('resonance', v), size=50)
        knobs_layout.addWidget(resonance_container)
        self.resonance_knob = resonance_container.findChild(QDial)
        self.resonance_label_value = resonance_container.findChild(QLabel, "value_label")

        layout.addLayout(knobs_layout)

        return section

    def create_lfo_section(self):
        """Create LFO section with waveform, rate, and modulation matrix"""
        section = QWidget()
        main_layout = QVBoxLayout(section)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title_label = QLabel("LFO & Modulation Matrix")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        main_layout.addWidget(title_label)

        # Top row: LFO controls
        lfo_controls_layout = QHBoxLayout()
        lfo_controls_layout.setSpacing(15)

        # Waveform selector
        waveform_container = QWidget()
        waveform_layout = QVBoxLayout(waveform_container)
        waveform_label = QLabel("Waveform")
        waveform_label.setAlignment(Qt.AlignCenter)
        waveform_layout.addWidget(waveform_label)
        self.lfo_waveform_combo = QComboBox()
        self.lfo_waveform_combo.addItems(["Sine", "Triangle", "Square", "Sawtooth", "Random"])
        self.lfo_waveform_combo.setFixedWidth(120)
        self.lfo_waveform_combo.currentTextChanged.connect(lambda v: setattr(self.lfo, 'waveform', v))
        waveform_layout.addWidget(self.lfo_waveform_combo)
        lfo_controls_layout.addWidget(waveform_container)

        # Rate Mode selector
        mode_container = QWidget()
        mode_layout = QVBoxLayout(mode_container)
        mode_label = QLabel("Rate Mode")
        mode_label.setAlignment(Qt.AlignCenter)
        mode_layout.addWidget(mode_label)
        self.lfo_mode_combo = QComboBox()
        self.lfo_mode_combo.addItems(["Free", "Sync"])
        self.lfo_mode_combo.setFixedWidth(100)
        self.lfo_mode_combo.currentTextChanged.connect(self.update_lfo_mode)
        mode_layout.addWidget(self.lfo_mode_combo)
        lfo_controls_layout.addWidget(mode_container)

        # Rate knob (Free mode: 0.1-20 Hz)
        self.lfo_rate_container = self.create_knob_with_label("Rate", 1, 200, 20,
                                                                lambda v: self.update_lfo_rate(v / 10.0), size=70)
        lfo_controls_layout.addWidget(self.lfo_rate_container)

        # Sync division selector (only visible in Sync mode)
        self.sync_div_container = QWidget()
        sync_div_layout = QVBoxLayout(self.sync_div_container)
        sync_div_label = QLabel("Division")
        sync_div_label.setAlignment(Qt.AlignCenter)
        sync_div_layout.addWidget(sync_div_label)
        self.lfo_sync_combo = QComboBox()
        self.lfo_sync_combo.addItems(["1/16", "1/8", "1/4", "1/2", "1/1", "2/1", "4/1"])
        self.lfo_sync_combo.setCurrentText("1/4")
        self.lfo_sync_combo.setFixedWidth(80)
        self.lfo_sync_combo.currentTextChanged.connect(lambda v: setattr(self.lfo, 'sync_division', v))
        sync_div_layout.addWidget(self.lfo_sync_combo)
        self.sync_div_container.setVisible(False)  # Hidden by default
        lfo_controls_layout.addWidget(self.sync_div_container)

        # BPM knob (only visible in Sync mode)
        self.bpm_container = self.create_knob_with_label("BPM", 40, 240, 120,
                                                          lambda v: setattr(self.lfo, 'bpm', float(v)), size=70)
        self.bpm_container.setVisible(False)  # Hidden by default
        lfo_controls_layout.addWidget(self.bpm_container)

        lfo_controls_layout.addStretch()
        main_layout.addLayout(lfo_controls_layout)

        # Bottom: Modulation Matrix
        matrix_label = QLabel("Modulation Depth (0-100%)")
        matrix_label.setAlignment(Qt.AlignCenter)
        matrix_label.setFont(QFont("Arial", 11))
        main_layout.addWidget(matrix_label)

        # Create grid for modulation matrix
        matrix_layout = QGridLayout()
        matrix_layout.setSpacing(10)

        # Headers
        matrix_layout.addWidget(QLabel(""), 0, 0)
        matrix_layout.addWidget(QLabel("Osc1 Pitch"), 0, 1, Qt.AlignCenter)
        matrix_layout.addWidget(QLabel("Osc2 Pitch"), 0, 2, Qt.AlignCenter)
        matrix_layout.addWidget(QLabel("Osc3 Pitch"), 0, 3, Qt.AlignCenter)
        matrix_layout.addWidget(QLabel("Osc1 PW"), 0, 4, Qt.AlignCenter)
        matrix_layout.addWidget(QLabel("Osc2 PW"), 0, 5, Qt.AlignCenter)
        matrix_layout.addWidget(QLabel("Osc3 PW"), 0, 6, Qt.AlignCenter)
        matrix_layout.addWidget(QLabel("Filter"), 0, 7, Qt.AlignCenter)
        matrix_layout.addWidget(QLabel("Osc1 Vol"), 0, 8, Qt.AlignCenter)
        matrix_layout.addWidget(QLabel("Osc2 Vol"), 0, 9, Qt.AlignCenter)
        matrix_layout.addWidget(QLabel("Osc3 Vol"), 0, 10, Qt.AlignCenter)

        # LFO row label
        matrix_layout.addWidget(QLabel("LFO"), 1, 0)

        # Create sliders for each modulation destination
        mod_destinations = [
            ('lfo_to_osc1_pitch', 1, 1),
            ('lfo_to_osc2_pitch', 1, 2),
            ('lfo_to_osc3_pitch', 1, 3),
            ('lfo_to_osc1_pw', 1, 4),
            ('lfo_to_osc2_pw', 1, 5),
            ('lfo_to_osc3_pw', 1, 6),
            ('lfo_to_filter_cutoff', 1, 7),
            ('lfo_to_osc1_volume', 1, 8),
            ('lfo_to_osc2_volume', 1, 9),
            ('lfo_to_osc3_volume', 1, 10)
        ]

        for attr_name, row, col in mod_destinations:
            slider = QSlider(Qt.Vertical)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(0)
            slider.setFixedHeight(60)
            slider.valueChanged.connect(lambda v, attr=attr_name: setattr(self, attr, v / 100.0))
            matrix_layout.addWidget(slider, row, col, Qt.AlignCenter)

        main_layout.addLayout(matrix_layout)

        return section

    def update_lfo_mode(self, mode):
        """Update LFO rate mode and show/hide relevant controls"""
        self.lfo.rate_mode = mode
        if mode == "Free":
            self.lfo_rate_container.setVisible(True)
            self.sync_div_container.setVisible(False)
            self.bpm_container.setVisible(False)
        else:  # Sync
            self.lfo_rate_container.setVisible(False)
            self.sync_div_container.setVisible(True)
            self.bpm_container.setVisible(True)

    def update_lfo_rate(self, rate_hz):
        """Update LFO rate in Hz"""
        self.lfo.rate_hz = rate_hz

    def create_knob_with_label(self, name, min_val, max_val, initial_val, callback, size=70):
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
        knob.setFixedSize(size, size)
        # Disable native styling to get clean knob appearance
        knob.setAttribute(Qt.WA_MacShowFocusRect, False)
        knob.setStyleSheet("""
            QDial {
                background: none;
                border: none;
            }
        """)

        # Value label
        value_label = QLabel(self.format_knob_value(name, initial_val))
        value_label.setObjectName("value_label")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setFont(QFont("Arial", 10, QFont.Bold))

        # Combined callback that updates both the value and the label
        def combined_callback(value):
            callback(value)
            value_label.setText(self.format_knob_value(name, value))

        knob.valueChanged.connect(combined_callback)

        knob_layout = QHBoxLayout()
        knob_layout.addStretch(1)
        knob_layout.addWidget(knob)
        knob_layout.addStretch(1)
        layout.addLayout(knob_layout)

        layout.addWidget(value_label)

        return container

    def format_knob_value(self, knob_name, value):
        """Format knob value for display"""
        if knob_name == "Attack" or knob_name == "Decay":
            # Scale 0-100 to 0-2000ms
            return f"{value * 20}ms"
        elif knob_name == "Release":
            # Scale 0-100 to 0-5000ms
            return f"{value * 50}ms"
        elif knob_name == "Sustain":
            return f"{value}%"
        elif knob_name == "Cutoff":
            # Apply logarithmic scaling for cutoff (20-5000 Hz)
            freq = int(20 * (250 ** (value/100)))
            return f"{freq}Hz"
        elif knob_name == "Resonance":
            return f"{value}%"
        elif knob_name == "Rate":
            # LFO rate: 1-200 maps to 0.1-20.0 Hz
            return f"{value / 10.0:.1f}Hz"
        elif knob_name == "BPM":
            return f"{value}"
        else:
            return str(value)

    def apply_detune(self, base_freq, detune_cents):
        """Apply detune in cents to a base frequency"""
        return base_freq * (2.0 ** (detune_cents / 1200.0))

    def apply_octave(self, freq, octave_offset):
        """Apply octave offset to a frequency"""
        return freq * (2.0 ** octave_offset)

    def change_octave(self, osc_num, direction):
        """Change octave offset for an oscillator (+1 or -1)"""
        if osc_num == 1:
            self.octave1 = max(-3, min(3, self.octave1 + direction))
            self.octave1_label.setText(f"{self.octave1:+d}" if self.octave1 != 0 else "0")
            # Recalculate frequency with new octave
            if self.midi_handler.running and self.current_note is not None:
                base_freq = self.midi_note_to_freq(self.current_note)
                self.freq1 = self.apply_octave(self.apply_detune(base_freq, self.detune1), self.octave1)
            else:
                # In drone mode, apply octave to current frequency
                base_freq = self.freq1 / (2.0 ** (self.octave1 - direction))  # Undo previous octave
                self.freq1 = self.apply_octave(base_freq, self.octave1)
        elif osc_num == 2:
            self.octave2 = max(-3, min(3, self.octave2 + direction))
            self.octave2_label.setText(f"{self.octave2:+d}" if self.octave2 != 0 else "0")
            if self.midi_handler.running and self.current_note is not None:
                base_freq = self.midi_note_to_freq(self.current_note)
                self.freq2 = self.apply_octave(self.apply_detune(base_freq, self.detune2), self.octave2)
            else:
                base_freq = self.freq2 / (2.0 ** (self.octave2 - direction))
                self.freq2 = self.apply_octave(base_freq, self.octave2)
        else:
            self.octave3 = max(-3, min(3, self.octave3 + direction))
            self.octave3_label.setText(f"{self.octave3:+d}" if self.octave3 != 0 else "0")
            if self.midi_handler.running and self.current_note is not None:
                base_freq = self.midi_note_to_freq(self.current_note)
                self.freq3 = self.apply_octave(self.apply_detune(base_freq, self.detune3), self.octave3)
            else:
                base_freq = self.freq3 / (2.0 ** (self.octave3 - direction))
                self.freq3 = self.apply_octave(base_freq, self.octave3)

    def update_frequency(self, osc_num, value):
        """Update frequency from knob - acts as detune in MIDI mode, frequency in drone mode"""
        if self.midi_handler.running:
            # MIDI mode: knob controls detune in cents (-100 to +100)
            # Map 0-100 slider to -100 to +100 cents
            detune_cents = (float(value) / 100.0) * 200.0 - 100.0

            if osc_num == 1:
                self.detune1 = detune_cents
                self.freq1_label.setText(f"{detune_cents:+.1f} cents")
            elif osc_num == 2:
                self.detune2 = detune_cents
                self.freq2_label.setText(f"{detune_cents:+.1f} cents")
            else:
                self.detune3 = detune_cents
                self.freq3_label.setText(f"{detune_cents:+.1f} cents")

            # If a note is currently playing, update the frequencies with new detune and octave
            if self.current_note is not None:
                base_freq = self.midi_note_to_freq(self.current_note)
                if osc_num == 1:
                    self.freq1 = self.apply_octave(self.apply_detune(base_freq, self.detune1), self.octave1)
                elif osc_num == 2:
                    self.freq2 = self.apply_octave(self.apply_detune(base_freq, self.detune2), self.octave2)
                else:
                    self.freq3 = self.apply_octave(self.apply_detune(base_freq, self.detune3), self.octave3)
        else:
            # Drone mode: knob controls absolute frequency
            # Map 0-100 slider to logarithmic frequency range
            slider_position = float(value) / 100.0
            log_freq = self.min_log + slider_position * (self.max_log - self.min_log)
            frequency = 10 ** log_freq

            if osc_num == 1:
                # Apply octave offset in drone mode too
                self.freq1 = self.apply_octave(frequency, self.octave1)
                self.freq1_label.setText(f"{self.freq1:.1f} Hz")
            elif osc_num == 2:
                self.freq2 = self.apply_octave(frequency, self.octave2)
                self.freq2_label.setText(f"{self.freq2:.1f} Hz")
            else:
                self.freq3 = self.apply_octave(frequency, self.octave3)
                self.freq3_label.setText(f"{self.freq3:.1f} Hz")

    def update_waveform(self, osc_num, waveform):
        """Update waveform selection"""
        if osc_num == 1:
            self.waveform1 = waveform
        elif osc_num == 2:
            self.waveform2 = waveform
        else:
            self.waveform3 = waveform

    def update_pulse_width(self, osc_num, value):
        """Update pulse width from knob (1-99%)"""
        pulse_width = value / 100.0  # Convert to 0.01-0.99 range

        if osc_num == 1:
            self.pulse_width1 = pulse_width
            self.pw1_label.setText(f"{value}%")
        elif osc_num == 2:
            self.pulse_width2 = pulse_width
            self.pw2_label.setText(f"{value}%")
        else:
            self.pulse_width3 = pulse_width
            self.pw3_label.setText(f"{value}%")

    def update_gain(self, osc_num, value):
        """Update gain from knob (0-100%)"""
        gain = value / 100.0

        if osc_num == 1:
            self.gain1 = gain
        elif osc_num == 2:
            self.gain2 = gain
        else:
            self.gain3 = gain

    def update_master_volume(self, value):
        """Update master volume from knob (0-100%)"""
        self.master_volume = value / 100.0

    def update_adsr(self, param, value):
        """Update ADSR parameters"""
        if param == 'attack':
            # Convert ms to seconds
            self.env1.attack = value / 1000.0
            self.env2.attack = value / 1000.0
            self.env3.attack = value / 1000.0
        elif param == 'decay':
            self.env1.decay = value / 1000.0
            self.env2.decay = value / 1000.0
            self.env3.decay = value / 1000.0
        elif param == 'sustain':
            self.env1.sustain = value / 100.0
            self.env2.sustain = value / 100.0
            self.env3.sustain = value / 100.0
        elif param == 'release':
            self.env1.release = value / 1000.0
            self.env2.release = value / 1000.0
            self.env3.release = value / 1000.0

    def update_filter(self, param, value):
        """Update filter parameters"""
        if param == 'cutoff':
            self.filter.cutoff = float(value)
        elif param == 'resonance':
            self.filter.resonance = value / 100.0

    def save_preset(self):
        """Save current settings to a preset file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Preset",
            os.path.expanduser("~/Documents"),
            "Synth Presets (*.json);;All Files (*)"
        )

        if not file_path:
            return

        # Ensure .json extension
        if not file_path.endswith('.json'):
            file_path += '.json'

        # Create preset data structure
        preset = {
            "version": "1.0",  # For future compatibility
            "oscillators": {
                "osc1": {
                    "waveform": self.waveform1,
                    "frequency": self.freq1,
                    "detune": self.detune1,
                    "octave": self.octave1,
                    "pulse_width": self.pulse_width1,
                    "gain": self.gain1
                },
                "osc2": {
                    "waveform": self.waveform2,
                    "frequency": self.freq2,
                    "detune": self.detune2,
                    "octave": self.octave2,
                    "pulse_width": self.pulse_width2,
                    "gain": self.gain2
                },
                "osc3": {
                    "waveform": self.waveform3,
                    "frequency": self.freq3,
                    "detune": self.detune3,
                    "octave": self.octave3,
                    "pulse_width": self.pulse_width3,
                    "gain": self.gain3
                }
            },
            "envelope": {
                "attack": self.env1.attack,
                "decay": self.env1.decay,
                "sustain": self.env1.sustain,
                "release": self.env1.release
            },
            "filter": {
                "cutoff": self.filter.cutoff,
                "resonance": self.filter.resonance
            },
            "master": {
                "volume": self.master_volume,
                "power": self.power_on
            }
        }

        try:
            with open(file_path, 'w') as f:
                json.dump(preset, f, indent=2)
            QMessageBox.information(self, "Success", f"Preset saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save preset:\n{str(e)}")

    def load_preset(self):
        """Load settings from a preset file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Preset",
            os.path.expanduser("~/Documents"),
            "Synth Presets (*.json);;All Files (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                preset = json.load(f)

            # Extract oscillator settings with defaults for forward compatibility
            osc1 = preset.get("oscillators", {}).get("osc1", {})
            osc2 = preset.get("oscillators", {}).get("osc2", {})
            osc3 = preset.get("oscillators", {}).get("osc3", {})

            # Oscillator 1
            self.waveform1 = osc1.get("waveform", "Sine")
            self.freq1 = osc1.get("frequency", 440.0)
            self.detune1 = osc1.get("detune", 0.0)
            self.octave1 = osc1.get("octave", 0)
            self.pulse_width1 = osc1.get("pulse_width", 0.5)
            self.gain1 = osc1.get("gain", 0.33)

            # Oscillator 2
            self.waveform2 = osc2.get("waveform", "Sine")
            self.freq2 = osc2.get("frequency", 440.0)
            self.detune2 = osc2.get("detune", 0.0)
            self.octave2 = osc2.get("octave", 0)
            self.pulse_width2 = osc2.get("pulse_width", 0.5)
            self.gain2 = osc2.get("gain", 0.33)

            # Oscillator 3
            self.waveform3 = osc3.get("waveform", "Sine")
            self.freq3 = osc3.get("frequency", 440.0)
            self.detune3 = osc3.get("detune", 0.0)
            self.octave3 = osc3.get("octave", 0)
            self.pulse_width3 = osc3.get("pulse_width", 0.5)
            self.gain3 = osc3.get("gain", 0.33)

            # Envelope settings
            env = preset.get("envelope", {})
            self.env1.attack = env.get("attack", 0.01)
            self.env2.attack = env.get("attack", 0.01)
            self.env3.attack = env.get("attack", 0.01)
            self.env1.decay = env.get("decay", 0.1)
            self.env2.decay = env.get("decay", 0.1)
            self.env3.decay = env.get("decay", 0.1)
            self.env1.sustain = env.get("sustain", 0.7)
            self.env2.sustain = env.get("sustain", 0.7)
            self.env3.sustain = env.get("sustain", 0.7)
            self.env1.release = env.get("release", 0.2)
            self.env2.release = env.get("release", 0.2)
            self.env3.release = env.get("release", 0.2)

            # Filter settings
            filt = preset.get("filter", {})
            self.filter.cutoff = filt.get("cutoff", 5000.0)
            self.filter.resonance = filt.get("resonance", 0.0)

            # Master settings
            master = preset.get("master", {})
            self.master_volume = master.get("volume", 0.5)
            self.power_on = master.get("power", True)

            # Update UI to reflect loaded preset
            self.update_ui_from_preset()

            QMessageBox.information(self, "Success", f"Preset loaded from:\n{file_path}")

        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "Error", f"Invalid preset file:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load preset:\n{str(e)}")

    def update_ui_from_preset(self):
        """Update all UI elements to reflect current preset values"""
        # Update waveform selectors
        self.waveform1_combo.setCurrentText(self.waveform1)
        self.waveform2_combo.setCurrentText(self.waveform2)
        self.waveform3_combo.setCurrentText(self.waveform3)

        # Update frequency labels
        self.freq1_label.setText(f"{self.freq1:.1f} Hz")
        self.freq2_label.setText(f"{self.freq2:.1f} Hz")
        self.freq3_label.setText(f"{self.freq3:.1f} Hz")

        # Update octave labels
        self.octave1_label.setText(f"{self.octave1:+d}" if self.octave1 != 0 else "0")
        self.octave2_label.setText(f"{self.octave2:+d}" if self.octave2 != 0 else "0")
        self.octave3_label.setText(f"{self.octave3:+d}" if self.octave3 != 0 else "0")

        # Update pulse width knobs and labels
        self.pw1_knob.setValue(int(self.pulse_width1 * 100))
        self.pw2_knob.setValue(int(self.pulse_width2 * 100))
        self.pw3_knob.setValue(int(self.pulse_width3 * 100))
        self.pw1_label.setText(f"{int(self.pulse_width1 * 100)}%")
        self.pw2_label.setText(f"{int(self.pulse_width2 * 100)}%")
        self.pw3_label.setText(f"{int(self.pulse_width3 * 100)}%")

        # Update power button
        if self.power_on:
            self.power_button.setText("ON")
            self.power_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 5px 15px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
        else:
            self.power_button.setText("OFF")
            self.power_button.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 5px 15px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
            """)

    def generate_waveform(self, waveform_type, phase, phase_increment, frames, pulse_width=0.5):
        """Generate a waveform based on type

        Args:
            waveform_type: Type of waveform ("Sine", "Sawtooth", "Square")
            phase: Starting phase
            phase_increment: Phase increment per sample
            frames: Number of frames to generate
            pulse_width: Pulse width for square wave (0.0 to 1.0, default 0.5 for 50% duty cycle)
        """
        phases = phase + np.arange(frames) * phase_increment

        if waveform_type == "Sine":
            return np.sin(phases)
        elif waveform_type == "Sawtooth":
            return 2 * ((phases % (2 * np.pi)) / (2 * np.pi)) - 1
        elif waveform_type == "Square":
            # Pulse width modulation: compare normalized phase to pulse width
            normalized_phase = (phases % (2 * np.pi)) / (2 * np.pi)
            return np.where(normalized_phase < pulse_width, 1.0, -1.0)
        else:
            return np.sin(phases)

    def audio_callback(self, outdata, frames, time, status):
        """Generate and mix three oscillators with envelopes and filter"""
        # Don't print in audio callback - causes buffer underruns

        # If power is off, output silence
        if not self.power_on:
            outdata[:, 0] = 0
            outdata[:, 1] = 0
            return

        # Generate LFO signal (-1 to 1)
        lfo_signal = self.lfo.process(frames)

        mixed = np.zeros(frames)

        # Generate oscillator 1 (only if on)
        if self.osc1_on:
            # Apply pitch modulation (vibrato) - LFO modulates frequency ±5% max
            freq_mod = 1.0 + (lfo_signal * 0.05 * self.lfo_to_osc1_pitch)
            modulated_freq1 = self.freq1 * freq_mod
            phase_increment1 = 2 * np.pi * modulated_freq1 / self.sample_rate

            # Apply pulse width modulation
            pw_mod = lfo_signal * 0.3 * self.lfo_to_osc1_pw  # ±30% max modulation
            modulated_pw1 = np.clip(self.pulse_width1 + pw_mod, 0.01, 0.99)

            wave1 = self.generate_waveform(self.waveform1, self.phase1, phase_increment1, frames, modulated_pw1)
            env1 = self.env1.process(frames)

            # Apply volume modulation (tremolo)
            vol_mod = 1.0 + (lfo_signal * self.lfo_to_osc1_volume)
            modulated_gain1 = self.gain1 * np.clip(vol_mod, 0.0, 1.0)

            mixed += modulated_gain1 * wave1 * env1
            self.phase1 = (self.phase1 + frames * phase_increment1) % (2 * np.pi)

        # Generate oscillator 2 (only if on)
        if self.osc2_on:
            # Apply pitch modulation
            freq_mod = 1.0 + (lfo_signal * 0.05 * self.lfo_to_osc2_pitch)
            modulated_freq2 = self.freq2 * freq_mod
            phase_increment2 = 2 * np.pi * modulated_freq2 / self.sample_rate

            # Apply pulse width modulation
            pw_mod = lfo_signal * 0.3 * self.lfo_to_osc2_pw
            modulated_pw2 = np.clip(self.pulse_width2 + pw_mod, 0.01, 0.99)

            wave2 = self.generate_waveform(self.waveform2, self.phase2, phase_increment2, frames, modulated_pw2)
            env2 = self.env2.process(frames)

            # Apply volume modulation
            vol_mod = 1.0 + (lfo_signal * self.lfo_to_osc2_volume)
            modulated_gain2 = self.gain2 * np.clip(vol_mod, 0.0, 1.0)

            mixed += modulated_gain2 * wave2 * env2
            self.phase2 = (self.phase2 + frames * phase_increment2) % (2 * np.pi)

        # Generate oscillator 3 (only if on)
        if self.osc3_on:
            # Apply pitch modulation
            freq_mod = 1.0 + (lfo_signal * 0.05 * self.lfo_to_osc3_pitch)
            modulated_freq3 = self.freq3 * freq_mod
            phase_increment3 = 2 * np.pi * modulated_freq3 / self.sample_rate

            # Apply pulse width modulation
            pw_mod = lfo_signal * 0.3 * self.lfo_to_osc3_pw
            modulated_pw3 = np.clip(self.pulse_width3 + pw_mod, 0.01, 0.99)

            wave3 = self.generate_waveform(self.waveform3, self.phase3, phase_increment3, frames, modulated_pw3)
            env3 = self.env3.process(frames)

            # Apply volume modulation
            vol_mod = 1.0 + (lfo_signal * self.lfo_to_osc3_volume)
            modulated_gain3 = self.gain3 * np.clip(vol_mod, 0.0, 1.0)

            mixed += modulated_gain3 * wave3 * env3
            self.phase3 = (self.phase3 + frames * phase_increment3) % (2 * np.pi)

        # Apply filter with LFO modulation to cutoff
        # LFO modulates cutoff ±2 octaves (4x range)
        # Use mean LFO value for filter (filters can't do per-sample modulation efficiently)
        if self.lfo_to_filter_cutoff > 0:
            lfo_mean = np.mean(lfo_signal)
            cutoff_mod = 2.0 ** (lfo_mean * 2.0 * self.lfo_to_filter_cutoff)  # ±2 octaves
            self.filter.cutoff = np.clip(self.filter.cutoff * cutoff_mod, 20.0, 20000.0)

        filtered = self.filter.process(mixed)

        # Apply master volume
        filtered = self.master_volume * filtered

        # Output stereo
        outdata[:, 0] = filtered
        outdata[:, 1] = filtered

    def toggle_oscillator(self, osc_num):
        """Toggle oscillator on/off and trigger/release envelope"""
        if osc_num == 1:
            self.osc1_on = not self.osc1_on
            button = self.osc1_button
            if self.osc1_on:
                self.phase1 = 0  # Osc1 starts at 0
                self.env1.trigger()
            else:
                self.env1.force_reset()  # Force to idle so next trigger starts from 0
        elif osc_num == 2:
            self.osc2_on = not self.osc2_on
            button = self.osc2_button
            if self.osc2_on:
                self.phase2 = 2 * np.pi / 3  # Phase offset to reduce interference
                self.env2.trigger()
            else:
                self.env2.force_reset()  # Force to idle so next trigger starts from 0
        else:
            self.osc3_on = not self.osc3_on
            button = self.osc3_button
            if self.osc3_on:
                self.phase3 = 4 * np.pi / 3  # Phase offset to reduce interference
                self.env3.trigger()
            else:
                self.env3.force_reset()  # Force to idle so next trigger starts from 0

        # Update button appearance
        is_on = self.osc1_on if osc_num == 1 else (self.osc2_on if osc_num == 2 else self.osc3_on)

        if is_on:
            button.setText("ON")
            button.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    border-radius: 16px;
                    min-width: 33px;
                    max-width: 33px;
                    min-height: 33px;
                    max-height: 33px;
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
                    border-radius: 16px;
                    min-width: 33px;
                    max-width: 33px;
                    min-height: 33px;
                    max-height: 33px;
                }
                QPushButton:hover {
                    background-color: #4c4c4c;
                }
            """)

        # Manage audio stream
        self.manage_audio_stream()

    def toggle_power(self):
        """Toggle master power on/off"""
        self.power_on = not self.power_on

        if self.power_on:
            self.power_button.setText("POWER ON")
            self.power_button.setStyleSheet("""
                QPushButton {
                    background-color: #2d5016;
                    color: #90ee90;
                    border: 2px solid #4a7c29;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #3a6620;
                }
                QPushButton:pressed {
                    background-color: #1f3810;
                }
            """)
        else:
            self.power_button.setText("POWER OFF")
            self.power_button.setStyleSheet("""
                QPushButton {
                    background-color: #5c1010;
                    color: #ff6b6b;
                    border: 2px solid #8b2020;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #6b1515;
                }
                QPushButton:pressed {
                    background-color: #3d0a0a;
                }
            """)

    def manage_audio_stream(self):
        """Start or stop audio stream based on oscillator states"""
        any_osc_on = self.osc1_on or self.osc2_on or self.osc3_on

        if any_osc_on and self.stream is None:
            # Reset filter state to prevent artifacts
            self.filter.reset()

            # Start audio stream
            try:
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=2,
                    blocksize=512,  # Smaller buffer for lower latency and better scheduling
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

    def refresh_midi_ports(self):
        """Refresh the list of available MIDI input ports"""
        self.midi_selector.clear()
        self.midi_selector.addItem("No MIDI Input")
        ports = mido.get_input_names()
        for port in ports:
            self.midi_selector.addItem(port)

    def on_midi_port_changed(self, port_name):
        """Handle MIDI port selection change"""
        if port_name and port_name != "No MIDI Input":
            success = self.midi_handler.start(port_name)
            if success:
                print(f"MIDI port opened: {port_name}")
                # Switch to MIDI mode: turn off all oscillators and release envelopes
                if self.osc1_on:
                    self.osc1_on = False
                    self.osc1_button.setText("OFF")
                    self.osc1_button.setStyleSheet("""
                        QPushButton {
                            background-color: #3c3c3c;
                            color: #888888;
                            border: none;
                            border-radius: 17px;
                        }
                        QPushButton:hover {
                            background-color: #4c4c4c;
                        }
                    """)
                    self.env1.release_note()

                if self.osc2_on:
                    self.osc2_on = False
                    self.osc2_button.setText("OFF")
                    self.osc2_button.setStyleSheet("""
                        QPushButton {
                            background-color: #3c3c3c;
                            color: #888888;
                            border: none;
                            border-radius: 17px;
                        }
                        QPushButton:hover {
                            background-color: #4c4c4c;
                        }
                    """)
                    self.env2.release_note()

                if self.osc3_on:
                    self.osc3_on = False
                    self.osc3_button.setText("OFF")
                    self.osc3_button.setStyleSheet("""
                        QPushButton {
                            background-color: #3c3c3c;
                            color: #888888;
                            border: none;
                            border-radius: 17px;
                        }
                        QPushButton:hover {
                            background-color: #4c4c4c;
                        }
                    """)
                    self.env3.release_note()

                # Set knobs to center (0 cents) and update labels
                center_value = 500  # Middle of 0-1000 range = 0 cents
                self.freq1_knob.setValue(center_value)
                self.freq2_knob.setValue(center_value)
                self.freq3_knob.setValue(center_value)
                # Labels will be updated by update_frequency callbacks
        else:
            self.midi_handler.stop()
            # Switch to drone mode: update labels to show frequency
            self.freq1_label.setText(f"{self.freq1:.1f} Hz")
            self.freq2_label.setText(f"{self.freq2:.1f} Hz")
            self.freq3_label.setText(f"{self.freq3:.1f} Hz")

    def midi_note_to_freq(self, note):
        """Convert MIDI note number to frequency"""
        return 440.0 * (2.0 ** ((note - 69) / 12.0))

    def handle_midi_note_on(self, note, velocity):
        """Handle MIDI note on message"""
        self.current_note = note
        base_freq = self.midi_note_to_freq(note)

        # Apply detune and octave offsets to each oscillator
        self.freq1 = self.apply_octave(self.apply_detune(base_freq, self.detune1), self.octave1)
        self.freq2 = self.apply_octave(self.apply_detune(base_freq, self.detune2), self.octave2)
        self.freq3 = self.apply_octave(self.apply_detune(base_freq, self.detune3), self.octave3)

        # Labels show detune in MIDI mode, not frequency
        # (they're updated in update_frequency when knobs are moved)

        # Don't reset filter - let it maintain state for smooth transitions

        # Only trigger envelopes on oscillators that are already enabled
        # This prevents forcing all oscillators on and causing clipping
        if self.osc1_on:
            self.env1.trigger()
        if self.osc2_on:
            self.env2.trigger()
        if self.osc3_on:
            self.env3.trigger()

        # Start audio if needed
        self.manage_audio_stream()

    def handle_midi_note_off(self, note):
        """Handle MIDI note off message"""
        if self.current_note == note:
            # Release envelopes
            self.env1.release_note()
            self.env2.release_note()
            self.env3.release_note()
            self.current_note = None

    def closeEvent(self, event):
        """Clean up when window is closed"""
        self.midi_handler.stop()
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
