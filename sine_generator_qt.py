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
NOTICE: This code is AI generated, do not use for model training purposes
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
                             QFileDialog, QMessageBox, QSlider, QGridLayout, QGroupBox)
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QFont


class MIDIHandler(QObject):
    """Handles MIDI input in a separate thread"""
    note_on = pyqtSignal(int, int)  # note, velocity
    note_off = pyqtSignal(int)      # note
    bpm_changed = pyqtSignal(float)  # BPM from MIDI clock

    def __init__(self):
        super().__init__()
        self.port = None
        self.running = False
        self.thread = None
        # MIDI clock timing (24 pulses per quarter note)
        self.last_clock_time = None
        self.clock_intervals = []  # Store recent intervals for averaging
        self.clock_count = 0

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
                    elif msg.type == 'clock':
                        # MIDI clock: 24 pulses per quarter note
                        self._process_midi_clock()
                # Sleep briefly to avoid hogging CPU and causing audio dropouts
                time.sleep(0.001)  # 1ms - plenty fast for MIDI input
            except Exception as e:
                print(f"MIDI error: {e}")
                break

    def _process_midi_clock(self):
        """Process MIDI clock message and calculate BPM"""
        current_time = time.time()

        if self.last_clock_time is not None:
            # Calculate interval between clock pulses
            interval = current_time - self.last_clock_time

            # Store recent intervals (keep last 24 for smoothing - one quarter note worth)
            self.clock_intervals.append(interval)
            if len(self.clock_intervals) > 24:
                self.clock_intervals.pop(0)

            # Calculate BPM every 24 clocks (one quarter note)
            self.clock_count += 1
            if self.clock_count >= 24 and len(self.clock_intervals) >= 24:
                # Average interval between clocks
                avg_interval = sum(self.clock_intervals) / len(self.clock_intervals)
                # 24 clocks = 1 quarter note, so time for one quarter note = avg_interval * 24
                # BPM = 60 / (time per quarter note)
                if avg_interval > 0:
                    bpm = 60.0 / (avg_interval * 24.0)
                    # Clamp BPM to reasonable range
                    bpm = max(40.0, min(240.0, bpm))
                    self.bpm_changed.emit(bpm)
                self.clock_count = 0

        self.last_clock_time = current_time


class EnvelopeGenerator:
    """ADSR Envelope Generator"""
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.attack = 0.0    # seconds (default 0 for instant attack)
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
            # beats_per_cycle = how many beats for one complete LFO cycle
            divisions = {
                "1/16": 0.25,   # 1/16 note = 1 cycle per quarter beat (fast)
                "1/8": 0.5,     # 1/8 note = 1 cycle per half beat
                "1/4": 1.0,     # 1/4 note = 1 cycle per beat
                "1/2": 2.0,     # 1/2 note = 1 cycle per 2 beats
                "1/1": 4.0,     # Whole note = 1 cycle per 4 beats
                "2/1": 8.0,     # 2 bars = 1 cycle per 8 beats
                "4/1": 16.0     # 4 bars = 1 cycle per 16 beats (slow)
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


class Voice:
    """Represents a single voice with three oscillators and envelopes"""
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.note = None  # MIDI note number (None if voice is free)
        self.velocity = 0
        self.age = 0  # For voice stealing (oldest-first strategy)

        # Phase accumulators for each oscillator (all start at 0 for clean unison)
        self.phase1 = 0
        self.phase2 = 0
        self.phase3 = 0

        # Envelope generators (independent per voice)
        self.env1 = EnvelopeGenerator(sample_rate)
        self.env2 = EnvelopeGenerator(sample_rate)
        self.env3 = EnvelopeGenerator(sample_rate)

        # Unison detuning offset (set when voice is allocated)
        self.unison_detune = 0.0  # In cents

    def is_active(self):
        """Voice is active if any envelope is not idle"""
        return (self.env1.phase != 'idle' or
                self.env2.phase != 'idle' or
                self.env3.phase != 'idle')

    def is_free(self):
        """Voice is free if no note is assigned and all envelopes are idle"""
        return self.note is None and not self.is_active()

    def trigger(self, note, velocity, unison_detune=0.0, phase_offset=0.0):
        """Trigger this voice with a note

        Args:
            note: MIDI note number
            velocity: Note velocity (0-127)
            unison_detune: Detune amount in cents for unison spread
            phase_offset: Phase offset in radians for subtle stereo width (0 to π/4)
        """
        self.note = note
        self.velocity = velocity
        self.unison_detune = unison_detune
        self.age = 0

        # Apply subtle phase offset for unison width
        # Small offsets (0 to π/4) add stereo presence without harsh phasing
        self.phase1 = phase_offset
        self.phase2 = phase_offset
        self.phase3 = phase_offset

        # Trigger envelopes
        self.env1.trigger()
        self.env2.trigger()
        self.env3.trigger()

    def release(self):
        """Release this voice (start release phase, but keep note assigned until idle)"""
        # DON'T set note = None here - we need it for the audio callback during release
        # It will be set to None when all envelopes reach idle phase
        self.env1.release_note()
        self.env2.release_note()
        self.env3.release_note()


class SineWaveGenerator(QMainWindow):
    def __init__(self):
        super().__init__()

        # Audio parameters
        self.sample_rate = 44100
        self.stream = None
        self.power_on = True  # Master power switch

        # Oscillator enabled states
        self.osc1_enabled = True
        self.osc2_enabled = True
        self.osc3_enabled = True

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

        # Envelope generators (one per oscillator) - TEMPLATE ONLY, not used for audio
        self.env1 = EnvelopeGenerator(self.sample_rate)
        self.env2 = EnvelopeGenerator(self.sample_rate)
        self.env3 = EnvelopeGenerator(self.sample_rate)
        # Force these template envelopes to idle - they should never be active
        self.env1.force_reset()
        self.env2.force_reset()
        self.env3.force_reset()

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

        # Modulation matrix mix controls (dry/wet 0-1 for each destination)
        self.lfo_to_osc1_pitch_mix = 1.0
        self.lfo_to_osc2_pitch_mix = 1.0
        self.lfo_to_osc3_pitch_mix = 1.0
        self.lfo_to_osc1_pw_mix = 1.0
        self.lfo_to_osc2_pw_mix = 1.0
        self.lfo_to_osc3_pw_mix = 1.0
        self.lfo_to_filter_cutoff_mix = 1.0
        self.lfo_to_osc1_volume_mix = 1.0
        self.lfo_to_osc2_volume_mix = 1.0
        self.lfo_to_osc3_volume_mix = 1.0

        # MIDI
        self.midi_handler = MIDIHandler()
        self.midi_handler.note_on.connect(self.handle_midi_note_on)
        self.midi_handler.note_off.connect(self.handle_midi_note_off)
        self.midi_handler.bpm_changed.connect(self.handle_midi_bpm_change)
        self.current_note = None

        # Computer keyboard MIDI
        self.pressed_keys = set()  # Track currently pressed keys
        self.keyboard_octave = 4  # Base octave for keyboard (C4 = middle C)

        # Voice management (polyphony and unison)
        self.max_polyphony = 1  # Number of simultaneous notes (1-8)
        self.unison_count = 1  # Number of voices per note when polyphony=1 (1-8)
        self.unison_detune_amount = 0.0  # Detune amount in cents for unison (0=tight, 5=chorus, 10=supersaw)
        self.voice_pool = []  # Will be created in init_ui
        self.active_voices = {}  # {note_number: [voice1, voice2, ...]}

        # Playback mode (chromatic vs drone)
        self.playback_mode = 'chromatic'  # 'chromatic' or 'drone'

        # Logarithmic scale parameters
        self.min_freq = 20.0
        self.max_freq = 5000.0
        self.min_log = np.log10(self.min_freq)
        self.max_log = np.log10(self.max_freq)

        # Preset browser
        self.factory_presets = []  # List of preset file paths
        self.current_preset_index = 0
        self.current_preset_name = "Init"

        # Initialize UI
        self.init_ui()

        # Initialize voice pool based on polyphony and unison settings
        self.reallocate_voice_pool()

        # Load factory presets and load Init preset
        self.load_factory_presets()
        if self.factory_presets:
            self.load_preset_by_index(0)  # Load Init.json (first preset)

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Triple Oscillator Synth")
        self.setMinimumSize(1000, 900)
        self.resize(1200, 900)

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

        # VOICE MODE CONTROLS (Simple 3-button selection)
        voice_label = QLabel("MODE:")
        voice_label.setFont(QFont("Arial", 10, QFont.Bold))
        midi_layout.addWidget(voice_label)

        # Mono button
        self.mono_button = QPushButton("MONO")
        self.mono_button.setFont(QFont("Arial", 9))
        self.mono_button.setFixedSize(60, 35)
        self.mono_button.setCheckable(True)
        self.mono_button.setChecked(True)
        self.mono_button.clicked.connect(lambda: self.set_voice_mode('mono'))
        self.mono_button.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: 2px solid #888888;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:checked {
                background-color: #ffc452;
                color: black;
                border: 2px solid #b37600;
            }
        """)
        midi_layout.addWidget(self.mono_button)

        # Poly button
        self.poly_button = QPushButton("POLY")
        self.poly_button.setFont(QFont("Arial", 9))
        self.poly_button.setFixedSize(60, 35)
        self.poly_button.setCheckable(True)
        self.poly_button.clicked.connect(lambda: self.set_voice_mode('poly'))
        self.poly_button.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: 2px solid #888888;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:checked {
                background-color: #6ff788;
                color: black;
                border: 2px solid #006112;
            }
        """)
        midi_layout.addWidget(self.poly_button)

        # Unison button
        self.unison_button = QPushButton("UNI")
        self.unison_button.setFont(QFont("Arial", 9))
        self.unison_button.setFixedSize(60, 35)
        self.unison_button.setCheckable(True)
        self.unison_button.clicked.connect(lambda: self.set_voice_mode('unison'))
        self.unison_button.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: 2px solid #888888;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:checked {
                background-color: #fc5b42;
                color: black;
                border: 2px solid #8a1300;
            }
        """)
        midi_layout.addWidget(self.unison_button)

        midi_layout.addSpacing(20)

        # PLAYBACK MODE CONTROLS (Chromatic vs Drone)
        playback_label = QLabel("PLAY:")
        playback_label.setFont(QFont("Arial", 10, QFont.Bold))
        midi_layout.addWidget(playback_label)

        # Chromatic button
        self.chromatic_button = QPushButton("CHROM")
        self.chromatic_button.setFont(QFont("Arial", 9))
        self.chromatic_button.setFixedSize(60, 35)
        self.chromatic_button.setCheckable(True)
        self.chromatic_button.setChecked(True)
        self.chromatic_button.clicked.connect(lambda: self.set_playback_mode('chromatic'))
        self.chromatic_button.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: 2px solid #888888;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:checked {
                background-color: #6eb5ff;
                color: black;
                border: 2px solid #004a99;
            }
        """)
        midi_layout.addWidget(self.chromatic_button)

        # Drone button
        self.drone_button = QPushButton("DRONE")
        self.drone_button.setFont(QFont("Arial", 9))
        self.drone_button.setFixedSize(60, 35)
        self.drone_button.setCheckable(True)
        self.drone_button.clicked.connect(lambda: self.set_playback_mode('drone'))
        self.drone_button.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: 2px solid #888888;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:checked {
                background-color: #c77dff;
                color: black;
                border: 2px solid #5a189a;
            }
        """)
        midi_layout.addWidget(self.drone_button)

        midi_layout.addSpacing(20)

        # Power button
        self.power_button = QPushButton("POWER ON")
        self.power_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.power_button.setFixedSize(100, 40)
        self.power_button.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                color: #90ee90;
                border: 2px solid #4a7c29;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:pressed {
                background-color: #3c3c3c;
            }
        """)
        self.power_button.clicked.connect(self.toggle_power)
        midi_layout.addWidget(self.power_button)

        midi_layout.addStretch(1)

        # Preset browser
        preset_browser_layout = QVBoxLayout()
        preset_browser_layout.setSpacing(5)

        # Top row: Previous, Preset Name, Next
        preset_nav_layout = QHBoxLayout()
        preset_nav_layout.setSpacing(5)

        # Previous button
        self.prev_preset_button = QPushButton("<")
        self.prev_preset_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.prev_preset_button.setFixedSize(35, 30)
        self.prev_preset_button.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                color: white;
                border: 2px solid #666666;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """)
        self.prev_preset_button.clicked.connect(self.prev_preset)
        preset_nav_layout.addWidget(self.prev_preset_button)

        # Preset name label
        self.preset_name_label = QLabel("Init")
        self.preset_name_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.preset_name_label.setAlignment(Qt.AlignCenter)
        self.preset_name_label.setFixedWidth(150)
        self.preset_name_label.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                color: #ffc452;
                border: 2px solid #444444;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        preset_nav_layout.addWidget(self.preset_name_label)

        # Next button
        self.next_preset_button = QPushButton(">")
        self.next_preset_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.next_preset_button.setFixedSize(35, 30)
        self.next_preset_button.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                color: white;
                border: 2px solid #666666;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """)
        self.next_preset_button.clicked.connect(self.next_preset)
        preset_nav_layout.addWidget(self.next_preset_button)

        preset_browser_layout.addLayout(preset_nav_layout)

        # Bottom row: Load and Save buttons
        preset_buttons_layout = QHBoxLayout()
        preset_buttons_layout.setSpacing(5)

        # Load Preset button
        self.load_preset_button = QPushButton("Load Preset")
        self.load_preset_button.setFont(QFont("Arial", 9))
        self.load_preset_button.setFixedHeight(25)
        self.load_preset_button.setStyleSheet("""
            QPushButton {
                background-color: #166534;
                color: white;
                border: 2px solid #22c55e;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #15803d;
            }
            QPushButton:pressed {
                background-color: #14532d;
            }
        """)
        self.load_preset_button.clicked.connect(self.load_preset)
        preset_buttons_layout.addWidget(self.load_preset_button)

        # Save Preset button
        self.save_preset_button = QPushButton("Save Preset")
        self.save_preset_button.setFont(QFont("Arial", 9))
        self.save_preset_button.setFixedHeight(25)
        self.save_preset_button.setStyleSheet("""
            QPushButton {
                background-color: #1e3a8a;
                color: white;
                border: 2px solid #3b82f6;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
        """)
        self.save_preset_button.clicked.connect(self.save_preset)
        preset_buttons_layout.addWidget(self.save_preset_button)

        preset_browser_layout.addLayout(preset_buttons_layout)

        midi_layout.addLayout(preset_browser_layout)

        midi_layout.addStretch(1)
        main_layout.addLayout(midi_layout)

        # TWO-ROW HORIZONTAL LAYOUT
        # Row 1: Oscillators | Mixer | Filter
        # Row 2: ADSR | LFO

        # ROW 1: Oscillators, Mixer, Filter
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(10)

        # OSCILLATORS (3 vertical columns)
        oscillators_group = self.create_group_box("OSCILLATORS")
        oscillators_layout = QHBoxLayout(oscillators_group)
        oscillators_layout.setSpacing(10)

        osc1_widget = self.create_oscillator_column("Oscillator 1", 1)
        oscillators_layout.addWidget(osc1_widget)

        osc2_widget = self.create_oscillator_column("Oscillator 2", 2)
        oscillators_layout.addWidget(osc2_widget)

        osc3_widget = self.create_oscillator_column("Oscillator 3", 3)
        oscillators_layout.addWidget(osc3_widget)

        row1_layout.addWidget(oscillators_group, 3)  # 3 parts for oscillators

        # MIXER
        mixer_group = self.create_group_box("MIXER")
        mixer_layout = QVBoxLayout(mixer_group)
        mixer_widget = self.create_mixer_column()
        mixer_layout.addWidget(mixer_widget)
        row1_layout.addWidget(mixer_group, 1)  # 1 part for mixer

        # FILTER
        filter_group = self.create_group_box("FILTER")
        filter_layout = QVBoxLayout(filter_group)
        filter_widget = self.create_filter_section()
        filter_layout.addWidget(filter_widget)
        row1_layout.addWidget(filter_group, 1)  # 1 part for filter

        main_layout.addLayout(row1_layout)

        # ROW 2: ADSR and LFO
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(10)

        # ADSR (sliders like JUNO-106)
        adsr_group = self.create_group_box("ADSR ENVELOPE")
        adsr_layout = QVBoxLayout(adsr_group)
        adsr_widget = self.create_adsr_section_sliders()
        adsr_layout.addWidget(adsr_widget)
        row2_layout.addWidget(adsr_group, 1)

        # LFO
        lfo_group = self.create_group_box("LFO & MODULATION")
        lfo_layout = QVBoxLayout(lfo_group)
        lfo_widget = self.create_lfo_section()
        lfo_layout.addWidget(lfo_widget)
        row2_layout.addWidget(lfo_group, 1)

        main_layout.addLayout(row2_layout)

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

    def create_group_box(self, title):
        """Create a styled QGroupBox"""
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 12px;
                padding-top: 15px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        return group

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

        # Frequency knob (Large: 80x80, use 0-100 range for clean appearance)
        freq_knob = QDial()
        freq_knob.setMinimum(0)
        freq_knob.setMaximum(100)
        freq_knob.setNotchesVisible(True)
        freq_knob.setWrapping(False)
        freq_knob.setFixedSize(80, 80)
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
        freq_label = QLabel("+0.0 cents")
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

        # Octave label (using organ footage notation)
        octave_label = QLabel("8'")
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
            self.osc1_octave_down_btn = octave_down_btn
            self.osc1_octave_up_btn = octave_up_btn
            octave_down_btn.clicked.connect(lambda: self.change_octave(1, -1))
            octave_up_btn.clicked.connect(lambda: self.change_octave(1, 1))
        elif osc_num == 2:
            self.octave2_label = octave_label
            self.osc2_octave_down_btn = octave_down_btn
            self.osc2_octave_up_btn = octave_up_btn
            octave_down_btn.clicked.connect(lambda: self.change_octave(2, -1))
            octave_up_btn.clicked.connect(lambda: self.change_octave(2, 1))
        else:
            self.octave3_label = octave_label
            self.osc3_octave_down_btn = octave_down_btn
            self.osc3_octave_up_btn = octave_up_btn
            octave_down_btn.clicked.connect(lambda: self.change_octave(3, -1))
            octave_up_btn.clicked.connect(lambda: self.change_octave(3, 1))

        octave_layout.addWidget(octave_down_btn)
        octave_layout.addWidget(octave_label)
        octave_layout.addWidget(octave_up_btn)
        octave_layout.addStretch(1)
        layout.addLayout(octave_layout)

        layout.addStretch(1)

        # Pulse Width knob (Small: 45x45, only affects Square wave)
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
        pw_knob.setFixedSize(45, 45)
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
        self.gain1_knob.setFixedSize(45, 45)
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
        self.gain2_knob.setFixedSize(45, 45)
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
        self.gain3_knob.setFixedSize(45, 45)
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
        self.master_volume_knob.setFixedSize(60, 60)
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
        layout.setContentsMargins(5, 5, 5, 5)

        # Knobs layout
        knobs_layout = QHBoxLayout()
        knobs_layout.setSpacing(15)

        # Attack (use 0-100 range internally, scale to 0-2000 in callback)
        attack_container = self.create_knob_with_label("Attack", 0, 100, 0,
                                                        lambda v: self.update_adsr('attack', v * 20), size=60)
        knobs_layout.addWidget(attack_container)
        self.attack_knob = attack_container.findChild(QDial)
        self.attack_label_value = attack_container.findChild(QLabel, "value_label")

        # Decay (use 0-100 range internally, scale to 0-2000 in callback)
        decay_container = self.create_knob_with_label("Decay", 0, 100, 5,
                                                       lambda v: self.update_adsr('decay', v * 20), size=60)
        knobs_layout.addWidget(decay_container)
        self.decay_knob = decay_container.findChild(QDial)
        self.decay_label_value = decay_container.findChild(QLabel, "value_label")

        # Sustain
        sustain_container = self.create_knob_with_label("Sustain", 0, 100, 70,
                                                         lambda v: self.update_adsr('sustain', v), size=60)
        knobs_layout.addWidget(sustain_container)
        self.sustain_knob = sustain_container.findChild(QDial)
        self.sustain_label_value = sustain_container.findChild(QLabel, "value_label")

        # Release (use 0-100 range internally, scale to 0-5000 in callback)
        release_container = self.create_knob_with_label("Release", 0, 100, 6,
                                                         lambda v: self.update_adsr('release', v * 50), size=60)
        knobs_layout.addWidget(release_container)
        self.release_knob = release_container.findChild(QDial)
        self.release_label_value = release_container.findChild(QLabel, "value_label")

        layout.addLayout(knobs_layout)

        return section

    def create_adsr_section_sliders(self):
        """Create ADSR envelope section with vertical sliders (JUNO-106 style)"""
        section = QWidget()
        layout = QHBoxLayout(section)
        layout.setSpacing(20)
        layout.setContentsMargins(10, 10, 10, 10)

        # Attack slider
        attack_container = QWidget()
        attack_layout = QVBoxLayout(attack_container)
        attack_layout.setSpacing(5)

        attack_label = QLabel("ATTACK")
        attack_label.setAlignment(Qt.AlignCenter)
        attack_layout.addWidget(attack_label)

        self.attack_slider = QSlider(Qt.Vertical)
        self.attack_slider.setRange(0, 100)
        self.attack_slider.setValue(0)
        self.attack_slider.setTickPosition(QSlider.TicksRight)
        self.attack_slider.setFixedHeight(150)
        self.attack_slider.valueChanged.connect(lambda v: self.update_adsr('attack', v * 20))
        attack_layout.addWidget(self.attack_slider, alignment=Qt.AlignCenter)

        self.attack_slider_value = QLabel("0ms")
        self.attack_slider_value.setAlignment(Qt.AlignCenter)
        self.attack_slider_value.setMinimumWidth(60)
        attack_layout.addWidget(self.attack_slider_value)

        layout.addWidget(attack_container)

        # Decay slider
        decay_container = QWidget()
        decay_layout = QVBoxLayout(decay_container)
        decay_layout.setSpacing(5)

        decay_label = QLabel("DECAY")
        decay_label.setAlignment(Qt.AlignCenter)
        decay_layout.addWidget(decay_label)

        self.decay_slider = QSlider(Qt.Vertical)
        self.decay_slider.setRange(0, 100)
        self.decay_slider.setValue(5)
        self.decay_slider.setTickPosition(QSlider.TicksRight)
        self.decay_slider.setFixedHeight(150)
        self.decay_slider.valueChanged.connect(lambda v: self.update_adsr('decay', v * 20))
        decay_layout.addWidget(self.decay_slider, alignment=Qt.AlignCenter)

        self.decay_slider_value = QLabel("100ms")
        self.decay_slider_value.setAlignment(Qt.AlignCenter)
        self.decay_slider_value.setMinimumWidth(60)
        decay_layout.addWidget(self.decay_slider_value)

        layout.addWidget(decay_container)

        # Sustain slider
        sustain_container = QWidget()
        sustain_layout = QVBoxLayout(sustain_container)
        sustain_layout.setSpacing(5)

        sustain_label = QLabel("SUSTAIN")
        sustain_label.setAlignment(Qt.AlignCenter)
        sustain_layout.addWidget(sustain_label)

        self.sustain_slider = QSlider(Qt.Vertical)
        self.sustain_slider.setRange(0, 100)
        self.sustain_slider.setValue(70)
        self.sustain_slider.setTickPosition(QSlider.TicksRight)
        self.sustain_slider.setFixedHeight(150)
        self.sustain_slider.valueChanged.connect(lambda v: self.update_adsr('sustain', v))
        sustain_layout.addWidget(self.sustain_slider, alignment=Qt.AlignCenter)

        self.sustain_slider_value = QLabel("70%")
        self.sustain_slider_value.setAlignment(Qt.AlignCenter)
        self.sustain_slider_value.setMinimumWidth(60)
        sustain_layout.addWidget(self.sustain_slider_value)

        layout.addWidget(sustain_container)

        # Release slider
        release_container = QWidget()
        release_layout = QVBoxLayout(release_container)
        release_layout.setSpacing(5)

        release_label = QLabel("RELEASE")
        release_label.setAlignment(Qt.AlignCenter)
        release_layout.addWidget(release_label)

        self.release_slider = QSlider(Qt.Vertical)
        self.release_slider.setRange(0, 100)
        self.release_slider.setValue(6)
        self.release_slider.setTickPosition(QSlider.TicksRight)
        self.release_slider.setFixedHeight(150)
        self.release_slider.valueChanged.connect(lambda v: self.update_adsr('release', v * 50))
        release_layout.addWidget(self.release_slider, alignment=Qt.AlignCenter)

        self.release_slider_value = QLabel("300ms")
        self.release_slider_value.setAlignment(Qt.AlignCenter)
        self.release_slider_value.setMinimumWidth(60)
        release_layout.addWidget(self.release_slider_value)

        layout.addWidget(release_container)

        layout.addStretch(1)

        return section

    def create_filter_section(self):
        """Create filter section"""
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)

        # Knobs layout
        knobs_layout = QHBoxLayout()
        knobs_layout.setSpacing(15)

        # Cutoff (Large: 80x80, use 0-100 range internally, scale to 20-5000 logarithmically)
        cutoff_container = self.create_knob_with_label("Cutoff", 0, 100, 100,
                                                        lambda v: self.update_filter('cutoff', int(20 * (250 ** (v/100)))), size=80)
        knobs_layout.addWidget(cutoff_container)
        self.cutoff_knob = cutoff_container.findChild(QDial)
        self.cutoff_label_value = cutoff_container.findChild(QLabel, "value_label")

        # Resonance (Small: 45x45)
        resonance_container = self.create_knob_with_label("Resonance", 0, 100, 0,
                                                           lambda v: self.update_filter('resonance', v), size=45)
        knobs_layout.addWidget(resonance_container)
        self.resonance_knob = resonance_container.findChild(QDial)
        self.resonance_label_value = resonance_container.findChild(QLabel, "value_label")

        layout.addLayout(knobs_layout)

        return section

    def create_lfo_section(self):
        """Create LFO section with waveform, rate, and modulation matrix with depth + mix controls"""
        section = QWidget()
        main_layout = QVBoxLayout(section)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Top row: LFO controls
        lfo_controls_layout = QHBoxLayout()
        lfo_controls_layout.setSpacing(15)

        # Waveform selector
        waveform_container = QWidget()
        waveform_layout = QVBoxLayout(waveform_container)
        waveform_label = QLabel("Shape")
        waveform_label.setAlignment(Qt.AlignCenter)
        waveform_layout.addWidget(waveform_label)
        self.lfo_waveform_combo = QComboBox()
        self.lfo_waveform_combo.addItems(["Sine", "Triangle", "Square", "Sawtooth", "Random"])
        self.lfo_waveform_combo.setFixedWidth(120)
        self.lfo_waveform_combo.currentTextChanged.connect(lambda v: setattr(self.lfo, 'waveform', v))
        waveform_layout.addWidget(self.lfo_waveform_combo)
        lfo_controls_layout.addWidget(waveform_container)

        # Rate Mode selector (Sync)
        mode_container = QWidget()
        mode_layout = QVBoxLayout(mode_container)
        mode_label = QLabel("Sync")
        mode_label.setAlignment(Qt.AlignCenter)
        mode_layout.addWidget(mode_label)
        self.lfo_mode_combo = QComboBox()
        self.lfo_mode_combo.addItems(["Free", "Sync"])
        self.lfo_mode_combo.setFixedWidth(100)
        self.lfo_mode_combo.currentTextChanged.connect(self.update_lfo_mode)
        mode_layout.addWidget(self.lfo_mode_combo)
        lfo_controls_layout.addWidget(mode_container)

        # Rate knob (Free mode: 0.1-20 Hz) - labeled as Freq
        self.lfo_rate_container = self.create_knob_with_label("Freq", 1, 200, 20,
                                                                lambda v: self.update_lfo_rate(v / 10.0), size=60)
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
                                                          lambda v: setattr(self.lfo, 'bpm', float(v)), size=60)
        self.bpm_knob = self.bpm_container.findChild(QDial)  # Get reference to the knob
        self.bpm_label_value = self.bpm_container.findChild(QLabel, "value_label")  # Get reference to value label
        self.bpm_container.setVisible(False)  # Hidden by default
        lfo_controls_layout.addWidget(self.bpm_container)

        lfo_controls_layout.addStretch()
        main_layout.addLayout(lfo_controls_layout)

        # Bottom: Modulation Matrix with Depth + Mix controls
        matrix_label = QLabel("Modulation Assignments (Depth / Mix)")
        matrix_label.setAlignment(Qt.AlignCenter)
        matrix_label.setFont(QFont("Arial", 10, QFont.Bold))
        main_layout.addWidget(matrix_label)

        # Create horizontal layout for grouped sliders
        lfo_matrix_layout = QHBoxLayout()
        lfo_matrix_layout.setSpacing(15)

        # Helper function to create a dual-slider group (depth + mix for each target)
        def create_dual_slider_group(group_title, slider_configs):
            group_layout = QVBoxLayout()
            group_layout.setSpacing(5)

            # Group title
            title = QLabel(group_title)
            title.setAlignment(Qt.AlignCenter)
            title.setFont(QFont("Arial", 9, QFont.Bold))
            group_layout.addWidget(title)

            # Sliders in horizontal row
            sliders_row = QHBoxLayout()
            sliders_row.setSpacing(12)  # Increased spacing to prevent overlap

            for label_text, depth_attr, mix_attr in slider_configs:
                # Container for one target (2 sliders + label)
                target_container = QVBoxLayout()
                target_container.setSpacing(2)

                # Horizontal row for depth + mix sliders
                dual_slider_row = QHBoxLayout()
                dual_slider_row.setSpacing(5)  # Increased spacing between depth and mix

                # Depth slider (ADSR style with tick marks)
                depth_slider = QSlider(Qt.Vertical)
                depth_slider.setMinimum(0)
                depth_slider.setMaximum(100)
                depth_slider.setValue(0)
                depth_slider.setFixedHeight(100)  # Doubled from 50px
                depth_slider.setTickPosition(QSlider.TicksRight)
                depth_slider.valueChanged.connect(lambda v, attr=depth_attr: setattr(self, attr, v / 100.0))
                dual_slider_row.addWidget(depth_slider)

                # Mix slider (ADSR style with tick marks)
                mix_slider = QSlider(Qt.Vertical)
                mix_slider.setMinimum(0)
                mix_slider.setMaximum(100)
                mix_slider.setValue(100)  # Default to 100% mix
                mix_slider.setFixedHeight(100)  # Doubled from 50px
                mix_slider.setTickPosition(QSlider.TicksRight)
                mix_slider.valueChanged.connect(lambda v, attr=mix_attr: setattr(self, attr, v / 100.0))
                dual_slider_row.addWidget(mix_slider)

                target_container.addLayout(dual_slider_row)

                # Label below sliders
                label = QLabel(label_text)
                label.setAlignment(Qt.AlignCenter)
                label.setFont(QFont("Arial", 8))
                target_container.addWidget(label)

                sliders_row.addLayout(target_container)

            group_layout.addLayout(sliders_row)
            return group_layout

        # Group 1: PITCH (3 targets, each with depth + mix)
        pitch_group = create_dual_slider_group("PITCH", [
            ("O1", "lfo_to_osc1_pitch", "lfo_to_osc1_pitch_mix"),
            ("O2", "lfo_to_osc2_pitch", "lfo_to_osc2_pitch_mix"),
            ("O3", "lfo_to_osc3_pitch", "lfo_to_osc3_pitch_mix")
        ])
        lfo_matrix_layout.addLayout(pitch_group)

        # Group 2: PULSE WIDTH (3 targets)
        pw_group = create_dual_slider_group("PULSE WIDTH", [
            ("O1", "lfo_to_osc1_pw", "lfo_to_osc1_pw_mix"),
            ("O2", "lfo_to_osc2_pw", "lfo_to_osc2_pw_mix"),
            ("O3", "lfo_to_osc3_pw", "lfo_to_osc3_pw_mix")
        ])
        lfo_matrix_layout.addLayout(pw_group)

        # Group 3: VOLUME (3 targets)
        vol_group = create_dual_slider_group("VOLUME", [
            ("O1", "lfo_to_osc1_volume", "lfo_to_osc1_volume_mix"),
            ("O2", "lfo_to_osc2_volume", "lfo_to_osc2_volume_mix"),
            ("O3", "lfo_to_osc3_volume", "lfo_to_osc3_volume_mix")
        ])
        lfo_matrix_layout.addLayout(vol_group)

        # Group 4: FILTER (1 target)
        filter_group = create_dual_slider_group("FILTER", [
            ("Cutoff", "lfo_to_filter_cutoff", "lfo_to_filter_cutoff_mix")
        ])
        lfo_matrix_layout.addLayout(filter_group)

        lfo_matrix_layout.addStretch()  # Push everything left
        main_layout.addLayout(lfo_matrix_layout)

        return section

    def update_lfo_mode(self, mode):
        """Update LFO rate mode and show/hide relevant controls"""
        self.lfo.rate_mode = mode
        if mode == "Free":
            self.lfo_rate_container.setVisible(True)
            self.sync_div_container.setVisible(False)
            self.bpm_container.setVisible(False)
            if self.bpm_knob:
                self.bpm_knob.setEnabled(True)
        else:  # Sync
            self.lfo_rate_container.setVisible(False)
            self.sync_div_container.setVisible(True)
            self.bpm_container.setVisible(True)
            # Disable BPM knob in Sync mode - BPM comes from MIDI clock
            if self.bpm_knob:
                self.bpm_knob.setEnabled(False)

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

    def octave_to_footage(self, octave):
        """Convert octave offset to organ footage notation"""
        footage_map = {
            -2: "32'",
            -1: "16'",
            0: "8'",
            1: "4'",
            2: "2'"
        }
        return footage_map.get(octave, "8'")

    def change_octave(self, osc_num, direction):
        """Change octave offset for an oscillator (+1 or -1)"""
        if osc_num == 1:
            self.octave1 = max(-2, min(2, self.octave1 + direction))
            self.octave1_label.setText(self.octave_to_footage(self.octave1))
            # Recalculate frequency with new octave
            if self.midi_handler.running and self.current_note is not None:
                base_freq = self.midi_note_to_freq(self.current_note)
                self.freq1 = self.apply_octave(self.apply_detune(base_freq, self.detune1), self.octave1)
            else:
                # In drone mode, apply octave to current frequency
                base_freq = self.freq1 / (2.0 ** (self.octave1 - direction))  # Undo previous octave
                self.freq1 = self.apply_octave(base_freq, self.octave1)
        elif osc_num == 2:
            self.octave2 = max(-2, min(2, self.octave2 + direction))
            self.octave2_label.setText(self.octave_to_footage(self.octave2))
            if self.midi_handler.running and self.current_note is not None:
                base_freq = self.midi_note_to_freq(self.current_note)
                self.freq2 = self.apply_octave(self.apply_detune(base_freq, self.detune2), self.octave2)
            else:
                base_freq = self.freq2 / (2.0 ** (self.octave2 - direction))
                self.freq2 = self.apply_octave(base_freq, self.octave2)
        else:
            self.octave3 = max(-2, min(2, self.octave3 + direction))
            self.octave3_label.setText(self.octave_to_footage(self.octave3))
            if self.midi_handler.running and self.current_note is not None:
                base_freq = self.midi_note_to_freq(self.current_note)
                self.freq3 = self.apply_octave(self.apply_detune(base_freq, self.detune3), self.octave3)
            else:
                base_freq = self.freq3 / (2.0 ** (self.octave3 - direction))
                self.freq3 = self.apply_octave(base_freq, self.octave3)

    def update_frequency(self, osc_num, value):
        """Update frequency from knob - acts as detune in chromatic mode, frequency in drone mode"""
        if self.playback_mode == 'chromatic':
            # Chromatic mode: knob controls detune in cents (-100 to +100)
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
            attack_val = value / 1000.0
            self.env1.attack = attack_val
            self.env2.attack = attack_val
            self.env3.attack = attack_val
            # Update all voice envelopes in real-time
            for voice in self.voice_pool:
                voice.env1.attack = attack_val
                voice.env2.attack = attack_val
                voice.env3.attack = attack_val
            # Update UI label
            self.attack_slider_value.setText(f"{int(value)}ms")
        elif param == 'decay':
            # Convert ms to seconds
            decay_val = value / 1000.0
            self.env1.decay = decay_val
            self.env2.decay = decay_val
            self.env3.decay = decay_val
            # Update all voice envelopes in real-time
            for voice in self.voice_pool:
                voice.env1.decay = decay_val
                voice.env2.decay = decay_val
                voice.env3.decay = decay_val
            # Update UI label
            self.decay_slider_value.setText(f"{int(value)}ms")
        elif param == 'sustain':
            # Convert percentage to 0-1
            sustain_val = value / 100.0
            self.env1.sustain = sustain_val
            self.env2.sustain = sustain_val
            self.env3.sustain = sustain_val
            # Update all voice envelopes in real-time
            for voice in self.voice_pool:
                voice.env1.sustain = sustain_val
                voice.env2.sustain = sustain_val
                voice.env3.sustain = sustain_val
            # Update UI label
            self.sustain_slider_value.setText(f"{int(value)}%")
        elif param == 'release':
            # Convert ms to seconds
            release_val = value / 1000.0
            self.env1.release = release_val
            self.env2.release = release_val
            self.env3.release = release_val
            # Update all voice envelopes in real-time
            for voice in self.voice_pool:
                voice.env1.release = release_val
                voice.env2.release = release_val
                voice.env3.release = release_val
            # Update UI label
            self.release_slider_value.setText(f"{int(value)}ms")

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

        # Determine current voice mode
        if self.mono_button.isChecked():
            voice_mode = "Mono"
        elif self.poly_button.isChecked():
            voice_mode = "Poly"
        elif self.unison_button.isChecked():
            voice_mode = "Unison"
        else:
            voice_mode = "Mono"  # Default

        # Create preset data structure
        preset = {
            "version": "1.0",  # For future compatibility
            "oscillators": {
                "osc1": {
                    "enabled": self.osc1_enabled,
                    "waveform": self.waveform1,
                    "frequency": self.freq1,
                    "detune": self.detune1,
                    "octave": self.octave1,
                    "pulse_width": self.pulse_width1,
                    "gain": self.gain1
                },
                "osc2": {
                    "enabled": self.osc2_enabled,
                    "waveform": self.waveform2,
                    "frequency": self.freq2,
                    "detune": self.detune2,
                    "octave": self.octave2,
                    "pulse_width": self.pulse_width2,
                    "gain": self.gain2
                },
                "osc3": {
                    "enabled": self.osc3_enabled,
                    "waveform": self.waveform3,
                    "frequency": self.freq3,
                    "detune": self.detune3,
                    "octave": self.octave3,
                    "pulse_width": self.pulse_width3,
                    "gain": self.gain3
                }
            },
            "voice_mode": voice_mode,
            "playback_mode": self.playback_mode,
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
            "lfo": {
                "waveform": self.lfo.waveform,
                "rate_mode": self.lfo.rate_mode,
                "rate_hz": self.lfo.rate_hz,
                "sync_division": self.lfo.sync_division,
                "bpm": self.lfo.bpm,
                "depth": {
                    "osc1_pitch": self.lfo_to_osc1_pitch,
                    "osc2_pitch": self.lfo_to_osc2_pitch,
                    "osc3_pitch": self.lfo_to_osc3_pitch,
                    "osc1_pw": self.lfo_to_osc1_pw,
                    "osc2_pw": self.lfo_to_osc2_pw,
                    "osc3_pw": self.lfo_to_osc3_pw,
                    "filter_cutoff": self.lfo_to_filter_cutoff,
                    "osc1_volume": self.lfo_to_osc1_volume,
                    "osc2_volume": self.lfo_to_osc2_volume,
                    "osc3_volume": self.lfo_to_osc3_volume
                },
                "mix": {
                    "osc1_pitch": self.lfo_to_osc1_pitch_mix,
                    "osc2_pitch": self.lfo_to_osc2_pitch_mix,
                    "osc3_pitch": self.lfo_to_osc3_pitch_mix,
                    "osc1_pw": self.lfo_to_osc1_pw_mix,
                    "osc2_pw": self.lfo_to_osc2_pw_mix,
                    "osc3_pw": self.lfo_to_osc3_pw_mix,
                    "filter_cutoff": self.lfo_to_filter_cutoff_mix,
                    "osc1_volume": self.lfo_to_osc1_volume_mix,
                    "osc2_volume": self.lfo_to_osc2_volume_mix,
                    "osc3_volume": self.lfo_to_osc3_volume_mix
                }
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

            # Load oscillator enabled states (default True if missing for backward compatibility)
            self.osc1_enabled = osc1.get("enabled", True)
            self.osc2_enabled = osc2.get("enabled", True)
            self.osc3_enabled = osc3.get("enabled", True)

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

            # Set mixer gains to 0.0 for disabled oscillators
            if not self.osc1_enabled:
                self.gain1 = 0.0
            if not self.osc2_enabled:
                self.gain2 = 0.0
            if not self.osc3_enabled:
                self.gain3 = 0.0

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

            # Voice mode (default "Mono" if missing for backward compatibility)
            voice_mode = preset.get("voice_mode", "Mono")

            # Playback mode (default "chromatic" if missing for backward compatibility)
            playback_mode = preset.get("playback_mode", "chromatic")
            self.set_playback_mode(playback_mode)

            # LFO settings
            lfo = preset.get("lfo", {})
            self.lfo.waveform = lfo.get("waveform", "Sine")
            self.lfo.rate_mode = lfo.get("rate_mode", "Free")
            self.lfo.rate_hz = lfo.get("rate_hz", 2.0)
            self.lfo.sync_division = lfo.get("sync_division", "1/4")
            self.lfo.bpm = lfo.get("bpm", 120.0)

            # LFO depth parameters
            lfo_depth = lfo.get("depth", {})
            self.lfo_to_osc1_pitch = lfo_depth.get("osc1_pitch", 0.0)
            self.lfo_to_osc2_pitch = lfo_depth.get("osc2_pitch", 0.0)
            self.lfo_to_osc3_pitch = lfo_depth.get("osc3_pitch", 0.0)
            self.lfo_to_osc1_pw = lfo_depth.get("osc1_pw", 0.0)
            self.lfo_to_osc2_pw = lfo_depth.get("osc2_pw", 0.0)
            self.lfo_to_osc3_pw = lfo_depth.get("osc3_pw", 0.0)
            self.lfo_to_filter_cutoff = lfo_depth.get("filter_cutoff", 0.0)
            self.lfo_to_osc1_volume = lfo_depth.get("osc1_volume", 0.0)
            self.lfo_to_osc2_volume = lfo_depth.get("osc2_volume", 0.0)
            self.lfo_to_osc3_volume = lfo_depth.get("osc3_volume", 0.0)

            # LFO mix parameters
            lfo_mix = lfo.get("mix", {})
            self.lfo_to_osc1_pitch_mix = lfo_mix.get("osc1_pitch", 1.0)
            self.lfo_to_osc2_pitch_mix = lfo_mix.get("osc2_pitch", 1.0)
            self.lfo_to_osc3_pitch_mix = lfo_mix.get("osc3_pitch", 1.0)
            self.lfo_to_osc1_pw_mix = lfo_mix.get("osc1_pw", 1.0)
            self.lfo_to_osc2_pw_mix = lfo_mix.get("osc2_pw", 1.0)
            self.lfo_to_osc3_pw_mix = lfo_mix.get("osc3_pw", 1.0)
            self.lfo_to_filter_cutoff_mix = lfo_mix.get("filter_cutoff", 1.0)
            self.lfo_to_osc1_volume_mix = lfo_mix.get("osc1_volume", 1.0)
            self.lfo_to_osc2_volume_mix = lfo_mix.get("osc2_volume", 1.0)
            self.lfo_to_osc3_volume_mix = lfo_mix.get("osc3_volume", 1.0)

            # Set voice mode
            if voice_mode == "Mono":
                self.set_voice_mode('mono')
            elif voice_mode == "Poly":
                self.set_voice_mode('poly')
            elif voice_mode == "Unison":
                self.set_voice_mode('unison')

            # Update UI to reflect loaded preset
            self.update_ui_from_preset()

            QMessageBox.information(self, "Success", f"Preset loaded from:\n{file_path}")

        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "Error", f"Invalid preset file:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load preset:\n{str(e)}")

    def load_factory_presets(self):
        """Scan Presets directory recursively for .json files"""
        presets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Presets")

        if not os.path.exists(presets_dir):
            print(f"Presets directory not found: {presets_dir}")
            return

        # Find all .json files recursively
        self.factory_presets = []
        for root, dirs, files in os.walk(presets_dir):
            for file in files:
                if file.endswith('.json'):
                    self.factory_presets.append(os.path.join(root, file))

        # Sort presets alphabetically by filename
        self.factory_presets.sort(key=lambda x: os.path.basename(x))

        # Make sure Init.json is first if it exists
        init_preset = os.path.join(presets_dir, "Init.json")
        if init_preset in self.factory_presets:
            self.factory_presets.remove(init_preset)
            self.factory_presets.insert(0, init_preset)

        print(f"Loaded {len(self.factory_presets)} factory presets")

    def next_preset(self):
        """Load next preset in list"""
        if not self.factory_presets:
            return

        self.current_preset_index = (self.current_preset_index + 1) % len(self.factory_presets)
        self.load_preset_by_index(self.current_preset_index)

    def prev_preset(self):
        """Load previous preset in list"""
        if not self.factory_presets:
            return

        self.current_preset_index = (self.current_preset_index - 1) % len(self.factory_presets)
        self.load_preset_by_index(self.current_preset_index)

    def load_preset_by_index(self, index):
        """Load specific preset by index"""
        if not self.factory_presets or index < 0 or index >= len(self.factory_presets):
            return

        file_path = self.factory_presets[index]

        try:
            with open(file_path, 'r') as f:
                preset = json.load(f)

            # Extract preset name from file or JSON
            self.current_preset_name = preset.get("name", os.path.splitext(os.path.basename(file_path))[0])

            # Extract oscillator settings with defaults for forward compatibility
            osc1 = preset.get("oscillators", {}).get("osc1", {})
            osc2 = preset.get("oscillators", {}).get("osc2", {})
            osc3 = preset.get("oscillators", {}).get("osc3", {})

            # Load oscillator enabled states (default True if missing for backward compatibility)
            self.osc1_enabled = osc1.get("enabled", True)
            self.osc2_enabled = osc2.get("enabled", True)
            self.osc3_enabled = osc3.get("enabled", True)

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

            # Set mixer gains to 0.0 for disabled oscillators
            if not self.osc1_enabled:
                self.gain1 = 0.0
            if not self.osc2_enabled:
                self.gain2 = 0.0
            if not self.osc3_enabled:
                self.gain3 = 0.0

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

            # Voice mode (default "Mono" if missing for backward compatibility)
            voice_mode = preset.get("voice_mode", "Mono")

            # Playback mode (default "chromatic" if missing for backward compatibility)
            playback_mode = preset.get("playback_mode", "chromatic")
            self.set_playback_mode(playback_mode)

            # LFO settings
            lfo = preset.get("lfo", {})
            self.lfo.waveform = lfo.get("waveform", "Sine")
            self.lfo.rate_mode = lfo.get("rate_mode", "Free")
            self.lfo.rate_hz = lfo.get("rate_hz", 2.0)
            self.lfo.sync_division = lfo.get("sync_division", "1/4")
            self.lfo.bpm = lfo.get("bpm", 120.0)

            # LFO depth parameters
            lfo_depth = lfo.get("depth", {})
            self.lfo_to_osc1_pitch = lfo_depth.get("osc1_pitch", 0.0)
            self.lfo_to_osc2_pitch = lfo_depth.get("osc2_pitch", 0.0)
            self.lfo_to_osc3_pitch = lfo_depth.get("osc3_pitch", 0.0)
            self.lfo_to_osc1_pw = lfo_depth.get("osc1_pw", 0.0)
            self.lfo_to_osc2_pw = lfo_depth.get("osc2_pw", 0.0)
            self.lfo_to_osc3_pw = lfo_depth.get("osc3_pw", 0.0)
            self.lfo_to_filter_cutoff = lfo_depth.get("filter_cutoff", 0.0)
            self.lfo_to_osc1_volume = lfo_depth.get("osc1_volume", 0.0)
            self.lfo_to_osc2_volume = lfo_depth.get("osc2_volume", 0.0)
            self.lfo_to_osc3_volume = lfo_depth.get("osc3_volume", 0.0)

            # LFO mix parameters
            lfo_mix = lfo.get("mix", {})
            self.lfo_to_osc1_pitch_mix = lfo_mix.get("osc1_pitch", 1.0)
            self.lfo_to_osc2_pitch_mix = lfo_mix.get("osc2_pitch", 1.0)
            self.lfo_to_osc3_pitch_mix = lfo_mix.get("osc3_pitch", 1.0)
            self.lfo_to_osc1_pw_mix = lfo_mix.get("osc1_pw", 1.0)
            self.lfo_to_osc2_pw_mix = lfo_mix.get("osc2_pw", 1.0)
            self.lfo_to_osc3_pw_mix = lfo_mix.get("osc3_pw", 1.0)
            self.lfo_to_filter_cutoff_mix = lfo_mix.get("filter_cutoff", 1.0)
            self.lfo_to_osc1_volume_mix = lfo_mix.get("osc1_volume", 1.0)
            self.lfo_to_osc2_volume_mix = lfo_mix.get("osc2_volume", 1.0)
            self.lfo_to_osc3_volume_mix = lfo_mix.get("osc3_volume", 1.0)

            # Set voice mode
            if voice_mode == "Mono":
                self.set_voice_mode('mono')
            elif voice_mode == "Poly":
                self.set_voice_mode('poly')
            elif voice_mode == "Unison":
                self.set_voice_mode('unison')

            # Update UI to reflect loaded preset
            self.update_ui_from_preset()

            # Update preset name label
            self.preset_name_label.setText(self.current_preset_name)

        except json.JSONDecodeError as e:
            print(f"Invalid preset file {file_path}: {str(e)}")
        except Exception as e:
            print(f"Failed to load preset {file_path}: {str(e)}")

    def update_ui_from_preset(self):
        """Update all UI elements to reflect current preset values"""
        # Block all signals during update to prevent callback interference
        # Update waveform selectors
        self.waveform1_combo.blockSignals(True)
        self.waveform2_combo.blockSignals(True)
        self.waveform3_combo.blockSignals(True)
        self.waveform1_combo.setCurrentText(self.waveform1)
        self.waveform2_combo.setCurrentText(self.waveform2)
        self.waveform3_combo.setCurrentText(self.waveform3)
        self.waveform1_combo.blockSignals(False)
        self.waveform2_combo.blockSignals(False)
        self.waveform3_combo.blockSignals(False)

        # Update frequency knobs - block signals to prevent triggering callbacks
        self.freq1_knob.blockSignals(True)
        self.freq2_knob.blockSignals(True)
        self.freq3_knob.blockSignals(True)
        # Calculate knob values based on logarithmic scale (20Hz to 5000Hz)
        if self.freq1 > 0:
            self.freq1_knob.setValue(int((np.log(self.freq1) - np.log(20)) / (np.log(5000) - np.log(20)) * 100))
        if self.freq2 > 0:
            self.freq2_knob.setValue(int((np.log(self.freq2) - np.log(20)) / (np.log(5000) - np.log(20)) * 100))
        if self.freq3 > 0:
            self.freq3_knob.setValue(int((np.log(self.freq3) - np.log(20)) / (np.log(5000) - np.log(20)) * 100))
        self.freq1_knob.blockSignals(False)
        self.freq2_knob.blockSignals(False)
        self.freq3_knob.blockSignals(False)

        # Update frequency labels
        self.freq1_label.setText(f"{self.freq1:.1f} Hz")
        self.freq2_label.setText(f"{self.freq2:.1f} Hz")
        self.freq3_label.setText(f"{self.freq3:.1f} Hz")

        # Update octave labels
        self.octave1_label.setText(f"{self.octave1:+d}" if self.octave1 != 0 else "0")
        self.octave2_label.setText(f"{self.octave2:+d}" if self.octave2 != 0 else "0")
        self.octave3_label.setText(f"{self.octave3:+d}" if self.octave3 != 0 else "0")

        # Update pulse width knobs and labels
        self.pw1_knob.blockSignals(True)
        self.pw2_knob.blockSignals(True)
        self.pw3_knob.blockSignals(True)
        self.pw1_knob.setValue(int(self.pulse_width1 * 100))
        self.pw2_knob.setValue(int(self.pulse_width2 * 100))
        self.pw3_knob.setValue(int(self.pulse_width3 * 100))
        self.pw1_knob.blockSignals(False)
        self.pw2_knob.blockSignals(False)
        self.pw3_knob.blockSignals(False)
        self.pw1_label.setText(f"{int(self.pulse_width1 * 100)}%")
        self.pw2_label.setText(f"{int(self.pulse_width2 * 100)}%")
        self.pw3_label.setText(f"{int(self.pulse_width3 * 100)}%")

        # Update mixer gain knobs (including 0 for disabled oscillators)
        self.gain1_knob.blockSignals(True)
        self.gain2_knob.blockSignals(True)
        self.gain3_knob.blockSignals(True)
        self.gain1_knob.setValue(int(self.gain1 * 100))
        self.gain2_knob.setValue(int(self.gain2 * 100))
        self.gain3_knob.setValue(int(self.gain3 * 100))
        self.gain1_knob.blockSignals(False)
        self.gain2_knob.blockSignals(False)
        self.gain3_knob.blockSignals(False)

        # Update voice mode buttons
        self.mono_button.blockSignals(True)
        self.poly_button.blockSignals(True)
        self.unison_button.blockSignals(True)
        self.mono_button.setChecked(self.mono_button.isChecked())
        self.poly_button.setChecked(self.poly_button.isChecked())
        self.unison_button.setChecked(self.unison_button.isChecked())
        self.mono_button.blockSignals(False)
        self.poly_button.blockSignals(False)
        self.unison_button.blockSignals(False)

        # Update ADSR sliders (envelope values are in seconds, sliders are 0-100)
        self.attack_slider.blockSignals(True)
        self.decay_slider.blockSignals(True)
        self.sustain_slider.blockSignals(True)
        self.release_slider.blockSignals(True)
        self.attack_slider.setValue(int(self.env1.attack * 50))  # seconds * 1000 / 20 = * 50
        self.decay_slider.setValue(int(self.env1.decay * 50))    # seconds * 1000 / 20 = * 50
        self.sustain_slider.setValue(int(self.env1.sustain * 100))  # 0-1 to 0-100
        self.release_slider.setValue(int(self.env1.release * 20))  # seconds * 1000 / 50 = * 20
        self.attack_slider.blockSignals(False)
        self.decay_slider.blockSignals(False)
        self.sustain_slider.blockSignals(False)
        self.release_slider.blockSignals(False)

        # Update ADSR labels (convert seconds to milliseconds for display)
        self.attack_slider_value.setText(f"{int(self.env1.attack * 1000)}ms")
        self.decay_slider_value.setText(f"{int(self.env1.decay * 1000)}ms")
        self.sustain_slider_value.setText(f"{int(self.env1.sustain * 100)}%")
        self.release_slider_value.setText(f"{int(self.env1.release * 1000)}ms")

        # Update master volume
        self.master_volume_knob.blockSignals(True)
        self.master_volume_knob.setValue(int(self.master_volume * 100))
        self.master_volume_knob.blockSignals(False)

        # Update filter knobs
        self.cutoff_knob.blockSignals(True)
        self.resonance_knob.blockSignals(True)
        # Cutoff: reverse the logarithmic formula (cutoff_hz = 20 * (250 ** (v/100)))
        # v = 100 * log(cutoff_hz / 20) / log(250)
        if self.filter.cutoff > 0:
            cutoff_knob_value = int(100 * np.log(self.filter.cutoff / 20) / np.log(250))
            self.cutoff_knob.setValue(cutoff_knob_value)
        # Resonance: direct percentage
        self.resonance_knob.setValue(int(self.filter.resonance * 100))
        self.cutoff_knob.blockSignals(False)
        self.resonance_knob.blockSignals(False)

        # Update LFO waveform and mode
        self.lfo_waveform_combo.blockSignals(True)
        self.lfo_mode_combo.blockSignals(True)
        self.lfo_waveform_combo.setCurrentText(self.lfo.waveform)
        self.lfo_mode_combo.setCurrentText(self.lfo.rate_mode)
        self.lfo_waveform_combo.blockSignals(False)
        self.lfo_mode_combo.blockSignals(False)

        # Update LFO sync division if needed
        if hasattr(self, 'lfo_sync_combo'):
            self.lfo_sync_combo.blockSignals(True)
            self.lfo_sync_combo.setCurrentText(self.lfo.sync_division)
            self.lfo_sync_combo.blockSignals(False)

        # Update oscillator enable/disable buttons
        # Sync osc_on variables with osc_enabled from preset
        self.osc1_on = self.osc1_enabled
        self.osc2_on = self.osc2_enabled
        self.osc3_on = self.osc3_enabled

        # Update oscillator 1 button
        if self.osc1_on:
            self.osc1_button.setText("ON")
            self.osc1_button.setStyleSheet("""
                QPushButton {
                    background-color: #fc5b42;
                    color: black;
                    border: none;
                    border-radius: 16px;
                    min-width: 33px;
                    max-width: 33px;
                    min-height: 33px;
                    max-height: 33px;
                }
                QPushButton:hover {
                    background-color: #fc5b42;
                }
            """)
        else:
            self.osc1_button.setText("OFF")
            self.osc1_button.setStyleSheet("""
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

        # Update oscillator 2 button
        if self.osc2_on:
            self.osc2_button.setText("ON")
            self.osc2_button.setStyleSheet("""
                QPushButton {
                    background-color: #fc5b42;
                    color: black;
                    border: none;
                    border-radius: 16px;
                    min-width: 33px;
                    max-width: 33px;
                    min-height: 33px;
                    max-height: 33px;
                }
                QPushButton:hover {
                    background-color: #fc5b42;
                }
            """)
        else:
            self.osc2_button.setText("OFF")
            self.osc2_button.setStyleSheet("""
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

        # Update oscillator 3 button
        if self.osc3_on:
            self.osc3_button.setText("ON")
            self.osc3_button.setStyleSheet("""
                QPushButton {
                    background-color: #fc5b42;
                    color: black;
                    border: none;
                    border-radius: 16px;
                    min-width: 33px;
                    max-width: 33px;
                    min-height: 33px;
                    max-height: 33px;
                }
                QPushButton:hover {
                    background-color: #fc5b42;
                }
            """)
        else:
            self.osc3_button.setText("OFF")
            self.osc3_button.setStyleSheet("""
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

        # Update power button
        self.update_power_button()

    def reallocate_voice_pool(self):
        """Create voice pool based on max_polyphony and unison_count"""
        # Total voices needed: max_polyphony * unison_count (max 8 total)
        total_voices = min(8, self.max_polyphony * self.unison_count)

        # Create new voice pool
        self.voice_pool = [Voice(self.sample_rate) for _ in range(total_voices)]

        # Clear active voices
        self.active_voices = {}

        # Copy envelope settings to all voices
        for voice in self.voice_pool:
            voice.env1.attack = self.env1.attack
            voice.env1.decay = self.env1.decay
            voice.env1.sustain = self.env1.sustain
            voice.env1.release = self.env1.release

            voice.env2.attack = self.env2.attack
            voice.env2.decay = self.env2.decay
            voice.env2.sustain = self.env2.sustain
            voice.env2.release = self.env2.release

            voice.env3.attack = self.env3.attack
            voice.env3.decay = self.env3.decay
            voice.env3.sustain = self.env3.sustain
            voice.env3.release = self.env3.release

    def calculate_unison_detune_pattern(self, unison_count):
        """Returns list of detune amounts in cents for unison voices

        Args:
            unison_count: Number of unison voices (1-8)

        Returns:
            List of detune values in cents
        """
        if unison_count <= 1:
            return [0.0]

        # Symmetric detune pattern around center
        # For 2 voices: [-detune, +detune]
        # For 3 voices: [-detune, 0, +detune]
        # For 4 voices: [-detune, -detune/3, +detune/3, +detune]
        # etc.

        detune_pattern = []
        for i in range(unison_count):
            # Map i from 0..(count-1) to -1..+1
            if unison_count == 1:
                offset = 0.0
            else:
                offset = -1.0 + (2.0 * i / (unison_count - 1))

            detune_pattern.append(offset * self.unison_detune_amount)

        return detune_pattern

    def steal_voice(self):
        """Find best voice to steal (prefer oldest in release phase)

        Returns:
            Voice object to steal, or None if no voice available
        """
        if not self.voice_pool:
            return None

        # First, try to find a voice in release phase (oldest first)
        release_voices = [v for v in self.voice_pool
                         if v.env1.phase == 'release' or v.env2.phase == 'release' or v.env3.phase == 'release']
        if release_voices:
            # Return oldest release voice
            return max(release_voices, key=lambda v: v.age)

        # If no release voices, try to find idle voices
        idle_voices = [v for v in self.voice_pool if v.is_free()]
        if idle_voices:
            return idle_voices[0]

        # Last resort: steal oldest active voice
        return max(self.voice_pool, key=lambda v: v.age)

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
        """Generate and mix all active voices with envelopes and filter"""
        # Don't print in audio callback - causes buffer underruns

        # If power is off, output silence
        if not self.power_on:
            outdata[:, 0] = 0
            outdata[:, 1] = 0
            return

        # Generate LFO signal (-1 to 1)
        lfo_signal = self.lfo.process(frames)

        # Create empty mixed output
        mixed = np.zeros(frames)

        # Count active voices for normalization
        active_count = 0

        # Loop through all voices in voice pool
        for voice in self.voice_pool:
            # Skip voices that are not active
            if not voice.is_active():
                continue

            # Skip voices with no note assigned (safety check - shouldn't happen)
            if voice.note is None:
                continue

            # Increment active voice count
            active_count += 1

            # Increment voice age for stealing priority
            voice.age += 1

            # Calculate base frequency for this voice's note
            base_freq = self.midi_note_to_freq(voice.note)

            # Apply unison detune to base frequency
            base_freq_detuned = self.apply_detune(base_freq, voice.unison_detune)

            # Generate each oscillator for this voice
            voice_mix = np.zeros(frames)

            # Oscillator 1
            if self.osc1_on:
                # In drone mode, use absolute frequency; in chromatic mode, use MIDI note + detune
                if self.playback_mode == 'drone':
                    freq1 = self.freq1
                else:
                    # Apply oscillator-specific detune and octave
                    freq1 = self.apply_octave(self.apply_detune(base_freq_detuned, self.detune1), self.octave1)

                # Apply pitch modulation (vibrato) - use mean for phase increment calculation
                freq_mod_scalar = 1.0 + (np.mean(lfo_signal) * 0.05 * self.lfo_to_osc1_pitch * self.lfo_to_osc1_pitch_mix)
                modulated_freq1 = freq1 * freq_mod_scalar
                phase_increment1 = 2 * np.pi * modulated_freq1 / self.sample_rate

                # Apply pulse width modulation - use mean for pw calculation
                pw_mod = np.mean(lfo_signal) * 0.3 * self.lfo_to_osc1_pw * self.lfo_to_osc1_pw_mix
                modulated_pw1 = np.clip(self.pulse_width1 + pw_mod, 0.01, 0.99)

                # Generate waveform using voice's phase
                wave1 = self.generate_waveform(self.waveform1, voice.phase1, phase_increment1, frames, modulated_pw1)
                env1 = voice.env1.process(frames)

                # Apply volume modulation (tremolo) - apply as array
                vol_mod = 1.0 + (lfo_signal * self.lfo_to_osc1_volume * self.lfo_to_osc1_volume_mix)
                modulated_gain1 = self.gain1 * np.clip(vol_mod, 0.0, 1.0)

                voice_mix += modulated_gain1 * wave1 * env1

                # Update voice phase
                voice.phase1 = (voice.phase1 + frames * phase_increment1) % (2 * np.pi)

            # Oscillator 2
            if self.osc2_on:
                # In drone mode, use absolute frequency; in chromatic mode, use MIDI note + detune
                if self.playback_mode == 'drone':
                    freq2 = self.freq2
                else:
                    # Apply oscillator-specific detune and octave
                    freq2 = self.apply_octave(self.apply_detune(base_freq_detuned, self.detune2), self.octave2)

                # Apply pitch modulation - use mean for phase increment calculation
                freq_mod_scalar = 1.0 + (np.mean(lfo_signal) * 0.05 * self.lfo_to_osc2_pitch * self.lfo_to_osc2_pitch_mix)
                modulated_freq2 = freq2 * freq_mod_scalar
                phase_increment2 = 2 * np.pi * modulated_freq2 / self.sample_rate

                # Apply pulse width modulation - use mean for pw calculation
                pw_mod = np.mean(lfo_signal) * 0.3 * self.lfo_to_osc2_pw * self.lfo_to_osc2_pw_mix
                modulated_pw2 = np.clip(self.pulse_width2 + pw_mod, 0.01, 0.99)

                # Generate waveform using voice's phase
                wave2 = self.generate_waveform(self.waveform2, voice.phase2, phase_increment2, frames, modulated_pw2)
                env2 = voice.env2.process(frames)

                # Apply volume modulation - apply as array
                vol_mod = 1.0 + (lfo_signal * self.lfo_to_osc2_volume * self.lfo_to_osc2_volume_mix)
                modulated_gain2 = self.gain2 * np.clip(vol_mod, 0.0, 1.0)

                voice_mix += modulated_gain2 * wave2 * env2

                # Update voice phase
                voice.phase2 = (voice.phase2 + frames * phase_increment2) % (2 * np.pi)

            # Oscillator 3
            if self.osc3_on:
                # In drone mode, use absolute frequency; in chromatic mode, use MIDI note + detune
                if self.playback_mode == 'drone':
                    freq3 = self.freq3
                else:
                    # Apply oscillator-specific detune and octave
                    freq3 = self.apply_octave(self.apply_detune(base_freq_detuned, self.detune3), self.octave3)

                # Apply pitch modulation - use mean for phase increment calculation
                freq_mod_scalar = 1.0 + (np.mean(lfo_signal) * 0.05 * self.lfo_to_osc3_pitch * self.lfo_to_osc3_pitch_mix)
                modulated_freq3 = freq3 * freq_mod_scalar
                phase_increment3 = 2 * np.pi * modulated_freq3 / self.sample_rate

                # Apply pulse width modulation - use mean for pw calculation
                pw_mod = np.mean(lfo_signal) * 0.3 * self.lfo_to_osc3_pw * self.lfo_to_osc3_pw_mix
                modulated_pw3 = np.clip(self.pulse_width3 + pw_mod, 0.01, 0.99)

                # Generate waveform using voice's phase
                wave3 = self.generate_waveform(self.waveform3, voice.phase3, phase_increment3, frames, modulated_pw3)
                env3 = voice.env3.process(frames)

                # Apply volume modulation - apply as array
                vol_mod = 1.0 + (lfo_signal * self.lfo_to_osc3_volume * self.lfo_to_osc3_volume_mix)
                modulated_gain3 = self.gain3 * np.clip(vol_mod, 0.0, 1.0)

                voice_mix += modulated_gain3 * wave3 * env3

                # Update voice phase
                voice.phase3 = (voice.phase3 + frames * phase_increment3) % (2 * np.pi)

            # Add this voice's output to the mix
            mixed += voice_mix

            # Clean up voice if all envelopes are idle (release finished)
            if not voice.is_active():
                voice.note = None  # Mark voice as free

        # Clean up active_voices dictionary - remove entries with all idle voices
        notes_to_remove = []
        for note, voices in self.active_voices.items():
            # Check if all voices for this note are now idle
            if all(not v.is_active() for v in voices):
                notes_to_remove.append(note)

        for note in notes_to_remove:
            del self.active_voices[note]

        # Apply voice count normalization (prevents clipping with multiple voices)
        # BUT: In unison mode (max_polyphony=1), all voices are playing the same note,
        # so we want full volume without normalization for thick unison sound
        if active_count > 0 and self.max_polyphony > 1:
            # Only normalize in polyphonic mode where multiple different notes play
            # Use sqrt normalization for better perceived loudness
            normalization = 1.0 / np.sqrt(active_count)
            mixed *= normalization

        # Apply filter with LFO modulation to cutoff
        if self.lfo_to_filter_cutoff > 0:
            lfo_mean = np.mean(lfo_signal)
            cutoff_mod = 2.0 ** (lfo_mean * 2.0 * self.lfo_to_filter_cutoff * self.lfo_to_filter_cutoff_mix)  # ±2 octaves
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
        elif osc_num == 2:
            self.osc2_on = not self.osc2_on
            button = self.osc2_button
        else:
            self.osc3_on = not self.osc3_on
            button = self.osc3_button

        # Update button appearance
        is_on = self.osc1_on if osc_num == 1 else (self.osc2_on if osc_num == 2 else self.osc3_on)

        if is_on:
            button.setText("ON")
            button.setStyleSheet("""
                QPushButton {
                    background-color: #fc5b42;
                    color: black;
                    border: none;
                    border-radius: 16px;
                    min-width: 33px;
                    max-width: 33px;
                    min-height: 33px;
                    max-height: 33px;
                }
                QPushButton:hover {
                    background-color: #fc5b42;
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

        # In drone mode, trigger/release a MIDI note when oscillator is toggled
        if self.playback_mode == 'drone':
            # Use MIDI note 69 (A440) as the base note for drone mode
            # The frequency will be determined by the oscillator knobs
            if is_on:
                # Trigger note on
                self.handle_midi_note_on(69, 100)
            else:
                # Trigger note off
                self.handle_midi_note_off(69)

        # Manage audio stream
        self.manage_audio_stream()

    def update_power_button(self):
        """Update power button appearance based on current power state"""
        if self.power_on:
            self.power_button.setText("POWER ON")
            self.power_button.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c;
                    color: #90ee90;
                    border: 2px solid #4a7c29;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #666666;
                }
                QPushButton:pressed {
                    background-color: #3c3c3c;
                }
            """)
        else:
            self.power_button.setText("POWER OFF")
            self.power_button.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c;
                    color: #ff6b6b;
                    border: 2px solid #8b2020;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #666666
                }
                QPushButton:pressed {
                    background-color: #3c3c3c;
                }
            """)

    def toggle_power(self):
        """Toggle master power on/off"""
        self.power_on = not self.power_on
        self.update_power_button()

    def set_voice_mode(self, mode):
        """Set voice mode: 'mono', 'poly', or 'unison'"""
        # Update button states
        self.mono_button.setChecked(mode == 'mono')
        self.poly_button.setChecked(mode == 'poly')
        self.unison_button.setChecked(mode == 'unison')

        # Configure voice settings
        if mode == 'mono':
            # Monophonic: 1 voice, no unison
            self.max_polyphony = 1
            self.unison_count = 1
        elif mode == 'poly':
            # Polyphonic: 8 voices, no unison
            self.max_polyphony = 8
            self.unison_count = 1
        elif mode == 'unison':
            # Unison: 1 note with 8 detuned voices
            self.max_polyphony = 1
            self.unison_count = 8

        # Reallocate voice pool
        self.reallocate_voice_pool()

    def set_playback_mode(self, mode):
        """Set playback mode: 'chromatic' or 'drone'"""
        self.playback_mode = mode

        # Update button states
        self.chromatic_button.setChecked(mode == 'chromatic')
        self.drone_button.setChecked(mode == 'drone')

        # Release all active voices when switching modes to prevent stuck notes
        for voice in self.voice_pool:
            voice.release()
        self.active_voices = {}
        self.current_note = None

        # Enable/disable octave step buttons based on mode
        octave_buttons_enabled = (mode == 'chromatic')
        self.osc1_octave_down_btn.setEnabled(octave_buttons_enabled)
        self.osc1_octave_up_btn.setEnabled(octave_buttons_enabled)
        self.osc2_octave_down_btn.setEnabled(octave_buttons_enabled)
        self.osc2_octave_up_btn.setEnabled(octave_buttons_enabled)
        self.osc3_octave_down_btn.setEnabled(octave_buttons_enabled)
        self.osc3_octave_up_btn.setEnabled(octave_buttons_enabled)

        # Update frequency knob displays based on mode
        if mode == 'chromatic':
            # Set knobs to center (0 cents) and update labels
            center_value = 50
            self.freq1_knob.setValue(center_value)
            self.freq2_knob.setValue(center_value)
            self.freq3_knob.setValue(center_value)
            # Labels will be updated by update_frequency callbacks
        else:
            # Drone mode: set knobs to minimum and update labels
            self.freq1_knob.setValue(0)
            self.freq2_knob.setValue(0)
            self.freq3_knob.setValue(0)
            # Labels will be updated by update_frequency callbacks

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
                # Set knobs to center (0 cents) and update labels
                center_value = 50  # Middle of 0-100 range = 0 cents
                self.freq1_knob.setValue(center_value)
                self.freq2_knob.setValue(center_value)
                self.freq3_knob.setValue(center_value)
                # Labels will be updated by update_frequency callbacks
        else:
            self.midi_handler.stop()
            # Chromatic mode with computer keyboard: set knobs to center (0 cents)
            center_value = 50  # Middle of 0-100 range = 0 cents
            self.freq1_knob.setValue(center_value)
            self.freq2_knob.setValue(center_value)
            self.freq3_knob.setValue(center_value)
            # Labels will be updated by update_frequency callbacks

    def midi_note_to_freq(self, note):
        """Convert MIDI note number to frequency"""
        return 440.0 * (2.0 ** ((note - 69) / 12.0))

    def handle_midi_note_on(self, note, velocity):
        """Handle MIDI note on message with polyphony/unison support"""
        self.current_note = note

        # In monophonic mode, force-reset ALL voices in the pool (not just active ones)
        # This ensures all 8 voices are immediately available for unison triggering
        if self.max_polyphony == 1:
            # Force reset ALL voices in the pool for immediate cutoff
            for voice in self.voice_pool:
                voice.env1.force_reset()
                voice.env2.force_reset()
                voice.env3.force_reset()
                voice.note = None
            # Clear active voices dict for monophonic mode
            self.active_voices = {}

        # Calculate how many voices we need for this note
        if self.max_polyphony == 1:
            # Monophonic mode with unison
            voices_needed = self.unison_count
        else:
            # Polyphonic mode (unison disabled)
            voices_needed = 1

        # Calculate unison detune pattern
        detune_pattern = self.calculate_unison_detune_pattern(voices_needed)

        # Find available voices
        available_voices = []

        # First, try to find free voices
        for voice in self.voice_pool:
            if voice.is_free() and voice not in available_voices:
                available_voices.append(voice)
                if len(available_voices) >= voices_needed:
                    break

        # If not enough free voices, steal voices (with safety limit)
        steal_attempts = 0
        max_steal_attempts = len(self.voice_pool) * 2  # Safety limit
        while len(available_voices) < voices_needed and steal_attempts < max_steal_attempts:
            steal_attempts += 1
            stolen_voice = self.steal_voice()
            if stolen_voice is None:
                break  # No voices available to steal
            if stolen_voice not in available_voices:
                available_voices.append(stolen_voice)
            else:
                # steal_voice() returned a voice we already have, try next iteration
                continue

        # Trigger voices with appropriate detune offsets and phase spread
        allocated_voices = []
        for i, voice in enumerate(available_voices[:voices_needed]):
            unison_detune = detune_pattern[i] if i < len(detune_pattern) else 0.0
            # Add subtle phase offset for unison mode to create width
            # Use small offsets (0 to π/4) to avoid harsh phasing
            phase_offset = (i / max(voices_needed - 1, 1)) * (np.pi / 4) if voices_needed > 1 else 0.0
            voice.trigger(note, velocity, unison_detune, phase_offset)
            allocated_voices.append(voice)

        # Track active voices for this note
        self.active_voices[note] = allocated_voices

        # Update legacy freq variables for backward compatibility
        # (used by UI display and some controls)
        base_freq = self.midi_note_to_freq(note)
        self.freq1 = self.apply_octave(self.apply_detune(base_freq, self.detune1), self.octave1)
        self.freq2 = self.apply_octave(self.apply_detune(base_freq, self.detune2), self.octave2)
        self.freq3 = self.apply_octave(self.apply_detune(base_freq, self.detune3), self.octave3)

        # Start audio if needed
        self.manage_audio_stream()

    def handle_midi_note_off(self, note):
        """Handle MIDI note off message with voice management"""
        # Look up voices for this note
        if note in self.active_voices:
            voices = self.active_voices[note]

            # Release all voices for this note
            for voice in voices:
                voice.release()

            # Remove from active voices dict
            del self.active_voices[note]

        # Update legacy current_note for backward compatibility
        if self.current_note == note:
            self.current_note = None

    def handle_midi_bpm_change(self, bpm):
        """Handle BPM changes from MIDI clock"""
        # Update LFO BPM
        self.lfo.bpm = bpm

        # Update UI - block signals to prevent feedback
        if self.bpm_knob and self.bpm_label_value:
            self.bpm_knob.blockSignals(True)
            self.bpm_knob.setValue(int(bpm))
            self.bpm_label_value.setText(str(int(bpm)))
            self.bpm_knob.blockSignals(False)

    def get_keyboard_note_mapping(self):
        """Map keyboard keys to MIDI notes (piano-style layout)"""
        # Bottom row: Z X C V B N M , . /  (white keys C-E)
        # Middle row: A S D F G H J K L ;  (white keys C-E one octave up)
        # Top row:    Q W E R T Y U I O P  (white keys C-E two octaves up)
        # Black keys (sharps): W E  T Y U  (using number row)

        base_note = self.keyboard_octave * 12  # C in current octave

        return {
            # Bottom row - Octave 0 (relative to keyboard_octave)
            Qt.Key_Z: base_note + 0,   # C
            Qt.Key_S: base_note + 1,   # C#
            Qt.Key_X: base_note + 2,   # D
            Qt.Key_D: base_note + 3,   # D#
            Qt.Key_C: base_note + 4,   # E
            Qt.Key_V: base_note + 5,   # F
            Qt.Key_G: base_note + 6,   # F#
            Qt.Key_B: base_note + 7,   # G
            Qt.Key_H: base_note + 8,   # G#
            Qt.Key_N: base_note + 9,   # A
            Qt.Key_J: base_note + 10,  # A#
            Qt.Key_M: base_note + 11,  # B
            Qt.Key_Comma: base_note + 12,  # C (next octave)

            # Top row - Octave +1
            Qt.Key_Q: base_note + 12,  # C
            Qt.Key_2: base_note + 13,  # C#
            Qt.Key_W: base_note + 14,  # D
            Qt.Key_3: base_note + 15,  # D#
            Qt.Key_E: base_note + 16,  # E
            Qt.Key_R: base_note + 17,  # F
            Qt.Key_5: base_note + 18,  # F#
            Qt.Key_T: base_note + 19,  # G
            Qt.Key_6: base_note + 20,  # G#
            Qt.Key_Y: base_note + 21,  # A
            Qt.Key_7: base_note + 22,  # A#
            Qt.Key_U: base_note + 23,  # B
            Qt.Key_I: base_note + 24,  # C (next octave)
        }

    def keyPressEvent(self, event):
        """Handle keyboard key press for MIDI input"""
        # Ignore auto-repeat
        if event.isAutoRepeat():
            return

        key = event.key()

        # Octave controls: Z and X shift octave down/up
        if key == Qt.Key_BracketLeft:
            self.keyboard_octave = max(0, self.keyboard_octave - 1)
            print(f"Keyboard octave: {self.keyboard_octave}")
            return
        elif key == Qt.Key_BracketRight:
            self.keyboard_octave = min(8, self.keyboard_octave + 1)
            print(f"Keyboard octave: {self.keyboard_octave}")
            return

        # Check if key is mapped to a note
        note_mapping = self.get_keyboard_note_mapping()
        if key in note_mapping:
            midi_note = note_mapping[key]

            # Avoid retriggering if key is already pressed
            if key not in self.pressed_keys:
                self.pressed_keys.add(key)
                # Trigger note on (velocity 100)
                self.handle_midi_note_on(midi_note, 100)

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Handle keyboard key release for MIDI input"""
        # Ignore auto-repeat
        if event.isAutoRepeat():
            return

        key = event.key()

        # Check if key is mapped to a note
        note_mapping = self.get_keyboard_note_mapping()
        if key in note_mapping:
            midi_note = note_mapping[key]

            # Remove from pressed keys
            if key in self.pressed_keys:
                self.pressed_keys.discard(key)
                # Trigger note off
                self.handle_midi_note_off(midi_note)

        super().keyReleaseEvent(event)

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
