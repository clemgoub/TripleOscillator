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
import queue
from scipy import signal as scipy_signal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QDial, QComboBox,
                             QFileDialog, QMessageBox, QSlider, QGridLayout, QGroupBox,
                             QProgressBar, QToolButton)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPainter, QPen, QColor, QRadialGradient
import pyqtgraph as pg


# Audio Processing Constants
# ---------------------------

# Filter Constants
Q_BUTTERWORTH = 0.707      # Butterworth filter Q (maximally flat response)
Q_SCALE = 9.293            # Scale factor for resonance to Q mapping (10.0 - 0.707)
Q_MIN = 0.5                # Minimum Q value for stability
Q_MAX = 10.0               # Maximum Q value (prevents self-oscillation, reduced for better stability)
FREQ_MIN_NORMALIZED = 0.001  # Minimum normalized frequency (cutoff/sample_rate)
FREQ_MAX_NORMALIZED = 0.499  # Maximum normalized frequency (prevents Nyquist aliasing)
FILTER_OUTPUT_MIN = -2.0   # Minimum filter output (prevents clipping)
FILTER_OUTPUT_MAX = 2.0    # Maximum filter output (prevents clipping)
FILTER_STATE_MIN = -10.0   # Minimum filter state value (prevents runaway)
FILTER_STATE_MAX = 10.0    # Maximum filter state value (prevents runaway)
FILTER_CUTOFF_MIN = 20.0   # Minimum filter cutoff frequency (Hz)
FILTER_CUTOFF_MAX = 20000.0  # Maximum filter cutoff frequency (Hz)

# LFO Modulation Depths
LFO_PITCH_MOD_DEPTH = 0.05  # Pitch modulation depth (±5% frequency deviation)
LFO_PW_MOD_DEPTH = 0.3      # Pulse width modulation depth (±30% pulse width deviation)
LFO_FILTER_MOD_OCTAVES = 2.0  # Filter cutoff modulation range (±2 octaves)

# LFO Destination Options
LFO_DESTINATIONS = [
    "None",
    "All OSCs Pitch",
    "Filter Cutoff",
    "All OSCs Volume",
    "OSC1 Pulse Width",
    "OSC2 Pulse Width",
    "OSC3 Pulse Width"
]

# Pulse Width Constraints
PW_MIN = 0.01  # Minimum pulse width (prevents duty cycle issues)
PW_MAX = 0.99  # Maximum pulse width (prevents duty cycle issues)

# Volume Modulation
VOL_MOD_MIN = 0.0  # Minimum volume modulation (silence)
VOL_MOD_MAX = 1.0  # Maximum volume modulation (full volume)

# Envelope Anti-Click
ENV_MIN_ATTACK = 0.005  # Minimum attack time in seconds (5ms anti-click fade)


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
        """Trigger note on (start attack phase with legato retrigger)

        Uses legato retriggering: if envelope is already active, start attack
        from current level instead of 0. This prevents clicks when stealing voices.
        """
        self.phase = 'attack'
        self.samples_in_phase = 0

        # Legato retrigger: Start attack from current level if already active
        # This prevents clicks when stealing voices (voice goes from sustain → attack smoothly)
        if self.level > 0.01:  # If envelope is already active
            self.attack_start_level = self.level  # Start from current level
        else:
            self.level = 0.0  # Only reset to 0 if envelope was idle
            self.attack_start_level = 0.0

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
                # Use minimum attack time to prevent clicks from instant attacks
                effective_attack = max(self.attack, ENV_MIN_ATTACK)
                attack_samples = max(1, int(effective_attack * self.sample_rate))
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


class MoogLadderFilter:
    """
    Moog-style 4-pole ladder filter with 24dB/octave rolloff.

    Implements a classic analog filter topology with resonance feedback.
    Supports three modes:
    - LP (Low-Pass): Output from 4th stage (24dB/octave)
    - BP (Band-Pass): Output from 2nd stage (12dB/octave)
    - HP (High-Pass): Input minus LP output

    Parameters:
        cutoff: Frequency in Hz (20-20000)
        resonance: Resonance amount (0-1), where 1 approaches self-oscillation
        filter_mode: "LP", "BP", or "HP"
    """
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.cutoff = 5000.0  # Hz
        self.resonance = 0.0  # 0-1
        self.filter_mode = "LP"  # "LP", "BP", or "HP"

        # 4 state variables (one per pole)
        self.stage1_state = 0.0
        self.stage2_state = 0.0
        self.stage3_state = 0.0
        self.stage4_state = 0.0

        # Coefficient caching for performance
        self.last_cutoff = None
        self.last_resonance = None
        self.last_mode = None
        self.g = None  # One-pole coefficient
        self.feedback_gain = None  # Resonance feedback

    def reset(self):
        """Reset filter state to prevent artifacts"""
        self.stage1_state = 0.0
        self.stage2_state = 0.0
        self.stage3_state = 0.0
        self.stage4_state = 0.0

    def _calculate_coefficients(self):
        """Calculate filter coefficients from cutoff and resonance"""
        # Normalize cutoff frequency (avoid Nyquist aliasing)
        freq = np.clip(self.cutoff / self.sample_rate, FREQ_MIN_NORMALIZED, FREQ_MAX_NORMALIZED)

        # Calculate one-pole coefficient (g) using bilinear transform with prewarp
        omega = 2.0 * np.pi * freq
        # Prewarp for better accuracy at high frequencies
        omega_warped = np.tan(omega / 2.0)
        self.g = omega_warped / (1.0 + omega_warped)

        # Resonance feedback gain (0-4 range typical for Moog)
        # Map 0-1 resonance to 0-3.5, with headroom to prevent signal kill
        # At resonance=1.0, should approach self-oscillation but remain musical
        self.feedback_gain = self.resonance * 3.5
        # Clamp to prevent instability
        self.feedback_gain = np.clip(self.feedback_gain, 0.0, 3.5)

        # Cache parameter values
        self.last_cutoff = self.cutoff
        self.last_resonance = self.resonance
        self.last_mode = self.filter_mode

    def process(self, input_signal):
        """Apply Moog ladder filter with state preservation"""
        # Recalculate coefficients if parameters changed
        if (self.last_cutoff != self.cutoff or
            self.last_resonance != self.resonance or
            self.last_mode != self.filter_mode):
            self._calculate_coefficients()

        num_samples = len(input_signal)
        output = np.zeros(num_samples, dtype=np.float32)

        # Process sample-by-sample (necessary for feedback loop)
        for i in range(num_samples):
            # Input with feedback subtraction (classic Moog topology)
            input_sample = input_signal[i] - self.feedback_gain * self.stage4_state

            # Stage 1: one-pole low-pass
            self.stage1_state += self.g * (input_sample - self.stage1_state)

            # Stage 2: one-pole low-pass
            self.stage2_state += self.g * (self.stage1_state - self.stage2_state)

            # Stage 3: one-pole low-pass
            self.stage3_state += self.g * (self.stage2_state - self.stage3_state)

            # Stage 4: one-pole low-pass
            self.stage4_state += self.g * (self.stage3_state - self.stage4_state)

            # Mode-dependent output selection
            if self.filter_mode == "LP":
                # Low-pass: output of 4th stage (24dB/octave)
                output[i] = self.stage4_state
            elif self.filter_mode == "BP":
                # Band-pass: output of 2nd stage
                output[i] = self.stage2_state
            elif self.filter_mode == "HP":
                # High-pass: use same input reference as filter stages (after feedback)
                # HP = (input - feedback) - LP(input - feedback)
                output[i] = input_sample - self.stage4_state

        # Simple output gain to compensate for 4-pole attenuation (same for all modes)
        # Moog filters naturally attenuate; a modest 2x boost is typical
        output = output * 2.0

        # Apply stability safeguards (preserve from original implementation)
        output = np.clip(output, FILTER_OUTPUT_MIN, FILTER_OUTPUT_MAX)

        # Clamp state variables
        self.stage1_state = np.clip(self.stage1_state, FILTER_STATE_MIN, FILTER_STATE_MAX)
        self.stage2_state = np.clip(self.stage2_state, FILTER_STATE_MIN, FILTER_STATE_MAX)
        self.stage3_state = np.clip(self.stage3_state, FILTER_STATE_MIN, FILTER_STATE_MAX)
        self.stage4_state = np.clip(self.stage4_state, FILTER_STATE_MIN, FILTER_STATE_MAX)

        # NaN/Inf detection and reset
        if (np.isnan(self.stage1_state) or np.isinf(self.stage1_state) or
            np.isnan(self.stage2_state) or np.isinf(self.stage2_state) or
            np.isnan(self.stage3_state) or np.isinf(self.stage3_state) or
            np.isnan(self.stage4_state) or np.isinf(self.stage4_state)):
            self.reset()

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
    """Represents a single voice with three oscillators and single envelope (applied post-mixer)"""
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.note = None  # MIDI note number (None if voice is free)
        self.velocity = 0
        self.age = 0  # For voice stealing (oldest-first strategy)

        # Phase accumulators for each oscillator (all start at 0 for clean unison)
        self.phase1 = 0
        self.phase2 = 0
        self.phase3 = 0

        # Single envelope generator (applied post-mixer for efficiency)
        self.env = EnvelopeGenerator(sample_rate)

        # Unison detuning offset (set when voice is allocated)
        self.unison_detune = 0.0  # In cents

    def is_active(self):
        """Voice is active if envelope is not idle"""
        return self.env.phase != 'idle'

    def is_free(self):
        """Voice is free if no note is assigned and envelope is idle"""
        return self.note is None and not self.is_active()

    def trigger(self, note, velocity, unison_detune=0.0, phase_offset=0.0):
        """Trigger this voice with a note

        Args:
            note: MIDI note number
            velocity: Note velocity (0-127)
            unison_detune: Detune amount in cents for unison spread
            phase_offset: Phase offset in radians (DEPRECATED - always starts at zero for click-free operation)
        """
        # Check if this is a fresh trigger (voice was idle) or a steal (voice was active)
        was_active = self.is_active()

        self.note = note
        self.velocity = velocity
        self.unison_detune = unison_detune
        self.age = 0

        # Only reset oscillator phases if voice was idle (fresh trigger)
        # If stealing an active voice, keep phases to avoid discontinuities (like analog synths)
        if not was_active:
            # Fresh trigger: Start at zero crossing for click-free note starts
            self.phase1 = 0.0
            self.phase2 = 0.0
            self.phase3 = 0.0
        # else: Voice stealing - keep current phases, just retune to new frequency
        # This allows oscillators to continue smoothly at their current phase position

        # Trigger envelope (uses legato retrigger if already active)
        self.env.trigger()

    def release(self):
        """Release this voice (start release phase, but keep note assigned until idle)"""
        # DON'T set note = None here - we need it for the audio callback during release
        # It will be set to None when envelope reaches idle phase
        self.env.release_note()


class NoiseGenerator:
    """Generates white, pink, and brown noise with envelope"""
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.noise_type = "White"  # White, Pink, or Brown

        # Envelope for noise (shares ADSR settings with oscillators)
        self.envelope = EnvelopeGenerator(sample_rate)

        # Pink noise filter state (using Paul Kellet's refined method)
        self.pink_b0 = 0.0
        self.pink_b1 = 0.0
        self.pink_b2 = 0.0
        self.pink_b3 = 0.0
        self.pink_b4 = 0.0
        self.pink_b5 = 0.0
        self.pink_b6 = 0.0

        # Brown noise integrator state
        self.brown_state = 0.0

    def trigger(self):
        """Trigger noise envelope"""
        self.envelope.trigger()

    def release(self):
        """Release noise envelope"""
        self.envelope.release_note()

    def generate_white(self, num_samples):
        """Generate white noise (uniform frequency spectrum)"""
        return np.random.uniform(-1.0, 1.0, num_samples)

    def generate_pink(self, num_samples):
        """Generate pink noise (1/f frequency spectrum) using Paul Kellet's method"""
        output = np.zeros(num_samples)

        for i in range(num_samples):
            white = np.random.uniform(-1.0, 1.0)

            # Apply pink noise filter (sum of multiple octaves)
            self.pink_b0 = 0.99886 * self.pink_b0 + white * 0.0555179
            self.pink_b1 = 0.99332 * self.pink_b1 + white * 0.0750759
            self.pink_b2 = 0.96900 * self.pink_b2 + white * 0.1538520
            self.pink_b3 = 0.86650 * self.pink_b3 + white * 0.3104856
            self.pink_b4 = 0.55000 * self.pink_b4 + white * 0.5329522
            self.pink_b5 = -0.7616 * self.pink_b5 - white * 0.0168980

            pink = (self.pink_b0 + self.pink_b1 + self.pink_b2 +
                   self.pink_b3 + self.pink_b4 + self.pink_b5 +
                   self.pink_b6 + white * 0.5362)

            self.pink_b6 = white * 0.115926

            # Normalize
            output[i] = pink * 0.11

        return output

    def generate_brown(self, num_samples):
        """Generate brown noise (Brownian/red noise, 1/f^2 spectrum)"""
        output = np.zeros(num_samples)

        for i in range(num_samples):
            white = np.random.uniform(-1.0, 1.0)
            # Integrate white noise with leak to prevent DC drift
            self.brown_state = (self.brown_state + white * 0.02) * 0.998
            # Clamp to prevent unbounded growth
            self.brown_state = np.clip(self.brown_state, -1.0, 1.0)
            output[i] = self.brown_state

        return output * 3.5  # Scale up brown noise

    def generate(self, num_samples):
        """Generate noise based on current type"""
        if self.noise_type == "Pink":
            return self.generate_pink(num_samples)
        elif self.noise_type == "Brown":
            return self.generate_brown(num_samples)
        else:  # White
            return self.generate_white(num_samples)


class SpectrumAnalyzerWindow(QMainWindow):
    """Separate window for real-time FFT spectrum analyzer"""
    def __init__(self, sample_rate=44100, parent=None):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.fft_size = 2048  # FFT size (power of 2 for efficiency)

        # Window setup
        self.setWindowTitle("Spectrum Analyzer")
        self.setMinimumSize(800, 400)
        self.resize(1000, 500)

        # Audio data queue (thread-safe communication from audio callback)
        self.audio_queue = queue.Queue(maxsize=10)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # Create PyQtGraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')  # Black background
        self.plot_widget.setLabel('left', 'Level', units='dB')
        self.plot_widget.setLabel('bottom', 'Frequency', units='Hz')

        # Fixed Y-axis scale (standard dB range)
        self.plot_widget.setYRange(-90, 100, padding=0)
        self.plot_widget.getPlotItem().setLimits(yMin=-90, yMax=100)

        # Fixed X-axis with conventional frequency breaks
        self.plot_widget.setXRange(np.log10(20), np.log10(20000), padding=0)
        self.plot_widget.setLogMode(x=True, y=False)

        # Set conventional frequency tick marks (20, 50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k)
        freq_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        freq_labels = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
        ax = self.plot_widget.getAxis('bottom')
        ax.setTicks([[(freq, label) for freq, label in zip(freq_ticks, freq_labels)]])

        # Standard dB tick marks
        db_ticks = list(range(0, -100, 80))
        ax_left = self.plot_widget.getAxis('left')
        ax_left.setTicks([[(db, str(db)) for db in db_ticks]])

        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Create plot curve
        self.curve = self.plot_widget.plot(pen=pg.mkPen('g', width=2))

        layout.addWidget(self.plot_widget)

        # Frequency bins for FFT
        self.freqs = np.fft.rfftfreq(self.fft_size, 1 / self.sample_rate)

        # Smoothing buffer for averaging FFT results
        self.smoothing = 0.7  # Smoothing factor (0 = no smoothing, 1 = maximum smoothing)
        self.magnitude_smooth = np.zeros(len(self.freqs))

        # Timer for updating display (30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_spectrum)
        self.timer.start(33)  # ~30 FPS

    def process_audio(self, audio_data):
        """Called from audio thread to pass audio data for analysis"""
        try:
            # Non-blocking put - drop data if queue is full (audio thread can't wait)
            self.audio_queue.put_nowait(audio_data.copy())
        except queue.Full:
            pass  # Drop frame if queue is full

    def update_spectrum(self):
        """Update spectrum display (called by timer in UI thread)"""
        # Get latest audio data from queue (non-blocking)
        audio_data = None
        try:
            # Get all queued data, keep only the latest
            while True:
                audio_data = self.audio_queue.get_nowait()
        except queue.Empty:
            pass

        if audio_data is None:
            return  # No new data

        # Pad or truncate to FFT size
        if len(audio_data) < self.fft_size:
            audio_data = np.pad(audio_data, (0, self.fft_size - len(audio_data)))
        else:
            audio_data = audio_data[:self.fft_size]

        # Apply Hann window to reduce spectral leakage
        windowed = audio_data * np.hanning(self.fft_size)

        # Compute FFT
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)

        # Convert to dB scale (with floor to avoid log(0))
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        # Apply smoothing (exponential moving average)
        self.magnitude_smooth = (self.smoothing * self.magnitude_smooth +
                                 (1 - self.smoothing) * magnitude_db)

        # Update plot (skip DC and very low frequencies for better display)
        # Find index for 20 Hz
        start_idx = np.searchsorted(self.freqs, 20)
        self.curve.setData(self.freqs[start_idx:], self.magnitude_smooth[start_idx:])

    def keyPressEvent(self, event):
        """Forward keyboard events to parent (main synth window) for MIDI control"""
        if self.parent():
            self.parent().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Forward keyboard release events to parent (main synth window) for MIDI control"""
        if self.parent():
            self.parent().keyReleaseEvent(event)
        else:
            super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        """Handle window resize to maintain proper spectrum display"""
        super().resizeEvent(event)
        # Re-apply fixed axis ranges after resize to ensure they're maintained
        self.plot_widget.setYRange(-90, 0, padding=0)
        self.plot_widget.setXRange(np.log10(20), np.log10(20000), padding=0)


class LFOLEDIndicator(QWidget):
    """LED-like indicator that displays LFO signal intensity"""
    def __init__(self, parent=None, size=30):
        super().__init__(parent)
        self.size = size
        self.value = 0.0  # -1.0 to 1.0
        self.setFixedSize(size, size)

    def set_value(self, value):
        """Set the LFO value (-1.0 to 1.0)"""
        self.value = max(-1.0, min(1.0, value))
        self.update()  # Trigger repaint

    def paintEvent(self, event):
        """Custom paint to draw the LED indicator"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Map value from -1..1 to 0..1 (absolute intensity)
        intensity = abs(self.value)

        # Background circle (dark)
        painter.setBrush(QColor(30, 30, 30))
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawEllipse(2, 2, self.size - 4, self.size - 4)

        # LED glow (color based on positive/negative, intensity based on abs value)
        if self.value >= 0:
            # Positive: cyan/blue
            color = QColor(0, int(180 * intensity), int(255 * intensity))
        else:
            # Negative: orange/red (just for visual variety, could be same color)
            color = QColor(0, int(180 * intensity), int(255 * intensity))

        # Create gradient for glow effect
        gradient = QRadialGradient(self.size / 2, self.size / 2, self.size / 2)
        gradient.setColorAt(0, color)
        gradient.setColorAt(0.6, color.darker(120))
        gradient.setColorAt(1, QColor(30, 30, 30))

        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(2, 2, self.size - 4, self.size - 4)


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

        # Single envelope generator - TEMPLATE ONLY, not used for audio
        # This template is copied to each voice with updated ADSR parameters
        self.env = EnvelopeGenerator(self.sample_rate)
        # Force template envelope to idle - it should never be active
        self.env.force_reset()

        # Filter
        self.filter = MoogLadderFilter(self.sample_rate)
        self.filter_cutoff_base = 5000.0  # Store the "dry" cutoff from knob (not modulated)

        # DC blocking filter state will be initialized on first use (no need to pre-allocate)

        # Dual LFOs
        self.lfo1 = LFOGenerator(self.sample_rate)
        self.lfo2 = LFOGenerator(self.sample_rate)

        # LFO1 Parameters (simplified destination system)
        self.lfo1_destination = "None"  # From LFO_DESTINATIONS
        self.lfo1_depth = 0.0  # 0.0 to 1.0
        self.lfo1_mix = 1.0    # 0.0 to 1.0 (dry/wet)

        # LFO2 Parameters
        self.lfo2_destination = "None"
        self.lfo2_depth = 0.0
        self.lfo2_mix = 1.0

        # LFO signal values for LED display (updated in audio callback)
        self.lfo1_signal_value = 0.0  # -1.0 to 1.0
        self.lfo2_signal_value = 0.0  # -1.0 to 1.0

        # Noise Generator
        self.noise = NoiseGenerator(self.sample_rate)
        self.noise_on = False
        self.noise_gain = 0.5  # 0-1

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

        # Level meter tracking
        self.peak_level = 0.0  # Current peak level (0.0 to 1.0+)
        self.clip_detected = False  # True if signal clipped (>=1.0)
        self.clip_hold_counter = 0  # Frames to hold clip indicator

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
        self.setMinimumSize(1420, 900)
        self.resize(1420, 900)

        # Create spectrum analyzer window (hidden initially)
        self.spectrum_analyzer_window = SpectrumAnalyzerWindow(sample_rate=self.sample_rate, parent=self)

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

        # Vertical layout for Spectrum button and Level meter (stacked)
        analysis_layout = QVBoxLayout()
        analysis_layout.setSpacing(5)

        # Spectrum Analyzer button (on top)
        spectrum_button = QPushButton("Spectrum")
        spectrum_button.setFont(QFont("Arial", 9))
        spectrum_button.setFixedSize(150, 25)
        spectrum_button.setStyleSheet("""
            QPushButton {
                background-color: #065f46;
                color: white;
                border: 2px solid #10b981;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #047857;
            }
            QPushButton:pressed {
                background-color: #064e3b;
            }
        """)
        spectrum_button.clicked.connect(self.toggle_spectrum_analyzer)
        analysis_layout.addWidget(spectrum_button)

        # Level meter (below)
        self.create_level_meter()
        analysis_layout.addWidget(self.level_meter_widget)

        midi_layout.addLayout(analysis_layout)

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

    def toggle_spectrum_analyzer(self):
        """Toggle spectrum analyzer window visibility"""
        if self.spectrum_analyzer_window.isVisible():
            self.spectrum_analyzer_window.hide()
        else:
            self.spectrum_analyzer_window.show()
            self.spectrum_analyzer_window.raise_()
            self.spectrum_analyzer_window.activateWindow()

    def create_level_meter(self):
        """Create a compact level meter with clip indicator"""
        self.level_meter_widget = QWidget()
        layout = QHBoxLayout(self.level_meter_widget)
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setSpacing(5)

        # Label
        label = QLabel("Level:")
        label.setFont(QFont("Arial", 9))
        layout.addWidget(label)

        # Progress bar for level
        self.level_meter_bar = QProgressBar()
        self.level_meter_bar.setOrientation(Qt.Horizontal)
        self.level_meter_bar.setFixedSize(100, 20)
        self.level_meter_bar.setRange(0, 100)
        self.level_meter_bar.setValue(0)
        self.level_meter_bar.setTextVisible(False)
        self.level_meter_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #1a1a1a;
            }
            QProgressBar::chunk {
                background-color: #10b981;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.level_meter_bar)

        # Clip indicator
        self.clip_indicator = QLabel("CLIP")
        self.clip_indicator.setFont(QFont("Arial", 9, QFont.Bold))
        self.clip_indicator.setFixedSize(40, 20)
        self.clip_indicator.setAlignment(Qt.AlignCenter)
        self.clip_indicator.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                color: #555555;
                border: 1px solid #555555;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.clip_indicator)

        # Add timer to update level meter (30 FPS)
        self.level_meter_timer = QTimer()
        self.level_meter_timer.timeout.connect(self.update_level_meter)
        self.level_meter_timer.start(33)  # ~30 FPS

        # Add timer to update LFO LED indicators (30 FPS)
        self.lfo_led_timer = QTimer()
        self.lfo_led_timer.timeout.connect(self.update_lfo_leds)
        self.lfo_led_timer.start(33)  # ~30 FPS

    def update_lfo_leds(self):
        """Update LFO LED indicators with current signal values"""
        if hasattr(self, 'lfo1_controls') and 'led' in self.lfo1_controls:
            self.lfo1_controls['led'].set_value(self.lfo1_signal_value)
        if hasattr(self, 'lfo2_controls') and 'led' in self.lfo2_controls:
            self.lfo2_controls['led'].set_value(self.lfo2_signal_value)

    def update_level_meter(self):
        """Update level meter display"""
        # Update progress bar (0-100 scale)
        level_percent = int(self.peak_level * 100)
        self.level_meter_bar.setValue(level_percent)

        # Update chunk color based on absolute threshold (not relative)
        if self.peak_level >= 0.85:
            # Red zone (85-100%)
            chunk_color = "#ef4444"
        elif self.peak_level >= 0.70:
            # Yellow zone (70-85%)
            chunk_color = "#f59e0b"
        else:
            # Green zone (0-70%)
            chunk_color = "#10b981"

        self.level_meter_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #1a1a1a;
            }}
            QProgressBar::chunk {{
                background-color: {chunk_color};
                border-radius: 2px;
            }}
        """)

        # Update clip indicator
        if self.clip_detected:
            self.clip_indicator.setStyleSheet("""
                QLabel {
                    background-color: #ef4444;
                    color: white;
                    border: 1px solid #dc2626;
                    border-radius: 3px;
                }
            """)
        else:
            self.clip_indicator.setStyleSheet("""
                QLabel {
                    background-color: #1a1a1a;
                    color: #555555;
                    border: 1px solid #555555;
                    border-radius: 3px;
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

        # Noise section
        noise_label = QLabel("Noise")
        noise_label.setAlignment(Qt.AlignCenter)
        noise_label.setFont(QFont("Arial", 9))
        layout.addWidget(noise_label)

        # Create horizontal layout for noise controls
        noise_controls_layout = QHBoxLayout()
        noise_controls_layout.addStretch(1)

        # Left side: ON/OFF button and type selector stacked vertically
        noise_left_layout = QVBoxLayout()
        noise_left_layout.setSpacing(3)

        # Noise ON/OFF button
        self.noise_button = QPushButton("OFF")
        self.noise_button.setFont(QFont("Arial", 8))
        self.noise_button.setFixedSize(40, 20)
        self.noise_button.setCheckable(True)
        self.noise_button.clicked.connect(self.toggle_noise)
        self.noise_button.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                color: #888888;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
            }
            QPushButton:checked {
                background-color: #fc5b42;
                color: black;
            }
        """)
        noise_left_layout.addWidget(self.noise_button)

        # Noise type selector
        self.noise_type_combo = QComboBox()
        self.noise_type_combo.addItems(["White", "Pink", "Brown"])
        self.noise_type_combo.setFixedWidth(65)
        self.noise_type_combo.setFixedHeight(18)
        self.noise_type_combo.currentTextChanged.connect(lambda t: setattr(self.noise, 'noise_type', t))
        noise_left_layout.addWidget(self.noise_type_combo)

        noise_controls_layout.addLayout(noise_left_layout)
        noise_controls_layout.addSpacing(5)

        # Right side: Gain knob
        self.noise_gain_knob = QDial()
        self.noise_gain_knob.setMinimum(0)
        self.noise_gain_knob.setMaximum(100)
        self.noise_gain_knob.setNotchesVisible(True)
        self.noise_gain_knob.setWrapping(False)
        self.noise_gain_knob.setFixedSize(45, 45)
        self.noise_gain_knob.setStyleSheet("""
            QDial {
                background: transparent;
            }
        """)
        self.noise_gain_knob.setValue(50)
        self.noise_gain_knob.valueChanged.connect(lambda v: setattr(self, 'noise_gain', v / 100.0))
        noise_controls_layout.addWidget(self.noise_gain_knob)

        noise_controls_layout.addStretch(1)
        layout.addLayout(noise_controls_layout)

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

        # Filter mode buttons (LP/BP/HP) in 2x3 grid
        # Row 0: Icons, Row 1: Buttons
        mode_grid = QGridLayout()
        mode_grid.setHorizontalSpacing(8)  # Spacing between columns
        mode_grid.setVerticalSpacing(2)  # No vertical spacing between icon and button rows
        mode_grid.setContentsMargins(0, 0, 0, 0)

        # Center the grid
        mode_layout_wrapper = QHBoxLayout()
        mode_layout_wrapper.addStretch(1)

        # Load icons with high-DPI support
        app = QApplication.instance()
        device_ratio = app.devicePixelRatio() if app else 2.0
        physical_size = int(20 * device_ratio)

        # === LP (Column 0) ===
        # LP Icon (row 0, col 0)
        lp_icon_label = QLabel()
        lp_icon_label.setFixedSize(20, 20)
        lp_icon_label.setAlignment(Qt.AlignCenter)
        lp_icon_label.setContentsMargins(0, 0, 0, 0)
        lp_pixmap = QPixmap("icons/1x/LP.png")
        if not lp_pixmap.isNull():
            scaled_pixmap = lp_pixmap.scaled(physical_size, physical_size,
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaled_pixmap.setDevicePixelRatio(device_ratio)
            lp_icon_label.setPixmap(scaled_pixmap)
        mode_grid.addWidget(lp_icon_label, 0, 0, Qt.AlignBottom | Qt.AlignHCenter)

        # LP Button (row 1, col 0)
        self.filter_lp_button = QPushButton("LP")
        self.filter_lp_button.setFont(QFont("Arial", 8, QFont.Bold))
        self.filter_lp_button.setFixedSize(35, 25)
        self.filter_lp_button.setCheckable(True)
        self.filter_lp_button.setChecked(True)
        self.filter_lp_button.clicked.connect(lambda: self.set_filter_mode("LP"))
        self.filter_lp_button.setStyleSheet("""
            QPushButton {
                background-color: #fc5b42;
                color: black;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #fc6b52;
            }
            QPushButton:!checked {
                background-color: #3c3c3c;
                color: #888888;
            }
        """)
        mode_grid.addWidget(self.filter_lp_button, 1, 0, Qt.AlignTop | Qt.AlignHCenter)

        # === BP (Column 1) ===
        # BP Icon (row 0, col 1)
        bp_icon_label = QLabel()
        bp_icon_label.setFixedSize(20, 20)
        bp_icon_label.setAlignment(Qt.AlignCenter)
        bp_icon_label.setContentsMargins(0, 0, 0, 0)
        bp_pixmap = QPixmap("icons/1x/BP.png")
        if not bp_pixmap.isNull():
            scaled_pixmap = bp_pixmap.scaled(physical_size, physical_size,
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaled_pixmap.setDevicePixelRatio(device_ratio)
            bp_icon_label.setPixmap(scaled_pixmap)
        mode_grid.addWidget(bp_icon_label, 0, 1, Qt.AlignBottom | Qt.AlignHCenter)

        # BP Button (row 1, col 1)
        self.filter_bp_button = QPushButton("BP")
        self.filter_bp_button.setFont(QFont("Arial", 8, QFont.Bold))
        self.filter_bp_button.setFixedSize(35, 25)
        self.filter_bp_button.setCheckable(True)
        self.filter_bp_button.clicked.connect(lambda: self.set_filter_mode("BP"))
        self.filter_bp_button.setStyleSheet("""
            QPushButton {
                background-color: #fc5b42;
                color: black;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #fc6b52;
            }
            QPushButton:!checked {
                background-color: #3c3c3c;
                color: #888888;
            }
        """)
        mode_grid.addWidget(self.filter_bp_button, 1, 1, Qt.AlignTop | Qt.AlignHCenter)

        # === HP (Column 2) ===
        # HP Icon (row 0, col 2)
        hp_icon_label = QLabel()
        hp_icon_label.setFixedSize(20, 20)
        hp_icon_label.setAlignment(Qt.AlignCenter)
        hp_icon_label.setContentsMargins(0, 0, 0, 0)
        hp_pixmap = QPixmap("icons/1x/HP.png")
        if not hp_pixmap.isNull():
            scaled_pixmap = hp_pixmap.scaled(physical_size, physical_size,
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaled_pixmap.setDevicePixelRatio(device_ratio)
            hp_icon_label.setPixmap(scaled_pixmap)
        mode_grid.addWidget(hp_icon_label, 0, 2, Qt.AlignBottom | Qt.AlignHCenter)

        # HP Button (row 1, col 2)
        self.filter_hp_button = QPushButton("HP")
        self.filter_hp_button.setFont(QFont("Arial", 8, QFont.Bold))
        self.filter_hp_button.setFixedSize(35, 25)
        self.filter_hp_button.setCheckable(True)
        self.filter_hp_button.clicked.connect(lambda: self.set_filter_mode("HP"))
        self.filter_hp_button.setStyleSheet("""
            QPushButton {
                background-color: #fc5b42;
                color: black;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #fc6b52;
            }
            QPushButton:!checked {
                background-color: #3c3c3c;
                color: #888888;
            }
        """)
        mode_grid.addWidget(self.filter_hp_button, 1, 2, Qt.AlignTop | Qt.AlignHCenter)

        mode_layout_wrapper.addLayout(mode_grid)
        mode_layout_wrapper.addStretch(1)
        layout.addLayout(mode_layout_wrapper)

        # Knobs layout
        knobs_layout = QHBoxLayout()
        knobs_layout.setSpacing(15)

        # Cutoff (Large: 80x80, use 0-100 range internally, scale to 20-20000 logarithmically)
        cutoff_container = self.create_knob_with_label("Cutoff", 0, 100, 100,
                                                        lambda v: self.update_filter('cutoff', int(20 * (1000 ** (v/100)))), size=80)
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
        """Create dual LFO section with dropdown-based UI"""
        section = QWidget()
        main_layout = QVBoxLayout(section)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Store UI references for preset loading
        self.lfo1_controls = {}
        self.lfo2_controls = {}

        # Title
        title = QLabel("LFO MODULATION")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 12, QFont.Bold))
        main_layout.addWidget(title)

        # LFO 1 Row
        lfo1_row = self.create_lfo_row(1, self.lfo1, 'lfo1_destination', 'lfo1_depth', 'lfo1_mix')
        main_layout.addLayout(lfo1_row)

        # Separator line
        separator = QLabel()
        separator.setFixedHeight(2)
        separator.setStyleSheet("background-color: #666666;")
        main_layout.addWidget(separator)

        # LFO 2 Row
        lfo2_row = self.create_lfo_row(2, self.lfo2, 'lfo2_destination', 'lfo2_depth', 'lfo2_mix')
        main_layout.addLayout(lfo2_row)

        return section

    def update_lfo_mode(self, lfo_num, mode):
        """Update LFO rate mode and show/hide controls"""
        lfo_obj = self.lfo1 if lfo_num == 1 else self.lfo2
        controls = self.lfo1_controls if lfo_num == 1 else self.lfo2_controls

        lfo_obj.rate_mode = mode

        # Show/hide controls based on mode
        if mode == "Free":
            controls['freq'].setVisible(True)
            controls['division'].setVisible(False)
            controls['bpm'].setVisible(False)
        else:  # Sync
            controls['freq'].setVisible(False)
            controls['division'].setVisible(True)
            controls['bpm'].setVisible(True)

    def update_lfo_rate(self, rate_hz):
        """Update LFO rate in Hz (legacy method - no longer used)"""
        # This method is no longer used with the new dual LFO UI
        pass

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
            # Apply logarithmic scaling for cutoff (20-20000 Hz)
            freq = int(20 * (1000 ** (value/100)))
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
        """Update ADSR parameters (single envelope per voice)"""
        if param == 'attack':
            # Convert ms to seconds
            attack_val = value / 1000.0
            self.env.attack = attack_val
            # Update noise envelope (4th oscillator)
            self.noise.envelope.attack = attack_val
            # Update all voice envelopes in real-time
            for voice in self.voice_pool:
                voice.env.attack = attack_val
            # Update UI label
            self.attack_slider_value.setText(f"{int(value)}ms")
        elif param == 'decay':
            # Convert ms to seconds
            decay_val = value / 1000.0
            self.env.decay = decay_val
            # Update noise envelope (4th oscillator)
            self.noise.envelope.decay = decay_val
            # Update all voice envelopes in real-time
            for voice in self.voice_pool:
                voice.env.decay = decay_val
            # Update UI label
            self.decay_slider_value.setText(f"{int(value)}ms")
        elif param == 'sustain':
            # Convert percentage to 0-1
            sustain_val = value / 100.0
            self.env.sustain = sustain_val
            # Update noise envelope (4th oscillator)
            self.noise.envelope.sustain = sustain_val
            # Update all voice envelopes in real-time
            for voice in self.voice_pool:
                voice.env.sustain = sustain_val
            # Update UI label
            self.sustain_slider_value.setText(f"{int(value)}%")
        elif param == 'release':
            # Convert ms to seconds
            release_val = value / 1000.0
            self.env.release = release_val
            # Update noise envelope (4th oscillator)
            self.noise.envelope.release = release_val
            # Update all voice envelopes in real-time
            for voice in self.voice_pool:
                voice.env.release = release_val
            # Update UI label
            self.release_slider_value.setText(f"{int(value)}ms")

    def update_filter(self, param, value):
        """Update filter parameters"""
        if param == 'cutoff':
            # Store the base "dry" cutoff value from knob
            self.filter_cutoff_base = float(value)
            # Only update filter.cutoff if no LFO modulation active
            # (otherwise it will be set in audio callback)
            if self.lfo_to_filter_cutoff == 0:
                self.filter.cutoff = self.filter_cutoff_base
        elif param == 'resonance':
            self.filter.resonance = value / 100.0

    def set_filter_mode(self, mode):
        """Set filter mode (LP/BP/HP) with mutual exclusion"""
        # Update filter object
        self.filter.filter_mode = mode

        # Update button states (mutual exclusion)
        self.filter_lp_button.setChecked(mode == "LP")
        self.filter_bp_button.setChecked(mode == "BP")
        self.filter_hp_button.setChecked(mode == "HP")

    def create_dropdown_container(self, label_text, items, initial_value, callback, width):
        """Create a labeled dropdown combo box

        Args:
            label_text: Text above dropdown
            items: List of strings for dropdown
            initial_value: Initial selection
            callback: Function called on selection change
            width: Fixed width in pixels

        Returns:
            QWidget container with label and combo box
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(label_text)
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Arial", 9))
        layout.addWidget(label)

        combo = QComboBox()
        combo.addItems(items)
        combo.setCurrentText(initial_value)
        combo.setFixedWidth(width)
        combo.currentTextChanged.connect(callback)
        layout.addWidget(combo)

        # Store combo reference in container for later access
        container.combo = combo

        return container

    def create_lfo_row(self, lfo_num, lfo_obj, dest_attr, depth_attr, mix_attr):
        """Create one LFO control row with dropdowns and knobs

        Args:
            lfo_num: 1 or 2
            lfo_obj: self.lfo1 or self.lfo2
            dest_attr: 'lfo1_destination' or 'lfo2_destination'
            depth_attr: 'lfo1_depth' or 'lfo2_depth'
            mix_attr: 'lfo1_mix' or 'lfo2_mix'

        Returns:
            QHBoxLayout with all controls for one LFO
        """
        row = QHBoxLayout()
        row.setSpacing(15)

        # Label
        label = QLabel(f"LFO {lfo_num}:")
        label.setFont(QFont("Arial", 10, QFont.Bold))
        label.setFixedWidth(60)
        row.addWidget(label)

        # LED Indicator
        led_indicator = LFOLEDIndicator(size=30)
        row.addWidget(led_indicator)

        # Shape dropdown
        shape_container = self.create_dropdown_container(
            "Shape", ["Sine", "Triangle", "Square", "Sawtooth", "Random"],
            lfo_obj.waveform, lambda v: setattr(lfo_obj, 'waveform', v), 120
        )
        row.addWidget(shape_container)

        # Destination dropdown
        dest_container = self.create_dropdown_container(
            "Destination", LFO_DESTINATIONS,
            getattr(self, dest_attr), lambda v: setattr(self, dest_attr, v), 150
        )
        row.addWidget(dest_container)

        # Sync Mode dropdown
        sync_container = self.create_dropdown_container(
            "Sync Mode", ["Free", "Sync"],
            lfo_obj.rate_mode, lambda v: self.update_lfo_mode(lfo_num, v), 100
        )
        row.addWidget(sync_container)

        # Division dropdown (visible in Sync mode)
        div_container = self.create_dropdown_container(
            "Division", ["1/16", "1/8", "1/4", "1/2", "1/1", "2/1", "4/1"],
            lfo_obj.sync_division, lambda v: setattr(lfo_obj, 'sync_division', v), 80
        )
        div_container.setVisible(lfo_obj.rate_mode == "Sync")
        row.addWidget(div_container)

        # Depth knob
        depth_knob = self.create_knob_with_label(
            "Depth", 0, 100, int(getattr(self, depth_attr) * 100),
            lambda v: setattr(self, depth_attr, v / 100.0), 60
        )
        row.addWidget(depth_knob)

        # Mix knob
        mix_knob = self.create_knob_with_label(
            "Mix", 0, 100, int(getattr(self, mix_attr) * 100),
            lambda v: setattr(self, mix_attr, v / 100.0), 60
        )
        row.addWidget(mix_knob)

        # Frequency knob (visible in Free mode)
        freq_knob = self.create_knob_with_label(
            "Freq", 1, 200, int(lfo_obj.rate_hz * 10),
            lambda v: setattr(lfo_obj, 'rate_hz', v / 10.0), 60
        )
        freq_knob.setVisible(lfo_obj.rate_mode == "Free")
        row.addWidget(freq_knob)

        # BPM knob (visible in Sync mode, disabled)
        bpm_knob = self.create_knob_with_label(
            "BPM", 40, 240, int(lfo_obj.bpm),
            lambda v: setattr(lfo_obj, 'bpm', float(v)), 60
        )
        bpm_knob.setVisible(lfo_obj.rate_mode == "Sync")
        bpm_dial = bpm_knob.findChild(QDial)
        if bpm_dial:
            bpm_dial.setEnabled(False)
        row.addWidget(bpm_knob)

        # Store references for preset loading and mode switching
        controls = {
            'led': led_indicator,
            'shape': shape_container, 'destination': dest_container,
            'sync_mode': sync_container, 'division': div_container,
            'depth': depth_knob, 'mix': mix_knob,
            'freq': freq_knob, 'bpm': bpm_knob
        }

        if lfo_num == 1:
            self.lfo1_controls = controls
        else:
            self.lfo2_controls = controls

        row.addStretch()
        return row

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
            "version": "1.3",  # Incremented: now saves dual LFO structure
            "oscillators": {
                "osc1": {
                    "enabled": self.osc1_on,  # Save actual runtime on/off state
                    "waveform": self.waveform1,
                    "frequency": self.freq1,
                    "detune": self.detune1,
                    "octave": self.octave1,
                    "pulse_width": self.pulse_width1,
                    "gain": self.gain1
                },
                "osc2": {
                    "enabled": self.osc2_on,  # Save actual runtime on/off state
                    "waveform": self.waveform2,
                    "frequency": self.freq2,
                    "detune": self.detune2,
                    "octave": self.octave2,
                    "pulse_width": self.pulse_width2,
                    "gain": self.gain2
                },
                "osc3": {
                    "enabled": self.osc3_on,  # Save actual runtime on/off state
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
                "attack": self.env.attack,
                "decay": self.env.decay,
                "sustain": self.env.sustain,
                "release": self.env.release
            },
            "filter": {
                "cutoff": self.filter.cutoff,
                "resonance": self.filter.resonance,
                "mode": self.filter.filter_mode
            },
            "lfo1": {
                "waveform": self.lfo1.waveform,
                "rate_mode": self.lfo1.rate_mode,
                "rate_hz": self.lfo1.rate_hz,
                "sync_division": self.lfo1.sync_division,
                "bpm": self.lfo1.bpm,
                "destination": self.lfo1_destination,
                "depth": self.lfo1_depth,
                "mix": self.lfo1_mix
            },
            "lfo2": {
                "waveform": self.lfo2.waveform,
                "rate_mode": self.lfo2.rate_mode,
                "rate_hz": self.lfo2.rate_hz,
                "sync_division": self.lfo2.sync_division,
                "bpm": self.lfo2.bpm,
                "destination": self.lfo2_destination,
                "depth": self.lfo2_depth,
                "mix": self.lfo2_mix
            },
            "noise": {
                "enabled": self.noise_on,
                "type": self.noise.noise_type,
                "gain": self.noise_gain
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

            # Extract preset name from file or JSON
            self.current_preset_name = preset.get("name", os.path.splitext(os.path.basename(file_path))[0])

            # Extract oscillator settings with defaults for forward compatibility
            osc1 = preset.get("oscillators", {}).get("osc1", {})
            osc2 = preset.get("oscillators", {}).get("osc2", {})
            osc3 = preset.get("oscillators", {}).get("osc3", {})

            # Load oscillator on/off states (default True if missing for backward compatibility)
            self.osc1_on = osc1.get("enabled", True)
            self.osc2_on = osc2.get("enabled", True)
            self.osc3_on = osc3.get("enabled", True)
            # Sync enabled states for backward compatibility
            self.osc1_enabled = self.osc1_on
            self.osc2_enabled = self.osc2_on
            self.osc3_enabled = self.osc3_on

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

            # Envelope settings (single envelope applied post-mixer)
            env = preset.get("envelope", {})
            self.env.attack = env.get("attack", 0.01)
            self.noise.envelope.attack = env.get("attack", 0.01)  # Noise as 4th oscillator
            self.env.decay = env.get("decay", 0.1)
            self.noise.envelope.decay = env.get("decay", 0.1)  # Noise as 4th oscillator
            self.env.sustain = env.get("sustain", 0.7)
            self.noise.envelope.sustain = env.get("sustain", 0.7)  # Noise as 4th oscillator
            self.env.release = env.get("release", 0.2)
            self.noise.envelope.release = env.get("release", 0.2)  # Noise as 4th oscillator

            # Filter settings
            filt = preset.get("filter", {})
            self.filter_cutoff_base = filt.get("cutoff", 5000.0)
            self.filter.cutoff = self.filter_cutoff_base  # Set both base and actual
            self.filter.resonance = filt.get("resonance", 0.0)
            self.filter.filter_mode = filt.get("mode", "LP")  # Default to LP for old presets

            # Master settings
            master = preset.get("master", {})
            self.master_volume = master.get("volume", 0.5)
            self.power_on = master.get("power", True)

            # Noise settings (default off if missing for backward compatibility)
            noise = preset.get("noise", {})
            self.noise_on = noise.get("enabled", False)
            self.noise.noise_type = noise.get("type", "White")
            self.noise_gain = noise.get("gain", 0.5)

            # Voice mode (default "Mono" if missing for backward compatibility)
            voice_mode = preset.get("voice_mode", "Mono")

            # Playback mode (default "chromatic" if missing for backward compatibility)
            playback_mode = preset.get("playback_mode", "chromatic")
            self.set_playback_mode(playback_mode)

            # LFO settings - detect old vs new format
            if "lfo1" in preset:
                # New dual LFO format
                self._load_lfo_settings(self.lfo1, preset["lfo1"], 'lfo1_destination', 'lfo1_depth', 'lfo1_mix')
                self._load_lfo_settings(self.lfo2, preset["lfo2"], 'lfo2_destination', 'lfo2_depth', 'lfo2_mix')
            else:
                # Old single LFO format - migrate to LFO1, reset LFO2
                self._migrate_legacy_lfo(preset.get("lfo", {}))
                self._reset_lfo_to_defaults(self.lfo2, 'lfo2_destination', 'lfo2_depth', 'lfo2_mix')

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

            # Load oscillator on/off states (default True if missing for backward compatibility)
            self.osc1_on = osc1.get("enabled", True)
            self.osc2_on = osc2.get("enabled", True)
            self.osc3_on = osc3.get("enabled", True)
            # Sync enabled states for backward compatibility
            self.osc1_enabled = self.osc1_on
            self.osc2_enabled = self.osc2_on
            self.osc3_enabled = self.osc3_on

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

            # Envelope settings (single envelope applied post-mixer)
            env = preset.get("envelope", {})
            self.env.attack = env.get("attack", 0.01)
            self.noise.envelope.attack = env.get("attack", 0.01)  # Noise as 4th oscillator
            self.env.decay = env.get("decay", 0.1)
            self.noise.envelope.decay = env.get("decay", 0.1)  # Noise as 4th oscillator
            self.env.sustain = env.get("sustain", 0.7)
            self.noise.envelope.sustain = env.get("sustain", 0.7)  # Noise as 4th oscillator
            self.env.release = env.get("release", 0.2)
            self.noise.envelope.release = env.get("release", 0.2)  # Noise as 4th oscillator

            # Filter settings
            filt = preset.get("filter", {})
            self.filter_cutoff_base = filt.get("cutoff", 5000.0)
            self.filter.cutoff = self.filter_cutoff_base  # Set both base and actual
            self.filter.resonance = filt.get("resonance", 0.0)
            self.filter.filter_mode = filt.get("mode", "LP")  # Default to LP for old presets

            # Master settings
            master = preset.get("master", {})
            self.master_volume = master.get("volume", 0.5)
            self.power_on = master.get("power", True)

            # Noise settings (default off if missing for backward compatibility)
            noise = preset.get("noise", {})
            self.noise_on = noise.get("enabled", False)
            self.noise.noise_type = noise.get("type", "White")
            self.noise_gain = noise.get("gain", 0.5)

            # Voice mode (default "Mono" if missing for backward compatibility)
            voice_mode = preset.get("voice_mode", "Mono")

            # Playback mode (default "chromatic" if missing for backward compatibility)
            playback_mode = preset.get("playback_mode", "chromatic")
            self.set_playback_mode(playback_mode)

            # LFO settings - detect old vs new format
            if "lfo1" in preset:
                # New dual LFO format
                self._load_lfo_settings(self.lfo1, preset["lfo1"], 'lfo1_destination', 'lfo1_depth', 'lfo1_mix')
                self._load_lfo_settings(self.lfo2, preset["lfo2"], 'lfo2_destination', 'lfo2_depth', 'lfo2_mix')
            else:
                # Old single LFO format - migrate to LFO1, reset LFO2
                self._migrate_legacy_lfo(preset.get("lfo", {}))
                self._reset_lfo_to_defaults(self.lfo2, 'lfo2_destination', 'lfo2_depth', 'lfo2_mix')

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

    def _load_lfo_settings(self, lfo_obj, lfo_data, dest_attr, depth_attr, mix_attr):
        """Load LFO settings from preset data"""
        lfo_obj.waveform = lfo_data.get("waveform", "Sine")
        lfo_obj.rate_mode = lfo_data.get("rate_mode", "Free")
        lfo_obj.rate_hz = lfo_data.get("rate_hz", 2.0)
        lfo_obj.sync_division = lfo_data.get("sync_division", "1/4")
        lfo_obj.bpm = lfo_data.get("bpm", 120.0)
        setattr(self, dest_attr, lfo_data.get("destination", "None"))
        setattr(self, depth_attr, lfo_data.get("depth", 0.0))
        setattr(self, mix_attr, lfo_data.get("mix", 1.0))

    def _reset_lfo_to_defaults(self, lfo_obj, dest_attr, depth_attr, mix_attr):
        """Reset LFO to default values"""
        lfo_obj.waveform = "Sine"
        lfo_obj.rate_mode = "Free"
        lfo_obj.rate_hz = 2.0
        lfo_obj.sync_division = "1/4"
        lfo_obj.bpm = 120.0
        setattr(self, dest_attr, "None")
        setattr(self, depth_attr, 0.0)
        setattr(self, mix_attr, 1.0)

    def _migrate_legacy_lfo(self, lfo_data):
        """Migrate old single-LFO preset format to LFO1"""
        self.lfo1.waveform = lfo_data.get("waveform", "Sine")
        self.lfo1.rate_mode = lfo_data.get("rate_mode", "Free")
        self.lfo1.rate_hz = lfo_data.get("rate_hz", 2.0)
        self.lfo1.sync_division = lfo_data.get("sync_division", "1/4")
        self.lfo1.bpm = lfo_data.get("bpm", 120.0)

        # Map old modulation targets to new simplified destinations
        depth = lfo_data.get("depth", {})
        mix = lfo_data.get("mix", {})

        destination_map = {
            "osc1_pitch": "All OSCs Pitch", "osc2_pitch": "All OSCs Pitch", "osc3_pitch": "All OSCs Pitch",
            "osc1_pw": "OSC1 Pulse Width", "osc2_pw": "OSC2 Pulse Width", "osc3_pw": "OSC3 Pulse Width",
            "filter_cutoff": "Filter Cutoff",
            "osc1_volume": "All OSCs Volume", "osc2_volume": "All OSCs Volume", "osc3_volume": "All OSCs Volume"
        }

        # Search for first active modulation in priority order
        search_order = ["osc1_pitch", "osc2_pitch", "osc3_pitch", "filter_cutoff",
                        "osc1_pw", "osc2_pw", "osc3_pw", "osc1_volume", "osc2_volume", "osc3_volume"]

        # Default to "None"
        self.lfo1_destination = "None"
        self.lfo1_depth = 0.0
        self.lfo1_mix = 1.0

        # Find first active modulation
        for key in search_order:
            if depth.get(key, 0.0) > 0.0:
                self.lfo1_destination = destination_map[key]
                self.lfo1_depth = depth[key]
                self.lfo1_mix = mix.get(key, 1.0)
                break

    def _update_lfo_ui(self, controls, lfo_obj, destination, depth, mix):
        """Update LFO UI controls from values"""
        # Update dropdowns (with signal blocking to prevent callbacks)
        for key in ['shape', 'destination', 'sync_mode', 'division']:
            if key in controls:
                combo = controls[key].combo
                combo.blockSignals(True)
                if key == 'shape':
                    combo.setCurrentText(lfo_obj.waveform)
                elif key == 'destination':
                    combo.setCurrentText(destination)
                elif key == 'sync_mode':
                    combo.setCurrentText(lfo_obj.rate_mode)
                elif key == 'division':
                    combo.setCurrentText(lfo_obj.sync_division)
                combo.blockSignals(False)

        # Update knobs
        self._update_knob_value(controls['depth'], depth * 100, f"{int(depth * 100)}%")
        self._update_knob_value(controls['mix'], mix * 100, f"{int(mix * 100)}%")
        self._update_knob_value(controls['freq'], lfo_obj.rate_hz * 10, f"{lfo_obj.rate_hz:.1f} Hz")
        self._update_knob_value(controls['bpm'], lfo_obj.bpm, f"{int(lfo_obj.bpm)}")

        # Show/hide controls based on mode
        is_sync = (lfo_obj.rate_mode == "Sync")
        controls['freq'].setVisible(not is_sync)
        controls['division'].setVisible(is_sync)
        controls['bpm'].setVisible(is_sync)

    def _update_knob_value(self, knob_container, value, display_text):
        """Update knob value and label"""
        knob = knob_container.findChild(QDial)
        label = knob_container.findChild(QLabel, "value_label")
        if knob:
            knob.blockSignals(True)
            knob.setValue(int(value))
            knob.blockSignals(False)
        if label:
            label.setText(display_text)

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
        self.attack_slider.setValue(round(self.env.attack * 50))  # seconds * 1000 / 20 = * 50 (use round, not int)
        self.decay_slider.setValue(round(self.env.decay * 50))    # seconds * 1000 / 20 = * 50 (use round, not int)
        self.sustain_slider.setValue(round(self.env.sustain * 100))  # 0-1 to 0-100 (use round, not int)
        self.release_slider.setValue(round(self.env.release * 20))  # seconds * 1000 / 50 = * 20 (use round, not int)
        self.attack_slider.blockSignals(False)
        self.decay_slider.blockSignals(False)
        self.sustain_slider.blockSignals(False)
        self.release_slider.blockSignals(False)

        # Update ADSR labels (convert seconds to milliseconds for display)
        self.attack_slider_value.setText(f"{int(self.env.attack * 1000)}ms")
        self.decay_slider_value.setText(f"{int(self.env.decay * 1000)}ms")
        self.sustain_slider_value.setText(f"{int(self.env.sustain * 100)}%")
        self.release_slider_value.setText(f"{int(self.env.release * 1000)}ms")

        # Update master volume
        self.master_volume_knob.blockSignals(True)
        self.master_volume_knob.setValue(int(self.master_volume * 100))
        self.master_volume_knob.blockSignals(False)

        # Update noise controls
        self.noise_button.setChecked(self.noise_on)
        self.noise_button.setText("ON" if self.noise_on else "OFF")
        self.noise_type_combo.setCurrentText(self.noise.noise_type)
        self.noise_gain_knob.blockSignals(True)
        self.noise_gain_knob.setValue(int(self.noise_gain * 100))
        self.noise_gain_knob.blockSignals(False)

        # Update filter knobs
        self.cutoff_knob.blockSignals(True)
        self.resonance_knob.blockSignals(True)
        # Cutoff: reverse the logarithmic formula (cutoff_hz = 20 * (1000 ** (v/100)))
        # v = 100 * log(cutoff_hz / 20) / log(1000)
        if self.filter.cutoff > 0:
            cutoff_knob_value = int(100 * np.log(self.filter.cutoff / 20) / np.log(1000))
            self.cutoff_knob.setValue(cutoff_knob_value)
        # Resonance: direct percentage
        self.resonance_knob.setValue(int(self.filter.resonance * 100))
        self.cutoff_knob.blockSignals(False)
        self.resonance_knob.blockSignals(False)

        # Update filter mode buttons
        self.filter_lp_button.setChecked(self.filter.filter_mode == "LP")
        self.filter_bp_button.setChecked(self.filter.filter_mode == "BP")
        self.filter_hp_button.setChecked(self.filter.filter_mode == "HP")

        # Update dual LFO UI
        if hasattr(self, 'lfo1_controls'):
            self._update_lfo_ui(self.lfo1_controls, self.lfo1,
                               self.lfo1_destination, self.lfo1_depth, self.lfo1_mix)
        if hasattr(self, 'lfo2_controls'):
            self._update_lfo_ui(self.lfo2_controls, self.lfo2,
                               self.lfo2_destination, self.lfo2_depth, self.lfo2_mix)

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

        # Copy envelope settings to all voices (single envelope)
        for voice in self.voice_pool:
            voice.env.attack = self.env.attack
            voice.env.decay = self.env.decay
            voice.env.sustain = self.env.sustain
            voice.env.release = self.env.release

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
        release_voices = [v for v in self.voice_pool if v.env.phase == 'release']
        if release_voices:
            # Return oldest release voice
            return max(release_voices, key=lambda v: v.age)

        # If no release voices, try to find idle voices
        idle_voices = [v for v in self.voice_pool if v.is_free()]
        if idle_voices:
            return idle_voices[0]

        # Last resort: steal oldest active voice
        return max(self.voice_pool, key=lambda v: v.age)

    def poly_blep_vectorized(self, t, dt):
        """Vectorized PolyBLEP (Polynomial Band-Limited stEP) residual for anti-aliasing

        Reduces aliasing in non-bandlimited waveforms by smoothing discontinuities
        using polynomial interpolation. Industry standard technique for high-quality
        synthesis.

        Args:
            t: Phase position array normalized to 0-1
            dt: Phase increment per sample normalized to 0-1

        Returns:
            Residual array to subtract from naive waveform
        """
        # Initialize output array
        residual = np.zeros_like(t)

        # Near rising edge (phase wrapping from 1 to 0)
        mask1 = t < dt
        t1 = t[mask1] / dt
        residual[mask1] = t1 + t1 - t1 * t1 - 1.0

        # Near falling edge (approaching phase = 1)
        mask2 = t > (1.0 - dt)
        t2 = (t[mask2] - 1.0) / dt
        residual[mask2] = t2 * t2 + t2 + t2 + 1.0

        return residual

    def generate_waveform(self, waveform_type, phase, phase_increment, frames, pulse_width=0.5):
        """Generate a waveform based on type with PolyBLEP anti-aliasing

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
            # Normalize phase to 0-1 for PolyBLEP
            normalized_phases = (phases % (2 * np.pi)) / (2 * np.pi)
            dt = phase_increment / (2 * np.pi)  # Normalized phase increment

            # Generate naive sawtooth
            naive_saw = 2 * normalized_phases - 1

            # Apply vectorized PolyBLEP correction at discontinuities
            blep_correction = self.poly_blep_vectorized(normalized_phases, dt)
            output = naive_saw - blep_correction

            return output

        elif waveform_type == "Square":
            # Normalize phase to 0-1 for PolyBLEP
            normalized_phases = (phases % (2 * np.pi)) / (2 * np.pi)
            dt = phase_increment / (2 * np.pi)  # Normalized phase increment

            # Generate naive square wave with PWM
            naive_square = np.where(normalized_phases < pulse_width, 1.0, -1.0)

            # Apply vectorized PolyBLEP correction at both rising and falling edges
            # Rising edge at t=0
            blep_rising = self.poly_blep_vectorized(normalized_phases, dt)
            # Falling edge at t=pulse_width (wrap phase for edge detection)
            phase_from_falling = (normalized_phases - pulse_width) % 1.0
            blep_falling = self.poly_blep_vectorized(phase_from_falling, dt)

            output = naive_square - blep_rising + blep_falling

            return output
        else:
            return np.sin(phases)

    def apply_lfo_modulation(self, lfo_num, lfo_signal, lfo_mean, destination, depth, mix):
        """Apply LFO modulation to specified destination(s)

        Args:
            lfo_num: 1 or 2 (for tracking which LFO)
            lfo_signal: Full LFO waveform array (-1 to 1)
            lfo_mean: Cached mean of lfo_signal
            destination: String from LFO_DESTINATIONS
            depth: Modulation depth (0-1)
            mix: Dry/wet mix (0-1)

        Returns:
            dict: {
                'pitch_mod': {osc_num: multiplier},
                'pw_mod': {osc_num: pw_offset},
                'vol_mod': {osc_num: gain_array},
                'filter_mod': cutoff_hz or None
            }
        """
        result = {
            'pitch_mod': {},  # {osc_num: multiplier}
            'pw_mod': {},     # {osc_num: pw_offset}
            'vol_mod': {},    # {osc_num: gain_array}
            'filter_mod': None  # cutoff_hz or None
        }

        if destination == "None" or depth == 0.0:
            return result

        if destination == "All OSCs Pitch":
            # Apply vibrato to all 3 oscillators
            pitch_scalar = 1.0 + (lfo_mean * LFO_PITCH_MOD_DEPTH * depth * mix)
            result['pitch_mod'] = {1: pitch_scalar, 2: pitch_scalar, 3: pitch_scalar}

        elif destination == "Filter Cutoff":
            # Apply filter sweep
            cutoff_mod = 2.0 ** (lfo_mean * LFO_FILTER_MOD_OCTAVES * depth)
            wet_cutoff = self.filter_cutoff_base * cutoff_mod
            dry_cutoff = self.filter_cutoff_base
            result['filter_mod'] = np.clip(
                dry_cutoff * (1.0 - mix) + wet_cutoff * mix,
                FILTER_CUTOFF_MIN, FILTER_CUTOFF_MAX
            )

        elif destination == "All OSCs Volume":
            # Apply tremolo to all 3 oscillators (uses full signal array)
            vol_mod = 1.0 + (lfo_signal * depth * mix)
            vol_array = np.clip(vol_mod, VOL_MOD_MIN, VOL_MOD_MAX)
            result['vol_mod'] = {1: vol_array, 2: vol_array, 3: vol_array}

        elif destination in ["OSC1 Pulse Width", "OSC2 Pulse Width", "OSC3 Pulse Width"]:
            # Apply PWM to specific oscillator
            osc_num = int(destination[3])  # Extract "1", "2", or "3"
            pw_offset = lfo_mean * LFO_PW_MOD_DEPTH * depth * mix
            result['pw_mod'] = {osc_num: pw_offset}

        return result

    def process_oscillator(self, osc_num, voice, base_freq_detuned, lfo1_mods, lfo2_mods, frames):
        """Process a single oscillator and return its output signal

        This helper eliminates code duplication across the three oscillators by using
        dynamic attribute access to get oscillator-specific parameters.

        Args:
            osc_num: Oscillator number (1, 2, or 3)
            voice: Voice object
            base_freq_detuned: Base frequency with unison detune applied
            lfo1_mods: Modulation dict from LFO1 (apply_lfo_modulation result)
            lfo2_mods: Modulation dict from LFO2 (apply_lfo_modulation result)
            frames: Number of frames to generate

        Returns:
            numpy array of oscillator output (or zeros if oscillator is off)
        """
        # Check if oscillator is enabled
        osc_on = getattr(self, f'osc{osc_num}_on')
        if not osc_on:
            return np.zeros(frames)

        # Get oscillator-specific parameters using dynamic attribute access
        drone_freq = getattr(self, f'freq{osc_num}')
        detune = getattr(self, f'detune{osc_num}')
        octave = getattr(self, f'octave{osc_num}')
        waveform = getattr(self, f'waveform{osc_num}')
        pulse_width = getattr(self, f'pulse_width{osc_num}')
        gain = getattr(self, f'gain{osc_num}')

        # Get voice phase
        phase_attr = f'phase{osc_num}'
        phase = getattr(voice, phase_attr)

        # Determine frequency based on playback mode
        if self.playback_mode == 'drone':
            freq = drone_freq
        else:
            # Apply oscillator-specific detune and octave
            freq = self.apply_octave(self.apply_detune(base_freq_detuned, detune), octave)

        # Combine pitch modulations from both LFOs (multiplicative)
        freq_mod_scalar = 1.0
        if osc_num in lfo1_mods['pitch_mod']:
            freq_mod_scalar *= lfo1_mods['pitch_mod'][osc_num]
        if osc_num in lfo2_mods['pitch_mod']:
            freq_mod_scalar *= lfo2_mods['pitch_mod'][osc_num]
        modulated_freq = freq * freq_mod_scalar
        phase_increment = 2 * np.pi * modulated_freq / self.sample_rate

        # Combine PW modulations from both LFOs (additive)
        pw_mod = 0.0
        if osc_num in lfo1_mods['pw_mod']:
            pw_mod += lfo1_mods['pw_mod'][osc_num]
        if osc_num in lfo2_mods['pw_mod']:
            pw_mod += lfo2_mods['pw_mod'][osc_num]
        modulated_pw = np.clip(pulse_width + pw_mod, PW_MIN, PW_MAX)

        # Generate waveform using voice's phase
        wave = self.generate_waveform(waveform, phase, phase_increment, frames, modulated_pw)

        # Combine volume modulations from both LFOs (multiplicative)
        modulated_gain = gain
        if osc_num in lfo1_mods['vol_mod']:
            modulated_gain = modulated_gain * lfo1_mods['vol_mod'][osc_num]
        if osc_num in lfo2_mods['vol_mod']:
            modulated_gain = modulated_gain * lfo2_mods['vol_mod'][osc_num]

        # Update voice phase
        setattr(voice, phase_attr, (phase + frames * phase_increment) % (2 * np.pi))

        return modulated_gain * wave

    def audio_callback(self, outdata, frames, time, status):
        """Generate and mix all active voices with envelopes and filter"""
        # Don't print in audio callback - causes buffer underruns

        # If power is off, output silence
        if not self.power_on:
            outdata[:, 0] = 0
            outdata[:, 1] = 0
            return

        # Generate both LFO signals (-1 to 1)
        lfo1_signal = self.lfo1.process(frames)
        lfo2_signal = self.lfo2.process(frames)

        # Cache LFO mean values (used for pitch, PW, and filter modulation)
        # This avoids recalculating np.mean() multiple times per callback
        lfo1_mean = np.mean(lfo1_signal)
        lfo2_mean = np.mean(lfo2_signal)

        # Store LFO values for LED display updates
        self.lfo1_signal_value = lfo1_mean
        self.lfo2_signal_value = lfo2_mean

        # Calculate modulations for both LFOs
        lfo1_mods = self.apply_lfo_modulation(
            1, lfo1_signal, lfo1_mean,
            self.lfo1_destination, self.lfo1_depth, self.lfo1_mix
        )
        lfo2_mods = self.apply_lfo_modulation(
            2, lfo2_signal, lfo2_mean,
            self.lfo2_destination, self.lfo2_depth, self.lfo2_mix
        )

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

            # Generate each oscillator for this voice (WITHOUT envelope - applied post-mixer)
            # Use helper function to eliminate code duplication
            voice_mix = np.zeros(frames)
            voice_mix += self.process_oscillator(1, voice, base_freq_detuned, lfo1_mods, lfo2_mods, frames)
            voice_mix += self.process_oscillator(2, voice, base_freq_detuned, lfo1_mods, lfo2_mods, frames)
            voice_mix += self.process_oscillator(3, voice, base_freq_detuned, lfo1_mods, lfo2_mods, frames)

            # Apply SINGLE envelope to mixed oscillators (post-mixer efficiency!)
            env = voice.env.process(frames)
            voice_mix *= env

            # Add this voice's output to the main mix
            mixed += voice_mix

            # Clean up voice if envelope is idle (release finished)
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
        # Use sqrt normalization for both poly and unison modes for better perceived loudness
        # With 8 voices: 1/sqrt(8) ≈ 0.35 per voice → ~2.8x total (thick but controlled)
        if active_count > 0:
            normalization = 1.0 / np.sqrt(active_count)
            mixed *= normalization

        # Mix in noise if enabled (with envelope like a 4th oscillator)
        if self.noise_on:
            noise_signal = self.noise.generate(frames)
            noise_env = self.noise.envelope.process(frames)
            mixed += noise_signal * noise_env * self.noise_gain

        # Apply LFO modulation to filter cutoff (combine both LFOs if both target filter)
        if lfo1_mods['filter_mod'] is not None and lfo2_mods['filter_mod'] is not None:
            # Both LFOs modulating filter - use average
            self.filter.cutoff = (lfo1_mods['filter_mod'] + lfo2_mods['filter_mod']) / 2.0
        elif lfo1_mods['filter_mod'] is not None:
            # Only LFO1 modulating filter
            self.filter.cutoff = lfo1_mods['filter_mod']
        elif lfo2_mods['filter_mod'] is not None:
            # Only LFO2 modulating filter
            self.filter.cutoff = lfo2_mods['filter_mod']
        else:
            # No modulation: use base cutoff
            self.filter.cutoff = self.filter_cutoff_base

        filtered = self.filter.process(mixed)

        # Apply master volume
        output_signal = self.master_volume * filtered

        # Apply DC blocking filter to remove any DC offset (vectorized)
        # Uses a simple first-order high-pass filter with ~5Hz cutoff
        # Transfer function: H(z) = (1 - z^-1) / (1 - 0.995*z^-1)
        # y[n] = x[n] - x[n-1] + R * y[n-1], where R = 0.995
        b_dc = [1.0, -1.0]  # Numerator coefficients
        a_dc = [1.0, -0.995]  # Denominator coefficients

        # Initialize DC blocker state on first use
        if not hasattr(self, 'dc_blocker_zi'):
            self.dc_blocker_zi = scipy_signal.lfilter_zi(b_dc, a_dc) * 0.0

        # Apply vectorized DC blocking filter
        dc_blocked, self.dc_blocker_zi = scipy_signal.lfilter(b_dc, a_dc, output_signal, zi=self.dc_blocker_zi)

        # Final safety clamp to prevent any clipping artifacts (soft limit at ±1.0)
        # This protects against edge cases where filter resonance or noise might push levels too high
        final_output = np.clip(dc_blocked, -1.0, 1.0)

        # Track peak level for level meter (before clipping)
        current_peak = np.max(np.abs(dc_blocked))
        self.peak_level = max(self.peak_level * 0.95, current_peak)  # Fast attack, slow decay

        # Detect clipping (signal reached or exceeded ±1.0 before clipping)
        if current_peak >= 1.0:
            self.clip_detected = True
            self.clip_hold_counter = int(self.sample_rate * 1.0)  # Hold for 1 second
        elif self.clip_hold_counter > 0:
            self.clip_hold_counter -= frames
            if self.clip_hold_counter <= 0:
                self.clip_detected = False

        # Send audio to spectrum analyzer for visualization
        if hasattr(self, 'spectrum_analyzer_window'):
            self.spectrum_analyzer_window.process_audio(final_output)

        # Output stereo
        outdata[:, 0] = final_output
        outdata[:, 1] = final_output

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

    def toggle_noise(self):
        """Toggle noise generator on/off"""
        self.noise_on = not self.noise_on

        if self.noise_on:
            self.noise_button.setText("ON")
            # In drone mode, trigger the noise envelope like an oscillator
            if self.playback_mode == 'drone':
                self.noise.trigger()
        else:
            self.noise_button.setText("OFF")
            # In drone mode, release the noise envelope
            if self.playback_mode == 'drone':
                self.noise.release()

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
                voice.env.force_reset()
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

        # Trigger voices with appropriate detune offsets
        # Always use phase_offset=0 for click-free triggering (zero crossing start)
        allocated_voices = []
        for i, voice in enumerate(available_voices[:voices_needed]):
            unison_detune = detune_pattern[i] if i < len(detune_pattern) else 0.0
            # Always start at zero phase for click-free operation
            voice.trigger(note, velocity, unison_detune, phase_offset=0.0)
            allocated_voices.append(voice)

        # Track active voices for this note
        self.active_voices[note] = allocated_voices

        # Update legacy freq variables for backward compatibility
        # (used by UI display and some controls)
        base_freq = self.midi_note_to_freq(note)
        self.freq1 = self.apply_octave(self.apply_detune(base_freq, self.detune1), self.octave1)
        self.freq2 = self.apply_octave(self.apply_detune(base_freq, self.detune2), self.octave2)
        self.freq3 = self.apply_octave(self.apply_detune(base_freq, self.detune3), self.octave3)

        # Trigger noise envelope if noise is enabled (like a 4th oscillator)
        # Only in chromatic mode - in drone mode, noise is triggered by its own button
        if self.noise_on and self.playback_mode == 'chromatic':
            self.noise.trigger()

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

        # Release noise envelope if noise is enabled (like a 4th oscillator)
        # Only in chromatic mode - in drone mode, noise is triggered by its own button
        if self.noise_on and self.playback_mode == 'chromatic':
            self.noise.release()

    def handle_midi_bpm_change(self, bpm):
        """Handle BPM changes from MIDI clock"""
        # Update both LFO BPMs
        self.lfo1.bpm = bpm
        self.lfo2.bpm = bpm

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
