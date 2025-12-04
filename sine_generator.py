#!/usr/bin/env python3
"""
Simple Sine Wave Generator with GUI
Features:
- Frequency control knob (20 Hz to 20,000 Hz)
- On/Off button to start/stop the tone
- Real-time frequency adjustment
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
import threading
import math


class CircularKnob(tk.Canvas):
    """Custom circular knob widget"""
    def __init__(self, parent, size=120, min_value=0, max_value=1000, initial_value=500, command=None):
        super().__init__(parent, width=size, height=size, bg='white', highlightthickness=0)

        self.size = size
        self.center = size / 2
        self.radius = size / 2 - 10
        self.min_value = min_value
        self.max_value = max_value
        self.value = initial_value
        self.command = command

        # Angle range: -140 to +140 degrees (280 degrees total)
        self.min_angle = -140
        self.max_angle = 140

        self.dragging = False

        # Draw the knob
        self.draw_knob()

        # Bind mouse events
        self.bind("<Button-1>", self.on_mouse_down)
        self.bind("<B1-Motion>", self.on_mouse_drag)
        self.bind("<ButtonRelease-1>", self.on_mouse_up)

    def draw_knob(self):
        """Draw the knob graphics"""
        self.delete("all")

        # Draw outer circle (knob body)
        self.create_oval(
            self.center - self.radius,
            self.center - self.radius,
            self.center + self.radius,
            self.center + self.radius,
            fill="#2c3e50",
            outline="#34495e",
            width=2
        )

        # Draw tick marks around the knob
        tick_radius_outer = self.radius - 5
        tick_radius_inner = self.radius - 15

        for i in range(0, 11):
            angle_deg = self.min_angle + (self.max_angle - self.min_angle) * i / 10
            angle_rad = math.radians(angle_deg)

            x1 = self.center + tick_radius_inner * math.sin(angle_rad)
            y1 = self.center - tick_radius_inner * math.cos(angle_rad)
            x2 = self.center + tick_radius_outer * math.sin(angle_rad)
            y2 = self.center - tick_radius_outer * math.cos(angle_rad)

            self.create_line(x1, y1, x2, y2, fill="#7f8c8d", width=2)

        # Draw indicator line
        current_angle = self.value_to_angle(self.value)
        angle_rad = math.radians(current_angle)

        indicator_length = self.radius - 20
        x = self.center + indicator_length * math.sin(angle_rad)
        y = self.center - indicator_length * math.cos(angle_rad)

        self.create_line(
            self.center, self.center, x, y,
            fill="#e74c3c",
            width=4,
            capstyle=tk.ROUND
        )

        # Draw center dot
        dot_radius = 8
        self.create_oval(
            self.center - dot_radius,
            self.center - dot_radius,
            self.center + dot_radius,
            self.center + dot_radius,
            fill="#e74c3c",
            outline=""
        )

    def value_to_angle(self, value):
        """Convert value to angle in degrees"""
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        return self.min_angle + normalized * (self.max_angle - self.min_angle)

    def angle_to_value(self, angle_deg):
        """Convert angle to value"""
        normalized = (angle_deg - self.min_angle) / (self.max_angle - self.min_angle)
        normalized = max(0, min(1, normalized))
        return self.min_value + normalized * (self.max_value - self.min_value)

    def mouse_to_angle(self, x, y):
        """Convert mouse position to angle"""
        dx = x - self.center
        dy = self.center - y
        angle_rad = math.atan2(dx, dy)
        angle_deg = math.degrees(angle_rad)

        # Clamp to min/max angles
        angle_deg = max(self.min_angle, min(self.max_angle, angle_deg))
        return angle_deg

    def on_mouse_down(self, event):
        """Handle mouse button press"""
        self.dragging = True
        self.update_from_mouse(event.x, event.y)

    def on_mouse_drag(self, event):
        """Handle mouse drag"""
        if self.dragging:
            self.update_from_mouse(event.x, event.y)

    def on_mouse_up(self, event):
        """Handle mouse button release"""
        self.dragging = False

    def update_from_mouse(self, x, y):
        """Update knob value from mouse position"""
        angle = self.mouse_to_angle(x, y)
        new_value = self.angle_to_value(angle)

        if new_value != self.value:
            self.value = new_value
            self.draw_knob()
            if self.command:
                self.command(self.value)

    def set(self, value):
        """Set the knob value programmatically"""
        self.value = max(self.min_value, min(self.max_value, value))
        self.draw_knob()


class SineWaveGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Sine Wave Generator")
        self.root.geometry("350x400")
        self.root.resizable(False, False)

        # Audio parameters
        self.sample_rate = 44100
        self.frequency = 440.0
        self.is_playing = False
        self.stream = None
        self.phase = 0

        # Logarithmic scale parameters
        self.min_freq = 20.0
        self.max_freq = 20000.0
        self.min_log = np.log10(self.min_freq)
        self.max_log = np.log10(self.max_freq)

        # Create GUI
        self.create_widgets()

    def create_widgets(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="Oscilator 1\n(Sine)",
            font=("Arial", 20, "bold")
        )
        title_label.pack(pady=20)

        # Circular frequency knob
        initial_position = 1000 * (np.log10(440) - self.min_log) / (self.max_log - self.min_log)

        self.freq_knob = CircularKnob(
            self.root,
            size=150,
            min_value=0,
            max_value=1000,
            initial_value=initial_position,
            command=self.update_frequency
        )
        self.freq_knob.pack(pady=20)

        # Frequency display
        self.freq_label = tk.Label(
            self.root,
            text=f"{self.frequency:.1f} Hz",
            font=("Arial", 20, "bold")
        )
        self.freq_label.pack(pady=10)

        # Frequency range labels
        range_frame = tk.Frame(self.root)
        range_frame.pack(pady=5)

        min_label = tk.Label(range_frame, text="20 Hz", font=("Arial", 10))
        min_label.pack(side=tk.LEFT, padx=40)

        max_label = tk.Label(range_frame, text="20k Hz", font=("Arial", 10))
        max_label.pack(side=tk.RIGHT, padx=40)

        # On/Off button
        self.toggle_button = tk.Button(
            self.root,
            text="ON",
            font=("Arial", 16, "bold"),
            bg="#4CAF50",
            fg="white",
            width=15,
            height=2,
            command=self.toggle_sound
        )
        self.toggle_button.pack(pady=20)

    def update_frequency(self, value):
        """Update frequency from slider using logarithmic scale"""
        # Convert slider value (0-1000) to logarithmic frequency
        slider_position = float(value) / 1000.0  # Normalize to 0-1
        log_freq = self.min_log + slider_position * (self.max_log - self.min_log)
        self.frequency = 10 ** log_freq
        self.freq_label.config(text=f"{self.frequency:.1f} Hz")

    def audio_callback(self, outdata, frames, time, status):
        """Generate sine wave audio with phase continuity"""
        if status:
            print(status)

        # Generate array of phase increments for each sample
        phase_increment = 2 * np.pi * self.frequency / self.sample_rate
        phases = self.phase + np.arange(frames) * phase_increment

        # Generate sine wave
        sine_wave = 0.3 * np.sin(phases)

        # Update phase and wrap to prevent overflow
        self.phase = (self.phase + frames * phase_increment) % (2 * np.pi)

        # Output stereo
        outdata[:, 0] = sine_wave
        outdata[:, 1] = sine_wave

    def toggle_sound(self):
        """Toggle sound on/off"""
        if self.is_playing:
            self.stop_sound()
        else:
            self.start_sound()

    def start_sound(self):
        """Start playing sine wave"""
        try:
            self.phase = 0
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                callback=self.audio_callback
            )
            self.stream.start()
            self.is_playing = True
            self.toggle_button.config(text="STOP", bg="#f44336")
        except Exception as e:
            print(f"Error starting audio: {e}")

    def stop_sound(self):
        """Stop playing sine wave"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False
        self.toggle_button.config(text="START", bg="#4CAF50")

    def cleanup(self):
        """Clean up resources"""
        self.stop_sound()


def main():
    root = tk.Tk()
    app = SineWaveGenerator(root)

    # Handle window close
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))

    root.mainloop()


if __name__ == "__main__":
    main()
