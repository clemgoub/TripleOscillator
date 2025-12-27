# Triple Oscillator VST3/AU Port - Implementation Plan

## Overview

This document outlines the complete plan for porting the Python Triple Oscillator synthesizer to a professional VST3/AU plugin using C++ and the JUCE framework.

**Python Reference Repository:** https://github.com/clemgoub/TripleOscillator
**Target:** VST3/AU/Standalone plugin compatible with all major DAWs
**Technology Stack:** C++17, JUCE Framework 7.x, CMake build system

---

## Why JUCE (Not VENOM)?

### ✅ JUCE Advantages:
- **Industry Standard:** Used by Serum, FabFilter, Native Instruments, etc.
- **Performance:** Native C++ for real-time audio (no Python overhead)
- **Multi-Format:** VST3/AU/AAX/Standalone from single codebase
- **Mature Ecosystem:** 15+ years, extensive docs, large community
- **Hardware Path:** C++ skills transfer directly to embedded/microcontroller synth development
- **Your Python Synth = Perfect Spec:** Well-documented reference implementation

### ❌ VENOM Limitations:
- Immature project (limited adoption, unknown stability)
- Python runtime overhead in real-time audio
- Limited DAW compatibility
- Not suitable for future hardware porting

**Decision:** C++/JUCE for professional quality and learning path

---

## Repository Structure

```
TripleOscillatorVST/
├── README.md                          # Overview, build instructions, credits
├── CMakeLists.txt                     # CMake build configuration
├── .gitignore                         # C++ build artifacts, IDE files
│
├── Docs/
│   ├── IMPLEMENTATION_PLAN.md         # This file (phase-by-phase roadmap)
│   ├── ARCHITECTURE.md                # System design, DSP flow diagrams
│   ├── PYTHON_REFERENCE.md            # Python code line number mappings
│   ├── PHASE_CHECKLISTS.md            # Validation criteria per phase
│   └── BUILD_GUIDE.md                 # Platform-specific build instructions
│
├── Source/
│   ├── PluginProcessor.h/cpp          # Main audio processing entry point
│   ├── PluginEditor.h/cpp             # UI (Phase 8)
│   │
│   ├── DSP/                           # Audio engine components
│   │   ├── Oscillator.h/cpp           # Waveform generation + PolyBLEP
│   │   ├── Envelope.h/cpp             # ADSR envelope generator
│   │   ├── Voice.h/cpp                # Single voice (3 osc + env)
│   │   ├── VoiceManager.h/cpp         # Voice allocation, stealing, modes
│   │   ├── MoogFilter.h/cpp           # 4-pole Moog ladder filter
│   │   ├── LFO.h/cpp                  # LFO generator (5 waveforms)
│   │   ├── NoiseGenerator.h/cpp       # White/Pink/Brown noise
│   │   └── AudioUtils.h/cpp           # PolyBLEP, DC blocker, utilities
│   │
│   ├── Parameters/                    # Parameter management
│   │   └── SynthParameters.h/cpp      # All synth parameters, ranges
│   │
│   └── UI/                            # GUI components (Phase 8)
│       ├── Components/
│       │   ├── OscillatorPanel.h/cpp
│       │   ├── FilterPanel.h/cpp
│       │   ├── LFOPanel.h/cpp
│       │   └── EnvelopePanel.h/cpp
│       └── LookAndFeel/
│           └── CustomLookAndFeel.h/cpp
│
├── Resources/                         # Plugin assets
│   ├── Presets/
│   │   ├── Init.vstpreset
│   │   └── Factory/
│   └── Images/
│       └── (filter icons, backgrounds)
│
└── Tests/                             # Unit tests (optional)
    └── DSP/
        ├── OscillatorTests.cpp
        └── EnvelopeTests.cpp
```

---

## Progressive Implementation Phases

### Phase 0: Project Setup & Verification ⏱️ 1-2 days

**Goal:** JUCE project builds successfully, outputs test sine wave

**Tasks:**
1. Install JUCE Framework
   - Download from https://juce.com/get-juce/download
   - Extract to `~/JUCE` or `/opt/JUCE`

2. Create new Audio Plugin project
   - Option A: Use Projucer GUI (easier for beginners)
   - Option B: Use CMake template (better for version control)

3. Configure project settings:
   - Plugin name: "TripleOscillator"
   - Manufacturer: Your name
   - Plugin code: "Trio" (4-char unique ID)
   - Formats: VST3, AU, Standalone

4. Test build on your platform (macOS/Windows/Linux)

5. Implement simple 440Hz sine wave test in `PluginProcessor::processBlock()`

6. Create documentation files in `Docs/`

**Validation Checklist:**
- [ ] Project builds without errors
- [ ] Standalone app launches
- [ ] VST3 loads in DAW (Reaper/Ableton/Logic)
- [ ] Plays 440Hz sine wave when MIDI note triggered
- [ ] No audio glitches or dropouts

**Python Reference:** N/A (new C++ project setup)

**Deliverables:**
- [ ] Working JUCE project
- [ ] README.md with build instructions
- [ ] Docs/ARCHITECTURE.md with initial design
- [ ] Git repo initialized

---

### Phase 1: Single Oscillator Engine ⏱️ 3-5 days

**Goal:** One voice, one oscillator, three waveforms (Sine/Saw/Square), MIDI input

**Implement:**

**`Source/DSP/Oscillator.h/cpp`**
```cpp
class Oscillator {
public:
    enum Waveform { Sine, Sawtooth, Square };

    void setSampleRate(double sampleRate);
    void setFrequency(float frequency);
    void setWaveform(Waveform waveform);
    void setPulseWidth(float pw);  // 0.01 - 0.99

    float processSample();         // Generate one sample
    void reset();                  // Reset phase to 0

private:
    double phase = 0.0;
    double phaseIncrement = 0.0;
    double sampleRate = 44100.0;
    float frequency = 440.0f;
    float pulseWidth = 0.5f;
    Waveform waveform = Sine;

    float generateSine();
    float generateSawtooth();      // With PolyBLEP
    float generateSquare();        // With PolyBLEP
    float polyBLEP(double t);      // Anti-aliasing
};
```

**`Source/DSP/AudioUtils.h/cpp`**
- `polyBLEP()` function (port from Python lines 640-680)
- MIDI note to frequency conversion: `440.0 * pow(2.0, (note - 69) / 12.0)`

**`Source/PluginProcessor.cpp`**
- Add `Oscillator oscillator;` member
- In `processBlock()`:
  - Get MIDI note-on/off from `midiMessages`
  - Update oscillator frequency
  - Generate samples in audio buffer loop

**Python Reference:**
- `sine_generator_qt.py:640-680` - PolyBLEP implementation
- `sine_generator_qt.py:3555-3615` - generate_waveform()
- `sine_generator_qt.py:3679-3750` - process_oscillator()

**Key Differences from Python:**
- No NumPy: Use scalar processing in sample loop
- Phase increment: `phaseIncrement = frequency / sampleRate * 2.0 * M_PI`
- PolyBLEP: Convert vectorized version to per-sample

**Validation Checklist:**
- [ ] Sine wave is pure (spectrum analyzer shows only fundamental)
- [ ] Sawtooth is bright but no aliasing artifacts
- [ ] Square wave sounds hollow
- [ ] Pulse width modulation changes square wave timbre
- [ ] MIDI notes play correct pitches (verify A4 = 440Hz)
- [ ] No clicks when changing waveform
- [ ] Phase is continuous when frequency changes

**Parameters to Add:**
- Waveform selector (Sine/Sawtooth/Square)
- Pulse width (for Square only)

---

### Phase 2: ADSR Envelope ⏱️ 2-3 days

**Goal:** Add envelope shaping to single-voice oscillator

**Implement:**

**`Source/DSP/Envelope.h/cpp`**
```cpp
class Envelope {
public:
    enum Phase { Idle, Attack, Decay, Sustain, Release };

    void setSampleRate(double sampleRate);
    void setParameters(float attack, float decay, float sustain, float release);

    void noteOn(float velocity = 1.0f);
    void noteOff();
    void reset();

    float processSample();  // Returns 0.0 - 1.0
    bool isActive() const { return phase != Idle; }

private:
    Phase phase = Idle;
    float level = 0.0f;
    float attackRate = 0.0f;
    float decayRate = 0.0f;
    float sustainLevel = 0.7f;
    float releaseRate = 0.0f;

    double sampleRate = 44100.0;

    void calculateRates();  // Convert time (ms) to per-sample increment
};
```

**Rate Calculation (Port from Python):**
```cpp
void Envelope::calculateRates() {
    // attack/decay/release are in seconds, convert to per-sample increment
    float minAttack = 0.005f;  // 5ms minimum (prevent clicks)
    float attackTime = std::max(attack, minAttack);

    attackRate = 1.0f / (attackTime * sampleRate);
    decayRate = (1.0f - sustainLevel) / (decay * sampleRate);
    releaseRate = sustainLevel / (release * sampleRate);
}
```

**`Source/PluginProcessor.cpp` Integration:**
```cpp
void processBlock(AudioBuffer& buffer, MidiBuffer& midi) {
    for (const auto metadata : midi) {
        auto message = metadata.getMessage();
        if (message.isNoteOn()) {
            oscillator.setFrequency(mtof(message.getNoteNumber()));
            envelope.noteOn(message.getFloatVelocity());
        } else if (message.isNoteOff()) {
            envelope.noteOff();
        }
    }

    for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
        float oscSample = oscillator.processSample();
        float envLevel = envelope.processSample();

        float output = oscSample * envLevel;

        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
            buffer.setSample(ch, sample, output);
    }
}
```

**Python Reference:**
- `sine_generator_qt.py:180-260` - EnvelopeGenerator class (full implementation)
- State machine logic: idle → attack → decay → sustain → release → idle

**Validation Checklist:**
- [ ] Short notes (< 100ms) decay properly
- [ ] Long notes sustain at correct level
- [ ] Release phase works when note released during sustain
- [ ] Attack phase prevents clicks (5ms minimum enforced)
- [ ] No audio glitches during phase transitions
- [ ] Velocity affects initial level correctly
- [ ] Multiple note-ons retrigger attack (monophonic for now)

**Parameters to Add:**
- Attack time (0-2000ms)
- Decay time (0-2000ms)
- Sustain level (0-100%)
- Release time (0-5000ms)

---

### Phase 3: Voice Polyphony & Management ⏱️ 4-5 days

**Goal:** 8-voice polyphony, voice stealing, MONO/POLY/UNI modes

**Implement:**

**`Source/DSP/Voice.h/cpp`**
```cpp
class Voice {
public:
    Voice();

    void setSampleRate(double sampleRate);
    void setOscillatorParameters(/* waveform, pw, etc */);
    void setEnvelopeParameters(float a, float d, float s, float r);

    void noteOn(int midiNote, float velocity, float unisonDetune = 0.0f);
    void noteOff();

    float processSample();  // Returns enveloped oscillator output

    bool isActive() const { return envelope.isActive(); }
    int getCurrentNote() const { return currentNote; }
    int getAge() const { return age; }
    void incrementAge() { ++age; }

private:
    Oscillator oscillator;  // Just 1 for now (will become 3 in Phase 4)
    Envelope envelope;

    int currentNote = -1;   // -1 = free, 0-127 = active
    int age = 0;            // For LRU voice stealing
    float unisonDetune = 0.0f;
};
```

**`Source/DSP/VoiceManager.h/cpp`**
```cpp
class VoiceManager {
public:
    enum VoiceMode { Mono, Poly, Unison };

    VoiceManager();

    void setSampleRate(double sampleRate);
    void setVoiceMode(VoiceMode mode);

    void noteOn(int midiNote, float velocity);
    void noteOff(int midiNote);
    void allNotesOff();

    float processSample();  // Sum all active voices

    // Forward parameter changes to all voices
    void setOscillatorParameters(/* params */);
    void setEnvelopeParameters(float a, float d, float s, float r);

private:
    static constexpr int MAX_VOICES = 8;
    std::array<Voice, MAX_VOICES> voices;
    VoiceMode voiceMode = Poly;

    Voice* findFreeVoice();
    Voice* stealVoice();           // LRU algorithm
    void allocateVoices(int midiNote, float velocity);
    float calculateUnisonDetune(int voiceIndex);  // Symmetric spread

    void incrementAllAges();
};
```

**Voice Stealing Algorithm (LRU):**
```cpp
Voice* VoiceManager::stealVoice() {
    Voice* oldest = nullptr;
    int maxAge = -1;

    for (auto& voice : voices) {
        if (voice.isActive() && voice.getAge() > maxAge) {
            maxAge = voice.getAge();
            oldest = &voice;
        }
    }

    return oldest;
}
```

**Unison Detune Calculation (Symmetric):**
```cpp
float VoiceManager::calculateUnisonDetune(int voiceIndex) {
    const float detuneDepth = 20.0f;  // ±20 cents
    const int numVoices = 8;

    // Symmetric spread: [-20, -14.3, -8.6, -2.9, +2.9, +8.6, +14.3, +20]
    float step = (2.0f * detuneDepth) / (numVoices - 1);
    return -detuneDepth + (step * voiceIndex);
}
```

**Python Reference:**
- `sine_generator_qt.py:100-177` - Voice class
- `sine_generator_qt.py:4050-4140` - handle_midi_note_on/off
- `sine_generator_qt.py:4230-4260` - steal_voice()
- `sine_generator_qt.py:998-1020` - reallocate_voice_pool()

**Validation Checklist:**
- [ ] POLY mode: Can play 8 simultaneous notes
- [ ] 9th note steals oldest voice (LRU)
- [ ] MONO mode: Only 1 note plays, new notes retrigger
- [ ] UNI mode: Single note triggers 8 detuned voices (chorus effect)
- [ ] Voice stealing is click-free (envelope retriggering)
- [ ] Note-off releases correct voice(s)
- [ ] All notes off clears all voices
- [ ] No audio glitches when switching modes

**Parameters to Add:**
- Voice mode selector (Mono/Poly/Unison)

---

### Phase 4: Three Oscillators + Mixer ⏱️ 2-3 days

**Goal:** 3 oscillators per voice, post-mixer envelope architecture

**Implement:**

**Update `Source/DSP/Voice.h/cpp`:**
```cpp
class Voice {
private:
    std::array<Oscillator, 3> oscillators;  // Changed from single
    Envelope envelope;

    // Per-oscillator parameters
    std::array<float, 3> oscillatorGains = {0.33f, 0.33f, 0.33f};
    std::array<float, 3> detuneCents = {0.0f, 0.0f, 0.0f};
    std::array<int, 3> octaveOffsets = {0, 0, 0};  // -3 to +3

public:
    void setOscillatorGain(int oscIndex, float gain);
    void setOscillatorDetune(int oscIndex, float cents);
    void setOscillatorOctave(int oscIndex, int offset);

    float processSample() {
        // Mix 3 oscillators
        float mix = 0.0f;
        for (int i = 0; i < 3; ++i) {
            if (oscillatorEnabled[i]) {
                mix += oscillators[i].processSample() * oscillatorGains[i];
            }
        }

        // Apply single envelope to mixed signal (post-mixer)
        float envLevel = envelope.processSample();
        return mix * envLevel;
    }
};
```

**Detune Application:**
```cpp
void Voice::noteOn(int midiNote, float velocity, float unisonDetune) {
    float baseFreq = 440.0f * pow(2.0f, (midiNote - 69) / 12.0f);

    for (int i = 0; i < 3; ++i) {
        // Apply octave offset
        float octaveMultiplier = pow(2.0f, octaveOffsets[i]);

        // Apply detune in cents (1 cent = 2^(1/1200))
        float totalDetune = detuneCents[i] + unisonDetune;  // cents
        float detuneMultiplier = pow(2.0f, totalDetune / 1200.0f);

        float finalFreq = baseFreq * octaveMultiplier * detuneMultiplier;
        oscillators[i].setFrequency(finalFreq);
    }

    envelope.noteOn(velocity);
}
```

**Python Reference:**
- `sine_generator_qt.py:3812-3814` - Processing 3 oscillators
- `sine_generator_qt.py:2467-2470` - apply_octave()
- `sine_generator_qt.py:2463-2465` - apply_detune()

**Validation Checklist:**
- [ ] All 3 oscillators audible when enabled
- [ ] Detune creates beating/chorus effect
- [ ] Octave offsets work (-3 to +3)
- [ ] Per-oscillator gains mix correctly
- [ ] Pulse width modulates square wave timbre
- [ ] Turning oscillators on/off works without clicks
- [ ] Unison mode with 3 oscillators = thick sound

**Parameters to Add (per oscillator):**
- Enabled (on/off)
- Waveform (Sine/Saw/Square)
- Detune (±100 cents)
- Octave (-3 to +3)
- Pulse Width (1-99%)
- Gain (0-100%)

---

### Phase 5: Moog Ladder Filter ⏱️ 3-4 days

**Goal:** 4-pole Moog ladder filter with LP/BP/HP modes

**Implement:**

**`Source/DSP/MoogFilter.h/cpp`**
```cpp
class MoogFilter {
public:
    enum Mode { LowPass, BandPass, HighPass };

    void setSampleRate(double sampleRate);
    void setMode(Mode mode);
    void setCutoff(float cutoffHz);        // 20 - 20000 Hz
    void setResonance(float resonance);    // 0.0 - 1.0

    float processSample(float input);
    void reset();

private:
    Mode mode = LowPass;
    double sampleRate = 44100.0;

    // 4 filter stages
    float stage1 = 0.0f, stage2 = 0.0f, stage3 = 0.0f, stage4 = 0.0f;

    // Cached coefficients
    float g = 0.0f;              // Cutoff coefficient
    float feedbackGain = 0.0f;   // Resonance feedback

    bool coefficientsNeedUpdate = true;
    void updateCoefficients();
};
```

**Filter Implementation (Simplified Moog Topology):**
```cpp
float MoogFilter::processSample(float input) {
    if (coefficientsNeedUpdate)
        updateCoefficients();

    // Feedback from output to input (resonance)
    float feedback = (mode == LowPass) ? stage4 : stage2;
    float inputWithFeedback = input - feedback * feedbackGain;

    // 4 one-pole lowpass stages
    stage1 = stage1 + g * (tanh(inputWithFeedback) - stage1);
    stage2 = stage2 + g * (stage1 - stage2);
    stage3 = stage3 + g * (stage2 - stage3);
    stage4 = stage4 + g * (stage3 - stage4);

    // Output selection
    switch (mode) {
        case LowPass:  return stage4;               // 24dB/octave
        case BandPass: return stage2;               // 12dB/octave
        case HighPass: return input - stage4;       // High-pass by subtraction
    }
}

void MoogFilter::updateCoefficients() {
    // Cutoff coefficient (simplified, normalized to sample rate)
    float normalizedCutoff = std::clamp(cutoff / sampleRate, 0.0f, 0.499f);
    g = tan(M_PI * normalizedCutoff);

    // Resonance to feedback gain (0.0 - 3.5 range)
    feedbackGain = resonance * 3.5f;

    coefficientsNeedUpdate = false;
}
```

**Python Reference:**
- `sine_generator_qt.py:370-440` - MoogLadderFilter class
- Coefficient calculation
- State limiting for stability

**Validation Checklist:**
- [ ] LP mode removes high frequencies smoothly
- [ ] BP mode isolates mid-range frequencies
- [ ] HP mode removes bass frequencies
- [ ] Resonance creates peak at cutoff frequency
- [ ] No instability or self-oscillation at high resonance
- [ ] Filter doesn't pop when sweeping cutoff
- [ ] Filter doesn't pop when switching modes
- [ ] Cutoff range covers 20Hz - 20kHz

**Parameters to Add:**
- Filter mode (LP/BP/HP)
- Cutoff frequency (20 - 20000 Hz, logarithmic)
- Resonance (0 - 100%)

---

### Phase 6: Dual LFO System ⏱️ 4-5 days

**Goal:** 2 independent LFOs with 7 destinations, modulation combining

**Implement:**

**`Source/DSP/LFO.h/cpp`**
```cpp
class LFO {
public:
    enum Waveform { Sine, Triangle, Square, Sawtooth, Random };
    enum RateMode { Free, Sync };
    enum Destination {
        None,
        AllOscsPitch,
        FilterCutoff,
        AllOscsVolume,
        Osc1PulseWidth,
        Osc2PulseWidth,
        Osc3PulseWidth
    };

    void setSampleRate(double sampleRate);
    void setWaveform(Waveform waveform);
    void setRateMode(RateMode mode);
    void setRateHz(float hz);              // 0.1 - 20 Hz
    void setSyncDivision(String division); // "1/16", "1/8", etc.
    void setBPM(float bpm);

    float processSample();  // Returns -1.0 to +1.0

private:
    Waveform waveform = Sine;
    RateMode rateMode = Free;
    double phase = 0.0;
    double phaseIncrement = 0.0;

    float generateWaveform();
};
```

**Modulation Application (in VoiceManager):**
```cpp
struct LFOModulation {
    std::map<int, float> pitchMod;     // oscNum -> multiplier
    std::map<int, float> pwMod;        // oscNum -> offset
    std::map<int, float> volumeMod;    // oscNum -> multiplier
    float filterMod = 0.0f;            // cutoff Hz or 0
};

LFOModulation applyLFOModulation(int lfoNum, float lfoSignal,
                                  Destination dest, float depth, float mix) {
    LFOModulation result;

    if (dest == None || depth == 0.0f)
        return result;

    switch (dest) {
        case AllOscsPitch: {
            float scalar = 1.0f + (lfoSignal * 0.05f * depth * mix);  // ±5%
            result.pitchMod = {{1, scalar}, {2, scalar}, {3, scalar}};
            break;
        }

        case FilterCutoff: {
            // ±2 octaves modulation
            float cutoffMod = pow(2.0f, lfoSignal * 2.0f * depth);
            result.filterMod = baseCutoff * cutoffMod * mix;
            break;
        }

        // ... other destinations
    }

    return result;
}
```

**Modulation Combining:**
```cpp
// Both LFO1 and LFO2 target pitch:
float combinedPitchMod = lfo1PitchMod * lfo2PitchMod;  // Multiplicative

// Both target pulse width:
float combinedPWMod = lfo1PWMod + lfo2PWMod;  // Additive

// Both target filter:
float combinedFilterMod = (lfo1FilterMod + lfo2FilterMod) / 2.0f;  // Average
```

**Python Reference:**
- `sine_generator_qt.py:580-630` - LFOGenerator class
- `sine_generator_qt.py:3687-3740` - apply_lfo_modulation()
- `sine_generator_qt.py:3825-3841` - Dual LFO signal generation

**Validation Checklist:**
- [ ] LFO1 creates vibrato on all oscillators
- [ ] LFO2 sweeps filter cutoff
- [ ] Both LFOs can target same destination
- [ ] Depth control adjusts modulation amount
- [ ] Mix control blends dry/wet
- [ ] Free mode: Hz rate control works
- [ ] Sync mode: MIDI divisions work
- [ ] All 5 waveforms sound different

**Parameters to Add (per LFO):**
- Waveform (5 choices)
- Rate mode (Free/Sync)
- Rate Hz (0.1 - 20 Hz)
- Sync division (1/16 to 4/1)
- Destination (7 choices)
- Depth (0 - 100%)
- Mix (0 - 100%)

---

### Phase 7: Noise Generator ⏱️ 1-2 days

**Goal:** White/Pink/Brown noise source with envelope

**Implement:**

**`Source/DSP/NoiseGenerator.h/cpp`**
```cpp
class NoiseGenerator {
public:
    enum NoiseType { White, Pink, Brown };

    void setSampleRate(double sampleRate);
    void setType(NoiseType type);
    void setGain(float gain);

    void trigger();   // Start envelope
    void release();   // Release envelope

    float processSample();  // Returns noise * envelope * gain

private:
    NoiseType type = White;
    float gain = 0.5f;
    Envelope envelope;

    // Pink noise state (Paul Kellet algorithm)
    float pinkState[5] = {0.0f};

    // Brown noise state
    float brownState = 0.0f;

    float generateWhite();
    float generatePink();   // Paul Kellet 5-stage
    float generateBrown();  // Integration
};
```

**Pink Noise (Paul Kellet Algorithm):**
```cpp
float NoiseGenerator::generatePink() {
    float white = generateWhite();

    pinkState[0] = 0.99886f * pinkState[0] + white * 0.0555179f;
    pinkState[1] = 0.99332f * pinkState[1] + white * 0.0750759f;
    pinkState[2] = 0.96900f * pinkState[2] + white * 0.1538520f;
    pinkState[3] = 0.86650f * pinkState[3] + white * 0.3104856f;
    pinkState[4] = 0.55000f * pinkState[4] + white * 0.5329522f;

    float pink = pinkState[0] + pinkState[1] + pinkState[2] +
                 pinkState[3] + pinkState[4];

    return pink * 0.11f;  // Normalize
}
```

**Python Reference:**
- `sine_generator_qt.py:490-577` - NoiseGenerator class
- Pink noise algorithm (Paul Kellet)

**Validation Checklist:**
- [ ] White noise is bright and hissy
- [ ] Pink noise is balanced across spectrum
- [ ] Brown noise is dark and rumbly
- [ ] Noise has independent envelope
- [ ] Gain control adjusts level
- [ ] No clicks when enabling/disabling

**Parameters to Add:**
- Noise type (White/Pink/Brown)
- Noise gain (0 - 100%)
- Noise enabled (on/off)

---

### Phase 8: JUCE GUI ⏱️ 7-10 days

**Goal:** Professional plugin UI matching Python version layout

**Implement:**

**`Source/PluginEditor.h/cpp`**
- Main window layout with sections
- JUCE component-based UI

**Component Hierarchy:**
```
PluginEditor
├── TopBar (MIDI port, voice modes, playback mode)
├── OscillatorSection (3 columns)
│   ├── OscillatorPanel x3
│   │   ├── OnOffButton
│   │   ├── WaveformCombo
│   │   ├── DetuneKnob
│   │   ├── OctaveButtons
│   │   ├── PulseWidthKnob
│   │   └── GainSlider
├── MixerSection
│   ├── Osc1GainSlider
│   ├── Osc2GainSlider
│   ├── Osc3GainSlider
│   └── MasterVolumeSlider
├── FilterSection
│   ├── ModeButtons (LP/BP/HP)
│   ├── CutoffKnob
│   └── ResonanceKnob
├── EnvelopeSection
│   ├── AttackSlider
│   ├── DecaySlider
│   ├── SustainKnob
│   └── ReleaseSlider
├── LFOSection x2
│   ├── WaveformCombo
│   ├── RateModeCombo
│   ├── RateKnob
│   ├── DestinationCombo
│   ├── DepthKnob
│   ├── MixKnob
│   └── LEDIndicator
└── PresetSection
    ├── PresetBrowser
    ├── SaveButton
    └── LoadButton
```

**JUCE Components to Use:**
- `juce::Slider` - Knobs and faders
- `juce::ComboBox` - Dropdowns
- `juce::TextButton` - Buttons
- `juce::Label` - Text labels
- Custom `LEDIndicator` component (similar to Python)

**Parameter Attachment:**
```cpp
// In PluginEditor constructor
cutoffAttachment = std::make_unique<SliderAttachment>(
    parameters, "cutoff", cutoffKnob
);
```

**Validation Checklist:**
- [ ] All parameters controllable from UI
- [ ] UI matches Python version layout
- [ ] Knobs update when parameters change
- [ ] Parameter automation works in DAW
- [ ] UI is responsive (no lag)
- [ ] Resizable window (optional)
- [ ] LED indicators pulse with LFO

---

### Phase 9: Preset System ⏱️ 2-3 days

**Goal:** Save/load presets, preset browser, factory presets

**Implement:**

**JUCE ValueTree State:**
```cpp
// In PluginProcessor.h
juce::AudioProcessorValueTreeState parameters;

juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Oscillator 1
    params.push_back(std::make_unique<AudioParameterChoice>(
        "osc1_waveform", "Osc 1 Waveform",
        juce::StringArray{"Sine", "Sawtooth", "Square"}, 0));

    params.push_back(std::make_unique<AudioParameterFloat>(
        "osc1_detune", "Osc 1 Detune", -100.0f, 100.0f, 0.0f));

    // ... all other parameters

    return { params.begin(), params.end() };
}
```

**Preset Save/Load:**
```cpp
void PluginProcessor::savePreset(const File& file) {
    auto state = parameters.copyState();
    auto xml = state.createXml();
    xml->writeTo(file);
}

void PluginProcessor::loadPreset(const File& file) {
    auto xml = parseXML(file);
    if (xml) {
        auto state = ValueTree::fromXml(*xml);
        parameters.replaceState(state);
    }
}
```

**Factory Presets:**
- Port presets from Python repo `Presets/` folder
- Embed as binary data in plugin (optional)
- Or ship as separate `.vstpreset` files

**Validation Checklist:**
- [ ] Presets save all parameters
- [ ] Presets load correctly
- [ ] Factory presets included (Init, Leads, Bass, Pads)
- [ ] Preset browser works
- [ ] DAW automation state saved/loaded
- [ ] No parameter jumps when loading preset

---

## Python → C++ Component Mapping Reference

| Python Component | Line Numbers | C++ Component | Notes |
|------------------|-------------|---------------|-------|
| `EnvelopeGenerator` | 180-260 | `DSP/Envelope.h/cpp` | State machine logic identical |
| `poly_blep_vectorized()` | 640-680 | `DSP/AudioUtils.cpp::polyBLEP()` | Convert NumPy to scalar |
| `LFOGenerator` | 580-630 | `DSP/LFO.h/cpp` | 5 waveforms, Free/Sync modes |
| `NoiseGenerator` | 490-577 | `DSP/NoiseGenerator.h/cpp` | Paul Kellet pink noise |
| `MoogLadderFilter` | 370-440 | `DSP/MoogFilter.h/cpp` | 4-pole topology, 3 modes |
| `Voice` | 100-177 | `DSP/Voice.h/cpp` | 3 osc + envelope container |
| `audio_callback()` | 3783-3864 | `PluginProcessor::processBlock()` | Main audio loop |
| `handle_midi_note_on/off` | 4050-4140 | `PluginProcessor::processBlock()` MIDI | Voice allocation |
| `generate_waveform()` | 3555-3615 | `Oscillator::processSample()` | Sine/Saw/Square generation |
| `apply_lfo_modulation()` | 3687-3740 | `VoiceManager::applyLFOModulation()` | Modulation calculation |
| `steal_voice()` | 4230-4260 | `VoiceManager::stealVoice()` | LRU algorithm |

---

## Key Architectural Differences: Python vs C++

### Vectorization → Scalar Processing
**Python (NumPy):**
```python
lfo_signal = np.sin(phase)  # 512 samples at once
wave = gain * lfo_signal    # Vectorized multiplication
```

**C++ (per-sample):**
```cpp
for (int i = 0; i < numSamples; ++i) {
    float lfoSample = sin(phase);
    float sample = gain * lfoSample;
    phase += phaseIncrement;
}
```

### Memory Management
**Python:** Garbage collected, dynamic typing
**C++:** Manual memory, use RAII, `std::unique_ptr`, avoid `new/delete`

### Audio Buffer Access
**Python (sounddevice):** Callback receives NumPy array
**C++ (JUCE):** `AudioBuffer<float>`, per-sample or block processing

### Threading
**Python:** GIL limitations
**C++:** JUCE handles audio thread automatically, use lock-free where possible

---

## Development Workflow

### Iteration Cycle:
1. **Implement:** Write C++ code for phase component
2. **Build:** Compile VST3 (5-30 seconds depending on changes)
3. **Test:** Load in DAW, play MIDI, verify audio
4. **Validate:** Check phase checklist, compare to Python version
5. **Document:** Update `IMPLEMENTATION_PLAN.md` with progress
6. **Commit:** Git commit with descriptive message
7. **Next:** Move to next phase

### Testing Strategy:
- **Standalone app:** Quick iteration, printf debugging
- **DAW testing:** Reaper (free), Ableton, Logic
- **A/B comparison:** Load Python version and C++ version side-by-side
- **Spectrum analyzer:** Verify waveform fidelity
- **Oscilloscope:** Check for DC offset, clipping

### Git Commits:
- One commit per component (e.g., "Add Oscillator class with PolyBLEP")
- Reference Python line numbers in commit messages
- Tag each phase completion (e.g., `git tag phase-1-complete`)

---

## Learning Resources

### JUCE:
- **Official Tutorials:** https://juce.com/learn/tutorials
- **Documentation:** https://docs.juce.com/
- **Forum:** https://forum.juce.com/
- **YouTube:** TheAudioProgrammer channel

### Books:
- **"Audio Plugin Development" by Will Pirkle** (excellent, comprehensive)
- **"Designing Audio Effect Plugins in C++" by Will Pirkle**
- **"The Audio Programming Book" by Boulanger & Lazzarini**

### DSP:
- **Julius O. Smith's Online Books:** https://ccrma.stanford.edu/~jos/
- **"Designing Sound" by Andy Farnell** (Pd but concepts apply)

### C++:
- **"A Tour of C++" by Bjarne Stroustrup** (if new to modern C++)
- **C++ Reference:** https://en.cppreference.com/

---

## Success Criteria

**Minimum Viable Plugin (MVP):** Phase 6 complete
- All core synthesis features working
- No UI required (parameter automation via DAW)
- Sounds identical to Python version

**Full Feature Parity:** Phase 9 complete
- Professional UI
- Preset system
- All features from Python version
- Ready for distribution

**Stretch Goals:**
- Additional waveforms (wavetables)
- More filter types (SVF, comb, etc.)
- Effects (reverb, delay, distortion)
- MPE support (MIDI Polyphonic Expression)
- Microtuning support

---

## Next Steps

1. **Create new GitHub repo:** `TripleOscillatorVST`
2. **Download JUCE Framework:** https://juce.com/get-juce/download
3. **Begin Phase 0:** Project setup
4. **Read this plan thoroughly**
5. **Reference Python repo frequently**

**When starting each phase:**
1. Read phase description in this document
2. Check Python reference line numbers
3. Implement C++ version
4. Validate against checklist
5. Update `IMPLEMENTATION_PLAN.md` with progress

---

**Ready to build a professional VST3/AU plugin! 🎹**

*This document serves as your complete roadmap. Copy it to the new VST repo and update progress as you go.*
