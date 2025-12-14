# UI Layout System Guide

This tutorial explains how the Triple Oscillator Synth UI layout system works and how to add or modify sections without causing widget overlapping.

## Table of Contents
1. [Layout Architecture Overview](#layout-architecture-overview)
2. [PyQt5 Layout Basics](#pyqt5-layout-basics)
3. [Main Layout Structure](#main-layout-structure)
4. [Creating Sections with QGroupBox](#creating-sections-with-qgroupbox)
5. [Adding New UI Sections](#adding-new-ui-sections)
6. [Best Practices](#best-practices)
7. [Common Pitfalls](#common-pitfalls)

---

## Layout Architecture Overview

The synth uses a **single-column vertical layout** where all major sections stack from top to bottom:

```
┌─────────────────────────────────────┐
│  Header Row (MIDI, Mode, Power)     │
├─────────────────────────────────────┤
│  OSCILLATORS                         │
│  (3 oscillators horizontal)          │
├─────────────────────────────────────┤
│  MIXER                               │
├─────────────────────────────────────┤
│  ADSR ENVELOPE                       │
├─────────────────────────────────────┤
│  FILTER                              │
├─────────────────────────────────────┤
│  LFO MODULATION                      │
└─────────────────────────────────────┘
```

**Window Size:** 1000x900 pixels (resizable)

---

## PyQt5 Layout Basics

### QVBoxLayout - Vertical Stacking
Arranges widgets **vertically** (top to bottom):

```python
layout = QVBoxLayout()
layout.addWidget(widget1)  # Top
layout.addWidget(widget2)  # Below widget1
layout.addWidget(widget3)  # Below widget2
```

### QHBoxLayout - Horizontal Arrangement
Arranges widgets **horizontally** (left to right):

```python
layout = QHBoxLayout()
layout.addWidget(widget1)  # Left
layout.addWidget(widget2)  # Right of widget1
layout.addWidget(widget3)  # Right of widget2
```

### Key Methods
- `addWidget(widget)` - Add a widget to the layout
- `addLayout(layout)` - Nest a layout inside another layout
- `addStretch(factor)` - Add flexible spacing
- `setSpacing(pixels)` - Set spacing between widgets
- `setContentsMargins(l, t, r, b)` - Set margins around the layout

---

## Main Layout Structure

The synth's main layout is defined in `init_ui()` method:

```python
def init_ui(self):
    """Initialize the user interface"""
    self.setWindowTitle("Triple Oscillator Synth")
    self.setMinimumSize(1000, 900)
    self.resize(1000, 900)

    # Create central widget and main layout
    central_widget = QWidget()
    self.setCentralWidget(central_widget)

    main_layout = QVBoxLayout(central_widget)
    main_layout.setSpacing(10)
    main_layout.setContentsMargins(10, 10, 10, 10)
```

**Explanation:**
1. `QVBoxLayout(central_widget)` - Creates vertical layout attached to central widget
2. `setSpacing(10)` - 10 pixels between each section
3. `setContentsMargins(10, 10, 10, 10)` - 10px margin on all sides

### Header Row

```python
# Header row with MIDI controls, mode selection, power, presets
midi_layout = QHBoxLayout()
midi_layout.setSpacing(10)

# Add MIDI dropdown
midi_label = QLabel("MIDI Input:")
midi_layout.addWidget(midi_label)
# ... more widgets ...

midi_layout.addStretch(1)  # Push everything to the left
main_layout.addLayout(midi_layout)  # Add to main vertical layout
```

**Key Points:**
- Uses `QHBoxLayout` for horizontal arrangement
- `addStretch(1)` creates flexible space that pushes widgets left
- Added to `main_layout` with `addLayout()`

---

## Creating Sections with QGroupBox

All major sections (OSCILLATORS, MIXER, etc.) use `QGroupBox` for visual grouping:

### The create_group_box() Helper Method

```python
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
```

**What it does:**
- Creates a `QGroupBox` with a title
- Applies consistent styling (border, font, colors)
- Returns the styled group box

### Example: Creating the OSCILLATORS Section

```python
# 1. Create the group box
oscillators_group = self.create_group_box("OSCILLATORS")

# 2. Create a layout for the group's content
oscillators_layout = QVBoxLayout(oscillators_group)

# 3. Create horizontal row for 3 oscillators
oscillators_row = QHBoxLayout()
oscillators_row.setSpacing(10)

# 4. Create and add each oscillator
osc1_widget = self.create_oscillator_column("Oscillator 1", 1)
oscillators_row.addWidget(osc1_widget, 1)

osc2_widget = self.create_oscillator_column("Oscillator 2", 2)
oscillators_row.addWidget(osc2_widget, 1)

osc3_widget = self.create_oscillator_column("Oscillator 3", 3)
oscillators_row.addWidget(osc3_widget, 1)

# 5. Add the horizontal row to the group's layout
oscillators_layout.addLayout(oscillators_row)

# 6. Add the group to the main layout
main_layout.addWidget(oscillators_group)
```

**Step-by-step:**
1. `create_group_box()` - Creates styled container
2. `QVBoxLayout(oscillators_group)` - Layout for group's content
3. `QHBoxLayout()` - Horizontal row for 3 oscillators
4. `create_oscillator_column()` - Create each oscillator widget
5. `addLayout()` - Add horizontal row to group's vertical layout
6. `addWidget()` - Add entire group to main layout

---

## Adding New UI Sections

### Example: Adding a New "EFFECTS" Section

Let's add a reverb/delay effects section between FILTER and LFO:

```python
# In init_ui(), after the FILTER section:

# EFFECTS SECTION
effects_group = self.create_group_box("EFFECTS")
effects_layout = QVBoxLayout(effects_group)
effects_widget = self.create_effects_section()  # Your custom method
effects_layout.addWidget(effects_widget)
main_layout.addWidget(effects_group)

# LFO SECTION (already exists below)
lfo_group = self.create_group_box("LFO & MODULATION")
# ... rest of LFO code ...
```

### Creating the Content Method

```python
def create_effects_section(self):
    """Create the effects section with reverb and delay"""
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setSpacing(20)

    # Reverb controls
    reverb_layout = QVBoxLayout()

    reverb_label = QLabel("Reverb")
    reverb_label.setAlignment(Qt.AlignCenter)
    reverb_layout.addWidget(reverb_label)

    reverb_knob = QDial()
    reverb_knob.setFixedSize(60, 60)
    reverb_knob.setRange(0, 100)
    reverb_knob.setValue(0)
    reverb_layout.addWidget(reverb_knob)

    reverb_value = QLabel("0%")
    reverb_value.setAlignment(Qt.AlignCenter)
    reverb_layout.addWidget(reverb_value)

    layout.addLayout(reverb_layout)

    # Delay controls (similar structure)
    # ... add delay knob ...

    layout.addStretch(1)
    return widget
```

**Pattern to follow:**
1. Create main `QWidget` container
2. Create layout for the widget (`QHBoxLayout` or `QVBoxLayout`)
3. Add controls (labels, knobs, buttons, etc.)
4. Return the widget

---

## Best Practices

### 1. Always Use Layouts
❌ **Bad:** Setting absolute positions
```python
button.move(100, 200)  # DON'T DO THIS
button.setGeometry(100, 200, 80, 30)  # DON'T DO THIS
```

✅ **Good:** Using layouts
```python
layout.addWidget(button)  # Automatic positioning
```

### 2. Consistent Spacing
Use consistent spacing values throughout:
```python
layout.setSpacing(10)  # Standard spacing
layout.setSpacing(15)  # Larger sections
layout.setSpacing(5)   # Compact elements
```

### 3. Fixed Sizes for Knobs
Set fixed sizes for `QDial` widgets to ensure consistency:
```python
knob = QDial()
knob.setFixedSize(60, 60)  # Same size for all similar knobs
```

### 4. Use Stretch for Alignment
Use `addStretch()` to control alignment:
```python
# Left-align widgets
layout.addWidget(widget1)
layout.addWidget(widget2)
layout.addStretch(1)  # Push widgets to the left

# Center widgets
layout.addStretch(1)
layout.addWidget(widget)
layout.addStretch(1)

# Right-align widgets
layout.addStretch(1)
layout.addWidget(widget1)
layout.addWidget(widget2)
```

### 5. Nest Layouts Logically
Create hierarchical structures:
```python
# Main vertical layout
main_layout = QVBoxLayout()

# Section 1: Horizontal row
section1 = QHBoxLayout()
section1.addWidget(widget1)
section1.addWidget(widget2)
main_layout.addLayout(section1)

# Section 2: Another horizontal row
section2 = QHBoxLayout()
section2.addWidget(widget3)
section2.addWidget(widget4)
main_layout.addLayout(section2)
```

---

## Common Pitfalls

### 1. Overlapping Widgets
**Problem:** Widgets render on top of each other

**Causes:**
- Not using layouts (using absolute positioning)
- Adding same widget to multiple layouts
- Forgetting to add layout to parent

**Solution:**
```python
# Always add layouts to their parent
parent_layout.addLayout(child_layout)

# Always add widgets to a layout
layout.addWidget(widget)
```

### 2. Widgets Not Showing
**Problem:** Widgets created but don't appear

**Causes:**
- Created widget but never added to layout
- Created layout but never added to parent layout
- Widget hidden or size set to 0

**Solution:**
```python
# Check the chain: widget → layout → parent layout → main layout
widget = QWidget()
layout = QVBoxLayout(widget)
layout.addWidget(control)  # Add control to layout
parent_layout.addWidget(widget)  # Add widget to parent
```

### 3. Inconsistent Sizing
**Problem:** Some knobs/buttons are different sizes

**Solution:**
```python
# Use setFixedSize() for consistent sizing
for knob in [knob1, knob2, knob3]:
    knob.setFixedSize(60, 60)
```

### 4. Too Much Nesting
**Problem:** Code becomes hard to read with too many nested layouts

**Solution:**
- Extract sections into separate methods (like `create_oscillator_column()`)
- Use helper methods for repeated patterns
- Keep nesting depth < 4 levels

---

## Layout Debugging Tips

### 1. Add Borders for Debugging
Temporarily add borders to see widget boundaries:
```python
widget.setStyleSheet("border: 1px solid red;")
```

### 2. Check Size Hints
Print widget sizes to debug:
```python
print(f"Widget size: {widget.size()}")
print(f"Size hint: {widget.sizeHint()}")
```

### 3. Use Qt.AlignCenter
Center widgets within their layout space:
```python
layout.addWidget(label, alignment=Qt.AlignCenter)
```

### 4. Inspect Layout Structure
Print layout hierarchy:
```python
def print_layout_tree(layout, indent=0):
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget():
            print("  " * indent + f"Widget: {item.widget().__class__.__name__}")
        elif item.layout():
            print("  " * indent + f"Layout: {item.layout().__class__.__name__}")
            print_layout_tree(item.layout(), indent + 1)
```

---

## Summary

**Key Principles:**
1. Use `QVBoxLayout` for vertical stacking
2. Use `QHBoxLayout` for horizontal arrangement
3. Use `QGroupBox` for visual section grouping
4. Always add widgets to layouts, never use absolute positioning
5. Use helper methods like `create_group_box()` for consistency
6. Extract complex sections into separate methods
7. Set fixed sizes for controls that should be uniform (knobs, buttons)
8. Use `addStretch()` for flexible spacing and alignment

**The Golden Rule:**
> Every widget must be added to a layout, and every layout must be added to a parent layout or widget.

Following this structure ensures a clean, maintainable UI without overlapping widgets!
