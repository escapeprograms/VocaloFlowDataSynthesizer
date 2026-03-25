import math

def freq_to_midi(freq):
    """Convert frequency in Hz to closest MIDI note number (0-127)."""
    if freq <= 0:
        return 0
    return int(round(69 + 12 * math.log2(freq / 440.0)))

def get_midi_pitch(freq_list):
    """Get the average MIDI pitch from a list of frequencies."""
    if isinstance(freq_list, (list, tuple)) and len(freq_list) > 0:
        avg_freq = sum(freq_list) / len(freq_list)
        return freq_to_midi(avg_freq)
    else:
        return 60 # Default middle C if parsing fails
