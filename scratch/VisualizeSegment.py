import matplotlib.pyplot as plt
import numpy as np
import math
import os

def freq_to_midi(freq):
    """Convert frequency in Hz to closest MIDI note number (0-127)."""
    if freq <= 0: return 0
    return 69 + 12 * math.log2(freq / 440.0)

def plot_segment(notes_annot, note_indices, chunk_start_sec, f0_matrix, f0_freqs, f0_time_r, output_path, melody_vector=None, melody_time_res=0.0):
    """
    Plots the segment:
    - x-axis: time in seconds (relative to chunk_start_sec)
    - y-axis: log frequency (measured in MIDI note values)
    """
    plt.figure(figsize=(14, 7))
    
    logged_midi = False
    
    # Plot MIDI Notes
    for idx in note_indices:
        note_info = notes_annot[idx]
        start_t = note_info['time'][0] - chunk_start_sec
        end_t = note_info['time'][1] - chunk_start_sec
        
        freq_list = note_info['freq']
        if isinstance(freq_list, (list, tuple)) and len(freq_list) > 0:
            avg_freq = sum(freq_list) / len(freq_list)
            midi_pitch = freq_to_midi(avg_freq)
        else:
            midi_pitch = 60
            
        label = 'MIDI Note (Base Frequency)' if not logged_midi else ""
        logged_midi = True
        
        plt.hlines(midi_pitch, start_t, end_t, colors='#d35400', linestyles='solid', lw=20, alpha=0.6, label=label)
        plt.text(start_t, midi_pitch + 0.6, note_info['text'], fontsize=10, color='white', 
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))

    # Plot annot2vector Melody Curve (DALI internal notes.freq representation)
    if melody_vector is not None and melody_time_res > 0:
        mel_times = []
        mel_pitches = []
        for i, val in enumerate(melody_vector):
            if val > 0:
                # time of this frame
                t = i * melody_time_res
                # Check if it falls within the chunk timeframe (plus a small buffer)
                end_chunk = notes_annot[note_indices[-1]]['time'][1]
                if t >= chunk_start_sec and t <= end_chunk:
                    mel_times.append(t - chunk_start_sec)
                    mel_pitches.append(freq_to_midi(val))
        
        if mel_times:
            plt.plot(mel_times, mel_pitches, color='#2ecc71', linestyle='-', linewidth=2.5, marker='.', markersize=4, alpha=0.8, label='annot2vector Melody Curve')

    # Plot F0 Curve
    if f0_matrix is not None and f0_freqs is not None and f0_time_r > 0:
        f0_times = []
        f0_pitches = []
        for idx in note_indices:
            note_info = notes_annot[idx]
            start_t = note_info['time'][0]
            end_t = note_info['time'][1]
            
            # The start and end positions within the f0_matrix array
            col_start = int(start_t / f0_time_r)
            col_end = int(end_t / f0_time_r)
            
            if col_end > col_start:
                slice_f0 = f0_matrix[:, col_start:col_end]
                peaks = np.argmax(slice_f0, axis=0)
                freqs = f0_freqs[peaks]
                
                for i_col, f in enumerate(freqs):
                    if f > 0:
                        t = (col_start + i_col) * f0_time_r - chunk_start_sec
                        f0_times.append(t)
                        f0_pitches.append(freq_to_midi(f))
                        
        if f0_times:
            plt.plot(f0_times, f0_pitches, color='#2980b9', linestyle='-', linewidth=2.5, marker='.', markersize=4, alpha=0.8, label='Extracted F0 Curve (.npz matrix)')
        
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Pitch (MIDI Note Number / Log Freq)', fontsize=12)
    plt.title('Segment Visualization: Base MIDI vs. Extracted F0 vs. annot2vector F0', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Force Y axis to show integer MIDI pitches nicely
    ax = plt.gca()
    y_min, y_max = ax.get_ylim()
    ax.set_yticks(np.arange(math.floor(y_min), math.ceil(y_max) + 1, 1))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path
