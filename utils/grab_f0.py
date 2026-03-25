import os
import numpy as np
import math

def load_f0_data(dali_id, dali_base_path):
    f0_npz_path = os.path.abspath(os.path.join(dali_base_path, "f0_v2.0", "f0_tismir", f"{dali_id}.f0.npz"))
    f0_matrix, f0_freqs, f0_time_r = None, None, None
    if os.path.exists(f0_npz_path):
        try:
            f0_data = np.load(f0_npz_path)
            f0_matrix = f0_data['f0']
            f0_freqs = f0_data['freqs']
            f0_time_r = f0_data['time_r'].item() if 'time_r' in f0_data.files else 0.058049886621315196
            print(f"Successfully loaded continuous f0 curves from {f0_npz_path}")
        except Exception as e:
            print(f"Warning: Could not load continuous f0 curves from {f0_npz_path}: {e}")
            f0_matrix = None
    else:
        print(f"Warning: F0 curve file NOT FOUND at {f0_npz_path}")
        f0_matrix = None
        
    return f0_matrix, f0_freqs, f0_time_r

def get_continuous_f0(f0_matrix, f0_freqs, f0_time_r, start_time_sec, end_time_sec, chunk_start_sec):
    continuous_f0 = []
    if f0_matrix is not None and f0_freqs is not None and f0_time_r > 0:
        col_start = int((start_time_sec + chunk_start_sec) / f0_time_r)
        col_end = int((end_time_sec + chunk_start_sec) / f0_time_r)
        if col_end > col_start:
            note_f0_slice = f0_matrix[:, col_start:col_end]
            peaks = np.argmax(note_f0_slice, axis=0)
            continuous_f0 = f0_freqs[peaks].tolist()
    return continuous_f0

def add_pitch_bends_to_array(pitch_array, f0_points, midi_pitch, start_time_sec, end_time_sec):
    if len(f0_points) > 1:
        base_freq = 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))
        # Note: F0 curve maps to the *original* duration of the note (before any extension in UTAU)
        # OpenUtau positions curves by ticks. At 120BPM, 1ms is approximately 1 tick.
        # However, for accuracy: BPM defaults to 120 (480 tpb) -> 960 ticks / sec -> 0.96 ticks / ms.
        orig_length_ticks = (end_time_sec - start_time_sec) * 960.0
        step_ticks = orig_length_ticks / len(f0_points)
        start_tick = int(start_time_sec * 960.0)
        
        for i_f, f in enumerate(f0_points):
            if f > 0: # Only compute bend for valid frequencies
                cents = 1200.0 * math.log2(f / base_freq)
                cents = max(-1200.0, min(1200.0, cents))  # Bound between 1 octave
                pt_tick = int(round(start_tick + i_f * step_ticks))
                
                # Store in dense array at 5-tick resolution
                idx = int(pt_tick // 5)
                if idx >= len(pitch_array):
                    # Extend array
                    pitch_array.extend([0] * (idx - len(pitch_array) + 1))
                pitch_array[idx] = int(round(cents))
                
        # Fill missing/zero values in the note interval with previous centroid by forward-filling
        start_idx = int(start_tick // 5)
        end_idx = int((start_tick + orig_length_ticks) // 5)
        if end_idx >= len(pitch_array):
            pitch_array.extend([0] * (end_idx - len(pitch_array) + 1))
        
        last_val = 0
        for i in range(start_idx, end_idx + 1):
            if pitch_array[i] != 0:
                last_val = pitch_array[i]
            else:
                pitch_array[i] = last_val
