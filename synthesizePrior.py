import os
import sys
import math
import pythonnet
pythonnet.load("coreclr")
import clr
import numpy as np

# Set up DALI import path
try:
    import DALI as dali_code
except ImportError:
    dali_code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DALI", "code"))
    sys.path.append(dali_code_path)
    try:
        import DALI as dali_code
    except ImportError:
        print("Error: Could not import DALI dataset library. Please ensure you have installed it or it is in the PYTHONPATH.")
        sys.exit(1)

# Set up UtauGenerate import
dependency_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "API", "UtauGenerate", "bin", "Debug", "net9.0", "UtauGenerate.dll")
if not os.path.exists(dependency_dir):
    print(f"Error: Could not find UtauGenerate at {dependency_dir}")
    sys.exit(1)

clr.AddReference(dependency_dir)
from UtauGenerate import Player
import System

def freq_to_midi(freq):
    """Convert frequency in Hz to closest MIDI note number (0-127)."""
    if freq <= 0:
        return 0
    return int(round(69 + 12 * math.log2(freq / 440.0)))

def process_dali_to_ustx(output_dir=None, use_continuations=True, dali_id="006b5d1db6a447039c30443310b60c6f", mode="paragraph", n_lines=4, use_f0=False):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
    
    os.makedirs(output_dir, exist_ok=True)
    # 1. Load the DALI dataset entry directly to bypass DALI library pathing bugs on Windows
    dali_data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DALI", "DALI_v2.0", "annot_tismir", f"{dali_id}.gz"))
    
    print(f"Loading DALI dataset entry from {dali_data_file} ...")
    try:
        # get_an_entry directly loads a .gz file without going through the broken Windows path splitting
        target_entry = dali_code.get_an_entry(dali_data_file)
    except Exception as e:
        print(f"Error loading DALI entry: {e}")
        return
        
    if target_entry is None:
        print("Failed to load test song.")
        return
        
    print(f"Successfully loaded song: {target_entry.info['title']} by {target_entry.info['artist']}")

    # 3. Extract annotations
    # From vertical format, extract words level to establish groupings
    try:
        target_entry.vertical2horizontal() 
    except Exception:
        pass # Already horizontal
    
    notes_annot = target_entry.annotations['annot'].get('notes', [])
    words_annot = target_entry.annotations['annot'].get('words', [])
    lines_annot = target_entry.annotations['annot'].get('lines', [])
    paragraphs_annot = target_entry.annotations['annot'].get('paragraphs', [])
    
    # 3. Load continuous F0 curves from .f0.npz
    dali_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DALI"))
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
        
    if not notes_annot or not words_annot or not lines_annot or not paragraphs_annot:
        print("Missing required annotations (notes, words, lines, or paragraphs).")
        return
        
    print(f"Generating ustx sequences using mode: {mode}...")

    # Group notes based on the selected mode
    chunks = [] # each chunk is a list of note indices
    chunk_start_times = []
    chunk_names = []

    if mode == "test":
        for line_idx, line_info in enumerate(lines_annot):
            line_notes = []
            for idx, note_info in enumerate(notes_annot):
                if words_annot[note_info['index']]['index'] == line_idx:
                    line_notes.append(idx)
            if line_notes:
                chunks.append(line_notes)
                chunk_start_times.append(line_info['time'][0])
                chunk_names.append("line_test")
                break

    elif mode == "paragraph":
        for para_idx, para_info in enumerate(paragraphs_annot):
            para_notes = []
            for idx, note_info in enumerate(notes_annot):
                line_idx = words_annot[note_info['index']]['index']
                if lines_annot[line_idx]['index'] == para_idx:
                    para_notes.append(idx)
            if para_notes:
                chunks.append(para_notes)
                chunk_start_times.append(para_info['time'][0])
                chunk_names.append(f"paragraph_{para_idx}")

    elif mode == "line":
        for line_idx, line_info in enumerate(lines_annot):
            line_notes = []
            for idx, note_info in enumerate(notes_annot):
                if words_annot[note_info['index']]['index'] == line_idx:
                    line_notes.append(idx)
            if line_notes:
                chunks.append(line_notes)
                chunk_start_times.append(line_info['time'][0])
                chunk_names.append(f"line_{line_idx}")

    elif mode == "n-line":
        chunk_idx = 0
        for para_idx, para_info in enumerate(paragraphs_annot):
            para_lines = [line_idx for line_idx, line_info in enumerate(lines_annot) if line_info['index'] == para_idx]
            
            for i in range(0, len(para_lines), n_lines):
                group_lines = para_lines[i:i+n_lines]
                group_notes = []
                for idx, note_info in enumerate(notes_annot):
                    if words_annot[note_info['index']]['index'] in group_lines:
                        group_notes.append(idx)
                
                if group_notes:
                    chunks.append(group_notes)
                    chunk_start_times.append(lines_annot[group_lines[0]]['time'][0])
                    chunk_names.append(f"chunk_{chunk_idx}")
                    chunk_idx += 1

    for chunk_i, (note_indices, chunk_start_sec, chunk_name) in enumerate(zip(chunks, chunk_start_times, chunk_names)):
        
        # 4. Initialize Player
        player = Player("OpenUtau.Plugin.Builtin.ArpasingPlusPhonemizer")
        
        # Keep track of dense pitch bends at 5-tick resolution
        # pitch_array[i] will store cents for the 5-tick chunk i.
        # Initialize an arbitrarily large array or resize later; let's use a dynamic list.
        pitch_array = []
        notes_added = 0
        for idx in note_indices:
            note_info = notes_annot[idx]
            word_idx = note_info['index']
            word_info = words_annot[word_idx]

            word_text = word_info['text']
            
            # Check if the previous note had the *same* word index. If so, it's a follow-up syllable
            is_continuation = False
            if use_continuations and idx > 0 and notes_annot[idx-1]['index'] == word_idx:
                is_continuation = True

            if use_continuations:
                if is_continuation:
                    text = "+"
                else:
                    text = word_text
            else:
                text = "a"
                
            start_time_sec = note_info['time'][0] - chunk_start_sec
            end_time_sec = note_info['time'][1] - chunk_start_sec
            
            freq_list = note_info['freq']
            if isinstance(freq_list, (list, tuple)) and len(freq_list) > 0:
                avg_freq = sum(freq_list) / len(freq_list)
                midi_pitch = freq_to_midi(avg_freq)
            else:
                midi_pitch = 60 # Default middle C if parsing fails
                
            position_ms = int(start_time_sec * 1000)
            length_ms = int((end_time_sec - start_time_sec) * 1000)
            
            if use_continuations and idx < len(notes_annot) - 1:
                next_idx = idx + 1
                next_note = notes_annot[next_idx]
                if next_note['index'] == word_idx:
                    next_start_time_sec = next_note['time'][0] - chunk_start_sec
                    length_ms = int((next_start_time_sec - start_time_sec) * 1000)
            
            player.addNote(position_ms, length_ms, midi_pitch, text)
            
            # --- Add pitch bends based on DALI F0 curve ---
            # Try to fetch true continuous F0 from npz, otherwise fallback to note info
            continuous_f0 = []
            if f0_matrix is not None and f0_freqs is not None and f0_time_r > 0:
                col_start = int((start_time_sec + chunk_start_sec) / f0_time_r)
                col_end = int((end_time_sec + chunk_start_sec) / f0_time_r)
                if col_end > col_start:
                    note_f0_slice = f0_matrix[:, col_start:col_end]
                    peaks = np.argmax(note_f0_slice, axis=0)
                    continuous_f0 = f0_freqs[peaks].tolist()
            
            if len(continuous_f0) > 1:
                f0_points = continuous_f0
            elif isinstance(freq_list, (list, tuple)) and len(freq_list) > 1:
                f0_points = freq_list
            else:
                f0_points = []
            
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

            notes_added += 1

        # Use setPitchBend to apply all collected pitch curves
        if use_f0 and pitch_array:
            arr = System.Array[int](pitch_array)
            player.setPitchBend(arr, 5)

        if notes_added > 0:
            # 6. Export to ustx
            chunk_out_dir = os.path.join(output_dir, dali_id, chunk_name)
            os.makedirs(chunk_out_dir, exist_ok=True)


            #wav export
            segment_wav = os.path.join(chunk_out_dir, "segment.wav")
            os.makedirs(os.path.dirname(segment_wav), exist_ok=True)
            player.exportWav(segment_wav)

            #ustx export
            segment_ustx = os.path.join(chunk_out_dir, "segment.ustx")
            os.makedirs(os.path.dirname(segment_ustx), exist_ok=True)
            player.exportUstx(segment_ustx)

            

            if mode == "test":
                try:
                    from VisualizeSegment import plot_segment
                    vis_path = os.path.join(chunk_out_dir, "visualization.png")
                    
                    # Extract annot2vector melody curve with high granularity (5ms resolution)
                    time_res = 0.005
                    end_time_total = notes_annot[-1]['time'][1] + 10.0 # Buffer end of song
                    melody_vector = dali_code.annot2vector(notes_annot, end_time_total, time_res, type='melody')
                    
                    plot_segment(notes_annot, note_indices, chunk_start_sec, f0_matrix, f0_freqs, f0_time_r, vis_path, melody_vector=melody_vector, melody_time_res=time_res)
                    print(f"Generated test visualization at {vis_path}")
                except Exception as e:
                    print(f"Visualization generation failed: {e}")

            # Also synthesize audio for this segment
            # Note: We use the base API command since the Player python wrapper has issues with audio export
                
    print(f"Finished generating {mode} ustx files.")
        
if __name__ == "__main__":
    process_dali_to_ustx(use_continuations=False, mode="test")

#conda run -n data_synthesizer python synthesizePrior.py