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

# Add the directory to PATH so native DLLs like onnxruntime.dll can be found by Pythonnet
bin_dir = os.path.dirname(dependency_dir)
native_dir = os.path.join(bin_dir, "runtimes", "win-x64", "native")
os.environ["PATH"] = bin_dir + os.pathsep + native_dir + os.pathsep + os.environ.get("PATH", "")

clr.AddReference(dependency_dir)
from UtauGenerate import Player
import System

from grab_midi import get_midi_pitch, freq_to_midi
from grab_f0 import load_f0_data, get_continuous_f0, add_pitch_bends_to_array
from determine_chunks import get_chunks

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
    f0_matrix, f0_freqs, f0_time_r = load_f0_data(dali_id, dali_base_path)
        
    if not notes_annot or not words_annot or not lines_annot or not paragraphs_annot:
        print("Missing required annotations (notes, words, lines, or paragraphs).")
        return
        
    print(f"Generating ustx sequences using mode: {mode}...")

    # Group notes based on the selected mode
    chunks, chunk_start_times, chunk_names = get_chunks(mode, notes_annot, words_annot, lines_annot, paragraphs_annot, n_lines=n_lines)

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
                text = note_info['text']
                
            start_time_sec = note_info['time'][0] - chunk_start_sec
            end_time_sec = note_info['time'][1] - chunk_start_sec
            
            freq_list = note_info['freq']
            midi_pitch = get_midi_pitch(freq_list)
                
            position_ms = int(start_time_sec * 1000)
            length_ms = int((end_time_sec - start_time_sec) * 1000)
            
            if use_continuations and idx < len(notes_annot) - 1:
                next_idx = idx + 1
                next_note = notes_annot[next_idx]
                if next_note['index'] == word_idx:
                    next_start_time_sec = next_note['time'][0] - chunk_start_sec
                    length_ms = int((next_start_time_sec - start_time_sec) * 1000)
            
            # Convert milliseconds to OpenUtau ticks (120BPM, 480 res = 0.96 multiplier)
            position_ticks = max(0, int(position_ms * 0.96))
            length_ticks = max(15, int(length_ms * 0.96))
            
            print(f"Adding note: text='{text}', pos={position_ticks}, len={length_ticks}, pitch={midi_pitch}")
            player.addNote(position_ticks, length_ticks, midi_pitch, text)
            
            # --- Add pitch bends based on DALI F0 curve ---
            # Try to fetch true continuous F0 from npz, otherwise fallback to note info
            continuous_f0 = get_continuous_f0(f0_matrix, f0_freqs, f0_time_r, start_time_sec, end_time_sec, chunk_start_sec)
            
            if len(continuous_f0) > 1:
                f0_points = continuous_f0
            elif isinstance(freq_list, (list, tuple)) and len(freq_list) > 1:
                f0_points = freq_list
            else:
                f0_points = []
            
            add_pitch_bends_to_array(pitch_array, f0_points, midi_pitch, start_time_sec, end_time_sec)

            notes_added += 1

        # Use setPitchBend to apply all collected pitch curves
        if use_f0 and pitch_array:
            arr = System.Array[int](pitch_array)
            player.setPitchBend(arr, 5)

        if notes_added > 0:
            # 7. Add notes to API and synthesize waveform
            print(f"Generating ustx sequences using mode: {mode}...")
            chunk_out_dir = os.path.join(output_dir, dali_id, chunk_name)
            os.makedirs(chunk_out_dir, exist_ok=True)

            #wav export
            segment_wav = os.path.join(chunk_out_dir, "prior.wav")
            os.makedirs(os.path.dirname(segment_wav), exist_ok=True)
            player.exportWav(segment_wav)

            import time
            time.sleep(1.0) # Hack to prevent OpenUtau C# thread race condition on Array.Clear()

            #ustx export
            segment_ustx = os.path.join(chunk_out_dir, "prior.ustx")
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
    process_dali_to_ustx(use_continuations=False)

#conda run -n data_synthesizer python synthesizePrior.py