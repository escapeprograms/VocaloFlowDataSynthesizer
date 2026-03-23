# DataSynthesizer Memory Palace

This document serves as the "memory palace" for the `DataSynthesizer` module. It describes the current up-to-date system details, architecture, and purpose of each file within this module. 

## Overview

The `DataSynthesizer` module is responsible for taking DALI dataset annotations (which include notes, words, lines, and paragraphs) and synthesizing audio using two different pipelines:
1. **Prior Synthesis** (`synthesizePrior.py`): Generates vocals via the OpenUtau API (usually outputting `prior.wav`).
2. **Post Synthesis** (`synthesizePost.py`): Generates vocals via the SoulX-Singer machine learning model (usually outputting `generated.wav`).

Finally, the module aligns the output of the Post synthesis to match the timing of the Prior synthesis using Dynamic Time Warping (DTW).

## File Details

### `synthesize.py`
Single-song driver — runs the complete pipeline for one DALI entry.
- Calls `process_dali_to_ustx`, `process_dali_to_soulx`, and `run_dtw_alignment` in sequence.
- SoulX-Singer subprocess is launched immediately (model loaded once per song).
- CLI args: `--dali_id`, `--mode`, `--vocoder`, `--use_phonemes`, `--use_continuations`, `--use_f0`, `--segmentation_mode`.
- Use conda env `data_synthesizer` for this file.

### `synthesize_dataset.py`
Dataset-scale driver for processing all English DALI entries efficiently.  This is the preferred entry point for full dataset runs.
- **Three-phase pipeline** that loads the SoulX-Singer model once per batch of songs rather than once per song:
  - **Phase 1 — Annotation**: Calls `process_dali_to_ustx` + `process_dali_to_soulx(defer_inference=True)` for every song. Writes all `prior.wav` and `music.json` files. Saves the combined inference task list to `pending_inference_tasks.json` in the output dir.
  - **Phase 2 — Inference**: Reads the task cache and splits it into batches of `songs_per_batch` songs. Each batch launches one `soulxsinger_batch_infer.py` subprocess — the model loads once and processes all chunks in the batch. Skips already-completed chunks automatically.
  - **Phase 3 — Alignment**: Calls `run_dtw_alignment` for every song, writing `alignment.json` per chunk.
- `--phases` flag (e.g. `'123'`, `'2'`, `'13'`) allows resuming any individual phase independently.
- `--dali_ids` allows processing a subset of songs.
- `get_english_dali_ids(dali_annot_dir)`: scans the DALI annotation directory and returns IDs whose `metadata.language == 'english'` (5,913 of 7,756 total).
- Internal helpers: `run_phase1_annotation`, `run_phase2_inference`, `run_phase3_alignment`, `_launch_inference_subprocess`.

### `synthesizePrior.py`
Generates vocal synthesis using the OpenUtau API.
- Loads DALI dataset annotations directly from `.gz` files.
- Groups notes into chunks using `determine_chunks.py`.
- Uses `pythonnet` (`clr`) to interface with `UtauGenerate.dll` (a C# wrapper for OpenUtau).
- Adds notes and pitch bends (extracting continuous F0 via `grab_f0.py`) into the OpenUtau `Player`.
- **Optimized**: Reuses a single `Player` instance and calls `resetParts()` before adding notes for each chunk to prevent dictionary collisions (G2P Key errors).
- Exports `prior.wav` and `prior.ustx` for each chunk.
- If run in `test` mode, it also calls `VisualizeSegment.py` to generate a visualization of the chunk's pitch curves.

### `synthesizePost.py`
Generates vocal synthesis using the SoulX-Singer model.
- `get_soulx_inference_config()`: Module-level function returning the dict of model/config/prompt paths for the SoulX-Singer subprocess. Single source of truth used by both `process_dali_to_soulx` and `synthesize_dataset.py`.
- `process_dali_to_soulx(...)`: Parses DALI annotations → `music.json` per chunk, optionally injects F0 curves.
  - `defer_inference=False` (default): immediately launches `soulxsinger_batch_infer.py` subprocess (per-song use).
  - `defer_inference=True`: skips the subprocess and **returns** the list of inference task dicts for the caller to batch across songs.
- Supports `save_mel=True` to also export `generated_mel.npy` alongside `generated.wav`.

### `time_align.py`
Performs time alignment between a generated audio file and a reference audio file.
- Uses `librosa` to compute the Dynamic Time Warping (DTW) cost between the Chromagrams of two audio signals.
- Warps the time scale of the first audio file (e.g. `generated.wav`) to closely match the timing of the second (e.g. `prior.wav`).
- Interpolates the signal and saves a newly aligned `.wav` file, returning the DTW cost.

### `segmented_dtw.py`
Implements a "Slice and Stitch" segmented Dynamic Time Warping (DTW) approach to align prior and post synthesized audio.
- `extract_features_and_timings`: Parses timing data and extracts mel-spectrograms. Can load a pre-extracted `post_mel.npy` directly if provided.
- `convert_timestamps_to_frames`: Converts real-world timestamps into matrix column (frame) indices.
- `align_phonemes_dtw`: Performs padded micro-DTW iteratively for each phoneme/word.
- `stitch_warped_slices`: Reassembles warped slices onto the prior timeline, preserving silence gaps.
- `_write_alignment_meta`: Helper that writes `alignment.json` to a chunk directory.
- `align_and_export_mel`: Main wrapper. **Always saves `aligned.wav` and `alignment.json`** regardless of DTW cost — `cost_threshold` is now a flag/label only, not a gate. `alignment.json` records `mean_dtw_cost`, `max_dtw_cost`, `per_phoneme_costs`, `under_threshold`, `mfa_prior_ok`, `mfa_post_ok`, `aligned_saved`, and `timestamp` for later dataset filtering.

### `soulxsinger_batch_infer.py`
Subprocess entry point that loads the SoulX-Singer model once and runs inference for a list of chunk tasks.
- Reads a `--tasks_json` file (list of dicts with `target_metadata_path`, `save_dir`, `control`, `save_mel`).
- **Resumable**: skips any chunk whose expected output (`generated_mel.npy` or `generated.wav`) already exists on disk.
- Called by both `process_dali_to_soulx` (per-song) and `synthesize_dataset.py` (dataset batches).
- Must be run under the `soulxsinger` conda environment.

### `determine_chunks.py`
A chunking utility for segmenting a song based on DALI annotations.
- Function `get_chunks` takes the complete sets of `notes`, `words`, `lines`, and `paragraphs` and clusters the notes into manageable processing segments.
- Supports different grouping granularities: `test` (first line only), `line` (each line individually), `paragraph` (stanza by stanza), and `n-line` (blocks of `N` lines).

### `mfa_align.py`
A modular wrapper executing the Montreal Forced Aligner (MFA) to automatically generate `generated.TextGrid` files dynamically representing the actual vocal timing constraints.
- `align_audio_text`: Wrapper for a single file alignment.
- `batch_align_audio_text`: **New**: Efficiently aligns multiple audio/text pairs in one MFA subprocess call.
- `ensure_models_downloaded`: Handles model dependencies.
- Now includes timestamped logging for performance tracking.

## Configuration & Usage
### Dependencies
Ensure prerequisites like Python, Conda, `librosa`, `mido`, `soundfile`, `pyworld`, and the `tgt` (TextGrid parser) python packages are available. 
**For MFA textgrid extraction:** This toolkit relies on an external MFA isolated Conda environment. Before synthesizing, ensure you create this environment via `conda create -n vocaloflow-mfa -c conda-forge montreal-forced-aligner -y`. 

### Running Scripts

**Single song** (development / testing):
```bash
python synthesize.py --dali_id <id> --mode line --use_phonemes
```

**Full English dataset** (production):
```bash
python synthesize_dataset.py --phases 123 --songs_per_batch 100 --use_phonemes
```

**Resume a specific phase** (e.g. inference only, after a crash):
```bash
python synthesize_dataset.py --phases 2 --songs_per_batch 50
```

Key Arguments (shared across both scripts):
- `--mode`: `[line, n-line, paragraph, test]` — chunk boundary granularity.
- `--segmentation_mode`: DTW granularity. `word` warps whole syllables; `phoneme` warps frame-by-frame. Default: `word`.
- `--use_continuations`: Extend note durations to fill intra-word gaps (does not affect phoneme hints).
- `--use_phonemes`: Inject ARPAbet phoneticHints via g2p_en to fix cross-note mispronunciation.
- `--use_f0`: Use DALI F0 curves for pitch prompting instead of flat MIDI.
- `--songs_per_batch` *(dataset only)*: Songs per SoulX-Singer subprocess (default 100 ≈ 5200 chunks). Controls model-load amortisation vs. crash recovery window.

### `grab_f0.py`
Handles fetching and interpreting fine-grained fundamental frequency (F0) tracking data.
- `load_f0_data`: Loads a continuous `.f0.npz` matrix for a DALI entry.
- `get_continuous_f0`: Slices the F0 data to extract the active pitch curve across a specific chronological and note boundaries.
- `add_pitch_bends_to_array`: Translates raw frequency data arrays into a dense array of OpenUtau pitch bends (cents at 5-tick intervals) so it can be passed over to `UtauGenerate`.

### `grab_midi.py`
A small helper for converting frequencies into MIDI notation.
- `freq_to_midi`: Converts a raw Hz frequency into an integer MIDI note number.
- `get_midi_pitch`: Returns the closest MIDI pitch for an average over a list of frequencies.

### `VisualizeSegment.py`
Graphing utility for debugging and verifying alignment of annotations vs. extracted F0.
- `plot_segment`: Uses `matplotlib` to plot the horizontal baseline MIDI notes alongside extracted continuous `.npz` F0 curves and the DALI `annot2vector` generated melody curve.
- Allows for visual inspection of whether the pitch bends line up properly with the intended base lyrics.
