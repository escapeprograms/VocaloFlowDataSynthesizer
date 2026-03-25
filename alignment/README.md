# Alignment Memory Palace

This directory contains all **alignment-related code** — Dynamic Time Warping (DTW), Montreal Forced Alignment (MFA), and iterative alignment approaches.

## Overview

Alignment is the step that reconciles timing differences between prior (OpenUtau) and post (SoulX-Singer) audio. Multiple approaches are available:

| File | Approach | Description |
|------|----------|-------------|
| `segmented_dtw.py` | Phoneme-level DTW | Slice-and-stitch mel-spectrogram warping per phoneme/word |
| `time_align.py` | Global DTW | Whole-audio chromagram-based DTW (simpler, less precise) |
| `iterative_align.py` | Iterative note adjustment | Adjusts source note durations instead of warping audio |
| `mfa_align.py` | Forced alignment | MFA wrapper for generating TextGrid timing boundaries |

## File Details

### `segmented_dtw.py`
Implements a "Slice and Stitch" segmented DTW approach to align prior and post synthesized audio.
- `extract_features_and_timings`: Parses timing data and extracts mel-spectrograms. Can load a pre-extracted `post_mel.npy` directly if provided.
- `convert_timestamps_to_frames`: Converts real-world timestamps into matrix column (frame) indices.
- `align_phonemes_dtw(... align_to="prior")`: Performs padded micro-DTW iteratively for each phoneme/word. The `align_to` parameter controls warping direction:
  - `"prior"` (v1 default): warps **post** mel slices onto **prior's** timeline.
  - `"post"` (v2): warps **prior** mel slices onto **post's** timeline.
- `stitch_warped_slices`: Reassembles warped slices onto the target timeline, preserving silence gaps.
- `_write_alignment_meta`: Helper that writes `alignment.json` to a chunk directory.
- `align_and_export_mel(... align_to="prior")`: Main wrapper. Always saves `aligned.wav` and `alignment.json` regardless of DTW cost.
- Imports: `utils.vocoders` for mel-spectrogram inversion.

### `time_align.py`
Performs global time alignment between two audio files.
- Uses `librosa` to compute DTW cost between chromagrams of two audio signals.
- Warps the time scale of the first audio to match the second.
- Interpolates the signal and saves a newly aligned `.wav` file, returning the DTW cost.
- No local imports — fully standalone.

### `iterative_align.py`
Iterative alignment: repeatedly adjusts UTAU note durations until prior timing converges to match post timing.
- Instead of warping audio after the fact, modifies the *source* (note durations) so OpenUtau naturally renders audio whose timing matches SoulX-Singer's output.
- Uses DTW (not MFA) to measure timing discrepancies — directly compares mel-spectrograms frame-by-frame.
- Algorithm: generate prior → compute DTW warp path → measure per-note time scale ratios → adjust durations → repeat until convergence.
- `iterative_align(chunk_dir, notes, post_audio_path, lyrics_text, player, ...)`: Main entry point. Returns `(adjusted_notes, metrics_dict)`.
- Imports: `utils.vocoders`, `stages.synthesizePrior` (lazy).

### `mfa_align.py`
Modular wrapper for the Montreal Forced Aligner (MFA) to generate TextGrid files.
- `align_audio_text`: Wrapper for a single file alignment.
- `batch_align_audio_text`: Efficiently aligns multiple audio/text pairs in one MFA subprocess call.
- `ensure_models_downloaded`: Handles model dependencies.
- Includes timestamped logging for performance tracking.
- Runs as a subprocess (called from `stages/synthesizeDTW.py` via conda in the `vocaloflow-mfa` environment).
