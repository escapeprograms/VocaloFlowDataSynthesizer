# Scratch Memory Palace

This directory contains **testing, debugging, and exploration** scripts. These are not part of the production pipeline and can be safely ignored or deleted without affecting synthesis.

## File Details

### `VisualizeSegment.py`
Graphing utility for debugging and verifying alignment of annotations vs. extracted F0.
- `plot_segment`: Uses `matplotlib` to plot horizontal baseline MIDI notes alongside extracted continuous `.npz` F0 curves and the DALI `annot2vector` generated melody curve.
- Allows visual inspection of whether pitch bends line up properly with intended base lyrics.

### `test_hifigan.py`
Roundtrip test: load `generated.wav` -> extract mel-spectrogram -> reconstruct via HiFiGAN.
- Verifies HiFiGAN is working correctly before any DTW warping is applied.
- Tests Griffin-Lim, HiFiGAN, and SoulX-Singer Vocos vocoders side-by-side.

### `inspect_player.py`
Quick diagnostic script for inspecting the OpenUtau `Player` class interface via pythonnet.
- Lists available methods and properties on the `Player` object.

### `detect_silences.py`
Scans all chunks in a song data directory to find internal silences/rests in SoulX-Singer generated audio.
- Uses two complementary signals: F0 gaps (`post_f0.npy`, frames where F0=0) and energy gaps (RMS below -40dB threshold from `generated.wav`).
- `find_gaps(mask, margin_frames, min_frames)`: Finds contiguous True runs in a boolean mask, trimming edge margins. Returns gap dicts with frame indices and timestamps.
- `analyze_chunk(chunk_dir, ...)`: Loads F0 + audio for one chunk, computes RMS, detects F0 and energy gaps, cross-references to classify as `f0_only` (consonant transitions) vs `both` (true silence). Computes severity score.
- `print_report(results)`: Ranked text report sorted by severity. Shows gap timestamps, types, and which note/lyric the gap falls within.
- `plot_chunk(result, output_dir)`: Two-subplot matplotlib figure — waveform+RMS on top, F0+note boundaries on bottom. Gap regions shaded red (confirmed) or orange (f0-only). Saved as PNG at 150 DPI.
- `main()`: argparse CLI with `--data_dir`, `--min_gap_ms`, `--energy_threshold_db`, `--top_n`, `--output_dir`, `--no_plots`. Defaults to the single-song data directory.
- Outputs to `{data_dir}/silence_report/` (PNG plots).

### `dali_test.ipynb`
Jupyter notebook for exploring the DALI dataset.
- Interactive exploration of DALI annotation structure, metadata, and song loading.
