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

### `dali_test.ipynb`
Jupyter notebook for exploring the DALI dataset.
- Interactive exploration of DALI annotation structure, metadata, and song loading.
