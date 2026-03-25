# Utils Memory Palace

This directory contains **standalone utility modules** with no local cross-dependencies. These are leaf-level helpers used by stages and pipelines.

## File Details

### `vocoders.py`
Vocoder integration for converting mel-spectrograms back to audio waveforms.
- `mel_to_soulx_mel(mel, config)`: Converts a standard librosa mel-spectrogram to SoulX-Singer format (log-compressed, transposed).
- `SOULX_MEL_CONFIG`: Default mel-spectrogram configuration dict (sr=22050, n_fft=1024, hop=256, n_mels=80).
- `invert_mel_to_audio(mel, config, vocoder)`: Dispatcher that routes to the appropriate vocoder.
- `invert_mel_to_audio_griffin_lim(mel, config)`: Griffin-Lim phase reconstruction (CPU, fast, lower quality).
- `invert_mel_to_audio_hifigan(mel, config)`: HiFIGAN neural vocoder (GPU, higher quality).
- `invert_mel_to_audio_soulxsinger(mel, config)`: SoulX-Singer's built-in Vocos vocoder (best quality for SoulX mel-spectrograms).
- Lazy-loads models on first call; caches globally.

### `grab_f0.py`
Handles fetching and interpreting fine-grained fundamental frequency (F0) tracking data.
- `load_f0_data(dali_id, dali_base_path)`: Loads a continuous `.f0.npz` matrix for a DALI entry.
- `get_continuous_f0(f0_matrix, f0_freqs, f0_time_r, start, end, notes)`: Slices the F0 data to extract the active pitch curve across specific chronological and note boundaries.
- `add_pitch_bends_to_array(f0_curve, f0_time_r, notes, start_time)`: Translates raw frequency data arrays into a dense array of OpenUtau pitch bends (cents at 5-tick intervals).

### `grab_midi.py`
Small helper for converting frequencies into MIDI notation.
- `freq_to_midi(freq)`: Converts a raw Hz frequency into an integer MIDI note number.
- `get_midi_pitch(freqs)`: Returns the closest MIDI pitch for an average over a list of frequencies.

### `determine_chunks.py`
Chunking utility for segmenting a song based on DALI annotations.
- `get_chunks(mode, notes, words, lines, paragraphs, n_lines)`: Takes the complete sets of DALI annotations and clusters notes into manageable processing segments.
- Supports grouping granularities: `test` (first line only), `line` (each line individually), `paragraph` (stanza by stanza), `n-line` (blocks of N lines).
- Returns `(chunks, chunk_start_times, chunk_names)`.
