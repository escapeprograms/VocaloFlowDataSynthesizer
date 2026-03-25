# Batch Memory Palace

This directory contains **subprocess entry points** — scripts that are launched as separate processes (typically in the `soulxsinger` conda environment) to run GPU-intensive model inference. They are never imported directly; instead they are invoked via `subprocess.run()` from pipeline or stage scripts.

## File Details

### `soulxsinger_batch_infer.py`
Subprocess entry point that loads the SoulX-Singer model once and runs inference for a list of chunk tasks.
- Reads a `--tasks_json` file (list of dicts with `target_metadata_path`, `save_dir`, `control`, `save_mel`).
- **Resumable**: skips any chunk whose expected output (`generated_mel.npy` or `generated.wav`) already exists on disk.
- Called by both `stages/synthesizePost.py` (per-song) and `pipelines/synthesize_dataset.py` (dataset batches).
- Must be run under the `soulxsinger` conda environment.

### `note_extraction_batch.py` (v2 only)
Subprocess entry point that loads ROSVOT + RMVPE once and extracts note timings/pitches + F0 from post audio.
- Reads a `--tasks_json` file (list of dicts with `chunk_dir`, `audio_path`, `item_name`, `words`).
- `build_extracted_notes(rosvot_out, words)`: Converts ROSVOT output into `extracted_notes.json` format. Maps DALI lyrics onto ROSVOT-detected notes via `_distribute_words_to_notes` (proportional fallback when `note2words` is unavailable). Applies intra-word note stretching and filters unvoiced notes (pitch=0).
- **Resumable**: skips chunks where both `extracted_notes.json` and `post_f0.npy` exist.
- Must be run under the `soulxsinger` conda environment.
