# Stages Memory Palace

This directory contains the **core pipeline stages** — the major processing steps called by the pipeline entry points in `pipelines/`.

## Overview

Each file represents one major step in the synthesis pipeline:
1. **Post synthesis** (`synthesizePost.py`): Generate vocals via SoulX-Singer.
2. **Prior synthesis** (`synthesizePrior.py`): Generate vocals via OpenUtau API.
3. **DTW orchestration** (`synthesizeDTW.py`, v1 only): Coordinate MFA + segmented DTW alignment.

## File Details

### `synthesizePost.py`
Generates vocal synthesis using the SoulX-Singer model.
- `get_soulx_inference_config()`: Module-level function returning the dict of model/config/prompt paths for the SoulX-Singer subprocess. Single source of truth used by both `process_dali_to_soulx` and `pipelines/synthesize_dataset.py`.
- `process_dali_to_soulx(...)`: Parses DALI annotations into `music.json` per chunk, optionally injects F0 curves.
  - `defer_inference=False` (default): immediately launches `batch/soulxsinger_batch_infer.py` subprocess (per-song use).
  - `defer_inference=True`: skips the subprocess and **returns** the list of inference task dicts for the caller to batch across songs.
- Supports `save_mel=True` to also export `generated_mel.npy` alongside `generated.wav`.
- Imports: `utils.grab_midi`, `utils.determine_chunks`.

### `synthesizePrior.py`
Generates vocal synthesis using the OpenUtau API.
- `process_dali_to_ustx(...)`: v1 entry point. Loads DALI annotations directly from `.gz` files, groups notes into chunks using `utils/determine_chunks.py`, adds notes and pitch bends into the OpenUtau `Player`. Exports `prior.wav` and `prior.ustx` for each chunk.
- `generate_prior_from_notes(chunk_dir, extracted_notes_path, player, use_phonemes)`: v2 entry point. Reads `extracted_notes.json` (from ROSVOT), groups notes by word, applies g2p phoneme distribution (`_distribute_phonemes`), converts to OpenUtau ticks, exports `prior.wav` + `prior.ustx`.
- Uses `pythonnet` (`clr`) to interface with `UtauGenerate.dll` (a C# wrapper for OpenUtau).
- **Optimized**: Reuses a single `Player` instance and calls `resetParts()` before adding notes for each chunk.
- Imports: `utils.grab_midi`, `utils.grab_f0`, `utils.determine_chunks`.

### `synthesizeDTW.py` (v1 only)
Orchestrates MFA forced alignment and segmented DTW for all chunks of a DALI entry. The v2 pipeline uses `alignment/iterative_align.py` directly instead.
- `run_dtw_alignment(dali_id, output_dir, mode, segmentation_mode, vocoder, align_to)`: Main entry point.
  1. Discovers chunks with `prior.wav`, `generated.wav`, and `music.json`.
  2. Runs batch MFA on prior and post audio via `alignment/mfa_align.py` subprocess.
  3. Runs `alignment/segmented_dtw.align_and_export_mel` per chunk.
- `run_batch_mfa(tasks, label, conda_exe, mfa_script)`: Runs MFA alignment in batch.
- `_write_mfa_failure_meta(chunk_dir, ...)`: Writes `alignment.json` with null costs when MFA fails.
- `align_to` parameter: `"prior"` (v1) warps post onto prior; `"post"` (v2) warps prior onto post.
