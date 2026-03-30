# Pipelines Memory Palace

This directory contains the **entry-point scripts** for the DataSynthesizer module. These are the files you actually run to kick off synthesis.

## Overview

There are two pipeline versions (v1 and v2), each with a single-song driver and a dataset-scale driver:

| Script | Version | Scale | Description |
|--------|---------|-------|-------------|
| `synthesize.py` | v1 (DEPRECATED) | Single song | Prior-first: generates prior and target independently, warps via segmented DTW |
| `synthesize_v2.py` | v2 | Single song | Target-first: generates target first, extracts notes via ROSVOT, iteratively aligns prior |
| `synthesize_dataset.py` | v1 (DEPRECATED) | Dataset | Three-phase pipeline with MFA + segmented DTW |
| `synthesize_dataset_v2.py` | v2 | Dataset | Five-phase pipeline: target-first with iterative alignment + manifest generation |

## File Details

### `synthesize.py` (v1, DEPRECATED)
Single-song driver — runs the complete v1 pipeline for one DALI entry. Use `synthesize_v2.py` instead.
- Calls `process_dali_to_ustx`, `process_dali_to_soulx`, and `run_dtw_alignment` in sequence.
- SoulX-Singer subprocess is launched immediately (model loaded once per song).
- CLI args: `--dali_id`, `--mode`, `--vocoder`, `--use_phonemes`, `--use_continuations`, `--use_f0`, `--segmentation_mode`.
- Use conda env `soulxsinger` for this file (the `data_synthesizer` env lacks torch, which is transitively required).

### `synthesize_v2.py` (v2)
Single-song driver — runs the target-first v2 pipeline for one DALI entry.
- **Four-phase pipeline** (all in one script, subprocesses for GPU phases):
  - **Phase 1**: Generate `music.json` + `chunk_words.json` from DALI annotations.
  - **Phase 2**: SoulX-Singer inference (subprocess in `soulxsinger` env).
  - **Phase 3**: ROSVOT note extraction + F0 (subprocess via `batch/note_extraction_batch.py`).
  - **Phase 4**: Prior generation from extracted notes via OpenUtau + iterative alignment until DTW convergence.
- Helper functions: `extract_chunk_words`, `save_chunk_words` (also used by `synthesize_dataset_v2.py`).
- CLI args: `--dali_id`, `--mode`, `--use_phonemes`, `--use_continuations`, `--use_f0`.

### `synthesize_dataset.py` (v1, DEPRECATED)
Dataset-scale driver for the v1 pipeline. Use `synthesize_dataset_v2.py` instead.
- **Three-phase pipeline** that loads the SoulX-Singer model once per batch:
  - **Phase 1 — Annotation**: Calls `process_dali_to_ustx` + `process_dali_to_soulx(defer_inference=True)` for every song. Writes `prior.wav` and `music.json`. Saves task list to `pending_inference_tasks.json`.
  - **Phase 2 — Inference**: Reads task cache, splits into batches of `songs_per_batch` songs. Each batch launches one `batch/soulxsinger_batch_infer.py` subprocess.
  - **Phase 3 — Alignment**: Calls `run_dtw_alignment` for every song.
- `--phases` flag (e.g. `'123'`, `'2'`, `'13'`) allows resuming individual phases.
- `get_english_dali_ids(dali_annot_dir)`: scans DALI annotations and returns IDs where `metadata.language == 'english'`.
- Internal helpers: `run_phase1_annotation`, `run_phase2_inference`, `run_phase3_alignment`, `_launch_inference_subprocess`.

### `synthesize_dataset_v2.py` (v2)
Dataset-scale driver for the target-first v2 pipeline.
- **Batch-oriented architecture**: Songs are split into batches of `songs_per_batch` (default 100). Each batch completes Phases 1–4 fully before the next batch starts, so you get complete entries early without waiting for the entire dataset to finish inference.
- **Five-phase pipeline** amortising all expensive model loads:
  - **Phase 1 — Target Metadata**: Generate `music.json` + `chunk_words.json` from DALI. Collect inference tasks. Merges into cumulative `pending_inference_tasks.json` cache.
  - **Phase 2 — Inference**: SoulX-Singer inference for this batch (reuses v1 `_launch_inference_subprocess`). If Phase 1 was not run in this invocation, loads and filters the cumulative cache.
  - **Phase 3 — Extraction**: ROSVOT note extraction + F0 via `batch/note_extraction_batch.py` subprocess for this batch.
  - **Phase 4 — Iterative Alignment**: Generate prior from `extracted_notes.json` via OpenUtau, iteratively adjusting note durations until DTW convergence. OpenUtau Player is initialised once before the batch loop and reused. Accepts optional `player` parameter.
  - **Phase 5 — Manifest**: Generate `manifest.csv` for ML training. Runs once after all batches complete.
- `--phases` accepts `'12345'`, any subset. Each selected phase runs per-batch (except Phase 5). All phases independently resumable via sentinel files.
- CLI args include `--max_iterations` (default 3) and `--duration_threshold` (default 0.15) for alignment tuning.
- Imports batch infrastructure from `synthesize_dataset.py`.

## Running Scripts

```bash
# v1 — Single song
conda run -n soulxsinger python pipelines/synthesize.py --dali_id <id> --mode line --use_phonemes

# v2 — Single song (post-first)
conda run -n soulxsinger python pipelines/synthesize_v2.py --dali_id <id> --mode line

# v1 — Full English dataset
conda run -n soulxsinger python pipelines/synthesize_dataset.py --phases 123 --songs_per_batch 100 --use_phonemes

# v2 — Full English dataset (post-first)
conda run -n soulxsinger python pipelines/synthesize_dataset_v2.py --phases 12345 --songs_per_batch 100

# Resume a specific phase
conda run -n soulxsinger python pipelines/synthesize_dataset_v2.py --phases 34 --songs_per_batch 50
```
