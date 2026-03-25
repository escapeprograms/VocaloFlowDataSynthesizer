# Pipelines Memory Palace

This directory contains the **entry-point scripts** for the DataSynthesizer module. These are the files you actually run to kick off synthesis.

## Overview

There are two pipeline versions (v1 and v2), each with a single-song driver and a dataset-scale driver:

| Script | Version | Scale | Description |
|--------|---------|-------|-------------|
| `synthesize.py` | v1 | Single song | Prior-first: generates prior and post independently from DALI, warps post onto prior's timeline |
| `synthesize_v2.py` | v2 | Single song | Post-first: generates post first, extracts notes via ROSVOT, generates prior from extracted notes |
| `synthesize_dataset.py` | v1 | Dataset | Three-phase pipeline amortising SoulX-Singer model load across batches |
| `synthesize_dataset_v2.py` | v2 | Dataset | Five-phase pipeline, post-first, with batch ROSVOT extraction |

## File Details

### `synthesize.py` (v1)
Single-song driver — runs the complete v1 pipeline for one DALI entry.
- Calls `process_dali_to_ustx`, `process_dali_to_soulx`, and `run_dtw_alignment` in sequence.
- SoulX-Singer subprocess is launched immediately (model loaded once per song).
- CLI args: `--dali_id`, `--mode`, `--vocoder`, `--use_phonemes`, `--use_continuations`, `--use_f0`, `--segmentation_mode`.
- Use conda env `data_synthesizer` for this file.

### `synthesize_v2.py` (v2)
Single-song driver — runs the post-first v2 pipeline for one DALI entry.
- **Five-phase pipeline** (all in one script, subprocesses for GPU phases):
  - **Phase 1**: Generate `music.json` + `chunk_words.json` from DALI annotations.
  - **Phase 2**: SoulX-Singer inference (subprocess in `soulxsinger` env).
  - **Phase 3**: ROSVOT note extraction + F0 (subprocess via `batch/note_extraction_batch.py`).
  - **Phase 4**: Prior generation from extracted notes via OpenUtau (`generate_prior_from_notes`) + iterative alignment.
  - **Phase 5**: MFA + DTW alignment with `align_to="post"`.
- Helper functions: `extract_chunk_words`, `save_chunk_words` (also used by `synthesize_dataset_v2.py`).
- CLI args: same as v1.

### `synthesize_dataset.py` (v1)
Dataset-scale driver for the v1 pipeline.
- **Three-phase pipeline** that loads the SoulX-Singer model once per batch:
  - **Phase 1 — Annotation**: Calls `process_dali_to_ustx` + `process_dali_to_soulx(defer_inference=True)` for every song. Writes `prior.wav` and `music.json`. Saves task list to `pending_inference_tasks.json`.
  - **Phase 2 — Inference**: Reads task cache, splits into batches of `songs_per_batch` songs. Each batch launches one `batch/soulxsinger_batch_infer.py` subprocess.
  - **Phase 3 — Alignment**: Calls `run_dtw_alignment` for every song.
- `--phases` flag (e.g. `'123'`, `'2'`, `'13'`) allows resuming individual phases.
- `get_english_dali_ids(dali_annot_dir)`: scans DALI annotations and returns IDs where `metadata.language == 'english'`.
- Internal helpers: `run_phase1_annotation`, `run_phase2_inference`, `run_phase3_alignment`, `_launch_inference_subprocess`.

### `synthesize_dataset_v2.py` (v2)
Dataset-scale driver for the post-first v2 pipeline.
- **Five-phase pipeline** amortising all expensive model loads across batches:
  - **Phase 1 — Post Metadata**: Generate `music.json` + `chunk_words.json` from DALI. Collect inference tasks.
  - **Phase 2 — Inference**: Batch SoulX-Singer inference (reuses v1 `_launch_inference_subprocess`).
  - **Phase 3 — Extraction**: Batch ROSVOT note extraction + F0 via `batch/note_extraction_batch.py` subprocess.
  - **Phase 4 — Prior Gen**: Generate `prior.wav` from `extracted_notes.json` via OpenUtau.
  - **Phase 5 — Alignment**: MFA + DTW with `align_to="post"`.
- `--phases` accepts `'12345'`, any subset. All phases independently resumable.
- Imports batch infrastructure from `synthesize_dataset.py`.

## Running Scripts

```bash
# v1 — Single song
python pipelines/synthesize.py --dali_id <id> --mode line --use_phonemes

# v2 — Single song (post-first)
python pipelines/synthesize_v2.py --dali_id <id> --mode line

# v1 — Full English dataset
python pipelines/synthesize_dataset.py --phases 123 --songs_per_batch 100 --use_phonemes

# v2 — Full English dataset (post-first)
python pipelines/synthesize_dataset_v2.py --phases 12345 --songs_per_batch 100

# Resume a specific phase
python pipelines/synthesize_dataset_v2.py --phases 34 --songs_per_batch 50
```
