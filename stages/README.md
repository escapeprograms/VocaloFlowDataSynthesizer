# Stages Memory Palace

This directory contains the **core pipeline stages** — the major processing steps called by the pipeline entry points in `pipelines/`.

## Overview

Each file represents one major step in the synthesis pipeline:
1. **Target synthesis** (`synthesizeTarget.py`): Generate vocals via SoulX-Singer.
2. **Prior synthesis** (`synthesizePrior.py`): Generate vocals via OpenUtau API.

## File Details

### `synthesizeTarget.py`
Generates vocal synthesis using the SoulX-Singer model.
- `get_soulx_inference_config(provider=None)`: Module-level function returning the dict of model/config/prompt paths for the SoulX-Singer subprocess. Looks up voice prompt paths from `voice_providers.VOICE_PROVIDERS` by provider name (defaults to `DEFAULT_PROVIDER`). Single source of truth used by both `process_dali_to_target` and `pipelines/synthesize_dataset_v2.py`.
- `process_dali_to_target(...)`: Parses DALI annotations into `music.json` per chunk, optionally injects F0 curves.
  - `defer_inference=False` (default): immediately launches `batch/soulxsinger_batch_infer.py` subprocess (per-song use).
  - `defer_inference=True`: skips the subprocess and **returns** the list of inference task dicts for the caller to batch across songs.
- Supports `save_mel=True` to also export `target_mel.npy` alongside `target.wav`.
- Imports: `utils.grab_midi`, `utils.determine_chunks`, `utils.prompt_selection`, `voice_providers`.

### `synthesizePrior.py`
Generates vocal synthesis using the OpenUtau API.
- `generate_prior_from_notes(chunk_dir, extracted_notes_path, player, use_phonemes)`: Reads `extracted_notes.json` (from ROSVOT), groups notes by word, applies g2p phoneme distribution (`_distribute_phonemes`), converts to OpenUtau ticks, exports `prior.wav` + `prior.ustx`.
- Uses `pythonnet` (`clr`) to interface with `UtauGenerate.dll` (a C# wrapper for OpenUtau).
- **Optimized**: Reuses a single `Player` instance and calls `resetParts()` before adding notes for each chunk.
- G2P helpers: `_normalize_arpabet`, `_distribute_phonemes` (coda-dominant syllabification), `_build_text_fallback`.
- Imports: `utils.grab_midi`.
