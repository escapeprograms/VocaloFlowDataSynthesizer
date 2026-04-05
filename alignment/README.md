# Alignment Memory Palace

This directory contains **alignment-related code** — specifically the iterative alignment approach used to reconcile timing differences between prior (OpenUtau) and target (SoulX-Singer) audio.

## Overview

Alignment reconciles timing differences between prior and target audio. Rather than warping audio after the fact, the approach modifies *source note durations* so OpenUtau naturally renders audio whose timing matches SoulX-Singer's output.

| File | Approach | Description |
|------|----------|-------------|
| `iterative_align.py` | Iterative note adjustment | Adjusts source note durations instead of warping audio |

## File Details

### `iterative_align.py`
Iterative alignment: repeatedly adjusts UTAU note durations until prior timing converges to match target timing. Used by both single-song (`synthesize_v2.py`) and dataset (`synthesize_dataset_v2.py`) pipelines.
- Uses DTW (not MFA) to measure timing discrepancies — directly compares mel-spectrograms frame-by-frame.
- Algorithm: generate prior → compute DTW warp path → measure per-note time scale ratios → adjust durations → repeat until convergence.
- `iterative_align(chunk_dir, notes, post_audio_path, lyrics_text, player, ...)`: Main entry point. Returns `(adjusted_notes, metrics_dict)`.
  - `adjusted_notes` is a list of note dicts (same schema as `extracted_notes.json` notes) with `start_s` and `note_dur` refined through iterative DTW. Each iteration: render prior via OpenUtau → DTW against target mel → compute per-note time-scale ratios → apply damped correction (clamped [0.3, 3.0]). Converges when all ratios within `duration_threshold` or `max_iterations` hit.
  - `metrics_dict` contains final alignment quality info (iteration count, max deviation, convergence flag).
  - The caller (pipeline) writes the output as `adjusted_notes.json` with structure `{"notes": [...], "source": "iterative"}`.
- `_save_prior_mel(prior_path, chunk_dir, sr)`: Extracts mel from final prior.wav and saves as `prior_mel.npy` in **(T, 128)** shape (time-first, matching target_mel convention).
- Imports: `utils.vocoders`, `stages.synthesizePrior` (lazy).
