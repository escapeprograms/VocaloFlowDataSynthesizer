# DataSynthesizer Memory Palace

This document serves as the "memory palace" for the `DataSynthesizer` module. It describes the current up-to-date system details, architecture, and directory structure.

## Overview

The `DataSynthesizer` module is responsible for taking DALI dataset annotations (which include notes, words, lines, and paragraphs) and synthesizing audio using two different pipelines:
1. **Prior Synthesis** (`stages/synthesizePrior.py`): Generates vocals via the OpenUtau API (usually outputting `prior.wav`).
2. **Target Synthesis** (`stages/synthesizeTarget.py`): Generates vocals via the SoulX-Singer machine learning model (usually outputting `target.wav`).

Two pipeline versions exist:
- **v1** (`pipelines/synthesize.py` / `pipelines/synthesize_dataset.py`): Generates prior and target independently from DALI, then warps target onto prior's timeline (`align_to="prior"`). `aligned.wav` = target timbre, prior timing.
- **v2** (`pipelines/synthesize_v2.py` / `pipelines/synthesize_dataset_v2.py`): **Target-first pipeline**. Generates target first, extracts note structure from `music.json` with F0-derived pitches from RMVPE, generates prior from those notes, then iteratively adjusts note durations via DTW until the prior timing converges to match the target. No audio warping — the prior is regenerated until it naturally aligns.

## Directory Structure

```
DataSynthesizer/
├── pipelines/          # Entry points — what you run
│   ├── synthesize.py           # v1 single-song driver
│   ├── synthesize_v2.py        # v2 single-song driver (target-first)
│   ├── synthesize_dataset.py   # v1 dataset-scale driver
│   └── synthesize_dataset_v2.py# v2 dataset-scale driver
│
├── stages/             # Core pipeline steps (called by pipelines)
│   ├── synthesizeTarget.py     # SoulX-Singer vocal generation
│   ├── synthesizePrior.py      # OpenUtau vocal generation
│   └── synthesizeDTW.py        # DTW alignment orchestration
│
├── alignment/          # All alignment-related code
│   ├── segmented_dtw.py        # Phoneme-level slice-and-stitch DTW
│   ├── iterative_align.py      # Iterative note duration adjustment
│   ├── time_align.py           # Global chromagram-based DTW
│   └── mfa_align.py            # Montreal Forced Aligner wrapper
│
├── utils/              # Standalone helpers (no local deps)
│   ├── vocoders.py             # Mel-to-audio inversion (Griffin-Lim, HiFiGAN, Vocos)
│   ├── grab_f0.py              # F0 curve loading and pitch bend generation
│   ├── grab_midi.py            # Frequency-to-MIDI conversion
│   └── determine_chunks.py     # Song segmentation by line/paragraph/n-line
│
├── batch/              # Subprocess entry points (run in soulxsinger env)
│   ├── soulxsinger_batch_infer.py  # Batch SoulX-Singer inference
│   └── note_extraction_batch.py    # Batch note extraction (music.json + RMVPE F0)
│
└── scratch/            # Testing, debugging, exploration
    ├── VisualizeSegment.py     # Pitch curve visualization
    ├── test_hifigan.py         # Vocoder roundtrip testing
    ├── inspect_player.py       # OpenUtau Player introspection
    └── dali_test.ipynb         # DALI dataset exploration notebook
```

Each subdirectory has its own `README.md` memory palace with detailed per-file documentation.

## Configuration & Usage

### Dependencies
Ensure prerequisites like Python, Conda, `librosa`, `mido`, `soundfile`, `pyworld`, and the `tgt` (TextGrid parser) python packages are available.
**For MFA textgrid extraction:** This toolkit relies on an external MFA isolated Conda environment. Before synthesizing, ensure you create this environment via `conda create -n vocaloflow-mfa -c conda-forge montreal-forced-aligner -y`.

### Running Scripts

All scripts should be run from the `DataSynthesizer/` directory:

**v1 — Single song** (development / testing):
```bash
python pipelines/synthesize.py --dali_id <id> --mode line --use_phonemes
```

**v1 — Full English dataset** (production):
```bash
python pipelines/synthesize_dataset.py --phases 123 --songs_per_batch 100 --use_phonemes
```

**v2 — Single song** (target-first pipeline):
```bash
python pipelines/synthesize_v2.py --dali_id <id> --mode line
```

**v2 — Full English dataset** (target-first, production):
```bash
python pipelines/synthesize_dataset_v2.py --phases 1234 --songs_per_batch 100
```

**Resume a specific phase** (e.g. extraction + alignment only, after a crash):
```bash
python pipelines/synthesize_dataset_v2.py --phases 34 --songs_per_batch 50
```

Key Arguments (shared across scripts):
- `--mode`: `[line, n-line, paragraph, test]` — chunk boundary granularity.
- `--segmentation_mode`: DTW granularity. `word` warps whole syllables; `phoneme` warps frame-by-frame. Default: `word`.
- `--use_continuations`: Extend note durations to fill intra-word gaps (does not affect phoneme hints).
- `--use_phonemes`: Inject ARPAbet phoneticHints via g2p_en to fix cross-note mispronunciation.
- `--use_f0`: Use DALI F0 curves for pitch prompting instead of flat MIDI.
- `--songs_per_batch` *(dataset only)*: Songs per SoulX-Singer subprocess (default 100 ≈ 5200 chunks). Controls model-load amortisation vs. crash recovery window.

### Output Per Chunk

**v1 outputs:**
```
Data/<dali_id>/<chunk_name>/
  music.json          # SoulX metadata from DALI
  prior.wav           # Prior from OpenUtau (from DALI notes)
  prior.ustx          # OpenUtau project file
  target.wav          # Target from SoulX-Singer
  target_mel.npy      # Target mel-spectrogram (pre-vocoder)
  aligned.wav         # Target timbre warped onto prior's timeline
  alignment.json      # DTW quality metadata (align_to: "prior")
```

**v2 adds:**
```
  chunk_words.json     # DALI word texts for lyric mapping
  extracted_notes.json # Notes from music.json structure + F0-derived pitches
  target_f0.npy        # Frame-level F0 from target audio (RMVPE)
  prior.wav            # Prior from OpenUtau (iteratively aligned to target)
  prior_mel.npy        # Mel-spectrogram of the final prior
  adjusted_notes.json  # Final note durations after iterative alignment
  alignment.json       # Iterative alignment convergence metrics
```
