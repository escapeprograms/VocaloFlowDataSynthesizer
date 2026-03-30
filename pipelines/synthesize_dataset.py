"""
DEPRECATED — v1 dataset-scale synthesis pipeline.

Use synthesize_dataset_v2.py instead. This file is retained only for
reproducing legacy v1 results. Do not add new features here.

Original description:
Dataset-scale synthesis pipeline for all English DALI entries.

Runs the three-phase pipeline that amortises the SoulX-Singer model load
across large batches of songs instead of reloading it per song:

  Phase 1 — Annotation  : Generate prior.wav (OpenUTAU) + music.json (SoulX
                          annotation) for every song.  Fast; fully in-process.
  Phase 2 — Inference   : Run SoulX-Singer once per batch of songs.  The model
                          loads once per subprocess, covering ~100 songs worth
                          of chunks before the next batch starts.
  Phase 3 — Alignment   : Run MFA + segmented DTW for every song.

All three phases are independently resumable:
  - Phase 1: prior.wav presence is checked per chunk inside process_dali_to_ustx.
  - Phase 2: soulxsinger_batch_infer.py skips chunks whose output already exists.
  - Phase 3: alignment.json presence signals a completed chunk inside run_dtw_alignment.

Usage:
    python synthesize_dataset.py --phases 123 --songs_per_batch 100
    python synthesize_dataset.py --phases 2   --songs_per_batch 50   # resume inference only
    python synthesize_dataset.py --dali_ids <id1> <id2>              # subset of songs
"""

import argparse
import gzip
import json
import os
import pickle
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import List, Optional

# Add DataSynthesizer root to path for cross-package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stages.synthesizePrior import process_dali_to_ustx
from stages.synthesizeTarget import process_dali_to_target, get_soulx_inference_config
from stages.synthesizeDTW import run_dtw_alignment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOULX_PYTHON   = r"C:\Users\archi\miniconda3\envs\soulxsinger\python.exe"
SOULX_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SoulX-Singer"))
DALI_ANNOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DALI", "DALI_v2.0", "annot_tismir"))
DEFAULT_OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data"))

# File written at end of Phase 1 so Phase 2 can be re-run independently
TASKS_CACHE_NAME = "pending_inference_tasks.json"

# ---------------------------------------------------------------------------
# Phase 0 — Dataset discovery
# ---------------------------------------------------------------------------

def get_english_dali_ids(dali_annot_dir: str) -> List[str]:
    """Scan the DALI annotation directory and return IDs for all English songs."""
    ids = []
    for fname in sorted(os.listdir(dali_annot_dir)):
        if not fname.endswith(".gz"):
            continue
        try:
            with gzip.open(os.path.join(dali_annot_dir, fname), "rb") as fh:
                obj = pickle.load(fh)
            lang = obj.info.get("metadata", {}).get("language", "")
            if lang.lower() == "english":
                ids.append(fname.replace(".gz", ""))
        except Exception:
            pass  # Corrupt or unreadable entry — skip silently
    return ids

# ---------------------------------------------------------------------------
# Phase 1 — Annotation
# ---------------------------------------------------------------------------

def run_phase1_annotation(
    dali_ids: List[str],
    output_dir: str,
    mode: str,
    n_lines: int,
    use_f0: bool,
    use_continuations: bool,
    use_phonemes: bool,
    save_mel: bool,
) -> List[dict]:
    """Generate prior.wav and music.json for every song; collect SoulX task dicts.

    Returns the combined list of all inference task dicts across all songs.
    Each task dict is passed verbatim to soulxsinger_batch_infer.py later.
    """
    all_tasks: List[dict] = []
    total = len(dali_ids)

    for i, dali_id in enumerate(dali_ids):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] [Phase 1 | {i+1}/{total}] {dali_id}")
        try:
            process_dali_to_ustx(
                output_dir=output_dir,
                dali_id=dali_id,
                mode=mode,
                n_lines=n_lines,
                use_f0=use_f0,
                use_continuations=use_continuations,
                use_phonemes=use_phonemes,
            )
            tasks = process_dali_to_target(
                dali_id=dali_id,
                output_dir=output_dir,
                mode=mode,
                n_lines=n_lines,
                use_f0=use_f0,
                use_continuations=use_continuations,
                save_mel=save_mel,
                defer_inference=True,
            )
            if tasks:
                all_tasks.extend(tasks)
                print(f"  -> {len(tasks)} chunk(s) queued.")
        except Exception as e:
            print(f"  [Phase 1] ERROR for {dali_id}: {e}")

    return all_tasks

# ---------------------------------------------------------------------------
# Phase 2 — Inference helpers
# ---------------------------------------------------------------------------

def _launch_inference_subprocess(tasks: List[dict], infer_cfg: dict) -> None:
    """Write tasks to a temp JSON file and launch soulxsinger_batch_infer.py."""
    batch_script = os.path.join(os.path.dirname(__file__), "..", "batch", "soulxsinger_batch_infer.py")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tf:
        json.dump(tasks, tf)
        tasks_file = tf.name

    cmd = [
        SOULX_PYTHON, batch_script,
        "--tasks_json",            tasks_file,
        "--model_path",            infer_cfg["model_path"],
        "--config",                infer_cfg["config_path"],
        "--prompt_wav_path",       infer_cfg["prompt_wav_path"],
        "--prompt_metadata_path",  infer_cfg["prompt_metadata_path"],
        "--phoneset_path",         infer_cfg["phoneset_path"],
        "--device",                "cuda",
        "--auto_shift",
        "--pitch_shift",           "0",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = SOULX_DIR + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(cmd, env=env, cwd=SOULX_DIR, check=True)
        print(f"  Batch complete ({len(tasks)} tasks).")
    except subprocess.CalledProcessError as e:
        print(f"  Batch FAILED with return code {e.returncode}.")
    finally:
        os.unlink(tasks_file)


def run_phase2_inference(all_tasks: List[dict], songs_per_batch: int) -> None:
    """Split the task list into song-sized batches and run one subprocess per batch.

    songs_per_batch controls how many songs' chunks are grouped per subprocess
    invocation — i.e. how many times the 704M model is loaded.
    A value of 100 means ~5200 chunks per subprocess (100 songs × ~52 lines/song).
    Tasks that already have output on disk are skipped by soulxsinger_batch_infer.py.
    """
    if not all_tasks:
        print("[Phase 2] No tasks to run.")
        return

    infer_cfg = get_soulx_inference_config()

    # Approximate chunks per song from the task list itself when possible.
    # We group by the dali_id embedded in the save_dir path segment.
    def dali_id_from_task(task: dict) -> str:
        # save_dir is  <output_dir>/<dali_id>/<chunk_name>
        return os.path.basename(os.path.dirname(task["save_dir"]))

    # Build ordered list of unique song IDs preserving task order
    seen: dict = {}
    for task in all_tasks:
        did = dali_id_from_task(task)
        seen.setdefault(did, []).append(task)

    song_ids = list(seen.keys())
    total_batches = max(1, (len(song_ids) + songs_per_batch - 1) // songs_per_batch)

    print(f"\n[Phase 2] {len(all_tasks)} tasks across {len(song_ids)} songs "
          f"→ {total_batches} batch(es) of up to {songs_per_batch} songs each.")

    for batch_i in range(total_batches):
        batch_song_ids = song_ids[batch_i * songs_per_batch : (batch_i + 1) * songs_per_batch]
        batch_tasks = [task for sid in batch_song_ids for task in seen[sid]]
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] [Phase 2 | Batch {batch_i+1}/{total_batches}] "
              f"{len(batch_song_ids)} songs, {len(batch_tasks)} chunks...")
        _launch_inference_subprocess(batch_tasks, infer_cfg)

# ---------------------------------------------------------------------------
# Phase 3 — DTW alignment
# ---------------------------------------------------------------------------

def run_phase3_alignment(
    dali_ids: List[str],
    output_dir: str,
    mode: str,
    segmentation_mode: str,
    vocoder: str,
) -> None:
    """Run MFA + segmented DTW for every song. Writes alignment.json per chunk."""
    total = len(dali_ids)
    for i, dali_id in enumerate(dali_ids):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] [Phase 3 | {i+1}/{total}] {dali_id}")
        try:
            run_dtw_alignment(
                dali_id=dali_id,
                output_dir=output_dir,
                mode=mode,
                segmentation_mode=segmentation_mode,
                vocoder=vocoder,
            )
        except Exception as e:
            print(f"  [Phase 3] ERROR for {dali_id}: {e}")

# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def synthesize_dataset(
    output_dir: str = DEFAULT_OUTPUT,
    dali_annot_dir: str = DALI_ANNOT_DIR,
    mode: str = "line",
    n_lines: int = 4,
    use_f0: bool = False,
    use_continuations: bool = False,
    use_phonemes: bool = True,
    segmentation_mode: str = "word",
    vocoder: str = "soulxsinger",
    songs_per_batch: int = 100,
    phases: str = "123",
    dali_ids: Optional[List[str]] = None,
) -> None:
    """Run the full dataset-scale synthesis pipeline.

    Args:
        output_dir:       Root directory for all generated files.
        dali_annot_dir:   Path to DALI annot_tismir directory.
        mode:             Chunk granularity — 'line', 'n-line', 'paragraph', 'test'.
        n_lines:          Lines per chunk when mode='n-line'.
        use_f0:           Use F0 curves for pitch instead of flat MIDI.
        use_continuations:Extend note durations to fill intra-word gaps.
        use_phonemes:     Use ARPAbet phoneticHints via g2p_en.
        segmentation_mode:DTW granularity — 'word' or 'phoneme'.
        vocoder:          Vocoder for mel inversion — 'soulxsinger', 'hifigan', 'griffin_lim'.
        songs_per_batch:  Songs grouped per SoulX-Singer subprocess call.
        phases:           Which phases to run, e.g. '123', '2', '13'. Enables resuming.
        dali_ids:         If given, process only these IDs; otherwise all English entries.
    """
    tasks_cache = os.path.join(output_dir, TASKS_CACHE_NAME)
    os.makedirs(output_dir, exist_ok=True)

    # Resolve song list
    if dali_ids is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning for English DALI entries in {dali_annot_dir}...")
        dali_ids = get_english_dali_ids(dali_annot_dir)
        print(f"Found {len(dali_ids)} English entries.")

    _sep = "=" * 62

    # ---- Phase 1 ----
    if "1" in phases:
        print(f"\n{_sep}")
        print(f"  PHASE 1 — Annotation  ({len(dali_ids)} songs)")
        print(_sep)
        all_tasks = run_phase1_annotation(
            dali_ids, output_dir, mode, n_lines,
            use_f0, use_continuations, use_phonemes, save_mel=True,
        )
        with open(tasks_cache, "w", encoding="utf-8") as f:
            json.dump(all_tasks, f)
        print(f"\n[Phase 1] Complete — {len(all_tasks)} inference tasks cached to {tasks_cache}")

    # ---- Phase 2 ----
    if "2" in phases:
        print(f"\n{_sep}")
        print(f"  PHASE 2 — SoulX Inference")
        print(_sep)
        if not os.path.exists(tasks_cache):
            raise FileNotFoundError(
                f"Task cache not found at {tasks_cache}. Run Phase 1 first, "
                "or pass --phases 12 to run both together."
            )
        with open(tasks_cache, "r", encoding="utf-8") as f:
            all_tasks = json.load(f)
        run_phase2_inference(all_tasks, songs_per_batch=songs_per_batch)

    # ---- Phase 3 ----
    if "3" in phases:
        print(f"\n{_sep}")
        print(f"  PHASE 3 — DTW Alignment  ({len(dali_ids)} songs)")
        print(_sep)
        run_phase3_alignment(dali_ids, output_dir, mode, segmentation_mode, vocoder)

    print(f"\n{_sep}")
    print(f"  Dataset synthesis complete.")
    print(_sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset-scale DALI synthesis pipeline (Phase 1 + 2 + 3)."
    )
    parser.add_argument("--output_dir",       default=DEFAULT_OUTPUT)
    parser.add_argument("--dali_annot_dir",   default=DALI_ANNOT_DIR)
    parser.add_argument("--mode",             default="line",
                        choices=["line", "n-line", "paragraph", "test"])
    parser.add_argument("--n_lines",          type=int, default=4)
    parser.add_argument("--use_f0",           action="store_true")
    parser.add_argument("--use_continuations",action="store_true")
    parser.add_argument("--use_phonemes",     action="store_true", default=True)
    parser.add_argument("--segmentation_mode",default="word", choices=["word", "phoneme"])
    parser.add_argument("--vocoder",          default="soulxsinger",
                        choices=["soulxsinger", "hifigan", "griffin_lim"])
    parser.add_argument("--songs_per_batch",  type=int, default=100,
                        help="Songs grouped per SoulX-Singer subprocess (default: 100).")
    parser.add_argument("--phases",           default="123",
                        help="Which phases to run, e.g. '123', '2', '13' (default: 123).")
    parser.add_argument("--dali_ids",         nargs="*", default=None,
                        help="Optional subset of DALI IDs to process.")

    args = parser.parse_args()
    args.use_continuations = True
    args.mode = "line"
    args.use_phonemes = True

    synthesize_dataset(
        output_dir=args.output_dir,
        dali_annot_dir=args.dali_annot_dir,
        mode=args.mode,
        n_lines=args.n_lines,
        use_f0=args.use_f0,
        use_continuations=args.use_continuations,
        use_phonemes=args.use_phonemes,
        segmentation_mode=args.segmentation_mode,
        vocoder=args.vocoder,
        songs_per_batch=args.songs_per_batch,
        phases=args.phases,
        dali_ids=args.dali_ids,
    )
