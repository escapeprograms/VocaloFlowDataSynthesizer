"""
Dataset-scale target-first synthesis pipeline (v2).

Flips the generation order: Target audio is generated first from DALI annotations,
then ROSVOT extracts note timings/pitches from the target audio, and Prior audio
is generated from those extracted notes.  This produces priors that match what
SoulX-Singer actually sang, improving DTW alignment quality.

Five-phase pipeline that amortises all expensive model loads:

  Phase 1 — Target Metadata : Generate music.json + chunk_words.json from DALI.
  Phase 2 — Inference     : Run SoulX-Singer once per batch of songs.
  Phase 3 — Extraction    : Run ROSVOT + F0 once per batch of songs.
  Phase 4 — Prior Gen     : Generate prior.wav from extracted notes (OpenUtau).
  Phase 5 — Alignment     : Run MFA + segmented DTW per song.

All five phases are independently resumable via sentinel files per chunk.

Usage:
    python synthesize_dataset_v2.py --phases 12345 --songs_per_batch 100
    python synthesize_dataset_v2.py --phases 34    --songs_per_batch 50
    python synthesize_dataset_v2.py --dali_ids <id1> <id2>
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import List, Optional

# Add DataSynthesizer root to path for cross-package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stages.synthesizeTarget import process_dali_to_target, get_soulx_inference_config
from stages.synthesizeDTW import run_dtw_alignment
from pipelines.synthesize_v2 import save_chunk_words

# Import batch infrastructure from v1
from pipelines.synthesize_dataset import (
    get_english_dali_ids,
    _launch_inference_subprocess,
    SOULX_PYTHON,
    SOULX_DIR,
    DALI_ANNOT_DIR,
    DEFAULT_OUTPUT,
    TASKS_CACHE_NAME,
)

# Cache file for extraction tasks between Phase 2 and Phase 3
EXTRACTION_CACHE_NAME = "pending_extraction_tasks.json"


# ---------------------------------------------------------------------------
# Phase 1 — Target Metadata
# ---------------------------------------------------------------------------

def run_phase1_target_metadata(
    dali_ids: List[str],
    output_dir: str,
    mode: str,
    n_lines: int,
    use_f0: bool,
    use_continuations: bool,
    save_mel: bool,
) -> List[dict]:
    """Generate music.json + chunk_words.json for every song; collect inference tasks.

    Returns the combined list of all SoulX-Singer inference task dicts.
    """
    all_tasks: List[dict] = []
    total = len(dali_ids)

    for i, dali_id in enumerate(dali_ids):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] [Phase 1 | {i+1}/{total}] {dali_id}")
        try:
            # Save chunk_words.json for each chunk (needed by Phase 3)
            save_chunk_words(dali_id, output_dir, mode, n_lines)

            # Generate music.json and collect inference tasks
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
# Phase 2 — Inference (reuses v1 infrastructure)
# ---------------------------------------------------------------------------

def run_phase2_inference(all_tasks: List[dict], songs_per_batch: int) -> None:
    """Split task list into song-sized batches and run one subprocess per batch.

    Identical to v1 — the SoulX-Singer model is loaded once per subprocess.
    """
    if not all_tasks:
        print("[Phase 2] No tasks to run.")
        return

    infer_cfg = get_soulx_inference_config()

    def dali_id_from_task(task: dict) -> str:
        return os.path.basename(os.path.dirname(task["save_dir"]))

    seen: dict = {}
    for task in all_tasks:
        did = dali_id_from_task(task)
        seen.setdefault(did, []).append(task)

    song_ids = list(seen.keys())
    total_batches = max(1, (len(song_ids) + songs_per_batch - 1) // songs_per_batch)

    print(f"\n[Phase 2] {len(all_tasks)} tasks across {len(song_ids)} songs "
          f"-> {total_batches} batch(es) of up to {songs_per_batch} songs each.")

    for batch_i in range(total_batches):
        batch_song_ids = song_ids[batch_i * songs_per_batch : (batch_i + 1) * songs_per_batch]
        batch_tasks = [task for sid in batch_song_ids for task in seen[sid]]
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] [Phase 2 | Batch {batch_i+1}/{total_batches}] "
              f"{len(batch_song_ids)} songs, {len(batch_tasks)} chunks...")
        _launch_inference_subprocess(batch_tasks, infer_cfg)


# ---------------------------------------------------------------------------
# Phase 3 — Note Extraction + F0
# ---------------------------------------------------------------------------

def collect_extraction_tasks(
    dali_ids: List[str],
    output_dir: str,
) -> List[dict]:
    """Scan output dirs for chunks with target.wav but no extracted_notes.json."""
    tasks = []
    for dali_id in dali_ids:
        dali_dir = os.path.join(output_dir, dali_id)
        if not os.path.isdir(dali_dir):
            continue
        for chunk_name in sorted(os.listdir(dali_dir)):
            chunk_dir = os.path.join(dali_dir, chunk_name)
            if not os.path.isdir(chunk_dir):
                continue

            audio_path = os.path.join(chunk_dir, "target.wav")
            notes_path = os.path.join(chunk_dir, "extracted_notes.json")
            f0_path = os.path.join(chunk_dir, "target_f0.npy")

            if not os.path.exists(audio_path):
                continue
            if os.path.exists(notes_path) and os.path.exists(f0_path):
                continue

            # Load chunk words for lyric mapping
            words = []
            words_path = os.path.join(chunk_dir, "chunk_words.json")
            if os.path.exists(words_path):
                with open(words_path, "r", encoding="utf-8") as f:
                    words = json.load(f)

            tasks.append({
                "chunk_dir": chunk_dir,
                "audio_path": audio_path,
                "item_name": chunk_name,
                "words": words,
            })
    return tasks


def _launch_extraction_subprocess(tasks: List[dict]) -> None:
    """Write tasks to a temp JSON file and launch note_extraction_batch.py."""
    batch_script = os.path.join(os.path.dirname(__file__), "..", "batch", "note_extraction_batch.py")
    rmvpe_base = os.path.join(SOULX_DIR, "pretrained_models", "SoulX-Singer-Preprocess")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tf:
        json.dump(tasks, tf)
        tasks_file = tf.name

    cmd = [
        SOULX_PYTHON, batch_script,
        "--tasks_json",        tasks_file,
        "--rmvpe_model_path",  os.path.join(rmvpe_base, "rmvpe", "rmvpe.pt"),
        "--device",            "cuda",
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


def run_phase3_extraction(
    all_tasks: List[dict],
    songs_per_batch: int,
) -> None:
    """Split extraction tasks into batches and run one subprocess per batch.

    ROSVOT + RMVPE models are loaded once per subprocess.
    """
    if not all_tasks:
        print("[Phase 3] No tasks to run.")
        return

    # Group by dali_id for batching
    def dali_id_from_task(task: dict) -> str:
        # chunk_dir is <output_dir>/<dali_id>/<chunk_name>
        return os.path.basename(os.path.dirname(task["chunk_dir"]))

    seen: dict = {}
    for task in all_tasks:
        did = dali_id_from_task(task)
        seen.setdefault(did, []).append(task)

    song_ids = list(seen.keys())
    total_batches = max(1, (len(song_ids) + songs_per_batch - 1) // songs_per_batch)

    print(f"\n[Phase 3] {len(all_tasks)} tasks across {len(song_ids)} songs "
          f"-> {total_batches} batch(es) of up to {songs_per_batch} songs each.")

    for batch_i in range(total_batches):
        batch_song_ids = song_ids[batch_i * songs_per_batch : (batch_i + 1) * songs_per_batch]
        batch_tasks = [task for sid in batch_song_ids for task in seen[sid]]
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] [Phase 3 | Batch {batch_i+1}/{total_batches}] "
              f"{len(batch_song_ids)} songs, {len(batch_tasks)} chunks...")
        _launch_extraction_subprocess(batch_tasks)


# ---------------------------------------------------------------------------
# Phase 4 — Prior Generation
# ---------------------------------------------------------------------------

def run_phase4_prior_generation(
    dali_ids: List[str],
    output_dir: str,
    use_phonemes: bool = True,
) -> None:
    """Generate prior.wav from extracted_notes.json for all chunks.

    Initialises OpenUtau Player once and iterates through all chunks.
    """
    from stages.synthesizePrior import generate_prior_from_notes, Player

    player = Player("OpenUtau.Plugin.Builtin.ArpasingPlusPhonemizer")
    total = len(dali_ids)

    for i, dali_id in enumerate(dali_ids):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] [Phase 4 | {i+1}/{total}] {dali_id}")
        dali_dir = os.path.join(output_dir, dali_id)
        if not os.path.isdir(dali_dir):
            continue

        for chunk_name in sorted(os.listdir(dali_dir)):
            chunk_dir = os.path.join(dali_dir, chunk_name)
            if not os.path.isdir(chunk_dir):
                continue

            notes_path = os.path.join(chunk_dir, "extracted_notes.json")
            prior_path = os.path.join(chunk_dir, "prior.wav")

            if not os.path.exists(notes_path):
                continue
            if os.path.exists(prior_path):
                continue

            try:
                generate_prior_from_notes(chunk_dir, notes_path, player, use_phonemes=use_phonemes)
            except Exception as e:
                print(f"  [Phase 4] ERROR for {chunk_name}: {e}")


# ---------------------------------------------------------------------------
# Phase 5 — DTW Alignment
# ---------------------------------------------------------------------------

def run_phase5_alignment(
    dali_ids: List[str],
    output_dir: str,
    mode: str,
    segmentation_mode: str,
    vocoder: str,
) -> None:
    """Run MFA + segmented DTW for every song.

    Uses align_to='target' so the prior is warped onto the target's timeline
    (v2 reversal: aligned.wav = prior timbre on target timing).
    """
    total = len(dali_ids)
    for i, dali_id in enumerate(dali_ids):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] [Phase 5 | {i+1}/{total}] {dali_id}")
        try:
            run_dtw_alignment(
                dali_id=dali_id,
                output_dir=output_dir,
                mode=mode,
                segmentation_mode=segmentation_mode,
                vocoder=vocoder,
                align_to="target",
            )
        except Exception as e:
            print(f"  [Phase 5] ERROR for {dali_id}: {e}")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def synthesize_dataset_v2(
    output_dir: str = DEFAULT_OUTPUT,
    dali_annot_dir: str = DALI_ANNOT_DIR,
    mode: str = "line",
    n_lines: int = 4,
    use_f0: bool = False,
    use_continuations: bool = True,
    use_phonemes: bool = True,
    segmentation_mode: str = "word",
    vocoder: str = "soulxsinger",
    songs_per_batch: int = 100,
    phases: str = "12345",
    dali_ids: Optional[List[str]] = None,
) -> None:
    """Run the full target-first dataset synthesis pipeline.

    Args:
        output_dir:         Root directory for all generated files.
        dali_annot_dir:     Path to DALI annot_tismir directory.
        mode:               Chunk granularity — 'line', 'n-line', 'paragraph', 'test'.
        n_lines:            Lines per chunk when mode='n-line'.
        use_f0:             Use F0 curves for SoulX-Singer conditioning.
        use_continuations:  Extend note durations to fill intra-word gaps.
        use_phonemes:       Use ARPAbet phoneticHints via g2p_en.
        segmentation_mode:  DTW granularity — 'word' or 'phoneme'.
        vocoder:            Vocoder for mel inversion.
        songs_per_batch:    Songs grouped per subprocess call.
        phases:             Which phases to run, e.g. '12345', '34', '5'.
        dali_ids:           If given, process only these IDs; otherwise all English.
    """
    inference_cache = os.path.join(output_dir, TASKS_CACHE_NAME)
    extraction_cache = os.path.join(output_dir, EXTRACTION_CACHE_NAME)
    os.makedirs(output_dir, exist_ok=True)

    # Resolve song list
    if dali_ids is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning for English DALI entries...")
        dali_ids = get_english_dali_ids(dali_annot_dir)
        print(f"Found {len(dali_ids)} English entries.")

    _sep = "=" * 62

    # ---- Phase 1 ----
    if "1" in phases:
        print(f"\n{_sep}")
        print(f"  PHASE 1 — Target Metadata  ({len(dali_ids)} songs)")
        print(_sep)
        all_tasks = run_phase1_target_metadata(
            dali_ids, output_dir, mode, n_lines,
            use_f0, use_continuations, save_mel=True,
        )
        with open(inference_cache, "w", encoding="utf-8") as f:
            json.dump(all_tasks, f)
        print(f"\n[Phase 1] Complete — {len(all_tasks)} inference tasks cached.")

    # ---- Phase 2 ----
    if "2" in phases:
        print(f"\n{_sep}")
        print(f"  PHASE 2 — SoulX-Singer Inference")
        print(_sep)
        if not os.path.exists(inference_cache):
            raise FileNotFoundError(
                f"Inference task cache not found at {inference_cache}. Run Phase 1 first."
            )
        with open(inference_cache, "r", encoding="utf-8") as f:
            all_tasks = json.load(f)
        run_phase2_inference(all_tasks, songs_per_batch=songs_per_batch)

    # ---- Phase 3 ----
    if "3" in phases:
        print(f"\n{_sep}")
        print(f"  PHASE 3 — Note Extraction + F0")
        print(_sep)
        extraction_tasks = collect_extraction_tasks(dali_ids, output_dir)
        with open(extraction_cache, "w", encoding="utf-8") as f:
            json.dump(extraction_tasks, f)
        print(f"  {len(extraction_tasks)} extraction tasks collected.")
        run_phase3_extraction(extraction_tasks, songs_per_batch=songs_per_batch)

    # ---- Phase 4 ----
    if "4" in phases:
        print(f"\n{_sep}")
        print(f"  PHASE 4 — Prior Generation  ({len(dali_ids)} songs)")
        print(_sep)
        run_phase4_prior_generation(dali_ids, output_dir, use_phonemes=use_phonemes)

    # ---- Phase 5 ----
    if "5" in phases:
        print(f"\n{_sep}")
        print(f"  PHASE 5 — DTW Alignment  ({len(dali_ids)} songs)")
        print(_sep)
        run_phase5_alignment(dali_ids, output_dir, mode, segmentation_mode, vocoder)

    print(f"\n{_sep}")
    print(f"  v2 Dataset synthesis complete.")
    print(_sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-first dataset synthesis pipeline (Phase 1-5)."
    )
    parser.add_argument("--output_dir",        default=DEFAULT_OUTPUT)
    parser.add_argument("--dali_annot_dir",    default=DALI_ANNOT_DIR)
    parser.add_argument("--mode",              default="line",
                        choices=["line", "n-line", "paragraph", "test"])
    parser.add_argument("--n_lines",           type=int, default=4)
    parser.add_argument("--use_f0",            action="store_true")
    parser.add_argument("--use_continuations", action="store_true", default=True)
    parser.add_argument("--use_phonemes",      action="store_true", default=True)
    parser.add_argument("--segmentation_mode", default="word", choices=["word", "phoneme"])
    parser.add_argument("--vocoder",           default="soulxsinger",
                        choices=["soulxsinger", "hifigan", "griffin_lim"])
    parser.add_argument("--songs_per_batch",   type=int, default=100,
                        help="Songs grouped per subprocess (default: 100).")
    parser.add_argument("--phases",            default="12345",
                        help="Which phases to run, e.g. '12345', '34', '5'.")
    parser.add_argument("--dali_ids",          nargs="*", default=None,
                        help="Optional subset of DALI IDs to process.")

    args = parser.parse_args()
    args.use_continuations = True
    args.mode = "line"
    args.use_phonemes = True

    synthesize_dataset_v2(
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
