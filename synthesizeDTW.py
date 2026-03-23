import json
import os
import subprocess
import tempfile
from datetime import datetime

def _write_mfa_failure_meta(chunk_dir: str, prior_ok: bool, post_ok: bool) -> None:
    path = os.path.join(chunk_dir, "alignment.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "mean_dtw_cost": None,
            "max_dtw_cost": None,
            "per_phoneme_costs": [],
            "n_phonemes_aligned": 0,
            "n_phonemes_total": 0,
            "under_threshold": False,
            "cost_threshold": None,
            "mfa_prior_ok": prior_ok,
            "mfa_post_ok": post_ok,
            "aligned_saved": False,
            "segmentation_mode": None,
            "vocoder": None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)


def run_batch_mfa(tasks, label, conda_exe, mfa_script):
    """Run MFA alignment in batch for a list of (audio_path, text, chunk_dir) tuples.

    Returns a list of TextGrid paths parsed from SUCCESS: lines in stdout.
    Order matches the input task list.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running Batch MFA for {label} ({len(tasks)} files)...")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        json.dump([(t[0], t[1], t[2]) for t in tasks], tf)
        temp_task_file = tf.name

    mfa_cmd = [
        conda_exe, "run", "-n", "vocaloflow-mfa",
        "python", mfa_script,
        "--batch_tasks", temp_task_file
    ]

    env = os.environ.copy()
    result = subprocess.run(mfa_cmd, env=env, capture_output=True, text=True)
    os.unlink(temp_task_file)

    if result.returncode != 0:
        print(f"Batch MFA for {label} failed:\n{result.stderr}")
        return []

    paths = []
    for line in result.stdout.split('\n'):
        if line.startswith("SUCCESS:"):
            paths.append(line.split("SUCCESS:")[1].strip())
    return paths


def run_dtw_alignment(dali_id, output_dir, mode, segmentation_mode, vocoder):
    """Run segmented DTW time alignment for all chunks of a given DALI entry.

    Steps:
        1. Discover chunks with prior.wav, generated.wav, and music.json.
        2. Run batch MFA on prior and post audio.
        3. Run align_and_export_mel (segmented DTW) per chunk.

    Args:
        dali_id: DALI entry ID, used to locate the output subdirectory.
        output_dir: Root data output directory.
        mode: Pipeline mode (e.g. 'line', 'paragraph', 'test'). 'test' enables diagnostic output.
        segmentation_mode: DTW segmentation granularity ('word' or 'phoneme').
        vocoder: Vocoder used for mel inversion ('griffin_lim', 'hifigan', 'soulxsinger').
    """
    from segmented_dtw import align_and_export_mel

    conda_exe = r"C:\Users\archi\miniconda3\Scripts\conda.exe"
    mfa_script = os.path.join(os.path.dirname(__file__), "mfa_align.py")

    dali_dir = os.path.join(output_dir, dali_id)
    if not os.path.exists(dali_dir):
        print(f"DALI output directory not found: {dali_dir}")
        return

    # 1. Collect chunks that have all required files
    prior_tasks = []
    post_tasks = []
    chunks_to_process = []

    for chunk_name in os.listdir(dali_dir):
        chunk_dir = os.path.join(dali_dir, chunk_name)
        if not os.path.isdir(chunk_dir):
            continue

        prior_path = os.path.join(chunk_dir, "prior.wav")
        post_path = os.path.join(chunk_dir, "generated.wav")
        music_json_path = os.path.join(chunk_dir, "music.json")

        if os.path.exists(prior_path) and os.path.exists(post_path) and os.path.exists(music_json_path):
            with open(music_json_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            full_text = " ".join([m.get("text", "") for m in meta_data])

            prior_tasks.append((prior_path, full_text, chunk_dir))
            post_tasks.append((post_path, full_text, chunk_dir))
            chunks_to_process.append(chunk_name)

    if not chunks_to_process:
        print("No chunks found to align.")
        return

    # 2. Batch MFA for prior and post audio
    all_prior_textgrids = run_batch_mfa(prior_tasks, "Prior", conda_exe, mfa_script)
    all_post_textgrids = run_batch_mfa(post_tasks, "Post", conda_exe, mfa_script)

    # 3. Segmented DTW per chunk
    dtw_config = {
        "sample_rate": 22050,
        "n_fft": 1024,
        "hop_length": 256,
        "n_mels": 80,
    }

    for i, chunk_name in enumerate(chunks_to_process):
        chunk_dir = os.path.join(dali_dir, chunk_name)
        prior_path, _, _ = prior_tasks[i]
        post_path, _, _ = post_tasks[i]

        prior_textgrid_path = all_prior_textgrids[i] if i < len(all_prior_textgrids) else None
        post_textgrid_path = all_post_textgrids[i] if i < len(all_post_textgrids) else None

        if not prior_textgrid_path or not post_textgrid_path:
            print(f"Warning: Missing TextGrids for {chunk_name}. Skipping DTW.")
            _write_mfa_failure_meta(
                chunk_dir,
                prior_ok=prior_textgrid_path is not None,
                post_ok=post_textgrid_path is not None,
            )
            continue

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Aligning {chunk_name} using Segmented DTW...")
        try:
            post_mel_path = os.path.join(chunk_dir, "generated_mel.npy")
            if not os.path.exists(post_mel_path):
                post_mel_path = None

            success = align_and_export_mel(
                prior_audio_path=prior_path,
                post_audio_path=post_path,
                prior_textgrid_path=prior_textgrid_path,
                post_textgrid_path=post_textgrid_path,
                config=dtw_config,
                pad_frames=4,
                cost_threshold=100.0,
                export_dir=chunk_dir,
                diagnostic_mode=(mode == 'test'),
                segmentation_mode=segmentation_mode,
                vocoder=vocoder,
                post_mel_path=post_mel_path,
            )

            if success:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] DTW complete for {chunk_name} (under threshold).")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] DTW complete for {chunk_name} (flagged: above threshold, aligned.wav still saved).")
        except Exception as e:
            print(f"Error aligning {chunk_name}: {e}")
