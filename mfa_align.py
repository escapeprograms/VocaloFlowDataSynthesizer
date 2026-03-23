import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

def ensure_models_downloaded(env: dict):
    """Ensures MFA models are downloaded."""
    mfa_bin = "mfa"
    download_dict_cmd = [mfa_bin, "model", "download", "dictionary", "english_us_arpa"]
    download_acc_cmd = [mfa_bin, "model", "download", "acoustic", "english_us_arpa"]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Checking/Downloading MFA models...")
    try:
        subprocess.run(download_dict_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        subprocess.run(download_acc_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    except subprocess.CalledProcessError:
        pass

def align_audio_text(audio_path: str, text: str, output_dir: str) -> str:
    """
    Creates a temporary MFA corpus from the given audio and text and runs mfa align.
    Returns the absolute path to the resulting TextGrid file.
    """
    return batch_align_audio_text([(audio_path, text)], output_dir)[0] if batch_align_audio_text([(audio_path, text)], output_dir) else None

def batch_align_audio_text(audio_text_pairs: List[Tuple[str, str]], output_dir: str) -> List[str]:
    """
    Aligns multiple audio files in a single MFA call for efficiency.
    Returns a list of absolute paths to the resulting TextGrid files.
    """
    # Create temporary corpus directory
    corpus_dir = os.path.join(output_dir, "mfa_corpus_batch")
    os.makedirs(corpus_dir, exist_ok=True)
    
    file_stems = []
    for audio_path, text in audio_text_pairs:
        base_name = Path(audio_path).stem
        file_stems.append(base_name)
        
        # 1. Copy audio to corpus
        corpus_audio_path = os.path.join(corpus_dir, f"{base_name}.wav")
        shutil.copy(audio_path, corpus_audio_path)
        
        # 2. Write text transcript to corpus
        corpus_text_path = os.path.join(corpus_dir, f"{base_name}.txt")
        with open(corpus_text_path, "w", encoding="utf-8") as f:
            f.write(text)
        
    # Output directory for the alignment TextGrid
    mfa_out_dir = os.path.join(output_dir, "mfa_out_batch")
    os.makedirs(mfa_out_dir, exist_ok=True)
    
    mfa_temp_root = os.path.join(output_dir, f"mfa_temp_batch")
    os.makedirs(mfa_temp_root, exist_ok=True)
    
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = mfa_temp_root
    
    ensure_models_downloaded(env)
    
    mfa_bin = "mfa"
    cmd = [
        mfa_bin, "align",
        corpus_dir,
        "english_us_arpa",
        "english_us_arpa",
        mfa_out_dir,
        "--clean",
        "--overwrite"
    ]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting MFA Batch Alignment for {len(audio_text_pairs)} files...")
    start_time = datetime.now()
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"MFA Alignment failed: {e}")
        return []
    
    elapsed = datetime.now() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] MFA Alignment complete in {elapsed.total_seconds():.2f}s.")
    
    # 4. Extract TextGrids
    final_tg_paths = []
    for base_name in file_stems:
        final_tg_path = os.path.join(output_dir, f"{base_name}.TextGrid")
        found = False
        for root, dirs, files in os.walk(mfa_out_dir):
            if f"{base_name}.TextGrid" in files:
                shutil.copy(os.path.join(root, f"{base_name}.TextGrid"), final_tg_path)
                found = True
                break
        if found:
            final_tg_paths.append(final_tg_path)
        else:
            print(f"Warning: Could not find generated TextGrid for {base_name} in {mfa_out_dir}")
            final_tg_paths.append(None)
        
    # Cleanup
    shutil.rmtree(corpus_dir, ignore_errors=True)
    shutil.rmtree(mfa_out_dir, ignore_errors=True)
    shutil.rmtree(mfa_temp_root, ignore_errors=True)
    
    return final_tg_paths

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path")
    parser.add_argument("--text")
    parser.add_argument("--output_dir")
    parser.add_argument("--batch_tasks", help="Path to a JSON file containing a list of (audio_path, text, output_dir) tuples")
    args = parser.parse_args()
    
    if args.batch_tasks:
        with open(args.batch_tasks, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        
        # grouping by output_dir is not strictly necessary but if they are different we might need to handle it.
        # But for now, let's assume they all go to the same base directory's mfa_corpus_batch.
        # However, the returning TextGrid paths should be printed.
        
        # Just use the first task's output_dir as the base for the batch if not provided?
        # Better: use a common temp dir or just the first one's dir.
        common_output_dir = tasks[0][2] if tasks else "."
        
        audio_text_pairs = [(t[0], t[1]) for t in tasks]
        tg_paths = batch_align_audio_text(audio_text_pairs, common_output_dir)
        
        if tg_paths:
            for path in tg_paths:
                if path:
                    print(f"SUCCESS:{path}")
                else:
                    print("FAILED_ITEM")
        else:
            print("FAILED")
            sys.exit(1)
            
    elif args.audio_path and args.text and args.output_dir:
        tg_path = align_audio_text(args.audio_path, args.text, args.output_dir)
        if tg_path:
            print(f"SUCCESS:{tg_path}")
        else:
            print("FAILED")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
