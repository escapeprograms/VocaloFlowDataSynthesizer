import os
import shutil
import subprocess
import sys
from pathlib import Path

def align_audio_text(audio_path: str, text: str, output_dir: str) -> str:
    """
    Creates a temporary MFA corpus from the given audio and text and runs mfa align.
    Returns the absolute path to the resulting TextGrid file.
    
    Args:
        audio_path: Path to the generated SoulX-Singer `.wav` (or similar).
        text: The plain text representation of the vocals.
        output_dir: The directory where the final `generated.TextGrid` will be saved.
    """
    # Create temporary corpus directory
    corpus_dir = os.path.join(output_dir, "mfa_corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    
    base_name = Path(audio_path).stem
    
    # 1. Copy audio to corpus
    corpus_audio_path = os.path.join(corpus_dir, f"{base_name}.wav")
    shutil.copy(audio_path, corpus_audio_path)
    
    # 2. Write text transcript to corpus (must share name with audio)
    corpus_text_path = os.path.join(corpus_dir, f"{base_name}.txt")
    with open(corpus_text_path, "w", encoding="utf-8") as f:
        f.write(text)
        
    # Output directory for the alignment TextGrid
    mfa_out_dir = os.path.join(output_dir, "mfa_out")
    os.makedirs(mfa_out_dir, exist_ok=True)
    
    # 3. Run MFA align with a unique temporary root to avoid database locks
    mfa_temp_root = os.path.join(output_dir, f"mfa_temp_{base_name}")
    os.makedirs(mfa_temp_root, exist_ok=True)
    
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = mfa_temp_root
    
    mfa_bin = "mfa"
    
    # First, ensure models are downloaded just in case
    download_dict_cmd = [mfa_bin, "model", "download", "dictionary", "english_us_arpa"]
    download_acc_cmd = [mfa_bin, "model", "download", "acoustic", "english_us_arpa"]
    
    try:
        subprocess.run(download_dict_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        subprocess.run(download_acc_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    except subprocess.CalledProcessError:
        pass # Ignore if already downloaded or network unreachable
        
    cmd = [
        mfa_bin, "align",
        corpus_dir,
        "english_us_arpa",
        "english_us_arpa",
        mfa_out_dir,
        "--clean" # Clean up old alignments for this corpus
    ]
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"MFA Alignment failed: {e}")
        return None
        
    # 4. Extract TextGrid and clean up
    textgrid_path = os.path.join(mfa_out_dir, "mfa_corpus", f"{base_name}.TextGrid")
    
    final_tg_path = os.path.join(output_dir, f"{base_name}.TextGrid")
    if os.path.exists(textgrid_path):
        shutil.copy(textgrid_path, final_tg_path)
    else:
        # MFA sometimes puts it in deep subdirectories depending on the version/corpus structure
        # Search for it
        found = False
        for root, dirs, files in os.walk(mfa_out_dir):
            if f"{base_name}.TextGrid" in files:
                shutil.copy(os.path.join(root, f"{base_name}.TextGrid"), final_tg_path)
                found = True
                break
        if not found:
            print(f"Warning: Could not find generated TextGrid in {mfa_out_dir}")
            final_tg_path = None
        
    # Cleanup corpus and temp root
    shutil.rmtree(corpus_dir, ignore_errors=True)
    shutil.rmtree(mfa_out_dir, ignore_errors=True)
    shutil.rmtree(mfa_temp_root, ignore_errors=True)

        
    # Cleanup corpus
    shutil.rmtree(corpus_dir, ignore_errors=True)
    shutil.rmtree(mfa_out_dir, ignore_errors=True)
    
    return final_tg_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    tg_path = align_audio_text(args.audio_path, args.text, args.output_dir)
    if tg_path:
        print(f"SUCCESS:{tg_path}")
    else:
        print("FAILED")
        sys.exit(1)
