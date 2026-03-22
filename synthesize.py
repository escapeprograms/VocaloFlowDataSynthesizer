import argparse
import os

from synthesizePrior import process_dali_to_ustx
from synthesizePost import process_dali_to_soulx

def synthesize(dali_id="006b5d1db6a447039c30443310b60c6f", language="English", use_continuations=False, output_dir=None, mode="paragraph", n_lines=4, use_f0=False, segmentation_mode="word", vocoder="griffin_lim"):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Prior (USTX generation)
    print(f"--- Running synthesizePrior (USTX Generation) for {dali_id} (mode={mode}) ---")
    process_dali_to_ustx(output_dir=output_dir, use_continuations=use_continuations, dali_id=dali_id, mode=mode, n_lines=n_lines, use_f0=use_f0)
    
    # Run Post (SoulX-Singer inference)
    print(f"\n--- Running synthesizePost (SoulX-Singer Inference) for {dali_id} (mode={mode}) ---")
    process_dali_to_soulx(dali_id=dali_id, language=language, output_dir=output_dir, use_continuations=use_continuations, mode=mode, n_lines=n_lines, use_f0=use_f0, save_mel=True)
    
    print("\n--- Running Segmented DTW Time Alignment ---")
    try:
        from segmented_dtw import align_and_export_mel
        import json
        

        dali_dir = os.path.join(output_dir, dali_id)
        if os.path.exists(dali_dir):
            for chunk_name in os.listdir(dali_dir):
                chunk_dir = os.path.join(dali_dir, chunk_name)
                if os.path.isdir(chunk_dir):
                    prior_path = os.path.join(chunk_dir, "prior.wav")
                    post_path = os.path.join(chunk_dir, "generated.wav")
                    music_json_path = os.path.join(chunk_dir, "music.json")
                    
                    if os.path.exists(prior_path) and os.path.exists(post_path) and os.path.exists(music_json_path):
                        print(f"Aligning {chunk_name} using Segmented DTW...")
                        try:
                            # 1. Gen text for MFA
                            with open(music_json_path, 'r', encoding='utf-8') as f:
                                meta_data = json.load(f)
                            # join text
                            full_text = " ".join([m.get("text", "") for m in meta_data])
                            
                            # 2. Run mfa_align via subprocess using the vocaloflow-mfa environment
                            conda_exe = r"C:\Users\archi\miniconda3\Scripts\conda.exe"
                            mfa_script = os.path.join(os.path.dirname(__file__), "mfa_align.py")
                            
                            def run_mfa(audio_to_align):
                                print(f"Running MFA for {os.path.basename(audio_to_align)} in {chunk_name}...")
                                mfa_cmd = [
                                    conda_exe, "run", "-n", "vocaloflow-mfa",
                                    "python", mfa_script,
                                    "--audio_path", audio_to_align,
                                    "--text", full_text,
                                    "--output_dir", chunk_dir
                                ]
                                
                                import subprocess
                                env = os.environ.copy()
                                result = subprocess.run(mfa_cmd, env=env, capture_output=True, text=True)
                                
                                out_tg = None
                                if result.returncode != 0:
                                    print(f"MFA Script failed with error:\n{result.stderr}")
                                else:
                                    for line in result.stdout.split('\n'):
                                        if line.startswith("SUCCESS:"):
                                            out_tg = line.split("SUCCESS:")[1].strip()
                                return out_tg
                                
                            post_textgrid_path = run_mfa(post_path)
                            prior_textgrid_path = run_mfa(prior_path)
                            
                            if not post_textgrid_path or not prior_textgrid_path:
                                print(f"Warning: Missing TextGrids generated for {chunk_name}. Skipping DTW.")
                                continue
                            
                            # Check for pre-extracted SoulX-Singer mel
                            post_mel_path = os.path.join(chunk_dir, "generated_mel.npy")
                            if not os.path.exists(post_mel_path):
                                post_mel_path = None

                            config = {
                                "sample_rate": 22050,
                                "n_fft": 1024,
                                "hop_length": 256,
                                "n_mels": 80
                            }
                            success = align_and_export_mel(
                                prior_audio_path=prior_path,
                                post_audio_path=post_path,
                                prior_textgrid_path=prior_textgrid_path,
                                post_textgrid_path=post_textgrid_path,
                                config=config,
                                pad_frames=4,
                                cost_threshold=100.0, # Lenient default threshold
                                export_dir=chunk_dir,
                                diagnostic_mode=(mode == 'test'),
                                segmentation_mode=segmentation_mode,
                                vocoder=vocoder,
                                post_mel_path=post_mel_path
                            )
                            if success:
                                print(f"Successfully ran segmented DTW for {chunk_name}.")
                            else:
                                print(f"Segmented DTW rejected for {chunk_name} due to high cost.")
                        except Exception as e:
                            print(f"Error aligning {chunk_name}: {e}")
    except Exception as e:
        print(f"Failed to run DTW alignment: {e}")

    print("\n--- Synthesis Pipeline Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete DataSynthesizer pipeline (Prior + Post).")
    parser.add_argument("--dali_id", default="006b5d1db6a447039c30443310b60c6f", help="DALI dataset entry ID")
    parser.add_argument("--language", default="English", help="Language for grapheme-to-phoneme (default: English)")
    parser.add_argument("--output_dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data")), help="Output directory")
    parser.add_argument("--use_continuations", action="store_true", help="Enable merging of syllable note continuations")
    parser.add_argument("--mode", choices=["line", "n-line", "paragraph", "test"], default="test", help="Granularity of audio segmentation")
    parser.add_argument("--n_lines", type=int, default=4, help="Number of lines to group when mode is n-line")
    parser.add_argument("--use_f0", action="store_true", help="Toggle F0 curve extraction tracking for synthesis")
    parser.add_argument("--segmentation_mode", choices=["word", "phoneme"], default="word", help="Granularity of DTW alignment (default: word)")
    parser.add_argument("--vocoder", choices=["griffin_lim", "hifigan", "soulxsinger"], default="soulxsinger", help="Vocoder used for final mel-spectrogram inversion (default: hifigan).")
    
    args = parser.parse_args()
    print(args)
    synthesize(
        dali_id=args.dali_id, 
        language=args.language, 
        use_continuations=args.use_continuations, 
        output_dir=args.output_dir,
        mode=args.mode,
        n_lines=args.n_lines,
        use_f0=args.use_f0,
        segmentation_mode=args.segmentation_mode,
        vocoder=args.vocoder
    )
