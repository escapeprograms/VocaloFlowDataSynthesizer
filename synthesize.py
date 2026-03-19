import argparse
import os

from synthesizePrior import process_dali_to_ustx
from synthesizePost import process_dali_to_soulx

def synthesize(dali_id="006b5d1db6a447039c30443310b60c6f", language="English", use_continuations=True, output_dir=None, mode="paragraph", n_lines=4, use_f0=False):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Prior (USTX generation)
    print(f"--- Running synthesizePrior (USTX Generation) for {dali_id} (mode={mode}) ---")
    process_dali_to_ustx(output_dir=output_dir, use_continuations=use_continuations, dali_id=dali_id, mode=mode, n_lines=n_lines, use_f0=use_f0)
    
    # Run Post (SoulX-Singer inference)
    print(f"\n--- Running synthesizePost (SoulX-Singer Inference) for {dali_id} (mode={mode}) ---")
    # process_dali_to_soulx(dali_id=dali_id, language=language, output_dir=output_dir, use_continuations=use_continuations, mode=mode, n_lines=n_lines, use_f0=True)
    
    print("\n--- Synthesis Pipeline Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete DataSynthesizer pipeline (Prior + Post).")
    parser.add_argument("--dali_id", default="006b5d1db6a447039c30443310b60c6f", help="DALI dataset entry ID")
    parser.add_argument("--language", default="English", help="Language for grapheme-to-phoneme (default: English)")
    parser.add_argument("--output_dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data")), help="Output directory")
    parser.add_argument("--no_continuations", action="store_true", help="Disable merging of syllable note continuations")
    parser.add_argument("--mode", choices=["line", "n-line", "paragraph", "test"], default="test", help="Granularity of audio segmentation")
    parser.add_argument("--n_lines", type=int, default=4, help="Number of lines to group when mode is n-line")
    parser.add_argument("--use_f0", action="store_true", help="Toggle F0 curve extraction tracking for synthesis")
    
    args = parser.parse_args()
    print(args)
    synthesize(
        dali_id=args.dali_id, 
        language=args.language, 
        use_continuations=not args.no_continuations, 
        output_dir=args.output_dir,
        mode=args.mode,
        n_lines=args.n_lines,
        use_f0=args.use_f0
    )
