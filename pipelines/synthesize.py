import argparse
import os
import sys
from datetime import datetime

# Add DataSynthesizer root to path for cross-package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stages.synthesizePrior import process_dali_to_ustx
from stages.synthesizeTarget import process_dali_to_target
from stages.synthesizeDTW import run_dtw_alignment

def synthesize(dali_id="006b5d1db6a447039c30443310b60c6f", language="English", use_continuations=False, output_dir=None, mode="paragraph", n_lines=4, use_f0=False, segmentation_mode="word", vocoder="griffin_lim", use_phonemes=False):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data"))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Prior (USTX generation)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] --- Running synthesizePrior (USTX Generation) for {dali_id} (mode={mode}) ---")
    process_dali_to_ustx(output_dir=output_dir, use_continuations=use_continuations, dali_id=dali_id, mode=mode, n_lines=n_lines, use_f0=use_f0, use_phonemes=use_phonemes)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Prior synthesis complete.")
    
    # Run Target (SoulX-Singer inference)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- Running synthesizeTarget (SoulX-Singer Inference) for {dali_id} (mode={mode}) ---")
    process_dali_to_target(dali_id=dali_id, language=language, output_dir=output_dir, use_continuations=use_continuations, mode=mode, n_lines=n_lines, use_f0=use_f0, save_mel=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Target synthesis complete.")
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- Running Segmented DTW Time Alignment ---")
    try:
        run_dtw_alignment(
            dali_id=dali_id,
            output_dir=output_dir,
            mode=mode,
            segmentation_mode=segmentation_mode,
            vocoder=vocoder,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to run DTW alignment: {e}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- Synthesis Pipeline Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete DataSynthesizer pipeline (Prior + Post).")
    parser.add_argument("--dali_id", default="006b5d1db6a447039c30443310b60c6f", help="DALI dataset entry ID")
    parser.add_argument("--language", default="English", help="Language for grapheme-to-phoneme (default: English)")
    parser.add_argument("--output_dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data")), help="Output directory")
    parser.add_argument("--use_continuations", action="store_true", help="Connect all consecutive notes into a legato phrase (intra-word legato is always on)")
    parser.add_argument("--mode", choices=["line", "n-line", "paragraph", "test"], default="test", help="Granularity of audio segmentation")
    parser.add_argument("--n_lines", type=int, default=4, help="Number of lines to group when mode is n-line")
    parser.add_argument("--use_f0", action="store_true", help="Toggle F0 curve extraction tracking for synthesis")
    parser.add_argument("--segmentation_mode", choices=["word", "phoneme"], default="word", help="Granularity of DTW alignment (default: word)")
    parser.add_argument("--vocoder", choices=["griffin_lim", "hifigan", "soulxsinger"], default="soulxsinger", help="Vocoder used for final mel-spectrogram inversion (default: hifigan).")
    parser.add_argument("--use_phonemes", action="store_true", help="Use g2p_en full-word G2P with ARPAbet phoneticHints per note for prior synthesis.")

    args = parser.parse_args()
    args.use_continuations = True
    args.mode = "line"
    args.use_phonemes = True
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
        vocoder=args.vocoder,
        use_phonemes=args.use_phonemes
    )
