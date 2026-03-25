"""
test_hifigan.py

Roundtrip test: load generated.wav -> extract mel spectrogram -> reconstruct via HiFiGAN.
This verifies HiFiGAN is working correctly before any DTW warping is applied.
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import torch

# Add DataSynthesizer root to path for cross-package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.vocoders import invert_mel_to_audio_hifigan, invert_mel_to_audio_griffin_lim, mel_to_soulx_mel, invert_mel_to_audio_soulxsinger

# --- Config (must match segmented_dtw.py settings) ---
GENERATED_WAV = r"C:\Users\archi\Documents\Research\Honors Thesis\Data\006b5d1db6a447039c30443310b60c6f\line_test\generated.wav"
OUTPUT_DIR    = r"C:\Users\archi\Documents\Research\Honors Thesis\Data\006b5d1db6a447039c30443310b60c6f\line_test"

CONFIG = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "hop_length": 256,
    "n_mels": 80,
}

def run():
    sr = CONFIG["sample_rate"]
    n_fft = CONFIG["n_fft"]
    hop_length = CONFIG["hop_length"]
    n_mels = CONFIG["n_mels"]

    print(f"Loading: {GENERATED_WAV}")
    y, file_sr = librosa.load(GENERATED_WAV, sr=sr, mono=True)
    print(f"  Audio loaded: {len(y)} samples @ {sr} Hz, duration={len(y)/sr:.2f}s")

    # --- Extract mel spectrogram (same as segmented_dtw.py) ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    print(f"  Mel shape: {mel.shape}  (n_mels={mel.shape[0]}, frames={mel.shape[1]})")

    # --- Reconstruct via Griffin-Lim as a reference baseline ---
    print("\n[1/2] Reconstructing via Griffin-Lim (baseline)...")
    gl_audio = invert_mel_to_audio_griffin_lim(mel, CONFIG)
    gl_out = os.path.join(OUTPUT_DIR, "test_hifigan_griffinlim.wav")
    sf.write(gl_out, gl_audio, sr)
    print(f"  Saved Griffin-Lim reconstruction -> {gl_out}")

    # --- Reconstruct via HiFiGAN ---
    print("\n[2/2] Reconstructing via HiFiGAN...")
    hifi_audio = invert_mel_to_audio_hifigan(mel, CONFIG)
    hifi_out = os.path.join(OUTPUT_DIR, "test_hifigan_reconstructed.wav")
    sf.write(hifi_out, hifi_audio, sr)
    print(f"  Saved HiFiGAN reconstruction -> {hifi_out}")

    # --- Reconstruct via SoulX-Singer Vocoder ---
    print("\n[3/3] Reconstructing via SoulX-Singer Vocos vocoder...")
    # Re-extract mel using SoulX-Singer's exact 24kHz/128-mel pipeline
    soulx_mel = mel_to_soulx_mel(y, sr=sr)
    print(f"  SoulX mel shape: {soulx_mel.shape}  (n_mels={soulx_mel.shape[0]}, frames={soulx_mel.shape[1]})")
    soulx_audio = invert_mel_to_audio_soulxsinger(soulx_mel, {})
    soulx_out = os.path.join(OUTPUT_DIR, "test_soulxsinger_reconstructed.wav")
    sf.write(soulx_out, soulx_audio, 24000)  # SoulX-Singer outputs at 24kHz
    print(f"  Saved SoulX-Singer reconstruction -> {soulx_out}")

    print("\nDone! Compare all outputs against the original generated.wav.")
    print(f"  Original   : {GENERATED_WAV}")
    print(f"  GriffinLim : {gl_out}")
    print(f"  HiFiGAN    : {hifi_out}")
    print(f"  SoulX-Vocos: {soulx_out}")

if __name__ == "__main__":
    run()
