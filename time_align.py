import librosa
import soundfile as sf
import numpy as np

def align_audio(audio_path_A, audio_path_B, output_path=None):
    """
    Time-aligns audio file A to coordinate with audio file B using Dynamic Time Warping (DTW).
    Returns the aligned audio A' and the DTW cost.
    
    A: The audio to be warped (e.g. AI Generated Post)
    B: The reference audio (e.g. Teto Prior)
    """
    # Load audio files
    y_A, sr_A = librosa.load(audio_path_A, sr=None)
    y_B, sr_B = librosa.load(audio_path_B, sr=sr_A) # Ensure same sample rate
    
    # Extract features (e.g. MFCCs or Chroma) for alignment
    # Chromagrams are good for aligning speech/music
    hop_length = 512
    chroma_A = librosa.feature.chroma_cqt(y=y_A, sr=sr_A, hop_length=hop_length)
    chroma_B = librosa.feature.chroma_cqt(y=y_B, sr=sr_A, hop_length=hop_length)
    
    # Use DTW to compute the alignment
    D, wp = librosa.sequence.dtw(X=chroma_A, Y=chroma_B, metric='cosine')
    
    # Warping path returns pairs (m, n) mapping chroma_A[m] to chroma_B[n]
    # wp is returning in reverse order, from end to start
    wp = wp[::-1]
    
    # To warp A to match B's timing, we need to resample A at the frame level.
    # For every frame n in B, we find the corresponding frame m in A.
    
    # Create time arrays for the frames
    time_A = librosa.frames_to_samples(wp[:, 0], hop_length=hop_length)
    time_B = librosa.frames_to_samples(wp[:, 1], hop_length=hop_length)
    
    # Interpolate to find the exact sample mapping for every sample in B
    target_samples = np.arange(len(y_B))
    
    # Handle cases where the warp path doesn't quite cover the ends
    # We map target_samples (B) back to source samples (A)
    source_samples = np.interp(target_samples, time_B, time_A)
    
    # Now interpolate the audio signal
    y_A_aligned = np.interp(source_samples, np.arange(len(y_A)), y_A)
    
    if output_path is not None:
        sf.write(output_path, y_A_aligned, sr_A)
        
    # The cost is D[-1, -1] representing the total accumulated cost
    cost = D[-1, -1]
    
    return y_A_aligned, sr_A, cost

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_A", help="Path to audio A (to be warped)")
    parser.add_argument("audio_B", help="Path to audio B (reference)")
    parser.add_argument("output", help="Path to save warped audio A'")
    args = parser.parse_args()
    
    print(f"Aligning {args.audio_A} to {args.audio_B}...")
    _, _, cost = align_audio(args.audio_A, args.audio_B, args.output)
    print(f"Done. DTW Cost: {cost}")
