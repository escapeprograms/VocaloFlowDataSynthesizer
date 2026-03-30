"""
Detect silences/rests in the middle of SoulX-Singer generated audio chunks.

Scans all chunks in a data directory, identifies internal gaps using two signals:
  1. F0 gaps (consecutive unvoiced frames where F0=0)
  2. Energy gaps (RMS below threshold)
Cross-references both to classify gaps and ranks chunks by severity.
Generates visualizations for the worst offenders.
"""

import argparse
import glob
import json
import os
import re

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# ── Constants ──────────────────────────────────────────────────────────────────
SR = 24000
HOP = 480
FRAME_SEC = HOP / SR  # 0.02s = 20ms

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "Data", "006b5d1db6a447039c30443310b60c6f"
)

MIN_GAP_FRAMES = 3       # 60ms minimum gap duration
MARGIN_FRAMES = 3         # ignore first/last 60ms of chunk
ENERGY_THRESH_DB = -40     # RMS below this = silence
TOP_N = 10                 # visualize top N chunks
RMS_FRAME_LENGTH = 2048


# ── Utilities ──────────────────────────────────────────────────────────────────

def find_gaps(mask, margin_frames, min_frames):
    """Find contiguous True runs in a boolean mask, ignoring edges.

    Args:
        mask: 1D boolean array (True = gap condition met)
        margin_frames: frames to trim from each end
        min_frames: minimum run length to report

    Returns:
        List of dicts with start_frame, end_frame, start_sec, end_sec, duration_sec.
        Frame indices are relative to the original (untrimmed) array.
    """
    n = len(mask)
    if n <= 2 * margin_frames:
        return []

    # Work on the interior only
    interior = mask[margin_frames: n - margin_frames].copy()

    # Pad with False at boundaries to detect edges cleanly
    padded = np.concatenate([[False], interior, [False]])
    diff = np.diff(padded.astype(np.int8))

    starts = np.where(diff == 1)[0]   # start of True runs
    ends = np.where(diff == -1)[0]    # end of True runs

    gaps = []
    for s, e in zip(starts, ends):
        length = e - s
        if length >= min_frames:
            # Convert back to original array indices
            abs_start = s + margin_frames
            abs_end = e + margin_frames
            gaps.append({
                "start_frame": int(abs_start),
                "end_frame": int(abs_end),
                "duration_frames": int(length),
                "start_sec": round(abs_start * FRAME_SEC, 3),
                "end_sec": round(abs_end * FRAME_SEC, 3),
                "duration_sec": round(length * FRAME_SEC, 3),
            })
    return gaps


def find_note_at_time(notes, time_sec):
    """Return the note text active at a given time, or '?' if none."""
    for note in notes:
        start = note["start_s"]
        end = start + note["note_dur"]
        if start <= time_sec < end:
            return note["note_text"]
    return "?"


# ── Per-chunk analysis ─────────────────────────────────────────────────────────

def analyze_chunk(chunk_dir, min_gap_frames, margin_frames, energy_thresh_db):
    """Analyze a single chunk for internal silences.

    Returns analysis dict or None if required files are missing.
    """
    chunk_name = os.path.basename(chunk_dir)

    # Load F0
    f0_path = os.path.join(chunk_dir, "post_f0.npy")
    if not os.path.exists(f0_path):
        f0_path = os.path.join(chunk_dir, "target_f0.npy")
    if not os.path.exists(f0_path):
        return None
    f0 = np.load(f0_path)

    # Load audio
    wav_path = os.path.join(chunk_dir, "generated.wav")
    if not os.path.exists(wav_path):
        wav_path = os.path.join(chunk_dir, "target.wav")
    if not os.path.exists(wav_path):
        return None
    audio, sr = sf.read(wav_path)
    audio_duration = len(audio) / sr

    # Compute RMS frames aligned with F0 hop
    rms = librosa.feature.rms(y=audio, frame_length=RMS_FRAME_LENGTH, hop_length=HOP)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Align lengths (F0 and RMS may differ by a frame or two)
    min_len = min(len(f0), len(rms_db))
    f0 = f0[:min_len]
    rms_db = rms_db[:min_len]

    # Detect gaps
    f0_mask = (f0 == 0)
    energy_mask = (rms_db < energy_thresh_db)

    f0_gaps = find_gaps(f0_mask, margin_frames, min_gap_frames)
    energy_gaps = find_gaps(energy_mask, margin_frames, min_gap_frames)

    # Cross-reference: find gaps where both signals agree
    both_mask = f0_mask & energy_mask
    combined_gaps = find_gaps(both_mask, margin_frames, min_gap_frames)

    # Classify each F0 gap
    for gap in f0_gaps:
        s, e = gap["start_frame"], gap["end_frame"]
        overlap = np.mean(energy_mask[s:e])
        gap["type"] = "both" if overlap > 0.5 else "f0_only"

    # Load notes for lyric context
    notes = []
    notes_path = os.path.join(chunk_dir, "extracted_notes.json")
    if os.path.exists(notes_path):
        with open(notes_path) as f:
            data = json.load(f)
            notes = data.get("notes", data) if isinstance(data, dict) else data

    lyrics = " ".join(n.get("note_text", "") for n in notes)

    # Annotate gaps with note context
    for gap in f0_gaps:
        mid = (gap["start_sec"] + gap["end_sec"]) / 2
        gap["during_note"] = find_note_at_time(notes, mid)

    # Severity: weight confirmed (both) gaps higher
    max_gap = max((g["duration_sec"] for g in f0_gaps), default=0)
    total_gap = sum(g["duration_sec"] for g in f0_gaps)
    both_max = max((g["duration_sec"] for g in combined_gaps), default=0)
    severity = both_max * 2 + max_gap + total_gap * 0.5

    return {
        "chunk_name": chunk_name,
        "chunk_dir": chunk_dir,
        "audio_duration_sec": round(audio_duration, 3),
        "f0_gaps": f0_gaps,
        "energy_gaps": energy_gaps,
        "combined_gaps": combined_gaps,
        "severity": round(severity, 3),
        "max_gap_sec": round(max_gap, 3),
        "total_gap_sec": round(total_gap, 3),
        "lyrics": lyrics,
        "notes": notes,
        "f0": f0,
        "rms_db": rms_db,
        "audio": audio,
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(results):
    """Print a ranked text report of silence detection results."""
    results.sort(key=lambda r: r["severity"], reverse=True)

    with_gaps = [r for r in results if r["f0_gaps"]]
    print(f"\n{'='*60}")
    print(f" Silence Detection Report")
    print(f"{'='*60}")
    print(f" Scanned: {len(results)} chunks | With internal gaps: {len(with_gaps)}")
    print(f" Sorted by severity (higher = more/longer silences)")
    print(f"{'='*60}\n")

    for i, r in enumerate(results):
        if not r["f0_gaps"]:
            continue

        both_count = sum(1 for g in r["f0_gaps"] if g.get("type") == "both")
        f0_only_count = sum(1 for g in r["f0_gaps"] if g.get("type") == "f0_only")

        print(f" #{i+1:>2}  {r['chunk_name']}  ({r['audio_duration_sec']:.2f}s)  "
              f"\"{r['lyrics'][:50]}{'...' if len(r['lyrics']) > 50 else ''}\"")
        print(f"     Severity: {r['severity']:.1f}  |  Max gap: {r['max_gap_sec']*1000:.0f}ms  |  "
              f"Total: {r['total_gap_sec']*1000:.0f}ms  |  "
              f"Gaps: {len(r['f0_gaps'])} ({both_count} confirmed, {f0_only_count} f0-only)")

        for j, gap in enumerate(r["f0_gaps"]):
            tag = "BOTH f0+energy" if gap["type"] == "both" else "f0 only"
            print(f"     Gap {j+1}: {gap['start_sec']:.3f}s - {gap['end_sec']:.3f}s "
                  f"({gap['duration_sec']*1000:.0f}ms) [{tag}] "
                  f"during \"{gap['during_note']}\"")

        print(f"     -> {os.path.join(r['chunk_dir'], 'generated.wav')}")
        print()

    if not with_gaps:
        print(" No internal silences detected with current thresholds.\n")


# ── Visualization ──────────────────────────────────────────────────────────────

def plot_chunk(result, output_dir):
    """Generate a waveform + F0 visualization for a chunk."""
    os.makedirs(output_dir, exist_ok=True)

    audio = result["audio"]
    f0 = result["f0"]
    rms_db = result["rms_db"]
    notes = result["notes"]
    f0_gaps = result["f0_gaps"]
    chunk_name = result["chunk_name"]

    time_audio = np.arange(len(audio)) / SR
    time_frames = np.arange(len(f0)) * FRAME_SEC

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"{chunk_name}  —  severity: {result['severity']:.1f}  "
                 f"—  \"{result['lyrics'][:60]}\"", fontsize=11)

    # ── Top: Waveform + RMS ──
    ax1.plot(time_audio, audio, color="lightgray", linewidth=0.5, label="Waveform")
    ax1.step(time_frames, rms_db / 100, where="post", color="steelblue",
             linewidth=1.2, label="RMS (dB/100)")
    ax1.axhline(y=ENERGY_THRESH_DB / 100, color="red", linestyle="--",
                linewidth=0.8, alpha=0.7, label=f"Threshold ({ENERGY_THRESH_DB} dB)")

    for gap in f0_gaps:
        color = "red" if gap["type"] == "both" else "orange"
        ax1.axvspan(gap["start_sec"], gap["end_sec"], alpha=0.25, color=color)

    ax1.set_ylabel("Amplitude / RMS")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(-1.0, 1.0)

    # ── Bottom: F0 + note boundaries ──
    f0_voiced = f0.copy().astype(float)
    f0_voiced[f0_voiced == 0] = np.nan
    ax2.plot(time_frames, f0_voiced, color="steelblue", linewidth=1.2, label="F0 (Hz)")

    for note in notes:
        start = note["start_s"]
        end = start + note["note_dur"]
        ax2.axvline(x=start, color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
        mid = (start + end) / 2
        ax2.text(mid, ax2.get_ylim()[0] if ax2.get_ylim()[0] > 0 else 50,
                 note.get("note_text", ""), ha="center", va="bottom",
                 fontsize=8, color="black",
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))

    for gap in f0_gaps:
        color = "red" if gap["type"] == "both" else "orange"
        ax2.axvspan(gap["start_sec"], gap["end_sec"], alpha=0.25, color=color)

    ax2.set_ylabel("F0 (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"detect_silences_{chunk_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect silences in generated audio chunks")
    parser.add_argument("--data_dir", default=os.path.normpath(DEFAULT_DATA_DIR),
                        help="Path to song data directory containing line_* chunks")
    parser.add_argument("--min_gap_ms", type=int, default=60,
                        help="Minimum gap duration in ms (default: 60)")
    parser.add_argument("--energy_threshold_db", type=float, default=ENERGY_THRESH_DB,
                        help=f"RMS silence threshold in dB (default: {ENERGY_THRESH_DB})")
    parser.add_argument("--top_n", type=int, default=TOP_N,
                        help=f"Number of worst chunks to visualize (default: {TOP_N})")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for plots (default: <data_dir>/silence_report)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip generating visualization plots")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir or os.path.join(data_dir, "silence_report")
    min_gap_frames = max(1, int(args.min_gap_ms / (FRAME_SEC * 1000)))

    # Discover chunks
    chunk_dirs = sorted(
        glob.glob(os.path.join(data_dir, "line_*")),
        key=lambda p: int(re.search(r"line_(\d+)", p).group(1))
    )

    if not chunk_dirs:
        print(f"No line_* directories found in {data_dir}")
        return

    print(f"Scanning {len(chunk_dirs)} chunks in {data_dir}...")
    print(f"Settings: min_gap={args.min_gap_ms}ms ({min_gap_frames} frames), "
          f"energy_threshold={args.energy_threshold_db}dB\n")

    # Analyze all chunks
    results = []
    for chunk_dir in chunk_dirs:
        result = analyze_chunk(chunk_dir, min_gap_frames, MARGIN_FRAMES,
                               args.energy_threshold_db)
        if result is not None:
            results.append(result)

    if not results:
        print("No chunks with required files (post_f0.npy + generated.wav) found.")
        return

    # Report
    print_report(results)

    # Visualize top N
    if not args.no_plots:
        ranked = sorted(results, key=lambda r: r["severity"], reverse=True)
        to_plot = [r for r in ranked if r["f0_gaps"]][:args.top_n]

        if to_plot:
            print(f"Generating plots for top {len(to_plot)} chunks...")
            for r in to_plot:
                path = plot_chunk(r, output_dir)
                print(f"  Saved: {path}")
            print(f"\nAll plots saved to {output_dir}")
        else:
            print("No chunks with gaps to visualize.")


if __name__ == "__main__":
    main()
