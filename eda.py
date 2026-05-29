"""
EDA (Exploratory Data Analysis) for UCSD Pedestrian Dataset
Run: python eda.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from pathlib import Path
from utils.preprocessing import load_frames_from_folder, generate_sequences

DATASET_ROOT = Path("data/raw/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2")


def inspect_clip(clip_path: Path, label: str = ""):
    frames = load_frames_from_folder(clip_path, target_size=(160, 120))
    if not frames:
        print(f"  ⚠ No frames found in {clip_path}")
        return
    seqs = generate_sequences(frames, sequence_length=8, stride=4)
    print(f"  {label}")
    print(f"    Frames loaded : {len(frames)}")
    print(f"    Frame shape   : {frames[0].shape}  dtype={frames[0].dtype}")
    print(f"    Pixel range   : min={frames[0].min():.3f}  max={frames[0].max():.3f}")
    print(f"    Sequences (8f,stride=4): {seqs.shape[0]}")
    return frames


def main():
    print("=" * 60)
    print("  UCSD Anomaly Detection — EDA Report")
    print("=" * 60)

    for ped in ["UCSDped1", "UCSDped2"]:
        ped_path = DATASET_ROOT / ped
        if not ped_path.exists():
            print(f"\n❌ {ped} not found.")
            continue

        print(f"\n[DIR] {ped}")
        for split in ["Train", "Test"]:
            split_path = ped_path / split
            if not split_path.exists():
                continue
            clip_dirs = sorted(
                d for d in split_path.iterdir()
                if d.is_dir() and not d.name.endswith("_gt") and not d.name.startswith(".")
            )
            print(f"\n  [{split}] - {len(clip_dirs)} clips")

            total_frames = 0
            total_seqs   = 0
            for d in clip_dirs:
                frames = load_frames_from_folder(d)
                seqs   = generate_sequences(frames)
                total_frames += len(frames)
                total_seqs   += (seqs.shape[0] if seqs.size else 0)

            print(f"    Total frames    : {total_frames}")
            print(f"    Total sequences : {total_seqs}  (8 frames, stride=4)")

    # Visual check — show 5 frames from first training clip of ped1
    print("\n\n[VISUAL] Visual check: first 5 frames of UCSDped1/Train/Train001")
    first_clip = DATASET_ROOT / "UCSDped1" / "Train" / "Train001"
    if first_clip.exists():
        frames = load_frames_from_folder(first_clip)
        for i, f in enumerate(frames[:5]):
            bgr = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow(f"Frame {i+1}", bgr)
        print("  Showing frames in windows... press any key in a window to close all.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("  Clip folder not found — skipping visual check.")

    print("\n[DONE] EDA complete. Ready to run preprocessing!\n")


if __name__ == "__main__":
    main()
