"""
Crowd Anomaly Detection - Data Preprocessing
Handles the UCSD Pedestrian Dataset format:
  - Frames stored as .tif images inside per-clip subdirectories
  - Structure: <split>/<ClipName>/<frame_number>.tif
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Frame I/O  (supports .tif, .png, .jpg, .bmp)
# ─────────────────────────────────────────────────────────────────────────────

FRAME_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def preprocess_frame(frame: np.ndarray,
                     target_size: Tuple[int, int] = (160, 120)) -> np.ndarray:
    """
    Resize and normalize a single BGR/grayscale frame.
    Returns (H, W, 3) float32 in [0, 1].
    """
    if frame is None:
        raise ValueError("preprocess_frame received None")
    # Convert grayscale → RGB if needed
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(frame, target_size)          # (H, W, 3)
    return frame.astype(np.float32) / 255.0


def load_frames_from_folder(folder: Path,
                             target_size: Tuple[int, int] = (160, 120)
                             ) -> List[np.ndarray]:
    """
    Load all image frames from a folder, sorted by filename.
    Skips hidden/system files (DS_Store, etc.).
    Returns list of (H, W, 3) float32 arrays.
    """
    image_files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in FRAME_EXTS and not p.name.startswith(".")
    )
    frames = []
    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        frames.append(preprocess_frame(img, target_size))
    return frames


def extract_frames_from_video(video_path: str,
                               target_size: Tuple[int, int] = (160, 120),
                               max_frames: Optional[int] = None
                               ) -> List[np.ndarray]:
    """Load frames from a video file (mp4/avi). Returns list of (H,W,3) float32."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame, target_size))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Sequence Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_sequences(frames: List[np.ndarray],
                       sequence_length: int = 8,
                       stride: int = 4) -> np.ndarray:
    """
    Slide a window over frames to generate (N, T, H, W, 3) sequences.
    """
    if len(frames) < sequence_length:
        return np.array([])
    sequences = []
    for start in range(0, len(frames) - sequence_length + 1, stride):
        seq = np.stack(frames[start: start + sequence_length], axis=0)
        sequences.append(seq)
    return np.array(sequences, dtype=np.float32)  # (N, T, H, W, 3)


# ─────────────────────────────────────────────────────────────────────────────
# UCSD-Specific Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_ucsd(raw_root: str,
                    output_dir: str,
                    config_path: str = "config.json") -> dict:
    """
    Full preprocessing pipeline for the UCSD Pedestrian dataset.

    Expected layout (auto-detected):
        raw_root/
          UCSD_Anomaly_Dataset.v1p2/
            UCSDped1/
              Train/
                Train001/  ← folder of .tif frames
                Train002/
                ...
              Test/
                Test001/
                Test001_gt/   ← ground truth (skipped for training)
            UCSDped2/
              ...

    Saves .npy files to:
        output_dir/ped1/train/
        output_dir/ped1/test/
        output_dir/ped2/train/
        output_dir/ped2/test/

    Returns dict with stats.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    ds_cfg = cfg["dataset"]
    target_size = (ds_cfg["frame_width"], ds_cfg["frame_height"])  # (W, H) for cv2
    seq_len = ds_cfg["sequence_length"]
    stride  = ds_cfg["stride"]

    # Auto-locate the v1p2 root
    raw_root = Path(raw_root)
    candidates = list(raw_root.rglob("UCSDped1"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find UCSDped1 inside {raw_root}. "
            "Make sure the dataset is extracted correctly."
        )
    dataset_root = candidates[0].parent   # ...UCSD_Anomaly_Dataset.v1p2/

    stats = {"splits": {}}
    total_sequences = 0

    for ped in ["UCSDped1", "UCSDped2"]:
        ped_path = dataset_root / ped
        if not ped_path.exists():
            print(f"  ⚠ {ped} not found, skipping.")
            continue

        for split in ["Train", "Test"]:
            split_path = ped_path / split
            if not split_path.exists():
                continue

            # Get clip folders (skip _gt folders and hidden files)
            clip_dirs = sorted(
                d for d in split_path.iterdir()
                if d.is_dir()
                and not d.name.endswith("_gt")
                and not d.name.startswith(".")
            )

            tag = f"{ped.lower()}/{split.lower()}"
            out_path = Path(output_dir) / ped.lower() / split.lower()
            out_path.mkdir(parents=True, exist_ok=True)

            clip_seq_count = 0
            print(f"\n📂 {tag} — {len(clip_dirs)} clips")

            for clip_dir in tqdm(clip_dirs, desc=f"  {tag}"):
                frames = load_frames_from_folder(clip_dir, target_size)
                if not frames:
                    continue
                seqs = generate_sequences(frames, seq_len, stride)
                if seqs.size == 0:
                    continue

                save_path = out_path / f"{clip_dir.name}.npy"
                np.save(str(save_path), seqs)
                clip_seq_count += seqs.shape[0]

            stats["splits"][tag] = clip_seq_count
            total_sequences += clip_seq_count
            print(f"  ✅ {clip_seq_count} sequences → {out_path}")

    stats["total_sequences"] = total_sequences
    print(f"\n🎉 Preprocessing complete! Total sequences: {total_sequences}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class UCSDSequenceDataset(Dataset):
    """
    Loads pre-generated .npy sequence files.
    Each item: (T, C, H, W) float32 tensor (ImageNet-normalized).
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, sequences_dir: str, label: int = 0, augment: bool = False):
        self.label = label
        npy_files = sorted(Path(sequences_dir).glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files in {sequences_dir}")

        parts = [np.load(str(f)) for f in npy_files]
        self.sequences = np.concatenate(parts, axis=0)   # (N, T, H, W, 3)

        self.normalize = T.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)
        self.flip = T.RandomHorizontalFlip(0.5) if augment else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx]).permute(0, 3, 1, 2)  # (T,3,H,W)
        seq = torch.stack([self.normalize(f) for f in seq])
        if self.flip:
            seq = torch.stack([self.flip(f) for f in seq])
        return seq, torch.tensor(self.label, dtype=torch.long)


def build_dataloaders(train_dir: str, val_dir: str,
                      config_path: str = "config.json"):
    with open(config_path) as f:
        cfg = json.load(f)
    bs = cfg["training"]["batch_size"]
    train_ds = UCSDSequenceDataset(train_dir, label=0, augment=True)
    val_ds   = UCSDSequenceDataset(val_dir,   label=0, augment=False)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} sequences  |  Val: {len(val_ds)} sequences")
    return train_dl, val_dl
