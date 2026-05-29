# COLAB NOTEBOOK FOR CROWD ANOMALY DETECTION TRAINING
# Optimized for Preprocessed .npy Sequences
# Copy this entire script and paste into a single Google Colab cell.

# ============================================================================
# CELL 1: SETUP & IMPORTS
# ============================================================================
import os
import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Check GPU availability
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# CELL 2: MOUNT GOOGLE DRIVE
# ============================================================================
from google.colab import drive
drive.mount('/content/gdrive')

# Create project directories
project_dir = '/content/gdrive/MyDrive/crowd_anomaly_colab'
model_dir = f'{project_dir}/models'
output_dir = f'{project_dir}/outputs'

os.makedirs(model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

print(f"✓ Project directory: {project_dir}")

# ============================================================================
# CELL 3: EXTRACT PREPROCESSED DATA
# ============================================================================
# Expected: /content/gdrive/MyDrive/crowd_anomaly_colab/sequences.zip
# contains: ped1/train/*.npy, ped1/test/*.npy, etc.

zip_path = f'{project_dir}/sequences.zip'
extract_path = '/content/sequences'

if not os.path.exists(extract_path):
    if os.path.exists(zip_path):
        print("Extracting preprocessed sequences... (this may take a minute)")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("✓ Extraction complete")
    else:
        print(f"❌ ERROR: {zip_path} not found in Google Drive!")
        print("Please upload your 'sequences.zip' to 'crowd_anomaly_colab' folder in Drive.")
else:
    print("✓ Sequences already extracted at /content/sequences")

# ============================================================================
# CELL 4: DATASET & DATALOADER
# ============================================================================

class UCSDSequenceDataset(Dataset):
    """
    Loads pre-generated .npy sequence files.
    Each item: (T, C, H, W) float32 tensor (ImageNet-normalized).
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, root_dir, ped_types=['ucsdped1', 'ucsdped2'], split='train', augment=False):
        self.split = split
        self.augment = augment
        self.sequences = []
        
        print(f"Loading {split} data from {root_dir}...")
        for ped in ped_types:
            split_dir = Path(root_dir) / ped / split
            if not split_dir.exists():
                # Try title case if not found
                split_dir = Path(root_dir) / ped / split.capitalize()
            
            if split_dir.exists():
                npy_files = sorted(split_dir.glob("*.npy"))
                for f in tqdm(npy_files, desc=f"  {ped}/{split}"):
                    data = np.load(str(f)) # (N, T, H, W, 3)
                    self.sequences.append(data)
        
        if not self.sequences:
            raise FileNotFoundError(f"No .npy files found for {split} split!")
            
        self.sequences = np.concatenate(self.sequences, axis=0) # (Total_N, T, H, W, 3)
        print(f"  ✓ Loaded {len(self.sequences)} sequences for {split}")

        self.normalize = T.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)
        self.flip = T.RandomHorizontalFlip(0.5) if augment else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # seq: (T, H, W, 3) float32 [0, 1]
        seq = torch.from_numpy(self.sequences[idx]).permute(0, 3, 1, 2)  # (T, 3, H, W)
        
        # Normalize each frame
        seq = torch.stack([self.normalize(f) for f in seq])
        
        if self.flip:
            seq = torch.stack([self.flip(f) for f in seq])
            
        return seq

# Build DataLoaders
# Using both ped1 and ped2 training data for better generalization
train_ds = UCSDSequenceDataset(extract_path, split='train', augment=True)
val_ds   = UCSDSequenceDataset(extract_path, split='test', augment=False) # Using test split for validation

batch_size = 32 # Colab GPU can handle larger batch sizes
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

# ============================================================================
# CELL 5: MODEL ARCHITECTURE
# ============================================================================

class CNNEncoder(nn.Module):
    """MobileNetV2 spatial feature extractor. Input: (B,3,120,160) → Output: (B,512)"""
    def __init__(self, pretrained=True):
        super().__init__()
        base = torchvision.models.mobilenet_v2(weights='DEFAULT' if pretrained else None)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(1280, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.dropout(self.relu(self.proj(x)))

class TemporalEncoder(nn.Module):
    """LSTM temporal encoder. Input: (B,T,512) → Output: (B,256)"""
    def __init__(self, input_size=512, hidden_size=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]

class TemporalDecoder(nn.Module):
    """LSTM temporal decoder. Input: (B,256) → Output: (B,T,512)"""
    def __init__(self, hidden_size=256, output_size=512, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, z, seq_len=8):
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm(z_repeated)
        return self.proj(out)

class LiteAnomalyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_encoder = CNNEncoder(pretrained=True)
        self.temporal_encoder = TemporalEncoder()
        self.temporal_decoder = TemporalDecoder()
        self.anomaly_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # Encode each frame
        frame_feats = self.cnn_encoder(x.view(B * T, C, H, W))
        frame_feats = frame_feats.view(B, T, -1) # (B, T, 512)
        
        # Temporal encode
        z = self.temporal_encoder(frame_feats) # (B, 256)
        
        # Decode/Reconstruct
        recon_feats = self.temporal_decoder(z, seq_len=T) # (B, T, 512)
        
        # Recon error
        recon_error = ((frame_feats - recon_feats) ** 2).mean(dim=(1, 2))
        
        # Anomaly score
        anomaly_score = self.anomaly_head(z).squeeze(-1)
        
        return anomaly_score, recon_error

model = LiteAnomalyDetector().to(device)
print("✓ Model initialized on", device)

# ============================================================================
# CELL 6: TRAINING SETUP
# ============================================================================
criterion = nn.MSELoss() # Reconstruction loss
# For the anomaly_head, we'll use a dummy target of 0 (normal) for training on normal data
# But primarily we train the Autoencoder to reconstruct normal behavior.

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# ============================================================================
# CELL 7: TRAINING LOOP
# ============================================================================
num_epochs = 30
best_val_loss = float('inf')

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch = batch.to(device)
        
        # Forward
        scores, recon_error = model(batch)
        
        # Loss: We want low reconstruction error for normal data
        loss = recon_error.mean()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            _, recon_error = model(batch)
            val_loss += recon_error.mean().item()
    
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step()
    
    print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f'{model_dir}/anomaly_detector.pth')
        print("  ✓ Saved best model")

# ============================================================================
# CELL 8: THRESHOLD & EXPORT
# ============================================================================
print("\nComputing final anomaly threshold...")
model.load_state_dict(torch.load(f'{model_dir}/anomaly_detector.pth'))
model.eval()

all_errors = []
with torch.no_grad():
    for batch in train_loader:
        batch = batch.to(device)
        _, recon_error = model(batch)
        all_errors.extend(recon_error.cpu().numpy())

all_errors = np.array(all_errors)
threshold = np.mean(all_errors) + 2 * np.std(all_errors)

print(f"Final Threshold: {threshold:.6f}")

# Update config.json
config = {
    "inference": {
        "anomaly_threshold": float(threshold),
        "fps_target": 10,
        "device": "cpu",
        "map_location": "cpu"
    }
}
with open(f'{model_dir}/config_updated.json', 'w') as f:
    json.dump(config, f, indent=4)

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print("Files to download from Google Drive:")
print(f"1. {model_dir}/anomaly_detector.pth")
print(f"2. {model_dir}/config_updated.json")
print("="*50)
