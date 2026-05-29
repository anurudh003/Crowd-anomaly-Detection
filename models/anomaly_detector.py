"""
Crowd Anomaly Detection - Core Model
Architecture: CNN (MobileNetV2) + LSTM + Autoencoder
"""

import torch
import torch.nn as nn
import torchvision.models as models
import json
import os


class LiteAnomalyDetector(nn.Module):
    """
    Full CNN-LSTM Autoencoder for crowd anomaly detection.
    Normal behavior -> low reconstruction error.
    Anomalous behavior -> high reconstruction error.
    """
    def __init__(self, config_path: str = "config.json"):
        super().__init__()
        # Use exact architecture from Colab to ensure state_dict loads correctly
        base = models.mobilenet_v2(weights=None)
        self.cnn = nn.Sequential(*list(base.features.children()), nn.AdaptiveAvgPool2d((1,1)))
        self.proj = nn.Linear(1280, 512)
        self.enc_lstm = nn.LSTM(512, 256, 2, batch_first=True, dropout=0.3)
        self.dec_lstm = nn.LSTM(256, 256, 2, batch_first=True, dropout=0.3)
        self.dec_proj = nn.Linear(256, 512)
        self.head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def extract_features(self, x):
        """Extract CNN features for a single frame or batch of frames."""
        # x: (B, C, H, W)
        f = self.cnn(x)
        f = f.view(f.size(0), -1)
        f = torch.relu(self.proj(f))
        return f

    def forward_features(self, f):
        """Process a sequence of features through LSTM/Autoencoder."""
        # f: (B, T, 512)
        B, T, D = f.shape
        _, (h, _) = self.enc_lstm(f)
        z = h[-1]
        d, _ = self.dec_lstm(z.unsqueeze(1).repeat(1, T, 1))
        recon = self.dec_proj(d)
        err = ((f - recon)**2).mean(dim=(1,2))
        return self.head(z).squeeze(-1), err

    def forward(self, x):
        """Full forward pass (kept for compatibility)."""
        B, T, C, H, W = x.shape
        f = self.extract_features(x.view(B*T, C, H, W)).view(B, T, -1)
        return self.forward_features(f)

    def predict(self, x, threshold: float = 0.5):
        self.eval()
        with torch.no_grad():
            score, recon_err = self.forward(x)
        return {
            "anomaly_score": score.cpu().numpy(),
            "reconstruction_error": recon_err.cpu().numpy(),
            "is_anomaly": (score > threshold).cpu().numpy().astype(bool),
        }

if __name__ == "__main__":
    model = LiteAnomalyDetector()
    dummy = torch.randn(2, 8, 3, 120, 160)
    score, err = model(dummy)
    print(f"Anomaly scores: {score.detach().numpy()}")
    print(f"Reconstruction errors: {err.detach().numpy()}")
    print("Model forward pass OK!")
