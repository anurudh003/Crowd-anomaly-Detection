"""
Crowd Anomaly Detection - Core Model
Architecture: CNN (MobileNetV2) + LSTM + Autoencoder
Author: Crowd Anomaly Detection Project
"""

import torch
import torch.nn as nn
import torchvision.models as models
import json
import os


class CNNEncoder(nn.Module):
    """
    MobileNetV2-based spatial feature extractor.
    Extracts spatial features from individual frames.
    Input:  (B, C, H, W) = (batch, 3, 120, 160)
    Output: (B, 512) feature vector per frame
    """

    def __init__(self, pretrained: bool = True):
        super(CNNEncoder, self).__init__()
        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        # Use everything except the classifier head
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(1280, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.features(x)       # (B, 1280, H', W')
        x = self.pool(x)           # (B, 1280, 1, 1)
        x = x.flatten(1)           # (B, 1280)
        x = self.relu(self.proj(x))  # (B, 512)
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    """
    LSTM-based temporal encoder.
    Processes a sequence of CNN features to capture motion patterns.
    Input:  (B, T, 512) — T = sequence_length frames
    Output: (B, 256) — compressed temporal code
    """

    def __init__(self, input_size: int = 512, hidden_size: int = 256, num_layers: int = 2):
        super(TemporalEncoder, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )

    def forward(self, x):
        # x: (B, T, 512)
        # Apply BN to each time step
        B, T, C = x.shape
        x = x.view(B * T, C)
        x = self.bn(x)
        x = x.view(B, T, C)
        
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, B, hidden) → take last layer
        return h_n[-1]  # (B, 256)


class TemporalDecoder(nn.Module):
    """
    LSTM-based temporal decoder.
    Reconstructs the sequence from the compressed code.
    Input:  (B, 256) — latent code + sequence_length
    Output: (B, T, 512) — reconstructed CNN feature sequence
    """

    def __init__(self, hidden_size: int = 256, output_size: int = 512, num_layers: int = 2):
        super(TemporalDecoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, z, seq_len: int = 8):
        # Repeat latent code across time steps
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, 256)
        out, _ = self.lstm(z_repeated)                       # (B, T, 256)
        return self.proj(out)                                 # (B, T, 512)


class LiteAnomalyDetector(nn.Module):
    """
    Full CNN-LSTM Autoencoder for crowd anomaly detection.

    Normal behavior → low reconstruction error
    Anomalous behavior → high reconstruction error

    Input:  sequence of frames (B, T, C, H, W)
    Output: anomaly score (scalar per sample), reconstructed sequence
    """

    def __init__(self, config_path: str = "config.json"):
        super(LiteAnomalyDetector, self).__init__()
        cfg = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)

        model_cfg = cfg.get("model", {})
        pretrained = model_cfg.get("pretrained", True)
        latent_dim = model_cfg.get("latent_dim", 512)
        lstm_hidden = model_cfg.get("lstm_hidden", 256)
        lstm_layers = model_cfg.get("lstm_layers", 2)

        self.cnn_encoder = CNNEncoder(pretrained=pretrained)
        self.temporal_encoder = TemporalEncoder(latent_dim, lstm_hidden, lstm_layers)
        self.temporal_decoder = TemporalDecoder(lstm_hidden, latent_dim, lstm_layers)

        # Anomaly score head: projects latent code to scalar anomaly score
        self.anomaly_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def encode_sequence(self, x):
        """
        x: (B, T, C, H, W)
        Returns: (B, T, latent_dim) CNN features for each frame
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn_encoder(x)       # (B*T, 512)
        return feats.view(B, T, -1)       # (B, T, 512)

    def forward(self, x):
        """
        x: (B, T, C, H, W) — batch of frame sequences
        Returns:
            anomaly_score: (B,) — 0 = normal, 1 = anomalous
            recon_error:   (B,) — mean squared reconstruction error
        """
        # Encode each frame
        frame_feats = self.encode_sequence(x)   # (B, T, 512)

        # Temporal encode → latent code
        z = self.temporal_encoder(frame_feats)  # (B, 256)

        # Decode → reconstruct features
        T = x.shape[1]
        recon_feats = self.temporal_decoder(z, seq_len=T)  # (B, T, 512)

        # Reconstruction error (MSE across time and feature dims)
        recon_error = ((frame_feats - recon_feats) ** 2).mean(dim=(1, 2))  # (B,)

        # Anomaly score from latent code
        anomaly_score = self.anomaly_head(z).squeeze(-1)   # (B,)

        return anomaly_score, recon_error

    def predict(self, x, threshold: float = 0.5):
        """
        Single-step inference. Returns dict with scores and binary prediction.
        x: (B, T, C, H, W)
        """
        self.eval()
        with torch.no_grad():
            score, recon_err = self.forward(x)
        return {
            "anomaly_score": score.cpu().numpy(),
            "reconstruction_error": recon_err.cpu().numpy(),
            "is_anomaly": (score > threshold).cpu().numpy().astype(bool),
        }


if __name__ == "__main__":
    # Quick smoke test
    model = LiteAnomalyDetector()
    dummy = torch.randn(2, 8, 3, 120, 160)  # batch=2, seq=8, 3-ch, 120×160
    score, err = model(dummy)
    print(f"Anomaly scores: {score}")
    print(f"Reconstruction errors: {err}")
    print("✅ Model forward pass OK")
