"""
Crowd Anomaly Detection - Crowd Density Heatmap Module
Algorithm: MOG2 background subtraction + morphological ops + Gaussian blur
"""

import cv2
import numpy as np
import json
from typing import Tuple, Optional


class CrowdDensityHeatmap:
    """
    Module 2: Real-time crowd density heatmap generator.

    Pipeline:
        1. MOG2 background subtraction → foreground mask
        2. Morphological closing   → fill holes
        3. Gaussian blur            → smooth density map
        4. Colormap overlay         → blue (sparse) → red (dense)
        5. Running accumulator      → temporal smoothing
    """

    def __init__(self, config_path: str = "config.json"):
        with open(config_path) as f:
            cfg = json.load(f)

        hm_cfg = cfg["heatmap"]
        self.history_length = hm_cfg.get("history_length", 30)
        self.morph_kernel   = hm_cfg.get("morphology_kernel", 5)
        self.blur_kernel    = hm_cfg.get("gaussian_blur_kernel", 15)

        # MOG2 background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history_length,
            varThreshold=16,
            detectShadows=False,
        )

        # Morphological kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel, self.morph_kernel),
        )

        # Running accumulator for temporal smoothing
        self.accumulator: Optional[np.ndarray] = None
        self.alpha = 0.1  # blend factor: higher = faster response

    def process_frame(self, frame: np.ndarray, return_overlay: bool = True) -> Tuple[Optional[np.ndarray], float]:
        """
        Process a single BGR frame.

        Returns:
            heatmap_overlay: (H, W, 3) BGR frame with colored heatmap overlay (or None if return_overlay is False)
            density_score:   float in [0, 1] — how crowded the scene is
        """
        h, w = frame.shape[:2]

        # 1. Background subtraction → foreground mask
        fg_mask = self.bg_subtractor.apply(frame)  # (H, W), uint8

        # 2. Morphological closing (fill holes)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        # 3. Gaussian blur → smooth density
        ks = self.blur_kernel
        density = cv2.GaussianBlur(fg_mask, (ks, ks), 0).astype(np.float32)

        # 4. Normalize to [0, 1]
        max_val = density.max()
        if max_val > 0:
            density /= max_val

        # 5. Temporal accumulation
        if self.accumulator is None:
            self.accumulator = density.copy()
        else:
            self.accumulator = (1 - self.alpha) * self.accumulator + self.alpha * density

        # 6. Colormap & Overlay (skip if not needed for speed)
        overlay = None
        if return_overlay:
            hm_uint8  = (self.accumulator * 255).clip(0, 255).astype(np.uint8)
            hm_color  = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 0.6, hm_color, 0.4, 0)

        # 7. Density score = mean of accumulator
        density_score = float(self.accumulator.mean())

        return overlay, density_score

    def reset(self):
        """Reset state (use between videos)."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history_length, varThreshold=16, detectShadows=False
        )
        self.accumulator = None

    def is_overcrowded(self, density_score: float, threshold: float = 0.75) -> bool:
        """Returns True if crowd density exceeds threshold."""
        return density_score > threshold


if __name__ == "__main__":
    # Quick test: run on webcam or video file
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else 0

    heatmap = CrowdDensityHeatmap()
    cap = cv2.VideoCapture(source)

    print(f"Running heatmap on source: {source} — press 'q' to quit")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        overlay, score = heatmap.process_frame(frame)
        cv2.putText(overlay, f"Density: {score:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Crowd Density Heatmap", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
