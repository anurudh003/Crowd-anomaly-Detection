# Failure Case Analysis: UCSD Pedestrian Dataset

This document provides a detailed breakdown of the model's false positive and false negative occurrences based on the recent evaluation metrics.

## Evaluation Summary
- **AUC-ROC**: 0.7564
- **Optimal Threshold**: 0.31
- **Precision**: 67.66%
- **Recall**: 85.96%
- **Confusion Matrix**:
  - True Negatives (Normal): 1493
  - False Positives (False Alarms): 1662
  - False Negatives (Missed Anomalies): 568
  - True Positives (Detected Anomalies): 3477

## Analysis of False Positives (1662 frames)
The high number of False Positives indicates that the model frequently flags normal pedestrian behavior as anomalous. Based on the integration of Motion Energy and Reconstruction Error, the primary causes are:

1.  **Dense Crowd Artifacts (Motion Overlap)**:
    *   *Issue*: When pedestrians overlap significantly (e.g., passing each other closely in opposite directions), the optical flow and motion energy spike artificially.
    *   *Model Reaction*: The reconstruction error from the CNN-LSTM autoencoder struggles to perfectly recreate these overlapping human shapes, resulting in a high anomaly score.
    *   *Mitigation Implemented*: Added Temporal Smoothing (moving average) and Gaussian Blur to soften these hard edges.

2.  **Environmental Shadows & Lighting Changes**:
    *   *Issue*: The UCSD Ped1 dataset contains walkways with trees that cast shifting shadows depending on the wind and time of day.
    *   *Model Reaction*: Rapid shadow movement triggers the `motion_score` heuristic, pushing the blended score above the 0.31 threshold.
    *   *Mitigation Implemented*: Applying CLAHE (Contrast Limited Adaptive Histogram Equalization) before motion extraction helps, but strong shadows still trigger minor false alarms.

3.  **Proximity to Camera (Scale Variation)**:
    *   *Issue*: Pedestrians walking very close to the camera occupy a much larger percentage of the frame.
    *   *Model Reaction*: The raw pixel difference between frames is exponentially higher for people close to the lens compared to people further down the walkway.

## Analysis of False Negatives (568 frames)
False negatives occur when the model fails to detect an actual anomaly (e.g., a bicycle, skateboarder, or small cart).

1.  **Slow-Moving Anomalies**:
    *   *Issue*: If a bicyclist or wheelchair is moving at the exact same pace as the surrounding pedestrians.
    *   *Model Reaction*: The `motion_score` remains low, and the autoencoder can sometimes recreate the shape well enough to stay below the threshold.
    *   *Mitigation Implemented*: The integration of the YOLOv5 object tracker ensures that even if the object is slow, if it is classified as a 'bicycle' or 'truck', the anomaly score is instantly boosted to 1.0.

2.  **Occlusion**:
    *   *Issue*: Small carts or skateboards being pushed behind a dense group of pedestrians.
    *   *Model Reaction*: The YOLO model cannot detect the object due to occlusion, and the CNN-LSTM focuses on recreating the prominent human figures, ignoring the small pixel variations of the skateboard.

## Conclusion
The current threshold of **0.31** was chosen to maximize the F1-Score, heavily favoring Recall (Safety First). In a real-world deployment, if the 1662 False Positives cause "alert fatigue" for the security operators, the threshold should be raised slightly (e.g., to 0.40) to trade a small amount of Recall for a massive gain in Precision.
