import time
import torch
import cv2
import json
import argparse
from app import InferencePipeline

def measure_latency(pipeline, video_path, num_frames=100):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
    
    times = {
        "preprocessing": [],
        "heatmap": [],
        "cnn_lstm_inference": [],
        "yolo_inference": [],
        "total": []
    }
    
    frame_id = 0
    while cap.isOpened() and frame_id < num_frames:
        ret, frame = cap.read()
        if not ret: break
        
        t0 = time.time()
        
        # Heatmap
        t_heat0 = time.time()
        pipeline.heatmap.process_frame(frame, return_overlay=False)
        times["heatmap"].append(time.time() - t_heat0)
        
        # Preprocessing
        t_prep0 = time.time()
        tensor = pipeline.preprocess_frame(frame)
        pipeline.frame_buffer.append(tensor)
        times["preprocessing"].append(time.time() - t_prep0)
        
        # CNN-LSTM Inference
        if frame_id % pipeline.seq_len == 0:
            t_cnn0 = time.time()
            pipeline.get_anomaly_score()
            times["cnn_lstm_inference"].append(time.time() - t_cnn0)
            
        # YOLO inference
        if frame_id % int(pipeline.cfg["inference"].get("yolo_every_n_frames", 10)) == 0:
            t_yolo0 = time.time()
            small = cv2.resize(frame, (320, 240))
            pipeline.obj_tracker.process_frame(small, frame_id)
            times["yolo_inference"].append(time.time() - t_yolo0)
            
        times["total"].append(time.time() - t0)
        frame_id += 1
        
    cap.release()
    
    print("\n--- Latency Benchmark ---")
    for k, v in times.items():
        if len(v) > 0:
            avg_ms = (sum(v) / len(v)) * 1000
            print(f"{k}: {avg_ms:.2f} ms")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="ucsd_test_001.mp4")
    args = parser.parse_args()
    
    print("Loading pipeline...")
    pipeline = InferencePipeline()
    print("Starting benchmark...")
    measure_latency(pipeline, args.video)
