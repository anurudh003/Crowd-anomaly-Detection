import cv2
import os
from pathlib import Path

def folder_to_video(folder_path, output_name="test_video.mp4", fps=10):
    images = sorted([img for img in os.listdir(folder_path) if img.endswith(".tif") or img.endswith(".png")])
    if not images:
        print(f"No images found in {folder_path}")
        return

    first_img = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, layers = first_img.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    print(f"Converting {len(images)} frames to {output_name}...")
    for image in images:
        img_path = os.path.join(folder_path, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"DONE: Video saved as: {output_name}")

if __name__ == "__main__":
    # Add 12 videos from UCSDped1/Test
    ped1_base = "data/raw/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"
    
    samples = [
        ("Test009", "ucsd_ped1_009.mp4"),
        ("Test010", "ucsd_ped1_010.mp4"),
        ("Test011", "ucsd_ped1_011.mp4"),
        ("Test012", "ucsd_ped1_012.mp4"),
        ("Test013", "ucsd_ped1_013.mp4"),
        ("Test014", "ucsd_ped1_014.mp4"),
        ("Test015", "ucsd_ped1_015.mp4"),
        ("Test016", "ucsd_ped1_016.mp4"),
        ("Test017", "ucsd_ped1_017.mp4"),
        ("Test018", "ucsd_ped1_018.mp4"),
        ("Test019", "ucsd_ped1_019.mp4"),
        ("Test020", "ucsd_ped1_020.mp4"),
    ]

    for folder, output in samples:
        full_path = os.path.join(ped1_base, folder)
        if os.path.exists(full_path):
            folder_to_video(full_path, output)
        else:
            print(f"Skipping {folder}, path not found.")
