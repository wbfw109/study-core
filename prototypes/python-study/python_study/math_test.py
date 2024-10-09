#%%
import threading
import time
import tkinter as tk
from typing import Optional
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import numpy as np
import logging
import os
import pickle
from http.client import HTTPResponse
from pathlib import Path
from urllib.request import Request, urlopen
import threading
import time
import tkinter as tk
from typing import Optional
#%%
test_dir = Path.home() / "Downloads/2024_EfficientGCN-B4_ntu-xset120.pth/archive"
file_dir = test_dir / "data.pkl"
with file_dir.open("rb") as file:
    data = pickle.load(file)
data



# %%
### requirees package: imageio[ffmpeg]
# 🆚 Sleep (or equivalent) vs waitKey ;  https://answers.opencv.org/question/228472/sleep-or-equivalent-vs-waitkey/
from pathlib import Path

import imageio
from pytubefix import YouTube
from pytubefix.cli import on_progress
from ultralytics.engine.results import Boxes
# Directory setup
remote_video_save_directory: Path = Path.home() / "remote_videos"
remote_video_save_directory.mkdir(parents=True, exist_ok=True)
captured_video_save_directory: Path = Path.home() / "remote_videos" / "captured"
captured_video_save_directory.mkdir(parents=True, exist_ok=True)
#%%
# YouTube video URL
youtube_url = "https://youtube.com/shorts/joJouBXdpys?si=osc0NGZEiTaW-Pkg"
# Function to download the video
def download_youtube_video(url: str, save_directory: Path, video_filename: str) -> None:
    try:
        yt = YouTube(url, on_progress_callback=on_progress)
        print(f"Downloading: {yt.title}")
        ys = yt.streams.get_highest_resolution()
        ys.download(output_path=save_directory, filename=video_filename)
        print(f"Download completed: {video_filename}")
    except Exception as e:
        print(f"Error during download: {e}")
# Generate filename and check if the video exists
try:
    yt = YouTube(youtube_url)
    video_filename: str = f"{yt.title}.mp4"
except Exception as e:
    print(f"Error retrieving video title: {e}")
    video_filename = "default_video_name.mp4"
video_path: Path = remote_video_save_directory / video_filename
if not video_path.exists():
    download_youtube_video(youtube_url, remote_video_save_directory, video_filename)
else:
    print(f"Video already exists: {video_path}")
# Use imageio to read the downloaded video
video_reader = imageio.get_reader(str(video_path), "ffmpeg")
# Load YOLO model
model = YOLO("yolo11n.pt")
#%%
import numpy as np
print(next(model.model.parameters()).dtype) # torch.float32

# 최대 472 GFLOPS (FP16 기준)
# FP32 기준으로는 236 GFLOPS입니다.
# EffcieintGCN-B0 2.73 FLOPS. FP32
# EffcieintGCN-B0 2.73 FLOPS

#%%
import pandas as pd
import itertools

# Constants for YOLO models (FLOPS and mAP values)
yolo_models = {
    'YOLO11n': {'FLOPS': 6.5, 'mAP': 39.5},
    'YOLO11s': {'FLOPS': 21.5, 'mAP': 47.0},
    'YOLO11m': {'FLOPS': 68.0, 'mAP': 51.5},
    'YOLO11l': {'FLOPS': 86.9, 'mAP': 53.4},
    'YOLO11x': {'FLOPS': 194.9, 'mAP': 54.7},
}

# Constants for EfficientGCN models
efficientgcn_models = {
    'EfficientGCN-B0': 2.73,
    'EfficientGCN-B2': 4.05,
    'EfficientGCN-B4': 8.36,
}

mediapipe_flops = 1.0  # FLOPS for MediaPipe models

# Jetson Nano performance constants
jetson_nano_fp16_performance = 472  # GFLOPS for FP16
jetson_nano_int8_performance = jetson_nano_fp16_performance * 2  # GFLOPS for INT8 (double FP16 performance)

# Function to calculate total GFLOPS based on the number of detected people and frames per second
def calculate_total_flops(yolo_flops, efficientgcn_flops, num_people, fps, optimization_factor=1.0):
    # Apply optimization factor (1.0 for no optimization, 0.5 for FP16, 0.25 for INT8)
    yolo_optimized = yolo_flops * optimization_factor
    efficientgcn_optimized = efficientgcn_flops * optimization_factor
    return (yolo_optimized + (num_people * (mediapipe_flops + efficientgcn_optimized))) * fps

# Define missing variables
optimization_types = {
    'None': 1.0,
    'FP16': 0.5,
    'INT8': 0.25
}

people_counts = list(range(2, 11, 2))  # People counts: 2, 4, 6, 8, 10
fps_values = [10, 20, 30]  # Frame rates

# Create all combinations using itertools.product
combinations = itertools.product(
    yolo_models.items(),
    efficientgcn_models.items(),
    optimization_types.items(),
    people_counts,
    fps_values
)

# Dictionary to store results
results = {
    'YOLO Model': [],
    'EfficientGCN Model': [],
    'Optimization': [],
    'Number of People': [],
    'FPS': [],
    'Total GFLOPS': []
}

# Calculate GFLOPS for each combination
for (yolo_name, yolo_info), (gcn_name, gcn_flops), (optimization, factor), num_people, fps in combinations:
    total_flops = calculate_total_flops(yolo_info['FLOPS'], gcn_flops, num_people, fps, factor)
    results['YOLO Model'].append(yolo_name)
    results['EfficientGCN Model'].append(gcn_name)
    results['Optimization'].append(optimization)
    results['Number of People'].append(num_people)
    results['FPS'].append(fps)
    results['Total GFLOPS'].append(total_flops)

# Convert results to a DataFrame
df_results = pd.DataFrame(results)

# Filter the DataFrame based on Jetson Nano's FP16 and INT8 performance limits
df_filtered = df_results[
    (
        ((df_results['Optimization'] == 'FP16') & (df_results['Total GFLOPS'] <= jetson_nano_fp16_performance)) |
        ((df_results['Optimization'] == 'INT8') & (df_results['Total GFLOPS'] <= jetson_nano_int8_performance))
    ) &
    (df_results['FPS'] == 20) &
    (df_results['Total GFLOPS'] < 472*2 - 16*5*2)
].reset_index(drop=True)

# Sort the filtered DataFrame by 'Total GFLOPS' in ascending order
df_sorted = df_filtered.sort_values(by='Total GFLOPS').reset_index(drop=True)

# Display the sorted DataFrame without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df_sorted



#%%
# Get FPS from the video metadata
fps = video_reader.get_meta_data().get("fps", 30)
if fps == 0:
    fps = 30
delay = int(1000 / fps)
# Frame loop
for frame_idx, frame in enumerate(video_reader):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Perform detection
    results: Results = model(frame_bgr)
    # Draw bounding boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = int(box.cls[0])
            label = model.names[class_id]
            if label == "person":
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_bgr,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
    # Display frame
    cv2.imshow("YOLOv8 Detection", frame_bgr)
    # Key press handling
    key = cv2.waitKey(delay=delay) & 0xFF
    if key == ord("q"):  # Quit
        break
    elif key == ord("c"):  # Capture ROIs
        for result in results:
            boxes: Boxes = result.boxes
            for i, box in enumerate(boxes):
                box: Boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame_bgr[y1:y2, x1:x2]  # Crop the region of interest (ROI)
                output_filename = f"capture_{frame_idx}_{i}.jpg"
                output_path = captured_video_save_directory / output_filename
                cv2.imwrite(str(output_path), roi)  # Save the cropped image
                print(f"Saved cropped image: {output_path}")
        cv2.waitKey()
    elif key == ord(" "):  # Pause
        cv2.waitKey()
# Close OpenCV windows
cv2.destroyAllWindows()

# %%
# url = "https://www.princeton.edu/sites/default/files/styles/1x_full_2x_half_crop/public/images/2022/02/KOA_Nassau_2697x1517.jpg?"
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Garden_strawberry_%28Fragaria_%C3%97_ananassa%29_single2.jpg/1920px-Garden_strawberry_%28Fragaria_%C3%97_ananassa%29_single2.jpg?20220126170106"
req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
response: HTTPResponse = urlopen(req)
img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
b, g, r = cv2.split(img)  # or img[:, :, 0], img[:, :, 1], img[:, :, 2]
rgb_split = np.concatenate((b, g, r), axis=1)
height, width, channel = img.shape
zero = np.zeros((height, width, 1), dtype=np.uint8)
bgz = cv2.merge((b, g, zero))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
hsv_split = np.concatenate((h, s, v), axis=1)
cropped = img[50:450, 100:400]
resized = cv2.resize(cropped, (400, 200))
rotated = cv2.rotate(img, rotateCode=cv2.ROTATE_90_CLOCKWISE)
bitwised_not = cv2.bitwise_not(img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshold_value, binary_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
bulrred = cv2.blur(img, (9, 9), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
## 🆚 ???
sobeled = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, 3)
laplacian = cv2.Laplacian(gray_img, cv2.CV_8U, ksize=3)
canny = cv2.Canny(gray_img, 100, 200)
stop_display = False
def show_image(window_name, image):
    global stop_display
    if stop_display:
        return
    cv2.imshow(window_name, image)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == ord("q"):
        stop_display = True
# 이미지 표시 함수 시퀀스
if img is not None:
    show_image("image", img)
    show_image("laplacian", laplacian)
    show_image("canny", canny)
    show_image("sobel", sobeled)
    show_image("Blurred", bulrred)
    show_image("Binary", binary_img)
    show_image("Bitwised Not", bitwised_not)
    show_image("Rotated", rotated)
    show_image("Cropped", cropped)
    show_image("Resized", resized)
    show_image("BGR Channels", rgb_split)
    show_image("Split HSV", hsv_split)
# %%
#### [31m[1mrequirements:[0m Ultralytics requirement ['onnx>=1.12.0'] not found, attempting AutoUpdate...
# Load a model
# model = YOLO("yolov8n.pt")
model = YOLO(
    "/home/wbfw109v2/datasets/kct-yh2hv-cat_dog_dataset-2023-01-30-2015/runs/detect/train/weights/best.pt"
)
# # Train the model
# train_results = model.train(
#     data="coco8.yaml",  # path to dataset YAML
#     epochs=100,  # number of training epochs
#     imgsz=640,  # training image size
#     device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
# )
# Evaluate model performance on the validation set
# results = model.val(data="your_data.yaml", project="~/detections", name="val_results")
# %%
# Perform object detection on an image
# reference https://docs.ultralytics.com/modes/predict/
results = model(
    "/home/wbfw109v2/datasets/kct-yh2hv-cat_dog_dataset-2023-01-30-2015/test/images/dogs_00019_jpg.rf.1bfb0cc9b333ce2a1db473b17a4fbcee.jpg"
)
# Show detection results (optional)
results[0].show()
# %%
results[0]
# %%
# Extract bounding boxes, class labels, and confidence scores
boxes = results[0].boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
class_id = results[0].boxes.cls  # Class ids
confidences = results[0].boxes.conf  # Confidence scores
# Print results
for i in range(len(boxes)):
    print(
        f"Bounding Box: {boxes[i].tolist()}, Class ID: {class_id[i]}, Confidence: {confidences[i]}"


#%%

def normalize_minmax(data: np.ndarray) -> np.ndarray:
    """
    Normalizes the values in `data` between 0 and 1, excluding invalid depth values.
    """
    valid_mask = data > 0  # Only normalize valid depth values
    if valid_mask.any():
        valid_data = data[valid_mask]
        return (valid_data - valid_data.min()) / (valid_data.max() - valid_data.min())
    else:
        return data  # Return unmodified if no valid depth values


def popup_warning(message: str, duration: int = 5) -> None:
    root = tk.Tk()
    root.title("Warning")
    label = tk.Label(root, text=message, padx=20, pady=20)
    label.pack()
    root.after(duration * 1000, root.destroy)
    root.mainloop()


def show_warning() -> None:
    warning_thread = threading.Thread(
        target=popup_warning, args=("Object too close!", 3)
    )
    warning_thread.start()


def process_stream_with_warning(
    use_webcam: bool = False,
    video_file: Optional[str] = None,
    threshold: float = 0.3,  # Adjust the threshold
    scale_output: float = 0.5,
    advance_frames: int = 2,
    debounce_time: float = 3,
    smoothing_window: int = 10,
) -> None:
    if use_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Could not open webcam.")
    else:
        if video_file is None:
            raise ValueError("No video file provided.")
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"The video at {video_file} cannot be read.")

    input_fps = cap.get(cv2.CAP_PROP_FPS) if not use_webcam else 30
    input_video_frame_height, input_video_frame_width = (
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    target_fps = input_fps / advance_frames
    target_frame_height = int(input_video_frame_height * scale_output)
    target_frame_width = int(input_video_frame_width * scale_output)

    cv2.namedWindow("Monodepth Estimation", cv2.WINDOW_NORMAL)

    last_warning_time = time.time()
    distances = []

    try:
        input_video_frame_nr = 0
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                cap.release()
                break

            resized_image = cv2.resize(
                src=image, dsize=(network_image_height, network_image_width)
            )
            input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

            result = compiled_model([input_image])[output_key]

            # Normalize valid depth values
            normalized_result = normalize_minmax(result)

            # Detect if object is too close based on normalized depth map and threshold
            min_distance = (
                np.min(normalized_result[normalized_result > 0])
                if normalized_result.any()
                else np.inf
            )
            print(f"Minimum distance (normalized) in frame: {min_distance}")

            distances.append(min_distance)

            if len(distances) > smoothing_window:
                distances.pop(0)

            smoothed_distance = np.mean(distances)
            print(f"Smoothed distance (normalized): {smoothed_distance}")

            current_time = time.time()

            if smoothed_distance < threshold:
                print("Object detected too close!")
                if current_time - last_warning_time > debounce_time:
                    show_warning()
                    last_warning_time = current_time

            result_frame = to_rgb(convert_result_to_image(normalized_result))

            result_frame = cv2.resize(
                result_frame, (target_frame_width, target_frame_height)
            )
            image = cv2.resize(image, (target_frame_width, target_frame_height))

            stacked_frame = np.hstack((image, result_frame))

            cv2.imshow("Monodepth Estimation", stacked_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            input_video_frame_nr += advance_frames
            cap.set(1, input_video_frame_nr)

    except KeyboardInterrupt:
        print("Processing interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


# Run with webcam stream
process_stream_with_warning(use_webcam=True, threshold=0.3)
