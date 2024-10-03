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
model = YOLO("yolov8n.pt")
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
