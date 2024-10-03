#%%

import cv2
import numpy as np
import openvino as ov
import threading
import tkinter as tk
import matplotlib.cm
import time
import cv2
import numpy as np
import openvino as ov
import threading
import tkinter as tk
import matplotlib.cm
import time
from openvino.runtime import ConstInput, ConstOutput
# %% [markdown]
# # Monodepth Estimation with OpenVINO
# 
# This tutorial demonstrates Monocular Depth Estimation with MidasNet in OpenVINO. Model information can be found [here](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/midasnet/README.md).
# 
# ![monodepth](https://user-images.githubusercontent.com/36741649/127173017-a0bbcf75-db24-4d2c-81b9-616e04ab7cd9.gif)
# 
# ### What is Monodepth?
# Monocular Depth Estimation is the task of estimating scene depth using a single image. It has many potential applications in robotics, 3D reconstruction, medical imaging and autonomous systems. This tutorial uses a neural network model called [MiDaS](https://github.com/intel-isl/MiDaS), which was developed by the [Embodied AI Foundation](https://www.embodiedaifoundation.org/). See the research paper below to learn more.
# 
# R. Ranftl, K. Lasinger, D. Hafner, K. Schindler and V. Koltun, ["Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer,"](https://ieeexplore.ieee.org/document/9178977) in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: `10.1109/TPAMI.2020.3019967`.
# 
# 
# #### Table of contents:
# 
# - [Preparation](#Preparation)
#     - [Install requirements](#Install-requirements)
#     - [Imports](#Imports)
#     - [Download the model](#Download-the-model)
# - [Functions](#Functions)
# - [Select inference device](#Select-inference-device)
# - [Load the Model](#Load-the-Model)
# - [Monodepth on Image](#Monodepth-on-Image)
#     - [Load, resize and reshape input image](#Load,-resize-and-reshape-input-image)
#     - [Do inference on the image](#Do-inference-on-the-image)
#     - [Display monodepth image](#Display-monodepth-image)
# - [Monodepth on Video](#Monodepth-on-Video)
#     - [Video Settings](#Video-Settings)
#     - [Load the Video](#Load-the-Video)
#     - [Do Inference on a Video and Create Monodepth Video](#Do-Inference-on-a-Video-and-Create-Monodepth-Video)
#     - [Display Monodepth Video](#Display-Monodepth-Video)
# 
# 
# ### Installation Instructions
# 
# This is a self-contained example that relies solely on its own code.
# 
# We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
# For details, please refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).
# 
# <img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/vision-monodepth/vision-monodepth.ipynb" />
# 

# %% [markdown]
# ## Preparation
# [back to top ⬆️](#Table-of-contents:)
# 
# ### Install requirements
# [back to top ⬆️](#Table-of-contents:)
# 

# %%
import platform

%pip install -q "openvino>=2023.1.0"
%pip install -q opencv-python requests tqdm

if platform.system() != "Windows":
    %pip install -q "matplotlib>=3.4"
else:
    %pip install -q "matplotlib>=3.4,<3.7"

# Fetch `notebook_utils` module
import requests

r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py", "w").write(r.text)

# %% [markdown]
# ### Imports
# [back to top ⬆️](#Table-of-contents:)
# 

# %%
import time
from pathlib import Path

import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import (
    HTML,
    FileLink,
    Pretty,
    ProgressBar,
    Video,
    clear_output,
    display,
)
import openvino as ov

from notebook_utils import download_file, load_image, device_widget

# %% [markdown]
# ### Download the model
# [back to top ⬆️](#Table-of-contents:)
# 
# The model is in the [OpenVINO Intermediate Representation (IR)](https://docs.openvino.ai/2024/documentation/openvino-ir-format.html) format.

# %%
model_folder = Path("model")

ir_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/"
ir_model_name_xml = "MiDaS_small.xml"
ir_model_name_bin = "MiDaS_small.bin"

download_file(
    ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=model_folder
)
download_file(
    ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=model_folder
)

model_xml_path = model_folder / ir_model_name_xml

# %% [markdown]
# ## Functions
# [back to top ⬆️](#Table-of-contents:)
# 

# %%
def normalize_minmax(data):
    """Normalizes the values in `data` between 0 and 1"""
    return (data - data.min()) / (data.max() - data.min())


def convert_result_to_image(result, colormap="viridis"):
    """
    Convert network result of floating point numbers to an RGB image with
    integer values from 0-255 by applying a colormap.

    `result` is expected to be a single network result in 1,H,W shape
    `colormap` is a matplotlib colormap.
    See https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    cmap = matplotlib.cm.get_cmap(colormap)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    result = result.astype(np.uint8)
    return result


def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

# %% [markdown]
# ## Select inference device
# [back to top ⬆️](#Table-of-contents:)
# 
# select device from dropdown list for running inference using OpenVINO

# %%
device = device_widget()

device

# %% [markdown]
# ## Load the Model
# [back to top ⬆️](#Table-of-contents:)
# 
# Load the model in OpenVINO Runtime with `core.read_model` and compile it for the specified device with `core.compile_model`. Get input and output keys and the expected input shape for the model.

# %%
import openvino.properties as props


# Create cache folder
cache_folder = Path("cache")
cache_folder.mkdir(exist_ok=True)

core = ov.Core()
core.set_property({props.cache_dir(): cache_folder})
model = core.read_model(model_xml_path)
compiled_model = core.compile_model(model=model, device_name=device.value)

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)

network_input_shape = list(input_key.shape)
network_image_height, network_image_width = network_input_shape[2:]

# %% [markdown]
# ## Monodepth on Image
# [back to top ⬆️](#Table-of-contents:)
# 
# ### Load, resize and reshape input image
# [back to top ⬆️](#Table-of-contents:)
# 
# The input image is read with OpenCV, resized to network input size, and reshaped to (N,C,H,W) (N=number of images,  C=number of channels, H=height, W=width). 

# %%
IMAGE_FILE = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg"
image = load_image(path=IMAGE_FILE)

# Resize to input shape for network.
resized_image = cv2.resize(src=image, dsize=(network_image_height, network_image_width))

# Reshape the image to network input shape NCHW.
input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

# %% [markdown]
# ### Do inference on the image
# [back to top ⬆️](#Table-of-contents:)
# 
# Do inference, convert the result to an image, and resize it to the original image shape.

# %%
result = compiled_model([input_image])[output_key]

# Convert the network result of disparity map to an image that shows
# distance as colors.
result_image = convert_result_to_image(result=result)

# Resize back to original image shape. The `cv2.resize` function expects shape
# in (width, height), [::-1] reverses the (height, width) shape to match this.
result_image = cv2.resize(result_image, image.shape[:2][::-1])

# %% [markdown]
# ### Display monodepth image
# [back to top ⬆️](#Table-of-contents:)
# 

# %%


# %%
fig, ax = plt.subplots(1, 2, figsize=(20, 15))
ax[0].imshow(to_rgb(image))
ax[1].imshow(result_image)

# %% [markdown]
# ## Monodepth on Video
# [back to top ⬆️](#Table-of-contents:)
# 
# By default, only the first 4 seconds are processed in order to quickly check that everything works. Change `NUM_SECONDS` in the cell below to modify this. Set `NUM_SECONDS` to 0 to process the whole video.

# %%

# Warning popup function
def popup_warning(message: str, duration: int = 5) -> None:
    """
    Show a popup window for a warning with a specified message and duration.
    
    This function creates a new Tkinter window using `tk.Tk()` to display the warning.
    It's important to note that only one instance of `tk.Tk()` should exist in the same program.
    Creating multiple instances of `tk.Tk()` (even in different scopes) can cause issues, such as conflicts or crashes
    , because Tkinter is designed to handle only one main event loop.
    For additional windows, use `tk.Toplevel()` instead of `tk.Tk()`.

    :param message: The message to display in the popup window.
    :param duration: The time (in seconds) for the popup to remain visible before closing.
    """

    root = tk.Tk()  # Create the main window (root) for the Tkinter application.
    root.title("Warning")  # Set the title of the window to "Warning".

    # Create a label widget inside the root window. 
    # padx and pady add padding (space) around the text (20 pixels horizontally and vertically).
    label = tk.Label(root, text=message, padx=20, pady=20)

    # Place the label in the window using the pack geometry manager, which arranges the widget automatically.
    label.pack()

    # Schedule the destruction (closing) of the root window after 'duration' seconds.
    # The 'after' method takes the time in milliseconds (duration * 1000 converts seconds to milliseconds).
    # After the specified time, the 'destroy' method will be called to close the window.
    root.after(duration * 1000, root.destroy)

    # Start the Tkinter event loop, which keeps the window open and responsive to events (e.g., button clicks).
    # This loop runs until the window is closed.
    root.mainloop()


def show_warning() -> None:
    """
    Launch the popup warning in a separate thread.
    """
    warning_thread = threading.Thread(
        target=popup_warning, args=("Object too close!", 3)
    )
    warning_thread.start()

# Depth normalization function
def normalize_minmax(data: np.ndarray) -> np.ndarray:
    """
    Normalize an array so that its values are between 0 and 1.
    
    :param data: The input array to normalize.
    :return: A normalized array with values ranging from 0 to 1.
    """
    return (data - data.min()) / (data.max() - data.min())




# Convert result to an image using a colormap
def convert_result_to_image(
    result: np.ndarray, colormap: str = "viridis"
) -> np.ndarray:
    """
    Convert the result from the depth model to an RGB image using a colormap.

    :param result: The result from the depth model (2D array representing depth).
    :param colormap: The name of the colormap to use (default is 'viridis').
    :return: An RGB image of the depth result in uint8 format.
    """

    # Get the colormap from matplotlib, 'viridis' is used as the default colormap
    cmap = matplotlib.cm.get_cmap(colormap)

    # 🔢 Shape before squeezing: (1, 1, H, W)
    # Remove the batch dimension (N) to convert (1, 1, H, W) -> (1, H, W).
    result = result.squeeze(0)

    # 🔢 Shape after squeeze: (1, H, W)
    # Normalize the result values to be between 0 and 1 (no dimension change).
    result = normalize_minmax(result)

    # 🔢 Shape before applying colormap: (1, H, W)
    # Apply the colormap to the normalized result, which returns (H, W, 4) because it includes RGBA.
    # Then, select only the first 3 channels (RGB) using `[:, :, :3]`.
    # ❔ cmap(); Element-wise calcuation
    result = cmap(result)[:, :, :3] * 255

    # 🔢 Shape after applying colormap: (H, W, 3) beccause
    # Convert the resulting values to uint8 type to represent as an image (0-255 range for each channel).
    result = result.astype(np.uint8)

    return result




def process_stream_with_warning() -> None:
    """
    Capture video stream from webcam, process it with a depth estimation model,
    and display a warning popup if the user is too close.

    Exception handling is added to ensure the camera is released even if an error occurs.
    """

    # Initialize the OpenVINO core and load the pre-trained model
    core = ov.Core()
    model = core.read_model(model_xml_path)
    compiled_model = core.compile_model(model=model, device_name="CPU")

    # Get the model's input and output tensor keys
    input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)

    # Start capturing video stream from the default webcam (0)
    cap = cv2.VideoCapture(0)

    # Extract the input height and width expected by the model (N, C, H, W)
    network_image_height, network_image_width = list(input_key.shape)[2:]

    # Debouncing variables
    cooldown_period = 3  # seconds between warnings
    last_warning_time = 0  # tracks when the last warning was shown

    try:
        while True:
            # Capture a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                break

            # 🔢 Shape of frame: (H, W, 3)
            # Resize the frame to match the model's input size.
            resized_frame = cv2.resize(
                frame, (network_image_width, network_image_height)
            )
            # 🔢 Shape after resizing: (H, W, 3)

            # Preprocess the image to be in the shape (N, C, H, W)
            input_image = np.expand_dims(np.transpose(resized_frame, (2, 0, 1)), 0)
            # 🔢 Shape after transpose and expand: (1, 3, H, W)

            # Perform inference on the model
            result = compiled_model([input_image])[output_key]
            # 🔢 Output shape of result: (1, 1, H, W)

            # Convert the raw result (depth map) to an RGB image for visualization
            depth_image = convert_result_to_image(result)
            # 🔢 Shape after colormap conversion: (H, W, 3)

            # Resize the depth image to match the original frame size for side-by-side display
            depth_image_resized = cv2.resize(
                depth_image, (frame.shape[1], frame.shape[0])
            )
            # 🔢 Shape after resizing: (H, W, 3)

            # Stack the original frame and depth result side by side for display
            stacked_frame = np.hstack((frame, depth_image_resized))
            # 🔢 Shape after stacking: (H, 2 * W, 3)

            # Check if the user is too close based on the valid minimum depth value (ignoring 0.0)
            valid_depth_values = result[result > 0]  # Ignore zero depth values
            if valid_depth_values.size > 0:
                min_depth_value = np.min(valid_depth_values)
            else:
                min_depth_value = float(
                    "inf"
                )  # If no valid values, set to infinity to avoid false warnings

            current_time = time.time()

            print(min_depth_value)

            # Only show the warning if enough time has passed since the last warning
            if (
                min_depth_value < 0.3
                and current_time - last_warning_time > cooldown_period
            ):
                show_warning()
                last_warning_time = current_time

            # Display the stacked frame
            cv2.imshow("Webcam and Depth", stacked_frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")


    finally:
        # Ensure that the camera is released and OpenCV windows are closed
        cap.release()
        cv2.destroyAllWindows()


process_stream_with_warning()


# %% [markdown]
# ### Video Settings
# [back to top ⬆️](#Table-of-contents:)
# 

# %%
# Video source: https://www.youtube.com/watch?v=fu1xcQdJRws (Public Domain)
VIDEO_FILE = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/Coco%20Walking%20in%20Berkeley.mp4"
# Number of seconds of input video to process. Set `NUM_SECONDS` to 0 to process
# the full video.
NUM_SECONDS = 4
# Set `ADVANCE_FRAMES` to 1 to process every frame from the input video
# Set `ADVANCE_FRAMES` to 2 to process every second frame. This reduces
# the time it takes to process the video.
ADVANCE_FRAMES = 2
# Set `SCALE_OUTPUT` to reduce the size of the result video
# If `SCALE_OUTPUT` is 0.5, the width and height of the result video
# will be half the width and height of the input video.
SCALE_OUTPUT = 0.5
# The format to use for video encoding. The 'vp09` is slow,
# but it works on most systems.
# Try the `THEO` encoding if you have FFMPEG installed.
# FOURCC = cv2.VideoWriter_fourcc(*"THEO")
FOURCC = cv2.VideoWriter_fourcc(*"vp09")

# Create Path objects for the input video and the result video.
output_directory = Path("output")
output_directory.mkdir(exist_ok=True)
result_video_path = output_directory / f"{Path(VIDEO_FILE).stem}_monodepth.mp4"

# %% [markdown]
# ### Load the Video
# [back to top ⬆️](#Table-of-contents:)
# 
# Load the video from a `VIDEO_FILE`, set in the *Video Settings* cell above. Open the video to read the frame width and height and fps, and compute values for these properties for the monodepth video.

# %%``
cap = cv2.VideoCapture(str(VIDEO_FILE))
ret, image = cap.read()
if not ret:
    raise ValueError(f"The video at {VIDEO_FILE} cannot be read.")
input_fps = cap.get(cv2.CAP_PROP_FPS)
input_video_frame_height, input_video_frame_width = image.shape[:2]

target_fps = input_fps / ADVANCE_FRAMES
target_frame_height = int(input_video_frame_height * SCALE_OUTPUT)
target_frame_width = int(input_video_frame_width * SCALE_OUTPUT)

cap.release()
print(
    f"The input video has a frame width of {input_video_frame_width}, "
    f"frame height of {input_video_frame_height} and runs at {input_fps:.2f} fps"
)
print(
    "The monodepth video will be scaled with a factor "
    f"{SCALE_OUTPUT}, have width {target_frame_width}, "
    f" height {target_frame_height}, and run at {target_fps:.2f} fps"
)

# %% [markdown]
# ### Do Inference on a Video and Create Monodepth Video
# [back to top ⬆️](#Table-of-contents:)
# 

# %%
# Initialize variables.
input_video_frame_nr = 0
start_time = time.perf_counter()
total_inference_duration = 0

# Open the input video
cap = cv2.VideoCapture(str(VIDEO_FILE))

# Create a result video.
out_video = cv2.VideoWriter(
    str(result_video_path),
    FOURCC,
    target_fps,
    (target_frame_width * 2, target_frame_height),
)

num_frames = int(NUM_SECONDS * input_fps)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if num_frames == 0 else num_frames
progress_bar = ProgressBar(total=total_frames)
progress_bar.display()

try:
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            cap.release()
            break

        if input_video_frame_nr >= total_frames:
            break

        # Only process every second frame.
        # Prepare a frame for inference.
        # Resize to the input shape for network.
        resized_image = cv2.resize(
            src=image, dsize=(network_image_height, network_image_width)
        )
        # Reshape the image to network input shape NCHW.
        input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

        # Do inference.
        inference_start_time = time.perf_counter()
        result = compiled_model([input_image])[output_key]
        inference_stop_time = time.perf_counter()
        inference_duration = inference_stop_time - inference_start_time
        total_inference_duration += inference_duration

        if input_video_frame_nr % (10 * ADVANCE_FRAMES) == 0:
            clear_output(wait=True)
            progress_bar.display()
            # input_video_frame_nr // ADVANCE_FRAMES gives the number of
            # Frames that have been processed by the network.
            display(
                Pretty(
                    f"Processed frame {input_video_frame_nr // ADVANCE_FRAMES}"
                    f"/{total_frames // ADVANCE_FRAMES}. "
                    f"Inference time per frame: {inference_duration:.2f} seconds "
                    f"({1/inference_duration:.2f} FPS)"
                )
            )

        # Transform the network result to a RGB image.
        result_frame = to_rgb(convert_result_to_image(result))
        # Resize the image and the result to a target frame shape.
        result_frame = cv2.resize(
            result_frame, (target_frame_width, target_frame_height)
        )
        image = cv2.resize(image, (target_frame_width, target_frame_height))
        # Put the image and the result side by side.
        stacked_frame = np.hstack((image, result_frame))
        # Save a frame to the video.
        out_video.write(stacked_frame)

        input_video_frame_nr = input_video_frame_nr + ADVANCE_FRAMES
        cap.set(1, input_video_frame_nr)

        progress_bar.progress = input_video_frame_nr
        progress_bar.update()

except KeyboardInterrupt:
    print("Processing interrupted.")
finally:
    clear_output()
    processed_frames = num_frames // ADVANCE_FRAMES
    out_video.release()
    cap.release()
    end_time = time.perf_counter()
    duration = end_time - start_time

    print(
        f"Processed {processed_frames} frames in {duration:.2f} seconds. "
        f"Total FPS (including video processing): {processed_frames/duration:.2f}."
        f"Inference FPS: {processed_frames/total_inference_duration:.2f} "
    )
    print(f"Monodepth Video saved to '{str(result_video_path)}'.")

# %% [markdown]
# ### Display Monodepth Video
# [back to top ⬆️](#Table-of-contents:)
# 

# %%
video = Video(result_video_path, width=800, embed=True)
if not result_video_path.exists():
    plt.imshow(stacked_frame)
    raise ValueError(
        "OpenCV was unable to write the video file. Showing one video frame."
    )
else:
    print(f"Showing monodepth video saved at\n{result_video_path.resolve()}")
    print(
        "If you cannot see the video in your browser, please click on the "
        "following link to download the video "
    )
    video_link = FileLink(result_video_path)
    video_link.html_link_str = "<a href='%s' download>%s</a>"
    display(HTML(video_link._repr_html_()))
    display(video)


