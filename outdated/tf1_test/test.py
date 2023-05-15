#%%
import IPython
from IPython.core.interactiveshell import InteractiveShell
import logging
import os
from pathlib import Path
import numpy
import sys
import subprocess
import ast
from enum import Enum
from PIL import Image


# * setting
# ** common environment setting for each python version
class EnvironmentLocation(Enum):
    LOCAL = 1
    LOCAL_WITH_DOCKER = 2
    GOOGLE_COLAB = 11


problem_name: str = "temp_1"
environment_location: EnvironmentLocation = EnvironmentLocation.LOCAL
# tesnorflow_version: str = 1 | 2
tensorflow_version_as_integer: int = 1

# + setting - fixed variables
installed_packages: list = [
    pip_list_as_json["name"]
    for pip_list_as_json in ast.literal_eval(
        subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format", "json"],
            capture_output=True,
            text=True,
        ).stdout
    )
]

# + setting - to install required packages and for default location
if environment_location == EnvironmentLocation.GOOGLE_COLAB:
    # if it is not local environment
    from google.colab import drive

    GOOGLE_COLAB_DRIVE_PATH: Path = Path("/content/drive")
    drive.mount(str(GOOGLE_COLAB_DRIVE_PATH))
    MY_GOOGLE_DRIVE_JUPYTER_PATH: Path = (
        GOOGLE_COLAB_DRIVE_PATH / "MyDrive/Colab_Notebooks"
    )
    SAVED_FOLDER_PATH: Path = MY_GOOGLE_DRIVE_JUPYTER_PATH

    # upgrade outdated pip
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
else:
    # if it local environment
    from starter.config import CONFIG_CLASS

    if environment_location == EnvironmentLocation.LOCAL:
        SAVED_FOLDER_PATH: Path = CONFIG_CLASS.MACHINE_LEARNING_ROOT_PATH
    elif environment_location == EnvironmentLocation.LOCAL_WITH_DOCKER:
        SAVED_FOLDER_PATH: Path = CONFIG_CLASS.GOOGLE_DRIVE_APP_PATH / "Colab_Notebooks"

PROBLEM_PATH = SAVED_FOLDER_PATH / problem_name
SAVED_FOLDER_PATH.mkdir(exist_ok=True)
PROBLEM_PATH.mkdir(exist_ok=True)

# + setting - tensorfow version
if tensorflow_version_as_integer == 2:
    import tensorflow as tf
elif tensorflow_version_as_integer == 1:
    # python <= 3.7
    if environment_location == EnvironmentLocation.GOOGLE_COLAB:
        IPython.get_ipython().run_line_magic("tensorflow_version", "1.x")
    import tensorflow as tf

    tf.compat.v1.enable_eager_execution()
    # import keras
tesnorflow_version: str = tf.__version__

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# + set verbose level
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.autograph.set_verbosity(3)

# %%
# for image_path in TEST_IMAGE_PATHS:
#     show_inference(detection_model, image_path)
"""
# https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python

"""
TFLITE_FILE_PATH = "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_5_map982/tflite/tflite_graph.pb"
# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
# There is only 1 signature defined in the model,
# so it will return it by default.
# If there are multiple signatures then we can pass the name.
my_signature = interpreter.get_signature_runner()
image_path = "/mnt/c/Users/wbfw109/MyDrive/shared_resource/images/test/cam01_drk_takeout_ice_02_30.jpg"
image_np = numpy.array(Image.open(image_path))
image = numpy.asarray(image_np)
# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

output = my_signature(input_tensor)

# my_signature is callable with input as arguments.
# output = my_signature(x=tf.constant([1.0], shape=(1,10), dtype=tf.float32))
# 'output' is dictionary with all outputs from the inference.
# In this case we have single output 'result'.
print(output["result"])

# https://github.com/tensorflow/tensorflow/issues/31224
# https://github.com/tensorflow/models/blob/master/research/lstm_object_detection/g3doc/exporting_models.md
