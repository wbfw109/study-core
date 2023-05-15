# %% [markdown]
# # Optimize a model
# Link: [Overview](https://www.tensorflow.org/lite/performance/model_optimization "Optimize a model - Overview")
#
# [INTEGER QUANTIZATION FOR DEEP LEARNING INFERENCE:PRINCIPLES AND EMPIRICAL EVALUATION](https://arxiv.org/pdf/2004.09602.pdf "INTEGER QUANTIZATION FOR DEEP LEARNING INFERENCE:PRINCIPLES AND EMPIRICAL EVALUATION")
#
# https://www.tensorflow.org/lite/performance/quantization_spec
#
# Todo: ~
## # Note
# If you use Tensorflow object detection API models, converting tflite can not be achieved in this way.
# [Error] ValueError: Failed to parse the model: Only models with a single subgraph are supported, model had 5 subgraphs.
#    [Solution]
#       https://github.com/tensorflow/tensorflow/issues/35194
#       When I do not use optimizations it not occurs error, but I cannot use qunatized optimizations
#       instead, use export_tflite_ssd_graph)....py in https://github.com/tensorflow/models/tree/master/research/object_detection
#       and refer to https://github.com/tensorflow/models/blob/master/research/lstm_object_detection/g3doc/exporting_models.md

# %%
from IPython.core.interactiveshell import InteractiveShell
import logging
import os
from pathlib import Path
import tensorflow as tf

# ** setting
# allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# set verbose level
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.autograph.set_verbosity(3)

problem_name: str = "cifar_10"
development_environment: str = "google_drive_app"
if development_environment == "colab":
    # if it is not local environment
    from google.colab import drive

    GOOGLE_COLAB_DRIVE_PATH: Path = Path("/", "content", "drive")
    drive.mount(str(GOOGLE_COLAB_DRIVE_PATH))
    MY_GOOGLE_DRIVE_JUPYTER_PATH: Path = (
        GOOGLE_COLAB_DRIVE_PATH / "MyDrive" / "Colab_Notebooks"
    )
    SAVED_FOLDER_PATH: Path = MY_GOOGLE_DRIVE_JUPYTER_PATH
else:
    # if it local environment
    from mysite.config import CONFIG_CLASS

    if development_environment == "local":
        SAVED_FOLDER_PATH: Path = CONFIG_CLASS.MACHINE_LEARNING_ROOT_PATH
    elif development_environment == "google_drive_app":
        SAVED_FOLDER_PATH: Path = CONFIG_CLASS.GOOGLE_DRIVE_APP_PATH / "Colab_Notebooks"

SAVED_FOLDER_PATH = SAVED_FOLDER_PATH / problem_name

# %%
# * Model optimization

# %%
# * Why models should be optimized
# ...

# %%
# * Type of optimization

# + Quantization
# + Full integer quantization with int16 activations and int8 weights
# + Pruning
# + Clustering
