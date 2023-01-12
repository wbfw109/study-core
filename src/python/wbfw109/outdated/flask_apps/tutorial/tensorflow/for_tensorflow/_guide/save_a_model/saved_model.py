# %% [markdown]
# # Save a model
# Link: [SavedModel](https://www.tensorflow.org/guide/saved_model "Save a model - SavedModel")
#
# https://www.tensorflow.org/api_docs/python/tf/saved_model
# https://www.tensorflow.org/api_docs/python/tf/saved_model/save
#   > @tf.function
# https://www.tensorflow.org/tfx/serving/docker

# Todo: ~ ★
# %%
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output
import logging
import os
import tensorflow as tf
import numpy as np

# ** setting
# allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# set verbose level
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.autograph.set_verbosity(3)

# %%
# * Using the SavedModel format
"""
A SavedModel contains a complete TensorFlow program, including trained parameters (i.e, tf.Variables) and computation. It does not require the original model building code to run, which makes it useful for sharing or deploying with TFLite, TensorFlow.js, TensorFlow Serving, or TensorFlow Hub.

You can save and load a model in the SavedModel format using the following APIs:
    - Low-level tf.saved_model API. This document describes how to use this API in detail.
        - Save: tf.saved_model.save(model, path_to_dir)
        - Load: model = tf.saved_model.load(path_to_dir)
    - High-level tf.keras.Model API. Refer to the keras save and serialize guide.
    - If you just want to save/load weights during training, refer to the checkpoints guide.
"""

# %%
# * Setup
# ...

# ...
# tf.saved_model.save, tf.saved_model.load.. predict 는 각 정의된 모델마다 사용가능한 함수에서 찾을 수 있을듯.?
# ...

# %%
# * Running a SavedModel in TensorFlow Serving
# Note: SavedModels are usable from Python (more on that below), but production environments typically use a dedicated service for inference without running Python code.
# This is easy to set up from a SavedModel using TensorFlow Serving.
# See the TensorFlow Serving REST tutorial for an end-to-end tensorflow-serving example.

# ... only in korean
# serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn( tf.feature_column.make_parse_example_spec([input_column]))
# ...
