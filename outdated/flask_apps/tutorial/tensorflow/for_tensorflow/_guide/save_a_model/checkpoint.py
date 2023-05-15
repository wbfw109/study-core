# %% [markdown]
# # Save a model
# Link: [Checkpoint](https://www.tensorflow.org/guide/checkpoint "Save a model - Checkpoint")
# https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
# https://www.tensorflow.org/api_docs/python/tf/train/CheckpointManager
#
# Todo: ~ ★
# %%
from IPython.core.interactiveshell import InteractiveShell
import logging
import os
import tensorflow as tf

# ** setting
# allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# set verbose level
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.autograph.set_verbosity(3)

# %%
# * Training checkpoints
"""
The phrase "Saving a TensorFlow model" typically means one of two things:
    - Checkpoints, OR
    - SavedModel.
Checkpoints capture the exact value of all parameters (tf.Variable objects) used by a model. Checkpoints do not contain any description of the computation defined by the model and thus are typically only useful when source code that will use the saved parameter values is available.

The SavedModel format on the other hand includes a serialized description of the computation defined by the model in addition to the parameter values (checkpoint). Models in this format are independent of the source code that created the model. They are thus suitable for deployment via TensorFlow Serving, TensorFlow Lite, TensorFlow.js, or programs in other programming languages (the C, C++, Java, Go, Rust, C# etc. TensorFlow APIs).

This guide covers APIs for writing and reading checkpoints.
"""

# %%
# * Setup
print("===== is eagerly?: {}".format(tf.executing_eagerly()))
