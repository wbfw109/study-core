# %% [markdown]
# # Tensorflow basics
# Link: [Automatic differentiation](https://www.tensorflow.org/guide/autodiff "Tensorflow basics - Automatic differentiation")
#
# https://wwwtensorflow.org/api_docs/python/tf/dtypes/DType
# Todo: About shapes ~ ★ Frist
#

# %%
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output
import logging
import os
import tensorflow as tf
import numpy as np
import pandas
import random

from tensorflow.python.keras.backend import dtype

# ** setting
# allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# set verbose level
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.autograph.set_verbosity(3)

# %%
# * Introduction to gradients and automatic differentiation

# %%
# * Setup
# ...

# %%
# * Getting a gradient of None
# ...
# + 4. Took gradients through a stateful object

# Similarly, tf.data.Dataset iterators and tf.queues are stateful, and will stop all gradients on tensors that pass through them.
