# %% [markdown]
# # Keras
# Link: [The Sequential model](https://www.tensorflow.org/guide/keras/sequential_model "Keras - The Sequential model")
#

# Todo: Dataset structure ~ ★ Second (Keras) ~ Training workflows

# %%
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output
import logging
import os
import tensorflow as tf
import numpy as np
import pandas
import random

# ** setting
# allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# set verbose level
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.autograph.set_verbosity(3)

# %%
# * Setup
# ..
