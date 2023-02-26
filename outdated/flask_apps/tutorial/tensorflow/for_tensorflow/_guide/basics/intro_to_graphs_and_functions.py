# %% [markdown]
# # Tensorflow basics
# Link: [Intro to graphs and functions](https://www.tensorflow.org/guide/tensor "Tensorflow basics - Intro to graphs and functions")
#
# Todo: ~
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
# * Overview
"""
This is a big-picture overview that covers how tf.function allows you to switch from eager execution to graph execution.
For a more complete specification of tf.function, go to the tf.function guide.
"""

# %%
# *

# If you want to be specific, you can set the dtype (see below) at creation time
rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16)
print("===== rank_2_tensor type: {}".format(type(rank_2_tensor)))
rank_2_tensor
# ** refer to the shape as picture in the URL
# clear_output(wait=True)
