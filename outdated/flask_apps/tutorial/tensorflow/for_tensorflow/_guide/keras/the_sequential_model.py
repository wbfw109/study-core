# %% [markdown]
# # Keras
# Link: [The Sequential model](https://www.tensorflow.org/guide/keras/sequential_model "Keras - The Sequential model")
#

# Todo: Dataset structure ~ ★ Second (Keras) ~ Training workflows

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
# * Setup
# ..
