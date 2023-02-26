# %% [markdown]
# # Estimator overview
# Link: [Estimator overview](https://www.tensorflow.org/guide/estimator "Estimators - Estimator overview")

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
# * Using the SavedModel format
"""
! Warning: Estimators are not recommended for new code. Estimators run v1.Session-style code which is more difficult to write correctly, and can behave unexpectedly, especially when combined with TF 2 code. Estimators do fall under our compatibility guarantees, but will receive no fixes other than security vulnerabilities. See the migration guide for details.
! = not learn this.
"""
