# %% [markdown]
# # Tensorflow basics
# Link: [Training loops](https://www.tensorflow.org/guide/basic_training_loops "Tensorflow basics - Training loops")
# https://www.tensorflow.org/guide/autodiff#getting_a_gradient_of_none
#
# Todo: ~ ★ Frist
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
# * Basic training loops
# In the previous guides, you have learned about tensors, variables, gradient tape, and modules. In this guide, you will fit these all together to train models.
# TensorFlow also includes the tf.Keras API, a high-level neural network API that provides useful abstractions to reduce boilerplate. However, in this guide, you will use basic classes.

# %%
# * Eager execution
"""
Eager execution is a flexible machine learning platform for research and experimentation, providing:
    - An intuitive interface—Structure your code naturally and use Python data structures. Quickly iterate on small models and small data.
    - Easier debugging—Call ops directly to inspect running models and test changes. Use standard Python debugging tools for immediate error reporting.
    - Natural control flow—Use Python control flow instead of graph control flow, simplifying the specification of dynamic models.
"""

# %%
# * Setup
# ...


# ...
# simple linear model
class MyModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be randomly initialized
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.w * x + self.b


model = MyModel()

# List the variables tf.modules's built-in variable aggregation.
print("===== Variables:")
model.variables
# Verify the model works
model(3.0)
assert model(3.0).numpy() == 15.0
