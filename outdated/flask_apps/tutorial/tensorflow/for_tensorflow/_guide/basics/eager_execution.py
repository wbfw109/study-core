# %% [markdown]
# # Tensorflow basics
# Link: [Eager execution](https://www.tensorflow.org/guide/eager "Tensorflow basics - Eager execution")
# ! Eager training 부터 진행 불가. Todo~

# %%
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output
import logging
import os
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
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
# * Eager execution
"""
Eager execution is a flexible machine learning platform for research and experimentation, providing:
    - An intuitive interface—Structure your code naturally and use Python data structures. Quickly iterate on small models and small data.
    - Easier debugging—Call ops directly to inspect running models and test changes. Use standard Python debugging tools for immediate error reporting.
    - Natural control flow—Use Python control flow instead of graph control flow, simplifying the specification of dynamic models.
"""

# %%
# * Setup and basic usage
# In Tensorflow 2.0, eager execution is enabled by default.
print("===== is eagerly?: {}".format(tf.executing_eagerly()))

# Eager execution works nicely with NumPy. NumPy operations accept tf.Tensor arguments. The TensorFlow tf.math operations convert Python objects and NumPy arrays to tf.Tensor objects. The tf.Tensor.numpy method returns the object's value as a NumPy ndarray.

x = [[10.0]]
m: EagerTensor = tf.matmul(x, x)
print("===== m type: {}".format(type(m)))
m
a: EagerTensor = tf.constant([[1, 2], [3, 4]])
print("===== a type: {}".format(type(a)))
a
# Obtain numpy value from a tensor:
aa: np.ndarray = a.numpy()
print("===== aa type: {}".format(type(aa)))
aa
# Broadcasting support
b: EagerTensor = tf.add(a, 1)
print("===== b type: {}".format(type(b)))
b
# Operator overloading is supported. it is not Matrix multiplication.
c: EagerTensor = a * b
print("===== c type: {}".format(type(c)))
c

# %%
# * Dynamic control flow

print("===== fizz buzz game")


def fizzbuzz(max_num):
    counter = tf.constant(0)
    max_num = tf.convert_to_tensor(max_num)
    for num in range(1, max_num.numpy() + 1):
        num = tf.constant(num)
        if int(num % 3) == 0 and int(num % 5) == 0:
            print("FizzBuzz")
        elif int(num % 3) == 0:
            print("Fizz")
        elif int(num % 5) == 0:
            print("Buzz")
        else:
            print(num.numpy())
        counter += 1


fizzbuzz(15)

# %%
# * Eager training
clear_output(wait=True)
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
    loss = w * w

grad = tape.gradient(loss, w)
print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)
