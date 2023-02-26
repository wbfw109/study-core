# %% [markdown]
# # Tensorflow basics
# Link: [Tensor](https://www.tensorflow.org/guide/tensor "Tensorflow basics - Tensor")
#
# https://wwwtensorflow.org/api_docs/python/tf/dtypes/DType
# Todo: About shapes ~ ★ Frist
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
# * Introduction to Tensors
"""
Tensors are multi-dimensional arrays with a uniform type (called a dtype). You can see all supported dtypes at tf.dtypes.DType.
If you're familiar with NumPy, tensors are (kind of) like np.arrays.
All tensors are immutable like Python numbers and strings: you can never update the contents of a tensor, only create a new one.


https://stackoverflow.com/questions/58429900/valueerror-all-inputs-to-concretefunctions-must-be-tensors
    Since tensorflow works with Tensors only, so it will not accept a python list as input and as the error also says, you need to convert the list to a Tensor and then feed it.
/*
"""

# %%
# * Basics

# Here is a "scalar" or "rank-0" tensor . A scalar contains a single value, and no "axes".
# This will be an int32 tensor by default; see "dtypes" below.
rank_0_tensor = tf.constant(4)
print("===== rank_0_tensor type: {}".format(type(rank_0_tensor)))
rank_0_tensor

# A "vector" or "rank-1" tensor is like a list of values. A vector has one axis:
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print("===== rank_1_tensor type: {}".format(type(rank_1_tensor)))
rank_1_tensor

# If you want to be specific, you can set the dtype (see below) at creation time
rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16)
print("===== rank_2_tensor type: {}".format(type(rank_2_tensor)))
rank_2_tensor
# ** refer to the shape as picture in the URL
# clear_output(wait=True)

rank_3_tensor = tf.constant(
    [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
        [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
)
print("===== rank_3_tensor type: {}".format(type(rank_3_tensor)))
rank_3_tensor

# You can convert a tensor to a NumPy array either using np.array or the tensor.numpy method:
# Tensors often contain floats and ints, but have many other types, including: complex numbers, strings

# Note: The base tf.Tensor class requires tensors to be "rectangular"---that is, along each axis, every element is the same size.
# However, there are specialized types of tensors that can handle different shapes: Ragged tensors (see RaggedTensor below), Sparse tensors (see SparseTensor below)

# You can do basic math on tensors, including addition, element-wise multiplication, and matrix multiplication.

a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[1, 1], [1, 1]])  # Could have also said `tf.ones([2,2])`
print("===== basic math on Tensor objects; a, b")
a
b

tensor_add = tf.add(a, b)
print("===== element-wise addition. same 'a + b'. type: {}".format(type(tensor_add)))
tensor_add

tensor_multiply = tf.multiply(a, b)
print(
    "=====  element-wise multiplication. same 'a * b'. type: {}".format(
        type(tensor_multiply)
    )
)
tensor_multiply

tensor_matmul = tf.matmul(a, b)
print(
    "===== matrix multiplication. same 'a @ b'. type: {}".format(type(tensor_multiply))
)
tensor_matmul

# Tensors are used in all kinds of operations (ops).
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# Find the largest value
tensor_reduce_max = tf.reduce_max(c)
print("===== reduce_max type: {}".format(type(tensor_multiply)))
tensor_reduce_max

# Find the index of the largest value
tensor_argmax = tf.argmax(c)
print("===== argma type: {}".format(type(tensor_argmax)))
tensor_argmax

# Compute the softmax
tensor_nn_softmax = tf.nn.softmax(c)
print("===== nn_softmax type: {}".format(type(tensor_nn_softmax)))
tensor_nn_softmax

# %%
# * About shapes

# %%
# Note: custom study
# ** test immutable and mutable
print("===== test immutable and mutable")
features_columns = ["a", "b", "c"]

w_n: dict[str, tf.Variable] = {
    column: tf.Variable(
        tf.random.uniform(shape=[1], dtype=tf.float64),
        trainable=True,
    )
    for column in features_columns
}

w_n
w_n_copy = [w_n[column] for column in features_columns]
w_n_copy

w_n["a"].assign_add([99.0])
w_n_copy
w_n["a"]

# %%
# ** test shape
print("===== test tf.variable shape basic")
aaa = tf.constant([[1, 2, 3]])
aaa.shape
bbb = tf.constant([[4], [5], [6]])
bbb.shape
tf.matmul(aaa, bbb)

# ** test tf.variable shape
print("===== test tf.variable shape")
random_nomarl_tf: tf.Variable = tf.Variable(
    tf.random.normal([4, 1], mean=2), name="weight_sum"
)
random_nomarl_tf
constant_shape_tf = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
constant_shape_tf

input_columns = [
    "AAA",
    "BBB",
    "CCC",
    "DDD",
    "EEE",
    "FFF",
]

random_test_data_df = pandas.DataFrame.from_dict(
    {
        x: [
            random.random() + random.randint(0, 5),
            random.random() + random.randint(0, 5),
        ]
        for x in input_columns
    }
)
random_test_data_df
random_test_data_df.shape

# ** test random
print("\n===== test random")
tf_random_uniform_tf = tf.random.uniform(shape=(), minval=1, maxval=5, dtype=tf.int32)
tf_random_uniform_tf
tf.random.uniform(shape=[1])

# %%
# ** test matmul
print("\n===== test matmul")

X_to_be_matmul_tf = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float32, shape=[1, 6])
X_to_be_matmul_tf

B_to_be_matmul_tf = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float32, shape=[6, 1])
B_to_be_matmul_tf

# it can be possible from tf.Tensor as well as from list[<values>]
print("\n----- tf.matmul from literal integer")
tf.matmul([[1, 2]], [[1], [2]])

# tf.Variable @ tf.Variable. sum of square values formular: n*(n+1)*(2*n+1)/6
print("\n----- tf.Tensor @ tf.Tensor")
mat_mul_multiple_linear = X_to_be_matmul_tf @ B_to_be_matmul_tf
mat_mul_multiple_linear

bbb: tf.Variable = tf.Variable(tf.random.uniform(shape=[1]), name="epsilon")
bbb
# flat these and operate in 0-rank
ccc = tf.reshape(mat_mul_multiple_linear, [])
ccc
ccc + bbb

# can cast primary type even if it's not flatten
float(ccc)

print("\n----- converted tf.Tensor @ tf.Tensor in one size")
series_to_tf = tf.convert_to_tensor([random_test_data_df.iloc[0]], dtype=tf.float32)
series_to_tf
series_to_tf @ B_to_be_matmul_tf

print("\n----- pandas.Dataframe @ tf.Variable in one size")
type(random_test_data_df.iloc[0])
df_at_tf = random_test_data_df.iloc[0] @ B_to_be_matmul_tf
df_at_tf
df_at_tf + bbb

print("\n----- pandas.Dataframe @ tf.Variable in batch")
random_test_data_df @ B_to_be_matmul_tf

# %%
a = tf.constant([[1, 2], [3, 4]])
# ** it is compatible operations with numpy, Pandas.DataFrame. return type is tf.Tensor.
a * np.arange(1, 5).reshape(2, 2)
as_df = pandas.DataFrame({"A": [1, 2], "B": [3, 4]})
as_df_2 = pandas.DataFrame({"A": [5], "B": [6]})

as_df_to_np = as_df.to_numpy()

as_df
a * as_df
as_df_to_np
a * as_df_to_np

# can be shuffle these
# https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
as_df = as_df.sample(frac=1).reset_index(drop=True)
as_df

# %%

tf.random.uniform(shape=[1])
