# %% [markdown]
# # Data input pipelines
# Link: [tf.data](https://www.tensorflow.org/guide/data "Data input pipelines - tf.data")
#
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch
# https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
# https://stackoverflow.com/questions/48552103/why-does-the-tensorflow-tf-data-dataset-shuffle-functions-reshuffle-each-iterat
#
# Todo: Dataset structure ~ ★ Frist: ~ Training workflows

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
# * tf.data: Build TensorFlow input pipelines
"""
The tf.data API enables you to build complex input pipelines from simple, reusable pieces. For example, the pipeline for an image model might aggregate data from files in a distributed file system, apply random perturbations to each image, and merge randomly selected images into a batch for training. The pipeline for a text model might involve extracting symbols from raw text data, converting them to embedding identifiers with a lookup table, and batching together sequences of different lengths. The tf.data API makes it possible to handle large amounts of data, read from different data formats, and perform complex transformations.

The tf.data API introduces a tf.data.Dataset abstraction that represents a sequence of elements, in which each element consists of one or more components. For example, in an image pipeline, an element might be a single training example, with a pair of tensor components representing the image and its label.

There are two distinct ways to create a dataset:
    - A data source constructs a Dataset from data stored in memory or in one or more files.
    - A data transformation constructs a dataset from one or more tf.data.Dataset objects.

"""

# %%
# * Basic mechanics
"""
To create an input pipeline, you must start with a data source. For example, to construct a Dataset from data in memory, you can use tf.data.Dataset.from_tensors() or tf.data.Dataset.from_tensor_slices(). Alternatively, if your input data is stored in a file in the recommended TFRecord format, you can use tf.data.TFRecordDataset().

Once you have a Dataset object, you can transform it into a new Dataset by chaining method calls on the tf.data.Dataset object. For example, you can apply per-element transformations such as Dataset.map(), and multi-element transformations such as Dataset.batch(). See the documentation for tf.data.Dataset for a complete list of transformations.

The Dataset object is a Python iterable. This makes it possible to consume its elements using a for loop:
"""
dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
dataset
for elem in dataset:
    print(elem.numpy())

# Or by explicitly creating a Python iterator using iter and consuming its elements using next:
print("\n===== next iter(dataset)")
it = iter(dataset)
print(next(it).numpy())

# Alternatively, dataset elements can be consumed using the reduce transformation, which reduces all elements to produce a single result. The following example illustrates how to use the reduce transformation to compute the sum of a dataset of integers.

print("\n===== reduce dataset")

print(dataset.reduce(0, lambda state, value: state + value).numpy())

# %%
# + Dataset structure
# ...

# %%
# * Reading input data
# + Consuming NumPy arrays
# ...

# %%
# + Consuming CSV data

# If your data fits in memory the same Dataset.from_tensor_slices method works on dictionaries, allowing this data to be easily imported:

# %%
# * Batching dataset elements
# + Simple batching
# ...

# %%
# * Training workflows
# + Processing multiple epoch

# .. The Dataset.repeat transformation concatenates its arguments without signaling the end of one epoch and the beginning of the next epoch. Because of this a Dataset.batch applied after Dataset.repeat will yield batches that straddle epoch boundaries:

# .. If you need clear epoch separation, put Dataset.batch before the repeat:

# Note: While large buffer_sizes shuffle more thoroughly, they can take a lot of memory, and significant time to fill. Consider using Dataset.interleave across files if this becomes a problem.

# %%
# + Randomly shuffling input data

# tf.dataset.batch()
#   Dataset.shuffle doesn't signal the end of an epoch until the shuffle buffer is empty. So a shuffle placed before a repeat will show every element of one epoch before moving to the next:
# .. But a repeat before a shuffle mixes the epoch boundaries together

# %%
# Note: custom study
# If your program depends on the batches having the same outer dimension, you should set the `drop_remainder` argument to `True` to prevent the smaller batch from being produced.

# ~ 최종 df.
winequality_red_file = tf.keras.utils.get_file(
    "winequality-red.csv",
    "https://raw.githubusercontent.com/wbfw109-park/light_resource/main/winequality-red.csv",
)

df = pandas.read_csv(winequality_red_file, sep=";")
df.head()

# dataframe 에서 정제 후 tensor_slices 로 만들기???
red_wine_slices_dataset = tf.data.Dataset.from_tensor_slices(dict(df))
batched_dataset = red_wine_slices_dataset.batch(7, drop_remainder=True)
print()

# test data

# print()
# count = 0
# for batch in red_wine_slices_dataset.repeat(3).batch(128):
#     batch
#     count +=1
#     if count > 0:
#         break

# %%
# print()
# test_dataset = tf.data.Dataset.range(100)

# for shuffled_batch in test_dataset.shuffle(buffer_size=15, reshuffle_each_iteration=True).batch(15, drop_remainder=True).repeat(2):
#     len(shuffled_batch)

columns = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]
FEATURE_COLUMNS = columns[:-1]
Y_COLUMN = columns[-1:]

bbb: tf.Variable = tf.Variable(tf.random.uniform(shape=[1]), name="epsilon")
random_test_data_df = pandas.DataFrame.from_dict(
    {x: [random.random() + random.randint(0, 5)] for x in columns}
)
random_test_data_df.shape
w_n: dict[str, tf.Variable] = {
    column: tf.Variable(
        tf.random.uniform(shape=[1]), trainable=True, name="/".join([column, "weight"])
    )
    for column in columns
}
len(w_n)

# tf.matmul(w_n, random_test_data_df.iloc[0])
# https://ipython.readthedocs.io/en/stable/interactive/python-ipython-diff.html
# 여기에서도 ipython 문법을 사용할 수 있다.
"""
https://hwiyong.tistory.com/341
    tf.vectorized_map을 사용하면 빠른 속도로 연산을 수행할 수 있습니다. 하지만 코드가 복잡합니다.
    제공하는 jacobian을 사용하면, 기존 코드보다 10배는 빠르다고 합니다.
"""

print()

MY_BATCH_SIZE = 32
len(red_wine_slices_dataset)
for shuffled_batch in (
    red_wine_slices_dataset.shuffle(
        buffer_size=MY_BATCH_SIZE, reshuffle_each_iteration=True
    )
    .batch(MY_BATCH_SIZE, drop_remainder=False)
    .repeat(2)
):
    # type(shuffled_batch)
    print(f"colmun size: {len(shuffled_batch)}")
    x_n_tf = tf.reshape(
        (
            tf.convert_to_tensor(
                [sum(shuffled_batch[column]) for column in FEATURE_COLUMNS],
                dtype=tf.float64,
            )
            / MY_BATCH_SIZE
        ),
        (1, len(FEATURE_COLUMNS)),
    )
    w_n_tf = tf.convert_to_tensor(
        [w_n[column] for column in FEATURE_COLUMNS], dtype=tf.float64
    )

    y_pred = tf.reshape(tf.matmul(x_n_tf, w_n_tf), []) + tf.Variable(
        [1.0], dtype=tf.float64
    )
    loss_rmse: tf.keras.metrics.RootMeanSquaredError = (
        tf.keras.metrics.RootMeanSquaredError(
            name="root_mean_squared_error", dtype=tf.float64
        )
    )
    y_true = (
        tf.convert_to_tensor([sum(shuffled_batch["quality"])], dtype=tf.float64)
        / MY_BATCH_SIZE
    )
    loss_rmse.update_state(y_true=y_true, y_pred=y_pred.numpy())
    loss_rmse.result()

    y_pred
    y_true
    break
