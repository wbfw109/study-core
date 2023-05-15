# %% [markdown]
# # Load and preprocess data
# Link: [TFRecord and tf.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord "Load and preprocess data - TFRecord and tf.Example")
#
#
# https://www.tensorflow.org/tutorials/quickstart/advanced
#
# Todo: ~ ★ Second

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
# * TFRecord and tf.train.Example
"""
The TFRecord format is a simple format for storing a sequence of binary records.

Protocol buffers are a cross-platform, cross-language library for efficient serialization of structured data.
    Protocol messages are defined by .proto files, these are often the easiest way to understand a message type.

Note: While useful, these structures are optional. There is no need to convert existing code to use TFRecords, unless you are using tf.data and reading data is still the bottleneck to training. You can refer to Better performance with the tf.data API for dataset performance tips.

Note: In general, you should shard your data across multiple files so that you can parallelize I/O (within a single host or across multiple hosts). The rule of thumb is to have at least 10 times as many files as there will be hosts reading data. At the same time, each file should be large enough (at least 10 MB+ and ideally 100 MB+) so that you can benefit from I/O prefetching. For example, say you have X GB of data and you plan to train on up to N hosts. Ideally, you should shard the data to ~10*N files, as long as ~X/(10*N) is 10 MB+ (and ideally 100 MB+). If it is less than that, you might need to create fewer shards to trade off parallelism benefits and I/O prefetching benefits.
"""

# %%
# * Setup
