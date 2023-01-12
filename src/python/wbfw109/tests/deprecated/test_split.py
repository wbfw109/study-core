# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn
from sklearn.model_selection import train_test_split

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# "===== using numpy ====="
# x = np.array([1, 2, 3, 4, 2, 3, 4, 5], dtype = np.float32)
# xs = np.split(x, indices_or_sections = [1, 3, 4, 5], axis = 0)
# x
# xs

# "===== using tensorflow ====="
# y = tf.Variable(np.array([1, 2, 3, 4, 2, 3, 4, 5], dtype = np.float32))
# ys = tf.split(y,num_or_size_splits = [1, 2, 5], axis = 0)
# y
# ys
"===== using sklearn (default: shuffle=True) ====="
x, y = np.arange(10).reshape((5, 2)), range(5)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)
x
list(y)
x_train
y_train
x_test
y_test
train_test_split(x, y, shuffle=False)
train_test_split(x, y, shuffle=True)

# %%
