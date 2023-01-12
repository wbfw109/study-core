# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import IPython
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from pathlib import Path
import sys
import subprocess
import ast

# ** setting
# allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %%
get_ipython().system("python --version")
get_ipython().run_line_magic("tensorflow_version", "1.x")
import tensorflow as tf

print(tf.__version__)
tf.enable_eager_execution()

get_ipython().system(
    'ls "/content/drive/MyDrive/Colab Notebooks/tensorflow/models/research/object_detection/protos/" | grep "\\\\.proto$"'
)

# %%
get_ipython().run_cell_magic(
    "bash",
    "",
    "cd models/research/\n\nprotoc object_detection/protos/*.proto --python_out=.\n\npip install .",
)
