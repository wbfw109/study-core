# %%
import itertools

import IPython
from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

"""
Note: Python list comprehension, unpacking and multiple operations
    https://stackoverflow.com/questions/13958998/python-list-comprehension-unpacking-and-multiple-operations
"""

#%%
# when you write in list comprehension loop
x = range(10)
outer_iterable = [(i, j**2) for i, j in zip(x, x)]
outer_iterable

# it is useful to unpack nested dict.
[element for inner_iterable in outer_iterable for element in inner_iterable]
# when you write not in for loop
list(itertools.chain.from_iterable(outer_iterable))
