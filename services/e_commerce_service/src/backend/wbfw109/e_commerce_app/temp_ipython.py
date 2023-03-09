# %%
from __future__ import annotations

from array import array
from collections.abc import Generator, Sequence
from enum import Enum
from heapq import heappop, heappush
from itertools import zip_longest
from pathlib import Path
from typing import (
    Any,
    Callable,
    Final,
    Iterable,
    Iterator,
    Literal,
    LiteralString,
    NamedTuple,
    Never,
    Optional,
    ParamSpec,
    Tuple,
    TypedDict,
    TypeVar,
)

import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import display
from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# %doctest_mode

# from google.protobuf.internal import api_implementation
# print(api_implementation.Type())

#%%

