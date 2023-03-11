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
# variant of binary search.. as predicate instead of look_up_target

while 1:
    n, a, b = map(int, input().split())
    if n == a == b == 0: break
    arr = sorted([[map(int, input().split())]for _ in range(n)], key=lambda x: -abs(x[1]-x[2]))
    ans = 0
    for k, x, y in arr:
        if x <= y: val = min(k, a)
        else: val = k - min(k, b)
        ans += val x + (k - val) * y
        a -= val
        b -= k - val
    print(ans)