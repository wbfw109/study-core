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

import math
from typing import Any, Callable

a = 100
target_list: list[int] = list(range(1000))


def solve(predicate: Callable[..., bool], target_list: list[Any]) -> None:
    n: int = len(target_list)
    left_i: int = 0
    right_i: int = n - 1
    while left_i != right_i:
        middle_i: int = math.ceil((left_i + right_i) / 2)
        if target_list[middle_i] > self.look_up_target:
            right_i = middle_i - 1
        else:
            left_i = middle_i
            # <look_up_target> is found in <middle_i>-th of <target_list>.
    else:  # if left_i == right_i
        if target_list[left_i] == self.look_up_target:
            self.target_location = left_i
            return


solve(lambda x: x + a == 666, target_list)
