# %%
from __future__ import annotations

import itertools
from collections.abc import Generator, Sequence
from enum import Enum
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
import sys
from bisect import bisect_left, bisect_right
from itertools import combinations

input = sys.stdin.readline


def getNum(arr, find):
    # target이 몇개 있는지 나온다
    return bisect_right(arr, find) - bisect_left(arr, find)


def setSum(arr, sumArr: list[int]):
    for i in range(1, len(arr) + 1):
        for a in combinations(arr, i):
            sumArr.append(sum(a))
    sumArr.sort()


n, s = map(int, input().split())
arr = list(map(int, input().split()))

left, right = arr[: n // 2], arr[n // 2 :]
leftSum, rightSum = [], []

setSum(left, leftSum)
setSum(right, rightSum)
ans = 0
for l in leftSum:
    find = s - l
    ans += getNum(rightSum, find)

ans += getNum(leftSum, s)
print(getNum(leftSum, s))
ans += getNum(rightSum, s)
print(getNum(leftSum, s))

print(ans)
