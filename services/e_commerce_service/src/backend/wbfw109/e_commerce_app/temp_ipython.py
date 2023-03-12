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


# [[x for x in map(int, input().split())] for _ in range(2)]  # type:ignore
# x = [1, 2, 3]
# automatically conversion to list
x = map(int, "123")
a, *b = x
a
b
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
#%%

import bisect

result = []
result2 = []
scores = [33, 99, 77, 70, 89, 90, 100]  # 애들이 받은 점수
grades = [60, 70, 80, 90]  # 기준점수를 등급으로 바꿔주자
for score in scores:
    # The return value i is such that all e in a[:i] have e <= x, and all e in
    # a[i:] have e > x. So if x already appears in the list, a.insert(i, x) will insert just after the rightmost x already there
    pos2 = bisect.bisect_right(grades, score)
    # The return value i is such that all e in a[:i] have e < x, and all e in a[i:] have e >= x
    pos1 = bisect.bisect_left(grades, score)
    # 경계에 있는 범위를 좌측 또는 우측의 인덱스에 포함시킬건지.
    # 범위 구분.. 0 A 1 B 2 C 3
    # 학생들의 점수를 grades = 기준에 넣는다고 가정했을때 어디 들어가면 될까
    # bisect는 인덱스만 리턴한다.
    result.append("FDCBA"[pos1])
    result2.append("FDCBA"[pos2])

print(result)
print(result2)
# bisect.insort_right, bisect.insort_left
# %%
