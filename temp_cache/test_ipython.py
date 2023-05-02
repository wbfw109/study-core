# %%
from __future__ import annotations

import itertools
import random
import string
import time
from collections.abc import Generator, Sequence
from enum import Enum
from itertools import permutations, zip_longest
from pathlib import Path
from pprint import pprint
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


# %%
import sys

# 직사각형 행렬 중 일부 정사각형 행렬만 대각 대칭행렬 만들기.
arr = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
for x in arr:
    x
x1x2 = (1, 2)
y1y2 = (2, 3)
offset_i = x1x2[0]
offset_j = y1y2[0]
for i in range(x1x2[0], x1x2[1] + 1):
    for j in range(y1y2[0] + 1, y1y2[1] + 1):
        vi = i - offset_i
        vj = j - offset_j
        if vj > vi:
            x = vj + offset_i
            y = vi + offset_j
            arr[i][j], arr[x][y] = arr[x][y], arr[i][j]
            # print(x, y)
for x in arr:
    x
#  피보나치 수, 세그먼트 트리, 플로이드, 다익스트라, 자카드 유사도
# 그래프 매칭 문제: 3번은 트리가 주어지고, 루트에서부터 각 자식 노드로 바이러스가 전파될 때 적절한 절단점(cut vertex)을 찾고 최소 감염 노드의 수를 구하는 문제였는데, 문제에서 입력 조건이 노드의 최대 개수가 50개였기 때문에 적절한 방법으로 백 트래킹 한 후 모든 경우를 구하고, bfs로 시뮬레이션하면 최소 감염 노드의 수를 구할 수 있습니다!, N-back 문제
# %%
i = 10
k = 3
q, r = divmod(i, k)
q
r


# %%
import datetime as dt

from dateutil.parser import parse

# string parse time
a = dt.datetime.strptime("2017-01-02 14:44", "%Y-%m-%d %H:%M")
b = dt.datetime.strptime("2017-01-02 18:44", "%Y-%m-%d %H:%M")

(a - b).total_seconds()
(a - b).seconds
parse("2017-01-02")
parse("6/7/2016")  # month/days/year

# string format time
dt.datetime.strftime


# %%
import sys

input_ = sys.stdin.readline
# 문제 선정하기 ; https://devth-preview.goorm.io/exam/53763/%EC%BD%94%EB%94%A9-%ED%85%8C%EC%8A%A4%ED%8A%B8-%EC%9D%91%EC%8B%9C-%ED%99%98%EA%B2%BD-%EC%B2%B4%ED%97%98%ED%95%98%EA%B8%B0/quiz/1
input_()
if len(set(map(int, input_().split()))) > 3:
    print("YES")
else:
    print("NO")


# 근묵자흑 ; https://devth-preview.goorm.io/exam/53763/%EC%BD%94%EB%94%A9-%ED%85%8C%EC%8A%A4%ED%8A%B8-%EC%9D%91%EC%8B%9C-%ED%99%98%EA%B2%BD-%EC%B2%B4%ED%97%98%ED%95%98%EA%B8%B0/quiz/2?_ga=2.22476799.531264434.1598839678-1290480941.1598839678
import math

n, k = map(int, input_().split())
input_()
print(math.ceil((n - 1) / (k - 1)))
# %%
# https://khu98.tistory.com/66

import sys

input_ = sys.stdin.readline
result = []
for _ in range(int(input_())):
    n, m = map(int, input_().split())
    print(min(n // 5, (n + m) // 12))


# %%
#
# 코테 4번과 비슷한듯? https://www.acmicpc.net/problem/17471
# dictionary (0, 0): 1, (0, 1): 2, (0, 2): 3, (1, 0): 4 ...  마스킹으로

# 맵 객체 자체는 sorted가 되지만 map 객체의 컨테이너는 sorted 가 안된다.
sorted([*map(int, "1 22 3".split()), *map(int, "1 22 3".split())])

# LCA ; https://www.acmicpc.net/problem/3830
# 다익스트라??? ; https://www.acmicpc.net/problem/13303
# https://www.acmicpc.net/problem/15713
# LCA, 이진 검색 트리, 피보나치 트리, 이항 트리, 다익스트라, 세그먼트 트리 하고 다시 알고리즘 재개..

## C 로 구현되잇는지 확인 (모든 Standard library 가 C로 랩핑되어 있음. functools.reduce 포함.)
type(sum)
repr(sum)

# %%

# From python 3.7 ast.literal_eval() is now stricter. Addition and subtraction of arbitrary numbers are no longer allowed. link
# https://stackoverflow.com/a/34904657


# 앞에서부터 읽을 대와 뒤에서부터 읽을 떄 똑같은 단어 팰린드롬.
# def solution(n, m):
#     """
#     홀수, 짝수 구분해서도 더 빠르게 만들 수 있을듯.
#     짝수개 ->
#     10*10*9*10
#     """
#     answer = 0
#     for i in range(n, m + 1):
#         if str(i) == "".join(reversed(str(i))):
#             answer += 1

#     return answer

a = [0] * 5
a[-3:-2] = [1] * 1
a  # [0, 0, 1, 0, 0]
a[-3:] = [2] * 2
a  # [0, 0, 2, 2]
a[-1:] = [3] * 1
a  # [0, 0, 2, 3]
a[-1:0] = [4] * 1
a  # [0, 0, 1, 0, 1, 0]

# When you set the slice a[-1:0], it's actually an empty slice. This is because the start index -1 refers to the last element of the list, and the stop index 0 is the start of the list. Since the slice is set with the start index being after the stop index, it results in an empty slice.

# Now, when you assign [1] * 1 to this empty slice, instead of overwriting any element in the list, Python inserts the new element into the list. As a result, the length of the list increases by 1. The list becomes [0, 0, 1, 0, 1, 0].


# %%

list(itertools.product(range(-1, 2), repeat=2))
i = 0
j = 0
list((i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1))
### 추가하기 - https://docs.python.org/3/library/itertools.html#itertools.product same two list comprehension
# Roughly equivalent to nested for-loops in a generator expression. For example, product(A, B) returns the same as ((x,y) for x in A for y in B).


# %%

# 기록하기

a = "0123456789"
s, e = 0, 7
a[e:s:-1]  # 7654321
a[e : None if s == 0 else s - 1 : -1]  # 76543210
# a[:s] + a[e:s:-1] + a[e + 1 :]


# %%


list(dict.fromkeys([0, 1, 1, 2, 2, 3]))
from datetime import datetime, timedelta

# 팰린드롬 단어
s[-1 : (len(s) + 1) // 2 - 1 : -1]  # 4 -> 2개 (i=3 to 2), 5 -> 2 (i=5 to 3)
s[: len(s) // 2]  # 4 -> 2(i=1) 개, 5 -> 2개

# return sum(
#     (
#         s[-1 : (len(s) + 1) // 2 - 1 : -1] == s[: len(s) // 2]
#         for s in (str(i) for i in range(n, m + 1))
#     )
# )


# 1. How to find the length of a substring of a given string where the frequency of consecutive matches of that string is the maximum?
# 2. How to use dynamic programming to find the number of ways to select exactly k non-duplicate elements from 1 to n so that their sum is n (in Python)?
# 3. In Python, in a list with a given positive integer, how to find the minimum number of times that a given element is swapped so that the size between elements is less than or equal to k? (should return -1 if not possible).
#   E.g. arr = [10, 40, 30, 20], k=10  ->  change arr[2], arr[4]; one time. so answer is 1.

# 프로그래머스 효율성 테스트에서 break 후 return 하는 것보다 바로 return 할떄 효율성 테스트에서 만점 받음.

# %%
import re

pattern = re.compile(r"[a-z]")
# complete Programmers lv 0 problems, add Set cover problem''