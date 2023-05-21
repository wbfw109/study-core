# %%
from __future__ import annotations

import itertools
import random
import string
import time
import timeit
import unittest
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

# datetime.weekday 기록.
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
# 코테 4번과 비슷한듯? https://www.acmicpc.net/problem/17471
# dictionary (0, 0): 1, (0, 1): 2, (0, 2): 3, (1, 0): 4 ...  마스킹으로

# 맵 객체 자체는 sorted가 되지만 map 객체의 컨테이너는 sorted 가 안된다.
sorted([*map(int, "1 22 3".split()), *map(int, "1 22 3".split())])

# LCA ; https://www.acmicpc.net/problem/3830
# 다익스트라??? ; https://www.acmicpc.net/problem/13303
# https://www.acmicpc.net/problem/15713
# LCA, 이진 검색 트리, 피보나치 트리, 이항 트리, 다익스트라, 세그먼트 트리 하고 다시 알고리즘 재개..

# LIS ; sequence[i] 보다 작은,  길이 l 에서 끝나는 인덱스의 값들 중  가지고 가장 큰 값의 길이 l 구하기


import re
from datetime import datetime, timedelta

# 프로그래머스 효율성 테스트에서 break 후 return 하는 것보다 바로 return 할떄 효율성 테스트에서 만점 받음.
pattern = re.compile(r"[a-z]")
a = " a  b   c"
a.split(" ")

# https://www.acmicpc.net/problem/1648

# Tessellation


# %%

## 구간합 문제; 배열에서 i 부터 j 까지 인덱스에 있는 값을 구할 때 접두사 합 배열 P 사용하자.

n = 5
data = [10, 20, 30, 40, 50]

prefix = [0]
sum_value = 0

for d in data:  # O(N)
    sum_value += d
    prefix.append(sum_value)

# 쿼리가 주어짐 -> 2~3번째까지 구간의 합 (인덱스가 아님)
left = 2
right = 3

result = prefix[right] - prefix[left - 1]
print(result)
10e3  # 10000. not 1000.


# %%
players = ["mumu", "soe", "poe", "kai", "mine"]
callings = ["kai", "kai", "mine", "mine"]


def method1():
    ranks = {name: i for i, name in enumerate(players)}
    for name in callings:
        rank = ranks[name]
        pre_rank = rank - 1


def method2():
    ranks = {name: i for i, name in enumerate(players)}
    for name in callings:
        pre_rank, rank = ranks[name] - 1, ranks[name]


timeit.timeit(method1, number=100000)  # 0.104612 s
timeit.timeit(method2, number=100000)  # 0.116712 s
# %%

range_obj = range(100000)


def method1():
    """The reason method1 is slower than method2 is due to the additional overhead introduced by the generator expression in method1.
    This overhead includes:
        Creating the generator object.
        Entering and exiting the generator context each time it yields a value.
        Handling the StopIteration exception when the generator is exhausted.
    """
    for i in (i for i in range_obj if i & 1 == 1):
        x = i


def method2():
    for i in range_obj:
        if i & 1 == 1:
            x = i


timeit.timeit(method1, number=100)  # 0.58332s
timeit.timeit(method2, number=100)  # 0.38724s


# %%
n = 5000


def method1():
    """When n is large enough, the accumulated cost of the extra addition operation in method1 could outweigh the one-time cost of the division operation in method2, resulting in method2 being faster overall."""
    for _ in range(0, n, 2):
        pass


def method2():
    for _ in range(n // 2):
        pass


# when n = 1000000001, 9.76544s and 10.79032s.
# when n = 5000, 4.48339e-05s and 4.80699e-05s.
timeit.timeit(method1, number=1)
timeit.timeit(method2, number=1)


# %%
## unpack receives iterable
s = "1" * 10
x = [*s]
x
from collections import deque

s = "1" * 1000


def method1():
    """Python lists are implemented as dynamic arrays in the backend.
    This means that they preallocate additional space for new elements.
    When you add elements to a list using the * operator, Python can take advantage of this preallocated space and insert all elements at once, which is quite efficient.

    The * operator, also known as the unpacking operator, is used in Python to unpack collections into separate elements.
    When you use this operator within a list, it essentially 'unpacks' the elements of the collection you're referencing and places them individually into the new list.
    """
    x: list[str] = ["intercept", "x", *s]


def method2():
    """
    deque.extend() method will have to individually append each character from s to the deque.
    Since deques are implemented as doubly-linked lists in Python, this individual addition requires changing pointers for each new element added, which incurs additional overhead, thus taking longer time.
    """
    x = deque(["intercept", "x"])
    x.extend(s)


timeit.timeit(method1)  # 5.04185 s
timeit.timeit(method2)  # 6.69755 s


# %%
## profile required
# https://www.geeksforgeeks.org/merge-two-sorted-arrays-python-using-heapq/
def solution(r1, r2):
    """
    # x > 0 경우의 점들 개수 *4
    # 100만개를 다 테스트해볼 수는 없음.
    x=1, y=r1 ~ r2-1 까지가 x=1일떄 개수.
    x=2, y=r2-1
    4분면 나누고, 4분면 안에서도 절반 나누면.
    a + b = r2 (a, b >= 0)


    count 3 + 5 + 7 ..
    r1    1   2
    a = 1
    a = 3, d = 2
    ; initial_term a+2*(r1-1)
    the number of terms = r2-r1

    등차수열 솔루션은 틀린듯.
        initial_term = 3+2*(r1-1)
        n = r2-r1
        return (n*(2*initial_term+(n-1)*2)//2 - n + 1)*4

    """
    initial_term = 3 + 2 * (r1 - 1)
    n = r2 - r1
    return (n * (2 * initial_term + (n - 1) * 2) // 2 - n + 1) * 4


def solution(sequence, k):
    """- `k는 항상 sequence의 부분 수열로 만들 수 있는 값입니다.`
    - `이때 수열의 인덱스는 0부터 시작합니다.`

    - `1 ≤ sequence의 원소 ≤ 1,000`
    - `sequence는 비내림차순으로 정렬되어 있습니다.`
    n^2 솔루션이 안된다. 각 sequence[i] 부터 뒤에꺼까지 부분합으로 k 보다 크기 바로 전까지 최소 pair 수 를 만들고 합이 안맞으면 패스.
    """
    n = len(sequence)
    pa = [0]  # prefix array
    for x in sequence:
        pa.append(x + pa[-1])

    for pair_num in range(1, n + 1):
        for i in range(n - pair_num + 1):
            x = pa[i + pair_num] - pa[i]
            if x == k:
                return [i, i + pair_num - 1]
            elif x > k:
                break
    return [-1, -1]


# [1, 2, 3].index(3, -1) #..


# %%
def solution(n):
    dp: list[int] = [1, 2]  # Corrected initialization
    for i in range(2, n):
        dp[i % 2] = (
            dp[0] + dp[1]
        ) % 1000000007  # add modulo operation for avoiding overflow
    return dp[(n - 1) % 2]


solution(4)

## any((_ for _ in range(0)))# valid
# %%

## It can be possible. unpacking operator. *
a: list[int] = [0, *(i for i in range(10))]

s = "0123456789"
s.rindex("4", 3, 9)
s.rfind("2", -1, 3)

s[:-1]

# %%
i = 0
for i in range(i, 5):
    pass
i
for i in range(i, 5):
    print("ok")
