# %%
from __future__ import annotations

import enum
from array import array

from IPython import display
from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

import collections
import dataclasses
import datetime
import functools
import inspect
import itertools
import json
import logging
import math
import operator
import os
import pprint
import random
import re
import shutil
import sys
import time
import unittest
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import (
    Any,
    Callable,
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
from urllib.parse import urlparse

import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from wbfw109.libs.utilities.ipython import (
    VisualizationRoot,
    VisualizationRootT,
    append_line_into_df_in_wrap,
    display_data_frame_with_my_settings,
    display_header_with_my_settings,
    get_data_frame_for_test,
)

# %doctest_mode


# This Cell is for common declaration, and Next Cells are for specified algorithms.
# comply my structure for writting algorithms like Next Cell:

# Note
# 1. if you are using Type hint, but python version is less than 3.9, Generic type hint is required instead of type object.
#   it applies some web site: https://programmers.co.kr/
"""
# line copy... 차이 확인..
# TODO: random graph generator, create a graph image from a dictionary
# dynamic programming.... 여기에다 메모해야 하는데 docstring 써서 메모하기?

Fraction + .nemerator, denoimator
math.isclose, Decimal import statistics
built-in @ Matrix Multiplication 은 numpy 에서만 사용가능한가
sorted multiple key: https://hello-bryan.tistory.com/43

import quopri
import textwrap.shorten(), textwrap.wrap(), textwrap.fill(long_text, width=70)
import struct
deque는 list보다 속도가 빠르다. pop(0)와 같은 메서드를 수행할 때 리스트라면 O(N) 연산을 수행하지만, deque는 O(1) 연산을 수행하기 때문이다. 스레드 환경에서 안전하다.
import heapq, heapq.heapify(data), heapq.nsmallest(3, data))
import bisect 점수에 따른 학점 구하기
import curses 터미널 프로그램
import lzma, bzip2, gzip, zip, tar, zlib
pickle 객체를 파일로 저장, import copyreg 로 객체 변경에 따른 오류 방지?, 딕셔너리를 파일로 저장하려면? ― shelve??
https://docs.python.org/3/library/csv.html
import linecache
import tempfile with test function? import dircmp, fileinput (여러개의 파일 한 번에 읽기)
import configparser
암호화.. hashlib, hmac (메시지 변조 확인), secrets
operator.itemgetter
conversion:    import quopri, binascii

Threading.. import concurrent.futures, async await
import sched
문장분석 shlex
import xmlrpc? import htt.server import html.escape
socketserver + nonblocking>?
crawling? nntplib
email poplib, IMAP4


datetime
datetime.fromtimestamp(float(split_line[0])).strftime("%Y-%m-%d %H:%M:%S")

# MAth: Permutation, Cartesian product, Discrete Fourier transform, Multiplication algorithm

navigate: 다익스트라, 플루이드 워셜 알고리즘
에라토스테네스의 채, 위상 정렬, 강한 결합 요소, 네트웤 ㅡ플로우, , 이분 매칭, KMP 문자열 매칭, 라빈 카프 문자열 매칭, 

달력에 특정한 달에 해당하는 일수 구하기
ACM-ICPC, Codeforce, 정보 올림피아드

??? sorted 두 개 이상 키 적용방법. 람다 말고 함수로 여러 개 적용가능?
전위 중위 후위순회
다익스트라+위상정렬
"""
#%%
# 📝 Last checking before coding test

# instead "with open" context, Use "with Path.open" context.
# if multiple version problems, use (Enum | IntEnum), for statement, Structural Pattern Matching


def itertools_test():
    import itertools

    pprint.pprint(
        [(key, list(group)) for key, group in itertools.groupby("AAAABBBCCD")]
    )


def itertools_combinatoric_test():
    from itertools import (
        combinations,
        combinations_with_replacement,
        permutations,
        product,
    )

    pprint.pprint(list(permutations([1, 2, 3], 2)))
    pprint.pprint(list(product([1, 2, 3], "AB")))
    pprint.pprint(list(product([1, 2, 3], repeat=2)))
    pprint.pprint(list(combinations([1, 2, 3], 2)))
    pprint.pprint(list(combinations_with_replacement([1, 2, 3], r=2)))


def math_test():
    # Greatest Common Divisor, Least Common Multiple
    from math import gcd, lcm

    pprint.pprint(gcd(72, 30))
    pprint.pprint(lcm(10, 3, 100))

    # divmod is useful in greedy algorithm
    pprint.pprint([20 % 13, 20 / 13, 20 // 13, divmod(20, 13)])


def datetime_test():
    import calendar

    pprint.pprint([calendar.isleap(year) for year in range(2020, 2025)])


def conversion_test():
    import base64

    encoded = base64.b64encode(b"data to be encoded")
    decoded_data = base64.b64decode(encoded)


def operators_test():
    import math

    pprint.pprint([type(2**2), type(math.pow(2, 2))])


def string_test():
    import string

    pprint.pprint([string.ascii_letters, string.digits])


@dataclasses.dataclass
class DataClassTimeLog:
    time: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime(datetime.MINYEAR, 1, 1, 0, 0, 0, 0)
    )
    log: list[Any] = dataclasses.field(default_factory=list)


import graphlib

graph = {"D": {"B", "C"}, "C": {"A"}, "B": {"A"}}
ts = graphlib.TopologicalSorter(graph)
tuple(ts.static_order())
# input_test

ts = graphlib.TopologicalSorter()
ts.add(3, 2, 1)
ts.add(1, 0)
print([*ts.static_order()])


ts2 = graphlib.TopologicalSorter()
ts2.add(1, 0)
ts2.add(3, 2, 1)
print([*ts2.static_order()])

# 제출 전 ctrl+, 으로 remove unused import 사용하기.
# a = [*map(int, input().split(" "))]
#%%
# graph # import graphlib.TopologicalSorter 위상 정렬(topological sorting)은 유향 그래프의 꼭짓점(vertex)을 변의 방향을 거스르지 않도록 나열하는 것을 의미한다.
# Note that: divide and conquer, dynamic programming, Recurrence relation, greedy

#%%


class PositionalSystemsByBase:
    """
    built-in format()
    https://docs.python.org/3/library/string.html#format-specification-mini-language
    """

    @staticmethod
    def get_other_based_number_from_ten_based_number(
        ten_based_number: int, other_based: Literal[2, 8, 16]
    ) -> str:
        match other_based:
            case 2:
                return format(ten_based_number, "b")
            case 8:
                return format(ten_based_number, "o")
            case 16:
                return format(ten_based_number, "X")
            case _:
                return None

    @staticmethod
    def get_ten_based_number_from_other_based_number(
        other_based_number: str, other_based: Literal[2, 8, 16]
    ):
        match other_based:
            case 2:
                return int("0b" + other_based_number, 2)
            case 8:
                return int("0o" + other_based_number, 8)
            case 16:
                return int("0x" + other_based_number, 16)
            case _:
                return None


def input_test():
    a, b = map(str, input().split())
    c: list[int] = list(map(int, input().split()))
    d = [list(map(int, input().split())) for _ in range(3)]


#%%

"""
TesCase
    itertools.chain (*v), 키 값으로 개수를 세면 x, y 좌표의 값이 한 곳에 뭉쳐져서 중복되지 않은 값이 x 인지 y인지 알 수가 없다.
"""


def binary_search(sorted_list, searchnum):
    left = 0
    right = len(sorted_list) - 1

    while left <= right:
        ## middle은 left와 right의 중간지점
        middle = (left + right) // 2

        ## searchnum이 list[middle]보다 작다. right = middle-1
        if searchnum < sorted_list[middle]:
            right = middle - 1
        ## searchnum이 list[middle]보다 크다. left = middle+1
        elif sorted_list[middle] < searchnum:
            left = middle + 1
        ## searchnum이 list[middle]과 같다. 인덱스 반환
        else:
            return middle
    ## while문을 빠져나오는 경우 : right < left
    ## 즉, 리스트 안에 searchnum이 존재하지 않는다
    return -1


def guessRectangleValue() -> None:
    # initial value
    answer: int = []
    # v: List[list[int, int]] = [list(map(int, input("input the two point value * 3").split(", "))) for _ in range(3)]
    ## test values
    v = [[1, 4], [3, 4], [3, 10]]
    # v = [[1, 1], [2, 2], [1, 2]]

    # validate value
    for i in zip(*v):
        y: dict = collections.Counter(i)

        # vaildate and add value
        for key, value in y.items():
            if key >= 1 and key < 100000000:
                if y[key] == 1:
                    answer.extend([key])
            else:
                raise Exception(
                    "[Error] Out of range value in requirement. The value must be greater than or equal to 1 and less than 100 million."
                )

    # return
    print("세 점이 {}, {}, {} 위치에 있을 때, {} 에 점이 위치하면 직사각형이 됩니다.".format(*v, answer))


if __name__ == "__main__":
    guessRectangleValue()


#%%

# number theory
def get_fibonacci_list(list_count: int):
    if list_count < 0:
        return []
    fibonacci_list: list[int] = [0, 1]
    if list_count < 2:
        return fibonacci_list[: list_count + 1]

    for _ in range(2, list_count + 1):
        fibonacci_list.append(fibonacci_list[-1] + fibonacci_list[-2])
    return fibonacci_list


def get_prime_number_list_by_using_sieve_of_eratosthenes(max_range: int) -> list[int]:
    """
    TODO: sieve_of_atkin
    Time Complexity
        bigO(n log (log n))

    Args:
        max_range (int): .

    Returns:
        list[int]: prime_number_list

    Example
        t0 = time.time()
        prime_number_list = get_prime_number_list_by_using_sieve_of_eratosthenes(1000000)
        print("Size of prime numbers in range:", len(prime_number_list))
        t1 = time.time()
        print("Time taken:", t1 - t0)
    """
    if max_range < 1:
        return list()

    # for one-based output.
    is_prime_number_list: list[bool] = [True for _ in range(max_range + 1)]
    is_prime_number_list[0] = False
    is_prime_number_list[1] = False
    # sieve
    for i in range(2, int(math.sqrt(max_range)) + 1):
        if is_prime_number_list[i]:
            for j in range(i**2, max_range + 1, i):
                is_prime_number_list[j] = False
    return [i for i, x in enumerate(is_prime_number_list) if x]


get_prime_number_list_by_using_sieve_of_eratosthenes(10)
