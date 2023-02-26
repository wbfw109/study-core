# %%
from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import datetime
import enum
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
import selectors
import shutil
import socket
import sys
import threading
import time
import unittest
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from array import array
from collections.abc import Generator, Sequence
from enum import Enum
from pathlib import Path
from queue import Queue
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
from urllib.parse import urlparse

import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import display
from IPython.core.interactiveshell import InteractiveShell
from PIL import Image
from wbfw109.libs.utilities.ipython import (
    VisualizationManager,
    VisualizationRoot,
    display_data_frame_with_my_settings,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode
#%%


class ObjectTypes:
    """Default types"""

    class PrimitiveTypes:
        """
        TODO: ... detail all method...
        """

        class Color(Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        def __init__(self) -> None:
            self.boolean: bool = True
            self.integer: int = 0
            self.floating_point: float = 0.0
            self.enumerated: ObjectTypes.PrimitiveTypes.Color = self.Color.RED

    class CompositeTypes:
        """
        TODO: ... detail all method...
        """

        def __init__(self) -> None:
            self.string: str = ""
            self.array: array[float] = array("d", [1.0, 2.0, 3.14])

    class AbstractDataTypes:
        """
        TODO: ... detail all method...
        """

        def __init__(self) -> None:
            self.tuple_: tuple = ()  # type:ignore
            self.set: set[Any] = set()
            self.list: list[Any] = []
            self.stack: list[Any] = self.list.copy()  # LIFO
            self.dict: dict[Any, Any] = {}
            self.queue: Queue[Any] = Queue()  # FIFO

        def test_stack_operations(self) -> None:
            self.stack.extend([1, 3, 5, 7])
            self.stack.append(10)
            print(
                "stack: LIFO (push, pop)",
                [self.stack.pop() for _ in range(len(self.stack))],
            )

        def test_queue_operations(self) -> None:
            self.queue.put(1)
            self.queue.put(3)
            self.queue.put(5)
            print(
                "queue: FIFO (put, get)",
                [self.queue.get() for _ in range(self.queue.qsize())],
            )


# ObjectTypes.AbstractDataTypes().test_queue_operations()
from collections import deque

# dequeue 랑 토폴로지, https://docs.python.org/3/library/graphlib.html 이거 하고 알고리즘 풀기.
# https://docs.python.org/3/library/collections.html#collections.deque
# Critical path method
# import heapq
# Disjoint-set for Kruskal's algorithm
# [[1, 2, 3]] * 2, [1, 2, 3] * 2     // * (repetition operation)
# xx = "Cursor2"
# tuple(xx)  vs  (xx,)
# [1, 2, 3] + [4]   -> concatenation operation: create new list
# [1, 2, 3].append(4) -> in-place update
# x = [i for i in range(5)]
# x[100:3]
# x[3 : 1000]
# x[-1:2]
# Also valid:   list(map(sum, zip(card_stack_list[::2], card_stack_list[1::2])))

# heap 이 동적으로 생성되고 삭제되는 데이터에서 이 데이터들끼리 비교가 필요할 때 사용하면 좋다. 그리디 알고리즘에서 자주 사용된다.
# heapq 는 작업할떄마다 자동으로 힙정렬되므로 사용하기 좋다.


def data_structure():
    """
    single linked list
    https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes
    TODO:
    https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues
    from collections import deque
        extendleft(iterable)
            Extend the left side of the deque by appending elements from iterable. Note, the series of left appends results in reversing the order of elements in the iterable argument.
    """


#%%
