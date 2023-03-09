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
    ChildAlgorithmVisualization,
    VisualizationManager,
    VisualizationRoot,
    display_data_frame_with_my_settings,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


#%%


class DataSize(VisualizationRoot):
    """...
    🧑‍🤝‍🧑 quotatio fromn 🔗 https://rushter.com/blog/python-strings-and-memory
    """

    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )
        self.df_caption = [
            "⚙️ Empty string occupy 49 bytes Because it stores supplementary information",
            "    , such as hash, length, length in bytes, encoding type and string flags.",
            "  - ASCII bytes additionally occupy 1 byte",
            "⚙️ Python uses interning way to names of string as well as constants, variables, functions, etc.",
            "  - It stores only one copy of same immutable object and shares in order to save space complexity",
        ]

    def __str__(self) -> str:
        return "-"

    @classmethod
    def test_case(cls):
        data_size: DataSize = cls()
        sys.getsizeof("")
        for obj in ["", "a", "abc", 100]:
            data_size.append_line_into_df_in_wrap(
                [f"sys.getsizeof( {obj} )", f"{sys.getsizeof(obj)} bytes"]
            )
        data_size.visualize()


class ZeroBasedNumbering(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "methods: zip(), itertools.zip_longest()"

    @classmethod
    def test_case(cls):
        zero_based_index: ZeroBasedNumbering = cls()

        for text, length_text in zip(
            ["raccecar", "kayak"], ["Even length", "Odd length"]
        ):
            text_len = len(text)
            zero_based_index.append_line_into_df_in_wrap(
                [
                    "[*zip(range(0, text_len // 2), range(text_len - 1, text_len // 2 - 1, -1))]",
                    [
                        *zip(
                            range(0, text_len // 2),
                            range(text_len - 1, text_len // 2 - 1, -1),
                        )
                    ],
                    f"{length_text}. <text>={text}",
                ]
            )
        zero_based_index.append_line_into_df_in_wrap()
        i = 5
        zero_based_index.append_line_into_df_in_wrap(
            ["2 * i + 1", 2 * i + 1, "[Heap] left child of <i>. <i>=5"]
        )
        zero_based_index.append_line_into_df_in_wrap(
            ["2 * i + 1", 2 * i + 2, "[Heap] right child of <i>. <i>=5"]
        )
        zero_based_index.append_line_into_df_in_wrap(
            ["(i - 1) // 2", (i - 1) // 2, "[Heap] parent of <i>. <i>=5"]
        )

        zero_based_index.visualize()


if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = []
    VisualizationManager.call_root_classes(only_class_list=only_class_list)


# for in loop 에서 하나씩 해당 iterator 의 요소를 pop 하는 경우, --
# 0 based 에서 배열의 개수 condition 확인: 3 index - 0 index = 4개

# # Title: transpose iteration order
# def get_2048_grid_iterator(
#     direction_i: Literal[0, 1, 2, 3],
#     max_row: int,
#     max_column: int,
# ) -> list[list[MutableSequence[int]]]:
#     # response against pressed arrow key
#     match direction_i:
#         case 0:  # when up arrow key (-row direction) is pressed
#             # column 별로 계산해야함.0, 0, 1,0 2,0, 3,0 끼리. 1, 0, 1,1, 1, 2, 1, 3 끼리
#             return [
#                 [array("b", [row, column]) for row in range(0, max_row, 1)]
#                 for column in range(0, max_column, 1)
#             ]
#         case 1:  # when down arrow key (+row direction) is pressed
#             return [
#                 [array("b", [row, column]) for row in range(max_row - 1, -1, -1)]
#                 for column in range(0, max_column, 1)
#             ]
#         case 2:  # when left arrow key (-column direction) is pressed
#             return [
#                 [array("b", [row, column]) for column in range(0, max_column, 1)]
#                 for row in range(0, max_row, 1)
#             ]
#         case 3:  # when right arrow key (+column direction) is pressed
#             return [
#                 [array("b", [row, column]) for column in range(max_column - 1, -1, -1)]
#                 for row in range(0, max_row, 1)
#             ]


# >>> ["."]*3     ['.', '.', '.'] .  not ["."], ["."], ["."]
