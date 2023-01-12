# %%
import collections
import copy
import dataclasses
import datetime
import inspect
import itertools
import logging
import math
import os
import pprint
import random
import re
import shutil
import string
import time
import xml.etree.ElementTree as ET
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import (
    Callable,
    ClassVar,
    Iterable,
    Literal,
    NamedTuple,
    Optional,
    Type,
    TypedDict,
    Union,
)

import IPython
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
pp = pprint.PrettyPrinter(compact=False)

# This Cell is for common declaration, and Next Cells are for specified algorithms.
# comply my structure for writting algorithms like Next Cell:

# Note
# 1. if you are using Type hint, but python version is less than 3.9, Generic type hint is required instead of type object.
#   it applies some web site: https://programmers.co.kr/
#%%


class PerfectString:
    """
    Example
        test_cases_input: list[PerfectString.InputDict] = [
            PerfectString.InputDict(
                sentence="His comments came after Pyongyang announced it had a plan to fire four missiles near the US territory of Guam."
            ),
            PerfectString.InputDict(sentence="Jackdaws love my big sphinx of quartz"),
        ]
        PerfectString.test(test_cases_input)
    """

    _perfect_string_set: set[str] = set(string.ascii_lowercase)

    @staticmethod
    def verify_perfect_string(sentence: str) -> str:
        set_difference: set[str] = PerfectString._perfect_string_set - set(
            sentence.lower()
        )

        if set_difference:
            return "".join(sorted(set_difference))
        else:
            return "perfect"

    class InputDict(TypedDict):
        sentence: str

    @staticmethod
    def test(test_cases_input: list[InputDict]):
        t0 = time.time()
        print("=== format: (required characters or 'perfect')")
        for test_case_input in test_cases_input:
            returned_string: str = PerfectString.verify_perfect_string(
                sentence=test_case_input["sentence"],
            )
            print((returned_string))
        t1 = time.time()
        print("Time taken:", t1 - t0)


class NaturalNumberDuplication:
    """
    Example
        test_cases_input: list[NaturalNumberDuplication.InputDict] = [
            NaturalNumberDuplication.InputDict(natural_number_list=[2, 1, 3, 3]),
            NaturalNumberDuplication.InputDict(natural_number_list=[3, 4, 4, 2, 5, 2, 5, 5]),
            NaturalNumberDuplication.InputDict(natural_number_list=[3, 5, 3, 5, 7, 5, 7]),
        ]
        NaturalNumberDuplication.test(test_cases_input)
    """

    @staticmethod
    def get_not_duplicated_number_list(natural_number_list: list[int]) -> list[int]:
        not_duplicated_number_list = []
        natural_number_to_count = collections.Counter(natural_number_list)
        for natural_number, count in natural_number_to_count.items():
            if count == 1:
                not_duplicated_number_list.append(natural_number)

        if not not_duplicated_number_list:
            not_duplicated_number_list = [-1]
        else:
            not_duplicated_number_list = list(sorted(not_duplicated_number_list))

        return not_duplicated_number_list

    class InputDict(TypedDict):
        natural_number_list: list[int]

    @staticmethod
    def test(test_cases_input: list[InputDict]):
        t0 = time.time()
        print("=== format: (not duplicated number_list or [-1])")
        for test_case_input in test_cases_input:
            not_duplicated_number_list: str = (
                NaturalNumberDuplication.get_not_duplicated_number_list(
                    natural_number_list=test_case_input["natural_number_list"],
                )
            )
            print((not_duplicated_number_list))
        t1 = time.time()
        print("Time taken:", t1 - t0)
