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
import string

a = 1
b = 20
sum_x = []
for x in range(a, b + 1):
    sum_y = 0
    for y in range(1, x):
        sum_y += y
    sum_x.append(sum_y)

print(sum_x)


class MoveToMaterialLocationInMarse:
    """
    Example
        test_cases_input: list[MoveToMaterialLocationInMarse.InputDict] = [
            MoveToMaterialLocationInMarse.InputDict(
                plain_text="hellopython", key_text="abcdefghijk", rotation_count=3
            )
        ]
        MoveToMaterialLocationInMarse.test(test_cases_input)
    """

    _encrypted_key = string.ascii_lowercase
    _encrypted_key_length = len(_encrypted_key)
    _ascii_lowercase_to_natural_number: dict[int, str] = {}
    _natural_number_to_ascii_lowercase: dict[int, str] = {}
    for i, character in enumerate(_encrypted_key, start=1):
        _ascii_lowercase_to_natural_number[character] = i
        _natural_number_to_ascii_lowercase[i] = character

    @staticmethod
    def get_encryption(plain_text: str, key_text: str, rotation_count: int) -> str:
        assert len(plain_text) == len(key_text)
        key_text_length: int = len(key_text)
        encrypted_text_as_deque: deque[str] = deque()
        temp_plain_text_as_list = [character for character in plain_text]
        for i in range(key_text_length):
            natural_number_index = (
                MoveToMaterialLocationInMarse._ascii_lowercase_to_natural_number[
                    temp_plain_text_as_list[i]
                ]
                + MoveToMaterialLocationInMarse._ascii_lowercase_to_natural_number[
                    key_text[i]
                ]
            ) % MoveToMaterialLocationInMarse._encrypted_key_length
            natural_number_index = (
                natural_number_index
                if natural_number_index != 0
                else MoveToMaterialLocationInMarse._encrypted_key_length
            )

            encrypted_text_as_deque.append(
                MoveToMaterialLocationInMarse._natural_number_to_ascii_lowercase[
                    natural_number_index
                ]
            )
        encrypted_text_as_deque.rotate(rotation_count)
        return "".join(encrypted_text_as_deque)

    @staticmethod
    def get_decryption(encrypted_text: str, key_text: str, rotation_count: int) -> str:
        key_text_length: int = len(key_text)
        decrypted_text_as_deque: deque = deque(encrypted_text)
        decrypted_text_as_deque.rotate(-rotation_count)
        plain_text_as_list: list[str] = []
        for i in range(key_text_length):
            natural_number_index = (
                MoveToMaterialLocationInMarse._ascii_lowercase_to_natural_number[
                    decrypted_text_as_deque[i]
                ]
                - MoveToMaterialLocationInMarse._ascii_lowercase_to_natural_number[
                    key_text[i]
                ]
            ) % MoveToMaterialLocationInMarse._encrypted_key_length
            natural_number_index = (
                natural_number_index
                if natural_number_index != 0
                else MoveToMaterialLocationInMarse._encrypted_key_length
            )
            plain_text_as_list.append(
                MoveToMaterialLocationInMarse._natural_number_to_ascii_lowercase[
                    natural_number_index
                ]
            )
        return "".join(plain_text_as_list)

    class InputDict(TypedDict):
        plain_text: str
        key_text: str
        rotation_count: int

    @staticmethod
    def test(test_cases_input: list[InputDict]):
        print("=== format: (plain_text, encrypted_text, decrypted_text)")
        for test_case_input in test_cases_input:
            encrypted_text: str = MoveToMaterialLocationInMarse.get_encryption(
                plain_text=test_case_input["plain_text"],
                key_text=test_case_input["key_text"],
                rotation_count=test_case_input["rotation_count"],
            )
            decrypted_text = MoveToMaterialLocationInMarse.get_decryption(
                encrypted_text=encrypted_text,
                key_text=test_case_input["key_text"],
                rotation_count=test_case_input["rotation_count"],
            )
            print((test_case_input["plain_text"], encrypted_text, decrypted_text))


#%%

"""


..save point 가 필요한가? 어떻게 알지..

directed x,y 

제한 사항
    로봇은 지도 밖으로 이동할 수 없습니다.
    지도의 가로, 세로 길이는 2 이상 100 이하입니다.
    장애물을 제거하는 비용 c는 0 이상 100 이하인 정수입니다.
    장애물이 없는 지역은 0, 장애물이 있는 지역은 1, 로봇의 현재 위치는 2 그리고 로봇이 도착해야 할 목적지는 3으로 주어집니다.
    로봇의 현재 위치와 도착해야 할 목적지는 반드시 1개만 있습니다.

입출력 예
board	c	result
[ [0,0,0,0,2,0,0,0,0,0],[0,0,1,1,1,1,1,0,0,0],[0,0,1,1,1,1,1,1,0,0],[0,0,1,1,1,1,1,0,1,0],[0,0,1,1,1,1,1,0,0,0],[0,0,0,0,3,0,0,0,1,0]]	1	9
[ [0,0,0,0,2,0,0,0,0,0],[0,0,1,1,1,1,1,0,0,0],[0,0,1,1,1,1,1,1,0,0],[0,0,1,1,1,1,1,0,1,0],[0,0,1,1,1,1,1,0,0,0],[0,0,0,0,3,0,0,0,1,0]]	2	11

다익스트라, 플로이드 워셜 알고리즘
"""
