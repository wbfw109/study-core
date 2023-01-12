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
    NamedTuple,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import IPython
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
pp = pprint.PrettyPrinter(compact=False)

# This Cell is for common declaration, and Next Cells are for specified algorithms.
# comply my structure for writting algorithms like Next Cell:

#%%
import string


class EncryptionAndDecryption_1:
    """
    Example
        test_cases_input: list[EncryptionAndDecryption_1.InputDict] = [
            EncryptionAndDecryption_1.InputDict(
                plain_text="hellopython", key_text="abcdefghijk", rotation_count=3
            )
        ]
        EncryptionAndDecryption_1.test(test_cases_input)
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
                EncryptionAndDecryption_1._ascii_lowercase_to_natural_number[
                    temp_plain_text_as_list[i]
                ]
                + EncryptionAndDecryption_1._ascii_lowercase_to_natural_number[
                    key_text[i]
                ]
            ) % EncryptionAndDecryption_1._encrypted_key_length
            natural_number_index = (
                natural_number_index
                if natural_number_index != 0
                else EncryptionAndDecryption_1._encrypted_key_length
            )

            encrypted_text_as_deque.append(
                EncryptionAndDecryption_1._natural_number_to_ascii_lowercase[
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
                EncryptionAndDecryption_1._ascii_lowercase_to_natural_number[
                    decrypted_text_as_deque[i]
                ]
                - EncryptionAndDecryption_1._ascii_lowercase_to_natural_number[
                    key_text[i]
                ]
            ) % EncryptionAndDecryption_1._encrypted_key_length
            natural_number_index = (
                natural_number_index
                if natural_number_index != 0
                else EncryptionAndDecryption_1._encrypted_key_length
            )
            plain_text_as_list.append(
                EncryptionAndDecryption_1._natural_number_to_ascii_lowercase[
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
            encrypted_text: str = EncryptionAndDecryption_1.get_encryption(
                plain_text=test_case_input["plain_text"],
                key_text=test_case_input["key_text"],
                rotation_count=test_case_input["rotation_count"],
            )
            decrypted_text = EncryptionAndDecryption_1.get_decryption(
                encrypted_text=encrypted_text,
                key_text=test_case_input["key_text"],
                rotation_count=test_case_input["rotation_count"],
            )
            print((test_case_input["plain_text"], encrypted_text, decrypted_text))
