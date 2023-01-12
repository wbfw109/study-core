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


class ForStatement(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self, columns=["eval", "print"], has_df_lock=False, should_highlight=True
        )

    def __str__(self) -> str:
        return "parameter <__start> is inclusive, parameter <__stop> is exclusive."

    @classmethod
    def test_case(cls):
        for_statement: ForStatement = cls()
        start_i = -1
        end_i = 4
        for_statement.append_line_into_df_in_wrap(["", "start_i = -1, end_i = 4"])
        for_statement.append_line_into_df_in_wrap(
            [
                "[i for i in range(start_i, end_i, 1)]",
                [i for i in range(start_i, end_i, 1)],
            ]
        )
        for_statement.append_line_into_df_in_wrap(
            [
                "[i for i in range(end_i, start_i, -1)]",
                [i for i in range(end_i, start_i, -1)],
            ]
        )

        for_statement.visualize()


class Operators(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self, columns=["eval", "print"], has_df_lock=False, should_highlight=True
        )

    def __str__(self) -> str:
        return "TODO: ...."

    @classmethod
    def test_case(cls):
        for_statement: Operators = cls()
        # start_i = -1
        # end_i = 4
        # for_statement.append_line_into_df_in_wrap(["", "start_i = -1, end_i = 4"])

        for_statement.visualize()


if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = []
    VisualizationManager.call_root_classes(only_class_list=only_class_list)

#%%
