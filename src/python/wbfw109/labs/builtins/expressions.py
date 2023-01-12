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


class LambdaExpression(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "terms (lambda expression == lambda form)"

    @classmethod
    def test_case(cls):
        lambda_expression: LambdaExpression = cls()
        lambda_expression.append_line_into_df_in_wrap(
            [
                "[(lambda: i)() for i in range(10)]",
                [(lambda: i)() for i in range(10)],
            ]
        )
        lambda_expression.append_line_into_df_in_wrap(
            [
                "[func() for func in [lambda: i for i in range(10)]]",
                [func() for func in [lambda: i for i in range(10)]],
            ]
        )
        lambda_expression.append_line_into_df_in_wrap(
            [
                "[func() for func in [lambda x=i: x for i in range(10)]]",
                [func() for func in [lambda x=i: x for i in range(10)]],  # type: ignore
                "if you require to use lambda in for-loop with different argument, save lambda into collection data type and use like this.",
            ]
        )
        lambda_expression.df_caption = [
            "⚙️ Note that functions created with lambda expressions cannot contain statements or annotations."
        ]

        lambda_expression.visualize()


if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = []
    VisualizationManager.call_root_classes(only_class_list=only_class_list)
