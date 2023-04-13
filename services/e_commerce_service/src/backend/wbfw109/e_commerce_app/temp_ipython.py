"""
1. distinguish data to use in NoSQL and PostgreSQL


주문, 상품, 고객, 결제



ERD, 정규화, 캐싱, 튜닝, 인덱싱, 파니셔닝, scalability
백업 전략, 테스팅
 availability 24/7,
"""

# %%
from __future__ import annotations

import itertools
import random
from collections.abc import Generator, Sequence
from enum import Enum
from itertools import zip_longest
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

import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import display
from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import display_data_frame_with_my_settings

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# %doctest_mode

# from google.protobuf.internal import api_implementation
# print(api_implementation.Type())


# %%
import math

math.comb(4, 2)


# %%
