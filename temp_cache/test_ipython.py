# %%
from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import dataclasses
import datetime
import enum
import functools
import importlib
import inspect
import itertools
import json
import logging
import math
import operator
import os
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
from pprint import pprint
from types import ModuleType
from typing import (
    Any,
    Callable,
    Final,
    Generic,
    Iterable,
    Iterator,
    Literal,
    LiteralString,
    Mapping,
    MutableMapping,
    MutableSequence,
    NamedTuple,
    Never,
    NotRequired,
    Optional,
    ParamSpec,
    Self,
    Tuple,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
)
from urllib.parse import urlparse

import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import display
from IPython.core.interactiveshell import InteractiveShell
from PIL import Image
from wbfw109.libs.objects.object import (
    MixInJsonSerializable,
    get_default_value_map,
    get_default_value_map_as_type,
)
from wbfw109.libs.parsing import ExplicitSimpleSyntaxNode
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

cnt_iter = itertools.count(0)
arr = [
    [0, 2, 3, 4, 0, 1],
    [3, 0, 3, 0, 0, 1],
    [6, 2, 4, 0, 0, 1],
    [0, 2, 0, 0, 2, 1],
    [4, 4, 3, 4, 4, 1],
]

for x in range(3):
    print(x)
else:
    # loop fell through without finding a factor
    print("is a prime number")
