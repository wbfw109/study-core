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
    VisualizationManager,
    VisualizationRoot,
    display_data_frame_with_my_settings,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode
#%%

# Closest pair of points problem
