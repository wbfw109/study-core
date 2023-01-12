# %%
#
import logging
import os

import IPython
from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

import collections
import copy
import dataclasses
import datetime
import itertools
import math
import random
import re
import shutil
import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path

#%%
from typing import Callable, Iterable, Iterator, NamedTuple, Optional, Tuple, Union

import numpy
import pandas
from PIL import Image, ImageDraw, ImageFilter

for x, group in itertools.groupby([5, 3, 1, 3, 3, 7, 8, 9, 9, 9], lambda x: x):
    # Note: all iterable from "Iterators terminating on the shortest input sequence" of itertools can be exhaustive.
    # use after object converting to such as list.
    group_as_list = list(group)
    is_exhaustive = 1

    copy_group = copy.copy(group)  # or itertools.permutations(group), etc...
    for x in group:
        # can not run these because of used iterable object in method copy.
        is_exhaustive = 2
    x, is_exhaustive

    for x in group:
        # can not run these
        is_exhaustive = 3
    x, is_exhaustive

    for x in group_as_list:
        # can not run these
        is_exhaustive = 4
    x, is_exhaustive

    break
