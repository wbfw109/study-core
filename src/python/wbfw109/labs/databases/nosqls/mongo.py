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
from bson.json_util import dumps, loads
from pymongo import InsertOne, MongoClient

# Json 정의파일 옮기기
# TODO: Visualization manager extension to API and Database Fixture
# ?? Json Scheme 정의 파일? .. 다른것들처럼 만드는방법
if __name__ == "__main__":
    my_db_name: LiteralString = "test"
    my_collection_name: LiteralString = "my_collection"

    client = MongoClient(
        host="localhost", port=27017, username="root", password="example"
    )
    my_db = client[my_db_name]
    my_db.name
    my_collection = my_db[my_collection_name]
    requesting = []

    with open("ref/algorithms_ref.json", mode="br") as f:
        bson_data = loads(f.read())
        previous_data = bson_data["google"]["code_jam"]
        for data in previous_data:
            ordered_data = collections.OrderedDict(data)
            ordered_data["company"] = "google"
            ordered_data["contest"] = "code_jam"
            ordered_data.move_to_end("contest", last=False)
            ordered_data.move_to_end("company", last=False)
            requesting.append(ordered_data)
        # for yy in bson_data:
        #     requesting.append(InsertOne(yy))
    result = my_collection.insert_many(requesting)
    result
    my_cursor = my_collection.find({"season": "2022", "round": "qualification"})
    counter = 3
    for i, x in enumerate(my_cursor):
        x
        if i >= counter:
            break
    client.close()

"""import svgling
svgling.draw_tree(("S", ("NP", ("D", "the"), ("N", "elephant")), ("VP", ("V", "saw"), ("NP", ("D", "the"), ("N", "rhinoceros")))))
"""
