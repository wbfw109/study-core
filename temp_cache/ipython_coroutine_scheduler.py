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
from types import ModuleType
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

VisualizationManager.call_modules([Path("wbfw109/labs/databases")])


# TODO: Svelte.. -> Deploy in Netlify by using GitHub workflow..

#%%
# Todo: write code After reading async, async generator PEP


class ComparisonIter:
    """
    https://docs.python.org/3/glossary.html#term-iterable
    https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
    https://docs.python.org/3/glossary.html#term-iterator
    https://peps.python.org/pep-0342/
    https://docs.python.org/3/reference/expressions.html#generator-iterator-methods
    """

    class MyTrampolineTest:
        def __str__(self) -> str:
            return "Generator usage e.g.: Trampoline"

    @classmethod
    def test_case_gen(cls):
        pass
        # my_echo_generator_test = ComparisonIter.MyEchoGeneratorTest()
        # my_echo_generator_test.speak()
        # my_thumb_generator_test = ComparisonIter.MyThumbGeneratorTest()
        # my_thumb_generator_test.write_thumbnails()

    @classmethod
    def main(cls):
        display_header_with_my_settings(columns=[cls.__name__])
        cls.test_case_gen()


ComparisonIter.main()


#%%

#  but must currently use awkward workarounds for the inability to pass values or exceptions into generators
class Trampoline:
    """Manage communications between coroutines"""

    def __init__(self) -> None:
        self.queue = collections.deque()
        self.running = False

    def add(self, coroutine: Generator):
        """Request that a coroutine be executed"""
        self.schedule(coroutine)

    def run(self):
        result = None
        self.running = True
        try:
            while self.running and self.queue:
                func = self.queue.popleft()
                result = func()
            return result
        finally:
            self.running = False

    def stop(self) -> None:
        self.running = False

    def schedule(self, coroutine: Generator, stack=(), val=None, *exceptions):
        """
        First loop
            add coroutine server as coroutine with no (stack, val, exceptions) arguments
            coroutine_send(value) equals next(), get connected sock yielded as value.
            There is no case of conditional True, so queue.append and run()
        """

        def resume():
            nonlocal coroutine, stack, val, exceptions
            value = val
            try:
                if exceptions:
                    value = coroutine.throw(value, *exceptions)
                else:
                    value = coroutine.send(value)
            except:
                # ? except as e 에서 e.traceback 뽑아낼 수 있나
                if stack:
                    # send the error back to the "caller"
                    self.schedule(stack[0], stack[1], *sys.exc_info())
                else:
                    # Nothing left in this pseudothread to handle it, let it propagate to the run loop
                    raise

            if isinstance(value, types.GeneratorType):
                # Note that nonblocking_read, nonblocking_wrtie, non_blocking_accept are I/O coroutines, namely Generators. => parallel
                # Yielded to a specific coroutine, push the current one on the stack, and call the new one with no args
                self.schedule(value, (coroutine, stack))

            elif stack:
                # Note that value from nonblocking IO coroutine is not Generator. so,
                # Yielded a result, pop the stack and send the value to the caller
                self.schedule(stack[0], stack[1], value)

            # else: this pseudothread has ended

        self.queue.append(resume)


# Note that nonblocking_read, nonblocking_wrtie, non_blocking_accept are I/O coroutines => parallel
# A simple echo server, and code to run it using a trampoline
# (presumes the existence of nonblocking_read, nonblocking_write, and other I/O coroutines
# , that e.g. raise ConnectionLost if the connection is closed):

# coroutine function that echos data back on a connected
# socket
#
def echo_handler(connected_sock):
    while True:
        try:
            data = yield nonblocking_read(connected_sock)
            yield nonblocking_write(connected_sock, data)
        except ConnectionLost:
            pass  # exit normally if connection lost


# coroutine function that listens for connections on a
# socket, and then launches a service "handler" coroutine
# to service the connection
#
def listen_on(trampoline, server_sock, handler):
    while True:
        # get the next incoming connection
        connected_sock = yield nonblocking_accept(server_sock)

        # start another coroutine to handle the connection
        trampoline.add(handler(connected_sock))


# Create a scheduler to manage all our coroutines
t = Trampoline()

# Create a coroutine instance to run the echo_handler on
# incoming connections
#
coroutine_server = listen_on(t, listening_socket("localhost", "echo"), echo_handler)

# Add the coroutine to the scheduler
t.add(coroutine_server)

# loop forever, accepting connections and servicing them
# "in parallel"
#
t.run()


#! Generator[YieldType, SendType, ReturnType]
