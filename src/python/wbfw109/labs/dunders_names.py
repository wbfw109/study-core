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
from decimal import Decimal
from enum import Enum
from fractions import Fraction
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


class DundersTruthyValue(VisualizationRoot):
    class NoBoolAndLen(VisualizationRoot):
        def __init__(self) -> None:
            VisualizationRoot.__init__(
                self,
                columns=["eval", "called_function", "print"],
                has_df_lock=False,
                should_highlight=True,
            )

        def __str__(self) -> str:
            return "methods: ()"

    class OnlyLen(VisualizationRoot):
        def __init__(self) -> None:
            VisualizationRoot.__init__(
                self,
                columns=["eval", "called_function", "print"],
                has_df_lock=False,
                should_highlight=True,
            )

        def __str__(self) -> str:
            return "methods: (__len__)"

        def __len__(self) -> int:
            return 0

    class BoolAndLen(VisualizationRoot):
        def __init__(self) -> None:
            VisualizationRoot.__init__(
                self,
                columns=["eval", "called_function", "print"],
                has_df_lock=False,
                should_highlight=True,
            )

        def __str__(self) -> str:
            return "methods: (__bool__, __len__)"

        def __bool__(self) -> bool:
            return False

        def __len__(self) -> int:
            return 1

    class FalsyValues(VisualizationRoot):
        def __init__(self) -> None:
            VisualizationRoot.__init__(
                self,
                columns=["eval", "print"],
                has_df_lock=False,
                should_highlight=True,
            )

        def __str__(self) -> str:
            return "It can be evaluated in condition as Pythonic way."

        def check_values(self) -> None:
            self.append_line_into_df_in_wrap(
                ["not all([None, False]", not all([None, False])]
            )
            self.append_line_into_df_in_wrap(
                [
                    "not all([0, 0.0, 0j, Decimal(0), Fraction(0, 1)]",
                    not all([0, 0.0, 0j, Decimal(0), Fraction(0, 1)]),
                ]
            )
            self.append_line_into_df_in_wrap(
                ["not all(['', (), [], {}, set(), range(0)])", not all(["", (), [], {}, set(), range(0)])]  # type: ignore
            )
            self.append_line_into_df_in_wrap(["not all([...]", not all([not ...])])

    @classmethod
    def test_case(cls) -> None:  # type: ignore
        no_bool_and_len = DundersTruthyValue.NoBoolAndLen()
        no_bool_and_len.append_line_into_df_in_wrap(
            ["bool(obj)", "", bool(no_bool_and_len)]
        )
        only_len = DundersTruthyValue.OnlyLen()
        only_len.append_line_into_df_in_wrap(
            ["bool(obj)", "__len__ as 0", bool(only_len)]
        )
        bool_and_len = DundersTruthyValue.BoolAndLen()
        bool_and_len.append_line_into_df_in_wrap(
            ["bool(obj)", "__bool__ as False", bool(bool_and_len)]
        )
        falsy_values = DundersTruthyValue.FalsyValues()
        falsy_values.check_values()

        no_bool_and_len.visualize()
        only_len.visualize()
        bool_and_len.visualize()
        falsy_values.visualize()


class DundersGetattr(VisualizationRoot):
    """It is required to manage to sub-class of <OnlyGetattr> in this class because of <__getattribute__> property.
    https://docs.python.org/3/reference/datamodel.html#object.__getattr__
    https://docs.python.org/3/reference/datamodel.html#object.__getattribute__

    """

    class OnlyGetattr:
        def __init__(self):
            self.my_attribute: str = "my_attribute_value"

        def __getattr__(self, name: str) -> str:
            return "__getattr__ called when attribute not found in class tree of self."

        def __str__(self) -> str:
            return "methods: __getattr__"

    class GetattrAndGetattribute1(OnlyGetattr):
        def __getattribute__(self, name: str) -> str:
            return "__getattribute__ called unconditionally."

        def __str__(self) -> str:
            return "methods: (__getattr__, __getattribute__)"

    class GetattrAndGetattribute2(OnlyGetattr):
        def __getattribute__(self, name: str) -> Never:
            """redirect to __getattr__"""
            raise AttributeError

        def __str__(self) -> str:
            return "methods: (__getattr__, __getattribute__ with AttributeError)"

    class GetattrAndGetattribute3(OnlyGetattr):
        def __getattribute__(self, name: str) -> str:
            return object.__getattribute__(self, name)

        def __str__(self) -> str:
            return "methods: (__getattr__, __getattribute__ with object.__getattribute__(self, name))"

    @classmethod
    def test_case(cls) -> None:  # type: ignore
        for obj in [
            DundersGetattr.OnlyGetattr(),
            DundersGetattr.GetattrAndGetattribute1(),
            DundersGetattr.GetattrAndGetattribute2(),
            DundersGetattr.GetattrAndGetattribute3(),
        ]:
            visualization_root = VisualizationRoot(
                columns=["eval", "print"],
                has_df_lock=False,
                should_highlight=True,
                header_string=str(obj),
            )
            visualization_root.append_line_into_df_in_wrap(
                ["obj.my_attribute", obj.my_attribute]
            )
            visualization_root.append_line_into_df_in_wrap(
                ["obj.unknown_attribute", obj.unknown_attribute]  # type:ignore
            )
            visualization_root.append_line_into_df_in_wrap(
                ["obj.__dict__", obj.__dict__]
            )
            visualization_root.visualize()


class DundersIter(VisualizationRoot):
    """
    https://docs.python.org/3/glossary.html#term-iterable
    https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
    https://docs.python.org/3/glossary.html#term-iterator
    https://peps.python.org/pep-0342/
    https://docs.python.org/3/reference/expressions.html#generator-iterator-methods
    """

    class Items(VisualizationRoot):
        def __init__(self) -> None:
            # Note that (lists, tuples, mappings) is iterable. not iterator.
            self.items: Sequence[Any] = list(range(5))
            VisualizationRoot.__init__(
                self,
                columns=["eval", "value", "print"],
                has_df_lock=False,
                should_highlight=True,
            )

    ItemsExtends = TypeVar("ItemsExtends", bound=Items)

    class Items1(Items):
        def __getitem__(self, index: int) -> Any:
            return self.items[index]

        def __str__(self) -> str:
            return "methods: __getitem__"

    class Items2(Items):
        def __iter__(self) -> Iterator[Any]:
            return iter(self.items)

        def __str__(self) -> str:
            return "methods: __iter__"

    @classmethod
    def test_case_iter(cls):  # type: ignore
        for obj in [
            DundersIter.Items1(),
            DundersIter.Items2(),
        ]:
            iterable_styled_name: str = "➖iterable➖"
            obj_styled_name: str = "➕obj➕"
            obj.append_line_into_df_in_wrap(
                [
                    f"isinstance({iterable_styled_name}, Iterable)",
                    isinstance(obj, Iterable),
                    "",
                ],
            )
            obj.append_line_into_df_in_wrap(
                [
                    f"isinstance(iter({iterable_styled_name}), Iterable)",
                    isinstance(iter(obj), Iterable),  # type: ignore
                    "",
                ],
            )
            obj.append_line_into_df_in_wrap()
            obj.append_line_into_df_in_wrap(
                [
                    f"isinstance({iterable_styled_name}, Iterator)",
                    isinstance(obj, Iterator),
                    "",
                ],
            )
            obj.append_line_into_df_in_wrap(
                [
                    f"isinstance(iter({iterable_styled_name}), Iterator)",
                    isinstance(iter(obj), Iterator),  # type: ignore
                    "",
                ],
            )
            obj.append_line_into_df_in_wrap()

            def append_continuous_elements(
                *, eval_name: str, my_iterable: Iterable[Any]
            ) -> None:
                # test continuous iteration
                for i in range(1, 3):
                    obj.append_line_into_df_in_wrap(
                        [
                            f"iteration: {i} of {eval_name}",
                            f"{[x for x in my_iterable]}",
                            "",
                        ],
                    )

            append_continuous_elements(eval_name=obj_styled_name, my_iterable=obj)  # type: ignore
            append_continuous_elements(eval_name=f"iter({obj_styled_name})", my_iterable=iter(obj))  # type: ignore
            obj.visualize()

    class GenExpressionExample(VisualizationRoot):
        def __init__(self) -> None:
            VisualizationRoot.__init__(
                self,
                columns=["eval", "print"],
                has_df_lock=False,
                should_highlight=True,
            )

        def __str__(self) -> str:
            return "Generator expression"

        def show_simple_gen_expression(self) -> None:
            self.append_line_into_df_in_wrap(
                ["list(_ for _ in range(5))", list(_ for _ in range(5))]
            )

        @classmethod
        def test_case(cls) -> None:  # type: ignore
            gen_expression_example = cls()
            gen_expression_example.show_simple_gen_expression()
            gen_expression_example.visualize()

    class EchoGenExample(VisualizationRoot):
        def __init__(self) -> None:
            VisualizationRoot.__init__(
                self,
                columns=["send_value", "yield_value", "eval", "print"],
                has_df_lock=False,
                should_highlight=True,
            )

        def __str__(self) -> str:
            return "Generator usage e.g.: Echo server"

        def echo(
            self, df: pd.DataFrame, *, value: Any = None
        ) -> Generator[Any, Any, None]:
            self.append_line_into_df_in_wrap(
                [
                    "➖",
                    "➖",
                    "",
                    "Execution starts when 'next()' is called for the first time.",
                ]
            )
            try:
                while True:
                    try:
                        value = yield value
                    except Exception as e:
                        value = e
            finally:
                self.append_line_into_df_in_wrap(
                    [
                        "➖",
                        "➖",
                        "",
                        "⭕ Don't forget to clean up when 'close()' is called.",
                    ]
                )

        def speak(self) -> None:
            words = 1
            generator = self.echo(self.df, value=words)
            response = next(generator)
            self.append_line_into_df_in_wrap([words, response, "", ""])
            response = next(generator)
            self.append_line_into_df_in_wrap([None, response, "", ""])
            words = 2
            response = generator.send(words)
            self.append_line_into_df_in_wrap([words, response, "", ""])
            self.append_line_into_df_in_wrap(
                ["➖", "➖", "generator.throw(TypeError)", ""]
            )
            generator.throw(TypeError)
            generator.close()

        @classmethod
        def test_case(cls) -> None:  # type: ignore
            echo_gen_example = cls()
            echo_gen_example.speak()
            echo_gen_example.visualize()

    class ThumbGenExample(VisualizationRoot):
        def __init__(self) -> None:
            VisualizationRoot.__init__(
                self,
                columns=["print"],
                has_df_lock=False,
                should_highlight=True,
            )
            # subplot(r,c) provide the no. of rows and columns
            _, self.axarr = plt.subplots(2, 2)  # type: ignore

        def __str__(self) -> str:
            return "Generator usage e.g.: Thumb Generator"

        @staticmethod
        def consumer(
            func: Callable[..., Generator[Any, Any, Any]]
        ) -> Callable[..., Generator[Any, Any, Any]]:
            P = ParamSpec("P")

            def wrapper(*args: P.args, **kw: P.kwargs) -> Generator[Any, Any, Any]:
                gen: Generator[Any, Any, Any] = func(*args, **kw)
                next(gen)
                return gen

            wrapper.__name__ = func.__name__
            wrapper.__dict__ = func.__dict__
            wrapper.__doc__ = func.__doc__
            return wrapper

        @consumer
        def thumbnail_pager(
            self,
            thumb_size: tuple[int, int],
            destination: Generator[None, Optional[Image.Image], None],
        ) -> Generator[None, np.ndarray[Any, np.dtype[np.uint]], None]:
            while True:
                pillow_image: Optional[Image.Image] = None
                try:
                    image: np.ndarray[Any, np.dtype[np.uint]] = yield
                    pillow_mode = ""
                    self.append_line_into_df_in_wrap(
                        [f"➕ [Thumbnail pager] receive image size: {image.shape}"]
                    )
                    if image.ndim == 3:
                        match image.shape[-1]:
                            case 1:
                                image = image.squeeze()
                                pillow_mode = "L"
                                self.axarr[0][0].imshow(image, cmap="gray")  # type: ignore
                            case 3:
                                pillow_mode = "RGB"
                                self.axarr[0][1].imshow(image)  # type: ignore
                            case _:
                                pass
                    else:
                        self.append_line_into_df_in_wrap(
                            [
                                "[Pass] Not Support Option: Image must be (Gray scale | RGB | RGBA)"
                            ]
                        )
                        raise GeneratorExit
                    pillow_image = Image.fromarray(image, mode=pillow_mode)  # type: ignore
                    pillow_image.thumbnail(thumb_size)
                except GeneratorExit:
                    # close() was called, so flush any pending output
                    destination.send(pillow_image)
                    # then close the downstream consumer, and exit
                    destination.close()
                    return
                else:
                    # we finished a page full of thumbnails, so send it
                    # downstream and keep on looping
                    destination.send(pillow_image)

        @consumer
        def image_writer(
            self,
            dir: Optional[Path] = None,
        ) -> Generator[None, Optional[Image.Image], None]:
            file_no = 1
            while True:
                thumb_image: Optional[Image.Image] = yield
                if not thumb_image:
                    continue
                if thumb_image.mode == "L":
                    self.axarr[1][0].imshow(thumb_image, cmap="gray")  # type: ignore
                else:
                    self.axarr[1][1].imshow(thumb_image)  # type: ignore
                self.append_line_into_df_in_wrap(
                    [
                        f"➖ [Image writer] pseudo write image_no: file {file_no}, size: {thumb_image.size}"
                    ]
                )
                file_no += 1

        def write_thumbnails(self):
            THUMB_SIZE: tuple[int, int] = (50, 50)
            byte_iinfo = np.iinfo(np.uint8)
            rng = np.random.default_rng()
            random_grayscale_np_image = rng.integers(  # type: ignore
                low=byte_iinfo.min,
                high=byte_iinfo.max + 1,
                size=(100, 100, 1),
                dtype=np.uint8,
            )
            random_true_color_np_image = rng.integers(  # type: ignore
                low=byte_iinfo.min,
                high=byte_iinfo.max + 1,
                size=(100, 100, 3),
                dtype=np.uint8,
            )

            plt.figure(clear=True)  # type: ignore
            pipeline: Generator[
                None, np.ndarray[Any, np.dtype[np.uint]], None
            ] = self.thumbnail_pager(THUMB_SIZE, destination=self.image_writer())
            pipeline.send(random_grayscale_np_image)
            pipeline.send(random_true_color_np_image)
            pipeline.close()

        def visualize(self):
            display_data_frame_with_my_settings(self.df)
            plt.show()  # type: ignore
            plt.close()  # type: ignore

        @classmethod
        def test_case(cls) -> None:  # type: ignore
            thumb_gen_example = cls()
            thumb_gen_example.write_thumbnails()
            thumb_gen_example.visualize()

    @classmethod
    def test_case_gen(cls):
        DundersIter.GenExpressionExample.main()
        DundersIter.EchoGenExample.main()
        DundersIter.ThumbGenExample.main()

    @classmethod
    def test_case(cls) -> None:  # type: ignore
        cls.test_case_iter()
        cls.test_case_gen()


if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [DundersTruthyValue]
    VisualizationManager.call_root_classes(only_class_list=only_class_list)
