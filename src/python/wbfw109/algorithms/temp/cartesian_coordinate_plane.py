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
    Literal,
    NamedTuple,
    Optional,
    Type,
    TypedDict,
    Union,
)

import IPython
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
pp = pprint.PrettyPrinter(compact=False)

# This Cell is for common declaration, and Next Cells are for specified algorithms.
# comply my structure for writting algorithms like Next Cell:

# Note
# 1. if you are using Type hint, but python version is less than 3.9, Generic type hint is required instead of type object.
#   it applies some web site: https://programmers.co.kr/
#%%
# It includes tiled graph.


class Point2D(NamedTuple):
    x: int
    y: int


@dataclasses.dataclass
class Rectangle:
    start_point: Point2D = dataclasses.field(default=Point2D)
    end_point: Point2D = dataclasses.field(default=Point2D)

    def get_area(self) -> int:
        return (self.end_point.x - self.start_point.x) * (
            self.end_point.y - self.start_point.y
        )


class SlicingByNaturalNumberAxis_1:
    """
    It requires additional Classes: Point2D, Rectangle.

    Example
        test_cases_input: list[SlicingByNaturalNumberAxis_1.Inputdict] = [
            SlicingByNaturalNumberAxis_1.Inputdict(
                max_x_point=4, max_y_point=4, x_axis_list=[1], y_axis_list=[3]
            ),
            SlicingByNaturalNumberAxis_1.Inputdict(
                max_x_point=3, max_y_point=4, x_axis_list=[2], y_axis_list=[1, 2]
            ),
        ]
        SlicingByNaturalNumberAxis_1.test(test_cases_input)

    제한사항
        - n, m은 오른쪽 위 꼭짓점의 좌표를 나타냅니다.
        - n, m은 2 이상 10,000 이하의 자연수입니다.
        - x_axis는 y축에 평행한 선분의 x축 좌표가 들어있는 배열이며, 길이는 100 이하입니다.
        - y_axis는 x축에 평행한 선분의 y축 좌표가 들어있는 배열이며, 길이는 100 이하입니다.
        - x_axis의 원소는 1 이상 n 미만의 자연수가 오름차순으로 들어있으며, 중복된 값이 들어있지 않습니다.
        - y_axis의 원소는 1 이상 m 미만의 자연수가 오름차순으로 들어있으며, 중복된 값이 들어있지 않습니다.
    """

    _filter_x_is_less_than_limit_value: Callable[
        [Point2D, int, int], Optional[Point2D]
    ] = (
        lambda point2d, x_limit, y_limit: True
        if point2d.x < x_limit and point2d.y < y_limit
        else False
    )
    _filter_x_is_greater_than_limit_value: Callable[
        [Point2D, int, int], Optional[Point2D]
    ] = (
        lambda point2d, x_limit, y_limit: True
        if point2d.x > x_limit and point2d.y > y_limit
        else False
    )

    @staticmethod
    def get_rectangle_list(
        max_x_point: int,
        max_y_point: int,
        x_axis_list: list[int],
        y_axis_list: list[int],
    ) -> int:
        # Note: it must be sorted for further processing.
        intersection_x_point_list_with_edge_of_axis: list[int] = [
            0,
            *x_axis_list,
            max_x_point,
        ]
        intersection_y_point_list_with_edge_of_axis: list[int] = [
            0,
            *y_axis_list,
            max_y_point,
        ]
        intersection_point_list_with_edge_of_axis: list[Point2D] = list(
            (
                Point2D(x, y)
                for x in intersection_x_point_list_with_edge_of_axis
                for y in intersection_y_point_list_with_edge_of_axis
            )
        )

        x_filtered = filter(
            lambda point2d: SlicingByNaturalNumberAxis_1._filter_x_is_less_than_limit_value(
                point2d, max_x_point, max_y_point
            ),
            intersection_point_list_with_edge_of_axis,
        )
        y_filtered = filter(
            lambda point2d: SlicingByNaturalNumberAxis_1._filter_x_is_greater_than_limit_value(
                point2d, 0, 0
            ),
            intersection_point_list_with_edge_of_axis,
        )
        rectangle_list: list[Rectangle] = [
            Rectangle(start_point=rectangle_points[0], end_point=rectangle_points[1])
            for rectangle_points in zip(x_filtered, y_filtered)
        ]

        return rectangle_list

    @staticmethod
    def get_maximum_area_of_rectangle_list(rectangle_list: list[Rectangle]) -> int:
        return max([rectangle.get_area() for rectangle in rectangle_list])

    class Inputdict(TypedDict):
        max_x_point: int
        max_y_point: int
        x_axis_list: list[int]
        y_axis_list: list[int]

    @staticmethod
    def test(test_cases_input: list[Inputdict]):
        t0 = time.time()
        print("=== format: (maximum area of rectangle list)")
        for test_case_input in test_cases_input:
            rectangle_list: list[
                Rectangle
            ] = SlicingByNaturalNumberAxis_1.get_rectangle_list(
                max_x_point=test_case_input["max_x_point"],
                max_y_point=test_case_input["max_y_point"],
                x_axis_list=test_case_input["x_axis_list"],
                y_axis_list=test_case_input["y_axis_list"],
            )
            maximum_area_of_rectangle_list: int = (
                SlicingByNaturalNumberAxis_1.get_maximum_area_of_rectangle_list(
                    rectangle_list
                )
            )
            print((maximum_area_of_rectangle_list))
        t1 = time.time()
        print("Time taken:", t1 - t0)
