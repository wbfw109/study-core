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
from queue import Queue
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
import svgling
from IPython import display
from IPython.core.interactiveshell import InteractiveShell
from PIL import Image
from wbfw109.libs.parsing import (
    convert_implicit_syntax_node_dict_to_tree,
    convert_syntax_tree_to_svgling_style,
)
from wbfw109.libs.utilities.ipython import (
    ChildAlgorithmVisualization,
    MixInParentAlgorithmVisualization,
    VisualizationManager,
    VisualizationRoot,
    display_data_frame_with_my_settings,
    visualize_implicit_tree,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode
#%%
# TODO: random graph generator, Trémaux tree.. and if larger, change code as MixInParentAlgorithmVisualization.
# TODO: Iterative deepening depth-first search and detail complexity...


class Glossary(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["context", "word", "meaning"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "📖 Glossary for Graph search"

    @classmethod
    def test_case(cls):
        glossary: Glossary = cls()
        glossary.append_line_into_df_in_wrap(["BigO", "V", "'Vertex' (node)"])
        glossary.append_line_into_df_in_wrap(["BigO", "E", "'Edge'"])
        glossary.append_line_into_df_in_wrap(
            ["BigO", "d", "'distance' from the start node"]
        )
        glossary.append_line_into_df_in_wrap(
            ["BigO", "b", "'branching factor'; the number of children at each node"]
        )
        glossary.append_line_into_df_in_wrap(
            ["node name", "<upper character>", "Explicit node"]
        )
        glossary.append_line_into_df_in_wrap(
            [
                "node name",
                "<lower character>",
                "Implicit node (not specified in edges in <given node>",
            ]
        )
        glossary.visualize()


@dataclasses.dataclass
class GraphDST:
    graph: dict[str, list[str]]
    root_node: str

    @staticmethod
    def get_default_implicit_dict() -> GraphDST:
        return GraphDST(
            graph={
                "A": ["B", "C", "X"],
                "B": ["A", "D", "E"],
                "C": ["A", "G", "H"],
                "D": ["B"],
                "E": ["B", "F"],
                "F": ["E"],
                "G": ["C"],
                "H": ["C", "y"],
                "X": ["A", "z"],
            },
            root_node="A",
        )


class DfsAndBfs(MixInParentAlgorithmVisualization):
    class BFS(ChildAlgorithmVisualization[GraphDST]):
        def __init__(self, /, dst: Optional[GraphDST]) -> None:
            super().__init__(columns=["elapsed time", "verification"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(
                ["-", "-", "|V| + |E|  =  b^d", "|V|"]
            )
            self.result: list[str] = []
            self.big_o_visualization.df_caption = [
                "⚙️ [Worse-case] Time complexity: O(|V| + |E|)",
                "  - In the case that every vertex and every edge will be explored in the worst case.",
                "  - Note that O(|E|) may vary between O(1) and O(|V|^{2}), depending on how sparse the input graph is.",
                "⚙️ [Worse-case] Space complexity: O(|V|)",
                "  - In the case that it is need to hold all vertices in the queue.",
                "",
                "⚙️ In BFS, usually a queue is needed to keep track of the child nodes that were encountered but not yet explored.",
                "  - It could be implemented by by replacing stack with queue in depth_first_search, but that is somewhat nonstandard one.",
            ]

        def __str__(self) -> str:
            return (
                "Non-recursive implementation that visits from left edge to right edge"
            )

        def solve(self) -> None:
            given_node_list: list[str] = list(self.dst.graph)
            explored_node_list: list[Any] = [self.dst.root_node]
            queue_for_child_nodes: Queue[Any] = Queue()
            queue_for_child_nodes.put(self.dst.root_node)

            while not queue_for_child_nodes.empty():
                discovered_node = queue_for_child_nodes.get()

                for connected_edge in self.dst.graph[discovered_node]:
                    if connected_edge not in explored_node_list:
                        # assert connected_edge.parent is discovered_node
                        explored_node_list.append(connected_edge)
                        if connected_edge in given_node_list:
                            queue_for_child_nodes.put(connected_edge)
            self.result = explored_node_list

        def visualize(self) -> None:
            visualize_implicit_tree(self.dst.graph, root_node=self.dst.root_node)
            return super().visualize()

        @classmethod
        def test_case(cls, dst: Optional[GraphDST]) -> None:  # type: ignore
            algorithm = DfsAndBfs.BFS(dst=dst)
            algorithm.solve()
            algorithm.append_line_into_df_in_wrap(["BFS with queue", algorithm.result])
            algorithm.visualize()

    class DFS(ChildAlgorithmVisualization[GraphDST]):
        def __init__(self, /, dst: Optional[GraphDST]) -> None:
            super().__init__(columns=["name", "explored order"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(
                ["-", "-", "|V| + |E|  |  b^d", "|V|"]
            )
            self.result: list[str] = []
            self.big_o_visualization.df_caption = [
                "⚙️ [Worse-case] Time complexity",
                "  - O(|V| + |E|): In the case that explicit graphs is traversed without repetition.",
                "  - O(b^d): In the case of implicit graphs.",
                "⚙️ [Worse-case] Space complexity",
                "  - O(|V| + |E|): In the case that explicit graphs is traversed without repetition.",
                "  - O(longest path length searched) = O(b*d): In the case of implicit graphs without elimination of duplicate nodes.",
                "",
                "⚙️ In DFS, usually a stack is needed to keep track of the nodes discovered so far along a specified branch which helps in backtracking of the graph.",
            ]

        def __str__(self) -> str:
            return (
                "Non-recursive implementation that visits from right edges to left edge"
            )

        def solve(self) -> None:
            explored_node_list: list[Any] = []
            stack_for_backtracking: list[Any] = [self.dst.root_node]

            while stack_for_backtracking:
                discovered_node = stack_for_backtracking.pop()

                if discovered_node not in explored_node_list:
                    explored_node_list.append(discovered_node)
                    self.dst.graph.update(
                        {
                            implicit_new_node: []
                            for implicit_new_node in set(
                                self.dst.graph[discovered_node]
                            ).difference(self.dst.graph)
                        }
                    )
                    stack_for_backtracking.extend(self.dst.graph[discovered_node])
            self.result = explored_node_list

        def visualize(self) -> None:
            visualize_implicit_tree(self.dst.graph, root_node=self.dst.root_node)
            return super().visualize()

        @classmethod
        def test_case(cls, dst: Optional[GraphDST]) -> None:  # type: ignore
            algorithm = DfsAndBfs.DFS(dst=dst)
            algorithm.solve()
            algorithm.append_line_into_df_in_wrap(["DFS with stack", algorithm.result])
            algorithm.visualize()


if __name__ == "__main__" or VisualizationManager.central_control_state:
    VisualizationManager.call_root_classes()

    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [DfsAndBfs]

    VisualizationManager.call_parent_algorithm_classes(
        dst=GraphDST.get_default_implicit_dict(),
        only_class_list=only_class_list,
    )


# if __name__ == "__main__":
#     # python src/python/wbfw109/labs/basics/graphs/search.py -v
#     import doctest

#     doctest.testmod()
