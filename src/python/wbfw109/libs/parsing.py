"""Parsing syntax, tree

In this file context,
- "Explicit" means that node level and parent node is specified.
- "Implicit" means that some node may be missing and level and parent node is not specified.
"""
from __future__ import annotations

import collections
import itertools
from collections.abc import Mapping
from typing import MutableMapping, TypeAlias, TypedDict, TypeVar, Union

from wbfw109.libs.typing import RecursiveTuple, SingleLinkedList, deprecated  # type: ignore

# Syntax tree that may be recursive
SyntaxTree: TypeAlias = MutableMapping[str, Union["SyntaxTree", str]]


def convert_implicit_syntax_node_dict_to_tree(
    implicit_syntax_node_dict: dict[str, list[str]],
    /,
    *,
    root_node: str,
) -> SyntaxTree:

    """
    TODO: currently only support for string node name

    Args:
        implicit_syntax_node_dict (dict[str, list[str]]): dict[<node_name>, list[<connected_node_name>]]
        root_node (str): .

    ---
    Implementation:
        - When construct <SyntaxTree>, Top-up implementation was used.
            It uses data structure: (deque, single linked list which indicates parent syntax tree).
            It will process deques by node level in similar way to Breadth First Search algorithm.
        - SingleLinkedList: [node: str, target: SyntaxTree]

    """
    syntax_tree: SyntaxTree = {}
    explored_node_name_list: list[str] = []
    processing_node_deque: collections.deque[
        SingleLinkedList[str, SyntaxTree]
    ] = collections.deque([SingleLinkedList(node=root_node, target=syntax_tree)])
    waiting_child_node_deque: collections.deque[
        SingleLinkedList[str, SyntaxTree]
    ] = collections.deque()
    given_node_list: list[str] = list(implicit_syntax_node_dict)

    # construct syntax tree
    while processing_node_deque:
        discovered_node = processing_node_deque.popleft()
        explored_node_name_list.append(discovered_node.node)
        discovered_node.target[discovered_node.node] = {}
        current_syntax_tree = discovered_node.target

        # if a connected_node exists but not given in <implicit_syntax_node_dict>, pass traverse (impossible).
        # however even if that case, it must be added to syntax tree.
        for node_name in implicit_syntax_node_dict[discovered_node.node]:
            if node_name in explored_node_name_list:
                continue
            if node_name in given_node_list:
                waiting_child_node_deque.append(
                    SingleLinkedList(  # type: ignore
                        node=node_name, target=current_syntax_tree[discovered_node.node]
                    )
                )
            else:
                current_syntax_tree[discovered_node.node][node_name] = {}  # type: ignore

        # process by node level
        if not processing_node_deque:
            processing_node_deque.extend(waiting_child_node_deque)
            waiting_child_node_deque.clear()

    return syntax_tree


def convert_syntax_tree_to_svgling_style(syntax_tree: SyntaxTree, /) -> RecursiveTuple:
    """Returned value can be used in as following argument structure (but must be used with prefix "*" (asterisk; unpack)).
    ```
    import svgling
    svgling.draw_tree(
        (
            "S",
            ("NP", ("D", "the"), ("N", "elephant")),
            ("VP", ("V", "saw"), ("NP", ("D", "the"), ("N", "rhinoceros"))),
        )
    )
    ```

    ---
    Example usage:
    ```
    graph_example = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "G", "H"],
        "D": ["B"],
        "E": ["B", "F"],
        "F": ["E"],
        "G": ["C"],
        "H": ["C"],
    }

    svgling.draw_tree(
        *convert_syntax_tree_to_svgling_style(
            convert_implicit_syntax_node_dict_to_tree(graph_example, root_node="A")
        )
    )
    ```

    ---
    Complexity:
        - Space complexity: O(n); the number of dict node.

    """
    syntax_tree_as_tuple_list: list[RecursiveTuple | str] = []
    for k, sub_syntax_tree in syntax_tree.items():
        # if leaf node whose have empty dict
        if not sub_syntax_tree:
            syntax_tree_as_tuple_list.append(k)
        elif isinstance(sub_syntax_tree, Mapping) and (
            sub_tuple := convert_syntax_tree_to_svgling_style(sub_syntax_tree)
        ):
            syntax_tree_as_tuple_list.append((k, *sub_tuple))
    return tuple(syntax_tree_as_tuple_list)


@deprecated
class ExplicitNode(TypedDict):
    level: int
    name: str


@deprecated
class ExplicitSyntaxNode(TypedDict):
    """
    Attributes
        node (ExplicitNode): TypedDict("ExplicitNode", {"level": int, "name": str})
        parent_node (ExplicitNode): TypedDict("ExplicitNode", {"level": int, "name": str})
    """

    node: ExplicitNode
    parent_node: ExplicitNode

@deprecated
class ExplicitSimpleSyntaxNode(ExplicitSyntaxNode):
    """
    Attributes
        node (ExplicitNode): TypedDict("ExplicitNode", {"level": int, "name": str})
        parent_node (ExplicitNode): TypedDict("ExplicitNode", {"level": int, "name": str})
        description list[str]: description by line.
    """

    description: list[str]

ExplicitSyntaxNodeT_co = TypeVar(
    "ExplicitSyntaxNodeT_co",
    bound=ExplicitSyntaxNode,
    covariant=True,
)


@deprecated
def get_explicit_syntax_node_group_by_level(
    explicit_syntax_node_t_list: list[ExplicitSyntaxNodeT_co],
) -> dict[int, list[ExplicitSyntaxNodeT_co]]:
    """It is useful when compare and search node by level."""
    node_group_by_level: dict[int, list[ExplicitSyntaxNodeT_co]] = {
        node_level: []
        for node_level in {x["node"]["level"] for x in explicit_syntax_node_t_list}
    }
    for node, g in itertools.groupby(
        explicit_syntax_node_t_list, key=lambda x: x["node"]
    ):
        node_group_by_level[node["level"]].extend(list(g))
    return node_group_by_level

@deprecated
def convert_explicit_syntax_node_list_to_tree(
    explicit_syntax_node_t_list: list[ExplicitSyntaxNodeT_co],
) -> SyntaxTree:
    """It assumes that:
        - There is a difference in 1 level between the node and its parent node.
        - No circular tree.
        - Parent node of a node must exists except for root node.
            📝 By doing so, same node name can exists simultaneously if parent node is different.
            But the root node does not have to exist in argument <explicit_syntax_node_t_list>)
            , because it automatically search from parent node of a node.

    ---
    Args:
        explicit_syntax_node_t_list (list[ExplicitSyntaxNodeT_co]): _description_

    Returns:
        SyntaxTree: TypeAlias = dict[str, Union["SyntaxTree", str]]

    ---
    Implementation
        - When construct <SyntaxTree>, Bottom-up implementation was used.
        - It uses indexing from method <get_explicit_syntax_node_group_by_level>, so tree search is enough fast.
    """
    node_group_by_level = get_explicit_syntax_node_group_by_level(
        explicit_syntax_node_t_list
    )

    # It uses stack data structure to construct SyntaxTree
    node_level_list: list[int] = sorted(node_group_by_level)
    previous_syntax_tree: SyntaxTree = {}
    while node_level_list and (current_node_level := node_level_list.pop()) >= 1:
        parent_node_name_list: list[str] = []
        current_node_name_list: list[str] = []
        for explicit_node in node_group_by_level[current_node_level]:
            parent_node_name_list.append(explicit_node["parent_node"]["name"])
            current_node_name_list.append(explicit_node["node"]["name"])
        parent_node_name_set: set[str] = set(parent_node_name_list)
        current_node_name_set: set[str] = set(current_node_name_list)

        # top-down implementation
        current_syntax_tree: SyntaxTree = {
            parent_node_name: {} for parent_node_name in parent_node_name_set
        }
        for explicit_node in node_group_by_level[current_node_level]:
            # connect <previous_syntax_tree> to <current_syntax_tree> by using default values from <dict.get>.
            current_syntax_tree[explicit_node["parent_node"]["name"]] = {
                explicit_node["node"]["name"]: previous_syntax_tree.get(  # type: ignore
                    explicit_node["node"]["name"], {}
                )
            }

        if previous_syntax_tree:
            # initialize new nodes that not have descendant nodes.
            current_syntax_tree.update(
                {x: {} for x in current_node_name_set.difference(previous_syntax_tree)}
            )
        previous_syntax_tree = current_syntax_tree
    return previous_syntax_tree
