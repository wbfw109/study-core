# %%
from __future__ import annotations

import dataclasses
import random
from typing import Any, Generator, Generic, Optional, TypeVar

import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    ChildAlgorithmVisualization,
    MixInParentAlgorithmVisualization,
    VisualizationManager,
    display_data_frame_with_my_settings,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


T = TypeVar("T")


@dataclasses.dataclass
class Vertex(Generic[T]):
    """In-tree for Disjoint-set data structure"""

    v: T
    parent: Vertex[T] = dataclasses.field(init=False)
    rank: int = dataclasses.field(default=0)

    def __post_init__(self) -> None:
        self.parent = self


@dataclasses.dataclass
class Edge(Generic[T]):
    v1: Vertex[T]
    v2: Vertex[T]
    weight: int = dataclasses.field(default=0)


@dataclasses.dataclass
class RawEdge(Generic[T]):
    v1: T
    v2: T
    weight: int = dataclasses.field(default=0)


@dataclasses.dataclass
class TreeDST(Generic[T]):
    raw_vertices_list: list[T] = dataclasses.field(default_factory=list)
    raw_edge_list: list[RawEdge[T]] = dataclasses.field(default_factory=list)
    vertex_dict: dict[T, Vertex[T]] = dataclasses.field(default_factory=dict)
    edge_list: list[Edge[T]] = dataclasses.field(default_factory=list)
    graph: nx.Graph = nx.Graph()

    def __post_init__(self) -> None:
        self.vertex_dict: dict[T, Vertex[T]] = {
            v: Vertex(v) for v in self.raw_vertices_list
        }

        if self.raw_edge_list:
            self.edge_list.extend(
                [
                    Edge(
                        v1=self.vertex_dict[raw_edge.v1],
                        v2=self.vertex_dict[raw_edge.v2],
                        weight=raw_edge.weight,
                    )
                    for raw_edge in self.raw_edge_list
                ]
            )
            self.graph.add_edges_from([(edge.v1.v, edge.v2.v, {"weight": edge.weight}) for edge in self.edge_list])  # type: ignore

    @staticmethod
    def get_default_tree_dst() -> TreeDST[int]:
        weights: Generator[int, None, None] = (random.randint(1, 5) for _ in range(9))
        return TreeDST(
            raw_vertices_list=list(range(6)),
            raw_edge_list=[
                RawEdge(0, 1, weight=next(weights)),
                RawEdge(0, 2, weight=next(weights)),
                RawEdge(0, random.choice([3, 5]), weight=next(weights)),
                RawEdge(1, 2, weight=next(weights)),
                RawEdge(2, 3, weight=next(weights)),
                RawEdge(2, 4, weight=next(weights)),
                RawEdge(2, 5, weight=next(weights)),
                RawEdge(3, 4, weight=next(weights)),
                RawEdge(4, 5, weight=next(weights)),
            ],
        )


class MinimumSpanningTree(MixInParentAlgorithmVisualization):
    """: 💻 Complexity Class of the decision problem is P and have strong polynomial time to exact algorithm.

    : 🍡 Borůvka, Kruskal, Prim, Reverse-delete"""

    class Kruskal(ChildAlgorithmVisualization[TreeDST[T]]):
        """It assumes that input vertices are already disjoint set."""

        def __init__(self, /, dst: Optional[TreeDST[T]]) -> None:
            super().__init__(columns=["elapsed time", "verification"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(
                ["-", "-", "E*log(E)", "|V| + |E|"]
            )
            self.big_o_visualization.df_caption = [
                "⚙️ about Time Complexity",
                "  - O( E*log(E) ) from Timsort, O(|E|) from iterating edges",
                "  - O( α(n) ) from Union-Find data structure (Inversed Ackermann function)",
                "⚙️ about Space Complexity",
                "  - O (|V|) from Union-Find data structure  +  O(|E|) in order to iterate all edges",
            ]
            # make_set() of Union-find operation is done in TreeDST.
            self.minimum_spanning_tree: list[Edge[T]] = []

        def __str__(self) -> str:
            return "\n".join(
                [
                    "methods: Union-Find",
                ]
            )

        def add_edges(self, raw_edges: list[RawEdge[T]]) -> None:
            """<raw_edges>: list[tuple[node, node, weight]]"""
            self.dst.edge_list.extend(
                [
                    Edge(
                        v1=self.dst.vertex_dict[raw_edge.v1],
                        v2=self.dst.vertex_dict[raw_edge.v2],
                        weight=raw_edge.weight,
                    )
                    for raw_edge in raw_edges
                ]
            )
            self.dst.graph.add_edges_from([(edge.v1.v, edge.v2.v, {"weight": edge.weight}) for edge in self.dst.edge_list])  # type: ignore

        # find() of Union-find functions while updating pointers (used way: Path Halving using Pointer jumping)
        def find_root(self, node: Vertex[T]) -> Vertex[T]:
            while node.parent != node:
                node.parent = node.parent.parent
                node = node.parent
            return node

        # union() of Union-find functions
        def merge_two_sets(self, v1: Vertex[T], v2: Vertex[T]) -> None:
            x = self.find_root(v1)
            y = self.find_root(v2)

            # x, y are already in the same set.
            if x == y:
                return
            # ensure that x has rank at least as large as that of y.
            if x.rank < y.rank:
                (x, y) = (y, x)

            # make x the new root.
            y.parent = x
            if x.rank == y.rank:
                x.rank += 1

        def solve(self) -> None:
            # Path will be saved in <self.minimum_spanning_tree>
            self.dst.edge_list.sort(key=lambda e: e.weight)
            could_connect: bool = False

            for edge in self.dst.edge_list:
                if self.find_root(edge.v1) != self.find_root(edge.v2):
                    self.minimum_spanning_tree.append(edge)
                    self.merge_two_sets(edge.v1, edge.v2)
                    if len(self.minimum_spanning_tree) == len(self.dst.vertex_dict) - 1:
                        could_connect = True
                        break

            if not could_connect:
                print("❕ This tree could not made up minimum spanning tree.")

        def visualize(self):
            edges_large = [
                (edge.v1.v, edge.v2.v) for edge in self.minimum_spanning_tree
            ]
            edges_small = {
                (edge.v1.v, edge.v2.v) for edge in self.dst.edge_list
            }.difference(edges_large)

            pos: dict[Any, Any] = nx.drawing.spring_layout(  # type: ignore
                self.dst.graph, seed=7
            )  # positions for all nodes - seed for reproducibility

            # nodes
            nx.drawing.draw_networkx_nodes(self.dst.graph, pos, node_size=700)  # type: ignore
            # edges
            nx.drawing.draw_networkx_edges(self.dst.graph, pos, edgelist=edges_large, width=6)  # type: ignore
            nx.drawing.draw_networkx_edges(  # type: ignore
                self.dst.graph,
                pos,
                edgelist=edges_small,
                width=6,
                alpha=0.2,
                edge_color="b",
                style="dashed",
            )

            # node labels
            nx.drawing.draw_networkx_labels(  # type: ignore
                self.dst.graph, pos, font_size=20, font_family="sans-serif"
            )
            # edge weight labels
            edge_labels: dict[Any, Any] = nx.function.get_edge_attributes(self.dst.graph, "weight")  # type: ignore
            nx.drawing.draw_networkx_edge_labels(self.dst.graph, pos, edge_labels)  # type: ignore

            ax: plt.Axes = plt.gca()  # type: ignore
            ax.margins(0.08)  # type: ignore
            plt.axis("off")  # type: ignore
            plt.tight_layout()  # type: ignore

            display_data_frame_with_my_settings(
                self.big_o_visualization.df, caption=self.big_o_visualization.df_caption
            )
            plt.show()  # type: ignore
            plt.close()  # type: ignore

        @classmethod
        def test_case(cls, dst: Optional[TreeDST[T]]) -> None:  # type: ignore
            algorithm = MinimumSpanningTree.Kruskal(dst=dst)  # type: ignore
            algorithm.append_line_into_df_in_wrap(algorithm.measure())
            algorithm.visualize()


# %%
if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [MinimumSpanningTree.Kruskal]  # type: ignore
    VisualizationManager.call_parent_algorithm_classes(
        dst=TreeDST.get_default_tree_dst(),
        only_class_list=only_class_list,  # type: ignore
    )


# Dijkstra's algorithm
# Prim’s algorithm runs faster in dense graphs.	Kruskal’s algorithm runs faster in sparse graphs.
