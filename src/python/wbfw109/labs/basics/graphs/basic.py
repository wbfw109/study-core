# %%
from __future__ import annotations

import dataclasses
import random
from collections import deque

import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    VisualizationManager,
    VisualizationRoot,
    display_data_frame_with_my_settings,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


@dataclasses.dataclass
class VerticesAndDistance:
    vertices: list[int]
    distance: int


class Glossary(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["context", "word", "meaning"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "📖 Glossary for graph"

    @classmethod
    def test_case(cls):
        glossary: Glossary = cls()
        glossary.append_line_into_df_in_wrap(
            [
                "Distance",
                "Graph metric",
                "A metric space defined over a set of points in terms of distances in a graph defined over the set, which metric space is formed if and only if the graph is connected.",
            ]
        )

        glossary.append_line_into_df_in_wrap(
            [
                "Graph metric",
                "eccentricity | ε(v)",
                "the greatest distance between v and any other vertex",
            ]
        )
        glossary.append_line_into_df_in_wrap(
            [
                "Graph metric",
                "radius | r",
                "the minimum eccentricity of any vertex",
            ]
        )
        glossary.append_line_into_df_in_wrap(
            [
                "Graph metric",
                "diameter | d",
                "the maximum eccentricity of any vertex",
            ]
        )
        glossary.append_line_into_df_in_wrap(
            [
                "Graph metric",
                "central vertex",
                "a vertex whose eccentricity is r",
            ]
        )
        glossary.append_line_into_df_in_wrap(
            [
                "Graph metric",
                "peripheral vertex",
                "a vertex whose eccentricity is d",
            ]
        )
        glossary.append_line_into_df_in_wrap(
            [
                "Graph metric",
                "pseudo-peripheral vertex",
                "vertices ε(v1) == ε(v2)",
            ]
        )
        glossary.append_line_into_df_in_wrap(
            [
                "Graph",
                "Geodetic graph",
                "a undirected graph such that there exists a unique (unweighted) shortest path between each two vertices.",
            ]
        )
        glossary.visualize()


class GraphDistance(VisualizationRoot):
    """
    ❔ The reason why the maximum number of steps required to find a pseudo-peripheral vertex in a tree is twice the height of the tree.
        1. From any <v1> node selected for the first time, find the node <v2> with the maximum eccentricity.
            This node will be a tip node. (root or leaves)
            This means that d(<v1>, <v2>) are included in the path of obtaining the maximum eccentricity of <v2>.

        2. Assume that vertex with maximum eccentricity with respect to <v2> is <v3>.
            That is, sum of (<v1>, <v2>) and d(<v1>, <v3>) is tree's diameter.

        It can apply in unweighted as well as weighted trees.
    """

    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["var", "print"],
            has_df_lock=False,
            should_highlight=True,
        )
        self.n = 9
        edges: list[tuple[int, int]] = [
            (0, 1),
            (1, 2),
            (1, 7),
            (1, 8),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (5, 7),
            (5, 8),
        ]
        self.graph: nx.Graph = nx.Graph()
        self.edges: list[dict[int, int]] = [{} for _ in range(self.n)]
        for edge in edges:
            self.edges[edge[0]][edge[1]] = 1
            self.edges[edge[1]][edge[0]] = 1
        self.graph.add_edges_from(edges)  # type: ignore
        self.df_caption = [
            "⚙️ Colored nodes are Pseudo-peripheral vertices",
            "  - Blue: Distance 3.",
            "  - Yellow and Red: Distance 4 (Diameter) and Peripheral vertices.",
            "⚙️ All trees are geodetic so that Pseudo-peripheral vertices in tree can be considered as Peripheral vertices.",
        ]

    def __str__(self) -> str:
        return "-"

    def find_farthest_v_with_minimal_degree(self, v1: int) -> VerticesAndDistance:
        p: int = 0
        distance: int = 1
        explored_deques: list[deque[int]] = [deque([v1]), deque()]
        trace_map: list[bool] = [False] * self.n
        trace_map[v1] = True
        tip_vertices_by_degree: dict[int, list[int]] = {}
        while explored_deques[p]:
            explored_v: int = explored_deques[p].popleft()

            is_tip: bool = True
            for v2 in self.edges[explored_v]:
                if not trace_map[v2]:
                    is_tip = False
                    trace_map[v2] = True
                    explored_deques[p ^ 1].append(v2)
            if is_tip:
                if (degree := len(self.edges[explored_v])) in tip_vertices_by_degree:
                    tip_vertices_by_degree[degree].append(explored_v)
                else:
                    tip_vertices_by_degree[degree] = [explored_v]

            if len(explored_deques[p]) == 0:
                if len(explored_deques[p ^ 1]) != 0:
                    # It only need to farthest some v2; ε(v1)
                    tip_vertices_by_degree.clear()

                    distance += 1
                    p ^= 1
        else:
            # distance was defined initially as 1. the next line is required post-process to do that.
            distance -= 1
        min_degree: int = min(tip_vertices_by_degree)

        return VerticesAndDistance(
            vertices=tip_vertices_by_degree[min_degree], distance=distance
        )

    def solve(self) -> None:
        """Algorithm for finding pseudo-peripheral vertices
        - condition: the number of vertices ≥ 2"""
        # 1. Choose a random vertex
        v1: int = random.randint(0, self.n - 1)

        # 2. BFS that finds vertices that are as far from <v1> as possible, let <v2> be one with minimal degree.
        found_v_and_d_from_v1 = self.find_farthest_v_with_minimal_degree(v1)
        self.append_line_into_df_in_wrap(["randomly selected initial v", v1])

        # If ε(v2) > ε(v1) then set <v1> = <v2> and repeat with step 2, else <v1> is a pseudo-peripheral vertex.
        while True:
            v2: int = random.choice(found_v_and_d_from_v1.vertices)
            found_v_and_d_from_v2 = self.find_farthest_v_with_minimal_degree(v2)
            # if ε(v2) > ε(v1):
            if found_v_and_d_from_v2.distance > found_v_and_d_from_v1.distance:
                v1 = v2
                found_v_and_d_from_v1 = found_v_and_d_from_v2
            else:  # if ε(v2) == ε(v1):
                break

        self.append_line_into_df_in_wrap(["found Pseudo-peripheral vertices", (v1, v2)])
        self.append_line_into_df_in_wrap(
            ["Pseudo-peripheral v's distance", found_v_and_d_from_v2.distance]
        )

    def visualize(self):
        # explicitly set positions
        pos: dict[int, tuple[int, int]] = {
            0: (0, 0),
            1: (1, 0),
            2: (2, 0),
            3: (3, 0),
            4: (4, 0),
            5: (5, 0),
            6: (6, 0),
            7: (3, 10),
            8: (3, -10),
        }
        #
        options = {
            "font_size": 18,
            "node_size": 1000,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 5,
            "width": 5,
        }
        nx.drawing.draw_networkx(self.graph, pos, **options)  # type: ignore
        nx.drawing.draw_networkx_nodes(  # type: ignore
            self.graph, pos, node_size=500, nodelist=[2, 4], node_color="tab:orange"
        )
        nx.drawing.draw_networkx_nodes(  # type: ignore
            self.graph, pos, node_size=500, nodelist=[3, 7, 8], node_color="tab:blue"
        )
        nx.drawing.draw_networkx_nodes(  # type: ignore
            self.graph, pos, node_size=500, nodelist=[0, 6], node_color="tab:red"
        )

        # Set margins for the axes so that nodes aren't clipped
        ax: plt.Axes = plt.gca()  # type: ignore
        ax.margins(0.20)  # type: ignore
        plt.axis("off")  # type: ignore
        plt.show()  # type: ignore
        plt.close()  # type: ignore
        display_data_frame_with_my_settings(self.df, caption=self.df_caption)

    @classmethod
    def test_case(cls) -> None:  # type: ignore
        algorithm = GraphDistance()
        algorithm.solve()
        algorithm.visualize()


# %%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        VisualizationManager.call_root_classes(
            only_class_list=[Glossary, GraphDistance]
        )
    else:
        VisualizationManager.call_root_classes(only_class_list=[GraphDistance])
