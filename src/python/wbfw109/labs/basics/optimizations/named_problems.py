""": 🍡 <Optimization algorithms>"""

# %%
from __future__ import annotations

import bisect
import dataclasses
import heapq
import itertools
import random
import string
from typing import Callable, Iterator, NamedTuple, Optional

import matplotlib.pyplot as plt
import networkx as nx
from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    ChildAlgorithmVisualization,
    MixInParentAlgorithmVisualization,
    VisualizationManager,
    display_data_frame_with_my_settings,
)
from wbfw109.open_source._networkx import my_draw_networkx_edge_labels  # type: ignore

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode
# %%


@dataclasses.dataclass
class SubsetSumDST:
    sequence: list[int] = dataclasses.field(default_factory=list)

    @staticmethod
    def get_default_subset_sum_dst(
        list_len: int = 8, random_range: tuple[int, int] = (-2, 5)
    ) -> SubsetSumDST:
        return SubsetSumDST(
            sequence=[
                random.randint(random_range[0], random_range[1])
                for _ in range(list_len)
            ]
        )


class SubsetSumProblem(MixInParentAlgorithmVisualization):
    """: 💻 Complexity Class of the decision problem is NP-Hard and weak NP-Complete.

    Time complexity can be thought as O(N*(B-A)).
        - Let A be the sum of the negative values and B the sum of the positive values.
        - This algorithm is polynomial in the values of A and B, which are exponential in their numbers of bits.
            so it includes pseudo-polynomial time algorithm.

    : 🍡 Horowitz and Sahni version, Three sum variant.
    """

    class HorowitzSahniVersion(ChildAlgorithmVisualization[SubsetSumDST]):
        """Variant of Horowitz and Sahni.

        - The number of subsets of a set with n elements is 2^n
            , since each element can either be included or excluded from a subset.
        """

        def __init__(self, /, dst: Optional[SubsetSumDST]) -> None:
            super().__init__(columns=["var", "print"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(
                ["-", "-", "n/2 * 2^(n/2)", "2^(n/2)"]
            )
            self.big_o_visualization.df_caption = [
                "⚙️ I used additionally binary search to find target-sum T",
                "  from this operation: ⏲️ 2^(n/2) * log_2( 2^(n/2) )  =  2^(n/2) * (n/2)",
            ]
            self.target_sum = 10
            self.total_count: int = 0

        def __str__(self) -> str:
            return "Variant of Horowitz and Sahni version"

        def solve(self) -> None:
            n: int = len(self.dst.sequence)

            # n halves are subsets of <sequence>.
            n_halves: list[list[int]] = [
                self.dst.sequence[: n // 2],
                self.dst.sequence[n // 2 :],
            ]
            sum_halves: list[list[int]] = [[], []]

            # save sum of possible combinations in <n_half> subset into <sum_half>, and sort.
            for n_half, sum_half in zip(n_halves, sum_halves):
                for i in range(1, len(n_half) + 1):
                    sum_half.extend(
                        (sum(comb) for comb in itertools.combinations(n_half, i))
                    )
                sum_half.sort()

            get_target_count: Callable[
                [list[int], int], int
            ] = lambda in_, target: bisect.bisect_right(
                in_, target
            ) - bisect.bisect_left(
                in_, target
            )

            # find target-sum T from each <n_half> not combined another subset.
            for sum_half in sum_halves:
                self.total_count += get_target_count(sum_half, self.target_sum)
            # find target-sum T from <n_half> combined another subset.
            for s in sum_halves[0]:
                self.total_count += get_target_count(sum_halves[1], self.target_sum - s)

        def visualize(self) -> None:
            self.append_line_into_df_in_wrap(
                ["Given sequence (set)", self.dst.sequence]
            )
            self.append_line_into_df_in_wrap(["Target sum", self.target_sum])
            self.append_line_into_df_in_wrap(["Total count", self.total_count])

            display_data_frame_with_my_settings(self.df, caption=self.df_caption)
            if not self.big_o_visualization.df.empty:
                display_data_frame_with_my_settings(
                    self.big_o_visualization.df,
                    caption=self.big_o_visualization.df_caption,
                )

        @classmethod
        def test_case(cls, dst: Optional[SubsetSumDST]) -> None:  # type: ignore
            algorithm = SubsetSumProblem.HorowitzSahniVersion(dst=dst)
            algorithm.solve()
            algorithm.visualize()

    class ThreeSumVariant(ChildAlgorithmVisualization[SubsetSumDST]):
        """
        Deduction
            O := Optimized value that make Minimum Abs with two control variable existed
                1. the Optimized value is different for each a independent variable.
                2. the Optimized value may exist or not in given Array

            Assume that the value is between consecutive two values in the Array.
            To determine which values are closest to zero
            , It should test both; each sum of (the two control variable, one of consecutive two values)

            if so, even if a pointer (<left_i> | <right_i> moves one by one, it could covers that range.
        """

        def __init__(self, /, dst: Optional[SubsetSumDST]) -> None:
            super().__init__(columns=["var", "print"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(["-", "-", "n^2", "n"])
            self.big_o_visualization.df_caption = [
                "⚙️ [Worse-case] Space complexity",
                "  - O(n) from Tim sort",
            ]
            self.found_list: list[str] = []

        def __str__(self) -> str:
            return "-"

        def solve(self) -> None:
            self.dst.sequence.sort()
            n: int = len(self.dst.sequence)
            for i in range(n - 2):
                j: int = i + 1
                k: int = n - 1

                while j < k:
                    if (
                        temp_sum := self.dst.sequence[i]
                        + self.dst.sequence[j]
                        + self.dst.sequence[k]
                    ) == 0:
                        self.found_list.append(
                            f"{self.dst.sequence[i]},{self.dst.sequence[j]},{self.dst.sequence[k]}"
                        )
                        # We need to update both end and start together since the array values are distinct.
                        k -= 1
                        j += 1

                    elif temp_sum > 0:
                        k -= 1
                    else:
                        j += 1

        def visualize(self) -> None:
            self.append_line_into_df_in_wrap(
                ["Given sequence (set)", self.dst.sequence]
            )
            self.append_line_into_df_in_wrap(["Found combinations", self.found_list])
            display_data_frame_with_my_settings(self.df, caption=self.df_caption)
            if not self.big_o_visualization.df.empty:
                display_data_frame_with_my_settings(
                    self.big_o_visualization.df,
                    caption=self.big_o_visualization.df_caption,
                )

        @classmethod
        def test_case(cls, dst: Optional[SubsetSumDST]) -> None:  # type: ignore
            algorithm = SubsetSumProblem.ThreeSumVariant(dst=dst)
            algorithm.solve()
            algorithm.visualize()


@dataclasses.dataclass
class TravellingSalesmanProblemDST:
    distance_matrix: list[list[int | float]]

    @staticmethod
    def get_default_tsf_dst(cities_count: int = 4) -> TravellingSalesmanProblemDST:
        distance_matrix: list[list[int | float]] = [
            [random.randint(1, 10) for _ in range(cities_count)]
            for _ in range(cities_count)
        ]
        for i in range(cities_count):
            distance_matrix[i][i] = float("inf")
        return TravellingSalesmanProblemDST(distance_matrix)


# TODO: Implementations of branch-and-bound and problem-specific cut generation (branch-and-cut)
class TravellingSalesmanProblem(MixInParentAlgorithmVisualization):
    """: 💻 Complexity Class of the decision problem is NP-Hard and strong NP-Complete.

    : 🍡 Held-Karp Algorithm."""

    # TODO: look again proof of complexity after doing Hamiltonian cycle and Binomial coefficient.
    class HeldKarpAlgorithm(ChildAlgorithmVisualization[TravellingSalesmanProblemDST]):
        """
        Assume that:
            - vertices are Integer equal or greater than 0.
            - starting point is 0.
                since the solution to TSP is a cycle, the choice of starting city doesn't matter.

        So It could uses Mask (bit-mask) for efficiency to Space and Time in according to condition that each Vertices (cities) are 0 or Integer.
            - and operations in this algorithm 0 vertex is not used.

        Definition
            - g(S, v) == <min_cycle_weights[route][vertex]> := minimum cost starting from vertex 0, through <route> and ending at <vertex> (destination).
                if <route> is 0, it means ∅.
            - k := cardinality of chose subset == <route>.bit_count()
        """

        def __init__(self, /, dst: Optional[TravellingSalesmanProblemDST]) -> None:
            super().__init__(columns=["var", "print"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(
                ["-", "Θ(n^2 * 2^n)", "-", "Θ(n * 2^n)"]
            )
            self.big_o_visualization.df_caption = [
                "-",
            ]

            n: int = len(self.dst.distance_matrix)
            # 0 ≤  route range  < <route_limit> - 1
            self.route_limit: int = 1 << n - 1
            self.min_cycle_weights: list[list[int | float]] = [
                [float("infinity")] * n for _ in range(self.route_limit)
            ]

            self.min_cycle_predecessor: list[int] = [-1] * self.route_limit
            self.cycle = [0] * n

            # for visualization
            self.graph: nx.DiGraph = nx.DiGraph()
            for i, row in enumerate(self.dst.distance_matrix):
                for j, x in enumerate(row):
                    if x != float("infinity"):
                        self.graph.add_edge(i, j, weight=x)  # type:ignore

        def __str__(self) -> str:
            return "Properties: Bit-masking"

        def solve(self) -> None:
            n: int = len(self.dst.distance_matrix)

            def set_min_cycle_weights(route: int, dest: int) -> None:
                """📝 Arguments: vertices of <route> and <dest> must not be overlapped.
                - It could cover Directed acyclic cycle in Travelling salesman problem.
                """
                # combinations
                for previous_dest in (
                    v for v in range(1, n) if 1 << v - 1 & route != 0
                ):
                    # <divided_route> can be also 0.
                    divided_route = route & ~(1 << previous_dest - 1)
                    new_distance = (
                        self.min_cycle_weights[divided_route][previous_dest]
                        + self.dst.distance_matrix[previous_dest][dest]
                    )
                    if new_distance < self.min_cycle_weights[route][dest]:
                        self.min_cycle_weights[route][dest] = new_distance
                        self.min_cycle_predecessor[route] = previous_dest

            ## set remained cycle cost
            # ∀i, min_cycle_weights[route_limit - 1][i] not need to be calculated.
            # one min_cycle_weights[route_limit - 1][0] will be calculated later.
            route_dict_by_v_count: dict[int, list[int]] = {
                v_count: [] for v_count in range(1, n - 1)
            }
            # Each key have as many Combinations (n-1, k). and total cases count 2^(n-1)-2.
            for route in range(1, self.route_limit - 1):
                route_dict_by_v_count[route.bit_count()].append(route)

            ## set g(k, v) when (k = ∅)
            for v in range(1, n):
                self.min_cycle_weights[0][v] = self.dst.distance_matrix[0][v]
            ## set g(k, v) when (1 ≤ k ≤ n-2)
            for route in itertools.chain.from_iterable(route_dict_by_v_count.values()):
                # the number of candidates of destination is (n-k-1)
                for dest in (v for v in range(1, n) if 1 << v - 1 & route == 0):
                    # loop of <set_min_cycle_weights> count is k.
                    set_min_cycle_weights(route=route, dest=dest)
            ## set g(k, v) when (k = n-1)
            set_min_cycle_weights(route=self.route_limit - 1, dest=0)

            # find a complete cycle
            route = self.route_limit - 1
            for i in range(n - 1, 0, -1):
                predecessor = self.min_cycle_predecessor[route]
                self.cycle[i] = predecessor
                route = route & ~(1 << predecessor - 1)

        def visualize(self) -> None:
            cycle_edges: list[tuple[int, int]] = list(itertools.pairwise(self.cycle))
            cycle_edges.append((self.cycle[-1], self.cycle[0]))
            # remained_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]

            # positions for all nodes - seed for reproducibility
            pos = nx.drawing.spring_layout(self.graph, seed=7)  # type: ignore
            _, ax = plt.subplots()  # type: ignore

            # draw nodes and node labels
            nx.drawing.draw_networkx_nodes(self.graph, pos, node_size=300, ax=ax)  # type: ignore
            nx.drawing.draw_networkx_labels(self.graph, pos, ax=ax)  # type: ignore

            # draw edges
            curved_edges: list[tuple[int, int]] = [edge for edge in self.graph.edges() if reversed(edge) in self.graph.edges()]  # type: ignore
            straight_edges: list[tuple[int, int]] = list(
                set(self.graph.edges()).difference(curved_edges)  # type:ignore
            )
            nx.drawing.draw_networkx_edges(  # type:ignore
                self.graph, pos, ax=ax, edgelist=straight_edges
            )
            arc_rad = 0.25
            nx.drawing.draw_networkx_edges(  # type:ignore
                self.graph,
                pos,
                ax=ax,
                edgelist=curved_edges,
                connectionstyle=f"arc3,rad={arc_rad}",
            )

            nx.drawing.draw_networkx_edges(  # type: ignore
                self.graph,
                pos,
                ax=ax,
                edgelist=cycle_edges,
                width=2,
                alpha=0.5,
                edge_color="b",
                style="dashed",
                connectionstyle=f"arc3,rad={arc_rad}",
            )

            # draw edges labels
            edge_weights: dict[tuple[int, int], int] = nx.function.get_edge_attributes(  # type: ignore
                self.graph, "weight"
            )
            curved_edge_labels: dict[tuple[int, int], int] = {
                edge: edge_weights[edge] for edge in curved_edges
            }
            straight_edge_labels: dict[tuple[int, int], int] = {
                edge: edge_weights[edge] for edge in straight_edges
            }
            my_draw_networkx_edge_labels(
                self.graph,
                pos,
                ax=ax,
                edge_labels=curved_edge_labels,
                rotate=False,
                rad=arc_rad,
            )
            nx.drawing.draw_networkx_edge_labels(  # type:ignore
                self.graph, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False
            )

            # Set margins for the axes so that nodes aren't clipped
            ax: plt.Axes = plt.gca()  # type: ignore
            ax.margins(0.08)  # type: ignore
            plt.axis("off")  # type: ignore
            plt.show()  # type: ignore
            # plt.tight_layout()  # type: ignore
            plt.close()  # type: ignore

            self.append_line_into_df_in_wrap(["edge weights", self.dst.distance_matrix])
            self.append_line_into_df_in_wrap(
                ["answer", self.min_cycle_weights[self.route_limit - 1][0]]
            )
            self.append_line_into_df_in_wrap(["cycle", self.cycle])
            display_data_frame_with_my_settings(self.df, caption=self.df_caption)
            if not self.big_o_visualization.df.empty:
                display_data_frame_with_my_settings(
                    self.big_o_visualization.df,
                    caption=self.big_o_visualization.df_caption,
                )

        @classmethod
        def test_case(cls, dst: Optional[TravellingSalesmanProblemDST]) -> None:  # type: ignore
            algorithm = TravellingSalesmanProblem.HeldKarpAlgorithm(dst=dst)
            algorithm.solve()
            algorithm.visualize()


@dataclasses.dataclass
class KnapsackProblemDST:
    none: None

    @staticmethod
    def get_default_knapsack_dst() -> KnapsackProblemDST:
        return KnapsackProblemDST(None)


class Item(NamedTuple):
    weight: int
    value: int


class Knapsack(NamedTuple):
    allowance: int


class KnapsackProblem(MixInParentAlgorithmVisualization):
    """: 💻 Complexity Class of the decision problem is NP-Hard and weak NP-Complete.

    : 🍡 0-1 multiple knapsack variant 1."""

    class ZeroOneMultipleKnapsackVariant1(
        ChildAlgorithmVisualization[KnapsackProblemDST]
    ):
        """
        Time Complexity (Worst-case): O(n(log n) + O(k log k))
            - O(n(log n)) + O(k(log k)) from Tim sort
                n is the number of jewels, k is the number of bags.
            - O(k) from bag loop  *
                ( O(1) comparison from Jewel consumed iteration  +  O(log k) from Hip (pop | push) )

        Space Complexity (Worst-case): O(n) + O(k) from Tim sort

        Definition
            - n: the number of jewels.
            - k: the number of bags.
        """

        def __init__(self, /, dst: Optional[KnapsackProblemDST]) -> None:
            super().__init__(columns=["var", "print"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(
                ["-", "-", "n*(log n) + k*(log k) ", "O(n) + O(k)"]
            )
            self.big_o_visualization.df_caption = [
                "-",
            ]
            items_name_iter: Iterator[str] = iter(string.ascii_lowercase)
            knapsacks_name_iter: Iterator[str] = iter(string.ascii_uppercase)
            self.items: dict[str, Item] = {
                next(items_name_iter): Item(
                    weight=random.randint(1, 9), value=random.randint(1, 9)
                )
                for _ in range(random.randint(5, 7))
            }
            self.knapsacks: dict[str, Knapsack] = {
                next(knapsacks_name_iter): Knapsack(allowance=random.randint(1, 9))
                for _ in range(random.randint(5, 7))
            }
            self.maximum_total_value: int = 0
            self.pairs: list[tuple[str, str]] = []

        def __str__(self) -> str:
            return "Limited: volumes of each item and knapsack are 1."

        def solve(self) -> None:
            # when sort <items>, item's value key is not important
            # , since <items> will  be checked as possible according to <knapsacks>' allowance
            sorted_knapsacks: list[tuple[str, Knapsack]] = sorted(
                self.knapsacks.items(), key=lambda kv: kv[1].allowance
            )
            sorted_items: list[tuple[str, Item]] = sorted(
                self.items.items(), key=lambda kv: -kv[1].weight
            )

            self.append_line_into_df_in_wrap(
                ["knapsacks", [(x[0], x[1].allowance) for x in sorted_knapsacks]]
            )
            self.append_line_into_df_in_wrap(
                ["items' weight", [(x[0], x[1].weight) for x in sorted_items]]
            )
            self.append_line_into_df_in_wrap(
                ["items' value", [(x[0], x[1].value) for x in sorted_items]]
            )

            checked_items_heapq: list[tuple[int, str]] = []
            for knapsack in sorted_knapsacks:
                while (
                    sorted_items and knapsack[1].allowance >= sorted_items[-1][1].weight
                ):
                    heapq.heappush(
                        checked_items_heapq,
                        (-sorted_items[-1][1].value, sorted_items[-1][0]),
                    )
                    sorted_items.pop()
                if checked_items_heapq:
                    target_item = heapq.heappop(checked_items_heapq)
                    self.pairs.append((knapsack[0], target_item[1]))
                    self.maximum_total_value += -target_item[0]

        def visualize(self) -> None:
            self.append_line_into_df_in_wrap(
                ["maximum_total_value", self.maximum_total_value]
            )
            self.append_line_into_df_in_wrap(["pairs", self.pairs])
            display_data_frame_with_my_settings(self.df, caption=self.df_caption)
            if not self.big_o_visualization.df.empty:
                display_data_frame_with_my_settings(
                    self.big_o_visualization.df,
                    caption=self.big_o_visualization.df_caption,
                )

        @classmethod
        def test_case(cls, dst: Optional[KnapsackProblemDST]) -> None:  # type: ignore
            algorithm = KnapsackProblem.ZeroOneMultipleKnapsackVariant1(dst=dst)
            algorithm.solve()
            algorithm.visualize()


# %%
if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        print(__doc__)
        VisualizationManager.call_parent_algorithm_classes(
            dst=SubsetSumDST.get_default_subset_sum_dst(),
            only_class_list=[SubsetSumProblem],
        )
        VisualizationManager.call_parent_algorithm_classes(
            dst=TravellingSalesmanProblemDST.get_default_tsf_dst(),
            only_class_list=[TravellingSalesmanProblem],
        )
        VisualizationManager.call_parent_algorithm_classes(
            dst=KnapsackProblemDST.get_default_knapsack_dst(),
            only_class_list=[KnapsackProblem],
        )
    else:
        VisualizationManager.call_parent_algorithm_classes(
            dst=SubsetSumDST.get_default_subset_sum_dst(),
            only_class_list=[SubsetSumProblem],
        )
# Knapsack problem
