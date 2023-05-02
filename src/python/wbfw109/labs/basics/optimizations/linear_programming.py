""": 🍡 <Covering/packing dualities>"""

# %%
from __future__ import annotations

import dataclasses
import itertools
import random
from typing import Optional

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
# %%


@dataclasses.dataclass
class SetCoverDST:
    n: int
    m: int
    universe: set[str]
    sub_collections: list[set[str]]

    @staticmethod
    def get_default_set_cover_dst() -> SetCoverDST:
        n: int = random.randint(8, 9)
        universe_list: str = "ABCDEFGHI"[:n]
        universe: set[str] = set(universe_list)
        m: int = random.randint(5, n - 2)
        sub_collections: list[set[str]] = []

        # Distribute elements of the universe randomly to sub_collections
        for elem in universe:
            if len(sub_collections) < m:
                sub_collections.append({elem})
            else:
                random.choice(sub_collections).add(elem)

        # Shuffle elements of sub_collections
        for subset in sub_collections:
            new_elements = random.sample(universe_list, random.randint(0, n // 2))
            subset.update(new_elements)

        return SetCoverDST(n, m, universe, sub_collections)


# TODO: Hitting set, Greedy method (Approximation)
class SetCoverProblem(MixInParentAlgorithmVisualization):
    """: 💻 Complexity Class of the decision problem is strong NP-Complete.

    the set cover problem is to identify the smallest sub-collection of S whose union equals the universe.

    Definition
        - n: is size of Universe.
        - collection S of m sets: is such that the union of all the sets in S equals the universe.

    : 🍡 Exact algorithm.

    🛍️ E.g.
        - Sensor placement problem
            Determining the minimum number of sensors required to monitor a specific area.
        - Relay station optimization problem
            Finding the minimum number of relay stations to connect all cities in a communication network.
        - Traffic control problem
            Find the minimum number of traffic lights needed to cover all intersections in a city, where each traffic light can manage a specific set of intersections
    """

    class ExactAlgorithm(ChildAlgorithmVisualization[SetCoverDST]):
        def __init__(self, /, dst: Optional[SetCoverDST]) -> None:
            super().__init__(columns=["var", "print"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(
                ["-", "-", "n * 2^m", "n"]
            )
            self.big_o_visualization.df_caption = [
                "⚙️ [Worse-case] Space complexity",
                "  - n: is taken time to compute the union for each case.",
                "  - 2^m: arises from considering all possible subsets of the given movie sets."
                "⚙️ [Worse-case] Space complexity",
                "  - O(n) union operation",
            ]
            self.min_cover_size: int = 0
            self.found_comb: tuple[set[str], ...] = ()

        def __str__(self) -> str:
            return "-"

        def solve(self) -> None:
            n, m, universe, sub_collections = (
                self.dst.n,
                self.dst.m,
                self.dst.universe,
                self.dst.sub_collections,
            )

            min_cover_size = m + 1
            is_found_min_cover_set = False
            found_comb: tuple[set[str], ...] = ()

            for i in range(1, m + 1):
                if is_found_min_cover_set:
                    break
                for combination in itertools.combinations(sub_collections, i):
                    union: set[str] = set().union(*combination)  # type:ignore
                    if universe <= union:
                        min_cover_size = min(min_cover_size, i)
                        is_found_min_cover_set = True
                        found_comb = combination
                        break

            self.min_cover_size = min_cover_size
            self.found_comb = found_comb

        def visualize(self) -> None:
            self.append_line_into_df_in_wrap(
                ["Given Universe", f"{self.dst.n}: {self.dst.universe}"]
            )
            self.append_line_into_df_in_wrap(
                ["Given Sub-collection", f"{self.dst.m}: {self.dst.sub_collections}"]
            )
            self.append_line_into_df_in_wrap(
                [
                    "Found Minimum combinations",
                    f"{self.min_cover_size}: {self.found_comb}",
                ]
            )
            display_data_frame_with_my_settings(self.df, caption=self.df_caption)
            if not self.big_o_visualization.df.empty:
                display_data_frame_with_my_settings(
                    self.big_o_visualization.df,
                    caption=self.big_o_visualization.df_caption,
                )

        @classmethod
        def test_case(cls, dst: Optional[SetCoverDST]) -> None:  # type: ignore
            algorithm = SetCoverProblem.ExactAlgorithm(dst=dst)
            algorithm.solve()
            algorithm.visualize()


# %%
if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        print(__doc__)
        VisualizationManager.call_parent_algorithm_classes(
            dst=SetCoverDST.get_default_set_cover_dst(),
            only_class_list=[SetCoverProblem],
        )
    else:
        VisualizationManager.call_parent_algorithm_classes(
            dst=SetCoverDST.get_default_set_cover_dst(),
            only_class_list=[SetCoverProblem],
        )
