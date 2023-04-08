""": 💻 Complexity Class of the decision problem is P and have strong polynomial time to exact algorithm.
    decision problem: to ask whether a given value exists in a given sequence of elements."""

# %%
from __future__ import annotations

import bisect
import dataclasses
import math
import random
from typing import Any, Optional

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    ChildAlgorithmVisualization,
    MixInParentAlgorithmVisualization,
    VisualizationManager,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


# binary search
@dataclasses.dataclass
class SortedDST:
    target_list: list[int] = dataclasses.field(default_factory=list)

    @staticmethod
    def get_default_sorting_dst(list_len: int = 1000) -> SortedDST:
        return SortedDST(target_list=[i for i in range(list_len)])

    @staticmethod
    def verify_found(look_up_target: int, target_location: Optional[int]) -> bool | Any:
        # check whether target found or not found in a array
        if target_location:
            return f"{look_up_target} Found in {target_location} index"
        else:
            return f"{look_up_target} Not Found"


class SortedListSearch(MixInParentAlgorithmVisualization):
    """: 🍡 (Binary, Fibonacci, Jump, Predictive, Uniform) search"""

    class BinarySearch(ChildAlgorithmVisualization[SortedDST]):
        """📝 see <BitonicSubsequence> sequences/subsequences.py about custom predicate.

        - A collection of improved binary search algorithms. ; https://github.com/scandum/binary_search
        """

        def __init__(self, /, dst: Optional[SortedDST]) -> None:
            super().__init__(columns=["elapsed time", "verification"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(
                ["1", "log n", "log n", "1"]
            )
            self.big_o_visualization.df_caption = [
                "⚙️ Binary search is faster than linear search except for small arrays.",
                "  It is useful  such as finding the next-smallest or next-largest element in the array relative to the target",
                "  , even if it is absent from the array.",
            ]
            self.target_location: Optional[int] = None
            self.look_up_target: int = 0

        def __str__(self) -> str:
            """the variant was first published by Hermann Bottenbruch.
            - to check that whether the middle element is equal to the target equal only would be performed when one element is left (when L=R).
                This results in a faster comparison loop, as one comparison is eliminated per iteration
                , while it requires only one more iteration on average
            - but it seems that if predicate is complex, it is inefficient than original binary search.
            """
            return "Advanced version: Deferred Detection of Equality"

        def solve(self) -> None:
            n: int = len(self.dst.target_list)
            self.look_up_target: int = random.randint(-(n - 1), n - 1)
            left_i: int = 0
            right_i: int = n - 1
            while left_i != right_i:
                middle_i: int = math.ceil((left_i + right_i) / 2)

                if self.dst.target_list[middle_i] > self.look_up_target:
                    right_i = middle_i - 1
                else:
                    left_i = middle_i
                    # <look_up_target> is found in <middle_i>-th of <target_list>.
            else:  # if left_i == right_i
                if self.dst.target_list[left_i] == self.look_up_target:
                    self.target_location = left_i
                    return

            # <look_up_target> is not found in <target_list>.
            return

        def verify(self) -> bool | Any:
            return SortedDST.verify_found(self.look_up_target, self.target_location)

        @classmethod
        def test_case(cls, dst: Optional[SortedDST]) -> None:  # type: ignore
            algorithm = SortedListSearch.BinarySearch(dst=dst)
            algorithm.append_line_into_df_in_wrap(algorithm.measure())
            algorithm.visualize()

    class BinarySearch2(ChildAlgorithmVisualization[SortedDST]):
        """bisect (built-in modules)"""

        def __init__(self, /, dst: Optional[SortedDST]) -> None:
            super().__init__(columns=["eval", "print 1", "print 2"], dst=dst)
            self.target_location: Optional[int] = None
            self.look_up_target: int = 0

        def __str__(self) -> str:
            return "🆚 bisect_left(), bisect_right()"

        def solve(self) -> None:
            a = [0, 1]
            target_list = [-0.5, 0, 0.5, 1, 1.5]
            self.append_line_into_df_in_wrap(
                ["a = [0, 1]", "bisect_left()", "bisect_right()"]
            )
            self.append_line_into_df_in_wrap()
            for x in target_list:
                self.append_line_into_df_in_wrap(
                    [f"(a, {x})", bisect.bisect_left(a, x), bisect.bisect_right(a, x)]
                )

        def verify(self) -> bool | Any:
            return SortedDST.verify_found(self.look_up_target, self.target_location)

        @classmethod
        def test_case(cls, dst: Optional[SortedDST]) -> None:  # type: ignore
            algorithm = SortedListSearch.BinarySearch2(dst=dst)
            algorithm.solve()
            algorithm.visualize()


# Fractional cascading. In particular, fractional cascading speeds up binary searches for the same value in multiple arrays
# Exponential search.

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        VisualizationManager.call_root_classes()
        only_class_list = []
    else:
        only_class_list = [SortedListSearch.BinarySearch2]
    VisualizationManager.call_parent_algorithm_classes(
        dst=SortedDST.get_default_sorting_dst(),
        only_class_list=only_class_list,
    )
