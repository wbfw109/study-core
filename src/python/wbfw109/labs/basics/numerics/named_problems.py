# %%
from __future__ import annotations

import dataclasses
import operator
import random
from typing import Any, Literal, Optional

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    ChildAlgorithmVisualization,
    MixInParentAlgorithmVisualization,
    VisualizationManager,
    VisualizationRoot,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


class BubbleSort(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )
        self.big_o_visualization = VisualizationRoot(
            columns=["Best", "Average", "Worst", "Memory"],
            has_df_lock=False,
            should_highlight=False,
        )
        self.big_o_visualization.append_line_into_df_in_wrap(["n", "n^2", "n^2", "1"])
        self.big_o_visualization.df_caption = [
            "⚙️ [Worse-case] Time complexity",
            "  - ( O(N^2) comparisons, O(N^2) swaps ) where n is the number of items being sorted",
        ]

    def __str__(self) -> str:
        return "\n".join(
            [
                "Properties: comparison-based, In-place",
                "methods: Exchanging",
            ]
        )

    def solve(self) -> None:
        target_list_len: int = len(self.dst.target_list)
        # Note: Optimization: by using <last_swapped_i>
        while True:
            last_swapped_i = 0
            for i in range(target_list_len - 1):
                # in sublist loop, Swapping all like bubble
                if self.dst.target_list[i] > self.dst.target_list[i + 1]:
                    self.dst.target_list[i], self.dst.target_list[i + 1] = (
                        self.dst.target_list[i + 1],
                        self.dst.target_list[i],
                    )
                    last_swapped_i = i + 1
            if last_swapped_i <= 1:
                break

    def verify(self) -> bool | Any:
        return SortingDST.verify_sorting(self.dst)

    @classmethod
    def test_case(cls, dst: Optional[SortingDST]) -> None:  # type: ignore
        algorithm = ExchangeSorts.BubbleSort(dst=dst)
        algorithm.append_line_into_df_in_wrap(algorithm.measure())
        algorithm.visualize()


#%%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        VisualizationManager.call_root_classes()
        only_class_list = []
    else:
        only_class_list = [MergeSorts.MergeSort]
    VisualizationManager.call_parent_algorithm_classes(
        dst=SortingDST.get_default_sorting_dst(),
        only_class_list=only_class_list,
    )
