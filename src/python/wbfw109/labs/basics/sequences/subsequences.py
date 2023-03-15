# %%
from __future__ import annotations

import random
import time
import unittest
from typing import Iterator, Optional

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    VisualizationManager,
    VisualizationRoot,
    display_data_frame_with_my_settings,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


class LongestIncreasingSubsequence(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["var", "eval", "print"],
            has_df_lock=False,
            should_highlight=True,
        )
        self.big_o_visualization = VisualizationRoot(
            columns=["Best", "Average", "Worst", "Memory"],
            has_df_lock=False,
            should_highlight=False,
        )
        self.big_o_visualization.append_line_into_df_in_wrap(
            ["-", "-", "n*log(n)", "n"]
        )
        self.big_o_visualization.df_caption = [
            "⚙️ [Worse-case] Space complexity",
            "  - O(n) from Tim sort",
        ]
        self.target_list = [random.randint(-10, 10) for _ in range(10)]
        self.found_list: list[str] = []

    def __str__(self) -> str:
        return "-"

    def solve(self) -> None:
        self.target_list.sort()
        n: int = len(self.target_list)
        for i in range(n - 2):
            j: int = i + 1
            k: int = n - 1

            while j < k:
                if (
                    temp_sum := self.target_list[i]
                    + self.target_list[j]
                    + self.target_list[k]
                ) == 0:
                    self.found_list.append(
                        f"{self.target_list[i]},{self.target_list[j]},{self.target_list[k]}"
                    )
                    # We need to update both end and start together since the array values are distinct.
                    k -= 1
                    j += 1

                elif temp_sum > 0:
                    k -= 1
                else:
                    j += 1

    def visualize(self) -> None:
        display_data_frame_with_my_settings(self.df, caption=self.df_caption)
        if not self.big_o_visualization.df.empty:
            display_data_frame_with_my_settings(
                self.big_o_visualization.df, caption=self.big_o_visualization.df_caption
            )

    @classmethod
    def test_case(cls) -> None:  # type: ignore
        algorithm = LongestIncreasingSubsequence()
        algorithm.solve()
        # Array range is -100 ~ 100, array size is 100.
        algorithm.append_line_into_df_in_wrap(
            ["Target list", "[random.randint(-10, 10) for _ in range(10)]"]
        )
        algorithm.append_line_into_df_in_wrap(
            ["Found combinations", "", algorithm.found_list]
        )
        algorithm.visualize()


if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [LongestIncreasingSubsequence]
    VisualizationManager.call_root_classes(only_class_list)


#%%
import random

p: list[int] = [random.randint(0, 20) for _ in range(20)]
p_len: int = len(p)
m: list[int] = [-1]  # undefined so can be set to any value
