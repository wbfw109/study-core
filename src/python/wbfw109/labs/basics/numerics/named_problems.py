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


class ThreeSum(VisualizationRoot):
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
        self.big_o_visualization.append_line_into_df_in_wrap(["-", "-", "n^2", "n"])
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
        algorithm = ThreeSum()
        algorithm.solve()
        # Array range is -100 ~ 100, array size is 100.
        algorithm.append_line_into_df_in_wrap(
            ["Target list", "[random.randint(-10, 10) for _ in range(10)]"]
        )
        algorithm.append_line_into_df_in_wrap(
            ["Found combinations", "", algorithm.found_list]
        )
        algorithm.visualize()


def three_sum_generalization(input_lines: Optional[Iterator[str]] = None) -> str:
    import math
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    n, target_sum = map(int, input_().split())
    sequence: list[int] = list(map(int, input_().split()))
    target_subsequences_count = 0
    control_var_i: deque[int] = deque()
    control_var_sum: int = 0
    consecutive_elements_sum: int = 0

    # Title: solve
    sequence.sort()

    # when the number of element of subsequence == 1,
    for e in sequence:
        if e == target_sum:
            target_subsequences_count += 1
        elif e > target_sum:
            break

    # when the number of element of subsequence >= 2,
    while len(control_var_i) + 2 <= n:

        inner_i: int = control_var_i[0] + 1 if control_var_i else 0
        inner_j: int = n - 1

        while inner_i < inner_j:
            temp_sum: int = control_var_sum + sequence[inner_i] + sequence[inner_j]
            if temp_sum == target_sum:
                # check elements with duplicated value
                # print([sequence[i] for i in [*control_var_i, inner_i, inner_j]])

                left_count = right_count = 1
                while inner_i < inner_j and sequence[inner_i] == sequence[inner_i + 1]:
                    inner_i += 1
                    left_count += 1
                while inner_i < inner_j and sequence[inner_j] == sequence[inner_j - 1]:
                    inner_j -= 1
                    right_count += 1

                # if elements of inner_i ~ right_i are same.
                if inner_i == inner_j and right_count == 1:
                    target_subsequences_count += math.comb(left_count, 2)
                else:
                    target_subsequences_count += left_count * right_count

                inner_i += 1
                inner_j -= 1
            elif temp_sum < target_sum:
                inner_i += 1
            else:
                inner_j -= 1

        # modify pointer when <control_var_i>s exist (n >= 3)
        for i in range(len(control_var_i)):
            control_var_sum -= sequence[control_var_i[i]]
            # if a pointer not exceeds valid range
            if control_var_i[i] + 1 != n - 2 - i:
                control_var_i[i] += 1
                control_var_sum += sequence[control_var_i[i]]
                for depth, ii in enumerate(range(i - 1, -1, -1), start=1):
                    control_var_i[ii] = control_var_i[i] + depth
                    control_var_sum += sequence[control_var_i[ii]]
                break
        else:
            # when combinations that can be made up with the number of <control_var_i> ends.
            previous_length: int = len(control_var_i)
            control_var_i.clear()
            control_var_i.extendleft(deque(range(previous_length + 1)))
            consecutive_elements_sum += sequence[len(control_var_i) - 1]
            control_var_sum = consecutive_elements_sum

    print(target_subsequences_count)
    return str(target_subsequences_count)


def test_three_sum_generalization() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "5 0",
                "-7 -3 -2 5 8",
            ],
            ["1"],
        ],
        [
            [
                "5 0",
                "0 0 0 0 0",
            ],
            ["31"],  # 5C1 ~ 5C5  =  5, 10, 10, 5, 1
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            three_sum_generalization(iter(input_lines)), output_lines[0]
        )
        print(f"elapsed time: {time.time() - start_time}")


#%%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [ThreeSum]
    VisualizationManager.call_root_classes(only_class_list)

# Knapsack problem
