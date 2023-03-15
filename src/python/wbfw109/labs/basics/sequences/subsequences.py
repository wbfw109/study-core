# %%
from __future__ import annotations

import random
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

arr: list[int] = [random.randint(1, 10) for _ in range(10)]
# undefined so can be set to any value
found_elements_len: list[int] = [-1, *[0] * 9]
arr
found_len = 0
for i in range(len(arr)):
    # Binary search for the smallest positive l ≤ L
    # such that X[M[l]] > X[i]
    lo = 1
    hi = found_len + 1
    print("mid", mid)
    while lo < hi:
        mid = lo + (hi - lo) // 2  # lo <= mid < hi
        if arr[found_elements_len[mid]] >= arr[i]:
            hi = mid
        else:  # if X[M[mid]] < X[i]
            lo = mid + 1
    print("lo", lo)
    # After searching, lo == hi is 1 greater than the
    # length of the longest prefix of X[i]
    new_found_len = lo
    # The predecessor of X[i] is the last index of
    # the subsequence of length newL-1
    arr[i] = found_elements_len[new_found_len - 1]
    found_elements_len[new_found_len] = i

    if new_found_len > found_len:
        # If we found a subsequence longer than any we've
        # found yet, update L
        found_len = new_found_len

# Reconstruct the longest increasing subsequence
# It consists of the values of X at the L indices:
# ...,  P[P[M[L]]], P[M[L]], M[L]
S = [0] * found_len
S
k = found_elements_len[found_len]
for j in range(found_len - 1, -1, -1):
    S[j] = arr[k]
    k = arr[k]
arr
S
#%%
N = 10
X: list[int] = [random.randint(1, 10) for _ in range(N)]
P = [0] * N
M = [0] * (N + 1)
M[0] = -1  # undefined so can be set to any value

L = 0
for i in range(N):
    # Binary search for the smallest positive l ≤ L
    # such that X[M[l]] > X[i]
    lo = 1
    hi = L + 1
    while lo < hi:
        mid = lo + (hi - lo) // 2  # lo <= mid < hi
        if X[M[mid]] >= X[i]:
            hi = mid
        else:  # if X[M[mid]] < X[i]
            lo = mid + 1

    # After searching, lo == hi is 1 greater than the
    # length of the longest prefix of X[i]
    newL = lo

    # The predecessor of X[i] is the last index of
    # the subsequence of length newL-1
    P[i] = M[newL - 1]
    M[newL] = i

    if newL > L:
        # If we found a subsequence longer than any we've
        # found yet, update L
        L = newL

# Reconstruct the longest increasing subsequence
# It consists of the values of X at the L indices:
# ...,  P[P[M[L]]], P[M[L]], M[L]
S = [0] * L
k = M[L]
for j in range(L - 1, -1, -1):
    S[j] = X[k]
    k = P[k]
S
# return S
