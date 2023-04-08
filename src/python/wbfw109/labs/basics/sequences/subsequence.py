# %%
from __future__ import annotations

import random

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    VisualizationManager,
    VisualizationRoot,
    display_data_frame_with_my_settings,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode
# %%


# TODO: look again after doing Patience sorting
class LongestIncreasingSubsequence(VisualizationRoot):
    """
    Variables:
        - <sequence>: X.
        - <predecessor_indexes>: P.
            P stores the index of the predecessor of X[k] in the longest increasing subsequence ending at X[K]
        - <smallest_indexes_at_l>: M.
            M stores the index k of the smallest value X[k] such that there is an increasing subsequence of length l ending at X[k] in the range k ≤ i.
            🚣 One of <smallest_indexes_at_l> could be updated whenever evaluate X[K].
                but Note that <longest_increasing_subsequence> is created only from last index of <found_subsequence_l> by tracing <predecessor_indexes>
                , which means creatable increasing subsequences ending at length l in <smallest_indexes_at_l> are independent.
            🚣 It uses binary search with custom predicate to find <new_l> for X[K.]
        - <longest_increasing_subsequence>: S (permutation (set)).
        - <found_subsequence_l>: L (length of Longest Increasing Subsequence).
        - <new_l>: l (suitable length for X[i]. 1 ≤ l ≤ L)
            Target of Binary search (== <mid>): smallest positive l such that X[M[l]] > X[i]

    Detail of Binary search used in this implementation: ➡️
        🔍 It seems that there is no duplicate calculation of the <mid>
        , because the loop is performed and broken out when <high - low> is less than 2.

        Note that range of Binary search (lo, high) is inclusive that could be set.
    """

    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["var", "print"],
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
            "⚙️ Based on Longest increasing subsequence, Longest decreasing subsequence could be created.",
            "",
            ": 💻 Complexity Class of the decision problem is P and have strong polynomial time to exact algorithm.",
            "  decision problem: to ask whether there exists an increasing subsequence of a given sequence of numbers with at least specified length.",
        ]
        n: int = 10
        self.sequence: list[int] = [random.randint(-10, 10) for _ in range(n)]
        self.predecessor_indexes = [0] * n
        self.smallest_indexes_at_l: list[int] = [0] * (n + 1)
        self.smallest_indexes_at_l[0] = -1  # undefined so can be set to any value
        self.found_subsequence_l: int = 0
        self.longest_increasing_subsequence: list[int] = []

    def __str__(self) -> str:
        return "-"

    def solve(self) -> None:
        for i in range(len(self.sequence)):
            ## Binary search to find suitable length of increasing subsequence in 1 ~ <found_subsequence_l> for X[i].
            # target is smallest positive l such that X[i] < X[M[l]]  (1 ≤ l ≤ L)

            # <low> and <high> are (minimum and maximum) length which could be set.
            low = 1
            high = self.found_subsequence_l + 1

            while low < high:
                mid = low + (high - low) // 2  # 📍 low <= mid < high
                if self.sequence[self.smallest_indexes_at_l[mid]] < self.sequence[i]:
                    low = mid + 1
                else:
                    # "high = mid" when high is included in valid range ("equal or less than" operator).
                    high = mid  # so the statements is not "mid -1"

            # 🚣 After searching, (lo == hi) is 1 greater than the length of the longest prefix of X[i].
            new_l = low
            # Note that minimum <new_l> is 1.
            # if value that could be root, <predecessor_indexes[i]> will be -1.
            self.predecessor_indexes[i] = self.smallest_indexes_at_l[new_l - 1]
            self.smallest_indexes_at_l[new_l] = i
            if new_l > self.found_subsequence_l:
                # If we found a subsequence longer than any we've found yet, update <found_L>.
                self.found_subsequence_l = new_l

        # Reconstruct the longest increasing subsequence.
        # It consists of the values of X at the L indices:
        # ...,  P[P[M[L]]], P[M[L]], M[L]
        self.longest_increasing_subsequence = [0] * self.found_subsequence_l
        k: int = self.smallest_indexes_at_l[self.found_subsequence_l]
        for j in range(self.found_subsequence_l - 1, -1, -1):
            self.longest_increasing_subsequence[j] = self.sequence[k]
            k = self.predecessor_indexes[k]  # last k will be -1 (root)

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
        algorithm.append_line_into_df_in_wrap(["Target list", algorithm.sequence])
        algorithm.append_line_into_df_in_wrap(
            ["predecessor_indexes", algorithm.predecessor_indexes]
        )
        algorithm.append_line_into_df_in_wrap(
            ["smallest_indexes_at_l", algorithm.smallest_indexes_at_l]
        )
        algorithm.append_line_into_df_in_wrap(
            ["found_subsequence_l", algorithm.found_subsequence_l]
        )
        algorithm.append_line_into_df_in_wrap(
            ["longest_increasing_subsequence", algorithm.longest_increasing_subsequence]
        )
        algorithm.visualize()


if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [LongestIncreasingSubsequence]
    VisualizationManager.call_root_classes(only_class_list)
