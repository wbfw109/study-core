""": 💻 Complexity Class of the decision problem is P and have strong polynomial time to exact algorithm.
    decision problem: to ask whether given a set of elements, determine if they are sorted in a specific order (ascending or descending)."""

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


class Glossary(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["context", "word", "meaning"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "📖 Glossary for Sorting"

    @classmethod
    def test_case(cls):
        glossary: Glossary = cls()
        glossary.append_line_into_df_in_wrap(
            [
                "Sorting Properties",
                "Adaptive",
                "efficient for data sets that are already substantially sorted",
            ]
        )
        glossary.append_line_into_df_in_wrap(
            [
                "Sorting Properties",
                "Stable",
                "does not change the relative order of elements with equal keys",
            ]
        )
        glossary.append_line_into_df_in_wrap(
            [
                "Sorting Properties",
                "In-place",
                "only requires a constant amount O(1) of additional memory space",
            ]
        )
        glossary.append_line_into_df_in_wrap(
            [
                "Sorting Properties",
                "Online",
                "can process its input piece-by-piece in a serial fashion. namely, can sort a list as it receives it.",
            ]
        )
        glossary.visualize()


@dataclasses.dataclass
class SortingDST:
    sequence: list[int] = dataclasses.field(default_factory=list)

    @staticmethod
    def get_default_sorting_dst(
        list_len: int = 1000, random_range: int = 1000
    ) -> SortingDST:
        """Simple sorts are efficient on small data, due to low overhead, but not efficient on large data."""
        return SortingDST(
            sequence=[random.randint(1, random_range) for _ in range(list_len)]
        )

    @staticmethod
    def verify_sorting(sorting_dst: SortingDST) -> bool | Any:
        # check whether ascending or descending
        for compare_operator in [operator.gt, operator.lt]:
            is_verified: bool = True
            for i in range(len(sorting_dst.sequence) - 1):
                if compare_operator(
                    sorting_dst.sequence[i], sorting_dst.sequence[i + 1]
                ):
                    is_verified = False
                    break
            if is_verified:
                if compare_operator == operator.gt:
                    return [is_verified, "ascending"]
                else:
                    return [is_verified, "descending"]
        return False


class ExchangeSorts(MixInParentAlgorithmVisualization):
    """: 🍡 Bubble sort, Cocktail shaker sort, Odd–even sort, Comb sort, Gnome sort, Proportion extend sort, Quicksort, Slowsort, Stooge sort, Bogosort"""

    class BubbleSort(ChildAlgorithmVisualization[SortingDST]):
        def __init__(self, /, dst: Optional[SortingDST]) -> None:
            super().__init__(columns=["elapsed time", "verification"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(
                ["n", "n^2", "n^2", "1"]
            )
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
            sequence = self.dst.sequence
            n: int = len(sequence)

            # Note: Optimization: by using <last_swapped_i>
            while True:
                last_swapped_i = 0
                for i in range(n - 1):
                    # in sublist loop, Swapping all like bubble
                    if sequence[i] > sequence[i + 1]:
                        sequence[i], sequence[i + 1] = sequence[i + 1], sequence[i]
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

    # TODO: Require to upgrade after Medians of medians, QuickSelection.
    class QuickSort(ChildAlgorithmVisualization[SortingDST]):
        def __init__(self, /, dst: Optional[SortingDST]) -> None:
            super().__init__(columns=["elapsed time", "verification"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(
                [
                    "n*log(n)",
                    "n*log(n)",
                    "n^2",
                    "log(n)",
                ]
            )
            self.big_o_visualization.df_caption = [
                "⚙️ [Worse-case] Time complexity: O(N^2)",
                "  - When one of the sublists returned by the partitioning routine is of size n − 1.",
                "    🛍️ e.g. This may occur if the pivot happens to be the smallest or largest element in the list, or when all the elements are equal.",
            ]

        def __str__(self) -> str:
            return "\n".join(
                [
                    "Properties: comparison-based, In-place",
                    "methods: Partitioning (Hoare partition scheme)",
                    "pivot selection method: Medium",
                ]
            )

        def quicksort(self, sequence: list[int], start_i: int, end_i: int) -> None:
            if 0 <= start_i < end_i:
                next_range = self.partition(sequence, start_i, end_i)
                # quicksort recursively left side of the pivot. Note that the pivot is included in left side.
                self.quicksort(sequence, start_i, next_range)
                self.quicksort(sequence, next_range + 1, end_i)

        def partition(self, sequence: list[int], start_i: int, end_i: int) -> int:
            # select value of middle index
            pivot: int = sequence[(start_i + end_i) // 2]

            # temp variables in do-while statement
            i = start_i - 1
            j = end_i + 1
            while True:
                while sequence[(i := i + 1)] < pivot:
                    pass
                while sequence[(j := j - 1)] > pivot:
                    pass
                if i >= j:
                    # If the indices crossed, return
                    return j
                sequence[i], sequence[j] = (
                    sequence[j],
                    sequence[i],
                )

        def solve(self) -> None:
            sequence = self.dst.sequence

            n: int = len(sequence)
            self.quicksort(sequence, 0, n - 1)

        def verify(self) -> bool | Any:
            return SortingDST.verify_sorting(self.dst)

        @classmethod
        def test_case(cls, dst: Optional[SortingDST]) -> None:  # type: ignore
            algorithm = ExchangeSorts.QuickSort(dst=dst)
            algorithm.append_line_into_df_in_wrap(algorithm.measure())
            algorithm.visualize()


class SelectionSorts(MixInParentAlgorithmVisualization):
    """: 🍡 Selection sort, Heapsort, Smoothsort, Cartesian tree sort, Tournament sort, Cycle sort, Weak-heap sort"""

    class SelectionSort(ChildAlgorithmVisualization[SortingDST]):
        def __init__(self, /, dst: Optional[SortingDST]) -> None:
            super().__init__(columns=["elapsed time", "verification"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(
                ["n", "n^2", "n^2", "1"]
            )
            self.big_o_visualization.df_caption = [
                "⚙️ [Worse-case] Time complexity: ( O(N^2) comparisons, O(N) swaps )",
                "  - Selecting the minimum requires scanning n elements (taking n-1 comparisons) and then swapping it into the first position. and n-2 ... so on.",
                "",
                "⚙️ It is useful where swapping cost is very expensive and when write performance is a limiting factor.",
            ]

        def __str__(self) -> str:
            return "\n".join(
                [
                    "Properties: comparison-based, In-place",
                    "methods: Selection",
                ]
            )

        def solve(self) -> None:
            sequence = self.dst.sequence
            n: int = len(sequence)
            for i in range(n):
                # Find the index that have minimum value in sublist.
                min_value_i: int = i
                for j in range(i + 1, n):
                    if sequence[min_value_i] > sequence[j]:
                        min_value_i = j

                # Swapping: (found index, start-index(<i>) from sublist)... So N(<LIST_LEN>) Times.
                if min_value_i != i:
                    sequence[i], sequence[min_value_i] = (
                        sequence[min_value_i],
                        sequence[i],
                    )

        def verify(self) -> bool | Any:
            return SortingDST.verify_sorting(self.dst)

        @classmethod
        def test_case(cls, dst: Optional[SortingDST]) -> None:  # type: ignore
            algorithm = SelectionSorts.SelectionSort(dst=dst)
            algorithm.append_line_into_df_in_wrap(algorithm.measure())
            algorithm.visualize()

    #  Therefore, the performance of this algorithm is O(n + n log n) = O(n log n).
    # TODO: equal key version that have O(n)?, sift up version
    class HeapSort(ChildAlgorithmVisualization[SortingDST]):
        """It uses complete binary tree
        - Python heapq library:
            Our pop method returns the smallest item, not the largest
            (called a “min heap” in textbooks; a “max heap” is more common in texts because of its suitability for in-place sorting).
        """

        def __init__(self, /, dst: Optional[SortingDST]) -> None:
            super().__init__(columns=["elapsed time", "verification"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(
                [
                    "n*log(n)",
                    "n*log(n)",
                    "n*log(n)",
                    "1",
                ]
            )
            self.big_o_visualization.df_caption = [
                "⚙️ [Worse-case] Time complexity: O(n*log(n))",
                "  - The <build_max_heap> method is run once, and is O(n) in performance.",
                "  - The <sift_down> method is O(log n), and is called n times.",
                "  = Therefore, the performance of this algorithm is O(n + n log n) = O(n log n).",
            ]

        def __str__(self) -> str:
            return "\n".join(
                [
                    "Properties: comparison-based, In-place",
                    "methods: Selection",
                    "sift method: sift down",
                ]
            )

        @staticmethod
        def get_heap_child_i(i: int, left_or_right: Literal[1, 2]) -> int:
            """
            Args:
                i (int): the zero-based index.
                left_or_right (Literal[1, 2]): set 1 if left child, otherwise 2
            """
            return 2 * i + left_or_right

        @staticmethod
        def get_heap_parent_i(i: int) -> int:
            """
            Args:
                i (int): the zero-based index.
            """
            return (i - 1) // 2

        def build_max_heap(self, sequence: list[int], n: int) -> None:
            # heapify
            ni = n - 1
            start_i = self.get_heap_parent_i(ni)
            while start_i >= 0:
                self.sift_down(sequence, start_i, end_i=ni)
                # next parent node
                start_i = start_i - 1

        def sift_down(self, sequence: list[int], start_i: int, end_i: int) -> None:
            """📝 Repair the heap whose root element is at start-index"""
            root_i: int = start_i
            while (left_child_i := self.get_heap_child_i(root_i, 1)) <= end_i:
                # find i_to_be_swapped. <left_child_i + 1> is <right_child_i>
                i_to_be_swapped = root_i
                right_child_i: int = left_child_i + 1

                # compare three nodes
                if sequence[root_i] < sequence[left_child_i]:
                    i_to_be_swapped = left_child_i
                if (
                    right_child_i <= end_i
                    and sequence[i_to_be_swapped] < sequence[right_child_i]
                ):
                    i_to_be_swapped = right_child_i

                if i_to_be_swapped == root_i:
                    # return if already valid
                    return
                else:
                    # else swap and continue sifting down the child.
                    sequence[root_i], sequence[i_to_be_swapped] = (
                        sequence[i_to_be_swapped],
                        sequence[root_i],
                    )
                    root_i = i_to_be_swapped

        def solve(self) -> None:
            # assume that sequence[0] is root node.
            sequence = self.dst.sequence
            n: int = len(self.dst.sequence)

            self.build_max_heap(sequence, n)

            end_i = n - 1
            while end_i >= 1:
                sequence[0], sequence[end_i] = (
                    sequence[end_i],
                    sequence[0],
                )
                end_i -= 1
                self.sift_down(sequence, start_i=0, end_i=end_i)

        def verify(self) -> bool | Any:
            return SortingDST.verify_sorting(self.dst)

        @classmethod
        def test_case(cls, dst: Optional[SortingDST]) -> None:  # type: ignore
            algorithm = SelectionSorts.HeapSort(dst=dst)
            algorithm.append_line_into_df_in_wrap(algorithm.measure())
            algorithm.visualize()


class InsertionSorts(MixInParentAlgorithmVisualization):
    """: 🍡 Insertion sort, Shellsort, Splaysort, Tree sort, Lsibrary sort, Patience sorting"""

    class InsertionSort(ChildAlgorithmVisualization[SortingDST]):
        def __init__(self, /, dst: Optional[SortingDST]) -> None:
            super().__init__(columns=["elapsed time", "verification"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(
                ["n", "n^2", "n^2", "1"]
            )
            self.big_o_visualization.df_caption = [
                "⚙️ [Worse-case] Time complexity: ( O(N^2) comparisons )",
                "  - The set of all worst case inputs consists of all arrays where each element is the smallest or second-smallest of the elements before it.",
                "  - In these cases, every iteration of the inner loop will scan and shift the entire sorted subsection of the array before inserting the next element",
                "  - Example: an array sorted in reverse order"
                "⚙️ [Best-case] Time complexity: ( O(N) comparisons, O(1) swaps) in the case of already sorted list",
                "",
                "⚙️ Adaptive: the time complexity is O(kn) when each element in the input is no more than k places away from its sorted position.",
            ]

        def __str__(self) -> str:
            return "\n".join(
                [
                    "Properties: comparison-based, Adaptive, Stable, In-place, Online",
                    "methods: Insertion",
                ]
            )

        def solve(self) -> None:
            sequence = self.dst.sequence

            n: int = len(sequence)
            for i in range(1, n):
                # Note: Optimization: actual insertion could only performs one assignment by using <target_value> instead of Swapping like bubble.
                comparison_value = sequence[i]
                j: int = i - 1
                # in sublist loop, 🔍 for advantage of spatial locality, is start-index of inner loop <i>?
                while j >= 0 and sequence[j] > comparison_value:
                    sequence[j + 1] = sequence[j]
                    j -= 1
                else:
                    sequence[j + 1] = comparison_value

        def verify(self) -> bool | Any:
            return SortingDST.verify_sorting(self.dst)

        @classmethod
        def test_case(cls, dst: Optional[SortingDST]) -> None:  # type: ignore
            algorithm = InsertionSorts.InsertionSort(dst=dst)
            algorithm.append_line_into_df_in_wrap(algorithm.measure())
            algorithm.visualize()


# TODO: Parallel sort, Quadsort, in-place version after binary searches
class MergeSorts(MixInParentAlgorithmVisualization):
    """: 🍡 Merge sort, Cascade merge sort, Oscillating merge sort, Polyphase merge sort"""

    class MergeSort(ChildAlgorithmVisualization[SortingDST]):
        def __init__(self, /, dst: Optional[SortingDST]) -> None:
            super().__init__(columns=["elapsed time", "verification"], dst=dst)
            self.big_o_visualization.append_line_into_df_in_wrap(
                [
                    "n*log(n)",
                    "n*log(n)",
                    "n*log(n)",
                    "n",
                ]
            )
            self.big_o_visualization.df_caption = [
                "⚙️ [Worse-case] Time complexity: ( n*log(n) )",
                "  - If the running time of merge sort for a list of length n is T(n), then the recurrence relation is T(n) = 2T(n/2) + n",
                "  - '+ n' is taken time to compare elements in a combining step.",
                "",
                "⚙️ It is also easily applied to lists, not only arrays, as it only requires sequential access, not random access.",
            ]

        def __str__(self) -> str:
            return "\n".join(
                [
                    "Properties: comparison-based, Stable, divide-and-conquer, sequential access",
                    "Optional X: In-place",
                    "methods: Merging",
                ]
            )

        def merge_sort(self, sequence: list[int]) -> list[int]:
            nn: int = len(sequence)
            if nn <= 1:
                return sequence
            middle_i: int = nn // 2
            # recursively sort two sublists.
            return self.merge(
                left_list=self.merge_sort(sequence[0:middle_i]),
                right_list=self.merge_sort(sequence[middle_i:nn]),
            )

        def merge(self, left_list: list[int], right_list: list[int]) -> list[int]:
            result_list: list[int] = []
            nn: int = len(left_list)
            mm: int = len(right_list)
            goal_size: int = len(left_list) + len(right_list)
            # compare
            i: int = 0
            j: int = 0
            while True:
                if left_list[i] <= right_list[j]:
                    result_list.append(left_list[i])
                    if (i := i + 1) == nn:
                        result_list.extend(right_list[j:])
                else:
                    result_list.append(right_list[j])
                    if (j := j + 1) == mm:
                        result_list.extend(left_list[i:])

                if len(result_list) == goal_size:
                    break
            return result_list

        def solve(self) -> None:
            # Top-down implementation
            # slice operator will copy original list, so the allocation is required.
            self.dst.sequence = self.merge_sort(self.dst.sequence)

        def verify(self) -> bool | Any:
            return SortingDST.verify_sorting(self.dst)

        @classmethod
        def test_case(cls, dst: Optional[SortingDST]) -> None:  # type: ignore
            algorithm = MergeSorts.MergeSort(dst=dst)
            algorithm.append_line_into_df_in_wrap(algorithm.measure())
            algorithm.visualize()


class DistributionSorts(MixInParentAlgorithmVisualization):
    """: 🍡 American flag sort, Bead sort, Bucket sort, Burstsort, Counting sort, Interpolation sort, Pigeonhole sort, Proxmap sort, Radix sort, Flashsort"""


class HybridSorts(MixInParentAlgorithmVisualization):
    """: 🍡 Block merge sort, Kirkpatrick–Reisch sort, Timsort, Introsort, Spreadsort, Merge-insertion sort"""

    # class TimSort(ChildAlgorithmVisualization[SortingDST]):
    #     def __init__(self, /, dst: Optional[SortingDST]) -> None:
    #         super().__init__(columns=["elapsed time", "verification"], dst=dst)
    #         self.big_o_visualization.append_line_into_df_in_wrap(
    #             [
    #                 "n",
    #                 "n*log(n)",
    #                 "n*log(n)",
    #                 "n",
    #             ]
    #         )
    #       self.big_o_visualization.df_caption = []
    #     def __str__(self) -> str:
    #         return "\n".join(
    #             [
    #                 "Properties: comparison-based, Stable, ???",
    #                 "methods: Insertion & Merging",
    #             ]
    #         )

    #     def solve(self) -> None:
    #         target_list_len: int = len(self.dst.target_list)
    #         self.quicksort(0, target_list_len - 1)

    #     def verify(self) -> bool | Any:
    #         return SortingDST.verify_sorting(self.dst)

    #     @classmethod
    #     def test_case(cls, dst: Optional[SortingDST]) -> None:  # type: ignore
    #         algorithm = HybridSorts.TimSort(dst=dst)
    #         algorithm.append_line_into_df_in_wrap(algorithm.measure())
    #         algorithm.visualize()


# Timsort...
# TODO: Counting sort, all (ascending, descending) version
# %%

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
