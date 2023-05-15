from __future__ import annotations

import time
import unittest
from typing import Iterator, Optional

### Title: Geometry ~


## Title: Sweep line ~
# TODO: ~, Segement tree with Lazy propagation


def solution_10534(input_lines: Optional[Iterator[str]] = None) -> str:
    """[Platinum 1] 락페스티벌 ; https://www.acmicpc.net/problem/10534"""


def solution_2261(input_lines: Optional[Iterator[str]] = None) -> str:
    """[Platinum 2] 가장 가까운 두 점 ; https://www.acmicpc.net/problem/2261"""


def solution_3392(input_lines: Optional[Iterator[str]] = None) -> str:
    """[Platinum 2] 화성 지도 ; https://www.acmicpc.net/problem/3392"""


def solution_10000(input_lines: Optional[Iterator[str]] = None) -> str:
    """[Platinum 4] 원 영역 ; https://www.acmicpc.net/problem/10000"""


### Title: Sequence ~


def solution_14003(input_lines: Optional[Iterator[str]] = None) -> str:
    """[Platinum 5] 가장 긴 증가하는 부분 수열 5 ; https://www.acmicpc.net/problem/14003
    Tag: Sequence (Subsequence)

    Time Complexity (Worst-case): O(n(log n))
        - O(n(log n)) from binary search to find suitable length of increasing subsequence for sequence[i]
        - O(n) from Reconstructing the longest increasing subsequence

    Space Complexity (Worst-case): O(n) from <predecessor_indexes>, <smallest_indexes_at_l>, <longest_increasing_subsequence>
    """
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (1 ≤ N ≤ 10^6)
    n: int = int(input_())
    # condition: (-(10^9) ≤ element of sequence ≤ 10^9)
    sequence: list[int] = list(map(int, input_().split()))

    # Title: solve
    ## find Longest Increasing Sequence
    predecessor_indexes = [0] * n
    indexes_of_smallest_v_at_l: list[int] = [0] * (n + 1)
    indexes_of_smallest_v_at_l[0] = -1
    found_subsequence_l: int = 0
    longest_increasing_subsequence: list[int] = []

    for i in range(len(sequence)):
        low = 1
        high = found_subsequence_l + 1

        while low < high:
            mid = low + (high - low) // 2
            if sequence[indexes_of_smallest_v_at_l[mid]] >= sequence[i]:
                high = mid
            else:
                low = mid + 1

        new_l = low
        predecessor_indexes[i] = indexes_of_smallest_v_at_l[new_l - 1]
        indexes_of_smallest_v_at_l[new_l] = i
        if new_l > found_subsequence_l:
            found_subsequence_l = new_l

    longest_increasing_subsequence = [0] * found_subsequence_l
    k: int = indexes_of_smallest_v_at_l[found_subsequence_l]
    for j in range(found_subsequence_l - 1, -1, -1):
        longest_increasing_subsequence[j] = sequence[k]
        k = predecessor_indexes[k]

    # Title: output
    result = "\n".join(
        [str(found_subsequence_l), " ".join(map(str, longest_increasing_subsequence))]
    )
    print(result)
    return result


def test_solution_14003() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "6",
                "10 20 10 30 20 50",
            ],
            ["4", "10 20 30 50"],
        ]
    ]:
        start_time = time.time()
        test_case.assertEqual(
            solution_14003(iter(input_lines)),
            "\n".join(output_lines),
        )
        print(f"elapsed time: {time.time() - start_time}")
