from __future__ import annotations

import time
import unittest
from typing import Iterator, Optional


def solve_subset_sum_with_strictly_positive_elements(
    input_lines: Optional[Iterator[str]] = None,
) -> str:
    """https://devth-preview.goorm.io/exam/53763/%EC%BD%94%EB%94%A9-%ED%85%8C%EC%8A%A4%ED%8A%B8-%EC%9D%91%EC%8B%9C-%ED%99%98%EA%B2%BD-%EC%B2%B4%ED%97%98%ED%95%98%EA%B8%B0/quiz/4?_ga=2.22476799.531264434.1598839678-1290480941.1598839678

    Time Complexity (Worst-case): O( n*k*n ) (pseudo-polynomial time)

    Space Complexity (Worst-case): O( n^2 ) from rolling window approach
        if rolling window approach is applied into one <n> not <k>, it will be O( n*k ).

    Key point
        3D Dynamic programming with rolling window approach

    Tag
        Dynamic programming, Masking, Memoization

    Debugging
        some combinations is tested by itertools.combinations and set.intersections.
        =====
        6C2
        -----
        0, 1; 0
        2C2; 1    01          ; sums=1
        3C2; 3    01 02 12    ; sums=1,  2, 3
        4C2; 6    01 02 12 03 13 23               ; sums=1,  2, 3,  3, 4, 5
        5C2; 10   01 02 12 03 13 23 04 14 24 34   ; sums=1,  2, 3,  3, 4, 5,  4, 5, 6, 7
        =====
        6C3
        -----
        0, 1, 2; 0
        3C3; 1    012                 ; sums=3
        4C3; 4    012 013 023  123    ; sums=3,  (4, 5, 6)
        5C3; 10   012 013 023  123 014 024 034 124 134 234  ; sums=3,  4, 5, 6,  (5, 6, 7,  7, 8, 9)
        6C3; 20   ...                                       ; ... (6, 7, 8, 9,  8, 9, 10,  10, 11, 12)
    From following debugging
        C(3, 3) = C(2, 3), (C(2, 2) with "2")
        C(4, 3) = C(3, 3), (C(3, 2) with "3")
        C(1, 1) = C(0, 1), (C(0, 0) with 0)))
    Recurrence Relation
        kk = 1, nn ≥ 1
            C(nn, kk) = C(n-1, kk), nn-1
        kk ≥ 2, nn ≥ 2
            C(nn, kk) = C(n-1, kk), (C(nn-1, kk-1) with "nn-1")
            - from third debugging, it also meets when k = 1 and nn = 1
    Implementation of with (n+1)*(k+1)*n 3D matrix (dynamic programming without optimization)
        It have n*k*n time and space complexity (pseudo polynomial time).
        dp: (n+1)*(k+1)*n; The number of combinations
            , such that the sum is (sum % n) from C(first n elements, k).
        ⚠️ but it causes "Memory Limit exceeded" and "Time out".
        so I used rolling window approach.
        It only needs dp[nn-1][kk] and dp[nn-1][kk-1] to calculate do[nn][kk] so that it requires maximum (n+1)*2*n space.
        -----
        Procedure (*<x> is order of calculation)
        1 0  0
        1 *1 0
        1 *2 *n+1 th
        1 *3 *n+2 th
        ----
        ⚠️ When submit the code if not remove not required code including comments and empty lines in coding test
        , it causes TimeOut error at the last test case.
    """
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    n, k = map(int, input_().split())
    dp: list[list[list[int]]] = [[[0] * n for _ in range(2)] for _ in range(n + 1)]

    # Title: solve
    # initialize dp
    for nn in range(n + 1):
        dp[nn][0][0] = 1
    ## when k ≥ 1, n ≥ 1
    for kk in range(1, k + 1):
        kki: int = kk & 1
        pkki: int = kki ^ 1  # previous kk index

        dp[kk - 1][kki] = [0] * n
        for nn in range(kk, n + 1):
            # set dp[nn][kk]
            pnn: int = nn - 1
            # bring previous combinations by copying it
            dp[nn][kki] = dp[pnn][kki].copy()

            # add new combinations from previous sums with new element
            # C(nn-1, kk-1) with "nn-1" denotes (nn-1+0), (nn-1+1) ... (nn-1)+(n-1).
            for a_sum, count in enumerate(dp[pnn][pkki], start=pnn):
                if a_sum >= n:
                    dp[nn][kki][a_sum - n] += count
                else:
                    dp[nn][kki][a_sum] += count

    # Title: output
    result: str = str(dp[n][k & 1][0] % 1000000007)
    print(result)
    return result


def test_subset_sum_with_strictly_positive_elements() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "7 4",
            ],
            ["5"],
        ],
        [
            [
                "10 3",
            ],
            ["12"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            solve_subset_sum_with_strictly_positive_elements(iter(input_lines)),
            "\n".join(output_lines),
        )
        print(f"elapsed time: {time.time() - start_time}")


test_subset_sum_with_strictly_positive_elements()
