"""Purpose is to solve in 22 minutes
- 💤: 문제 이해 제대로 하기
    "문제 설명"에 모든 제한사항이 함게 주어지지 않는 경우가 있어서, 완벽히 이해하기 어려울 수 있다.
    에매한 경우, 명확하게 이해할 수 있도록 "예시" 가 주어지므로 꼭 함께 보도록 한다.
- 🧠: (신박한 풀이, 수학식을 활용한 풀이, Benchmark)
- 💦: built-in functions, grammar
"""

from __future__ import annotations

import time
import unittest
from typing import Iterator, Optional


def solve_subset_sum_with_strictly_positive_elements(
    input_lines: Optional[Iterator[str]] = None,
) -> str:
    """https://devth-preview.goorm.io/exam/53763/%EC%BD%94%EB%94%A9-%ED%85%8C%EC%8A%A4%ED%8A%B8-%EC%9D%91%EC%8B%9C-%ED%99%98%EA%B2%BD-%EC%B2%B4%ED%97%98%ED%95%98%EA%B8%B0/quiz/4?_ga=2.22476799.531264434.1598839678-1290480941.1598839678

    Time Complexity (Worst-case): O( n*k*n ) (pseudo-polynomial time)

    Space Complexity (Worst-case): O( n*2 ) from rolling window approach

    Tag
        Dynamic programming (Rolling window approach for 3D dp array), Masking, Memoization

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
        🚣 C is combinations(first n elements, k subset)
        C(4, 3) = C(3, 3), (C(3, 2) with "3")
        C(3, 3) = C(2, 3), (C(2, 2) with "2")
        C(1, 1) = C(0, 1), (C(0, 0) with 0)))
    Recurrence Relation
        k = 1, n ≥ 1
            C(n, k) = C(n-1, k), n-1
        k ≥ 2, n ≥ 2
            C(n, k) = C(n-1, k), (C(n-1, k-1) with "n-1")
            - from third debugging, it also meets when k = 1 and nn = 1
        ➡️ dp: (n+1)*(k+1)*n; The number of combinations such that the sum is (sum % n) from C(first n elements, k).
        It have n*k*n time and space complexity (pseudo polynomial time).
        ⚠️ but it causes "Memory Limit exceeded" and "Time out".
    Implementation
        1st attempt; (n+1)*(k+1)*n 3D array for Dynamic programming
            👎 [Memory limit exceeded, Time out]
            I will use rolling window approach.
            ➡️ It only needs dp[nn-1][kk] and dp[nn-1][kk-1] to calculate do[nn][kk] so that it requires maximum (n+1)*2*n space.
            -----
            Procedure (*<x> is order of calculation in n*k array)
                0 1  2    3  ..k
            0   - 0  0    0
            1   - *1 0    0
            2   - *2 *n+1 0
            3   - *3 *n+2 ...
            ..n
        2nd attempt; Naive Reification of Recurrence releation
            👎 [Time out]
                dp[nn][kki] = dp[pnn][kki].copy()
                for remainder, count in enumerate(dp[pnn][pkki]):
                    dp[nn][kki][(remainder+pnn) % n] += count
            ➡️ It can be thought as Bijective function; domain and codomain can be invertible.
                # <n>, <nn>, <pnn> is fixed in the loop.
                DP_SUM(i); dp[nn][kki][i] = dp[pnn][kki][i] + dp[pnn][pkki][❔]
                    `(remainder+pnn) % n = i`    can be converted `dp[pnn][pkki][(remainder+pnn) - n]` (Memoization technique)
                    and `(remainder+pnn) - n` can be converted indexing of enumerate() function.
        3rd attempt: Reification of Recurrence releation with 🚣 Memoization
            dp[nn][kki] = [count + dp[pnn][pkki][new_remainder] for new_remainder, count in enumerate(dp[pnn][kki], start=pnn - n)]
            ## Same solution, but list comprehension is slightly faster.
                for new_remainder, count in enumerate(dp[pnn][pkki], start=pnn - n):
                    dp[nn][kki][new_remainder] = count + dp[pnn][kki][new_remainder]
            ➡️ But It may be inefficient because it not uses spatial locality. (The cache fetches data in regular 'block' units.)
                in the case, a new block of memory must be fetched into the cache for each element accessed
                , which lowers the cache hit rate and thus the overall performance.
                # Title: solve
                dp: list[list[list[int]]] = [[[0] * n for _ in (0, 1)] for _ in range(n + 1)]
                for nn in range(n + 1):
                    dp[nn][0][0] = 1
                ## when k ≥ 1, n ≥ 1
                for kk in range(1, k + 1):
                    # move pointer of k
                    kki: int = kk & 1
                    pkki: int = kki ^ 1  # previous kk index

                    # set dp[nn][kk]; Bring previous sums, and Add new combinations from previous sums with new element.
                    # 💡 but for-loop starts from `for first_n in range(kk, n + 1)` because cases of nn < kk is zero in C(nn, kk).
                    # 💡 `dp[kk - 1][kki] = [0] * n` is required to additionally clear part not to be newly allocated from `for first_n in range(kk, n + 1)`
                    # , since it currently uses Sliding window approach.
                    dp[kk - 1][kki] = [0] * n
                    for nn in range(kk, n + 1):
                        pnn: int = nn - 1
                        dp[nn][kki] = [
                            count + dp[pnn][pkki][new_remainder]
                            for new_remainder, count in enumerate(dp[pnn][kki], start=pnn - n)
                        ]
                    #     print(f"nn {nn}, kk {kk}")
                    #     pprint(dp[nn][kki])
                    # print()
                # Title: output
                result: str = str(dp[n][k & 1][0] % 1000000007)
                print(result)
                return result
        4rd attempt: Rolling window approach with Row-major order for improving spatial locality
                0  1  2     3    .. n
            0   1  1  1     1
            1   - *1  *2   *3    ...
            2   -  -  *n+1 *n+2  ...
            3   -  -  -     ...
            ..k

    """
    # 11 lines except for input
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    n, k = map(int, input_().split())

    # Title: solve
    dp: list[list[list[int]]] = [[[0] * n for _ in range(n + 1)] for _ in (0, 1)]
    for nn in range(n + 1):
        dp[0][nn][0] = 1
    for kk in range(1, k + 1):
        kki = kk & 1
        pkki = kki ^ 1
        dp[kki][kk - 1] = [0] * n
        for nn in range(kk, n + 1):
            pnn = nn - 1
            # new_remainder := (i + pnn) % n
            dp[kki][nn] = [
                count + dp[pkki][pnn][new_remainder]
                for new_remainder, count in enumerate(dp[kki][pnn], start=pnn - n)
            ]

    # Title: output
    result: str = str(dp[k & 1][n][0] % 1000000007)
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
