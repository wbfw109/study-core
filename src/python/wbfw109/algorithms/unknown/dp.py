from __future__ import annotations

import sys
import time
import unittest
from pprint import pprint
from typing import Iterator, Optional

input_ = sys.stdin.readline


def rabbit_jump(input_lines: Optional[Iterator[str]] = None) -> str:
    """🔍 추가 테스트로 케이스 검증 필요 (코딩 사이트에서 비슷한 문제 발견 못함)
    X축 위에 있는 토끼가 0 번에서 시작하여 n 번까지 k 번 점프로 이동할 수 있는 경우의 수 구하기.
        - 토끼는 한 번 점프할 때 양의 정수 거리만큼 점프할 수 있으며, 다음 점프 시, 이전 점프 거리보다 작아야 한다. (최소 점프 거리는 1 이상)
        - 결과값이 커질 수 있으니 1000000007 로 나눈 나머지를 반환한다 (?)

    Debugging
        k=3, n=9
            6+3+1, 4+3+2
        k=1  ->  n >= 1
        k=2  ->  n >= 3
        k=3  ->  n >= 6
    Steps
        1. find properties which may be affected from next result value (the number of cases)
            - elements of multi-dimensional array must be one of array of the number of cases
            - properties:
                destination <n>, <k> jump
                point after a jump
                jump distance in last jump
                the number of cases
        2. convert proper process
            = The number of combinations that select k out of 1 to n to make n".
                but it requires sums of previous <first n elements> in order to propagate next state.
            - it may requires `dp[Comb(first n elements, k), sum ≤ n]`; 3D array for dynamic programming.
        3. guess Recurrence relation
            🚣 C is combinations(first n elements, k subset)
            a.
                dp[C(0, 0), 0] = 1
                dp[C(1, 0), 0] = 1
                ...
            b.
                dp[C(n, k), *] includes dp[C(n-1, k), *] in according to Recursion rules.
                dp[C(n, k), *] includes dp[C(n-1, k-1) with "n"] if C(n-1, k-1) with "n"  <= n
                = It can uses Sliding Window approach.
            c. Sliding Window approach
                <n> is greater or equal than <k>. Therefore, a one-dimensional space reduction of length <n> is more efficient than a one-dimensional space reduction of length <k>.
                    so dp table will be [[[0] * n for _ in (0, 1)] for _ in range(k+1)]
                But in according to following condition,
                    `Minimum distance in k jump  :=  a + a+d + a+2d + ... jump  = n*(n+1)/2 ;  Arithmetic sequence`
                    🚣 in order to use for-loop faster than while-loop while only calculate required indices
                    , dp table will be [[[0] * n for _ in (0, 1)] for _ in range(n+1)].
                    otherwise, it require additional memory of the list for memoization of sums of 1 ~ k.
    """
    if input_lines:
        input_ = lambda: next(input_lines)

    # Title: input
    n, k = map(int, input_().split())

    # Title: solve
    dp: list[list[list[int]]] = [[[0] * (n + 1) for _ in (0, 1)] for _ in range(n + 1)]
    for nn in range(n + 1):
        dp[nn][0][0] = 1
    for kk in range(1, k + 1):
        # move pointer of k
        kki: int = kk & 1
        pkki: int = kki ^ 1  # previous i
        dp[kk - 1][kki] = [0] * (n + 1)
        for nn in range(kk, n + 1):
            pnn: int = nn - 1
            dp[nn][kki] = [
                count + dp[pnn][pkki][psum] if psum >= 0 else count
                for psum, count in enumerate(dp[pnn][kki], start=-nn)
            ]
        #     print(f"nn {nn}, kk {kk}")
        #     pprint(dp[nn][kki])
        # print()

    # Title: output
    # pprint(dp)
    result: str = str(dp[n][k & 1][n] % 1000000007)
    print(result)
    return result


def test_rabbit_jump() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "3 2",  #  1+2
            ],
            ["1"],
        ],
        [
            [
                "4 3",
            ],
            ["0"],
        ],
        [
            [
                "9 3",  # 6+2+1, 5+3+1, 4+3+2
            ],
            ["3"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            rabbit_jump(iter(input_lines)),
            "\n".join(output_lines),
        )
        print(f"elapsed time: {time.time() - start_time}")


test_rabbit_jump()


def balance_list_through_swap(input_lines: Optional[Iterator[str]] = None) -> str:
    """❓
    고유한 양의 정수들이 들어있는 리스트에서, 인접한 정수의 차이가 k 이하가 되도록 최소 swap 횟수 구하기
    e.g. [10, 30, 20, 40] -> k = 10일 때, 30과 20을 바꿔서 1 번.

    정렬+DP 관계된거같은데.. 손도 못대겠다.
    """
    if input_lines:
        input_ = lambda: next(input_lines)

    # Title: input
    # Title: solve
    # Title: output


def test_balance_list_through_swap() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "4 10",  #  n=4, k=10
                "10 30 20 40",
            ],
            ["1"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            balance_list_through_swap(iter(input_lines)),
            "\n".join(output_lines),
        )
        print(f"elapsed time: {time.time() - start_time}")


# test_balance_list_through_swap()
