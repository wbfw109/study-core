from __future__ import annotations

import time
import unittest
from pprint import pprint
from typing import Iterator, Optional

# 중량 제한 ; https://www.acmicpc.net/problem/1939
# 공유기 설치 ; https://www.acmicpc.net/problem/2110
# 개똥벌레 ; https://www.acmicpc.net/problem/3020

# data structure 업데이트.. visualization root 포함되게
# escape_marble_2 구슬만 움직이도록 해서 다시-


def hang_balloons_to_teams(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/4716

    우선순위가 주어짐. 힙 문제?
    기회비용 비교?
    sorted 대신 sort 쓰기


    """

    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition (1 ≤ N < 1000)
    n: int = int(input_())
    # condition (1 ≤ each weight of weights ≤ 10^6)
    weights: list[int] = list(map(int, input_().split()))
    # condition (1 ≤ weights to be measured ≤ n)
    not_found_weight: int = 1
    measurable_weight: int = 0

    # Title: solve
    weights.sort()

    # Title: output
    print(not_found_weight)
    return str(not_found_weight)


def test_hang_balloons_to_teams() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "3 15 35",
                "10 20 10",
                "10 10 30",
                "10 40 10",
                "0 0 0",
            ],
            ["300"],
        ]
    ]:
        start_time = time.time()
        test_case.assertEqual(
            hang_balloons_to_teams(iter(input_lines)), output_lines[0]
        )
        print(f"elapsed time: {time.time() - start_time}")


test_hang_balloons_to_teams()
