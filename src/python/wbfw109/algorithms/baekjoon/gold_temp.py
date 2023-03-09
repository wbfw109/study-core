from __future__ import annotations

import time
import unittest
from pprint import pprint
from typing import Iterator, Optional


def mix_three_solutions(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/2473

    두 용액 합쳣을떄처럼 적절한 범위 내에서 경우의 수 다 확인해봐야함.
    A .. B  ... C

    맞나? A C 는 하나씩 움직인다 치면, 이진탐색 될거같은데
    가운데꺼 계산해보고 abs(sum) 이 이전꺼보다 더 작을떄까지만.

    Inner loop
        assume that B starts from A+1,
        abs(new_sum) in B loop
        if abs(new_sum) < abs(sum):
            update sum as new_sum

        if new_sum > 0:
            break inner loop (B-- since I starts with B from A+1, impossible so break.)
        else:
            B++
    
    Outer loop
        A 를 다음꺼로 옮겨야할지 C를 이전꺼로 옮겨야할지 뭐부터?
        마지막으로 계산한 용액의 합이 0 미만이면 A+1
        아니라면 C-1
        겹치지 않게.

    third_i 를 처음부터 이진탐색해서 결정하기.

    escape_marble_2 구슬만 움직이도록 해서 다시-
    tuple 덧셈 operator.add map 으로 바꾸기-?
    
    A ........B  C
    sum() < 0:
    =================
    
    """
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition (3 ≤ N < 5000)
    n: int = int(input_())
    # condition (-(10^9) ≤ each number of solution ≤ 10^9)
    # negative integer is acid solution, positive integer is alkaline solution.
    solutions = list(map(int, input_().split()))
    zero_closest_solutions_i: tuple[int, int, int] = (-1, -1, -1)
    zero_closest_abs: int = sys.maxsize

    # Title: solve
    solutions.sort()
    left_i: int = 0
    right_i: int = n - 1
    third_i: int = -1
    while left_i < right_i:
        while left_i < third_i < right_i:
            # binary search
        
        temp_sum: int = sum((solutions[i] for i in [left_i, third_i, right_i]))

        if (new_abs := abs(temp_sum)) < zero_closest_abs:
            zero_closest_solutions_i = (left_i, third_i, right_i)

        if temp_sum < 0:
            left_i += 1
        else:
            right_i -= 1


def test_mix_three_solutions() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "5",
                "-2 6 -97 -6 98",
            ],
            ["-97 -2 98"],
        ],
        [
            [
                "7",
                "-2 -3 -24 -6 98 100 61",
            ],
            ["-6 -3 -2"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(mix_three_solutions(iter(input_lines)), output_lines[0])
        print(f"elapsed time: {time.time() - start_time}")


test_mix_three_solutions()
