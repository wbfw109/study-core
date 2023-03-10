from __future__ import annotations

import time
import unittest
from pprint import pprint
from typing import Iterator, Optional


def mix_three_solutions(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/2473


    data structure 업데이트.. visualization root 포함되게
    escape_marble_2 구슬만 움직이도록 해서 다시-
    두 용액 합쳣을떄처럼 적절한 범위 내에서 경우의 수 다 확인해봐야함.
    A .. B  ... C
    B 를 움직이고

    It is similar but in this case, binary search can be used on third-pointer.

    맞나? A C 는 하나씩 움직인다 치면, 이진탐색 될거같은데
    가운데꺼 계산해보고 abs(sum) 이 이전꺼보다 더 작을떄까지만.
    속도가 더 빨라지긴하는데 이거는 다른 모듈에서

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


    A x......m..y
        means that A is negative and abs(A) is similar with m.
        that means, it just need to check additionally two solutions: one of A ~ m, one of m ~ y
    sum() < 0:
    A .B`......B
        means that  B` ~ B value is

    이진탐색.. 음.. 모듈화?할까? predicate 만 달라지도록
    def binary_search(predicate):


    애초에 두 용액 합치는거도.. 0... n 있으면
    n 을 2진탐색하면서 구해보고, 최종위치까지 구해보고,
    =================

    """
    import math
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    class TargetFound(Exception):
        pass

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
    # outer i
    left_i: int = 0
    right_i: int = n - 1
    # inner j
    left_j: int = 1
    right_j: int = n - 2
    try:
        while left_i < right_i:
            while left_j != right_j:
                # binary search
                middle_j: int = math.ceil((left_j + right_j) / 2)

                temp_sum: int = sum((solutions[x] for x in [left_i, middle_j, right_i]))
                if (new_abs := abs(temp_sum)) < zero_closest_abs:
                    zero_closest_solutions_i = (left_i, middle_j, right_i)
                    if new_abs == 0:
                        raise TargetFound

                if temp_sum > 0:
                    right_j = middle_j - 1
                else:
                    left_j = middle_j
            else:
                temp_sum: int = sum((solutions[x] for x in [left_i, left_j, right_i]))
                if (new_abs := abs(temp_sum)) < zero_closest_abs:
                    zero_closest_solutions_i = (left_i, left_j, right_i)
                    if new_abs == 0:
                        raise TargetFound

    except TargetFound:
        pass


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


# test_mix_three_solutions()


def mix_two_solutions(input_lines: Optional[Iterator[str]] = None) -> str:
    """get the Zero-closest sum of two solution ; https://www.acmicpc.net/problem/2470

    assume that A is control variable (unchanged)
    A ...aa..bb..B
    A 에서 찾은게 최적의 bb
    B 에서 찾은 최적의 aa

    A 는 aa 전까지만 하면 되고, Y 는 bb 전까지만 하면 된다?

    means that A is negative and abs(A) is similar with m.
    that means, it just need to check additionally two solutions: one of A ~ m, one of m ~ y
    sum() < 0:
    A .B`......B
        means that  B` ~ B value is
    언제 break 해야함?

    """
    import math
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition (2 ≤ N ≤ 100,000)
    n: int = int(input_())
    # condition (-(10^9) ≤ each number of solution ≤ 10^9)
    # negative integer is acid solution, positive integer is alkaline solution.
    solutions = list(map(int, input_().split()))
    zero_closest_solutions_indexes: tuple[int, int] = (-1, -1)
    zero_closest_abs = sys.maxsize

    # Title: solve
    solutions.sort()
    left_i: int = 0
    right_i: int = n - 1

    def predicate_found(i: int, j: int) -> Optional[int]:
        nonlocal zero_closest_abs, zero_closest_solutions_indexes
        temp_sum: int = solutions[i] + solutions[j]
        print("temp_sum", temp_sum, "i,j", i, j)
        if (new_abs := abs(temp_sum)) < zero_closest_abs:
            zero_closest_abs = new_abs
            zero_closest_solutions_indexes = (i, j)
            if temp_sum == 0:
                return None
        return temp_sum

    def binary_search(left_j: int, right_j: int, fixed_i: int) -> bool:
        while left_j <= right_j:
            middle_j = math.ceil((left_j + right_j) / 2)
            print("middle_j (", fixed_i, middle_j, ")")
            temp_sum = predicate_found(fixed_i, middle_j)
            if not temp_sum:
                return True

            # if temp_sum == 0:  is already checked upper expressions
            if temp_sum < 0:
                left_j = middle_j + 1
            else:
                right_j = middle_j - 1
        return False

    indexing_result_1 = binary_search(left_i + 1, right_i, fixed_i=left_i)
    left_i_limit: int = zero_closest_solutions_indexes[1]
    zero_closest_abs = sys.maxsize
    indexing_result_2 = binary_search(left_i, right_i - 1, fixed_i=right_i)
    right_i_limit: int = zero_closest_solutions_indexes[1]
    print("limit", left_i_limit, right_i_limit)
    if not indexing_result_1 and not indexing_result_2:
        are_remained: list[bool] = [left_i <= left_i_limit, right_i >= right_i_limit]
        print(left_i, left_i_limit, right_i, right_i_limit)
        while any(are_remained):
            if are_remained[0]:
                if left_i + 1 <= left_i_limit:
                    left_i += 1
                    if predicate_found(left_i, right_i):
                        break
                else:
                    are_remained[0] = False
            if are_remained[1]:
                if right_i - 1 >= right_i_limit:
                    right_i += 1
                    if predicate_found(left_i, right_i):
                        break
                else:
                    are_remained[1] = False

    result_as_str = " ".join(
        map(
            str,
            (
                solutions[zero_closest_solutions_indexes[0]],
                solutions[zero_closest_solutions_indexes[1]],
            ),
        )
    )

    # Title: output
    print(result_as_str)
    return result_as_str
