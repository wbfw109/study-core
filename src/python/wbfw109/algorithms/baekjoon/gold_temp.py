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
    독립변인.. 테스트를 잘해야한다..
    A .. B  ... C
    B 를 움직이고
    A -1

    It is similar <mix_two_solutions>.
    even if <third_i>
    binary search can be used on third-pointer.
    bu

    음.. 차례대로 왼쪽 -1 오른쪽 -1 하면 비는 경우가 생김...
    0 1 2 3 4
    L=0, R=4
    L=1, R=4
    L=1, R=3    -> L= 0, R=3 일 경우가 그렇다..
    이런경우라면 차라리 for 루프를 돌리자.
    이진 검색이 의미가 있나?.. 약간 있긴 하다 . 왜냐하면 "절대값"이 최소가 되는 것을 구해야 하기 때문에
    i, j 의 중첩 루프 안에서도 j 가 변화한다면 k 의 위치가 왼쪽 또는 오른쪽으로 변할 수가 있다.
    최악의 경우 n*n*log n 이 되는데,
    +1씩 비교하며 접근하면, n*2 이 된다.

    sum (generator) 보다 이미 있는 리스트라면 인덱스로 접근해서 직접 더하는게 더 빠른가?
    ㅇㅇ 맞다.. 유의미한 차이를 보인다.. 통과 못햇음.. generator 도 결국 그만큼 순환해서 생성해야 하므로..



    """
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
    zero_closest_solutions_i: tuple[int, int, int] = (0, 1, 2)
    zero_closest_abs: int = sys.maxsize

    # Title: solve
    solutions.sort()
    try:
        for i in range(n - 2):
            j = i + 1
            k = n - 1

            while j < k:
                temp_sum: int = solutions[i] + solutions[j] + solutions[k]
                if (new_abs := abs(temp_sum)) < zero_closest_abs:
                    zero_closest_solutions_i = (i, j, k)
                    zero_closest_abs = new_abs
                    if temp_sum == 0:
                        raise TargetFound

                # 'if temp_sum == 0' is evaluated in upper expressions.
                if temp_sum < 0:
                    j += 1
                else:
                    k -= 1
    except TargetFound:
        pass

    result_as_str = " ".join(
        map(
            str,
            ((solutions[i] for i in zero_closest_solutions_i)),
        )
    )

    # Title: output
    print(result_as_str)
    return result_as_str


def test_mix_three_solutions() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        # [
        #     [
        #         "4",
        #         "1 2 3 -1",
        #     ],
        #     ["-1 1 2"],
        # ],
        [
            [
                "5",
                "-99 -100 -100 -105 -100",
            ],
            ["-100 -100 -99"],
        ],
        # [
        #     [
        #         "6",
        #         "-104 239 997 627 722 -942",
        #     ],
        #     ["-942 239 722"],
        # ],
        # [
        #     [
        #         "10",
        #         "254336095 47691541 257341582 -144645454 861485597 33299316 -291023334 -255047743 -645353494 329443014",
        #     ],
        #     ["-291023334 33299316 257341582"],
        # ],
    ]:
        start_time = time.time()
        test_case.assertEqual(mix_three_solutions(iter(input_lines)), output_lines[0])
        print(f"elapsed time: {time.time() - start_time}")


test_mix_three_solutions()
