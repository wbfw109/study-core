from __future__ import annotations

import time
import unittest
from pprint import pprint
from typing import Iterator, Optional


def sum_subsequences_2(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/1208
    3SUM 에 대한 일반화를 해봤는데 DFS 와 같이 경우의 수가 너무 많이 생겨서 N 이 커지면 처리가 너무 느려진다.
    square root decomposition

    시간복잡도가 왜 시간복잡도 O(2^n) 왜 지수승..
    현재 원소가 들어가있는 이전에 만들어진 부분수열을 포함한 부분수열의 수.. 저울 참고..

    Meet-in-the-middle attack


    다하고서 오답노트.
    # dataclass 사용하기?.. 객체 중심으로 접근?

    """
    import math
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition (1 ≤ N < 40)
    # condition (1 ≤ <target_sum> < 10^6)
    n, target_sum = map(int, input_().split())
    m = n // 2
    n = n - m
    sequence: list[int] = list(map(int, input_().split()))
    target_subsequences_count = 0
    control_var_i: deque[int] = deque()
    control_var_sum: int = 0
    consecutive_elements_sum: int = 0

    # Title: solve

    sequence.sort()

    # when the number of element of subsequence == 1,
    for e in sequence:
        if e == target_sum:
            target_subsequences_count += 1
        elif e > target_sum:
            break

    # when the number of element of subsequence >= 2,
    while len(control_var_i) + 2 <= n:

        inner_i: int = control_var_i[0] + 1 if control_var_i else 0
        inner_j: int = n - 1

        while inner_i < inner_j:
            temp_sum: int = control_var_sum + sequence[inner_i] + sequence[inner_j]
            if temp_sum == target_sum:
                # check elements with duplicated value
                # print([sequence[i] for i in [*control_var_i, inner_i, inner_j]])

                left_count = right_count = 1
                while inner_i < inner_j and sequence[inner_i] == sequence[inner_i + 1]:
                    inner_i += 1
                    left_count += 1
                while inner_i < inner_j and sequence[inner_j] == sequence[inner_j - 1]:
                    inner_j -= 1
                    right_count += 1

                # if elements of inner_i ~ right_i are same.
                if inner_i == inner_j and right_count == 1:
                    target_subsequences_count += math.comb(left_count, 2)
                else:
                    target_subsequences_count += left_count * right_count

                inner_i += 1
                inner_j -= 1
            elif temp_sum < target_sum:
                inner_i += 1
            else:
                inner_j -= 1

        # modify pointer when <control_var_i>s exist (n >= 3)
        for i in range(len(control_var_i)):
            control_var_sum -= sequence[control_var_i[i]]
            # if a pointer not exceeds valid range
            if control_var_i[i] + 1 != n - 2 - i:
                control_var_i[i] += 1
                control_var_sum += sequence[control_var_i[i]]
                for depth, ii in enumerate(range(i - 1, -1, -1), start=1):
                    control_var_i[ii] = control_var_i[i] + depth
                    control_var_sum += sequence[control_var_i[ii]]
                break
        else:
            # when combinations that can be made up with the number of <control_var_i> ends.
            previous_length: int = len(control_var_i)
            control_var_i.clear()
            control_var_i.extendleft(deque(range(previous_length + 1)))
            consecutive_elements_sum += sequence[len(control_var_i) - 1]
            control_var_sum = consecutive_elements_sum

    print(target_subsequences_count)
    return str(target_subsequences_count)


def test_sum_subsequences_2() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "32 1000",
                "100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100",
            ],
            ["129024480"],  # 5C1 ~ 5C5  =  5, 10, 10, 5, 1
        ],
        [
            [
                "5 0",
                "-7 -3 -2 5 8",
            ],
            ["1"],
        ],
        [
            [
                "5 0",
                "0 0 0 0 0",
            ],
            ["31"],  # 5C1 ~ 5C5  =  5, 10, 10, 5, 1
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(sum_subsequences_2(iter(input_lines)), output_lines[0])
        print(f"elapsed time: {time.time() - start_time}")


# test_sum_subsequences_2()

# 가장 긴 증가하는 부분 수열 2 주어진 조건에 따라 (정수여야 함) 조화수열은 아니다.
def get_longest_increasing_subsequence_2(
    input_lines: Optional[Iterator[str]] = None,
) -> str:
    """https://www.acmicpc.net/problem/12015
    수열의 정의..
    최장 증가 부분 수열
    여러 개의 LIS 가 나올 수잇음. nlogn 짜리 알고리즘이 있다.
    1차원 배열의 그래프 탐색과 비슷한듯.

    """
    pass


def drive_with_valid_weight(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/1939
    BFS 중량제한...
    넓게 생각하자...
    경로 탐색 = 노드/방향에 따라 움직이면 BFS 고려해보기?
    sorted... than sort

    """
    pass


def set_up_home_routers(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/2110
    3SUM 의 변환.
    중복되는거 제거하는 게 없음.

    공유기 어려워보임..
    """
    pass


def move_straight_in_cave(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/3020
    중앙값에서부터 양방향 탐색

    종유석, 석순 각 정렬하고-- 그 크기보다 큰거는 더 이상 탐색안하고 나머지 길이만 더하면 됨.


    data structure 업데이트.. visualization root 포함되게
    escape_marble_2 구슬만 움직이도록 해서 다시-
    정렬해서 해당 높이를 지난 것을 합하여 테이블에 저장해놓자.
    """
    pass


def test_move_straight_in_cave() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "6 7",
                "1",
                "5",
                "3",
                "3",
                "5",
                "1",
            ],
            ["2 3"],
        ],
        [
            [
                "14 5",
                "1",
                "3",
                "4",
                "2",
                "2",
                "4",
                "3",
                "4",
                "3",
                "3",
                "3",
                "2",
                "3",
                "3",
            ],
            ["7 2"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            move_straight_in_cave(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


test_move_straight_in_cave()
