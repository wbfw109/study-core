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

    data structure 업데이트.. visualization root 포함되게, README 업데이트
    escape_marble_2 구슬만 움직이도록 해서 다시-
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

    정렬을 사용하긴 하지만 일정한 규칙을 적용할 수 있도력 정렬의 범위를 나누는 것이 핵심인듯.
        여기에서는 종유석과 석순을 나눔.
    5번째 문제 Meet-in-the-middle attack 과 연결됨.

    종유석, 석순 각 정렬하고-- 그 크기보다 큰거는 더 이상 탐색안하고 나머지 길이만 더하면 됨.


    정렬해서 해당 높이를 지난 것을 합하여 테이블에 저장해놓자.

    근데 모든 거를 다 봐야하나?
    조기 종료 시점.
    해시테이블을 만들 수는 있는데
    아래에서부터 첫번쨰 구간을 지나는거 구하려면 석순 최소 첫번쨰 길이까지만 찾고

    0-1-2-3 section
    .1.2.3. stalagmite
    .3.2.1. stalactites

    구간 (높이) 가 증가할꺠마다 stalagmite 는 감소하는 구조, 반대는 감소하는 흐름.
    H = 4 라면, 각 통과를 시도해볼 수 있는 구간이 4개가 나옴.
    앞에서부터 K (자연수) 번쨰 구간
    if k in [1, H]:
        the number of obstacles = n // 2
    else:
        the number of stalagmite that is >= k
        + the number of stalactites that is >= H-k+1

    구간 잘 나누고 초기값을 잘 잡아야할듯.... 루프에서 한번에 처리할 수 있도록

    겹치는게 많다면 bisect 가 나을 수 있음.
    bisect 를 사용해서 풀 수도 있지만, 찾을 구간이 탐색할떄마다 좁아지므로 굳이 필요가 없을 듯 보임.
    그리고 이 문제는 어차피 최솟값 뿐 아니라 그러한 구간의 수까지 구해야 하기 떄문에 다 돌아야할듯보임.

    2차원 그림에서 컴퓨터로는 구간을 타이핑해서 디버깅하기 어려우니 가로로 배치해서 생각해볼것.
    짝수 받는게 뭇느 의미지

    h 만큼 루프를 도는게 아니라 마지막 종유석, 석순의 높이를 확인하고 점핑할 수 있게?

    for 은 순차 접근 방식에서 좋다. i 를 2 이상 한번에 점프하는 경우가 없을 때. 속도도 더 빠름.
    그래서 attempt_section 은 while 로, 종유석 석순에 대한 루프는 for 로 구성함.

    숫자만 세는거 counter
    """
    import sys
    from collections import Counter

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (2 ≤  (N, H)  < 2*10^5).
    # condition: N is always even number.
    # condition: (1 ≤ size of obstacle < H)
    stalagmites: list[int] = []
    stalactites: list[int] = []
    n, h = map(int, input_().split())
    half_of_stones: int = n // 2
    for _ in range(half_of_stones):
        stalagmites.append(int(input_()))
        stalactites.append(int(input_()))
    minimum_collision_section: int = 0
    minimum_collision_count: int = 0

    # Title: solve
    # reverse=True so that <stalagmite_i> could indicates cumulated obstacles' count.
    stalagmites.sort(reverse=True)
    # reverse=True so that stalactites can be processed with stalagmite together in order.
    stalactites.sort(reverse=True)

    stalagmite_i: int = half_of_stones - 1
    stalactite_i: int = 0
    collision_counter: Counter[int] = Counter()
    attempt_section: int = 0
    while True:
        next_attempt_section: int = h
        for _ in range(stalagmite_i, -1, -1):
            if stalagmites[stalagmite_i] == attempt_section:
                stalagmite_i -= 1
            else:
                next_attempt_section = stalagmites[stalagmite_i]
                break

        for _ in range(stalactite_i, half_of_stones):
            if (y := h - stalactites[stalactite_i]) == attempt_section:
                stalactite_i += 1
            else:
                if y < next_attempt_section:
                    next_attempt_section = y
                break

        # skip sections whose the number of collisions is same.
        collision_counter[stalagmite_i + 1 + stalactite_i] += (
            next_attempt_section - attempt_section
        )

        # early stopping when stalagmites, stalactites are exhausted
        if next_attempt_section == h:
            break
        else:
            attempt_section = next_attempt_section

    minimum_collision_section = min(collision_counter)
    minimum_collision_count = collision_counter[minimum_collision_section]

    # Title: output
    print(minimum_collision_section, minimum_collision_count)
    return " ".join(map(str, [minimum_collision_section, minimum_collision_count]))


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
