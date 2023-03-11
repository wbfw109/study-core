from __future__ import annotations

import time
import unittest
from pprint import pprint
from typing import Iterator, Optional


def drive_with_valid_weight(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/1939"""
    pass


def set_up_home_routers(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/2110"""
    pass


def move_straight_in_cave(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/3020
    중앙값에서부터 양방향 탐색


    data structure 업데이트.. visualization root 포함되게
    escape_marble_2 구슬만 움직이도록 해서 다시-
    """
    pass


def hang_balloons_to_teams(input_lines: Optional[Iterator[str]] = None) -> str:
    """🔍 get Minimum distance to hang balloons to teams ; https://www.acmicpc.net/problem/4716

    우선순위가 주어짐. 힙 문제?
    기회비용 비교?
    sorted 대신 sort 쓰기

    ??? >> 풍선은 한 가지 색상을 여러 개 달아준다고 가정한다. 무슨 의미 ..
        달아줘야 하는 풍선 색상이 중요하다면 해당 색을 한 팀이 선택한 후 부족하면 다른 팀은 다른 풍선 색상을 선택해야 한다.
    그리디같음. 가중치가 뭐가 우선이 되야 할까..
    한쪽이 선택하면 다른 선택지가 없어지는거를 해결ㅎ야하는듯.


    =====
    n A  B
    -----
    3 25 25
    10 20 10        # 1 team
    10 10 30        # 2 team
    10 30 15        # 3 team
    10 40 20        # 4 team
    0 0 0           # ignore this
    = 650 (?)
    -----
    When:
        Two teams with same distance apart from B; 1, 2 teams that should select B balloons if possible.
            , and 1 team is closer than 3 team apart from A.
        So precedence of B balloons are owned to 1 team.
            but 1 team can not obtain all required B balloons because it is not enough.
            After obtain, the side effect of having to choose A balloons is exists for remained teams that should select B balloons if possible.
                It is unavoidable.
        the counterpart is likewise.
        Result of the cases that the either queue (A or B) is prioritized than another queue is different.
    -> use Two heapq to save opportunity cost.

    여기서 기회비용은 abs(A 와 떨어진 거리 - B 와 떨어진 거리) 이다.

    //''';;;각 테스트케이스에 대해서...

    """
    import dataclasses
    import heapq
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    @dataclasses.dataclass(eq=True, order=True)
    class Team:
        distance_a: int
        distance_b: int
        required_balloons: int

    # Title: input
    # condition (1 ≤ N < 1000)
    # condition (0 ≤  (A, B) balloons  < 10^4)
    # 🚣 condition (Σ(required_balloons) ≤ A+B)
    n, a, b = list(map(int, input_().split()))
    given_balloons: list[int] = [a, b]

    # condition (0 ≤  distance apart from (A, B)  ≤ 10^3)
    team_heapq: list[tuple[int, Team]] = []
    for _ in range(n):
        line: list[int] = list(map(int, input_().split()))

        # if required balloons is none, it is not need to process.
        if line[0] == 0:
            continue

        team_heapq.append(
            (-abs(line[1] - line[2]), Team(line[1], line[2], required_balloons=line[0]))
        )
    minimum_sum_of_distance: int = 0

    # Title: solve
    a_b_pointer: int = -1
    heapq.heapify(team_heapq)

    # this loop must be run at least one time according to given conditions.
    while team_heapq:
        _, team = team_heapq[0]
        if team.distance_a < team.distance_b:
            a_b_pointer = 0
            is_a: bool = True
        else:
            a_b_pointer = 1
            is_a: bool = False

        if given_balloons[a_b_pointer] != 0:
            used_balloons = min(given_balloons[a_b_pointer], team.required_balloons)
            given_balloons[a_b_pointer] -= used_balloons
            team.required_balloons -= used_balloons
            if is_a:
                minimum_sum_of_distance += used_balloons * team.distance_a
            else:
                minimum_sum_of_distance += used_balloons * team.distance_b
            if team.required_balloons == 0:
                heapq.heappop(team_heapq)
        else:
            # reverse operation since previous <given_balloons[a_b_pointer]> is 0
            if is_a:
                minimum_sum_of_distance += sum(
                    (team.distance_b * team.required_balloons for _, team in team_heapq)
                )
            else:
                minimum_sum_of_distance += sum(
                    (team.distance_a * team.required_balloons for _, team in team_heapq)
                )
            break

    # Title: output
    print(minimum_sum_of_distance)
    return str(minimum_sum_of_distance)


def test_hang_balloons_to_teams() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "1 5 5",
                "10 2 1",
                "0 0 0",
            ],
            ["15"],
        ],
        [
            [
                "3 15 50",
                "10 1 2",  #  20   |  5+10
                "10 2 3",  # 10+15 |  30
                "10 4 1000",  # 40
                "0 0 0",
            ],
            ["85"],
        ],
        [
            [
                "3 15 35",
                "10 20 10",
                "10 10 30",
                "10 40 10",
                "0 0 0",
            ],
            ["300"],
        ],
        [
            [
                "4 25 25",
                "10 20 10",  # 50+100
                "10 10 30",  # 100
                "10 30 15",  # 150
                "10 40 20",  # 200
                "0 0 0",
            ],
            ["600"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            hang_balloons_to_teams(iter(input_lines)), output_lines[0]
        )
        print(f"elapsed time: {time.time() - start_time}")


test_hang_balloons_to_teams()
