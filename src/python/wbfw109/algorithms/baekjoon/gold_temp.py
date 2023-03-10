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
    """🔍 get Minimum distance to hang balloons to teams ; https://www.acmicpc.net/problem/4716

    벨만포드 알고리즘? MCMF  알고리즘?
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

    ... 알수가없음..

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
        distance_1: int
        distance_2: int
        required_balloons: int

    # Title: input
    # condition (1 ≤ N < 1000)
    # condition (0 ≤  (A, B) balloons  < 10^4)
    # 🚣 condition (Σ(required_balloons) ≤ A+B)
    n, a, b = list(map(int, input_().split()))
    given_balloons: list[int] = [a, b]
    # condition (0 ≤  distance apart from (A, B)  ≤ 10^3)
    a_heapq: list[Team] = []
    b_heapq: list[Team] = []
    # why n+1? ignore last line ("0 0 0")
    for _ in range(n + 1):
        line: list[int] = list(map(int, input_().split()))

        # if required balloons is none, it is not need to process.
        if line[0] == 0:
            continue

        if line[1] < line[2]:
            a_heapq.append(Team(line[1], -line[2], required_balloons=line[0]))
        else:
            b_heapq.append(Team(line[2], -line[1], required_balloons=line[0]))
    minimum_sum_of_distance: int = 0

    # Title: solve
    heapq.heapify(a_heapq)
    heapq.heapify(b_heapq)
    a_b_pointer: int = 0
    heapq_list: list[list[Team]] = [a_heapq, b_heapq]

    # this loop must be run at least one time according to given conditions.
    while True:
        match bool(a_heapq), bool(b_heapq):
            case True, True:
                if a_heapq[0].distance_1 < b_heapq[0].distance_1:
                    target_team: Team = a_heapq[0]
                    a_b_pointer = 0
                else:
                    target_team: Team = b_heapq[0]
                    a_b_pointer = 1
            case False, True:
                target_team: Team = b_heapq[0]
                a_b_pointer = 1
            case True, False:
                target_team: Team = a_heapq[0]
                a_b_pointer = 0
            case _:
                break

        if given_balloons[a_b_pointer] != 0:
            used_balloons = min(
                given_balloons[a_b_pointer], target_team.required_balloons
            )
            given_balloons[a_b_pointer] -= used_balloons
            target_team.required_balloons -= used_balloons
            minimum_sum_of_distance += used_balloons * target_team.distance_1
            if target_team.required_balloons == 0:
                heapq.heappop(heapq_list[a_b_pointer])
        else:
            minimum_sum_of_distance += sum(
                (
                    -team.distance_2 * team.required_balloons
                    for team in heapq_list[a_b_pointer]
                )
            )
            a_b_pointer = (a_b_pointer + 1) % 2
            minimum_sum_of_distance += sum(
                (
                    team.distance_1 * team.required_balloons
                    for team in heapq_list[a_b_pointer]
                )
            )
            break

    # Title: output
    print(minimum_sum_of_distance)
    return str(minimum_sum_of_distance)


# 우선순위가 변위가 큰 것 abs(a-b)
def test_hang_balloons_to_teams() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "3 15 50",
                "10 1 2",
                "10 2 3",
                "10 4 1000",
                "0 0 0",
            ],
            ["30"],
        ],
        # [
        #     [
        #         "3 15 35",
        #         "10 20 10",
        #         "10 10 30",
        #         "10 40 10",
        #         "0 0 0",
        #     ],
        #     ["300"],
        # ],
        # [
        #     [
        #         "4 25 25",
        #         "10 20 10",
        #         "10 10 30",
        #         "10 30 15",
        #         "10 40 20",
        #         "0 0 0",
        #     ],
        #     ["650"],
        # ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            hang_balloons_to_teams(iter(input_lines)), output_lines[0]
        )
        print(f"elapsed time: {time.time() - start_time}")


test_hang_balloons_to_teams()
