from typing import Iterator, Optional


def hang_balloons_to_teams(input_lines: Optional[Iterator[str]] = None) -> str:

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
