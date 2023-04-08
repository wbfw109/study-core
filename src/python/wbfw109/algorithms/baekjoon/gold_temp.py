from __future__ import annotations

import time
import unittest
from pprint import pprint
from typing import Iterator, Optional


def make_az_dictionary(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/1256
    알파벳 순서대로 기록되어있다...
    a*3 z*3 에 대해서 최소한 하나씩 나온다.
    0+1; aaazzz     ; 1 + (1+2)*3
        앞에 aaa 3개를 고정해놓고 뒤에서 변경할수있는 거를 확인하며 루프.
    1+1*3; aazazz aazzaz aazzza
        앞에 aa+za 로 시작. 뒤에서 변경할수있는거를 확인하며 루프.
        a 를 하나씩 옮겨서 끝에 도달.
        z 뒤에 [azz] 를 사용해서 3개 조합하는것으로 볼 수 있음.
    4+2*3; azaazz azazaz azazza . azzaaz azzaza azzzaa
        앞에 a+zaa 로 시작. 뒤에서 변경할수있는거를 확인하며 루프.
        a 를 하나씩 옮겨서 끝에 도달.
        뒤의 aazz 를 하나로 보고서 이전꺼와 똑같이 적용이 가능함.

    ~ 10+
    1; zaaazz   ; 1 + (1+2)*2. aaazz 로부터 나올 수 있는 경우의수
    1+1*2; zaazaz zaazza
    3+2*2; zazaaz ...

    ~ 10+7+
    1; zzaaaz ; 1 + (1+2)*1 aaaz 로부터 나올 수 있는 경우의수
    1+1*3; zzaaza ...

    ~ 10+7+5+  ; 1 + 1+2)*0. aaa 로부터 나올수 있는 경우의 수.
    1; zzzaaa

    aaazzz 에서 z 가 고정될떄까지
        각 내부루프에서 나올 수 있는 것의 개수가
        고정되지 않은 z 개수
        z 가 고정되기 전까지(가상 z 를 -1 에 두고 시작. 사이 거리 -1)
        multiplication operand = 내부루프횟수 partial sum * 초기의 가상 z (-1)이 아닌 고정되지않은 z의 개수
        내부루프횟수 partial sum = (루프 시작 전 첫 z 와 다음 z 사이의 거리 -1) = 1+2+3...
            내부루프횟수 == 주어진 a 의 개수 -1 과 동일
    모든 z를 가장 앞에까지 보내기까지의 가장 외부 루프가 z 개수만큼 반복되는건데

    전체 루프가 z의 개수만큼 반복된다. 시작할떄는 +1씩.. 처음 자기 위치표현. z가 고정됫을떄의 위치.
    z 를 맨 처음까지 고정시키고 다음 aaazz 에 대해서 반복하는 그런 것.
    a 등장 이후 z 가 처음발견되는 곳에서부터 마지막위치까지. a의 개수 * z 의 개수만큼 나온다.
        즉 z 처음 발견 위치부터 끝까지의 길이에서 한쪽은 a 개수만큼 뺸 것 * a의 개수
        없으면

    가장 앞의 z 의 위치에 대한 포인터를 초기화하고, 이 z가 index 0 이 될떄가지 루프
    해당 과정이 끝나면 z 의 위치를 새로운 포

    누적합+문자열+추적 문제인가
    z 의 처음 위치부터 마지막 z의 위치 사이에 a의 개수*그 팩토리얼?


    진짜로 사전을 만들 필요는 없을것?같기도 규칙성 찾아서 해당 문자열만 만들기?
    리스트로 만들어서 편집하고 "".join 으로 사전 만들기.

    O: 각 처음 시작 문자열을 보고 이 루프에서 나올 수 있는 개수를 알면 좋을거같은데
    그럼 여기에서 얻어내야 하는 문자열을 보고 어떤 문자열인지 예측하는 방법?

    사실 모든 경우의 수가 다 a z를 앞으로 보내는것으로 (deque 로 치면 pop 한거를 appendleft 한 것)
        이는 deque 가

    한 번 루프 돌 때마다. 현재 루프를 모두 돌았을 때, z 포인터 가장 앞에 고정. (추가) 하고
    테스트할 z 경우의 수 -1.
    내부에서는 각 z 포인터의 위치에 따라 현재 수가 달라진다.
        남은 z의 포인터가 처음 뺴고 1 z 개수를 루프만큼 계속 더해가면서 구해야 하는 범위 안에 있는지 확인한다.
        확인되면 해당 위치의 내부 루프?? 를 진행. 범위안 a 의 개수 * z 으로 "Z" 가 몇인지 확인한다.
        이를 무한루프로 반복하다ㅁ보면 좁혀질듯.
    """
    import dataclasses
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline


# test_make_az_dictionary()


def run_acm_craft(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/1005
    중복없는 방향성 트리. 노드->노드 순으로 배열하되
    역순으로 탐지하고, 걸린시간을 탐색별로 합쳐서 종료될떄까지. 진행해서 가장 작은 시간초 출력.
    """
    import dataclasses
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline
    # Title: input


def pack_in_normal_backpack(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/12865
    음.. 무게 대비 가치를 비교해야하나.. 가방 하나에 싸는 문제라서..
    예전것과 비교..
    eary stopping: K < backpack.Weight
    무게 대비 가치를 비교해야하나.. ㄴㄴ.. 에외 있음.
    NP-complete
    K=10
    weight  value
    7       12
    3       5
    5       11
    5       9

    짐을 쪼갤 수 없기 때문에 다이나믹 프로그래밍(DP)로 해결해야한다.
    어차피 이전 i 들에 대해 모두 순환하니까 상관ㅇ벗음

    """
    import dataclasses
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    @dataclasses.dataclass
    class Backpack:
        weight: int
        value: int

    # Title: input
    # condition: (1 ≤ N ≤ 10^2)
    # condition: (1 ≤ weight_limit ≤ 10^5)
    n, weight_limit = map(int, input_().split())
    # condition: (1 ≤ baccpack weight ≤ 10^5)
    # condition: (0 ≤ bkacpack value ≤ 10^3)
    backpacks: list[Backpack] = [
        Backpack(*map(int, input_().split())) for _ in range(n)
    ]
    backpacks.sort()


def test_pack_in_normal_backpack() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "4 7",
                "6 13",
                "4 8",
                "3 6",
                "5 12",
            ],
            ["14"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            pack_in_normal_backpack(iter(input_lines)), output_lines[0]
        )
        print(f"elapsed time: {time.time() - start_time}")


# test_pack_in_normal_backpack()


# week 4-1: graph theory: 5
def escape_maze(
    input_lines: Optional[Iterator[str]] = None,
) -> str:
    """get Minimum distance to escape maze ; https://www.acmicpc.net/problem/1194

    Time Complexity (Worst-case): O( BFS(maze) ) (but some routes with key subsets are added.)

    Space Complexity (Worst-case): O( BFS(maze) )

    Implementation
        - It uses Masking.
        - Things I've done in the implementation.
            - If I create new n*m boolean table whenever new key is obtained
                , it occurs "Memory Limit Exceeded".
            - When I use heapq instead of deque to eliminate some duplicated exploration that it occured because BFS still proceeds one by one
                , this rather causes "Timeout" because of time complexity of heapq.
                I had used max heap with bit count of <key_state>.

    A or operator B = B 이면 subset 인지 확인하는것이 비용이 subset problem 너무 크다.
    'a' key     'ab' keys

    """
    import operator
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    class FoundExit(Exception):
        pass

    DIRECTIONS: list[tuple[int, int]] = [
        (-1, 0),  # (-row direction)
        (1, 0),  # (+row direction)
        (0, -1),  # (-column direction)
        (0, 1),  # (+column direction)
    ]
    RouteState = tuple[int, tuple[int, int]]

    # Title: input
    # condition: (2 ≤  N, M  ≤ 5*10^1)
    n, m = map(int, input_().split())
    # n*m. ".": EMTPY.  "#": WALL.  "[a-f]": KEYS.  "[A-F]": DOORS corresponding KEYS
    # "0": START POINT. "1": EXIT
    maze: list[str] = []
    start_point: tuple[int, int] = (-1, -1)
    for row in range(n):
        line: str = input_()
        if (column := line.find("0")) != -1:
            start_point = (row, column)
        maze.append(line)

    # Title: solve
    # <explored_deque>: deque[<key_state>, point]. max heap about <key_bit_count>.
    explored_deques: list[deque[RouteState]] = [
        deque([(0, start_point)]),
        deque([]),
    ]
    # n*m dict[bit_count, list[key_state]]. <key_state> is mask.
    trace_map: list[list[dict[int, list[int]]]] = [
        [{} for _ in range(m)] for _ in range(n)
    ]
    trace_map[start_point[0]][start_point[1]][0] = [0]
    distance: int = 1
    minimum_distance: int = -1
    try:
        p: int = 0  # pointer which indicates current and next exploration deque.
        while explored_deques[p]:
            key_state, explored_point = explored_deques[p].popleft()
            key_bit_count = key_state.bit_count()

            for direction in DIRECTIONS:
                new_point: tuple[int, int] = tuple(
                    map(operator.add, explored_point, direction)
                )
                if (
                    0 <= new_point[0] < n
                    and 0 <= new_point[1] < m
                    and (c := maze[new_point[0]][new_point[1]]) != "#"
                ):
                    # is goal?
                    if c == "1":
                        raise FoundExit(distance)
                    elif c.isupper():
                        # can it go through the door? 65 is ord("A")
                        if key_state & 1 << (ord(c) - 65) == 0:
                            continue

                    # check duplication in trace
                    is_valid: bool = True
                    trace = trace_map[new_point[0]][new_point[1]]
                    for trace_bit_count, trace_subset in reversed(trace.items()):
                        if key_bit_count > trace_bit_count:
                            break
                        elif (
                            key_bit_count == trace_bit_count
                            and key_state in trace_subset
                        ):
                            is_valid = False
                            break
                        else:
                            if any(
                                (
                                    key_state | subset == subset
                                    for subset in trace_subset
                                )
                            ):
                                is_valid = False
                                break
                    if not is_valid:
                        continue

                    # update <new_key_state>
                    if c.islower():
                        # 97 is ord("a")
                        new_key_state = key_state | 1 << (ord(c) - 97)
                    else:
                        new_key_state = key_state
                    new_key_bit_count = new_key_state.bit_count()

                    # apply new trace
                    if new_key_bit_count in trace:
                        trace[new_key_bit_count].append(new_key_state)
                    else:
                        trace[new_key_bit_count] = [new_key_state]

                    explored_deques[p ^ 1].append((new_key_state, new_point))

            if len(explored_deques[p]) == 0:
                distance += 1
                p ^= 1
    except FoundExit as e:
        minimum_distance = e.args[0]

    # Title: output
    result: str = str(minimum_distance)
    print(result)
    return result


def test_escape_maze() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "1 7",
                "f0.F..1",
            ],
            ["7"],
        ],
        [
            [
                "5 5",
                "....1",
                "#1###",
                ".1.#0",
                "....A",
                ".1.#.",
            ],
            ["-1"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            escape_maze(iter(input_lines)),
            "\n".join(output_lines),
        )
        print(f"elapsed time: {time.time() - start_time}")


test_escape_maze()
