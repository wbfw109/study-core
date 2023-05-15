from __future__ import annotations

import time
import unittest
from typing import Iterator, Optional


def run_acm_craft(input_lines: Optional[Iterator[str]] = None) -> str:
    """get Minimum elapsed seconds to build target ; https://www.acmicpc.net/problem/1005

    TODO: Topologial sorting 따로 하고 나서 정리하기. (Complexity(BFS)) strongly polynomial time.

    It can be Connected or Forest shape. so It is reasonable to explore from <target_building>
    target building 부터 거꾸로 탐색하여 답을 찾는 것이 가능한 것 처럼 보이지만 그렇지 않다.
    거꾸로 탐색하는 도중 in-degree 개수가 0이 될 때만 큐에 추가하려 하지만
    , 중간 노드 x 가 target building 에 건설시간에 영향을 주지 않는 노드를 in-degree 로 두고 있다면 이를 알 방법이 없어 무시가 불가능하다.
    A -> (B, C) -> D
    A -> C -> E

    """
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    test_cases_answer: list[str] = []
    for _ in range(int(input_())):
        # Title: input
        # condition: (2 ≤  n; the number of buildings  ≤ 10^3)
        # condition: (1 ≤  k; the number of order rules between buildings  ≤ 10^5)
        n, k = map(int, input_().split())
        # condition: (1 ≤  (X, Y); building names,  W; target building ≤ N)
        # condition: (0 ≤ D_i ≤ 10^5)
        building_seconds: list[int] = [0]
        building_seconds.extend(map(int, input_().split()))
        cumulated_seconds: list[int] = [0] * (n + 1)

        # <remained_in_degree> can be used as traces
        remained_in_degrees: list[int] = [0] * (n + 1)
        edges: list[list[int]] = [[] for _ in range(n + 1)]
        for _ in range(k):
            line: list[int] = list(map(int, input_().split()))
            edges[line[0]].append(line[1])
            remained_in_degrees[line[1]] += 1
        target_building: int = int(input_())

        # Title: solve
        explored_deque: deque[int] = deque()
        for start_node, in_degree in enumerate(remained_in_degrees):
            if in_degree == 0:
                cumulated_seconds[start_node] = building_seconds[start_node]
                explored_deque.append(start_node)

        while explored_deque:
            explored_building = explored_deque.popleft()
            if edges[explored_building]:
                for next_building in edges[explored_building]:
                    remained_in_degrees[next_building] -= 1
                    cumulated_seconds[next_building] = max(
                        cumulated_seconds[next_building],
                        cumulated_seconds[explored_building]
                        + building_seconds[next_building],
                    )
                    if remained_in_degrees[next_building] == 0:
                        explored_deque.append(next_building)

        test_cases_answer.append(str(cumulated_seconds[target_building]))

    # Title: output
    result: str = "\n".join(test_cases_answer)
    print(result)
    return result


def test_run_acm_craft() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "2",
                "4 4",
                "10 1 100 10",
                "1 2",
                "1 3",
                "2 4",
                "3 4",
                "4",
                "8 8",
                "10 20 1 5 8 7 1 43",
                "1 2",
                "1 3",
                "2 4",
                "2 5",
                "3 6",
                "5 7",
                "6 7",
                "7 8",
                "7",
            ],
            ["120", "39"],
        ]
    ]:
        start_time = time.time()
        test_case.assertEqual(
            run_acm_craft(iter(input_lines)),
            "\n".join(output_lines),
        )
        print(f"elapsed time: {time.time() - start_time}")


def make_az_dictionary(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/1256
    알파벳 순서대로 기록되어있다...
    만들 수 있는 조합
        a 와 z가 있으면 z 1개를 가장 앞으로 빼기까지의 경우의 수를 구하는 것이 핵심.
    a1; 1
        a
    z1; 1
        z
    a1z1; 2
        a+z    z+a
    a1z2;
        a+zz    z+az z+za
    a2z1; 3
        a+az a+za   z+aa
    a2z2;
        a+azz a+zaz a+zza   z+aaz z+aza z+zaa

    if i >= 1 and j >= 1
        A[i][j] = A[i-1][j] + A[i][j-1]
    if i == 0 or j == 0
        A[i][j] = 1

    누적 개수를 저장할 곳 테이블 1개
    N("a") * N("z")
      0 1 2 3
    0 0 1 1 1
    1 1 2 3 4
    2 1 3 6
    3 1
    홀수짝수 판별 is_odd: bool = i & 1
    """
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (1 ≤  (n, m); the number of "a" and "z"  ≤ 10^2)
    # condition: (1 ≤  k  ≤ 10^9)
    n, m, k = map(int, input_().split())

    # Title: solve
    found_word: list[str] = []
    result: str = "-1"
    # <az_dictionary> have the number of words to make up by using given "a", "z".
    az_dictionary = [[1] * (m + 1) for _ in range(n + 1)]
    az_dictionary[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            az_dictionary[i][j] = az_dictionary[i - 1][j] + az_dictionary[i][j - 1]

    if k <= az_dictionary[-1][-1]:
        kk: int = k
        i: int = n
        j: int = m
        while i > 0 and j > 0:
            if kk <= az_dictionary[i - 1][j]:
                found_word.append("a")
                i -= 1
            else:
                # set <kk> as relative index
                kk -= az_dictionary[i - 1][j]
                found_word.append("z")
                j -= 1
        else:
            if i == 0:
                found_word.append("z" * j)
            else:
                found_word.append("a" * i)

        result = "".join(found_word)

    # Title: output
    print(result)
    return result


def test_make_az_dictionary() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "2 2 2",
            ],
            ["azaz"],
        ],
        [
            [
                "2 2 6",
            ],
            ["zzaa"],
        ],
        [
            [
                "10 10 1000000000",
            ],
            ["-1"],
        ],
        [
            [
                "7 4 47",
            ],
            ["aaazazaazaz"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            make_az_dictionary(iter(input_lines)),
            "\n".join(output_lines),
        )
        print(f"elapsed time: {time.time() - start_time}")
