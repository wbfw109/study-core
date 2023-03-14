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

    정렬 범위 나누기.... Meet-in-the-middle attack 과 이어짐

    - using bisect is inefficient in this problem because elements once searched is not used in later loop.
    다하고서 오답노트.
    # dataclass 사용하기?.. 객체 중심으로 접근?
    이전꺼 보고 필요하면 Counter 쓰기

    정렬을 사용하면 이 문제에서 어떤 이점을 만들 수 있는지 찾아내기

    정렬된것의 특징을 문제에서 어떻게 살릴것인지가 포인트
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

    n .. 섬 개수가 너무 커서 set 로 놓기에도 좀..

    A B C
    BFS.. 인데 시뮬레이션.. combination 이 없는?

    갈수있는 경로탐색부터.. dictionary 로 만들기 다리?
    순환되는 케이스를 버려야 한다.
    같은 v1 to v2 다리에 중량제한이 다른게 있으면 최대중량만 저장한다...
        Adjacency matrix + Symmetric matrix (대칭행렬) 로.. 하려햇는데 100000*100000 행렬이 최대 생성되서 안될듯..
        양방향 처리 방법이 이거도 잇지만.. 너무 비효율적이다.  두개 리스트 만드는것.. 흠..
        데이터클래스 딱히 필요없을듯? 무게 하나만 표시하면되서.
        - Used data structure: deque with Adjacency matrix in BFS.

    근데 단순 BFS 면 메모리초과날거같음.
    .. 감이 안온다... 이진탐색을 대체 어떻게 사용해야하는건지
    #  공장이 있는 두 섬을 연결하는 경로는 항상 존재하는 데이터만 입력으로 주어진다.
        으로부터 크루스칼도 사용가능한듯? 시작하는 다리개수만큼 경우의수가 주어진 경우 + 도착지 만나면 종료

    시작점과 끝점이 정해져있어서 시작점에서 이어진 섬 개수만큼 시뮬레이션하고,
    (시작점 끝점이 동일한 다리는 최대중량의 다리만 사용.)
    크루스칼 알고리즘으로 시작점을 루트로 두고 있는 집합에 끝점이 들어오면 마지막 다리의 중량제한이 정답이 될듯 보임.

    하지만 바이너리 서치 문제이므로 매개변수 검색 문제로 풀어봐야겟다..

    양방향이라는거를 어떻게 해석하지

    이분 탐색 + bfs를 이용한 파라메트릭 서치..

    큐에 10000개 정도들어가는거는 상관이 없나..
    메모리 초과가 안나나보다 흠..
    BFS 사용하기에.. 날줄알안슨데 안나나보네

    # predicate

    일단 양방향으

    중복된 다리 어떻게 처리할거냐..
    파라메트릯 ㅓ치에서 문제에서 구하는 값이 BFS 의 가능한 경로의 수가 아니고
    ★ , mid 값이 가능한지를 보는것이기 떄문에 통과한 다리의 개수가 다르더라도 한번 방문한 곳에 대해서 visited 를 확인하고
    해당 브릿지는 큐에 넣지 않아도 된다.

    <install_home_routers> 과 다르게
    출발지에서 이동가능한 다리에 대해 최대중량으로 초기화해줄 수는 있지만, max 를 개수만큼 해야해서
    최대 10만개만큼 비교해야 한다. 차라리 MAX_WEIGHT 으로 초기화하는 것이 좋음.
    log_2 로 취해보면 30 이하이기 떄문에 훨씬 빠름.

    BFS할떄 도달가능한 경로의 수를 세는 용도 visit count 누적 + heapq

    정렬 써야할듯. 너무 느리다. break가 필요할듯보임.
    만약 도달가능한 경로의 개수가 필요없고 도달가능한지만 확인하는 문제라면 is_explored 를 단순 set 구성 가능.

    """
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    class FoundPath(Exception):
        pass

    # Title: input
    # condition: (2 ≤ island count ≤ 10^4), (1 ≤ M ≤ 10^6)
    island_count, m = map(int, input_().split())

    # 0, 1 index are not used.
    map_: list[list[tuple[int, int]]] = [[]] * (island_count + 1)
    # condition: (1 ≤ two vertexes ≤ island count), (1 ≤ weight limit ≤ 10^9)
    for _ in range(m):
        island_1, island_2, weight_limit = map(int, input_().split())
        map_[island_1].append((island_2, weight_limit))
        map_[island_2].append((island_1, weight_limit))
    start_island, end_island = map(int, input_().split())
    maximum_weight_limit: int = 1

    # Title: solve
    for i in range(1, island_count + 1):
        map_[i].sort(key=lambda x: -x[1])

    # <endpoints>: minimum and maximum weight limits
    endpoints: list[int] = [1, 10**9]

    while True:
        # Test algorithm
        mid_weight_limit: int = (endpoints[0] + endpoints[1]) // 2

        is_exploration_success: bool = False
        explored_deque: deque[int] = deque([start_island])

        are_visited: list[bool] = [False] * (island_count + 1)
        are_visited[start_island] = True
        # is_visited_set: set[int] = set([start_island])
        try:
            while explored_deque:
                explored_island: int = explored_deque.popleft()
                for dest_island, weight_limit in map_[explored_island]:
                    if mid_weight_limit <= weight_limit:
                        if not are_visited[dest_island]:
                            if dest_island == end_island:
                                raise FoundPath
                            are_visited[dest_island] = True
                            explored_deque.append(dest_island)
                    else:
                        break
        except FoundPath:
            is_exploration_success = True

        # Decision algorithm (predicate)
        if is_exploration_success:
            endpoints[0] = mid_weight_limit + 1
            if endpoints[0] > endpoints[1]:
                maximum_weight_limit = mid_weight_limit
                break
        else:
            endpoints[1] = mid_weight_limit - 1

    # Title: output
    print(maximum_weight_limit)
    return str(maximum_weight_limit)


def test_drive_with_valid_weight() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "3 3",
                "1 2 2",
                "3 1 3",
                "2 3 2",
                "1 3",
            ],
            ["3"],
        ]
    ]:
        start_time = time.time()
        test_case.assertEqual(
            drive_with_valid_weight(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


test_drive_with_valid_weight()
