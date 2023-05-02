from __future__ import annotations

import time
import unittest
from pprint import pprint
from typing import Iterator, Optional

# TODO: remove "❔" in beginning of docstring, create Complexity class
# TODO: use sys.stdout.write instead of print()
# TODO: re-solve <escape_maze>, <escape_marble_2>, gold_temp.py
# week 5-1: bitmask
# week 4-2: string


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
        - ❓ It seems that implementation using 3D trace_map by subset of keys is faster than current implementation to compare subset
            , because comparsing subsets (latter) is expensive than aloowing some duplicated exploration (former).
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
                        if key_state & 1 << ord(c) - 65 == 0:
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
                        new_key_state = key_state | 1 << ord(c) - 97
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


def get_diameter_in_tree(
    input_lines: Optional[Iterator[str]] = None,
) -> str:
    """get Tree's diameter ; https://www.acmicpc.net/problem/1167

    Time Complexity (Worst-case): O( BFS(1d matrix) ) (two times)
        - O(x) from max() function to obtain a tip node. x is the number of tips nodes. (leaf or root)

    Space Complexity (Worst-case): O( BFS(1d matrix) )

    Consideration:
        - 🚣 Do not trust completely sample test case. It not guarantees order of vertices input.
    """
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (2 ≤ N ≤ 10^5)
    n: int = int(input_())
    # condition: (1 ≤ each V ≤ 10^5)
    # <tree_graph_metric>: vertices[list[to another vertex, distance]]. 0 index is not used.
    tree_graph_metric: dict[int, list[tuple[int, int]]] = {}
    for _ in range(n):
        line: list[int] = list(map(int, input_().split()))
        edges_iterator: Iterator[int] = iter(line[1:-1])
        tree_graph_metric[line[0]] = list(zip(edges_iterator, edges_iterator))

    # Title: solve
    def find_farthest_v(v1: int) -> tuple[int, int]:
        """
        Returns:
            tuple[int, int]: discovered farthest vertex from <v1>, distance from the vertex from <v1>
        """
        cumulated_distance: int = 0
        # <explored_deques>: deque[vertex, cumulated distance]
        explored_deque: deque[tuple[int, int]] = deque([(v1, cumulated_distance)])
        trail_map: list[bool] = [False] * (n + 1)
        trail_map[v1] = True
        # <tip_vertices>: list[tuple[vertex, cumulated distance]]
        tip_vertices: list[tuple[int, int]] = []
        while explored_deque:
            explored_v, cumulated_distance = explored_deque.popleft()
            is_tip: bool = True
            for v2, distance in tree_graph_metric[explored_v]:
                if not trail_map[v2]:
                    is_tip = False
                    trail_map[v2] = True
                    explored_deque.append((v2, cumulated_distance + distance))

            if is_tip:
                tip_vertices.append((explored_v, cumulated_distance))

        # <tip_vertices[i][1]> is cumulated distance of <tip_vertices[i]>.
        max_distance_i = max(
            (i for i in range(len(tip_vertices))), key=lambda i: tip_vertices[i][1]
        )
        return tip_vertices[max_distance_i]

    # 1. Choose a random vertex
    v1: int = 1
    # 2. BFS that finds vertices that are as far from <v1> as possible
    v2, distance = find_farthest_v(v1)
    _, distance = find_farthest_v(v2)

    # Title: output
    result: str = str(distance)
    print(result)
    return result


def test_get_diameter_in_tree() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "5",
                "1 3 2 -1",
                "2 4 4 -1",
                "3 1 2 4 3 -1",
                "4 2 4 3 3 5 6 -1",
                "5 4 6 -1",
            ],
            ["11"],
        ],
        [
            [
                "5",
                "1 5 1 -1",
                "5 1 1 4 10 -1",
                "4 3 10 5 10 -1",
                "3 2 10 4 10 -1",
                "2 3 10 -1",
            ],
            ["31"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            get_diameter_in_tree(iter(input_lines)),
            "\n".join(output_lines),
        )
        print(f"elapsed time: {time.time() - start_time}")


def move_with_breaking_one_wall(input_lines: Optional[Iterator[str]] = None) -> str:
    """get Minimum distance to endpoint ; https://www.acmicpc.net/problem/2206

    Time Complexity (Worst-case): O( BFS(number_map) ) (but some routes with <has_broken_wall>=True are added.)

    Space Complexity (Worst-case): O( BFS(number_map) )

    Implementation
        - In the online test cases
            , this implementation finished 0.5 seconds faster than the implementation using one deque
            , and used 9.5MB less memory space.
    """
    import operator
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    DIRECTIONS: list[tuple[int, int]] = [
        (-1, 0),  # (-row direction)
        (1, 0),  # (+row direction)
        (0, -1),  # (-column direction)
        (0, 1),  # (+column direction)
    ]

    RouteState = tuple[int, bool, tuple[int, int]]

    class FoundEndPoint(Exception):
        pass

    # Title: input
    # condition: (1 ≤  N, M  ≤ 10^3)
    # condition: (1 ≤ weight_limit ≤ 10^5)
    n, m = map(int, input_().split())
    # n*m matrix. "0": movable cell.  "1": unmovable cell
    number_map: list[str] = [input_() for _ in range(n)]
    # <trace_map>'s elements are masking.
    # 00 (0): not visited.  01 (1): Normal trace.  10 (2): Another trace.  11 (3): both
    trace_map: list[list[int]] = [[0] * m for _ in range(n)]
    minimum_distance: int = -1

    # Title: solve
    # 0, 1 index will be used alternatively to distinguish verticies at same depth level.
    # list[deque[distance_from_base, has_broken_wall, point]]
    explored_deques: list[deque[RouteState]] = [
        deque([(1, False, (0, 0))]),
        deque(),
    ]
    trace_map[0][0] = 1
    try:
        ## check initial queue
        if (0, 0) == (n - 1, m - 1):
            raise FoundEndPoint(1)

        # next exploration level pointer: 0 or 1
        p: int = 0
        while explored_deques[p]:
            distance_from_base, has_broken_wall, point = explored_deques[p].popleft()

            for direction in DIRECTIONS:
                new_point: tuple[int, int] = tuple(map(operator.add, point, direction))

                if 0 <= new_point[0] < n and 0 <= new_point[1] < m:
                    new_distance_from_base = distance_from_base + 1
                    new_has_broken_wall = has_broken_wall
                    new_number = number_map[new_point[0]][new_point[1]]
                    new_trace = trace_map[new_point[0]][new_point[1]]

                    if new_has_broken_wall:
                        if new_number == "1" or new_trace != 0:
                            continue
                        # case 1: has_broken_wall and trace == 0
                    else:
                        if new_number == "1":
                            if new_trace != 0:
                                continue
                            else:
                                # case 2: not has_broken_wall and number == "1" and trace == 0
                                new_has_broken_wall = True
                        else:
                            if new_trace not in [0, 2]:
                                continue
                            # case 3 : not has_broken_wall and number == "0" and if trace in [0, 2]:
                            # even if route that is <new_has_broken_wall>=True is preceed, this may not pass throguh to goal point.

                    if new_point == (n - 1, m - 1):
                        raise FoundEndPoint(new_distance_from_base)

                    trace_map[new_point[0]][new_point[1]] = (
                        trace_map[new_point[0]][new_point[1]] | new_has_broken_wall + 1
                    )

                    x: RouteState = (
                        new_distance_from_base,
                        new_has_broken_wall,
                        new_point,
                    )
                    if new_has_broken_wall:
                        explored_deques[p ^ 1].append(x)
                    else:
                        explored_deques[p ^ 1].appendleft(x)
            # next exploration queue
            if not explored_deques[p]:
                p ^= 1
    except FoundEndPoint as e:
        minimum_distance = e.args[0]

    # Title: output
    result: str = str(minimum_distance)
    print(result)
    return result


def test_move_with_breaking_one_wall() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "1 1",
                "0",
            ],
            ["1"],
        ],
        [
            [
                "6 4",
                "0100",
                "1110",
                "1000",
                "0000",
                "0111",
                "0000",
            ],
            ["15"],
        ],
        [
            [
                "4 4",
                "0111",
                "1111",
                "1111",
                "1110",
            ],
            ["-1"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            move_with_breaking_one_wall(iter(input_lines)), output_lines[0]
        )
        print(f"elapsed time: {time.time() - start_time}")


def get_minimum_weight_to_all_vertices(
    input_lines: Optional[Iterator[str]] = None,
) -> str:
    """get Minimum weight to all vertiecs ; https://www.acmicpc.net/problem/1753
    🔍 다익스트라로 다시?
    
    Time Complexity (Worst-case): ...
        - O( N(edges) ) from selecting minimum weight of same edge points
        - O( BFS(directed_graph_weights) ) (but except for overlapped edges)
        - O(V * log n) from Hip (pop | push)

    Space Complexity (Worst-case): O( BFS(directed_graph_weights) )
    """
    import heapq
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (1 ≤ v_count ≤ 20,000,    1 ≤ e_count ≤ 300,000)
    v_count, e_count = map(int, input_().split())
    # condition: (1 ≤ start_v ≤ v_count)
    start_v: int = int(input_())
    # condition: (1 ≤ weight ≤ 10)
    # <directed_graph_weights>: v1[dict[v2, minimum weight]]]
    directed_graph_weights: list[dict[int, int]] = [{} for _ in range(v_count + 1)]
    for _ in range(e_count):
        v1, v2, weight = map(int, input_().split())
        if v2 in directed_graph_weights[v1].keys():
            directed_graph_weights[v1][v2] = min(directed_graph_weights[v1][v2], weight)
        else:
            directed_graph_weights[v1][v2] = weight

    # <min_weights> also could be used as <is_explored>. <min_weights> from <start point> to a vertex.
    min_weights: list[int | str] = ["INF"] * (v_count + 1)

    # Title: solve
    # <explored_heapq>: (cumulated_weight, v)
    explored_heapq: list[tuple[int, int]] = [(0, start_v)]
    while explored_heapq:
        cumulated_weight, explored_v = heapq.heappop(explored_heapq)
        # if already minimum weight to a vertex is modified by heapq
        if min_weights[explored_v] != "INF":
            continue
        min_weights[explored_v] = cumulated_weight

        for dest, weight in directed_graph_weights[explored_v].items():
            # if already a vertex is visited
            if min_weights[dest] == "INF":
                heapq.heappush(explored_heapq, (cumulated_weight + weight, dest))

    # Title: output
    result: str = "\n".join(map(str, min_weights[1:]))
    print(result)
    return result


def test_get_minimum_weight_to_all_vertices() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "5 6",
                "1",
                "5 1 1",
                "1 2 2",
                "1 3 3",
                "2 3 4",
                "2 4 5",
                "3 4 6",
            ],
            ["0", "2", "3", "7", "INF"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            get_minimum_weight_to_all_vertices(iter(input_lines)),
            "\n".join(output_lines),
        )
        print(f"elapsed time: {time.time() - start_time}")


def see_colors_as_red_green_color_blindness(
    input_lines: Optional[Iterator[str]] = None,
) -> str:
    """get Color zone count ; https://www.acmicpc.net/problem/10026

    Time Complexity (Worst-case): O( BFS(color_map) )

    Space Complexity (Worst-case): O( BFS(color_map) )
    """
    import operator
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    DIRECTIONS: list[tuple[int, int]] = [
        (-1, 0),  # (-row direction)
        (1, 0),  # (+row direction)
        (0, -1),  # (-column direction)
        (0, 1),  # (+column direction)
    ]

    # Title: input
    # condition: (1 ≤ N ≤ 100)
    n: int = int(input_())
    color_map: list[str] = [input_() for _ in range(n)]

    # Title: solve
    def get_color_zone_count(predicate: str) -> int:
        is_visited_color_map: list[list[bool]] = [[False] * n for _ in range(n)]
        count: int = 0
        for row in range(n):
            for column in range(n):
                if not is_visited_color_map[row][column]:
                    colors: str = color_map[row][column]
                    if colors in predicate:
                        colors = predicate
                    count += 1

                    # BFS
                    explored_deque: deque[tuple[int, int]] = deque([(row, column)])
                    is_visited_color_map[row][column] = True
                    while explored_deque:
                        explored_point = explored_deque.popleft()
                        for direction in DIRECTIONS:
                            new_point: tuple[int, int] = tuple(
                                map(operator.add, explored_point, direction)
                            )
                            if (
                                0 <= new_point[0] < n
                                and 0 <= new_point[1] < n
                                and not is_visited_color_map[new_point[0]][new_point[1]]
                                and color_map[new_point[0]][new_point[1]] in colors
                            ):
                                explored_deque.append(new_point)
                                is_visited_color_map[new_point[0]][new_point[1]] = True
        return count

    count_as_normal: int = get_color_zone_count(predicate="")
    count_as_blidness: int = get_color_zone_count(predicate="RG")

    # Title: output
    result: str = " ".join(map(str, [count_as_normal, count_as_blidness]))
    print(result)
    return result


def test_see_colors_as_red_green_color_blindness() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "5",
                "RRRBB",
                "GGBBB",
                "BBBRR",
                "BBRRR",
                "RRRRR",
            ],
            ["4 3"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            see_colors_as_red_green_color_blindness(iter(input_lines)), output_lines[0]
        )
        print(f"elapsed time: {time.time() - start_time}")


# week 3-2: dynamic programming
def solve_travelling_salesman_problem(
    input_lines: Optional[Iterator[str]] = None,
) -> str:
    """solve Travelling salesman problem ; https://www.acmicpc.net/problem/2098

    Time Complexity (Worst-case): Θ(2^n * n^2) from "main algorithm" part

    Space Complexity (Worst-case): Θ(2^n * n) from <min_cycle_weights>
    """
    import itertools
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (2 ≤ N ≤ 16)
    n = int(input_())
    # weight: (0 ≤ weight in edges ≤ 10^6). 0 means INFINITY.
    edge_weights: list[list[int | float]] = []
    for _ in range(n):
        line: list[int | float] = list((map(int, input_().split())))
        for i, x in enumerate(line):
            if x == 0:  # given condition
                line[i] = float("infinity")
        edge_weights.append(line)

    # Title: solve
    # 0 ≤  route range  < <route_limit> - 1
    route_limit: int = 1 << n - 1
    min_cycle_weights: list[list[int | float]] = [
        [float("infinity")] * n for _ in range(route_limit)
    ]

    def set_min_cycle_weights(route: int, dest: int) -> None:
        for previous_dest in [v for v in range(1, n) if 1 << v - 1 & route != 0]:
            divided_route = route & ~(1 << previous_dest - 1)
            new_distance = (
                min_cycle_weights[divided_route][previous_dest]
                + edge_weights[previous_dest][dest]
            )
            if new_distance < min_cycle_weights[route][dest]:
                min_cycle_weights[route][dest] = new_distance

    route_dict_by_v_count: dict[int, list[int]] = {
        v_count: [] for v_count in range(1, n - 1)
    }
    for route in range(1, route_limit - 1):
        route_dict_by_v_count[route.bit_count()].append(route)

    # main algorithm
    for v in range(1, n):
        min_cycle_weights[0][v] = edge_weights[0][v]
    for route in itertools.chain.from_iterable(route_dict_by_v_count.values()):
        for dest in [v for v in range(1, n) if 1 << v - 1 & route == 0]:
            set_min_cycle_weights(route=route, dest=dest)
    set_min_cycle_weights(route=route_limit - 1, dest=0)

    # Title: output
    result: str = str(min_cycle_weights[route_limit - 1][0])
    print(result)
    return result


def test_solve_travelling_salesman_problem() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "4",
                "0 10 15 20",
                "5 0 9 10",
                "6 13 0 12",
                "8 8 9 0",
            ],
            ["35"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            solve_travelling_salesman_problem(iter(input_lines)), output_lines[0]
        )
        print(f"elapsed time: {time.time() - start_time}")


def get_longest_length_of_bitonic_like_subsequence(
    input_lines: Optional[Iterator[str]] = None,
) -> str:
    """get longest length of bitonic-like subsequence ; https://www.acmicpc.net/problem/11054

    Time Complexity (Worst-case): O(n(log n))
        - O(n(log n)) from binary search to find suitable length of increasing subsequence for sequence[i]. (and from reversed thing)
        - O(n) from counting the length of increasing subsequence for each element of sequence. (and from reversed thing)

    Space Complexity (Worst-case): O(n) from auxiliary data structures

    Consideration
        - Actual bitonic sequence is defined as x_0 ≤ ... ≤ x_k ≥ ... ≥ x_(n-1) for some (k, 0 ≤ k < n).
            but this problem define that as x_0 < ... < x_k > ... > x_(n-1) for some (k, 0 ≤ k < n).
            so an algorithm for longest increasing sequences can be applied to this problem.
    """
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (1 ≤ N ≤ 10^6)
    n: int = int(input_())
    # condition: (-(10^9) ≤ element of sequence ≤ 10^9)
    sequence: list[int] = list(map(int, input_().split()))
    longest_bitonic_subsequence_l: int = 0

    # Title: solve
    smallest_indexes_at_l: list[int] = [0] * (n + 1)
    smallest_indexes_at_l[0] = -1
    increasing_subsequence_lengths = [0] * n
    found_subsequence_l: int = 0

    for i in range(len(sequence)):
        low = 1
        high = found_subsequence_l + 1

        while low < high:
            mid = low + (high - low) // 2
            if sequence[smallest_indexes_at_l[mid]] >= sequence[i]:
                high = mid
            else:
                low = mid + 1

        new_l = low
        smallest_indexes_at_l[new_l] = i
        if new_l > found_subsequence_l:
            found_subsequence_l = new_l

        increasing_subsequence_lengths[i] = new_l

    # iterate in reverse. similar with upper algorithm.
    smallest_indexes_at_l_2: list[int] = [0] * (n + 1)
    smallest_indexes_at_l_2[0] = -1
    increasing_subsequence_lengths_2 = [0] * n
    found_subsequence_l_2: int = 0

    for i in range(len(sequence) - 1, -1, -1):
        low = 1
        high = found_subsequence_l_2 + 1

        while low < high:
            mid = low + (high - low) // 2
            if sequence[smallest_indexes_at_l_2[mid]] >= sequence[i]:
                high = mid
            else:
                low = mid + 1

        new_l = low
        smallest_indexes_at_l_2[new_l] = i
        if new_l > found_subsequence_l_2:
            found_subsequence_l_2 = new_l

        increasing_subsequence_lengths_2[i] = new_l

    longest_bitonic_subsequence_l = (
        max(
            (
                a + b
                for a, b in zip(
                    increasing_subsequence_lengths, increasing_subsequence_lengths_2
                )
            )
        )
        - 1
    )

    # Title: output
    result: str = str(longest_bitonic_subsequence_l)
    print(result)
    return result


def test_get_longest_length_of_bitonic_like_subsequence() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "10",
                "1 5 2 1 4 3 4 5 2 1",
            ],
            ["7"],
        ]
    ]:
        start_time = time.time()
        test_case.assertEqual(
            get_longest_length_of_bitonic_like_subsequence(iter(input_lines)),
            "\n".join(output_lines),
        )
        print(f"elapsed time: {time.time() - start_time}")


def pack_in_normal_backpack(input_lines: Optional[Iterator[str]] = None) -> str:
    """solve 0-1 knapsack problem ; https://www.acmicpc.net/problem/12865

    Time Complexity (Worst-case): O(n*W) (pseudo-polynomial time. NP-complete)

    Space Complexity (Worst-case): O(n*W)
    """
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (1 ≤  n; the number of distinct items  ≤ 10^2)
    # condition: (1 ≤ knapsack_capacity ≤ 10^5)
    n, knapsack_capacity = map(int, input_().split())
    # condition: (1 ≤ item weight ≤ 10^5)
    # condition: (0 ≤ item value ≤ 10^3)
    # items[tuple[weight, value]] are assumed to store all relevant values starting at index 1.
    items: list[tuple[int, int]] = [(0, 0)] + [
        tuple(map(int, input_().split())) for _ in range(n)
    ]

    # Title: solve
    # start with initializing each first line to 0: m[0][x], m[x][0] = 0
    m: list[list[int]] = [[0] * (knapsack_capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for weight_limit in range(1, knapsack_capacity + 1):
            if items[i][0] > weight_limit:  # <items[i][0]> is weight of the item
                m[i][weight_limit] = m[i - 1][weight_limit]
            else:
                m[i][weight_limit] = max(
                    m[i - 1][weight_limit],
                    m[i - 1][weight_limit - items[i][0]] + items[i][1],
                )
                # <items[i][0]> is value of the item

    # Title: output
    result: str = str(m[-1][-1])
    print(result)
    return result


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


# week 3-1: binary search: 4
def sum_subsequences_2(input_lines: Optional[Iterator[str]] = None) -> str:
    """get Number of cases of subsequences to be able to create target Sum ; https://www.acmicpc.net/problem/1208

    Time Complexity (Worst-case): O( 2^(n/2) * (n/2) )
        - O(k log_2(k)) from iterating subset and running binary search
        - O(k log_2(k)) from Tim sort

        As k is 2^(n/2), O(k log k) == O( 2^(n/2) * (n/2) * log_2(2) ) == O( 2^(n/2) * (n/2) )

    Space Complexity (Worst-case): O(2^(n/2)) from auxiliary data structures

    Definition
        - k := the number of made up all subset from (n // 2). == 2^(n/2)
    """
    import bisect
    import itertools
    import sys
    from typing import Callable

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition (1 ≤ N < 40)
    # condition (1 ≤ <target_sum> < 10^6)
    n, target_sum = map(int, input_().split())
    sequence: list[int] = list(map(int, input_().split()))
    total_count = 0

    # Title: solve
    n_halves: list[list[int]] = [sequence[: n // 2], sequence[n // 2 :]]
    sum_halves: list[list[int]] = [[], []]
    get_target_count: Callable[
        [list[int], int], int
    ] = lambda in_, target: bisect.bisect_right(in_, target) - bisect.bisect_left(
        in_, target
    )

    for n_half, sum_half in zip(n_halves, sum_halves):
        for i in range(1, len(n_half) + 1):
            sum_half.extend((sum(comb) for comb in itertools.combinations(n_half, i)))
        sum_half.sort()

    for sum_half in sum_halves:
        total_count += get_target_count(sum_half, target_sum)
    for s in sum_halves[0]:
        total_count += get_target_count(sum_halves[1], target_sum - s)

    # Title: output
    result: str = str(total_count)
    print(result)
    return result


def test_sum_subsequences_2() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "6 0",
                "-7 -3 -2 5 8 0",
            ],
            ["3"],
        ],
        [
            [
                "5 0",
                "0 0 0 0 0",
            ],
            ["31"],  # 5C1 ~ 5C5  =  5, 10, 10, 5, 1
        ],
        [
            [
                "32 1000",  # 21 "100" and 11 "-100"
                "100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100",
            ],
            ["129024480"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(sum_subsequences_2(iter(input_lines)), output_lines[0])
        print(f"elapsed time: {time.time() - start_time}")


def drive_with_valid_weight(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Maximum weight limit ; https://www.acmicpc.net/problem/1939

    Time Complexity (Worst-case): ...
        - O(log <maximum_weight_limit>) from Parametric search loop
                ; <maximum_weight_limit> was given 10^9
            * O( BFS(graph_metric) ) (but except overlapped edges)
        - O( N(bridges) ) from selecting maximum weight limit of same edge points

    Space Complexity (Worst-case): O( BFS(graph_metric) ) (bidirectional edges)

    Definition
        - Number( BFS(bridges) ): |V| + |E|; Time occurred to find valid paths
            - the number of valid islands including <start_island> are vertexes.
            - the number of valid bridges are edges.

    Consideration:
        - ❔ How to process "bidirectional edges"?
            - if it is processed in Adjacency matrix (Symmetric matrix)
                , It is expected to cause "Memory Limit Exceeded".
            - 🚣 instead append edges into two vertices as Adjancecy list.

    Implementation:
        - It can be thought as of a successor of problem <install_home_routers> (function)
        - Current implmentation is 0.25 seconds slower than without deduplication of bridge (; to select maximum weight_limit).
        - 🚣 It seem that It can be implemented by using Kruscal's algorithm.
            when disjoint set where <start_island> is root meets disjoint of <end_island>
            , weight limit of the lastly merged edge will be answer.

    Time Complexity (Worst-case): ...
        - O( N(edges) ) from selecting minimum weight of same edge points
        - O( BFS(graph_metric) ) (but except overlapped edges)

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
    # <graph_metric>: v1[dict[v2, maximum weight_limit]]]
    # condition: (1 ≤ island id vertexes ≤ island count), (1 ≤ weight limit ≤ 10^9)
    graph_metric: list[dict[int, int]] = [{} for _ in range(island_count + 1)]

    def set_maximum_weight(v1: int, v2: int, weight: int) -> None:
        if v2 in graph_metric[v1].keys():
            graph_metric[v1][v2] = max(graph_metric[v1][v2], weight)
        else:
            graph_metric[v1][v2] = weight

    for _ in range(m):
        island_1, island_2, weight_limit = map(int, input_().split())
        set_maximum_weight(island_1, island_2, weight_limit)
        set_maximum_weight(island_2, island_1, weight_limit)

    start_island, end_island = map(int, input_().split())
    maximum_weight_limit: int = 1

    # Title: solve
    # <endpoints>: minimum and maximum weight limits
    endpoints: list[int] = [1, 10**9]

    while endpoints[0] <= endpoints[1]:
        ## Test algorithm
        mid_weight_limit: int = endpoints[0] + (endpoints[1] - endpoints[0]) // 2  # type: ignore

        is_exploration_success: bool = False
        explored_deque: deque[int] = deque([start_island])
        is_visited_set: set[int] = set([start_island])
        try:
            while explored_deque:
                explored_island: int = explored_deque.popleft()
                for dest_island, weight_limit in graph_metric[explored_island].items():
                    if (
                        dest_island not in is_visited_set
                        and mid_weight_limit <= weight_limit
                    ):
                        if dest_island == end_island:
                            raise FoundPath
                        is_visited_set.add(dest_island)
                        explored_deque.append(dest_island)
        except FoundPath:
            is_exploration_success = True

        ## Decision algorithm (predicate)
        if is_exploration_success:
            endpoints[0] = mid_weight_limit + 1  # type: ignore
        else:
            endpoints[1] = mid_weight_limit - 1  # type: ignore
    else:
        maximum_weight_limit = endpoints[0] - 1

    # Title: output
    result: str = str(maximum_weight_limit)
    print(result)
    return result


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


def install_home_routers(input_lines: Optional[Iterator[str]] = None) -> str:
    """🚤 get Distance that maximizes the distance between all adjacent routers ; https://www.acmicpc.net/problem/2110

    Time Complexity (Worst-case): O(n(log^2 n))
        - O(n(log n)) from Tim sort
        - O(log n) from Parametric search loop (<mid_distance> is updated logarithmically)
            * O(n) from inner While loop
                but efficient since search range is narrower for each iteration from "lo=start_coord_i + 1".
            * O(log n) in inner loop from binary search

    Space Complexity (Worst-case): O(n) from Tim sort

    Implementation
        - It is efficient I uses binary search because target coordinate is always in routers list.
        - key point is to use Parametric search with Bisection method.
            - The structure can be largely divided into two categories
                : (Test algorithm, Decision algorithm (predicate))
    """
    import bisect
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (2 ≤ N ≤ 2*10^5)
    # condition: (2 ≤ home routers count ≤ N)
    # condition: (0 ≤ routers coordinate ≤ 10^9). home's coordinates do not overlap.
    n, routers_count = map(int, input_().split())
    routers_coordinates: list[int] = [int(input_()) for _ in range(n)]
    maximum_distance: int = 0

    # Title: solve
    routers_coordinates.sort()
    # endpoints are search range of distance to get Installable maximum distance.
    # Note that endpoints represent a range that has not yet been tested.
    # endpoints[0] is minimum Installable distance.
    # endpoints[1] is maximum distance between two routers.
    endpoints: list[int] = [1, routers_coordinates[-1] - routers_coordinates[0]]

    # Parametric search with Bisection method
    while endpoints[0] <= endpoints[1]:
        ## Test algorithm
        mid_distance: int = endpoints[0] + (endpoints[1] - endpoints[0]) // 2  # type: ignore
        start_coord_i: int = 0  # initial coordinate to install router.
        installed_router_count: int = 0
        # while if result of bisect.bisect_left() is not last of <start_coord_i>
        while start_coord_i < len(routers_coordinates):
            installed_router_count += 1

            # update next start coordinate
            start_coord_i = bisect.bisect_left(
                routers_coordinates,
                routers_coordinates[start_coord_i] + mid_distance,  # type: ignore
                lo=start_coord_i + 1,
            )

        ## Decision algorithm (predicate)
        if installed_router_count >= routers_count:
            # update minimum Installable distance if installed_router_count >= routers_count
            # current <mid_distance> is valid. but test is required for a longer distance. so set with "+1".
            endpoints[0] = mid_distance + 1  # type: ignore
        else:
            # update maximum Installable distance.
            # current <mid_distance> is not valid. so set with "-1".
            endpoints[1] = mid_distance - 1  # type: ignore
    else:
        # When root is found.
        maximum_distance = endpoints[0] - 1

    # Title: output
    result: str = str(maximum_distance)
    print(result)
    return result


def test_install_home_routers() -> None:
    """Debugging
    +: router counts ++
    *: next start point.


    =====
    5 3
    0 1 2 3 4   # coordinates. middle coordinate is 2
    +   *
        +   *
            +

    endpoints
    1 4
    3 4  # 2: O
    3 2  # 3: X
    3 2  # 2: O   end.
    """
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "5 3",
                "0",
                "1",
                "2",
                "3",
                "4",
            ],
            ["2"],
        ],
        [
            [
                "5 3",
                "1",
                "2",
                "8",
                "4",
                "9",
            ],
            ["3"],
        ],
        [
            [
                "3 2",
                "8",
                "9",
                "10",
            ],
            ["2"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            install_home_routers(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


def move_straight_in_cave(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Minimum collision count ; https://www.acmicpc.net/problem/3020

    Time Complexity (Worst-case): O(n(log n))
        - O(2 * n/2(log n/2)) from Tim sort on two stalagmites, stalactites list.
        - O(2* n/2) iteration from given stalagmites, stalactites.

    Space Complexity (Worst-case): O(n) from Tim sort

    Inductive reasoning
        =====
        N = 6, H = 4
        -----
        0 | 1 | 2 | 3  section (0-based numbering assuming that section 0 occupies height 0 ~ 1)
        1   2   3   -  stalagmite heights
        -   3   2   1  stalactites heights
        -----

        As the section number increases, the number of collisions on stalagmite decreases.
            initial value is n / 2 and last value is 0 in H section.
        As the section number increases, the number of collisions on stalactites increases.
            initial value is 0 and last value is n / 2 in H section.

        sections whose new a stalagmite or stalactite collided is not appeared can be skipped.

        key point is to use feature of partial sum.

    Implementation
        - According to given condition "N is always even number."
            , I distinguished stalagmites and stalactites once when input data.
        - I used collections.Counter() instead of storing all collision count into list.
            it will save memory footprint.
        - using bisect is inefficient in this problem because elements once searched is not used in later loop.
    """
    import sys
    from collections import Counter

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (2 ≤ N ≤ 2*10^5). N is always even number.
    # condition: (2 ≤ H ≤ 5*10^5). (1 ≤ height of obstacle < H)
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

        # early stop when stalagmites, stalactites are exhausted
        if next_attempt_section == h:
            break
        else:
            attempt_section = next_attempt_section

    minimum_collision_section = min(collision_counter)
    minimum_collision_count = collision_counter[minimum_collision_section]

    # Title: output
    result: str = " ".join(
        map(str, [minimum_collision_section, minimum_collision_count])
    )
    print(result)
    return result


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
    ]:
        start_time = time.time()
        test_case.assertEqual(
            move_straight_in_cave(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


# week 2-2: sorting: 5
def hang_balloons_to_teams(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Minimum distance to hang balloons to teams ; https://www.acmicpc.net/problem/4716

    Time Complexity (Worst-case): O(n(log n)) from Tim sort
        - O(n) loop from given teams.

    Space Complexity (Worst-case): O(n) from Tim sort

    Consideration
        - To compare Opportunity costs
            If team's balloons that have higher opportunity cost is processed in order
            , the number of remained (A, B) balloons will not matter.
            In this problem, opportunity cost is absolute value of difference between distances apart from A room and B room.

    Implementation
        - 🚣 key point is Memoization of <ab_pointer>.
            <ab_pointer> represents a pointer to the ballon that the team should prioritize among A, B balloons.
            <ab_pointer> is used in <distance_ab> and <remained_ab>.
        - I used <list>.sort() instead of sorted(<list>).
            - <list>.sort() is in-place operation.
            - sorted(<list>) returns new sorted object so that it causes overhead as much copy operation.
        - condition "0 0 0" input can be thought of as a first line input of each test cases.
    """
    import dataclasses
    import sys
    from typing import Literal

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    @dataclasses.dataclass(init=False, eq=True, order=True)
    class Team:
        required_balloons: int
        distance_ab: list[int]
        ab_pointer: Literal[0, 1]

        def __init__(self, *args: int):
            self.required_balloons = args[0]
            self.distance_ab = [args[1], args[2]]
            self.ab_pointer = 0 if self.distance_ab[0] <= self.distance_ab[1] else 1

    minimum_sum_of_distance_list: list[str] = []  # for local debugging

    # condition: when "0 0 0" is present, break loop.
    while (start_line := list(map(int, input_().split()))) != [0, 0, 0]:
        # Title: input
        # condition: (1 ≤ N ≤ 1000)
        n: int = start_line[0]
        # condition: (0 ≤  (A, B) balloons  < 10^4)
        remained_ab: list[int] = start_line[1:]

        # condition: (0 ≤  distance apart from (A, B)  ≤ 10^3)
        teams: list[Team] = [Team(*map(int, input_().split())) for _ in range(n)]
        teams.sort(key=lambda team: -abs(team.distance_ab[0] - team.distance_ab[1]))
        minimum_sum_of_distance: int = 0

        # Title: solve
        for i, team in enumerate(teams):
            used_balloons: int = (
                team.required_balloons
                if team.required_balloons < remained_ab[team.ab_pointer]
                else remained_ab[team.ab_pointer]
            )

            minimum_sum_of_distance += used_balloons * team.distance_ab[team.ab_pointer]
            remained_ab[team.ab_pointer] -= used_balloons

            # if remained A or B balloons are exhausted.
            if remained_ab[team.ab_pointer] == 0:
                team.required_balloons -= used_balloons
                fixed_ab_pointer: int = (team.ab_pointer + 1) % 2

                # 🚣 condition: (Σ(required_balloons) ≤ A+B)
                minimum_sum_of_distance += sum(
                    (
                        teams[j].required_balloons
                        * teams[j].distance_ab[fixed_ab_pointer]
                        for j in range(i, len(teams))
                    )
                )
                break

        # Title: output
        print(minimum_sum_of_distance)
        minimum_sum_of_distance_list.append(str(minimum_sum_of_distance))

    return "\n".join(minimum_sum_of_distance_list)


def test_hang_balloons_to_teams() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "1 5 5",  # First case
                "10 2 1",  #  2*5 + 1*5
                "3 15 50",  # Second case
                "10 1 2",  # 2. 1*5 + 2*5
                "10 2 3",  # 3. 3*10
                "10 4 1000",  # 1. 4*10
                "0 0 0",
            ],
            ["15", "85"],
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
            hang_balloons_to_teams(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


def weigh_weights_on_the_scales(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Maximum measurable weight by using weights ; https://www.acmicpc.net/problem/2437

    Time Complexity (Worst-case): O(n(log n)) from Tim sort
        - O(n) loop from given weights.

    Space Complexity (Worst-case): O(n) from Tim sort

    Inductive reasoning
        =====
        4
        1 2 4 9
        -----
        -> 1        -> 2, 3     -> 4, 5, 6, 7        -> [X] 9

        if Current maximum measurable weight + 1   >=   still not checked a Next weight:
            Current maximum measurable weight  +=  a Next weight
        else:
            Not found weight  =  Current maximum measurable weight +1
            break
        and, one exception: loop of all given weights is over,
            Not found weight  =  Current maximum measurable weight +1

    Implementation
        - Things I've done in the implementation which uses hash table as set type.
            In this case, always loop is performed as many length of measurable weights until before for each weight.
            namely, it's Recurrence Relation will be:
                when loop count = 1, Sum(1) = 1.
                when loop count > 1 , Sum(loop_count) = 1 + Sum(loop_count-1)
            Each loop's length's is 2^(loop_count-1).
            , and causes "Out of Memory" in the submit site.
        - 🚣 key point is reasoning with Inductive reasoning.
            (debug some cases, set temporarily hypothesis, find the rules)
            refer to Debugging of <test_weigh_weights_on_the_scales>
    """

    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (1 ≤ N ≤ 1000)
    n: int = int(input_())
    # condition: (1 ≤ each weight of weights ≤ 10^6)
    weights: list[int] = list(map(int, input_().split()))
    # condition: (1 ≤ weights to be measured ≤ n)
    not_found_weight: int = 1
    measurable_weight: int = 0

    # Title: solve
    weights.sort()
    for weight in weights:
        if measurable_weight + 1 >= weight:
            measurable_weight += weight
        else:
            not_found_weight = measurable_weight + 1
            break
    else:
        not_found_weight = measurable_weight + 1

    # Title: output
    result: str = str(not_found_weight)
    print(result)
    return result


def test_weigh_weights_on_the_scales() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "7",
                "3 1 6 2 7 30 1",
            ],
            ["21"],
        ],
        [
            [
                "5",
                "1 2 3 4 5",
            ],
            ["16"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            weigh_weights_on_the_scales(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


def mix_three_solutions(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get the Zero-closest sum of three solution ; https://www.acmicpc.net/problem/2473

    Time Complexity (Worst-case): O(n^2) from loop of variant of 3SUM problem
        - O(n(log n)) from Tim sort

    Space Complexity (Worst-case): O(n) from Tim sort

    Purpose
        - to compare absolute sums of two solution by explicitly control two index pointers with exclusive range.

    Consideration
        - It is almost similar with function <mix_two_solutions> (problem). refer to that.

    Implementation
        - when get the sum of three values, to use sum() functions is slower than a way of direct indexing access.
            This appears to be because sum() creates an iterator once every execution.
            It causes "Timeout" in the submit site.
    """
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    class TargetFound(Exception):
        pass

    # Title: input
    # condition: (3 ≤ N ≤ 5000)
    n: int = int(input_())
    # condition: (-(10^9) ≤ each number of solution ≤ 10^9)
    # negative integer is acid solution, positive integer is alkaline solution.
    solutions: list[int] = list(map(int, input_().split()))
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
        [
            [
                "4",
                "1 2 3 -1",
            ],
            ["-1 1 2"],
        ],
        [
            [
                "5",
                "-99 -100 -100 -105 -100",
            ],
            ["-100 -100 -99"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            mix_three_solutions(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


def bundle_up_numbers(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Maximum sum by optionally bundling up numbers ; https://www.acmicpc.net/problem/1744

    Time Complexity (Worst-case): O(n(log n)) from Tim sort

    Space Complexity (Worst-case): O(n) from Tim sort

    Purpose
        - to compare absolute sums of two solution by explicitly control two index pointers with exclusive range.

    Consideration
        - Multiplication of two negative integers is positive integer.              Good
        - Multiplication of two (negative, positive) integers is negative integer.  Bad
        - Multiplication of two positive integers is positive integer.              Good, but Exception exists
            if next operand is 1, adding current operand and process next loop is greater than just multiplication.
        - Multiplication of (negative integer, zero) is zero.                       Good
        - Multiplication of (positive integer, zero) is zero.                       Bad
    """
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (1 ≤ N < 50)
    n: int = int(input_())
    # condition: (-(10^3) ≤ sequence number < 10^3)
    sequences: list[int] = [int(input_()) for _ in range(n)]
    # condition: (maximum result < 2^31)
    maximum_sum: int = 0

    # Title: solve
    sequences.sort()
    left_i = 0
    right_i = n - 1

    while left_i < n and sequences[left_i] < 0:
        # could I bundle up two negative integer including the case that next element is zero?
        if left_i + 1 < n and sequences[left_i + 1] <= 0:
            maximum_sum += sequences[left_i] * sequences[left_i + 1]
            left_i += 2
        else:
            maximum_sum += sequences[left_i]
            left_i += 1
    while right_i >= 0 and sequences[right_i] > 0:
        # could I bundle up two positive integer except the case that next element is zero?
        if right_i - 1 >= 0 and sequences[right_i - 1] > 0:
            if sequences[right_i - 1] == 1:
                maximum_sum += sequences[right_i]
                right_i -= 1
            else:
                maximum_sum += sequences[right_i] * sequences[right_i - 1]
                right_i -= 2
        else:
            maximum_sum += sequences[right_i]
            right_i -= 1

    # Title: output
    result: str = str(maximum_sum)
    print(result)
    return result


def test_bundle_up_numbers() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "4",
                "-1",
                "2",
                "1",
                "3",
            ],
            ["6"],
        ],
        [
            [
                "2",
                "1",
                "1",
            ],
            ["2"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            bundle_up_numbers(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


def mix_two_solutions(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get the Zero-closest sum of two solution ; https://www.acmicpc.net/problem/2470

    Time Complexity (Worst-case): O(n(log n)) from Tim sort
        - O(n) from loop of variant of 2SUM problem

    Space Complexity (Worst-case): O(n) from Tim sort

    Definition
        Abs := Absolute value of the sum closest to zero
        Array := Sorted Array
        Sum : = sum of two values

        O := Optimized value that make Minimum Abs with one control variable existed
            - the Optimized value is different for each a independent variable.
            - the Optimized value may exist or not in given Array
                Assume that the value is between consecutive two values in the Array.
                To determine which values minimize Abs
                🚣, It should test both; each sum of (the control variable, one of consecutive two values)

                if so, even if a pointer (<left_i> | <right_i> moves one by one, it could covers that range.

    Purpose
        - to compare absolute sums of two solution by explicitly control two index pointers with exclusive range.

    Consideration
        - ❔ How do I compose up inside of the loop?
            chaining one by one in order to test sum of solution, one loop must have one predicate.

    Implementation
        - even if I implement by using Binary Search
            , since purpose is testing sum of two solution rather than searching one target
            , Anyway after Binary search it must be tested in condition where fixed one control variable (unmodified) and modify another variable.
            so Binary Search useless.
        - if list of solution is sorted state, It can calculate systematically all solution in order.
            if <temp_sum> is less than zero, two selections exist: increase (<left_i> | <right_i>).
            but <right_i> starts with end of the list, so it doesn't have to be bothered.
            the counterpart is likewise.
    """
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (2 ≤ N ≤ 100,000)
    n: int = int(input_())
    # condition: (-(10^9) ≤ each number of solution ≤ 10^9)
    # negative integer is acid solution, positive integer is alkaline solution.
    solutions = list(map(int, input_().split()))
    zero_closest_solutions_i: tuple[int, int] = (0, 1)
    zero_closest_abs: int = sys.maxsize

    # Title: solve
    solutions.sort()
    left_i: int = 0
    right_i: int = n - 1
    while left_i < right_i:
        temp_sum: int = solutions[left_i] + solutions[right_i]

        if (new_abs := abs(temp_sum)) < zero_closest_abs:
            zero_closest_abs = new_abs
            zero_closest_solutions_i = (left_i, right_i)
            if temp_sum == 0:
                break

        # if temp_sum == 0' is evaluated in upper expressions.
        if temp_sum < 0:
            left_i += 1
        else:
            right_i -= 1
    result_as_str = " ".join(
        map(
            str,
            ((solutions[i] for i in zero_closest_solutions_i)),
        )
    )

    # Title: output
    print(result_as_str)
    return result_as_str


def test_mix_two_solutions() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "5",
                "-2 4 -99 -1 98",
            ],
            ["-99 98"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            mix_two_solutions(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


# week 2-1: (BFS, DFS): 5
def construct_bridges_2(input_lines: Optional[Iterator[str]] = None) -> str:
    """🚤 get Length of optimized path through bridges to connect all islands ; https://www.acmicpc.net/problem/17472

    Time Complexity (Worst-case): ...
        - O( Number( BFS(islands) ) ) from BFS loop
        - O( Number( BFS(bridges) ) ) from BFS loop
        - O( MST(bridges) ) by using Kruscal's algorithm

    Space Complexity (Worst-case): ...
        - O( Number( BFS(islands)) ) from BFS
            = the number of given islands cells.
        - O( Number( BFS(bridges)) ) from BFS
            = the number of (given islands cells, bridges)
        - O( Number( bridges) ) from Kruscal's algorithm
            - O(E) from Tim sort

    Definition
        - Number( BFS(islands) ): |V| + |E|; Time occurred to distinguish islands in given 10*10 grid
            - the number of given islands cells are vertexes.
            - the number of cells identified in order to check same island are edges (up to 4 directions in each the cell).
        - Number( BFS(bridges) ): |V| + |E|; Time occurred to check whether bridges are construct in given 10*10 grid
            - the number of (given islands cells, bridges) are vertexes.
            - the number of cells identified in order to check constructable bridge are edges (up to 4 directions in each the cell).
        - MST(bridges)
            - O(E(log E)) from Tim sort
            - O(E) from iterating Edges in Kruscal's algorithm
            - O(α(n)) from Union-Find in Kruscal's algorithm

    Purpose
        - to get minimum length of bridges to route all islands.

    Consideration
        - "섬 A와 B를 연결하는 다리가 중간에 섬 C와 인접한 바다를 지나가는 경우에 섬 C는 A, B와 연결되어있는 것이 아니다."
        - "교차하는 다리의 길이를 계산할 때는 각 칸이 각 다리의 길이에 모두 포함되어야 한다."
        - given map uses numbers (0, 1) to represent OCEAN and ISLAND. so be careful when numbering islands.

    Implementation
        - Used data structure: adjacent list in BFS
        - I used Kruscal's algorithm to get optimized path (Minimum Spanning Tree).
            - used Find function of Union-find: Halving
        - Things I've done if adjacent islands (4 directions) of bridges are included in "connection"
            - BFS to constructable bridges in one depth, and simulate combinations with (1 ~ the number of edges)
                , even if duplicated bridges are chose.
    """
    import dataclasses
    import itertools
    import operator
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # point and direction
    PointVector = tuple[tuple[int, int], tuple[int, int]]

    @dataclasses.dataclass
    class Island:  # Vertex
        v: int
        parent: Island = dataclasses.field(init=False)
        rank: int = dataclasses.field(default=0)

        def __post_init__(self) -> None:
            self.parent = self

    @dataclasses.dataclass(eq=True, frozen=True)
    class Bridge:  # Edge
        v1: Island
        v2: Island
        length: int = dataclasses.field(default=0)

    DIRECTIONS: list[tuple[int, int]] = [
        (-1, 0),  # (-row direction)
        (1, 0),  # (+row direction)
        (0, -1),  # (-column direction)
        (0, 1),  # (+column direction)
    ]

    # find() of Union-find functions
    def find_root(node: Island) -> Island:
        while node.parent != node:
            node.parent = node.parent.parent
            node = node.parent
        return node

    # union() of Union-find function
    def merge_two_sets(v1: Island, v2: Island) -> None:
        x = find_root(v1)
        y = find_root(v2)

        # x, y are already in the same set.
        if x == y:
            return
        # ensure that x has rank at least as large as that of y
        if x.rank < y.rank:
            (x, y) = (y, x)

        # make x the new root
        y.parent = x
        if x.rank == y.rank:
            x.rank += 1

    # Title: input
    # condition: (1 ≤ N, M ≤ 10), (3 ≤ N * M ≤ 100)
    # condition: (2 ≤ the number of islands ≤ 6)
    n, m = map(int, input_().split())
    # OCEAN = 0, LAND = 1
    map_: list[list[int]] = [list(map(int, input_().split())) for _ in range(n)]
    island_key_gen = itertools.count(2)
    island_zones_dict: dict[int, list[tuple[int, int]]] = {}
    island_dict: dict[int, Island] = {}
    bridge_list: list[Bridge] = []
    minimum_mst_bridges: list[Bridge] = []
    minimum_mst_bridges_length: int = -1

    # Title: solve
    # distinguish islands by using BFS. I used number from 2 according to given number on map.
    for row, line in enumerate(map_):
        for column, x in enumerate(line):
            if x != 1:
                continue

            island_key = next(island_key_gen)
            island_zones_dict[island_key] = [(row, column)]
            island_dict[island_key] = Island(v=island_key)
            # <island_key> can be substitution of <is_explored>.
            map_[row][column] = island_key
            explored_deque: deque[tuple[int, int]] = deque([(row, column)])
            while len(explored_deque) > 0:
                point = explored_deque.popleft()
                for direction in DIRECTIONS:
                    new_point: tuple[int, int] = tuple(
                        map(operator.add, point, direction)
                    )

                    # Each islands are at least one square apart, horizontally or vertically.
                    if (
                        0 <= new_point[0] < n
                        and 0 <= new_point[1] < m
                        and map_[new_point[0]][new_point[1]] == 1
                    ):
                        island_zones_dict[island_key].append(new_point)
                        map_[new_point[0]][new_point[1]] = island_key
                        explored_deque.append(new_point)

    # create bridges information
    dest_cell_point_vector_set: set[PointVector] = set()
    for island, zone in island_zones_dict.items():
        # check cells in depth 1 from a cell
        start_points_and_vectors: list[PointVector] = []
        for cell_point in zone:
            for direction in DIRECTIONS:
                new_point = tuple(map(operator.add, cell_point, direction))
                # condition: (bridge is constructed on OCEAN.).
                if (
                    0 <= new_point[0] < n
                    and 0 <= new_point[1] < m
                    and map_[new_point[0]][new_point[1]] == 0
                    and (cell_point, direction) not in dest_cell_point_vector_set
                ):
                    start_points_and_vectors.append((new_point, direction))

        # BFS to check to be able to construct bridges until to find another island zone
        for start_point, direction in start_points_and_vectors:
            explored_deque: deque[tuple[int, int]] = deque([start_point])
            bridge_points: list[tuple[int, int]] = [start_point]
            dest_cell_point_vector: Optional[PointVector] = None
            dest_cell_no: int = -1
            while len(explored_deque) > 0:
                point = explored_deque.popleft()
                new_point = tuple(map(operator.add, point, direction))
                if 0 <= new_point[0] < n and 0 <= new_point[1] < m:
                    if (cell_no := map_[new_point[0]][new_point[1]]) != 0:
                        dest_cell_point_vector = new_point, tuple(
                            map(operator.mul, direction, (-1, -1))
                        )
                        dest_cell_no = cell_no
                        break
                    bridge_points.append(new_point)
                    explored_deque.append(new_point)

            # condition: (2 <= bridge length)
            if len(bridge_points) < 2 or not dest_cell_point_vector:
                continue
            # if bridges can be constructed on OCEAN
            dest_cell_point_vector_set.add(dest_cell_point_vector)

            # bridges with different points but same connections (v1, v2) will be filtered in MST later.
            bridge_list.append(
                Bridge(
                    v1=island_dict[island],
                    v2=island_dict[dest_cell_no],
                    length=len(bridge_points),
                )
            )

    # MST
    bridge_list.sort(key=lambda e: e.length)
    for bridge in bridge_list:
        if find_root(bridge.v1) != find_root(bridge.v2):
            minimum_mst_bridges.append(bridge)
            merge_two_sets(bridge.v1, bridge.v2)
            if len(minimum_mst_bridges) == len(island_dict) - 1:
                # MST creation complete
                minimum_mst_bridges_length = sum(
                    bridge.length for bridge in minimum_mst_bridges
                )
                break

    # Title: output
    result: str = str(minimum_mst_bridges_length)
    print(result)
    return result


def test_construct_bridges_2() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "7 8",
                "0 0 0 0 0 0 1 1",
                "1 1 0 0 0 0 1 1",
                "1 1 0 0 0 0 0 0",
                "1 1 0 0 0 1 1 0",
                "0 0 0 0 0 1 1 0",
                "0 0 0 0 0 0 0 0",
                "1 1 1 1 1 1 1 1",
            ],
            ["9"],
        ],
        [
            [
                "7 8",
                "0 0 0 1 1 0 0 0",
                "0 0 0 1 1 0 0 0",
                "1 1 0 0 0 0 1 1",
                "1 1 0 0 0 0 1 1",
                "1 1 0 0 0 0 0 0",
                "0 0 0 0 0 0 0 0",
                "1 1 1 1 1 1 1 1",
            ],
            ["10"],
        ],
        [
            [
                "7 8",
                "1 0 0 1 1 1 0 0",
                "0 0 1 0 0 0 1 1",
                "0 0 1 0 0 0 1 1",
                "0 0 1 1 1 0 0 0",
                "0 0 0 0 0 0 0 0",
                "0 1 1 1 0 0 0 0",
                "1 1 1 1 1 1 0 0",
            ],
            ["9"],
        ],
        [
            [
                "7 7",
                "1 1 1 0 1 1 1",
                "1 1 1 0 1 1 1",
                "1 1 1 0 1 1 1",
                "0 0 0 0 0 0 0",
                "1 1 1 0 1 1 1",
                "1 1 1 0 1 1 1",
                "1 1 1 0 1 1 1",
            ],
            ["-1"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            construct_bridges_2(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


def sort_2d_array_puzzle(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Minimum number of turns to make number map as sorted state ; https://www.acmicpc.net/problem/1525

    Time Complexity (Worst-case): O( Number( BFS(routes) ) ) from BFS loop

    Space Complexity (Worst-case): O( Number(given_start_point) ) from BFS at least
        It is one.

    Definition
        - Number( BFS(routes) ): |V| + |E|; Time occurred until finding answer or all available cases are exhausted in given 3*3 grid
            - not duplicated number map including initial number map are vertexes.
            - adjacent number cells of a "0" node are edges (up to 4 directions in each the cell)

    Purpose
        - to reduce memory usage occurred in BFS deque.

    Implementation
        - Used data structure: adjacent list in BFS with set
            It is similar with the problem "move_alphabet_piece" (function).
            🚣 To use hashable object as string type is key point in set type in order to check duplication.
            Empty string occupy 49 bytes and each ASCII bytes occupy 1 byte.
            so output of    sys.getsizeof("123456780")    is 58.
        - Things I've done:
            - Using hashable objects as tuples that indicates points in two-dimensional array.
                It causes "Out of Memory" in the submit site.
    """
    import dataclasses
    import itertools
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    @dataclasses.dataclass(eq=True, frozen=True)
    class MapState:
        empty_i: int
        line_map: str

    class GameWin(Exception):
        pass

    PURPOSE_LINE_MAP: str = "123456780"

    def get_new_i_list(i: int) -> list[int]:
        """returns <new_i_list> based on DIRECTIONS as 1D array converted from 2D array"""
        # maximum the number of elements in one line is 3, and have 3 lines.
        new_i_list: list[int] = []
        # (-row direction)
        if i >= 3:
            new_i_list.append(i - 3)
        # (+row direction)
        if i < 6:
            new_i_list.append(i + 3)
        # (-column direction)
        if i % 3 != 2:
            new_i_list.append(i + 1)
        # (+column direction)
        if i % 3 != 0:
            new_i_list.append(i - 1)
        return new_i_list

    # Title: input
    temp_number_map: list[list[int]] = [
        list(map(int, input_().split())) for _ in range(3)
    ]
    line_map: str = "".join(map(str, itertools.chain(*temp_number_map)))
    # EMPTY = 0 (start point)
    start_i: int = line_map.index("0")
    turns: int = 0

    # Title: solve
    explored_deque: deque[MapState] = deque()
    explored_map_state_set: set[MapState] = set()
    next_exploration_deque: deque[MapState] = deque()
    if line_map == PURPOSE_LINE_MAP:
        print(turns)
        return str(turns)
    else:
        turns += 1
        map_state = MapState(start_i, line_map)
        explored_deque.append(map_state)
        explored_map_state_set.add(map_state)

    try:
        while len(explored_deque) >= 1:
            explored_map_state = explored_deque.popleft()
            empty_i = explored_map_state.empty_i

            for new_i in get_new_i_list(empty_i):
                temp_line_map = list(explored_map_state.line_map)
                (
                    temp_line_map[empty_i],
                    temp_line_map[new_i],
                ) = (temp_line_map[new_i], temp_line_map[empty_i])
                new_map_state = MapState(new_i, "".join(temp_line_map))

                if new_map_state not in explored_map_state_set:
                    if (
                        new_map_state.line_map[-1] == "0"
                        and new_map_state.line_map == PURPOSE_LINE_MAP
                    ):
                        raise GameWin
                    else:
                        next_exploration_deque.append(new_map_state)
                        explored_map_state_set.add(new_map_state)

            # <turns>-th ends
            if len(explored_deque) == 0:
                turns += 1

                # settings for next exploration
                explored_deque = next_exploration_deque.copy()
                next_exploration_deque.clear()
    except GameWin:
        pass
    else:
        turns = -1

    # Title: output
    result: str = str(turns)
    print(result)
    return result


def test_sort_2d_array_puzzle() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "1 0 3",
                "4 2 5",
                "7 8 6",
            ],
            ["3"],
        ],
        [
            [
                "1 2 3",
                "4 5 6",
                "7 8 0",
            ],
            ["0"],
        ],
        [
            [
                "3 6 0",
                "8 1 2",
                "7 4 5",
            ],
            ["-1"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            sort_2d_array_puzzle(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


def go_down_downhill(input_lines: Optional[Iterator[str]] = None) -> str:
    """🚤 get Number of cases where a person can go to the point of destination ; https://www.acmicpc.net/problem/1520

    Time Complexity (Worst-case): ...
        - O( Number( BFS(routes) ) ) from BFS loop
        - O(log x) from Hip (pop | push) ) in BFS loop
            1 hip pop is occurred when first visits vertex
            1 hip push is occurred when first visits edge

    Space Complexity (Worst-case): O( Number(given_start_point) ) from BFS at least
        It is one.

    Definition
        - n, m: size to create map (n*m grid).
        - Number( BFS(routes) ): |V| + |E|; Time occurred from routes including given start point in given n*m grid
            - movable number cells from start point are vertexes that meet given condition.
            - available adjacent number cells of a number node are edges (up to 4 directions in each the cell)

    Purpose
        - to reduce calculation of duplicated path
            , It must propagate visit count to next cell
            , and the cells with the higher number must be calculated first to propagate all cumulated visit count.

    Implementation
        - Used data structure: Heap
            - Max heap for (number, point) on cell.
    """
    import dataclasses
    import heapq
    import operator
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    @dataclasses.dataclass(eq=True, order=True)
    class RouteState:
        number: int
        point: tuple[int, int]

    DIRECTIONS: list[tuple[int, int]] = [
        (-1, 0),  # (-row direction)
        (1, 0),  # (+row direction)
        (0, -1),  # (-column direction)
        (0, 1),  # (+column direction)
    ]
    START_POINT: tuple[int, int] = (0, 0)
    NumberMap = list[list[int]]

    # Title: input
    # condition: (1 ≤ N, M ≤ 500)
    # condition: (1 ≤ number on each cell ≤ 10000)
    n, m = map(int, input_().split())
    number_map: NumberMap = [list(map(int, input_().split())) for _ in range(n)]
    # condition: (1 ≤ the number of route success ≤ 1,000,000,000)
    visit_count_map: NumberMap = [[0 for _ in range(m)] for _ in range(n)]
    end_point: tuple[int, int] = (n - 1, m - 1)

    # Title: solve
    route_heapq: list[RouteState] = [
        RouteState(-number_map[START_POINT[0]][START_POINT[1]], START_POINT)
    ]
    visit_count_map[START_POINT[0]][START_POINT[1]] += 1
    while route_heapq:
        route_state = heapq.heappop(route_heapq)
        route_state.number *= -1
        if route_state.point == end_point:
            continue

        for direction in DIRECTIONS:
            new_point: tuple[int, int] = tuple(
                map(operator.add, route_state.point, direction)
            )
            if 0 <= new_point[0] < n and 0 <= new_point[1] < m:
                new_number = number_map[new_point[0]][new_point[1]]
                if new_number < route_state.number:
                    if visit_count_map[new_point[0]][new_point[1]] == 0:
                        heapq.heappush(route_heapq, RouteState(-new_number, new_point))

                    # propagate previous cell's visit count to new point
                    visit_count_map[new_point[0]][new_point[1]] += visit_count_map[
                        route_state.point[0]
                    ][route_state.point[1]]

    # Title: output
    result: str = str(visit_count_map[end_point[0]][end_point[1]])
    print(result)
    return result


def test_go_down_downhill() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "4 4",
                "20 19 18 17",
                "10 9 8 16",
                "100 100 7 15",
                "100 100 6 14",
            ],
            ["1"],
        ],
        [
            [
                "4 5",
                "50 45 37 32 30",
                "35 50 40 20 25",
                "30 30 25 17 28",
                "27 24 22 15 10",
            ],
            ["3"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            go_down_downhill(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


def move_alphabet_piece(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Maximum movable distance ; https://www.acmicpc.net/problem/1987

    Time Complexity (Worst-case): O( Number( BFS(routes) ) ) from BFS loop

    Space Complexity (Worst-case): O( Number(given_start_point) ) from BFS at least
        It is one.

    Definition
        - n, m: size to create alphabet board (n*m grid).
        - Number( BFS(routes) ): |V| + |E|; Time occurred from routes including given start point in given n*m grid
            - movable alphabet cells from start point are vertexes that meet given condition.
            - available adjacent alphabet cells of a alphabet node are edges (up to 4 directions in each the cell)

    Implementation
        - Used data structure: set in BFS
            By making hashable class or by using tuples, the container can be used as element of queue.
            <track> in class <RouteState> is used instead of hash table that indicates whether cell is explored.
            even if <track> is same but actual route is different at some point, it doesn't matter.
            purpose is <maximum_passing_distance> and these <RouteState> can't invade each other's path in according to given condition.
        - Used data structure: array for reducing memory usage.
        - DFS non-recursive implementation was evaluated as Timeout (slow) because DFS must be backtracking and it's cost is expensive.
            If the size of the board increases, the number of cases to be tracked increases exponentially.
            Things I've done:
            - to convert input characters to integer
            - to make <passing_distance_list> as set type for membership testing
    """
    import dataclasses
    import operator
    import sys
    from array import array
    from collections.abc import MutableSequence

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    @dataclasses.dataclass(eq=True, frozen=True)
    class RouteState:
        point: tuple[int, int]
        track: str

    DIRECTIONS: list[tuple[int, int]] = [
        (-1, 0),  # (-row direction)
        (1, 0),  # (+row direction)
        (0, -1),  # (-column direction)
        (0, 1),  # (+column direction)
    ]
    START_POINT: tuple[int, int] = (0, 0)
    AlphabetBoard = list[list[str]]

    # Title: input
    # condition: (1 ≤ N, M ≤ 20)
    n, m = map(int, input_().split())
    alphabet_board: AlphabetBoard = [list(input_()) for _ in range(n)]
    passing_distance_list: MutableSequence[int] = array("b", [])
    maximum_passing_distance: int = 0

    # Title: solve
    route_state_set: set[RouteState] = set()
    route_state_set.add(
        RouteState(START_POINT, alphabet_board[START_POINT[0]][START_POINT[1]])
    )
    passing_distance_list.append(1)

    while route_state_set:
        route_state = route_state_set.pop()
        for direction in DIRECTIONS:
            new_point: tuple[int, int] = tuple(
                map(operator.add, route_state.point, direction)
            )
            if (
                0 <= new_point[0] < n
                and 0 <= new_point[1] < m
                and alphabet_board[new_point[0]][new_point[1]] not in route_state.track
            ):
                route_state_set.add(
                    RouteState(
                        new_point,
                        route_state.track + alphabet_board[new_point[0]][new_point[1]],
                    )
                )
                passing_distance_list.append(len(route_state.track) + 1)
    maximum_passing_distance = max(passing_distance_list)

    # Title: output
    result: str = str(maximum_passing_distance)
    print(result)
    return result


def test_move_alphabet_piece() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "3 6",
                "HFDFFB",
                "AJHGDH",
                "DGAGEH",
            ],
            ["6"],
        ],
        [
            [
                "5 5",
                "IEFCJ",
                "FHFKC",
                "FFALF",
                "HFGCF",
                "HMCHH",
            ],
            ["10"],
        ],
        [
            [
                "10 10",
                "ASWERHGCFH",
                "QWERHDLKDG",
                "ZKFOWOHKRK",
                "SALTPWOKSS",
                "BMDLKLKDKF",
                "ALSKEMFLFQ",
                "GMHMBPTIYU",
                "DMNKJZKQLF",
                "HKFKGLKEOL",
                "OTOJKNKRMW",
            ],
            ["22"],
        ],
        [
            [
                "20 20",
                "ZYXWVUTSRQPONMLKJIHG",
                "YXWVUTSRQPONMLKJIHGF",
                "XWVUTSRQPONMLKJIHGFE",
                "WVUTSRQPONMLKJIHGFED",
                "VUTSRQPONMLKJIHGFEDC",
                "UTSRQPONMLKJIHGFEDCB",
                "TSRQPONMLKJIHGFEDCBA",
                "SRQPONMLKJIHGFEDCBAA",
                "RQPONMLKJIHGFEDCBAAA",
                "QPONMLKJIHGFEDCBAAAA",
                "PONMLKJIHGFEDCBAAAAA",
                "ONMLKJIHGFEDCBAAAAAA",
                "NMLKJIHGFEDCBAAAAAAA",
                "MLKJIHGFEDCBAAAAAAAA",
                "LKJIHGFEDCBAAAAAAAAA",
                "KJIHGFEDCBAAAAAAAAAA",
                "JIHGFEDCBAAAAAAAAAAA",
                "IHGFEDCBAAAAAAAAAAAA",
                "HGFEDCBAAAAAAAAAAAAA",
                "GFEDCBAAAAAAAAAAAAAA",
            ],
            ["26"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            move_alphabet_piece(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


def ripen_tomatoes(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Elapsed days to all tomatoes to be ripen in conditions ; https://www.acmicpc.net/problem/7576

    Time Complexity (Worst-case): O( Number( BFS(spread_ripened_tomatoes) ) ) from BFS loop

    Space Complexity (Worst-case): O( Number(given_ripened_tomatoes) ) from BFS at least

    Definition
        - n, m: size to create space of tomatoes 2D tank (n*m grid).
        - Number(given_ripened_tomatoes): The number of ripened tomatoes in given n*m grid before ripened state spread.
        - Number( BFS(spread_ripened_tomatoes) ): |V| + |E|; Time occurred from spread ripened tomatoes including given ripened tomatoes in given n*m grid
            - spread ripened tomatoes are vertexes
            - adjacent tomatoes of a ripened tomato are edges (up to 4 directions in each the cell)

    Implementation
        - Used data structure: deque in BFS.
        - This solution does not require <is_explored> variable in BFS.
            instead code to compare <is_explored> can be replaced by checking for ripened tomatoes
            , so that code will be simple.
    """
    import operator
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    DIRECTIONS: list[tuple[int, int]] = [
        (-1, 0),  # (-row direction)
        (1, 0),  # (+row direction)
        (0, -1),  # (-column direction)
        (0, 1),  # (+column direction)
    ]

    # Title: input
    # condition: (2 ≤ N, M ≤ 1,000)
    m, n = map(int, input_().split())
    tomato_2d_tank: list[list[int]] = []
    ripe_tomatoes_points: list[tuple[int, int]] = []
    not_ripe_tomatoes_count: int = 0
    for row in range(n):
        # n*m loop
        line: list[int] = list(map(int, input_().split()))
        tomato_2d_tank.append(line)
        for column, x in enumerate(line):
            # EMPTY = -1, NOT_RIPE = 0, RIPE = 1
            if x == 0:
                not_ripe_tomatoes_count += 1
            elif x == 1:
                ripe_tomatoes_points.append((row, column))
    elapsed_day: int = 0

    # Title: solve
    explored_deque: deque[tuple[int, int]] = deque()
    if not_ripe_tomatoes_count > 0:
        elapsed_day += 1
        explored_deque.extend(ripe_tomatoes_points)

    # simulate
    next_exploration_deque: deque[tuple[int, int]] = deque()
    while not_ripe_tomatoes_count > 0 and len(explored_deque) > 0:
        # add some tomatoes to <next_exploration_deque> from <explored_deque>.
        explored_point = explored_deque.popleft()
        for direction in DIRECTIONS:
            new_point: tuple[int, int] = tuple(
                map(operator.add, explored_point, direction)
            )

            if (
                0 <= new_point[0] < n
                and 0 <= new_point[1] < m
                and tomato_2d_tank[new_point[0]][new_point[1]] == 0
            ):
                tomato_2d_tank[new_point[0]][new_point[1]] = 1
                not_ripe_tomatoes_count -= 1
                next_exploration_deque.append(new_point)

        # <elapsed_day>-th ends. judge <next_exploration_deque>.
        if len(explored_deque) == 0:
            # when tomatoes can not spread "ripe" state or all tomatoes are ripened.
            if not_ripe_tomatoes_count == 0 or len(next_exploration_deque) == 0:
                break

            # <elapsed_day> ends
            elapsed_day += 1

            # settings for next exploration
            explored_deque = next_exploration_deque.copy()
            next_exploration_deque.clear()

    # when loop is break due to empty <next_exploration_deque> but not ripe tomatoes are remained.
    if not_ripe_tomatoes_count > 0:
        elapsed_day = -1

    # Title: output
    result: str = str(elapsed_day)
    print(result)
    return result


def test_ripen_tomato() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "6 4",
                "0 0 0 0 0 0",
                "0 0 0 0 0 0",
                "0 0 0 0 0 0",
                "0 0 0 0 0 1",
            ],
            ["8"],
        ],
        [
            [
                "6 4",
                "0 -1 0 0 0 0",
                "-1 0 0 0 0 0",
                "0 0 0 0 0 0",
                "0 0 0 0 0 1",
            ],
            ["-1"],
        ],
        [
            [
                "2 2",
                "1 -1",
                "-1 1",
            ],
            ["0"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            ripen_tomatoes(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


# week 1-2: Implementation: 5
def escape_marble_2(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Minimum turn number where the game win within turns 10 ; https://www.acmicpc.net/problem/13460
    🔍 다시. iterator를 4개로 잘 나누기.

    Time Complexity (Worst-case): ...
        - O( Number( BFS(board_for_marble) ) ) from BFS loop.
        - O(n*m) in BFS loop
            operations to realign locations of marbles.

    Space Complexity (Worst-case): O(1) from BFS at least
        - 1 is element in queue to indicates initial state

    Definition
        - LAST_TURN: 10. the number of maximum turns.
        - Number(directions): 4. (left, right, down, up). available motions in every turn.
        - n, m: size to create marble board (n*m grid).
        - Number( BFS(board_for_marble) ): |V| + |E|; Time occurred from branching as many Number(directions) until turn 10
            - the number of (blue, red) marbles points are vertexes
            - sum of Number(directions) from each turn (1 ~ 9 turns) are edges.

    Consideration
        ❔ Where should I put the code for determining the win or loss (failure or continuation) of the game and the code for early termination?
            - for faster exit game, I compared locations of hole and marbles by line. not by block.
                and I used try ~ except statement in order to break nested loop once.
        - Even if blue marble falls into the hole (failure), It should not raise GameEnd exception to immediately break all loops
            , because remained cases are still not tested.

    Implementation
        - Used data structure: array for reducing memory usage.
        - Used data structure: deque in BFS.
            branches (2d array) are created by assemble elements of original 2d array.
            but in this problem, it should trace blue and red marble point to avoid duplicated exploration.
            so I used the way that assemble original 2d array and reassemble to recover original 2d array.
            this makes that code to process in the iterator can have consistency.
        - I made the class <MarblesPoint> hashable to compare faster whether the same marble points are in <explored_marbles_points> or not.
    """
    import dataclasses
    import re
    import sys
    from collections import deque
    from typing import TypeVar

    T = TypeVar("T")

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    BoardForMarble = list[list[str]]
    LAST_TURN: int = 10

    def assemble(two_d_iterator: list[list[T]], direction_i: int) -> list[list[T]]:
        """Assemble <two_d_iterator> based on <direction_i> and returns new iterator."""
        # direction_i is index in DIRECTIONS
        match direction_i:
            # assemble based on edge of <(-, +) row, (-, +) column> direction
            case 0:  # against -row axis (+vertical)
                return list(zip(*two_d_iterator))
            case 1:  # against +row axis (-vertical)
                return list(map(list, map(reversed, zip(*two_d_iterator))))  # type: ignore
            case 2:  # against -column axis (-horizontal)
                return two_d_iterator
            case 3:  # against +column axis (+horizontal)
                return list(map(list, map(reversed, two_d_iterator)))  # type: ignore
            case _:
                return two_d_iterator

    def reassemble(two_d_iterator: list[list[T]], direction_i: int) -> list[list[T]]:
        """Reassemble result from function <assemble> and returns original <two_d_iterator>.
        It similar with processing decryption in cryptography"""
        # direction_i is index in DIRECTIONS
        match direction_i:
            # reassemble based on edge of <(-, +) row, (-, +) column> direction
            case 0:  # against -row axis (+vertical)
                return list(zip(*two_d_iterator))
            case 1:  # against +row axis (-vertical)
                return list(zip(*map(reversed, two_d_iterator)))  # type: ignore
            case 2:  # against -column axis (-horizontal)
                return two_d_iterator
            case 3:  # against +column axis (+horizontal)
                return list(map(list, map(reversed, two_d_iterator)))  # type: ignore
            case _:
                return two_d_iterator

    def reassemble_i(
        column_len: int, direction_i: int, points: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Similar with function <reassemble> but target is points and returns points as indexes.

        Corresponding operations in n*m iterator
        |<reassemble>       |<reassemble_i>|
        |:---  |:---|
        |zip                |(i, j) -> (j, i)|
        |reversed columns   |(i, j) -> (i, m-j-1)|
        """
        # direction_i is index in DIRECTIONS
        match direction_i:
            # assemble based on edge of <(-, +) row, (-, +) column> direction
            case 0:  # against -row (+vertical)
                return [(point[1], point[0]) for point in points]
            case 1:  # against +row (-vertical)
                return [(column_len - point[1] - 1, point[0]) for point in points]
            case 2:  # against -column (-horizontal)
                return points
            case 3:  # against +column (+horizontal)
                return [(point[0], column_len - point[1] - 1) for point in points]
            case _:
                return points

    @dataclasses.dataclass(frozen=True, eq=True)
    class MarblesPoint:
        red: tuple[int, int]
        blue: tuple[int, int]

    class GameEnd(Exception):
        pass

    # Title: input
    # condition: (3 ≤ N, M ≤ 10)
    n, m = map(int, input_().split())
    # n*m board input. but actually (n-2)*(m-2) board. (except edge "#" (OBSTACLE) cells)
    board_for_marble: BoardForMarble = []
    red_marble_point: tuple[int, int] = None  # type: ignore
    blue_marble_point: tuple[int, int] = None  # type: ignore
    for row in range(n):
        # OBSTACLE = "#", EMPTY = ".", HOLE = "O", RED_MARBLE = "R", BLUE_MARBLE = "B"
        point = list(input_())
        if 0 < row < n - 1:
            board_for_marble.append([])
            for column in range(1, m - 1):
                board_for_marble[-1].append(point[column])
                if point[column] == "R":
                    red_marble_point = (row - 1, column - 1)
                elif point[column] == "B":
                    blue_marble_point = (row - 1, column - 1)
    turn_no: int = 1
    game_end_pattern = re.compile(r"O([RB]{1,2})")
    # condition: <pass_result> will be -1 when can not pass the game in 10 turns or Blue marble falls into a hole (fail),
    # , otherwise will be turn number (success). "None" indicates "before starting game".
    pass_result: Optional[int] = None

    # Title: solve
    # simulate
    explored_deque: deque[BoardForMarble] = deque([board_for_marble])
    explored_marbles_points: set[MarblesPoint] = {
        MarblesPoint(red=red_marble_point, blue=blue_marble_point)
    }
    next_exploration_deque: deque[BoardForMarble] = deque()
    try:
        while len(explored_deque) >= 1:
            # add boards to <next_exploration_deque> from <explored_deque>
            explored_board = explored_deque.popleft()
            column_len: int = len(explored_board[0])

            for direction_i in range(4):
                new_board: BoardForMarble = []
                red_marble_point: tuple[int, int] = None  # type: ignore
                blue_marble_point: tuple[int, int] = None  # type: ignore
                is_fail: bool = False

                for i, line in enumerate(assemble(explored_board, direction_i)):
                    new_line: list[str] = []
                    empty_cell_count_before_edge: int = 0
                    other_cells_before_edge: list[str] = []

                    # realign characters
                    for char in line:
                        match char:
                            case x if x in ["#", "O"]:
                                # when meet OBSTACLE or hole
                                new_line.extend(other_cells_before_edge)
                                new_line.extend(["."] * empty_cell_count_before_edge)
                                new_line.append(x)

                                # clear used variables for empty cells
                                empty_cell_count_before_edge = 0
                                other_cells_before_edge.clear()
                            case ".":
                                empty_cell_count_before_edge += 1
                            case _:
                                # case x if x in ["R", "B"]:
                                other_cells_before_edge.append(char)
                    else:
                        new_line.extend(other_cells_before_edge)
                        new_line.extend(["."] * empty_cell_count_before_edge)

                    # find marbles's actual points on the new line after realign
                    if not red_marble_point or not blue_marble_point:
                        for j, char in enumerate(new_line):
                            if char == "R":
                                red_marble_point = reassemble_i(
                                    column_len, direction_i, [(i, j)]
                                )[0]
                            elif char == "B":
                                blue_marble_point = reassemble_i(
                                    column_len, direction_i, [(i, j)]
                                )[0]

                    # judge whether player wins by line
                    if match_object := game_end_pattern.search(
                        string="".join(new_line)
                    ):
                        marbles: str = match_object.group(1)
                        # if len(marbles) > 1: fail, but only in the case.
                        match marbles:
                            case "R":
                                raise GameEnd(turn_no)
                            case _:
                                # x if x in ["B", "RB", "BR"]:
                                is_fail = True
                                break

                    new_board.append(new_line)

                # only add the case that is not fail.
                if (
                    not is_fail
                    and (
                        marbles_point := MarblesPoint(
                            red=red_marble_point, blue=blue_marble_point
                        )
                    )
                    not in explored_marbles_points
                ):
                    explored_marbles_points.add(marbles_point)
                    next_exploration_deque.append(reassemble(new_board, direction_i))

            # <turn_no>-th ends. judge <next_exploration_deque>.
            if len(explored_deque) == 0:
                turn_no += 1
                if turn_no > LAST_TURN:
                    raise GameEnd(-1)

                # settings for next exploration
                explored_deque = next_exploration_deque.copy()
                next_exploration_deque.clear()
    except GameEnd as e:
        pass_result = e.args[0]
    else:
        # early exit when it can not explore valid cases. (<next_exploration_deque> is empty)
        pass_result = -1

    # Title: output
    result: str = str(pass_result)
    print(result)
    return result


def test_escape_marbles_2() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "3 10",
                "##########",
                "#.O....RB#",
                "##########",
            ],
            ["-1"],
        ],
        [
            [
                "7 8",
                "########",
                "#.#O.#R#",
                "#....#B#",
                "#.#....#",
                "#......#",
                "#......#",
                "########",
            ],
            ["9"],
        ],
        [
            [
                "10 10",
                "##########",
                "#.#....###",
                "#........#",
                "#........#",
                "##B..#...#",
                "#.#......#",
                "#.#..R...#",
                "#...O#...#",
                "#.#....###",
                "##########",
            ],
            ["6"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            escape_marble_2(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


def play_2048_easy(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Maximum score within turns 5 ; https://www.acmicpc.net/problem/12100

    Time Complexity (Worst-case): ...
        - O( Number( DFS(board_2048) ) ) from DFS loop.
            1281 = 1 + 4^5 + 4^4
        - O(n*n) in DFS loop
            - 1024 iteration times  ==  the number of vertexes in DFS.
                *  O(n) from <grid_iterator> loop by directions.
                *  O(3n) in <line> loop; two list(filter(...)) and one while statement.

    Space Complexity (Worst-case): O(1) from DFS
        - by using <dfs_stack_by_turn>, maximum sum of stack size is 13.
            maximum turns is fixed in given condition as 5.
            <dfs_stack_by_turn>[1]: 0
            <dfs_stack_by_turn>[2]: 3
            <dfs_stack_by_turn>[3]: 3
            <dfs_stack_by_turn>[4]: 3
            <dfs_stack_by_turn>[5]: 4

    Definition
        - LAST_TURN: 5. the number of maximum turns.
        - Number(directions): 4. (left, right, down, up). available motions in every turn.
        - n: size to create 2048 board (n*n grid).
        - Number( DFS(board_2048) ): |V| + |E|; Time occurred from branching as many available motions until turn 5
            This can be thought of as a tree structure with a depth of 6 and a branching factor of 4.
                - DFS explores 1 ~ 6 depth, but it does not new boards on 6-th depth into stack according to given condition.
            - 1 + 4^5 are vertexes ==  Number(directions)^(LAST_TURN)
            - 4^4 are edges  ==  Number(directions)^(LAST_TURN - 1).

    Consideration
        - ❔ Is there a way to find the best score without simulating every case?
            I don't know it's best strategy.
            It would be better to test the number of all cases (Brute-force) by using DFS.
        - refer to original game link: https://play2048.co/


    Implementation
        - Used data structure: stack in DFS using <dfs_stack_by_turn>
            graph that have all cases can be thought as a directed rooted out-tree.
            but a variable to verify it has been explored is not required.
            If instead it uses BFS or simple Brute-force (itertools.product), space complexity is bigger.
        - when calculates numbers, to transpose calculated board to true location of cells doesn't matter.
            because this can be thought of as a transposed and sorted matrix
            , and branch types in each branch is same. (Number(directions))
        - function <get_square_iterators_against_gravity> makes that code to process in the iterator can have consistency.
    """
    import itertools
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    TwoDIterator = list[list[tuple[int, int]]]
    Board2048 = list[list[int]]
    TURN_LIMIT: int = 5

    def get_square_iterators_against_gravity(
        line_len: int,
    ) -> list[TwoDIterator]:
        """returns n*n (<line_len>*<line_len>) iterators against gravity.

        It is efficient because once returns iterators, it can be used continuously without any additional operations to create iterators.

        ⚠️ This function can not be used in following situations:
            - tracing some points of original square is required.
            - the number of row and column is not same.
        """

        # responses against pressed arrow key
        # when up arrow key (-row direction) is pressed
        response_for_row_minus: TwoDIterator = [
            [(row, column) for row in range(0, line_len, 1)]
            for column in range(0, line_len, 1)
        ]
        # when down arrow key (+row direction) is pressed
        response_for_row_plus: TwoDIterator = [
            [(row, column) for row in range(line_len - 1, -1, -1)]
            for column in range(0, line_len, 1)
        ]
        # when left arrow key (-column direction) is pressed
        response_for_column_minus: TwoDIterator = [
            [(row, column) for column in range(0, line_len, 1)]
            for row in range(0, line_len, 1)
        ]
        # when right arrow key (+column direction) is pressed
        response_for_column_plus: TwoDIterator = [
            [(row, column) for column in range(line_len - 1, -1, -1)]
            for row in range(0, line_len, 1)
        ]
        return [
            response_for_row_minus,
            response_for_row_plus,
            response_for_column_minus,
            response_for_column_plus,
        ]

    # Title: input
    # condition: (1 ≤ N ≤ 20)
    n: int = int(input_())
    # 0 cell indicates empty, other cells indicates the number.
    # condition: number in valid cell is 2^x (x is natural number)
    # n*n board
    board_2048: Board2048 = [[*map(int, input_().split())] for _ in range(n)]
    square_iterators = get_square_iterators_against_gravity(line_len=n)
    depth_level_pointer: int = 1
    # 0 is not used.
    dfs_stacks_by_turn: list[list[Board2048]] = [[] for _ in range(TURN_LIMIT + 1)]
    dfs_stacks_by_turn[1].append(board_2048)
    score_list: list[int] = []
    maximum_score: int = 0

    # Title: solve
    while True:
        # DFS according to the priority where next depth pointer is not empty.
        if (
            depth_level_pointer + 1 <= TURN_LIMIT
            and len(dfs_stacks_by_turn[depth_level_pointer + 1]) > 0
        ):
            depth_level_pointer += 1
        elif len(dfs_stacks_by_turn[depth_level_pointer]) > 0:
            pass
        else:
            depth_level_pointer -= 1
            if depth_level_pointer < 1:
                break
            else:
                continue

        # simulate
        explored_board = dfs_stacks_by_turn[depth_level_pointer].pop()
        for grid_iterator in square_iterators:
            new_board: Board2048 = []
            for line in grid_iterator:
                new_line = list(
                    filter(
                        lambda x: x > 0,
                        (explored_board[point[0]][point[1]] for point in line),
                    )
                )
                if (new_line_len := len(new_line)) >= 2:
                    i = 0
                    while i < new_line_len - 1:
                        if new_line[i] == new_line[i + 1]:
                            new_line[i] *= 2
                            new_line[i + 1] = 0
                            i += 2
                        else:
                            i += 1
                    new_line = list(filter(lambda x: x > 0, new_line))

                new_board.append(new_line + [0] * (n - len(new_line)))

            if depth_level_pointer < TURN_LIMIT:
                # original locations of new lines don't matter.
                dfs_stacks_by_turn[depth_level_pointer + 1].append(new_board)
            else:
                # elif depth_level_pointer == TURN_LIMIT:
                score_list.append(max(itertools.chain(*new_board)))
        # <dfs_stack_by_turn>-th turn ends.

    maximum_score = max(score_list)

    # Title: output
    result: str = str(maximum_score)
    print(result)
    return result


def test_play_2048_easy() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            ["3", "2 2 2", "4 4 4", "8 8 8"],
            ["16"],
        ],
        [
            [
                "7",
                "2 2 2 2 2 2 2",
                "2 0 2 2 2 2 2",
                "2 0 2 2 2 2 2",
                "2 0 2 2 2 2 2",
                "2 2 2 0 2 2 2",
                "2 2 2 2 2 2 0",
                "2 2 2 2 2 2 0",
            ],
            ["32"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            play_2048_easy(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


def prey_on_fishes(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Minimum sum of distances in which baby shark moved ; https://www.acmicpc.net/problem/16236

    Time Complexity (Worst-case): ...
        - O( Number( BFS(available_cells) ) ) from BFS loop until given condition is not meet.
            🛍️ e.g. when fishes to be pred on are far away from each other.
        - O(n^2) * number of times baby shark moved in from <is_explored_space>

    Space Complexity (Worst-case): O( Number(baby_shark) ) from BFS at least
        - It is O(1) according to given condition.

    Definition
        - n: size to create marine space (n*n grid).
        - Number( BFS(available_cells) ): |V| + |E|; Time occurred from available cells where baby shark can pass through in given n*n space
            - available cells are vertexes
            - adjacent available cells of a available cell are edges (up to 4 directions in each the cell)
        - Number(baby_shark): The number of virus cells in given n*n grid before virus cells spread.
            given value is fixed as one.

    Purpose
        - Simulate: situation where baby shark moves in optimized path step by step and by distance.

    Consideration
        - Baby shark starts with 2 size and size is increased in +1 if prey on fishes as many the number of current baby shark's size
        - Baby shark can only move in directions: (-1, 0), (1, 0), (0, -1), (0, 1)
        - Baby shark can only prey on fishes that have a size less than baby shark's size.
        - Baby shark can only pass through fishes that have a size equal or less than baby shark's size.
        - Precedence of fishes to be preyed on
            - smaller (distance, row, column)
        - Each cell in marine space have 0 ~ 1 number of fish that have a size.
        - Baby Shark is only one in the marine space.

    Implementation
        - Used data structure: deque in BFS
            It uses BFS every time that baby shark prey on a fish until baby shark can not prey on fish.
            : the number of remained fishes are 0  |  all size of remained fishes is bigger than baby shark.
        - It is divided in main two step.
            1. explore toward each directions from explored points (initial point is baby shark's point)
                , and collect next exploration points.
                According to the given conditions of what fish to be predated
                , there is no need to navigate all the map space each time to search the fish.
            2. analyze next exploration points.
            2.1. if fishes that baby shark can prey on exist, baby shark preys on one of fishes by the precedence.
            2.2. else, re-run "1" step.
    """
    import operator
    import sys
    from collections import deque
    from typing import Literal

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    DIRECTIONS: list[tuple[int, int]] = [
        (-1, 0),  # (-row direction)
        (1, 0),  # (+row direction)
        (0, -1),  # (-column direction)
        (0, 1),  # (+column direction)
    ]
    BABY_SHARK_SIZE: int = 2
    VALID_FISH_SIZES: set[int] = set(range(1, 7))

    class BabyShark:
        def __init__(self, point: tuple[int, int], size: int) -> None:
            self.point: tuple[int, int] = point
            self.xp: int = 0
            self.size: int = size

        def _add_xp(self) -> None:
            self.xp += 1
            if self.xp == self.size:
                self.size += 1
                self.xp = 0

        def compare_size(self, target_size: int) -> Optional[Literal[-1, 0, 1]]:
            if target_size == 0:
                return None
            elif target_size < self.size:
                # Baby Shark can prey on fish, can pass through the fish
                return 1
            elif target_size == self.size:
                # Baby Shark can not prey on fish, can pass through the fish
                return 0
            else:
                # Baby Shark can not prey on fish
                return -1

        def prey_on_fish(self, fish_point: tuple[int, int]) -> None:
            self._add_xp()
            self.point = fish_point

    # Title: input
    # condition: (2 ≤ N ≤ 20)
    n: int = int(input_())
    marine_space: list[list[int]] = [[] for _ in range(n)]
    given_fishes_count: int = 0
    baby_shark: Optional[BabyShark] = None
    for row, line in enumerate(marine_space):
        # n*n loop
        for column, x in enumerate(map(int, input_().split())):
            # EMPTY = 0, FISH = 1 ~ 6 (the size of fish), BABY_SHARK = 9
            # condition: (1 == the number of BABY_SHARK)
            match x:
                case x if x in VALID_FISH_SIZES:
                    given_fishes_count += 1
                case 9:
                    baby_shark = BabyShark(point=(row, column), size=BABY_SHARK_SIZE)
                case _:
                    pass
            line.append(x)
    if baby_shark is None:
        return ""
    distance: int = 0
    travel_distance: int = 0

    # Title: solve
    explored_deque: deque[tuple[int, int]] = deque([baby_shark.point])
    is_explored_space: list[list[bool]] = [[False for _ in range(n)] for _ in range(n)]
    is_explored_space[baby_shark.point[0]][baby_shark.point[1]] = True
    next_exploration_deque: deque[tuple[int, int]] = deque()
    next_exploration_size_comparisons: list[Optional[Literal[0, 1]]] = []
    while given_fishes_count > 0 and len(explored_deque) >= 1:
        # add points to <next_exploration_deque> from <explored_deque>
        explored_point: tuple[int, int] = explored_deque.popleft()
        for direction in DIRECTIONS:
            new_point: tuple[int, int] = tuple(
                map(operator.add, explored_point, direction)
            )
            # if it is new exploration point,
            if (
                0 <= new_point[0] < n
                and 0 <= new_point[1] < n
                and not is_explored_space[new_point[0]][new_point[1]]
            ):
                # compare whether baby shark can or not prey on any fish.
                size_comparison = baby_shark.compare_size(
                    marine_space[new_point[0]][new_point[1]]
                )
                # if baby shark can pass through the fish
                if size_comparison != -1:
                    next_exploration_deque.append(new_point)
                    next_exploration_size_comparisons.append(size_comparison)

                is_explored_space[new_point[0]][new_point[1]] = True

        # same <distance> exploration apart from baby shark ends. judge <next_exploration_deque>.
        if len(explored_deque) == 0:
            distance += 1
            valid_fishes_i: list[int] = [
                i for i, x in enumerate(next_exploration_size_comparisons) if x
            ]
            target_fish_point: tuple[int, int]

            # judge whether baby shark can prey on fishes in current distance
            if len(valid_fishes_i) < 1:
                # if baby shark can not prey on any fish in current distance
                explored_deque = next_exploration_deque.copy()
                next_exploration_deque.clear()
                next_exploration_size_comparisons.clear()
                continue
            else:
                # decide which fish to be pred on according to the priority of the index of the smaller row, column
                target_fish_point = min((next_exploration_deque[i] for i in valid_fishes_i))  # type: ignore

            # move and prey on the fish
            marine_space[baby_shark.point[0]][baby_shark.point[1]] = 0
            travel_distance += distance
            baby_shark.prey_on_fish(target_fish_point)
            marine_space[baby_shark.point[0]][baby_shark.point[1]] = 9
            given_fishes_count -= 1

            # post-process; makes variables initial state for new exploration in new Baby shark point
            distance = 0
            explored_deque = deque([baby_shark.point])
            is_explored_space = [[False for _ in range(n)] for _ in range(n)]
            is_explored_space[baby_shark.point[0]][baby_shark.point[1]] = True
            next_exploration_deque.clear()
            next_exploration_size_comparisons.clear()

    # Title: output
    result: str = str(travel_distance)
    print(result)
    return result


def test_prey_on_fishes() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "3",
                "0 0 0",
                "0 0 0",
                "0 9 0",
            ],
            ["0"],
        ],
        [
            ["3", "0 0 1", "0 0 0", "0 9 0"],
            ["3"],
        ],
        [
            [
                "4",
                "4 3 2 1",
                "0 0 0 0",
                "0 0 9 0",
                "1 2 3 4",
            ],
            ["14"],
        ],
        [
            [
                "6",
                "5 4 3 2 3 4",
                "4 3 2 3 4 5",
                "3 2 9 5 6 6",
                "2 1 2 3 4 5",
                "3 2 1 6 5 4",
                "6 6 6 6 6 6",
            ],
            ["60"],
        ],
        [
            [
                "6",
                "6 0 6 0 6 1",
                "0 0 0 0 0 2",
                "2 3 4 5 6 6",
                "0 0 0 0 0 2",
                "0 2 0 0 0 0",
                "3 9 3 0 0 1",
            ],
            ["48"],
        ],
        [
            [
                "6",
                "1 1 1 1 1 1",
                "2 2 6 2 2 3",
                "2 2 5 2 2 3",
                "2 2 2 4 6 3",
                "0 0 0 0 0 6",
                "0 0 0 0 0 9",
            ],
            ["39"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            prey_on_fishes(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


def block_virus_from_leaking(input_lines: Optional[Iterator[str]] = None) -> str:
    """🚤 get Maximum the number of secure cells in given situation ; https://www.acmicpc.net/problem/14502

    Time Complexity (Worst-case): ...
        - O( 3 * ( Number(empty_cells) choose 3 ) from combination operation
            3 is given <WALL_COUNT>
        - O( Combinations(empty_cells) ) from loop
            *  O( Number( BFS(spread_virus_cells) ) ) from BFS loop

    Space Complexity (Worst-case): O( Number(given_virus_cell) ) from BFS at least
        🛍️ e.g. when the number of wall cells is minimum and (virus cells, empty cells) is maximum.

    Definition
        - n, m: size to create space of laboratory (n*m grid).
        - Combinations(empty_cells): The number of combinations in which 3 empty cells are replaced with wall cell.
        - Number(empty_cells): The number of empty cells in given n*m grid.
        - Number(given_virus_cell): The number of virus cells in given n*m grid before virus cells spread.
        - Number( BFS(spread_virus_cells) ): |V| + |E|; Time occurred from spread virus cells including given virus cells in given n*m grid
            - spread virus cells are vertexes
            - adjacent virus cells of a virus cell are edges (up to 4 directions in each the cell)

    Purpose
        - Simulate: situations for each available combination where given all walls are constructed.
            and select maximum number of empty cells.

    Consideration
        - ❔ Are there cases where walls can not be constructed because there is not enough empty cell?
            No. minimum the number of empty cell is 3 in given condition.
        - Virus cell can only spread in directions: (-1, 0), (1, 0), (0, -1), (0, 1)

    Implementation
        - Used data structure: deque in BFS (It can be also implemented by using DFS.)
            It uses BFS that visits from right edges to left edge (except initial deque) for each initial virus entry points.
        - If points and cell in given space are represented as NamedTuple and Enum type,
            , it causes considerable delay because this algorithm calculates many operations for each simulated case.
        - This solution does not require <is_explored> variable in BFS.
            because virus cells are input in deque before explore, and spread one by one
            , and the condition of spread is that it judges whether adjacent cells are only empty cells
            , but operations to do in other branch of condition is none.
        - It uses list for backtracking to recover changed cell types for each simulation.
            because cost of deep copy is expensive, especially in combinations loop.
        - Brute-force makes sense; itertools.combinations
        - It can be also implemented in the way like event propagation, but it is slower 2 times than current implementation.
            before explore in a direction, filter some directions of surrounding points toward the wall points
            , and if edge points, filter some directions in order to prevent the out of range.
            , and copy this original <explored_directions_space> and use copied variable
            , and after virus_entry_point.get(), filter some directions of surrounding points toward the entry point.
    """
    import itertools
    import operator
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    DIRECTIONS: list[tuple[int, int]] = [
        (-1, 0),  # (-row direction)
        (1, 0),  # (+row direction)
        (0, -1),  # (-column direction)
        (0, 1),  # (+column direction)
    ]
    WALL_COUNT: int = 3

    # Title: input
    # condition: (3 ≤ N, M ≤ 8)
    n, m = map(int, input_().split())
    lab_space: list[list[int]] = [[] for _ in range(n)]
    # virus spaces will be entry point for graph search.
    virus_points: list[tuple[int, int]] = []
    empty_points: list[tuple[int, int]] = []
    for row, line in enumerate(lab_space):
        # n*m loop
        for column, x in enumerate(map(int, input_().split())):
            # EMPTY = 0, WALL = 1, # VIRUS = 2
            match x:
                case 0:
                    # condition: (3 ≤ the number of EMPTY)
                    empty_points.append((row, column))
                case 1:
                    # condition: (3 == the number of WALL)
                    pass
                case 2:
                    # condition: (2 ≤ the number of VIRUS ≤ 10)
                    virus_points.append((row, column))
                case _:
                    pass
            line.append(x)
    initial_empty_point_len: int = len(empty_points) - WALL_COUNT
    maximum_empty_points_count: int = 0

    # Title: solve
    # simulate
    empty_points_len_list: list[int] = []
    for wall_point_comb in itertools.combinations(empty_points, WALL_COUNT):
        explored_deque: deque[tuple[int, int]] = deque(virus_points)
        backtracking_to_empty: list[tuple[int, int]] = []
        for wall_point in wall_point_comb:
            # construct wall on empty points
            lab_space[wall_point[0]][wall_point[1]] = 1
            backtracking_to_empty.append(wall_point)
        empty_point_len: int = initial_empty_point_len

        # spread virus; Graph search from <virus_points>
        while len(explored_deque) > 0:
            explored_point = explored_deque.popleft()
            for direction in DIRECTIONS:
                new_point: tuple[int, int] = tuple(
                    map(operator.add, explored_point, direction)
                )

                # check that <new_point> can be spread by virus.
                if (
                    0 <= new_point[0] < n
                    and 0 <= new_point[1] < m
                    and lab_space[new_point[0]][new_point[1]] == 0
                ):
                    # spread virus cell to empty cell.
                    lab_space[new_point[0]][new_point[1]] = 2
                    explored_deque.appendleft(new_point)
                    empty_point_len -= 1
                    backtracking_to_empty.append(new_point)
        empty_points_len_list.append(empty_point_len)

        # post-process. original <lab_space> will be recovered by backtracking.
        for point in backtracking_to_empty:
            lab_space[point[0]][point[1]] = 0
    maximum_empty_points_count = max(empty_points_len_list)

    # Title: output
    result: str = str(maximum_empty_points_count)
    print(result)
    return result


def test_block_virus_from_leaking() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "7 7",
                "2 0 0 0 1 1 0",
                "0 0 1 0 1 2 0",
                "0 1 1 0 1 0 0",
                "0 1 0 0 0 0 0",
                "0 0 0 0 0 1 1",
                "0 1 0 0 0 0 0",
                "0 1 0 0 0 0 0",
            ],
            ["27"],
        ],
        [
            ["4 6", "0 0 0 0 0 0", "1 0 0 0 0 2", "1 1 1 0 0 2", "0 0 0 0 0 2"],
            ["9"],
        ],
        [
            [
                "8 8",
                "2 0 0 0 0 0 0 2",
                "2 0 0 0 0 0 0 2",
                "2 0 0 0 0 0 0 2",
                "2 0 0 0 0 0 0 2",
                "2 0 0 0 0 0 0 2",
                "0 0 0 0 0 0 0 0",
                "0 0 0 0 0 0 0 0",
                "0 0 0 0 0 0 0 0",
            ],
            ["3"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            block_virus_from_leaking(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


def deliver_chicken(input_lines: Optional[Iterator[str]] = None) -> str:
    """get Minimum of sum of distances between houses and chickens places in given space ; https://www.acmicpc.net/problem/15686

    Decision problem
        ❔ Given a positive integer m, is there a way to choose m chicken places out of the given set of mm places
        , such that the total distance between the selected chicken places and all the houses in the city is at most d?

    Time Complexity (Worst-case): ...
        - O( m * (mm choose m) ) from combination operation
        - O( Combinations(m) ) from loop
            *  O( Number(houses) ) from loop
            *  O(m) from min() function

    Space Complexity (Worst-case): O(1)

    Definition
        - n: size to create space of city (n*n space).
        - m: the number of chicken places that will not be shut down.
        - mm: the number of chicken places in given n*n space.
        - Combinations(m): The number of combinations in which only <m> of <mm> chicken houses remain.
        - Number(houses): The number of houses in given n*n space.
        - Distance(x, y): | x.row - y.row | + | x.column - y.column |

    Purpose
        - Simulate: Assuming that some of chicken places given are missing
            and compare distances between houses and chickens places
            and select minium sum of distances.

    Implementation
        - To create N*N array is unnecessary.
            This algorithms uses only information of houses and chicken places.
        - Brute-force makes sense; itertools.combinations
    """
    import itertools
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (2 ≤ N ≤ 50), (1 ≤ M ≤ 13)
    n, m = map(int, input_().split())
    houses: list[tuple[int, int]] = []
    chicken_places: list[tuple[int, int]] = []
    for row in range(1, n + 1):
        # n*n loop
        for column, x in enumerate(input_().split(), start=1):
            # EMPTY = "0", HOUSE = "1", CHICKEN_PLACE = "2"
            # condition: (1 ≤ the number of chicken places in n*n space ≤ 13)
            # condition: (1 ≤ the number of houses in n*n space ≤ 2*N)
            if x == "1":
                houses.append((row, column))
            elif x == "2":
                chicken_places.append((row, column))
    minimum_total_chicken_distance: int = 0

    # Title: solve
    total_distance_list: list[int] = []
    for chicken_place_comb in itertools.combinations(chicken_places, m):
        total_distance: int = 0
        for house in houses:
            total_distance += min(
                (
                    abs(house[0] - chicken_place[0]) + abs(house[1] - chicken_place[1])
                    for chicken_place in chicken_place_comb
                )
            )
        total_distance_list.append(total_distance)
    minimum_total_chicken_distance = min(total_distance_list, key=None)

    # Title: output
    result: str = str(minimum_total_chicken_distance)
    print(result)
    return result


def test_deliver_chicken() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            ["5 3", "0 0 1 0 0", "0 0 2 0 1", "0 1 2 0 0", "0 0 1 0 0", "0 0 0 0 2"],
            ["5"],
        ],
        [
            ["5 2", "0 2 0 1 0", "1 0 1 0 0", "0 0 0 0 0", "2 0 0 1 1", "2 2 0 1 2"],
            ["10"],
        ],
        [
            ["5 1", "1 2 0 0 0", "1 2 0 0 0", "1 2 0 0 0", "1 2 0 0 0", "1 2 0 0 0"],
            ["11"],
        ],
        [
            ["5 1", "1 2 0 2 1", "1 2 0 2 1", "1 2 0 2 1", "1 2 0 2 1", "1 2 0 2 1"],
            ["32"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            deliver_chicken(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


# week 1-1: Greedy: 5
def schedule_multi_tap(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Minimum the count to unplug appliance on multi-tap ; https://www.acmicpc.net/problem/1700

    Time Complexity (Worst-case): O(k*n) (pseudo-polynomial time because of k)
        - O(k) from appliances loop
            *  O(n) from (<appliances_in_use> loop, max() function)

    Space Complexity (Worst-case): O(1)

    Definition
        - n: the number of sockets on the multi-tap.
        - k: the sum of the number of times that each appliance will be used.

    Purpose
        - Which appliance should I unplug from appliances in use on multi-tap?

    Consideration
        - ❔ Is the number of appliance for each appliance meaningful to determine appliance to be unplugged?
            There is no relationship.
            It is only related with next index of same appliance to be used for each appliance in use.
            One of those with the biggest index is the target to be unplugged. (; "those" are <comparison_indexes>)
        - When any socket on multi-tap is empty.
        - When next appliance to be used is already plugged on multi-tap.
        - When consecutive appliances is same.

    Implementation
        - Used data structure: Deque for each appliance.
            The algorithm compares appliance to be used and schedules in order of given appliance list.
            To use Deque makes that to set "search range" is unnecessary.
        - When plug initially all available appliances to <appliances_in_use>
            , "if x in slicing" in loop is inefficient if <appliances_in_use> was initialized as many the number of sockets
            , because slicing operator always shallow copy of a list.
    """
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (1 ≤ N, K ≤ 100)
    n, k = map(int, input_().split())
    # condition: appliance name is Natural Number equal or less than K.
    appliances: list[int] = list(map(int, input_().split()))
    # -1 index indicates that the multi-tap socket is not currently being used.
    appliances_in_use: list[int] = []
    # 0 index will not be used because K is Natural Number. <appliances_i_queue> will be treated as fixed size.
    appliances_i_queue: list[deque[int]] = [deque() for _ in range(k + 1)]
    minimum_unplugged_count: int = 0

    # Title: solve
    # create queues that have index of appliances sorted in ascending order for each appliance
    for i, appliance in enumerate(appliances):
        appliances_i_queue[appliance].append(i)

    # plug electrical appliances into multi-tap until all sockets on the multi-tap are full.
    i: int = 0
    for i, appliance in enumerate(appliances):
        if appliance in appliances_in_use:
            appliances_i_queue[appliance].popleft()
        else:
            appliances_in_use.append(appliance)
            # if full of sockets on multi-tap
            if len(appliances_in_use) >= n:
                break

    # plan to unplug
    for i in range(i + 1, len(appliances)):
        # filter appliances already in sockets on the multi-tap
        if appliances[i] in appliances_in_use:
            appliances_i_queue[appliances[i]].popleft()
            continue

        # choose socket to be unplugged
        comparison_indexes: list[int] = []
        for x in appliances_in_use:
            if len(appliances_i_queue[x]) > 1:
                comparison_indexes.append(appliances_i_queue[x][1])
            else:
                # If no plan to use same appliance, set index temporarily to max
                comparison_indexes.append(sys.maxsize)

        socket_to_be_unplug: int = max(
            (x for x in range(len(comparison_indexes))),
            key=lambda x: comparison_indexes[x],
        )
        unplug_target: int = appliances_in_use[socket_to_be_unplug]

        # unplug
        appliances_i_queue[unplug_target].popleft()
        appliances_in_use[socket_to_be_unplug] = appliances[i]
        minimum_unplugged_count += 1

    # Title: output
    result: str = str(minimum_unplugged_count)
    print(result)
    return result


def test_schedule_multi_tap() -> None:
    """Debugging
    =====
    3 9
    1 2 1 1 2 3 5 2 1
    * * - - - * ?
    * *         * - -   +1
    =====
    3 9
    1 2 3 4 2 3 3 4 5
    * * * ? ~ ~
      * * * - - - ?     +1
        * *       *     +1
          *       * *   +1
    """
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [["2 5", "1 2 3 2 1"], ["2"]],
        [["2 7", "2 3 2 3 1 2 7"], ["2"]],
        [["3 9", "1 2 1 1 2 3 5 2 1"], ["1"]],
        [["3 9", "1 2 3 4 2 3 3 4 5"], ["2"]],
        [["3 5", "1 1 1 1 1"], ["0"]],
        [
            [
                "3 100",
                "56 71 70 25 52 77 76 8 68 71 51 65 13 23 7 16 19 54 95 18 86 74 29 76 61 93 44 96 32 72 64 19 50 49 22 14 7 64 24 83 6 3 2 76 99 7 76 100 60 60 6 50 90 49 27 51 37 61 16 84 89 51 73 28 90 77 73 39 78 96 78 13 92 54 70 69 62 78 7 75 30 67 97 98 19 86 90 90 2 39 41 58 57 84 19 8 52 39 26 7",
            ],
            ["80"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            schedule_multi_tap(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")


def thieve_jewels(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Maximum sum of value of available jewels ; https://www.acmicpc.net/problem/1202

    Time Complexity (Worst-case): O(n(log n) + O(k log k))
        - O(n(log n)) + O(k(log k)) from Tim sort
            n is the number of jewels, k is the number of bags.
        - O(k) from bag loop  *
            ( O(1) comparison from Jewel consumed iteration  +  O(log k) from Hip (pop | push) )

    Space Complexity (Worst-case): O(n) + O(k) from Tim sort

    Definition
        - n: the number of jewels.
        - k: the number of bags.

    Purpose
        - to maximize value of jewels, a bag should select available Jewel with highest value in each iteration.

    Consideration
        - One bag can select a unique Jewel.
        - ❔ should I check all available Jewels in every bag?
            Inefficient. in the case, if a bag select a jewel it should updates all available Jewels in remained bag.
        - ❔ Which bag should I check a jewel first?
            If a bag with a large allowance is first, it is difficult to choose jewel for a smaller bag.
        - When the number of bag is greater than jewel.

    Implementation
        - Used data structure: Heap
            - Min heap for Jewels.
            - Max heap for available values of Jewels by bag through ad-hoc.
        - To explore <jewel_list> and <bag_list> sorted in ascending order makes that:
            - Once explored jewel's weight will be not important in remained bag's iteration.
                because any added jewel into Max heap can be put in any remained bags.
        - This problem is variant of Knapsack problem.
    """
    import heapq
    import sys
    from typing import NamedTuple

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    class Jewel(NamedTuple):
        weight: int
        value: int

    class Bag(NamedTuple):
        allowance: int

    # Title: input
    # condition: (1 ≤ N, K ≤ 300,000)
    n, k = map(int, input_().split())
    # condition Jewel: (0 ≤ weight, value ≤ 1,000,000)
    jewel_list: list[Jewel] = [Jewel(*map(int, input_().split())) for _ in range(n)]
    # condition bag: (0 ≤ weight_allowance ≤ 100,000,000)
    bag_list: list[Bag] = [Bag(int(input_())) for _ in range(k)]
    maximum_total_value: int = 0

    # Title: solve
    jewel_list.sort()
    bag_list.sort()
    checked_jewel_value_heapq: list[int] = []
    for bag in bag_list:
        while jewel_list and bag.allowance >= jewel_list[0].weight:
            heapq.heappush(checked_jewel_value_heapq, -jewel_list[0].value)
            heapq.heappop(jewel_list)
        if checked_jewel_value_heapq:
            maximum_total_value += -heapq.heappop(checked_jewel_value_heapq)

    # Title: output
    result: str = str(maximum_total_value)
    print(result)
    return result


def test_thieve_jewels() -> None:
    """Debugging
    =====
    4 3
    2 4 6

    Jewels
    -----
    weight      value
    1            65
    2            99
    5            23
    8            44

    Bag
    -----
    allowance
    2, 4, 6

    2   -> (2, 99)
    4   -> (1, 65)
    10  -> (8, 44)
    """
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [["2 1", "5 10", "100 100", "11"], ["10"]],
        [["3 2", "1 65", "5 23", "2 99", "10", "2"], ["164"]],
        [["4 3", "1 65", "2 99", "5 23", "8 44", "2", "4", "10"], ["208"]],
    ]:
        start_time = time.time()
        test_case.assertEqual(thieve_jewels(iter(input_lines)), "\n".join(output_lines))
        print(f"elapsed time: {time.time() - start_time}")


def make_bigger(input_lines: Optional[Iterator[str]] = None) -> str:
    """🚤 get Maximum Integer by deleting digits as many "k" number ; https://www.acmicpc.net/problem/2812

    Time Complexity (Worst-case): O(n)
        - O(n) from characters loop  *  O(1) from some operations
            some operations: comparisons, (append, pop) in stack

    Space Complexity (Worst-case): O(1)

    Definition
        - n: length of a number.
        - k: the number of characters to be deleted.
        - kk: the number of remaining characters to be deleted.

    Consideration
        - When the same number is consecutive.
        - When the next explored number is greater than previous explored number.
            - if kk > 0 or not.

    Implementation
        - characters can be compared without having to convert "str" to "int" as Unicode.
            Unicode Integer value: ord("0") is 48, ord ("9") is 57.
        - It can be implemented by using function max().
            but in this case, Time Complexity (Worse-case) is O(kk+1) * O(kk) from max() function.
            because it causes to compare duplicated range.
            For each loop, the search range is set to kk+1, it picks one by one from max() function.
            the algorithm not removes picked characters.
        - Used data structure: Stack
            Quoting "while kk > 0 and char_stack and char_stack[-1] < n_digits[i]:"
            , it may be required to post-process to slice elements added in last after loop.
            see <def test_make_bigger()>
    """
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (1 ≤ K < N ≤ 500,000)
    n, k = map(int, input_().split())
    # condition: <n digit number> not starts with 0
    n_digits: str = input_()
    kk: int = k
    char_stack: list[str] = []
    maximum_digits: str = ""

    # Title: solve
    for i in range(n):
        while kk > 0 and char_stack and char_stack[-1] < n_digits[i]:
            char_stack.pop()
            kk -= 1
        char_stack.append(n_digits[i])
    maximum_digits = "".join(char_stack[: n - k])
    # same as ... = "".join(char_stack[:-kk]) if kk > 0 else "".join(char_stack)

    # Title: output
    print(maximum_digits)
    return maximum_digits


def test_make_bigger() -> None:
    """Debugging
    5 4
    0 1 4 3 5
    -----
    char_stack    kk
    : 0           4
    : 1           3
    : 4           2
    : 4, 3        2
    : 5           0

    =====
    7 3
    1 2 3 2 4 3 4
    -----
    char_stack    kk
    : 1           3
    : 2           2
    : 3           1
    : 3, 2        1
    : 3, 4        0

    =====
    10 8
    4 1 7 7 2 5 2 8 4 1
    -----
    char_stack    kk
    : 4           8
    : 4, 1        8
    : 7           6
    : 7, 7        6
    : 7, 7, 2     6
    : 7, 7, 5     5
    : 7, 7, 5, 2  5
    : 8           1
    : 8, 4        1
    : 8, 4, 1     1
    """
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [["5 4", "01435"], ["5"]],
        [["7 3", "1232434"], ["3434"]],
        [["10 8", "4177252841"], ["84"]],
    ]:
        start_time = time.time()
        test_case.assertEqual(make_bigger(iter(input_lines)), "\n".join(output_lines))
        print(f"elapsed time: {time.time() - start_time}")


def sort_cards(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Minimum sum of the number of comparisons to sort cards ; https://www.acmicpc.net/problem/1715

    Time Complexity (Worst-case): O(n(log n))
        - O(n) from heapify function
        - O(n-1) from loop  *  3*O(log n) from Hip (pop | push)

    Space Complexity (Worst-case): O(1)

    Definition
        - n: the number of card stack groups.
        - CardStacks(i): the number of cards of i-th card stack.
        - Sum(i): sum of the number of comparisons to cards until i-th merge.

    Recurrence relation
        - Sum(i) ::=
            - if n == 1, 0
            - if n > 1, Sum(i-1) + ( Sum(i-1) + CardStacks(i) )

    Purpose
        - to minimize Sum(x), it requires to select minimum CardStacks(x) in lower merge level.

    Implementation
        - Used data structure: Heap
            - If n == 1, It is not need to compare card stacks. so <result> is 0.
            - If n > 1, Regardless of the number of remained card stacks is odd or even
                , it required to merge with two smallest cards stack in sequence.
    """
    import heapq
    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (1 ≤ N ≤ 100,000)
    n: int = int(input_())
    # condition: (1 ≤ CardStacks(i) ≤ 1000)
    card_stack_list: list[int] = [int(input_()) for _ in range(n)]
    minimum_total_comparison_count: int = 0

    # Title: solve
    heapq.heapify(card_stack_list)
    # merge
    while len(card_stack_list) > 1:
        a, b = heapq.heappop(card_stack_list), heapq.heappop(card_stack_list)
        merge_value: int = a + b
        minimum_total_comparison_count += merge_value
        heapq.heappush(card_stack_list, merge_value)

    # Title: output
    result: str = str(minimum_total_comparison_count)
    print(result)
    return result


def test_sort_cards() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [["4", "30", "40", "50", "100"], ["410"]],
        [["4", "30", "40", "50", "60"], ["360"]],
        [["8", "30", "40", "50", "20", "10", "100", "60", "120"], ["1160"]],
        [["8", "30", "40", "50", "20", "10", "100", "60", "10"], ["860"]],
        [["4", "120", "40", "100", "20"], ["500"]],
    ]:
        start_time = time.time()
        test_case.assertEqual(sort_cards(iter(input_lines)), "\n".join(output_lines))
        print(f"elapsed time: {time.time() - start_time}")


def assign_lecture_room(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Minimum the number of lecture rooms that make all lecture available ; https://www.acmicpc.net/problem/11000

    Time Complexity (Worst-case): O(n(log n))
        - O(n(log n)) from Tim sort
        - O(n-1) from loop  *  ( O(1) comparison  +  O(log n) from Hip (pop | push) at least )

    Space Complexity (Worst-case): O(n) from Tim sort

    Definition
        - n: the number of lectures.

    Types
        - varaint of Interval scheduling problem

    Implementation
        - It uses sort for input data in order to compare <end time> of lecture in order.
        - Used data structure: Heap (Python heapq library uses min heap)
            if it uses simple list, it must compare all <end time> as many as lecture rooms. so inefficient.
    """
    import heapq
    import sys
    from typing import NamedTuple

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    class Period(NamedTuple):
        start: int
        end: int

    # Title: input
    # condition: (1 ≤ N ≤ 200,000)
    n: int = int(input_())
    # condition: (0 ≤ lecture.start < lecture.end ≤ 10^9)
    lecture_period_list: list[Period] = [
        Period(*map(int, input_().split())) for _ in range(n)
    ]
    minimum_total_lecture_room: int = 0

    # Title: solve
    lecture_period_list.sort()
    lecture_end_time_heapq: list[int] = [lecture_period_list[0].end]
    for i in range(1, n):
        if lecture_end_time_heapq[0] <= lecture_period_list[i].start:
            heapq.heapreplace(lecture_end_time_heapq, lecture_period_list[i].end)
        else:
            heapq.heappush(lecture_end_time_heapq, lecture_period_list[i].end)
    minimum_total_lecture_room = len(lecture_end_time_heapq)

    # Title: output
    result: str = str(minimum_total_lecture_room)
    print(result)
    return result


def test_assign_lecture_room() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [["3", "1 3", "2 4", "3 5"], ["2"]],
        [["8", "1 8", "9 16", "3 7", "8 10", "10 14", "5 6", "6 11", "11 12"], ["3"]],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            assign_lecture_room(iter(input_lines)), "\n".join(output_lines)
        )
        print(f"elapsed time: {time.time() - start_time}")
