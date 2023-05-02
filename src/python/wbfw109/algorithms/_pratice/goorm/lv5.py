from __future__ import annotations

import time
import unittest
from typing import Iterator, Optional


def determine_distribution_center(
    input_lines: Optional[Iterator[str]] = None,
) -> str:
    """https://level.goorm.io/exam/47875/%EB%AC%BC%EB%A5%98%EC%84%BC%ED%84%B0/quiz/1

    Time Complexity (Worst-case): O( BFS(tree) ) (*3)

    Space Complexity (Worst-case): O( BFS(tree) )

    Key point
        ❔ when changing distribution center, What are variables to affect total cost?
        ➡️ <subtree_size_by_neighbor> can be obtained by
            , calculating the found <subtree_size> for each node at some point in BFS
            , similar with topological sorting.
    Tag
        Dynamic programming, Topological sort, BFS
    """
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # graph is weighted tree, and <v> starts from "0"
    v_count: int = int(input_())
    subtree_size_by_neighbor: list[dict[int, int]] = [{} for _ in range(v_count)]
    tree_metric: list[list[tuple[int, int]]] = [[] for _ in range(v_count)]
    # if value > 1 in <traces_as_degree>, it denotes "not explored".
    traces_as_degree: list[int] = [0] * v_count
    found_subtree_size: list[int] = [1] * v_count

    for _ in range(v_count - 1):
        v1, nv, weight = map(int, input_().split())
        tree_metric[v1].append((nv, weight))
        tree_metric[nv].append((v1, weight))
        traces_as_degree[v1] += 1
        traces_as_degree[nv] += 1
        subtree_size_by_neighbor[v1][nv] = 0
        subtree_size_by_neighbor[nv][v1] = 0

    # Title: solve
    ## find leaves and BFS which fills <subtree_size_by_neighbor> similar with topological sorting
    # route state: tuple[v, found subtree size]
    leaves: list[int] = []
    for v, degree in enumerate(traces_as_degree):
        if degree == 1:
            leaves.append(v)

    dq1: deque[int] = deque(leaves)
    while dq1:
        ev = dq1.popleft()
        for nv, _ in tree_metric[ev]:
            if traces_as_degree[nv] > 0:
                traces_as_degree[nv] -= 1
                traces_as_degree[ev] -= 1
                subtree_size_by_neighbor[nv][ev] = found_subtree_size[ev]
                subtree_size_by_neighbor[ev][nv] = v_count - found_subtree_size[ev]

                # <found_subtree_size> in order to calculate <subtree_size_by_neighbor>
                found_subtree_size[nv] += found_subtree_size[ev]
                if traces_as_degree[nv] == 1:
                    dq1.append(nv)

    total_costs: list[int] = []
    ## Simulate from a random <v>
    # start_v
    v: int = 0
    total_cost: int = 0
    traces: list[bool] = [False] * v_count
    traces[v] = True
    # dq[tuple[vertex, cumulated cost]]
    dq2: deque[tuple[int, int]] = deque([(v, 0)])
    while dq2:
        ev, cumulated_cost = dq2.popleft()
        for nv, weight in tree_metric[ev]:
            if not traces[nv]:
                traces[nv] = True
                cost = cumulated_cost + weight
                total_cost += cost
                dq2.append((nv, cost))
    else:
        total_costs.append(total_cost)

    # Simulate with calculating new total cost
    total_cost: int = 0
    traces: list[bool] = [False] * v_count
    traces[v] = True
    # dq[tuple[vertex, total cost in the vertex]]
    dq3: deque[tuple[int, int]] = deque([(v, total_costs[0])])
    while dq3:
        ev, total_cost = dq3.popleft()
        for nv, weight in tree_metric[ev]:
            if not traces[nv]:
                traces[nv] = True
                increased_cost: int = weight * (subtree_size_by_neighbor[nv][ev] - 1)
                decreased_cost: int = weight * (subtree_size_by_neighbor[ev][nv] - 1)
                new_total_cost: int = total_cost + increased_cost - decreased_cost
                total_costs.append(new_total_cost)
                dq3.append((nv, new_total_cost))

    # Title: output
    result: str = str(min(total_costs))
    print(result)
    return result


def test_determine_distribution_center() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "5",
                "0 1 2",
                "1 2 1",
                "1 3 7",
                "3 4 5",
            ],
            ["22"],
        ],
        [
            [
                "6",
                "0 1 1",
                "1 2 4",
                "2 3 1",
                "3 4 4",
                "4 5 1",
            ],
            ["21"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            determine_distribution_center(iter(input_lines)),
            "\n".join(output_lines),
        )
        print(f"elapsed time: {time.time() - start_time}")


test_determine_distribution_center()
