from typing import Iterator, Optional


def move_alphabet_piece(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/198"""
    import sys
    from array import array
    from collections.abc import MutableSequence

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
    START_POINT: tuple[int, int] = (0, 0)
    AlphabetBoard = list[list[int]]

    # Title: input
    # condition: (1 ≤ N,M ≤ 20)
    n, m = map(int, input_().split())
    alphabet_board: AlphabetBoard = [[ord(x) - 65 for x in input_()] for _ in range(n)]
    dfs_stacks: list[tuple[int, int]] = [START_POINT]
    is_explored_space: list[list[bool]] = [[False for _ in range(m)] for _ in range(n)]
    # len(remained_branches_counts) is depth level pointer
    remained_branches_counts: list[int] = [len(dfs_stacks)]
    backtracking_by_stack: list[list[tuple[int, int]]] = [[]]
    is_in_ascii_list: list[bool] = [False for _ in range(26)]
    passing_ascii: MutableSequence[int] = array("b", [])
    passing_distance_list: MutableSequence[int] = array("b", [])
    maximum_passing_distance: int = 0

    # Title: solve
    while len(dfs_stacks) > 0:
        discovered_point = dfs_stacks.pop()
        remained_branches_counts[-1] -= 1

        if not is_explored_space[discovered_point[0]][discovered_point[1]]:
            is_explored_space[discovered_point[0]][discovered_point[1]] = True
            backtracking_by_stack[-1].append(discovered_point)

            # check that the node meets given conditions.
            if not is_in_ascii_list[
                alphabet_board[discovered_point[0]][discovered_point[1]]
            ]:
                is_in_ascii_list[
                    alphabet_board[discovered_point[0]][discovered_point[1]]
                ] = True
                passing_ascii.append(
                    alphabet_board[discovered_point[0]][discovered_point[1]]
                )

                # check whether look deeper (almost true because nodes of DFS is evaluated after pop()).
                new_points: list[tuple[int, int]] = []
                for direction in DIRECTIONS:
                    new_point: tuple[int, int] = (
                        discovered_point[0] + direction[0],
                        discovered_point[1] + direction[1],
                    )
                    if 0 <= new_point[0] < n and 0 <= new_point[1] < m:
                        new_points.append(new_point)
                if new_points:
                    dfs_stacks.extend(new_points)
                    remained_branches_counts.append(len(new_points))
                    backtracking_by_stack.append([])

        # backtracking when leaves nodes are explored if remained root nodes to be explored exist
        while remained_branches_counts and remained_branches_counts[-1] == 0:
            # judge <passing_elements> of this route
            passing_distance_list.append(len(passing_ascii))

            # backtracking
            if len(passing_ascii) > 0:
                # if parent node exists
                is_in_ascii_list[passing_ascii.pop()] = False
            remained_branches_counts.pop()
            for point in backtracking_by_stack[-1]:
                is_explored_space[point[0]][point[1]] = False
            backtracking_by_stack.pop()

            if len(backtracking_by_stack) > 0:
                # if parent node exists
                # <backtracking_by_stack[-1][-1]> is the point of node that has explored all of its child nodes.
                is_explored_space[backtracking_by_stack[-1][-1][0]][
                    backtracking_by_stack[-1][-1][1]
                ] = False
    else:
        maximum_passing_distance = max(passing_distance_list)

    # Title: output
    sys.stdout.write(str(maximum_passing_distance))
    return str(maximum_passing_distance)
