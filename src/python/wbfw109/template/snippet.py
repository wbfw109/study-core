from typing import Any, TypeVar

T = TypeVar("T")

# Title: variables in 2D space
n, p = 10, [1, 5]
# (left, right, bottom, top) directions from a point
for nx, ny in ((p[0] + dx, p[1] + dy) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))):
    pass
# 3*3 adjacent points from a point (i, j)
for nx, ny in ((p[0] + dx, p[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2)):
    if 0 <= nx < n and 0 <= ny < n:
        pass

# Title: transpose iteration ~
TwoDIterator = list[list[Any]]


def get_square_iterators_against_gravity(
    line_len: int,
) -> list[TwoDIterator]:
    """returns n*n (<line_len>*<line_len>) iterators against gravity by points.

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
