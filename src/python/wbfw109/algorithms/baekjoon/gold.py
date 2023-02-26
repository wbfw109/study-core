import unittest
from pprint import pprint
from typing import Iterator, Optional


# week 1-2: Implementation
def escape_marbles_2(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/13460"""


def play_2048_easy(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/12100"""


def prey_on_fishes(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/16236"""


def block_virus_from_leaking(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/14502"""


def deliver_chicken(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/15686"""


# week 1-1: Greedy
def schedule_multi_tap(input_lines: Optional[Iterator[str]] = None) -> str:
    """get Minimum the count to unplug appliance on multi-tap ; https://www.acmicpc.net/problem/1700

    Time Complexity (Worst-case): O(k*n)
        - O(k) from appliances loop  *  O(n) from max() function
            🛍️ e.g. when all appliances is different type

    Space Complexity: O(1)

    Purpose
        - Which appliance should I unplug from appliances in use on multi-tap?
    Consideration
        - ❔ Is the number of appliance for each appliances meaningful to determine appliance to be unplugged?
            There is no relationship.
            It is only related with next index of same appliance to be used for each appliances in use.
            One of those with the biggest index is the target to be unplugged. (; "those" are <comparison_indexes>)
        - When any socket on multi-tap is empty.
        - When next appliance to be used is already plugged on multi-tap.
        - When consecutive appliances is same.

    Implementation
        - Used data structure: Queue for each appliances.
            The algorithm compares appliance to be used and schedules in order of given appliance list.
            To use Queue makes that to set "search range" is not need to.

            but instead I used Deque because Python Queue is not subscriptable like queue[1]
    """
    import sys
    from collections import deque

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition: (1 ≤ N, K ≤ 100).
    # N is the number of sockets on the multi-tap.
    # K is the sum of the number of times that each appliance will be used.
    n, k = map(int, input_().split())
    # condition: appliance name is Natural Number equal or less than K.
    appliance_list: list[int] = list(map(int, input_().split()))
    result: int = 0
    # -1 index indicates that the multi-tap socket is not currently being used.
    appliances_in_use: list[int] = [-1 for _ in range(n)]
    # 0 index will not be used because K is Natural Number. <appliances_i_queue> will be treated as fixed size.
    appliances_i_queue: list[deque[int]] = [deque() for _ in range(k + 1)]

    # Title: solve
    # create queues that have index of appliances sorted in ascending order for each appliances
    for i, appliance in enumerate(appliance_list):
        appliances_i_queue[appliance].append(i)

    # plug electrical appliances into multi-tap until all sockets on the multi-tap are full.
    i: int = -1
    j: int = 0
    for i, appliance in enumerate(appliance_list):
        if appliance in appliances_in_use[: j + 1]:
            appliances_i_queue[appliance].popleft()
        else:
            appliances_in_use[j] = appliance
            j += 1

        # if full of sockets on multi-tap
        if j >= n:
            break

    # plan to unplug
    start_i: int = i + 1
    for i, appliance in enumerate(appliance_list[start_i:], start=start_i):
        # filter appliances already in sockets on the multi-tap
        if appliance in appliances_in_use:
            appliances_i_queue[appliance].popleft()
            continue

        # choose socket to be unplugged
        comparison_indexes: list[int] = []
        for appliance_in_use in appliances_in_use:
            if len(appliances_i_queue[appliance_in_use]) > 1:
                comparison_indexes.append(appliances_i_queue[appliance_in_use][1])
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
        appliances_in_use[socket_to_be_unplug] = appliance
        result += 1

    # Title: output
    print(result)
    return str(result)


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
        test_case.assertEqual(schedule_multi_tap(iter(input_lines)), output_lines[0])


test_schedule_multi_tap()


def thieve_jewels(input_lines: Optional[Iterator[str]] = None) -> str:
    """get Maximum sum of value of available jewels ; https://www.acmicpc.net/problem/1202

    Time Complexity (Worst-case): O(n(log n) + O(k log k))
        - O(n(log n)) + O(k(log k)) from Tim sort
            n is the number of jewels, k is the number of bags.
        - O(k) from bag loop  *
            ( O(1) comparison from Jewel consumed iteration  +  O(log k) from Hip (pop | push) )

    Space Complexity: O(1)

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
        - It uses Min heap on Jewels.
        - It uses Max heap on available values of Jewels by bag through ad-hoc.
        - To explore <jewel_list> and <bag_list> sorted in ascending order makes that:
            - Once explored jewel's weight will be not important in remained bag's iteration.
                because any added jewel into Max heap can be put in any remained bags.
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
    result: int = 0

    # Title: solve
    jewel_list.sort()
    bag_list.sort()
    checked_jewel_value_heap: list[int] = []
    for bag in bag_list:
        while jewel_list and bag.allowance >= jewel_list[0].weight:
            heapq.heappush(checked_jewel_value_heap, -jewel_list[0].value)
            heapq.heappop(jewel_list)
        if checked_jewel_value_heap:
            result += -heapq.heappop(checked_jewel_value_heap)

    # Title: output
    print(result)
    return str(result)


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
        test_case.assertEqual(thieve_jewels(iter(input_lines)), output_lines[0])


def make_bigger(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Maximum Integer by deleting digits as many "k" number ; https://www.acmicpc.net/problem/2812

    Time Complexity (Worst-case): O(n)
        - O(n) from characters loop  *  O(1) from some operations
            some operations: comparisons, (append, pop) in stack

    Space Complexity: O(1)

    Definition
        - kk: Number of remaining characters to be deleted.

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
        - Following implementation uses "Add picked character to stack" and "If in a condition, pop and kk -= 1".
            Quoting "while kk > 0 and char_stack and char_stack[-1] < n_digit_number[i]:"
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
    n_digit_number: str = input_()
    kk: int = k
    char_stack: list[str] = []

    # Title: solve
    for i in range(n):
        while kk > 0 and char_stack and char_stack[-1] < n_digit_number[i]:
            char_stack.pop()
            kk -= 1
        char_stack.append(n_digit_number[i])
    result: str = "".join(char_stack[: n - k])
    # same as ... = "".join(char_stack[:-kk]) if kk > 0 else "".join(char_stack)

    # Title: output
    print(result)
    return result


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
        test_case.assertEqual(make_bigger(iter(input_lines)), output_lines[0])


def sort_cards(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Minimum sum of the number of comparisons to sort cards ; https://www.acmicpc.net/problem/1715

    Time Complexity (Worst-case): O(n(log n))
        - O(n) from heapify function
        - O(n-1) from loop  *  3*O(log n) from Hip (pop | push)

    Space Complexity: O(1)

    Definition
        - CardStacks(i): the number of cards of i-th card stack.
        - Sum(i): sum of the number of comparisons to cards until i-th merge.

    Recurrence relation
        - Sum(i) ::=
            - if n == 1, 0
            - if n > 1, Sum(i-1) + ( Sum(i-1) + CardStacks(i) )

    Purpose
        - to minimize Sum(x), it requires to select minimum CardStacks(x) in lower merge level.

    Implementation
        - If n == 1, It is not need to compare card stacks. so <result> is 0.
        - If n > 1, Regardless of the number of remained card stacks is odd or even
            , it required to merge with two smallest cards stack in sequence.
            so in this problem heap queue is useful.
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
    result: int = 0

    # Title: solve
    heapq.heapify(card_stack_list)
    # merge
    while len(card_stack_list) > 1:
        a, b = heapq.heappop(card_stack_list), heapq.heappop(card_stack_list)
        merge_value: int = a + b
        result = result + merge_value
        heapq.heappush(card_stack_list, merge_value)

    # Title: output
    print(result)
    return str(result)


def test_sort_cards() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [["4", "30", "40", "50", "100"], ["410"]],
        [["4", "30", "40", "50", "60"], ["360"]],
        [["8", "30", "40", "50", "20", "10", "100", "60", "120"], ["1160"]],
        [["8", "30", "40", "50", "20", "10", "100", "60", "10"], ["860"]],
        [["4", "120", "40", "100", "20"], ["500"]],
    ]:
        test_case.assertEqual(sort_cards(iter(input_lines)), output_lines[0])


def assign_lecture_room(input_lines: Optional[Iterator[str]] = None) -> str:
    """❔ get Minimum the number of lecture rooms that make all lecture available ; https://www.acmicpc.net/problem/11000

    Time Complexity (Worst-case): O(n(log n))
        - O(n(log n)) from Tim sort
        - O(n-1) from loop  *  ( O(1) comparison  +  O(log n) from Hip (pop | push) at least )

    Space Complexity: O(1)

    Implementation
        - It uses sort for input data in order to compare <end time> of lecture in order.
        - It uses heap data structure (Python heapq library uses min heap)
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
    lecture_end_time_heap: list[int] = []

    # Title: solve
    lecture_period_list.sort()
    heapq.heappush(lecture_end_time_heap, lecture_period_list[0].end)
    for i in range(1, n):
        if lecture_end_time_heap[0] <= lecture_period_list[i].start:
            heapq.heapreplace(lecture_end_time_heap, lecture_period_list[i].end)
        else:
            heapq.heappush(lecture_end_time_heap, lecture_period_list[i].end)
    result: str = str(len(lecture_end_time_heap))

    # Title: output
    print(result)
    return result


def test_assign_lecture_room() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [["3", "1 3", "2 4", "3 5"], ["2"]],
        [["8", "1 8", "9 16", "3 7", "8 10", "10 14", "5 6", "6 11", "11 12"], ["3"]],
    ]:
        test_case.assertEqual(assign_lecture_room(iter(input_lines)), output_lines[0])
