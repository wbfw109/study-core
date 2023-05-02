# %%
from __future__ import annotations

import functools
import itertools
import math
import timeit

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    VisualizationManager,
    VisualizationRoot,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


class MinFunc(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "functions: min()"

    @classmethod
    def test_case(cls):
        min_func: MinFunc = cls()
        min_func.append_line_into_df_in_wrap(
            [
                "min(range(10, 20))",
                min(range(10, 20)),
                "If one positional argument is provided, it should be an iterable. 📝 The smallest item in the iterable is returned.",
            ]
        )
        min_func.append_line_into_df_in_wrap(
            [
                "min([], default=100)",
                min([], default=100),
                "if <iterable> is empty, <default> argument is not specified raise ValueError.",
            ]
        )
        min_func.append_line_into_df_in_wrap(
            [
                "min(*range(10, 20))",
                min(*range(10, 20)),
                "If can be done as arguments instead of iterable.",
            ]
        )
        my_dict = {
            1: [2, 5],
            2: [3, 4, 7],
            3: [1, 4],
            4: [],
            5: [1],
        }
        min_func.append_line_into_df_in_wrap()
        min_func.append_line_into_df_in_wrap(
            [
                "",
                "",
                "<my_dict> = {1: [2, 5], 2: [3, 4, 7], 3: [1, 4], 4: [], 5: [1]}",
            ]
        )
        min_func.append_line_into_df_in_wrap(
            [
                "min((k for k, value in my_dict.items() if value), key=lambda k: len(my_dict[k]))",
                min(
                    (k for k, value in my_dict.items() if value),
                    key=lambda k: len(my_dict[k]),
                ),
                "🛍️ e.g. get 'key' of dict in a condition. not 'value'.",
            ]
        )
        min_func.visualize()


class SortFunc(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "functions: sorted(), MutableSequence.sort()"

    @classmethod
    def test_case(cls):
        sort_func: SortFunc = cls()
        sort_func.append_line_into_df_in_wrap(
            [
                "sorted({'a': 1, 'b': 2})",
                sorted({"a": 1, "b": 2}),
                "sorted(<dict>) returns types: list[tuple[<key>, <value>]]",
            ]
        )
        sort_func.visualize()


class ZipFunc(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "functions: zip(), itertools.zip_longest()"

    @classmethod
    def test_case(cls):
        zip_func: ZipFunc = cls()
        x = range(3)
        y = range(3, 6)
        zip_func.append_line_into_df_in_wrap(["", "", "x = range(3), y = range(3, 6)"])
        zip_func.append_line_into_df_in_wrap(
            [
                "list(zip(*zip(x, y))) == [tuple(x), tuple(y)]",
                list(zip(*zip(x, y))) == [tuple(x), tuple(y)],
                "Transpose matrix. and one more.",
            ]
        )
        zip_func.append_line_into_df_in_wrap(
            [
                "list(zip(list(range(2)), 'abcd'))",
                list(zip(list(range(2)), "abcd")),
                "By default <strict>=False. so zip() stops when the shortest iterable is exhausted.",
            ]
        )
        zip_func.append_line_into_df_in_wrap(
            [
                "list(itertools.zip_longest(list(range(2)), 'abcd', fillvalue=None))",
                list(itertools.zip_longest(list(range(2)), "abcd", fillvalue=None)),
                "Shorter iterables can be padded with a constant value.",
            ]
        )
        zip_func.append_line_into_df_in_wrap(
            [
                "list(zip(*[[1, 2, 3]] * 3, strict=True))",
                list(zip(*[[1, 2, 3]] * 3, strict=True)),
                "This repeats the same iterator 3 times.",
            ]
        )
        zip_func.visualize()


class Iterators(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "-"

    def profile_list_comprehension_for_flattening(self) -> None:
        """

        [Speed]
        -----
        - List comprehension (with tuple)  ; 🪟 Win
        - List comprehension (with range object)
            While this approach can save memory, it introduces overhead due to the creation and management of the range() objects.
        ➡️ using tuples in the list comprehension is more efficient than using range() objects 🚣 for small, fixed-size sequences.
            The reduced overhead from iterating through tuples results in better performance.
            
            🚣 When slicing a sequence, This is the same reason why there is a speed difference between using slice objects and not using slice objects.
        """
        n = 10
        p = (5, 5)

        def method1():
            for nx, ny in (
                (p[0] + dx, p[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            ):
                if 0 <= nx < n and 0 <= ny < n:
                    pass

        def method2():
            for nx, ny in (
                (p[0] + dx, p[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2)
            ):
                if 0 <= nx < n and 0 <= ny < n:
                    pass

        result1 = timeit.timeit(method1, number=100000)
        result2 = timeit.timeit(method2, number=100000)
        print(f"List comprehension (with tuple): {result1}")  # 0.19769s
        print(f"List comprehension (with range object): {result2}")  # 0.27276s

    @classmethod
    def test_case(cls):
        iterators: Iterators = cls()
        iterators.append_line_into_df_in_wrap(
            [
                "str(reversed('abc'))",
                str(reversed("abc")),
                "it converts the object to a string representation of the object itself, not the content of the object.",
            ]
        )
        iterators.append_line_into_df_in_wrap(
            ["''.join(reversed('abc'))", "".join(reversed("abc"))]
        )

        # iterators.profile_list_comprehension_for_flattening()

        iterators.visualize()


class SumAndProduction(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "-"

    def profile_functions_that_returns_same_result(self) -> None:
        """

        [Speed]
        -----
        - `sum(num_list)`    ; 🪟 Win
            The built-in sum() function has a single function call, minimizing this overhead.
        - `functools.reduce(lambda x, y: x + y, num_list)`
            There is additional overhead in the execution and creation of the lambda function at runtime.
            The lambda function is called for each pair of elements in the list
                , which means there are multiple function calls throughout the process. (🚣 Call chains)
        ➡️ This comparison also can apply to `math.prod(num_list)` and `functools.reduce(lambda x, y: x* y, num_list)`.
        """
        num_list = list(range(1, 101))
        result1 = timeit.timeit(lambda: sum(num_list), number=1000)
        result2 = timeit.timeit(
            lambda: functools.reduce(lambda x, y: x + y, num_list), number=1000
        )
        print(f"Sum Method 1 (sum): {result2}")  # 0.00057s
        print(f"Sum Method 2 (functools.reduce): {result1}")  # 0.00825s

    @classmethod
    def test_case(cls):
        sum_and_production: SumAndProduction = cls()

        sum_and_production.profile_functions_that_returns_same_result()

        # sum_and_production.visualize()


# %%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [SumAndProduction]
    VisualizationManager.call_root_classes(only_class_list=only_class_list)


# partial 에서 positional.. lambda 에서는 순서 중요. 함수에서는 args, keywords 가 나눠져잇지만 여긴 아님.
# "TypeError: got multiple values for argument" after applying functools.partial() 그래서 발생 가능.
#    get_max_relative_parent_len: Callable[
#     [int, list[PathPair]], int
# ] = lambda i, path_pair_list: len(path_pair_list[i].relative_parent.parts)
# for k, g in itertools.groupby(data, key=lambda x: x["word"]):
# k is key, g is group

# input vs sys.stdin.readline speed  https://stackoverflow.com/a/57200421/15252251
# It checks if it is TTY every time as input() runs by syscall and it works much more slow than sys.stdin.readline()

# a = [ (1, 3),(2, 3),(3, 3),(3, 2),]
# sorted(a, reverse=True)
# sorted(a, key=lambda x: (-x[0], x[1]))

# 동일한 리스트에 대해 iterator 로 만들고 zip 을 하면 하나의 리스트 안에 있는 원소들을 2개씩 짝지을 수 있다.
