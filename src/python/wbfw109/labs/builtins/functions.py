# %%
from __future__ import annotations

import functools
import itertools
import timeit

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    VisualizationManager,
    VisualizationRoot,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


class FormatFunc(VisualizationRoot):
    """References
    - https://docs.python.org/3/library/string.html#format-specification-mini-language
    """

    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["function", "eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "-"

    @classmethod
    def test_case(cls):
        format_func: FormatFunc = cls()
        format_func.append_line_into_df_in_wrap(
            [
                "Precision",
                "format(12.345679, '.4f')",
                format(12.345679, ".4f"),
            ]
        )
        format_func.append_line_into_df_in_wrap(
            [
                "Integer",
                "( format(18, 'b'), format(18, 'o'), format(18, 'x'), format(18, 'X') )",
                (format(18, "b"), format(18, "o"), format(18, "x"), format(18, "X")),
            ]
        )
        format_func.visualize()


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


class RangeFunc(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )
        self.df_caption = [
            "💡 Memoization technique: Range object is immutable sequence.",
            "  - Range object is not iterator, but it is iterable which means it is not exhausted unlike iterator.",
        ]

    def __str__(self) -> str:
        return "-"

    @classmethod
    def test_case(cls):
        range_func: RangeFunc = cls()

        range_obj = range(2)
        range_func.append_line_into_df_in_wrap(
            [
                "([i for i in range_obj], [i for i in range_obj])",
                (list(range_obj), list(range_obj)),
                "# range_obj = range(2)",
            ]
        )
        range_func.visualize()


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

    def profile_stable_sorting_advantage(self, /, *, only_conclusion: bool) -> str:
        """
        [Speed]
            - Stable Sorting without already sorted key  ; Win : 🥇
                - It is fastest because it not compare already sorted key.
            - Stable Sorting with all key ; 🥈
                - It is slower because it compares all key instead of minimum key in Stable sorting

        - ➡️ This code demonstrates that if we have a list of tuples where the second element is already sorted
            , we can get a speed improvement by only sorting on the second element (method1).
        """
        if only_conclusion:
            return "\n".join(
                [
                    "⌛ [Profile conclusion: stable sorting advantage] 🔪 Speed",
                    "  in the case in which Stable property of TimSort can be used, sorting without already sorted key is faster than sorting with all key.",
                    "  'Stable' means; it does not change the relative order of elements with equal keys",
                ]
            )

        values: list[int] = [3, 5, 3, 3, 6, 7]
        attributes_1: list[tuple[int, float]] = []
        for num, value in enumerate(values, start=1):
            attributes_1.append((num, value / 10))

        assert sorted(attributes_1, key=lambda x: -x[1]) == sorted(
            attributes_1, key=lambda x: (-x[1], x[0])
        )

        def method1():
            sorted(attributes_1, key=lambda x: -x[1])  # 1.36242s

        def method2():
            sorted(attributes_1, key=lambda x: (-x[1], x[0]))  # 2.23172 s

        results = [timeit.timeit(x) for x in (method1, method2)]
        print(f"Stable Sorting without already sorted key: {results[0]}")
        print(f"Stable Sorting with all key: {results[1]}")
        return ""

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
        arr: list[tuple[int, int]] = [(1, 2), (3, 2), (3, 1), (3, 3)]
        sort_func.append_line_into_df_in_wrap(
            [
                "sorted(arr, reverse=True) == sorted(arr, key=lambda x: (-x[0], -x[1]))",
                sorted(arr, reverse=True) == sorted(arr, key=lambda x: (-x[0], -x[1])),
                "# arr = [(1, 2), (3, 2), (3, 1), (3, 3)]",
            ]
        )
        print(sort_func.profile_stable_sorting_advantage(only_conclusion=True))
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
        zip_func.append_line_into_df_in_wrap(
            [
                "list(zip(range(3), range(3)))",
                list(zip(range(3), range(3))),
                "zip() iterates over several iterables in parallel, producing 🚣 tuples with an item from each one.",
            ]
        )
        zip_func.append_line_into_df_in_wrap(
            ["", "", "# x = range(3); y = range(3, 6)"]
        )
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


# groups ~
class SumProdAndMinMax(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "-"

    def profile_sum_and_prod_functions_that_returns_same_result(
        self, /, *, only_conclusion: bool
    ) -> str:
        """
        [Speed]
            - Sum with sum()     ; 🥇 Win
                - The built-in sum() function has a single function call, minimizing this overhead.
            - Sum with functools.reduce() ; 🥈
                - There is additional overhead in the execution and creation of the lambda function at runtime.
                - The lambda function is called for each pair of elements in the list
                    , which means there are multiple function calls throughout the process. (🚣 Call chains)
        -----
        - ➡️ similarly, this comparison also can apply to `math.prod(num_list)` and `functools.reduce(lambda x, y: x* y, num_list)`.
        """
        if only_conclusion:
            return "\n".join(
                [
                    "⌛ [Profile conclusion: ( sum(), math.prod() ) and functools.reduce() which returns same result] 🔪 Speed",
                    "  sum() and math.prod() are faster than using functools.reduce().",
                ]
            )

        num_list = list(range(1, 101))

        def method1():
            sum(num_list)

        def method2():
            functools.reduce(lambda x, y: x + y, num_list)

        results = [timeit.timeit(x) for x in (method1, method2)]
        print(f"Sum with sum(): {results[0]}")  # 0.65607s
        print(f"Sum with functools.reduce(): {results[1]}")  #  8.98534s
        return ""

    def profile_sum_with_two_arguments(self, /, *, only_conclusion: bool) -> str:
        """
        [Speed]
            - sum directly (a+b)   ; 🥇 Win
            - sum() with two arguments  ; 🥈
                sum() have overhead of function.
        -----
        - ➡️ similarly, this comparison also can apply to `min()`, `max()`.
        """
        if only_conclusion:
            return "\n".join(
                [
                    "⌛ [Profile conclusion: arr[0]+arr[1], sum(arr)]             🔪 Speed",
                    "  `arr[0]+arr[1]` is faster than `sum(arr)`.",
                    "  - 🚣 similarly, this comparison also can apply to `min()`, `max()`.",
                    "  - ❔ pylint said 'consider-using-max-builtin / R1731', but based on the results of this profile, this message is negligible.",
                ]
            )

        arr = [10, -20]

        def method1():
            arr[0] + arr[1]

        def method2():
            sum(arr)

        results = [timeit.timeit(x) for x in (method1, method2)]
        print(f"sum directly (a+b): {results[0]}")  # 0.08576s
        print(f"sum() with two arguments: {results[1]}")  # 0.14830s
        return ""

    @classmethod
    def test_case(cls):
        sum_and_production: SumProdAndMinMax = cls()
        print(
            sum_and_production.profile_sum_and_prod_functions_that_returns_same_result(
                only_conclusion=True
            )
        )
        print(sum_and_production.profile_sum_with_two_arguments(only_conclusion=True))
        # sum_and_production.visualize()


# %%

# %%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [FormatFunc]
    VisualizationManager.call_root_classes(only_class_list=only_class_list)


# partial 에서 positional.. lambda 에서는 순서 중요. 함수에서는 args, keywords 가 나눠져잇지만 여긴 아님.
# "TypeError: got multiple values for argument" after applying functools.partial() 그래서 발생 가능.
#    get_max_relative_parent_len: Callable[
#     [int, list[PathPair]], int
# ] = lambda i, path_pair_list: len(path_pair_list[i].relative_parent.parts)


# input vs sys.stdin.readline speed  https://stackoverflow.com/a/57200421/15252251
