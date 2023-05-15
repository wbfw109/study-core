"""💡 some Statement can be considered as also Expression in Python. e.g. function
    - a statement may have internal components but it not means the statement is also expression.
    🛍️ e.g. statement ∩ expression: function.
    🛍️ e.g. only statement: if-else, loops
"""

# %%
from __future__ import annotations

import itertools
import math
import timeit

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    VisualizationManager,
    VisualizationRoot,
    display_data_frame_with_my_settings,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


class UnpackingVariables(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self, columns=["eval", "print"], has_df_lock=False, should_highlight=True
        )

    def __str__(self) -> str:
        return "-"

    def profile_consecutive_vars_in_loop(self, /, *, only_conclusion: bool) -> str:
        """
        [Speed]
            - (i, i-1) in loop with Arithmetic operation  ; 🥇 Win
            - (i, i-1) in loop with zip()
                - it is slower because overhead of calling zip() and tuple unpacking's cost are exist
        """
        if only_conclusion:
            return "\n".join(
                [
                    "⌛ [Profile conclusion: consecutive vars in loop] 🔪 Speed",
                    "  (i, i-1) in loop with Arithmetic operation is faster than with zip().",
                ]
            )

        ran1 = range(1, 1001)
        ran2 = range(1000)

        def method1():
            for i in ran1:
                j = i - 1

        def method2():
            for i, j in zip(ran1, ran2):
                pass

        results = [timeit.timeit(x, number=100000) for x in (method1, method2)]
        print(f"(i, i-1) in loop with Arithmetic operation: {results[0]}")  # 3.22418s
        print(f"(i, i-1) in loop with zip(): {results[1]}")  # 3.65471s
        return ""

    def profile_assignment(self, /, *, only_conclusion: bool) -> str:
        """
        [Speed]
            - assignment with condition statement without unpacking  ; 🥇 Win
            - assignment using ternary operation with unpacking
                - it is is slower because of the extra overhead introduced by the creation and unpacking tuple.
        """
        if only_conclusion:
            return "\n".join(
                [
                    "⌛ [Profile conclusion: assignment] 🔪 Speed",
                    "  assignment with condition statement without unpacking is faster than using ternary operation with unpacking.",
                ]
            )

        x, y = 15, 30

        def method1():
            if x < y:
                shorter_side, longer_side = x, y
            else:
                shorter_side, longer_side = y, x

        def method2():
            shorter_side, longer_side = (x, y) if x < y else (y, x)

        results = [timeit.timeit(x) for x in (method1, method2)]
        print(
            f"assignment with condition statement without unpacking: {results[0]}"
        )  # 0.06060s
        print(
            f"assignment using ternary operation with unpacking: {results[1]}"
        )  # 0.09834s
        return ""

    @classmethod
    def test_case(cls):
        unpacking_vars: UnpackingVariables = cls()
        a, b, *c = [1, 2, 3, 4, 5]
        unpacking_vars.append_line_into_df_in_wrap(
            [
                "a, b, *c = [1, 2, 3, 4, 5]",
                f"a = {a}; b = {b}; c= {c}",
            ]
        )
        [[x1, y1], [x2, y2]] = [[1, 3], [2, 4]]
        unpacking_vars.append_line_into_df_in_wrap(
            [
                "[[x1, y1], [x2, y2]]=[[1, 3], [2, 4]]",
                f"x1, y1, x2, y2 = {x1}, {y1}, {x2}, {y2}",
            ]
        )
        unpacking_vars.append_line_into_df_in_wrap()

        unpacking_vars.append_line_into_df_in_wrap(
            [
                "tuple(*[[1, 2, 3]])",
                tuple(*[[1, 2, 3]]),
            ]
        )
        unpacking_vars.append_line_into_df_in_wrap(
            [
                "a = [[1, 2], 3][0]",
                f"a = {[[1, 2], 3][0]}",
            ]
        )

        unpacking_vars.append_line_into_df_in_wrap()
        unpacking_vars.append_line_into_df_in_wrap(
            [
                "[(x, y) for x, y in [(1, 2), (3, 4)]]",
                [(x, y) for x, y in [(1, 2), (3, 4)]],  # pylint: disable=R1721
            ]
        )
        unpacking_vars.append_line_into_df_in_wrap(
            [
                "[(i, x, y) for i, (x, y) in enumerate(zip([10, 15], 'AB'))]",
                [(i, x, y) for i, (x, y) in enumerate(zip([10, 15], "AB"))],
            ]
        )

        print(unpacking_vars.profile_consecutive_vars_in_loop(only_conclusion=True))
        print(unpacking_vars.profile_assignment(only_conclusion=True))
        unpacking_vars.visualize()


class ForStatement(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self, columns=["eval", "print"], has_df_lock=False, should_highlight=True
        )

    def __str__(self) -> str:
        return "parameter <__start> is inclusive, parameter <__stop> is exclusive."

    @classmethod
    def test_case(cls):
        for_statement: ForStatement = cls()
        start_i = -1
        end_i = 4
        for_statement.append_line_into_df_in_wrap(["", "start_i = -1, end_i = 4"])
        for_statement.append_line_into_df_in_wrap(
            [
                "list(range(start_i, end_i, 1))",
                list(range(start_i, end_i, 1)),
            ]
        )
        for_statement.append_line_into_df_in_wrap(
            [
                "list(range(end_i, start_i, -1))",
                list(range(end_i, start_i, -1)),
            ]
        )

        for_statement.visualize()


class Operators(VisualizationRoot):
    """
    Test environment:
        - 🆚 Operation (Left Shift, Exponentiation):
            Average of time occurred from each operations in inner loop range(1000000) of outer loop range(100).
            - Exponentiation (**):    0.6202
            - Left Shift (**):        0.6166
        - 🆚 Operation (list.extend, list + list):
            Another list is range(1000000)
            - [0] and list.extend(another list):    0.1973
            - [0] + another list:                   0.3829

    https://docs.python.org/3/reference/expressions.html#comparisons
    https://stackoverflow.com/a/20970087/15252251
    https://stackoverflow.com/questions/65445377/python-list-comprehension-performance
    """

    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )
        self.df_caption.extend(
            [
                "⚙️ Comparisons can be chained arbitrarily.",
                "  'a == b == c' equals 'a == b and b == c'.",
                "",
                "⚙️ Left Shift (<<) operation is slightly faster than Exponentiation (**) operation.",
                "  and Exponentiation (**) operation faster than math.pow().",
                "",
            ]
        )

    def __str__(self) -> str:
        return "-"

    @classmethod
    def test_case(cls):
        operators: Operators = cls()
        c = b = a = 1
        operators.append_line_into_df_in_wrap(
            ["a == b == c", a == b == c, "# c = b = a = 1"]
        )
        operators.append_line_into_df_in_wrap()
        operators.append_line_into_df_in_wrap(
            [
                "( 1 << 4, 2**4, math.pow(2, 4) )",
                (1 << 4, 2**4, math.pow(2, 4)),
            ]
        )
        # TODO: move this other file with float("inf")
        operators.append_line_into_df_in_wrap(
            [
                "( math.sqrt(4), 4**0.5, float(4**0.5).is_integer() )",
                (math.sqrt(4), 4**0.5, float(4**0.5).is_integer()),
            ]
        )
        operators.append_line_into_df_in_wrap()
        operators.append_line_into_df_in_wrap(
            [
                "[] or [100,200]",
                [] or [100, 200],
                "Python `and`, `or` returns the value itself in according to Truthy value.",
            ]
        )
        operators.append_line_into_df_in_wrap(
            [
                "min([] or [100]) == min([], default=100)",
                min([] or [100]) == min([], default=100),
                "so, this expression is valid.",
            ]
        )
        operators.append_line_into_df_in_wrap()

        operators.visualize()


class LambdaExpression(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )
        self.df_caption = [
            "⚙️ Note that lambda expressions can only contain expressions; cannot contain statements or annotations.",
            "",
            "⚠️ if lambda expression is defined in a loop like normal functions, Note that the function is created and run on every iteration",
            "  , so that it may cause the many overhead.",
        ]

    def __str__(self) -> str:
        return "terms (lambda expression == lambda form)"

    @classmethod
    def test_case(cls):
        lambda_expression: LambdaExpression = cls()
        lambda_expression.append_line_into_df_in_wrap(
            [
                "[(lambda: i)() for i in range(10)]",
                [(lambda: i)() for i in range(10)],
            ]
        )
        lambda_expression.append_line_into_df_in_wrap(
            [
                "[func() for func in [lambda: i for i in range(10)]]",
                [func() for func in [lambda: i for i in range(10)]],
            ]
        )
        lambda_expression.append_line_into_df_in_wrap(
            [
                "[func() for func in [lambda x=i: x for i in range(10)]]",
                [func() for func in [lambda x=i: x for i in range(10)]],  # type: ignore
                "if you require to use lambda in for-loop with different argument, save lambda into collection data type and use like this.",
            ]
        )

        lambda_expression.visualize()


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

    @staticmethod
    def test_lazy_evaluation():
        visualization = VisualizationRoot(
            columns=["name", "eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
            header_string="Lazy evaluation property of Iterators (including Generator)",
        )

        visualization.append_line_into_df_in_wrap(
            [
                "Early stopping",
                "list(itertools.takewhile(lambda x: x > 0, [1, 3, -1, 3, 5])))",
                list(itertools.takewhile(lambda x: x > 0, [1, 3, -1, 3, 5])),
            ]
        )
        visualization.append_line_into_df_in_wrap(
            [
                "get First matched element",
                "next((x for x in [1, 2, 5, 8, 9] if x == 5))",
                next((x for x in [1, 2, 5, 8, 9] if x == 5)),
            ]
        )
        visualization.append_line_into_df_in_wrap(
            [
                "get First matched element (with default keyword)",
                "next((x for x in [1, 2, 5, 8, 9] if x == 100), -10)",
                next((x for x in [1, 2, 5, 8, 9] if x == 100), -10),
                "🚣 If default is given and the iterator is exhausted, it is returned instead of raising StopIteration.",
            ]
        )

        display_data_frame_with_my_settings(
            visualization.df, caption=visualization.df_caption
        )

    def profile_loop_for_flattening(self, /, *, only_conclusion: bool) -> str:
        """
        [Speed]
            - Loop for flattening with itertools.product  ; Win : 🥇
                - It is fastest method because `itertools.product` is implemented in C, and it can create and iterate over the Cartesian product more efficiently than equivalent Python code.
                - This speed advantage applies even if the itertools library is not already imported or the range objects are already created.
            - Loop for flattening with tuple ; 🥈
                - It is slower than using `itertools.product` because it relies on Python's built-in generator expressions and for loops
                , which are not as optimized as the C implementation of `itertools.product`.
            - Loop for flattening with range object ; 🥉
                - It is slowest because it introduces overhead due to the creation and management of the range() objects.
                    - Even if the range objects are created beforehand, this method remains slower due to the overhead of managing the range() objects.
                - This is similar case to the overhead introduced by using slice() or slicing syntax [::] for slicing sequences.

        🔍 Python, as a high-level language, doesn't natively support SIMD operations.
        Instead, Python libraries such as NumPy, which are written in lower-level languages like C, can be designed to take advantage of SIMD instructions for certain operations.
        """
        if only_conclusion:
            return "\n".join(
                [
                    "⌛ [Profile conclusion: loop for flattening] 🔪 Speed",
                    "  if possible, to use itertools.product() with already created range object is fastest.",
                    "  , than with tuple or range object.",
                ]
            )

        n = 10
        p = (5, 5)
        range_obj = range(-1, 2)

        def method1():
            for dx, dy in itertools.product((-1, 0, 1), repeat=2):
                nx, ny = p[0] + dx, p[1] + dy
                if 0 <= nx < n and 0 <= ny < n:
                    pass

        def method2():
            for nx, ny in (
                (p[0] + dx, p[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            ):
                if 0 <= nx < n and 0 <= ny < n:
                    pass

        def method3():
            for nx, ny in (
                (p[0] + dx, p[1] + dy) for dx in range_obj for dy in range_obj
            ):
                if 0 <= nx < n and 0 <= ny < n:
                    pass

        results = [timeit.timeit(x, number=100000) for x in (method1, method2, method3)]
        print(f"Loop for flattening with itertools.product: {results[0]}")  # 0.17098s
        print(f"Loop for flattening with tuple: {results[1]}")  # 0.24233s
        print(f"Loop for flattening with range object: {results[2]}")  # 0.25859s
        return ""

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

        iterators.append_line_into_df_in_wrap(
            [
                "[(k, tuple(g)) for k, g in itertools.groupby('AAACCA')]",
                [(k, tuple(g)) for k, g in itertools.groupby("AAACCA")],
                "Return type is pairs of key, group (iterator); it only groups by consecutive same elements.",
            ]
        )

        print(iterators.profile_loop_for_flattening(only_conclusion=True))
        iterators.visualize()
        Iterators.test_lazy_evaluation()


# %%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        print(__doc__)
        only_class_list = []
    else:
        only_class_list = [Iterators]
    VisualizationManager.call_root_classes(only_class_list=only_class_list)
