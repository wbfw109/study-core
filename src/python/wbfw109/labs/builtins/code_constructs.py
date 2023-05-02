"""💡 some Statement can be considered as also Expression in Python. e.g. function
    - a statement may have internal components but it not means the statement is also expression.
    🛍️ e.g. statement ∩ expression: function.
    🛍️ e.g. only statement: if-else, loops
"""

# %%
from __future__ import annotations

import functools
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
                "⚙️ Sequence operation (s*n or n*s) is faster than List comprehension.",
                "  ⚠️ Note that the List multiplication uses shallow copy.",
                "  if then, [[dict()] * m for _ in range(n)] will share one dictionary.",
            ]
        )

    def __str__(self) -> str:
        return "TODO: ...."

    @classmethod
    def test_case(cls):
        operators: Operators = cls()
        c = b = a = 1
        operators.append_line_into_df_in_wrap(["", "c = b = a = 1"])
        operators.append_line_into_df_in_wrap(
            [
                a == b == c,
                "a == b == c",
            ]
        )
        operators.append_line_into_df_in_wrap()
        operators.append_line_into_df_in_wrap(
            [
                (1 << 4, 2**4, math.pow(2, 4)),
                "( 1 << 4, 2**4, math.pow(2, 4) )",
            ]
        )
        operators.append_line_into_df_in_wrap()
        operators.append_line_into_df_in_wrap(
            [
                ([0] * 3, [0 for _ in range(3)]),
                "( [0]*3, [0 for _ in range(3)] )",
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
        lambda_expression.df_caption = [
            "⚙️ Note that functions created with lambda expressions cannot contain statements or annotations."
        ]

        lambda_expression.visualize()


class UnpackingVariables(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self, columns=["eval", "print"], has_df_lock=False, should_highlight=True
        )

    def __str__(self) -> str:
        return "-"

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
        unpacking_vars.visualize()


# %%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        print(__doc__)
        only_class_list = []
    else:
        only_class_list = [UnpackingVariables]
    VisualizationManager.call_root_classes(only_class_list=only_class_list)

# list comprehension


# while else: https://docs.python.org/3/reference/compound_stmts.html#the-while-statement
