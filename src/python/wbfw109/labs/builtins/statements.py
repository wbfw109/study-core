# %%
from __future__ import annotations

import math

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
                "[i for i in range(start_i, end_i, 1)]",
                [i for i in range(start_i, end_i, 1)],
            ]
        )
        for_statement.append_line_into_df_in_wrap(
            [
                "[i for i in range(end_i, start_i, -1)]",
                [i for i in range(end_i, start_i, -1)],
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
        operators.visualize()


#%%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = []
    VisualizationManager.call_root_classes(only_class_list=only_class_list)

# while else: https://docs.python.org/3/reference/compound_stmts.html#the-while-statement
