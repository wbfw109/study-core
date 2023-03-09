# %%
from __future__ import annotations

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    VisualizationManager,
    VisualizationRoot,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


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


#%%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = []
    VisualizationManager.call_root_classes(only_class_list=only_class_list)

# list comprehension
