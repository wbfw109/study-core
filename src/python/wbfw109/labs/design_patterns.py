# %%
from __future__ import annotations

import sys

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    VisualizationManager,
    VisualizationRoot,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


class Interning(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )
        self.df_caption = [
            "⚙️ Interning is re-using objects of equal value on-demand instead of creating new objects.",
            "  - Interned objects must be immutable.",
            "  - ⭕ Interning saves memory and can thus improve performance and memory footprint of a program.",
            "    The downside is time required to search for existing values of objects which are to be interned.",
            "",
            "⚙️ in Python, refer to https://docs.python.org/3/library/sys.html#sys.intern",
            "  - It seems that immutable objects in function scope automatically interned.",
            "    in global scope, only string not including whitespace or other primitive objects is automatically interned.",
        ]

    def __str__(self) -> str:
        return "Creational pattern 🔪 Interning"

    @classmethod
    def test_case(cls):
        interning: Interning = cls()

        a = (1, 2)
        b = (1, 2)
        interning.append_line_into_df_in_wrap(
            [f"(a == b, a is b)", (a == b, a is b), "a = (1, 2); b = (1, 2)"]
        )
        a = "abcd"
        b = "abcd"
        interning.append_line_into_df_in_wrap(
            [f"(a == b, a is b)", (a == b, a is b), "a = 'abcd'; b = 'abcd'"]
        )

        a = "ab cd"
        b = "ab cd"
        interning.append_line_into_df_in_wrap(
            [f"(a == b, a is b)", (a == b, a is b), "a = 'ab cd'; b = 'ab cd'"]
        )
        a = sys.intern("ab cd")
        b = sys.intern("ab cd")
        interning.append_line_into_df_in_wrap(
            [
                f"(a == b, a is b)",
                (a == b, a is b),
                "a = sys.intern('ab cd'); b = sys.intern('ab cd')",
            ]
        )

        interning.visualize()


# %%

# class DisposePattern:
#     """

#     with statement
#     https://peps.python.org/pep-0343/
#     """

#     pass
# Singleton, bridge, strategy, builder, factory method, facade pattern

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [Interning]
    VisualizationManager.call_root_classes(only_class_list=only_class_list)
