# %%
from __future__ import annotations

import itertools
import sys
import timeit

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type:ignore
    VisualizationManager,
    VisualizationRoot,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


class CPythonAdvantage(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )
        self.df_caption = [
            "💡 Optimization tip; use built-in function or method implemented in C language as possible as",
            "  , rather than expressions and statement evaluated as Python",
        ]

    def __str__(self) -> str:
        return "-"

    @classmethod
    def test_case(cls):
        c_python_advantage: CPythonAdvantage = cls()
        c_python_advantage.append_line_into_df_in_wrap(
            [
                "f'( {type(sum)}, {repr(sum)} )'",
                f"( {type(sum)}, {repr(sum)} )",
                "we can check whether a function is implemented in C language.",
            ]
        )
        c_python_advantage.visualize()


class ArrayInMemory(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["condition", "eval", "print"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "💡 Memoization technique: Python uses Row-major order"

    def profile_memory_access_pattern_of_array(
        self, /, *, only_conclusion: bool
    ) -> str:
        """
        [Speed]
            - Row-major iteration in Row-major order   ; 🥇 Win
                - it can take advantage of spatial locality.
            - Column-major iteration in Row-major order
                - it is hard to take advantage of spatial locality.
        """
        if only_conclusion:
            return "\n".join(
                [
                    "⌛ [Profile conclusion: memory access pattern of array] 🔪 Speed",
                    "  In Row-major order, Row-major iteration is faster than Column-major iteration.",
                ]
            )

        x = itertools.count(1)
        arr = [[next(x) for _ in range(1000)] for _ in range(1000)]

        def method1():
            for i in range(1000):
                for j in range(1000):
                    yy = arr[i][j]

        def method2():
            for i in range(1000):
                for j in range(1000):
                    yy = arr[j][i]

        result1 = timeit.timeit(method1, number=1)
        result2 = timeit.timeit(method2, number=1)
        print(f"List comprehension (with tuple): {result1}")  # 0.03409 second
        print(f"List comprehension (with range object): {result2}")  #  0.11772 second
        return ""

    @classmethod
    def test_case(cls):
        array_in_memory: ArrayInMemory = cls()
        print(
            array_in_memory.profile_memory_access_pattern_of_array(only_conclusion=True)
        )
        # array_in_memory.visualize()


class DataSize(VisualizationRoot):
    """https://rushter.com/blog/python-strings-and-memory"""

    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )
        self.df_caption = [
            "⚙️ Empty string occupy 49 bytes Because it stores supplementary information",
            "    , such as hash, length, length in bytes, encoding type and string flags.",
            "  - ASCII bytes additionally occupy 1 byte",
        ]

    def __str__(self) -> str:
        return "-"

    @classmethod
    def test_case(cls):
        data_size: DataSize = cls()

        for obj in ["", "a", "abc"]:
            data_size.append_line_into_df_in_wrap(
                [f"sys.getsizeof( '{obj}' )", f"{sys.getsizeof(obj)} bytes"]
            )

        data_size.visualize()


class ZeroBasedNumbering(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "methods: zip(), itertools.zip_longest()"

    @classmethod
    def test_case(cls):
        zero_based_index: ZeroBasedNumbering = cls()

        for text, length_text in zip(
            ["raccecar", "kayak"], ["even length", "odd length"]
        ):
            text_len = len(text)
            zero_based_index.append_line_into_df_in_wrap(
                [
                    "[*zip(range(0, text_len // 2), range(text_len - 1, text_len // 2 - 1, -1))]",
                    [
                        *zip(
                            range(0, text_len // 2),
                            range(text_len - 1, text_len // 2 - 1, -1),
                        )
                    ],
                    f"{length_text}. <text>={text}",
                ]
            )
        zero_based_index.append_line_into_df_in_wrap()
        i = 5
        zero_based_index.append_line_into_df_in_wrap(
            ["2 * i + 1", 2 * i + 1, "[Heap] left child of <i>. <i>=5"]
        )
        zero_based_index.append_line_into_df_in_wrap(
            ["2 * i + 1", 2 * i + 2, "[Heap] right child of <i>. <i>=5"]
        )
        zero_based_index.append_line_into_df_in_wrap(
            ["(i - 1) // 2", (i - 1) // 2, "[Heap] parent of <i>. <i>=5"]
        )

        zero_based_index.visualize()


# %%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [Indexing]
    VisualizationManager.call_root_classes(only_class_list=only_class_list)
