# %%
from __future__ import annotations

import sys

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type:ignore
    VisualizationManager,
    VisualizationRoot,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


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
            ["raccecar", "kayak"], ["Even length", "Odd length"]
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


#%%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [DataSize]
    VisualizationManager.call_root_classes(only_class_list=only_class_list)


# for in loop 에서 하나씩 해당 iterator 의 요소를 pop 하는 경우, --
# 0 based 에서 배열의 개수 condition 확인: 3 index - 0 index = 4개
