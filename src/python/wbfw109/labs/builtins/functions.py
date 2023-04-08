# %%
from __future__ import annotations

import itertools

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    VisualizationManager,
    VisualizationRoot,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


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

#%%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = []
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
