"""Package for easy debugging and visualize in iPython"""
import contextlib
import copy
import functools
import importlib
import inspect
import re
import sys
import threading
import time
import unittest
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Generator, Generic, Iterable, Optional, TypeVar

import pandas as pd
import svgling  # type: ignore
from IPython import display  # type: ignore
from IPython.core import getipython  # type: ignore
from IPython.core.interactiveshell import InteractiveShell  # type: ignore
from pandas.io.formats.style import Styler
from wbfw109.libs.objects.object import (  # type: ignore
    get_child_classes,
    get_outer_class,
)
from wbfw109.libs.parsing import (  # type: ignore
    convert_implicit_syntax_node_dict_to_tree,
    convert_syntax_tree_to_svgling_style,
)
from wbfw109.libs.path import (  # type: ignore
    PathPair,
    get_module_name_list,
    get_valid_path_pair_from_sys_path,
)
from wbfw109.libs.typing import DST, JudgeResult, T  # type: ignore


def visualize_implicit_tree(
    implicit_syntax_node_dict: dict[str, list[str]], /, *, root_node: str
) -> None:
    """Simple function for visualize normal graph."""
    display.display(  # type:ignore
        svgling.draw_tree(  # type:ignore
            *convert_syntax_tree_to_svgling_style(
                convert_implicit_syntax_node_dict_to_tree(
                    implicit_syntax_node_dict, root_node=root_node
                )
            )
        )
    )


def get_data_frame_for_test(
    iterables_for_product: list[list[str]],
    multi_index_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    if not multi_index_names:
        multi_index_names = [
            *["title" for _ in range(len(iterables_for_product) - 1)],
            "contents",
        ]

    return pd.DataFrame(
        data={},
        columns=pd.MultiIndex.from_product(  # type: ignore
            iterables_for_product,
            names=multi_index_names,
        ),
    )


def get_default_df_style_with_my_settings(df: pd.DataFrame) -> Styler:
    return df.style.hide().format(precision=20, escape="html")  # type: ignore


def display_header_with_my_settings(columns: list[str]) -> None:
    df = pd.DataFrame(data={}, columns=columns)
    display.display(  # type: ignore
        get_default_df_style_with_my_settings(df).apply_index(  # type: ignore
            lambda elements: [  # type: ignore
                "color: deeppink;" for _ in elements  # type: ignore
            ],
            axis=1,
        )
    )


def display_data_frame_with_my_settings(
    df: pd.DataFrame, caption: Optional[list[str]] = None
) -> None:
    def highlight_header_titles(elements: pd.Series):  # type: ignore
        return [
            "color: darkorange;" if value != "title" else ""
            for value in elements  # type: ignore
        ]

    def highlight_header_contents(elements: pd.Series):  # type: ignore
        return [
            "; ".join(["text-align: center", ""]) if value != "title" else ""
            for value in elements  # type: ignore
        ]

    if not caption:
        caption = []

    default_df_my_style: Styler = (
        get_default_df_style_with_my_settings(df)
        .set_table_styles(
            [
                {"selector": "*", "props": [("text-align", "left")]},
                {
                    "selector": "td, th",
                    "props": [("border", "1px solid grey !important")],
                },
            ]
        )
        .set_caption(
            "<br>".join(["&nbsp;" * (len(x) - len(x.lstrip())) + x for x in caption])
        )
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("white-space", "nowrap"),
                        ("overflow", "hidden"),
                        ("caption-side", "bottom"),
                        ("font-size", "1.00em"),
                    ],
                }
            ],
            overwrite=False,
        )
    )
    if df.columns.nlevels >= 2:
        default_df_my_style = default_df_my_style.apply_index(  # type: ignore
            highlight_header_titles,
            axis="columns",
            level=list(range(df.columns.nlevels - 1)),
        ).apply_index(
            highlight_header_contents,
            axis="columns",
            level=[df.columns.nlevels - 1],
        )
    display.display(default_df_my_style)  # type: ignore


def append_line_into_df(
    df: pd.DataFrame,
    /,
    line: Optional[list[Any]] = None,
    lock: Optional[threading.Lock] = None,
) -> None:
    """
    Args:
        df (pd.DataFrame): .
        line (list[Any]): if <line> is Falsy value, automatically create empty line.
        lock (Optional[threading.Lock], optional): lock of <df>.
    """
    df_columns_len: int = len(df.columns)
    if not line:
        line = ["" for _ in range(df_columns_len)]
    elif (empty_value_len := df_columns_len - len(line)) != 0:
        line.extend(["" for _ in range(empty_value_len)])

    if lock:
        with lock:
            df.loc[len(df.index)] = line  # type: ignore
    else:
        df.loc[len(df.index)] = line  # type: ignore


def append_line_into_df_in_wrap(
    df: pd.DataFrame, /, lock: Optional[threading.Lock] = None
) -> functools.partial[None]:
    """On common use, (df, lock) is always in fixed so wrapping that.

    💡 You would only call with "line: list[Any]" argument as positional in returned partial object.
    """
    return functools.partial(append_line_into_df, df, lock=lock)


class VisualizationRoot:
    """It creates <self.df> attribute and <self.append_line_into_df_in_wrap> method that appends line on <self.df>.
    - It also provide class method (<main>, <test_case>), method <visualize> in order to classes used with no inheritance
    - You can set optional <self.df_caption> for detail description.
        - Recommend to use <self.df_caption> instead of docstring when use this class.
        - Recommend with emojis like (⚙️) or header like "-" with whitespace character for better readability.
    """

    def __init__(
        self,
        /,
        columns: list[str],
        has_df_lock: bool,
        should_highlight: bool,
        header_string: Optional[str] = None,
    ) -> None:
        """
        Args:
            columns (list[str]): columns of <self.df>
            has_df_lock (bool): do you required to lock when write <self.df>?
            should_highlight (bool): If <should_highlight> is True, it will make Multi-index pd.DataFrame.
            header_string (Optional[str], optional): when <should_highlight> is true and when it is not None
                , header string to set the value instead of "str(self)".
        """
        self.df_caption: list[str] = []

        if should_highlight:
            if header_string is None:
                header_string = str(self)
            self.df = get_data_frame_for_test(
                iterables_for_product=[  # type: ignore
                    *[
                        ["🔪 " + "🔪 ".join(x.split(":"))]
                        for x in header_string.split("\n")
                    ],
                    columns,
                ]
            )
        else:
            self.df = pd.DataFrame(data={}, columns=columns, index=None)

        if has_df_lock:
            self.df_lock = threading.Lock()
            self.append_line_into_df_in_wrap = append_line_into_df_in_wrap(
                self.df, lock=self.df_lock
            )
        else:
            self.append_line_into_df_in_wrap = append_line_into_df_in_wrap(self.df)

    @classmethod
    def test_case(cls):
        """📝[Optional] override. It may or may not be used. Example:
        ```
        ...
        obj.visualize()
        ```
        """

    def visualize(self) -> None:
        """📝[Optional] override. default shows <self.df> with caption"""
        display_data_frame_with_my_settings(self.df, caption=self.df_caption)

    @classmethod
    def main(cls):
        display_header_with_my_settings(columns=[cls.__name__])
        cls.test_case()


VisualizationRootT = TypeVar("VisualizationRootT", bound=VisualizationRoot)


class AlgorithmVisualization(VisualizationRoot):
    """- <self.big_o_visualization> columns: ["Best", "Average", "Worst", "Memory"]"""

    def __init__(self, /, columns: list[str]) -> None:
        VisualizationRoot.__init__(
            self, columns, has_df_lock=True, should_highlight=True
        )
        self.big_o_visualization = VisualizationRoot(
            columns=["Best", "Average", "Worst", "Memory"],
            has_df_lock=False,
            should_highlight=False,
        )


class ChildAlgorithmVisualization(AlgorithmVisualization, ABC, Generic[DST]):
    """📝~ Note that some function must be override in subclasses.
    - method (<solve>), class method <test_case>, Optional method (<verify>, <judge_acceptance>, <visualize>)

    - <self.big_o_visualization> columns: ["Best", "Average", "Worst", "Memory"]"""

    def __init__(self, /, columns: list[str], dst: Optional[DST] = None) -> None:
        AlgorithmVisualization.__init__(self, columns)
        if dst:
            self.dst: DST = dst

    def measure(self) -> JudgeResult:
        """
        Returns:
            JudgeResult: ( <elapsed_time>, judge_result )
        """
        start_time = time.time()
        self.solve()
        elapsed_time = time.time() - start_time
        return JudgeResult(
            elapsed_time=elapsed_time, judge_result=self.judge_acceptance()
        )

    @abstractmethod
    def solve(self) -> None:
        """📝 Require override."""

    def verify(self) -> bool | Any:
        """📝 [Optional] override. recommend to create and call to staticmethod of DST because verification processes can overlap.
        - It is designed to return "True" by default in case verification is not required. So it is not abstract method.

        Returns
            bool | Any: Truthy or other values.
        """
        return True

    def judge_acceptance(self) -> bool | Any:
        """📝[Optional] override. default is AssertTrue"""
        verified_result = self.verify()
        unittest.TestCase().assertTrue(verified_result)
        return verified_result

    def visualize(self) -> None:
        """📝 [Optional] override. default shows <self.df>, <self.big_o_visualization.df> with caption"""
        display_data_frame_with_my_settings(self.df, caption=self.df_caption)
        if not self.big_o_visualization.df.empty:
            display_data_frame_with_my_settings(
                self.big_o_visualization.df, caption=self.big_o_visualization.df_caption
            )

    @classmethod
    @abstractmethod
    def test_case(cls, dst: Optional[DST] = None) -> None:  # type: ignore
        """📝 Require override. It must be called specifically. Example:
        ```
        algorithm = ExchangeSorts.BubbleSort(dst=dst)
        algorithm.append_line_into_df_in_wrap(algorithm.measure())
        algorithm.visualize()
        ```
        """

    @classmethod
    def main(
        cls,
        dst: Optional[DST] = None,
    ) -> None:
        display_header_with_my_settings(columns=[cls.__name__])
        cls.test_case(dst=dst)


ChildAlgorithmVisualizationT_co = TypeVar(
    "ChildAlgorithmVisualizationT_co",
    bound=ChildAlgorithmVisualization[Any],
    covariant=True,
)


class MixInParentAlgorithmVisualization:
    """It contains class method <main> that calls sub-abstract classes."""

    @classmethod
    def call_child_classes(
        cls,
        dst: Optional[DST] = None,
        only_class_list: Optional[list[type[ChildAlgorithmVisualizationT_co]]] = None,
    ) -> None:
        available_classes_list: list[
            type[ChildAlgorithmVisualizationT_co]
        ] = get_child_classes(
            obj_to_be_inspected=cls,
            parent_class=ChildAlgorithmVisualization,
        )
        filtered_list: list[type[ChildAlgorithmVisualizationT_co]] = []

        if only_class_list:
            # ??? Todo: PyLance error report
            filtered_list.extend(
                [  # type:ignore
                    child_class
                    for child_class in only_class_list
                    if child_class in available_classes_list
                ]
            )
        else:
            filtered_list = available_classes_list

        if filtered_list:
            display_header_with_my_settings(columns=[f"🅰️ {cls.__name__}"])
            for child_class in filtered_list:
                child_class.main(dst=dst)


MixInParentAlgorithmVisualizationT_co = TypeVar(
    "MixInParentAlgorithmVisualizationT_co",
    bound=MixInParentAlgorithmVisualization,
    covariant=True,
)


class VisualizationManager:
    """It manipulates classes that extends <VisualizationRoot> or <MixInParentAlgorithmVisualization>, <ChildAlgorithmVisualization>.
    - Use class method <call_parent_algorithm_classes>, <call_root_classes>, or <call_modules>.

    Tip:
        - If you want to draw graph as .svg format, refer to function <convert_syntax_tree_to_svgling_style> in "src/python/wbfw109/libs/parsing.py".
            Or you could function <visualize_implicit_tree> for simple graph.
        - If you want to draw plot, you could use a library such <matplotlib>.

    ---
    Implementation:
        - Parameter <obj_to_be_inspected> at class method <call ~> is not required. (argument as sys.modules["__main__"] or sys.modules[__name__] in caller file.)
            modules path are automatically resolved by tracing call stack in this class.
        - classVar <_previous_loaded_classes_dict> is needed to avoid re-calling classes which are previously run in Interactive Window
            , because modules from running a file in IPython are added in sys.modules["__main__"] for lifetime of Interactive Window.
            It is same when not in Interactive Window (normal run and call chain).
        - classVar (<central_control_state>), method <_central_control> is required in order to control centrally.
        - when pass <dst>, you don't have to copy data. in method, automatically copy that.
    """

    central_control_state: ClassVar[bool] = False
    _previous_loaded_classes_dict: ClassVar[
        OrderedDict[str, list[type[object]]]
    ] = OrderedDict()

    @classmethod
    @contextlib.contextmanager
    def _central_control(cls) -> Generator[bool, None, None]:
        interactive_shell: InteractiveShell = getipython.get_ipython()  # type: ignore
        if not interactive_shell:
            yield False
        else:
            # setup
            cls.central_control_state = True
            if "IPython.extensions.autoreload" not in interactive_shell.extension_manager.loaded:  # type: ignore
                interactive_shell.run_line_magic("load_ext", "autoreload")  # type: ignore
            interactive_shell.run_line_magic("autoreload", "")  # type: ignore

            yield True
            # exit context
            interactive_shell.run_line_magic("autoreload", "0")  # type: ignore
            cls.central_control_state = False

    @classmethod
    def _get_module_to_be_inspected(
        cls, *, call_stack_level: int
    ) -> Optional[ModuleType]:
        current_call_stack_level = call_stack_level + 1
        # the value will be not None if cls.center_control_state or directly import libraries with not using class method <call_modules>
        obj_to_be_inspected: Optional[ModuleType] = inspect.getmodule(
            inspect.stack()[current_call_stack_level].frame
        )
        # else if caller is module whose classes is inherited from <MixInParentAlgorithmVisualization>, <VisualizationRoot>.
        if not obj_to_be_inspected:
            obj_to_be_inspected = sys.modules["__main__"]
        return obj_to_be_inspected

    @classmethod
    def _get_valid_classes(
        cls, loaded_classes: Iterable[type[T]], *, call_stack_level: int
    ) -> list[type[T]]:
        current_call_stack_level: int = call_stack_level + 1
        # It must be loaded in other file because it use inspect.stack()[1] (previous caller file)
        available_classes: list[type[T]] = []
        ran_code_name: str = inspect.stack()[current_call_stack_level].filename
        ran_code_hash: str = ""
        if match_obj := re.fullmatch(
            r"<ipython-input-\d+-(?P<cell_hash>.*)>", ran_code_name
        ):
            # when run as cell at Python Interactive shell
            ran_code_hash = match_obj.group("cell_hash")
        elif ran_code_name.startswith("/"):
            # when run normally
            ran_code_hash = str(Path(ran_code_name).relative_to(Path.cwd()))

        loaded_classes_set = set(loaded_classes)
        # add new added files to modules.
        available_classes.extend(
            loaded_classes_set.difference(*cls._previous_loaded_classes_dict.values())
        )
        # even if same <ran_code_hash>, same module could be added to modules.
        # default value is required in order to avoid KeyError when new <ran_code_hash>
        available_classes.extend(
            loaded_classes_set.intersection(
                cls._previous_loaded_classes_dict.get(ran_code_hash, {})
            )
        )

        # This ensures that valid classes is returned when a module that uses class method <call ~ > is called directly from another file
        # , with not using class method <call_modules>.
        if loaded_classes_set and not available_classes:
            available_classes.extend(loaded_classes)

        cls._previous_loaded_classes_dict.update({ran_code_hash: available_classes})  # type: ignore
        return available_classes

    @classmethod
    def call_root_classes(
        cls,
        only_class_list: Optional[Iterable[type[VisualizationRootT]]] = None,
    ) -> None:
        """It calls classes that extends <VisualizationRoot>.

        Example in caller file:
        ```
        if __name__ == "__main__" or VisualizationManager.center_control_state:
            VisualizationManager.call_root_classes()
            VisualizationManager.call_root_classes(only_class_list=[ZipFunc])
        ```
        """
        call_stack_level: int = 1
        if not (
            obj_to_be_inspected := cls._get_module_to_be_inspected(
                call_stack_level=call_stack_level
            )
        ):
            return

        loaded_classes: Iterable[type[VisualizationRootT]] = get_child_classes(
            obj_to_be_inspected=obj_to_be_inspected, parent_class=VisualizationRoot
        )
        available_classes: Iterable[
            type[VisualizationRootT]
        ] = VisualizationManager._get_valid_classes(
            loaded_classes, call_stack_level=call_stack_level
        )
        filtered_list: list[type[VisualizationRootT]] = []
        if only_class_list:
            filtered_list.extend(
                [cls_ for cls_ in only_class_list if cls_ in available_classes]
            )
        else:
            filtered_list = available_classes

        if filtered_list:
            for child_class in filtered_list:
                child_class.main()

    @classmethod
    def call_parent_algorithm_classes(
        cls,
        dst: Optional[DST] = None,
        only_class_list: Optional[
            Iterable[
                type[MixInParentAlgorithmVisualizationT_co]
                | type[ChildAlgorithmVisualizationT_co]
            ]
        ] = None,
    ) -> None:
        """It calls classes that extends <MixInParentAlgorithmVisualization>.

        Example in sorting.py:
        ```
        if __name__ == "__main__" or VisualizationManager.center_control_state:
            VisualizationManager.call_parent_algorithm_classes(
                dst=SortingDST.get_default_sorting_dst(),
                only_class_list=[ExchangeSorts, MergeSorts.MergeSort],
            )
        ```
        """
        call_stack_level: int = 1
        if not (
            obj_to_be_inspected := cls._get_module_to_be_inspected(
                call_stack_level=call_stack_level
            )
        ):
            return

        loaded_classes: Iterable[
            type[MixInParentAlgorithmVisualizationT_co]
        ] = get_child_classes(
            obj_to_be_inspected=obj_to_be_inspected,
            parent_class=MixInParentAlgorithmVisualization,
        )
        available_classes: Iterable[
            type[MixInParentAlgorithmVisualizationT_co]
        ] = VisualizationManager._get_valid_classes(loaded_classes, call_stack_level=1)

        filtered_dict: dict[
            type[MixInParentAlgorithmVisualizationT_co],
            list[type[ChildAlgorithmVisualizationT_co]],
        ] = {}
        if only_class_list:
            for cls_ in only_class_list:
                cls__mro = inspect.getmro(cls_)
                if MixInParentAlgorithmVisualization in cls__mro:
                    if cls_ not in filtered_dict:
                        filtered_dict[cls_] = []  # type:ignore
                elif ChildAlgorithmVisualization in cls__mro:
                    if outer_cls := get_outer_class(
                        outer_candidate_classes=filtered_dict, inner_class=cls_
                    ):
                        filtered_dict[outer_cls].append(cls_)  # type:ignore
                    elif outer_cls := get_outer_class(
                        outer_candidate_classes=available_classes,
                        inner_class=cls_,
                    ):
                        filtered_dict[outer_cls] = [cls_]  # type:ignore
        if not filtered_dict:
            filtered_dict = {x: [] for x in available_classes}

        for parent_class, child_class_list in filtered_dict.items():
            parent_class.call_child_classes(
                dst=copy.deepcopy(dst), only_class_list=child_class_list
            )

    @classmethod
    def call_modules(
        cls, package_or_module_paths: Optional[Iterable[Path]] = None
    ) -> Optional[str]:
        """<package_or_module_paths> will be called in order with sorted state where key is absolute modules path.

        You can call such as:

        ```
        VisualizationManager.call_modules(
            [
                Path("src/python/wbfw109/labs/builtins/statements.py"),
                Path(
                    "/home/wbfw109/repository/study-core/src/python/wbfw109/labs/builtins/system.py"
                ),
                Path("wbfw109/labs/basics/sequences"),
            ]
        )
        ```
        
        ---
        It assume that
        - Similar code of the following code is written in last of each file:

        ```
        if __name__ == "__main__" or VisualizationManager.center_control_state:
            VisualizationManager.call_parent_algorithm_classes(...)
            VisualizationManager.call_root_classes(...)
        ```
        - Importing of each file in one time does not raise circular import error.

        ---
        Args:
            package_or_module_paths: Optional[Iterable[Path]]: if it is None, search modules from "wbfw109/labs".
                valid path is (absolute path | relative path from (<working Directory> | one of "sys.path")

        Returns:
            bool: When <interactive_shell> not exists in Runtime, returns message. otherwise None.

        Implementation:
            It must delegate responsibility to a file because usage of method call in each file is different,
            To do so, module re-importing is required.
        """
        with cls._central_control() as retval:
            if not retval:
                return "[Pass] method <call_modules>. No active Interactive Shell."

            # search modules
            target_module_str_iter: Optional[list[str]] = None
            if package_or_module_paths:
                target_module_str_iter = get_module_name_list(package_or_module_paths)
            else:
                valid_path_pair: Optional[PathPair] = get_valid_path_pair_from_sys_path(
                    Path("wbfw109/labs")
                )
                if valid_path_pair:
                    target_module_str_iter = get_module_name_list(
                        valid_path_pair.absolute.glob("**/*.py")
                    )
            if not target_module_str_iter:
                return "[Pass] method <call_modules>. can not find valid modules"

            # import or re-import modules
            # dict value indicates that False is new module, True is already loaded module.
            modules_to_be_loaded: dict[str, bool] = {}
            modules_to_be_loaded.update(
                {
                    module_str: True
                    for module_str in set(target_module_str_iter).intersection(
                        sys.modules.keys()
                    )
                }
            )
            modules_to_be_loaded.update(
                {
                    module_str: False
                    for module_str in set(target_module_str_iter).difference(
                        modules_to_be_loaded
                    )
                }
            )
            # it guarantees call order to module in sort
            for module_str, is_previous_loaded_module in sorted(
                modules_to_be_loaded.items()
            ):
                print(f"\n\n===== {module_str} =====")
                match is_previous_loaded_module:
                    case True:
                        importlib.reload(sys.modules[module_str])
                    case False:
                        importlib.import_module(module_str)
