"""Path manipulation"""
import functools
import itertools
import sys
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Optional


class PathPair(NamedTuple):
    absolute: Path
    relative_parent: Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent


def get_valid_path_pair_from_sys_path(path: Path) -> Optional[PathPair]:
    """
    Returns:
        Optional[PathPair]: if resolved path is not exists, return None.
    """
    if path.root.startswith("/"):
        # if absolute path
        for sys_path in reversed(sys.path):
            if path.is_relative_to(sys_path) and path.exists():
                return PathPair(absolute=path, relative_parent=Path(sys_path))
    else:
        # if relative path:
        for sys_path in reversed(sys.path):
            absolute_candidate_path: Path = sys_path / path
            if sys_path and absolute_candidate_path.exists():
                return PathPair(
                    absolute=absolute_candidate_path, relative_parent=Path(sys_path)
                )
    return None


def filter_module_file(path: Path, stem_filter: str = "__init__") -> Optional[Path]:
    return path if path.suffix == ".py" and path.stem != stem_filter else None


def get_module_path_pair_list(path_pair: PathPair) -> list[PathPair]:
    """Note that it does not iterates sub-packages."""
    module_path_pair_list: list[PathPair] = []
    if path_pair.absolute.is_dir():
        # if path is a package
        for absolute_path in path_pair.absolute.iterdir():
            if module_path := filter_module_file(absolute_path):
                module_path_pair_list.append(
                    PathPair(
                        absolute=module_path, relative_parent=path_pair.relative_parent
                    )
                )
    else:
        # if path is a module
        if module_path := filter_module_file(path_pair.absolute):
            module_path_pair_list.append(path_pair)
    return module_path_pair_list


def convert_module_path_to_qualified_name(path: Path) -> str:
    return ".".join([*path.parts[:-1], path.stem])


def get_module_name_list(paths: Iterable[Path]) -> list[str]:
    """📝 Note that it automatically filters duplication modules when absolute paths are same from relative paths argument.
    So result module name will be shortest name of available names.

    But, it does not guarantee sorted modules.

    Args:
        paths (Iterable[Path]): searchable paths from one of "sys.path".
            You could check the value by running normally the statement rather than in Interactive Windows.

    Returns:
        list[str]: list that have fully qualified module names like "wbfw109.labs.builtin.statements".

    Implementation:
        To remove duplication modules, it uses a algorithm in similar way to Sieve of Eratosthenes.
    """
    valid_path_pair_list: list[PathPair] = []
    module_path_pair_list: list[PathPair] = []
    for path in paths:
        if valid_path_pair := get_valid_path_pair_from_sys_path(path):
            valid_path_pair_list.append(valid_path_pair)
    for path_pair in valid_path_pair_list:
        module_path_pair_list.extend(get_module_path_pair_list(path_pair))

    # remove duplication in similar way to Sieve of Eratosthenes.
    module_path_pair_list_len: int = len(module_path_pair_list)
    is_checked_list: list[bool] = [False for _ in range(module_path_pair_list_len)]
    is_unique_list: list[bool] = [False for _ in range(module_path_pair_list_len)]
    get_max_relative_parent_len: Callable[
        [int, list[PathPair]], int
    ] = lambda i, path_pair_list: len(path_pair_list[i].relative_parent.parts)

    for i in range(module_path_pair_list_len):
        if is_checked_list[i]:
            continue
        same_absolute_path_pair_list: list[PathPair] = [module_path_pair_list[i]]
        is_checked_list[i] = True
        for j in range(i + 1, module_path_pair_list_len):
            if (
                not is_checked_list[j]
                and module_path_pair_list[j].absolute
                == module_path_pair_list[i].absolute
            ):
                is_checked_list[j] = True
                same_absolute_path_pair_list.append(module_path_pair_list[j])

        is_unique_list[
            i
            + max(
                (i for i in range(len(same_absolute_path_pair_list))),
                key=functools.partial(
                    get_max_relative_parent_len,
                    path_pair_list=same_absolute_path_pair_list,
                ),
            )
        ] = True

    return list(
        map(
            convert_module_path_to_qualified_name,
            [
                path_pair.absolute.relative_to(path_pair.relative_parent)
                for path_pair in itertools.compress(
                    module_path_pair_list, is_unique_list
                )
            ],
        )
    )
