"""Common functions to dictionary of object"""
import copy
from collections.abc import Mapping
from typing import Any, Optional, TypedDict

from wbfw109.libs.objects.object import (  # type: ignore
    get_default_value_map,
    get_field_annotations,
)


class TypedDictFactory:
    """# Todo: Implementation to nested default typed_dict"""

    def __init__(self, default_map: Optional[dict[str, Any]]) -> None:
        if default_map:
            self._default_map: dict[str, Any] = default_map
        else:
            self._default_map: dict[str, Any] = get_default_value_map()

    @property
    def default_map(self) -> dict[str, Any]:
        return self._default_map

    @default_map.setter
    def default_map(self, value: dict[str, Any]) -> None:
        self._default_map = value

    def create_default_typed_dict(
        self, typed_dict_class: type[TypedDict], /
    ) -> dict[str, Any]:

        return {
            field_name: copy.deepcopy(self.default_map[type_name])
            for field_name, type_name in get_field_annotations(typed_dict_class).items()
        }


def update_deep_dict(
    original_dict: dict[Any, Any], new_dict: dict[Any, Any]
) -> dict[Any, Any]:
    """https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"""
    for k, v in new_dict.items():
        if isinstance(v, Mapping):
            temp = update_deep_dict(original_dict.get(k, {}), v)  # type: ignore
            original_dict[k] = temp
        elif isinstance(v, list):
            original_dict[k] = original_dict.get(k, []) + v
        else:
            original_dict[k] = new_dict[k]
    return original_dict
