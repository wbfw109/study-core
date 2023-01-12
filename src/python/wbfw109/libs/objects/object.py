"""object package in next level to typing package"""
import copy
import inspect
import re
import typing
from pprint import pprint
from typing import (
    Any,
    ForwardRef,
    Generic,
    Iterable,
    Mapping,
    MutableSequence,
    Optional,
)

from wbfw109.libs.typing import JsonType, T  # type: ignore

# ? TODO use function typing.get_type_hints() instead of function get_annotations()


# Title: default, annotations ~
def get_base_default_value_map() -> dict[type[object], Any]:
    return {
        bool: False,
        int: 0,
        float: 0.0,
        str: "",
        tuple: (),
        set: set(),
        list: [],
        dict: {},
    }


def get_default_value_map(
    *,
    modify_dict: Optional[dict[type[object], Any]] = None,
) -> dict[str, Any]:
    """
    Returns:
        dict[str, Any]: dict[type_name, default_value]
    """
    default_value_map = get_base_default_value_map()
    if modify_dict:
        for k, v in modify_dict.items():
            default_value_map[k] = v
    return {k.__name__: v for k, v in default_value_map.items()}  # type: ignore


def get_default_value_map_as_type(
    *, modify_dict: Optional[dict[type[object], Any]] = None
) -> dict[type[object], Any]:
    default_value_map = get_base_default_value_map()
    if modify_dict:
        for k, v in modify_dict.items():
            default_value_map[k] = v
    return default_value_map


def get_field_annotations(cls_or_obj: type[object] | object, /) -> dict[str, str]:
    """
    📝 Note that you must know it not distinguish namespace of class or object.
    Returns:
        dict[str, str]: dict[field_name, type_name]
    """
    if inspect.isclass(cls_or_obj):
        field_annotations: dict[str, Any] = inspect.get_annotations(cls_or_obj)
        for field_name, field_type in field_annotations.copy().items():
            if isinstance(field_type, ForwardRef):
                field_annotations[field_name] = field_type.__forward_arg__
            if inspect.isclass(field_type):
                field_annotations[field_name] = field_type.__name__
        return field_annotations
    else:
        return {k: type(v).__name__ for k, v in vars(cls_or_obj).items()}


def get_not_constant_field_annotations(
    cls_or_obj: type[object] | object, /
) -> dict[str, str]:
    """find not constant variables by using naming convention
    Returns:
        dict[str, str]: dict[field_name, type_name]
    """
    find_uppercase = re.compile("[A-Z]")
    field_annotations: dict[str, Any] = get_field_annotations(cls_or_obj)

    for field_name in field_annotations.copy().keys():
        if re.search(find_uppercase, field_name):
            del field_annotations[field_name]
    return field_annotations


def initialize_fields(
    cls_or_obj: type[object] | object, /, *, default_map: dict[str, Any]
) -> None:
    for field_name, type_name in get_field_annotations(cls_or_obj).items():
        setattr(cls_or_obj, field_name, copy.deepcopy(default_map[type_name]))


def initialize_not_constant_fields(
    cls_or_obj: type[object] | object, /, *, default_map: dict[str, Any]
) -> None:
    for field_name, type_name in get_not_constant_field_annotations(cls_or_obj).items():
        setattr(cls_or_obj, field_name, copy.deepcopy(default_map[type_name]))


# Title: nested class, inheritance ~
def get_child_classes(
    obj_to_be_inspected: object | type[object],
    parent_class: type[object],
) -> list[Any]:
    """<obj_to_be_inspected> e.g.: sys.modules["__main__"], classes
    - 📝 Note that type of returned elements is <parent_class>_co.
        but in Python, "TypeVar bound type cannot be generic" so type hint is list[Any].
    """
    target_list: list[Any] = []
    for _, obj in inspect.getmembers(obj_to_be_inspected, predicate=inspect.isclass):
        if issubclass(obj, parent_class) and obj != parent_class:
            obj_mro: tuple[type[object]] = inspect.getmro(obj)
            # obj_mro[0] is leaf a descendant class of mro, obj_mro[-1] is "object."
            if obj_mro[1] == parent_class:
                target_list.append(obj)
    return target_list


def get_descendant_classes(
    obj_to_be_inspected: object | type[object],
    parent_class: type[object],
) -> list[Any]:
    """<obj_to_be_inspected> e.g.: sys.modules["__main__"], classes
    - 📝 Note that type of returned elements is <parent_class>_co.
        but in Python, "TypeVar bound type cannot be generic" so type hint is list[Any].
    """
    target_list: list[Any] = []
    for _, obj in inspect.getmembers(obj_to_be_inspected, predicate=inspect.isclass):
        if issubclass(obj, parent_class) and obj != parent_class:
            target_list.append(obj)

    return target_list


def is_inner_class(outer_candidate_cls: type[object], inner_cls: type[object]) -> bool:
    """It checks only 1 level difference."""
    return inspect.getmro(inner_cls)[0].__qualname__.split(".")[:-1] == inspect.getmro(
        outer_candidate_cls
    )[0].__qualname__.split(".")


def get_outer_class(
    outer_candidate_classes: Iterable[type[object]], inner_class: type[object]
) -> None | type[object]:
    """It searches only 1 level difference."""
    for outer_candidate_cls in outer_candidate_classes:
        if is_inner_class(
            outer_candidate_cls=outer_candidate_cls, inner_cls=inner_class
        ):
            return outer_candidate_cls


class SelfReferenceHelper(Generic[T]):
    """It helps to distinguish nested type of the class that have self-referenced type in list or in dict as value).

    Usage 🔪 initialize this class, and use method <get_cls_object>.

    - It contains common used attributes. specially, <self.type> can be used in wrapping values as the class type by using __new__().
    - Recommend to use <cls_> with <dataclasses.dataclass>.
        It delegates the responsibility to <cls_> whether keywords and types matches to <cls_> or not including <Optional>.
    ---
    Limitation: It assumes that
        - the class to be serialized does not have any union and optional type.
    ---
    Implementation:
        - It uses a algorithm based on type annotations

    ---
    Motivation:
        - library <dataclass_wizard> not supports <from_dict> from self reference typing; it raise error: RecursionError: maximum recursion depth exceeded while calling a Python object.
    """

    def __init__(
        self,
        cls_: type[T],
    ) -> None:
        self.type = cls_
        self.type_hints = typing.get_type_hints(cls_)
        self.self_type_attributes: list[str] = []
        self.self_type_list_attributes: list[str] = []
        self.self_type_dict_attributes: list[str] = []

        for attribute_name, obj_type in self.type_hints.items():
            origin_type = typing.get_origin(obj_type)
            args_type = typing.get_args(obj_type)
            if not origin_type:
                origin_type = obj_type

            if issubclass(origin_type, MutableSequence) and args_type[0] == cls_:
                self.self_type_list_attributes.append(attribute_name)
            elif issubclass(origin_type, Mapping) and args_type[1] == cls_:
                self.self_type_dict_attributes.append(attribute_name)
            elif origin_type == self.type:
                self.self_type_attributes.append(attribute_name)

    def get_cls_object(self, *, dict_obj: JsonType) -> T:
        new_obj = self.type(**dict_obj)
        self.set_self_reference_object_from_json(new_obj)
        return new_obj

    def set_self_reference_object_from_json(self, obj: object) -> None:
        """
        Implementation:
            - Attributes in <self.not_self_type_attributes> are already processed in initializing object.
                So the conditional expression is not required.
        """
        for attribute in self.type_hints:
            if attribute in self.self_type_list_attributes:
                self_type_list: list[T] = getattr(obj, attribute)
                for i in range(len(self_type_list)):
                    self_type_list[i] = self.type(**self_type_list[i])
                    self.set_self_reference_object_from_json(self_type_list[i])
            elif attribute in self.self_type_dict_attributes:
                self_type_dict: dict[str, T] = getattr(obj, attribute)
                for k in self_type_dict:
                    self_type_dict[k] = self.type(**self_type_dict[k])
                    self.set_self_reference_object_from_json(self_type_dict[k])
            elif attribute in self.self_type_attributes:
                setattr(obj, attribute, self.type(**getattr(obj, attribute)))
                self.set_self_reference_object_from_json(getattr(obj, attribute))


class MixInJsonSerializable:
    """it provides classmethod <from_dict>.
    TODO: __str__ ?
    """

    @classmethod
    def from_dict(cls: type[T], dict_obj: JsonType, /) -> list[T] | T:
        self_reference_helper = SelfReferenceHelper(cls)
        if isinstance(dict_obj, MutableSequence):
            return [self_reference_helper.get_cls_object(dict_obj=x) for x in dict_obj]  # type: ignore
        else:
            return self_reference_helper.get_cls_object(dict_obj=dict_obj)
