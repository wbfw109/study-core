from typing import Any, Iterable, TypeVar
import dataclasses

# Object
O = TypeVar("O")


def get_dict_from_attribute_to_object(
    objects: Iterable[O], attribute_name: str
) -> dict[Any, O]:
    return {
        getattr(origin_object, attribute_name): origin_object
        for origin_object in objects
    }

def get_dict_from_key_to_dict(
    dicts: Iterable[dict], key_name: str
) -> dict[Any, dict]:
    return {
        origin_dict[key_name]: origin_dict
        for origin_dict in dicts
    }
