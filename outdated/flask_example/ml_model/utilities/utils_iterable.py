import copy
from typing import Any, Iterable, Sequence, TypedDict
from ml_model.utilities.utils_typing import (
    O,
    get_dict_from_attribute_to_object,
    get_dict_from_key_to_dict,
)


def bulk_update_objects_from_dicts(
    objects: Iterable[O],
    dicts: Iterable[dict],
    key_or_attribute_name: str,
    fields_to_be_updated: Sequence[str],
) -> Iterable[O]:
    object_dict = get_dict_from_attribute_to_object(
        objects=objects, attribute_name=key_or_attribute_name
    )
    new_data_dict = get_dict_from_key_to_dict(
        dicts=dicts, key_name=key_or_attribute_name
    )

    for key in object_dict.keys():
        for field_to_be_updated in (
            new_data_key
            for new_data_key in new_data_dict[key].keys()
            if new_data_key in fields_to_be_updated
        ):
            setattr(
                object_dict[key],
                field_to_be_updated,
                new_data_dict[key][field_to_be_updated],
            )

    return objects


def validate_iterable_comply_optional_typed_dict(
    targets_iterable: Iterable[dict],
    typed_dict_class: TypedDict,
    required_key_list: list[Any] = None,
) -> bool:
    temp_temp_targets = copy.deepcopy(targets_iterable)
    if len(targets_iterable) < 1:
        return False
    if required_key_list is None:
        required_key_list = []
    optional_typed_dict_annotations: dict = copy.deepcopy(
        typed_dict_class.__annotations__
    )
    required_typed_dict_annotations: dict = {
        key: optional_typed_dict_annotations.pop(key) for key in required_key_list
    }

    optional_type_dict_keys = optional_typed_dict_annotations.keys()
    required_type_dict_keys = required_typed_dict_annotations.keys()
    required_type_dict_keys_length: int = len(required_type_dict_keys)
    for validation_target in temp_temp_targets:
        if len(validation_target) <= required_type_dict_keys_length:
            return False

        if set(required_type_dict_keys).issubset(validation_target) and all(
            [
                isinstance(
                    validation_target[required_key],
                    required_typed_dict_annotations[required_key],
                )
                for required_key in required_type_dict_keys
            ]
        ):
            for required_key in required_type_dict_keys:
                del validation_target[required_key]
        else:
            return False

        for optional_key in validation_target.keys():
            if not (
                optional_key in optional_type_dict_keys
                and isinstance(
                    validation_target[optional_key],
                    optional_typed_dict_annotations[optional_key],
                )
            ):
                return False
    return True
