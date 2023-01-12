"""
⚠️ deprecated. require to reconstruct file

todo: generator_with_suffle for_pandas.Dataframe.iterrows() and apply in existing logic
"""
import collections.abc
import copy
import dataclasses
import datetime
import inspect
import itertools
import math
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    NamedTuple,
    Sequence,
    TypedDict,
    Union,
)

from wbfw109.libs.typing import ObjectT


def get_dict_from_attribute_to_object(
    objects: Iterable[ObjectT], attribute_name: str
) -> dict[Any, ObjectT]:
    return {
        getattr(origin_object, attribute_name): origin_object
        for origin_object in objects
    }


def get_dict_from_key_to_dict(dicts: Iterable[dict], key_name: str) -> dict[Any, dict]:
    return {origin_dict[key_name]: origin_dict for origin_dict in dicts}


def bulk_update_objects_from_dicts(
    objects: Iterable[ObjectT],
    dicts: Iterable[dict],
    key_or_attribute_name: str,
    fields_to_be_updated: Sequence[str],
) -> Iterable[ObjectT]:
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


def get_unique_id_set(
    count_to_be_added: int,
    get_function: Callable,
    existing_id_set: set = None,
    timeout_seconds: int = 1,
) -> set:
    """
    it is useful when create serial number.

    Args:
        existing_ids (set): existing ids
        additional_id_count (int): count to be generated
        get_function (Callable): function that can create mostly a unique element. e.g. lambda: add(a, b)
        timeout_seconds (int, optional): parameter to prevent infinite loop. Defaults to 5. if this value < 1, it will be set as 1.

    Returns:
        list: additional list to database that have unique elements

    Examples:
        import string
        import secrets

        STRING_ASCII_PLUS_DIGITS: str = string.ascii_uppercase + string.digits


        def get_my_normal_serial_number() -> str:
            return "-".join(
                textwrap.wrap(
                    text="".join(secrets.choice(STRING_ASCII_PLUS_DIGITS) for _ in range(1)),
                    width=5,
                )
            )

        get_unique_id_set(count_to_be_added=5, get_function=get_my_normal_serial_number)

    """
    # preprocess
    if existing_id_set is None:
        existing_id_set = set()
    updated_id_set: set = copy.deepcopy(existing_id_set)
    required_loop_count: int = count_to_be_added
    goal_ids_count: int = len(updated_id_set) + count_to_be_added
    if timeout_seconds < 1:
        timeout_seconds = 1
    start_time: float = time.time()

    # process
    while required_loop_count > 0:
        updated_id_set.update([get_function() for _ in range(required_loop_count)])
        required_loop_count = goal_ids_count - len(updated_id_set)
        if time.time() - start_time >= timeout_seconds:
            raise Exception(f"TimeoutError: {timeout_seconds} seconds")

    return updated_id_set - existing_id_set


def convert_WSL_path_to_windows_path_in_batch(
    wsl_path_list: list[Union[Path, str]]
) -> list[Path]:
    """
    Convert WSL path to Windows path.

    Args:
        wsl_path_list (list[Union[Path, str]]): _description_

    Returns:
        list[Path]: _description_
    """
    new_wsl_path_list: list[Path] = []
    if sys.platform.startswith("win"):
        for wsl_path in wsl_path_list:
            if wsl_path.root == "\\":
                new_wsl_path_list.append((Path(str(wsl_path).replace("\\", "/"))))
    else:
        new_wsl_path_list = copy.copy(wsl_path_list)

    # replace name like "/mnt/f/a_directory" to "f:/a_directory".
    for index, new_wsl_path in enumerate(new_wsl_path_list):
        if new_wsl_path.root == "/" or new_wsl_path.root == "\\":
            new_wsl_path_list[index] = Path(
                "/".join([new_wsl_path.parts[2] + ":", *new_wsl_path.parts[3:]])
            )

    return new_wsl_path_list


def get_file_list(
    input_directory: Union[Path, str],
    start_with_string: str = "",
    extensions: list[str] = [],
) -> list[Path]:
    """
    Args:
        input_directory (Union[Path, str]): [description]
        start_with_string (str, optional): [description]. Defaults to "".
        extensions (list[str], optional): [description]. Defaults to [].

    Returns:
        list[Path]: [description]
    """
    input_directory: Path = Path(input_directory)
    start_with_string_regex = f"^{start_with_string}"
    extensions_string_regex = "|".join(extensions)
    if extensions_string_regex:
        extensions_string_regex = f"\.({extensions_string_regex})$"

    regex_pattern: re.Pattern = re.compile(
        f"{start_with_string_regex}.*{extensions_string_regex}"
    )

    return [
        file_path
        for file_path in input_directory.iterdir()
        if regex_pattern.search(
            string=f"{file_path.name}",
        )
    ]


def get_joined_string_filtering_empty_elements(
    join_list: list[str],
    iterator_list: list[Iterable[str]],
    prefix_list: list[str] = [],
) -> str:
    """It uses true iterator to be operated as [prefix_list[i], joined_string, *iterator_list[i]] order.

    Args:
        join_list (list[str]): [description]
        iterator_list (list[Iterable[str]]): [description]
        prefix_list (list[str], optional): [description]. Defaults to [].

    Returns:
        str: joined_string
    """
    assert len(join_list) == len(iterator_list)
    if not prefix_list:
        prefix_list = ["" for _ in range(len(join_list))]
    assert len(join_list) == len(prefix_list)

    joined_string: str = ""
    for i in range(len(join_list)):
        joined_string = join_list[i].join(
            list(filter(None, [prefix_list[i], joined_string, *iterator_list[i]]))
        )

    return joined_string


def get_changed_file_without_suffix(
    input_file: Union[Path, str],
    output_directory: Union[Path, str],
    revision_list: list[str],
) -> Path:
    """
    + This function use following mechanism:
        - separate "base_stem", "revision_stem", "created datetime" by "-" (dash) character.
        - separate "revision_stem" shards by "_" (underscore) characters
        - file_stem[0] is base_stem, file_stem[1] is revision list, file_stem[2] is created datetime

    e.g.
        for file in [
            Path("/mnt/c/Users/wbfw109/images_temp_1/cam01_drk_tumbler_02_00.jpg"),
            Path(
                "/mnt/c/Users/wbfw109/images_temp_1/cam01_drk_tumbler_02_00-superimposed_noised-20210805_154047_017283.jpg"
            ),
            Path(
                "/mnt/c/Users/wbfw109/images_temp_1/cam01_drk_tumbler_02_00-superimposed-20210805_154047_017283.jpg"
            ),
        ]:
            print(file.stem)
            for revision in [[], ["rotated", "blurred"]]:
                print(
                    get_changed_file_without_suffix(
                        input_file=file,
                        output_directory="/mnt/c/Users/wbfw109/images_temp_2",
                        revision_list=revision,
                    ).stem
                )
            print()

    Args:
        input_file (Union[Path, str]): [description]
        output_directory (Union[Path, str]): [description]
        revision_list (list[str]): [description]

    Returns:
        Path: path with file stem. not including file suffix
    """
    input_file: Path = Path(input_file)
    output_directory: Path = Path(output_directory)

    file_stem: list[str] = input_file.stem.split("-")
    if len(file_stem) > 2:
        revision_list: list = [file_stem[1], *revision_list]
    else:
        revision_list: list = revision_list

    return output_directory / "{file_stem}-{datetime}".format(
        file_stem=get_joined_string_filtering_empty_elements(
            join_list=["_", "-"],
            iterator_list=[revision_list, ""],
            prefix_list=["", file_stem[0]],
        ),
        datetime=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    )


def get_sample_weight_from_dataset_directory_as_same_ratio(
    input_directory: Union[Path, str],
    sample_unique_class_list: list[str],
    base_weight_mode: str = "min",
    prints_object_class_size: bool = False,
) -> dict[str, float]:
    """
    + about Formula
        sample_weight = dividend / the number of image in a directory

    e.g.
        tf_object_detection_api_config_format = [
            f"sample_from_datasets_weights: {value}"
            for value in get_balanced_sample_weight_from_path(
                input_directory=Path("/mnt/c/Users/wbfw109/MyDrive/shared_resource/images/train"),
                sample_unique_class_list=sorted(
                    list(
                        set(
                            machine_learning.get_converted_dict_from_extension_pbtxt(
                                "/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/my_label_map.pbtxt"
                            )
                        )
                        - {"tray", "unknown"}
                    )
                ),
                base_weight_mode="max",
                prints_object_class_size=True,
            ).values()
        ]
        print(tf_object_detection_api_config_format)

    Args:
        input_directory (Union[Path, str]): [description]
        sample_unique_class_list (list[str]): [description]
        base_weight_mode (str, optional): [description]. Defaults to "min".
        prints_object_class_size (bool, optional): [description]. Defaults to False.

    Returns:
        dict[str, float]: [description]
    """
    input_directory: Path = Path(input_directory)
    assert input_directory.is_dir()
    assert base_weight_mode in ["min", "max", "mean"]

    sample_class_size_dict: dict[str, int] = {}
    for sample_class in sample_unique_class_list:
        sample_class_size_dict[sample_class] = len(
            list(input_directory.glob(f"*{sample_class}*.jpg"))
        )

    if base_weight_mode == "min":
        base_weight_unit: int = min(sample_class_size_dict.values())
    elif base_weight_mode == "max":
        base_weight_unit: int = max(sample_class_size_dict.values())
    elif base_weight_mode == "mean":
        base_weight_unit: int = sum(sample_class_size_dict.values()) / len(
            sample_class_size_dict
        )

    sample_class_weight_dict: dict[str, float] = {}
    for sample_class in sample_unique_class_list:
        sample_class_weight_dict[sample_class] = (
            base_weight_unit / sample_class_size_dict[sample_class]
        )

    if prints_object_class_size:
        for sample_class in sample_unique_class_list:
            print(
                (
                    sample_class,
                    sample_class_size_dict[sample_class],
                    sample_class_weight_dict[sample_class],
                    sample_class_size_dict[sample_class]
                    * sample_class_weight_dict[sample_class],
                )
            )
    return sample_class_weight_dict


def get_sample_weight_from_dataset_directory_list_as_same_ratio(
    input_directory_list: list[Union[Path, str]],
    base_weight_mode: str = "min",
    prints_dataset_size: bool = False,
) -> dict[str, float]:
    """
    + about Formula
        sample_weight = dividend / the number of image in a directory

    e.g.
        tf_object_detection_api_config_format = [
            value
            for value in get_sample_weight_from_dataset_directory_list_as_same_ratio(
                input_directory_list=[
                    Path(
                        "/mnt/c/users/wbfw109/image_tasks_sorted_true/pickup_bar-image_and_pascal_voc_set-1"
                    ),
                    Path(
                        "/mnt/c/users/wbfw109/image_tasks_sorted_true/pickup_bar-image_and_pascal_voc_set-2"
                    ),
                    Path("/mnt/c/users/wbfw109/image_task_211019_AreaSample_no_exif_true"),
                ],
                base_weight_mode="max",
                prints_dataset_size=False,
            )
        ]
        print(tf_object_detection_api_config_format)

    Args:
        input_directory_list (list[Union[Path, str]]): [description]
        base_weight_mode (str, optional): [description]. Defaults to "min".
        prints_dataset_size (bool, optional): [description]. Defaults to False.

    Returns:
        dict[str, float]: [description]
    """
    print(f"===== Start function; {inspect.currentframe().f_code.co_name}")

    assert base_weight_mode in ["min", "max", "mean"]
    for i in range(len(input_directory_list)):
        input_directory_list[i] = Path(input_directory_list[i])
    input_directory_size_list = [
        int(len(list(input_directory.iterdir())) / 2)
        for input_directory in input_directory_list
    ]

    if base_weight_mode == "min":
        base_weight_unit: int = min(input_directory_size_list)
    elif base_weight_mode == "max":
        base_weight_unit: int = max(input_directory_size_list)
    elif base_weight_mode == "mean":
        base_weight_unit: int = sum(input_directory_size_list) / len(
            input_directory_size_list
        )

    sample_dataset_weight_dict: dict[int, float] = {}
    for i in range(len(input_directory_list)):
        sample_dataset_weight_dict[i] = base_weight_unit / input_directory_size_list[i]

    if prints_dataset_size:
        for i in range(len(input_directory_list)):
            print(
                (
                    input_directory_list[i],
                    input_directory_size_list[i],
                    sample_dataset_weight_dict[i],
                    input_directory_size_list[i] * sample_dataset_weight_dict[i],
                )
            )
    return list(
        zip(
            [input_directory.name for input_directory in input_directory_list],
            sample_dataset_weight_dict.values(),
        )
    )


def generator_with_shuffle(iterator: Iterable, iteration: int, random_seed: int):
    random_seed_rd: random.Random = random.Random(random_seed)
    for _ in range(iteration):
        random_seed_rd.shuffle(iterator)
        for j in iterator:
            yield j


def split_customarily_train_and_test_with_annotation(
    src_path: Path,
    names: list[str],
    extensions: list[str],
    angle_count: int,
    seed: int = 42,
) -> None:
    """
    Split if each image has the same number of files with different angles

    e.g.
        from wbfw109.utility.machine_learning import get_converted_dict_from_extension_pbtxt
        from pathlib import Path
        import subprocess

        annotations_path: Path = Path("/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations")
        labels_names: dict = get_converted_dict_from_extension_pbtxt(
            annotations_path / "my_label_map.pbtxt",
        )

        IMAGES_PATH: Path = Path("/mnt/c/Users/wbfw109/MyDrive/shared_resource/images")
        for name in list(labels_names):
            print(
                call_root_classes(
                    f"ls '{str(IMAGES_PATH)}' | grep .*{name}.*.jpg | wc -l",
                    capture_output=True,
                    text=True,
                    shell=True,
                ).stdout
            )

        split_customarily_train_and_test_with_annotation(
            src_path=IMAGES_PATH,
            names=names,
            extensions=[
                "jpg", "xml"
            ],
            angle_count=4,
        )
    """
    # initialize
    assert src_path.is_dir()
    train_path: Path = src_path / "train"
    test_path: Path = src_path / "test"
    train_path.mkdir(exist_ok=True)
    assert not any(train_path.iterdir())
    test_path.mkdir(exist_ok=True)
    assert not any(test_path.iterdir())

    random.seed(seed)
    all_input_file_list: list[list[str]] = []
    all_train_index_list: list[list[str]] = []
    all_test_index_list: list[list[str]] = []

    # process
    for name in names:
        input_file_list: list[Path] = sorted(
            [
                file_path
                for file_path in src_path.iterdir()
                if re.search(
                    f".*({name}).*.{extensions}$".format(
                        name=name, extensions="|".join(extensions)
                    ),
                    str(file_path),
                )
            ],
            key=lambda f: f.suffix,
        )

        original_unit_count: int = int(len(input_file_list) / len(extensions))

        all_input_file_list.append(input_file_list)

        index_test = []

        for unit_index in range(int(original_unit_count / angle_count)):
            random_index = random.randint(
                unit_index * angle_count, (unit_index + 1) * angle_count - 1
            )
            index_test.extend(
                [random_index + original_unit_count * i for i in range(len(extensions))]
            )
        index_test = sorted(index_test)
        all_test_index_list.append(index_test)
        all_train_index_list.append(
            list(set(range(len(input_file_list))) - set(index_test))
        )

    for i, index_test_list in enumerate(all_test_index_list):
        for index_test in index_test_list:
            shutil.copy2(str(all_input_file_list[i][index_test]), str(test_path))

            # test
            # print(file_list[i][index_test])

    for i, index_test_list in enumerate(all_train_index_list):
        for index_test in index_test_list:
            shutil.copy2(str(all_input_file_list[i][index_test]), str(train_path))

            # test
            # print(file_list[i][index_test])

    print("train and test split (copy) completes")


@dataclasses.dataclass
class MeasurementOfCombinations:
    """
    # Todo: prevent overflow from get_bytes_on_case_list()

    e.g.
        measurement_of_combinations = MeasurementOfCombinations(
            case_dict={}, original_data_bytes=38936576
        )
        # add cases
        tensorflow_api_normal_augmentation_list: list = [iaa.Fliplr(1.0), iaa.Flipud(1.0)]
        measurement_of_combinations.create_case_for_consecutive_k(
            key="tensorflow_api_normal_augmentation",
            iterator=tensorflow_api_normal_augmentation_list,
            end_k=len(tensorflow_api_normal_augmentation_list),
        )

        base_angle = 15
        angle_max_multiplier = 6
        available_rotation_list: list = [
            iaa.Affine(rotate=base_angle * multiplier, mode="edge", fit_output=True)
            for multiplier in range(1, angle_max_multiplier + 1)
        ]
        measurement_of_combinations.create_case_for_consecutive_k(
            key="available_rotation", iterator=available_rotation_list, end_k=1
        )

        available_case_list: list = measurement_of_combinations.get_available_case_list()
        available_case_size: int = len(available_case_list)
        # size; 1: KibiByte, 2: MebiByte, 3: GibiByte
        measurement_of_combinations.get_bytes_on_case_list() / pow(1024, 3) * 2

        augmentation_case_list: list = [
            iaa.Sequential(agumentation_case, random_order=False)
            for agumentation_case in available_case_list
        ]

        for augmnetation in augmentation_case_list:
            "_".join([x.__class__.__name__ for x in augmnetation.get_all_children()]).lower()
    """

    def __init__(self, original_data_bytes: int = 1) -> None:
        self.case_dict: dict[str, list] = {}
        self.original_data_bytes: int = original_data_bytes

    def create_case_for_consecutive_k(
        self, key, iterator: Iterable, end_k: int, start_k: int = 1
    ) -> None:
        """
        + About start_k, end_k
            - it includes start_k, end_k. not excludes. default value is 1.
            - later to product it with other cases in oder to use function get_available_case_list(), it append one non-selected at the last.
                so if you set start_k as 0, duplicated [] element will be created.


        Reference
            https://en.wikipedia.org/wiki/Combination

        Args:
            key ([type]): [description]
            iterator (Iterable): [description]
            end_k (int): [description]
            start_k (int, optional): [description]. Defaults to 1.
        """
        self.case_dict[key] = list(
            itertools.chain.from_iterable(
                [
                    [
                        list(one_case)
                        for one_case in itertools.combinations(
                            iterator,
                            i,
                        )
                    ]
                    for i in range(start_k, end_k + 1)
                ]
            )
        )
        self.case_dict[key].append([])

    def get_available_case_list(
        self, case_key_list: list[str] = [], should_include_empty_case: bool = True
    ) -> list[list]:
        """
        Args:
            case_key_list (list[str], optional): if this value is empty, it uses all key. Defaults to [].
            should_include_empty_case (bool, optional): [description]. Defaults to True.

        Returns:
            list: available_case_list
        """
        if case_key_list:
            iterator_list: list = [
                self.case_dict[true_key]
                for true_key in case_key_list in self.case_dict.keys()
            ]
        else:
            iterator_list: list = self.case_dict.values()

        available_case_list: list[list] = [
            list(
                itertools.chain.from_iterable(
                    [
                        unpacked_existing_combination_case
                        for unpacked_existing_combination_case in packed_new_combination_case
                        if unpacked_existing_combination_case
                    ]
                )
            )
            for packed_new_combination_case in list(itertools.product(*iterator_list))
        ]
        if not should_include_empty_case:
            available_case_list = list(filter(None, available_case_list))

        return available_case_list

    def get_bytes_on_case_list(self, case_key_list: list[str] = []) -> int:
        """
        Args:
            case_key_list (list[str], optional): refer to the method get_available_case_list(). Defaults to [].

        Returns:
            int: bytes
        """
        return self.original_data_bytes * len(
            self.get_available_case_list(case_key_list=case_key_list)
        )
