"""
⚠️ deprecated. require to reconstruct file
utility for machine learning (ML)
"""
import collections
import dataclasses
import datetime
import inspect
import itertools
import math
import random
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional, Tuple, Type, Union

import cv2
import imgaug.augmenters as iaa
import imgaug.augmenters.meta as iaa_meta
import keras
import numpy
import pandas
import pascal_voc_writer
import tensorflow as tf
import wbfw109.libs.utilities.iterable
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from PIL import Image, ImageDraw, ImageFilter
from scipy import stats
from wbfw109.libs.typing import RangeOfTwoPoint

# locally installed library
try:
    TEMP_FILES_PATH: Path = Path.home() / ".local_files"
    NEW_SLIM_PATH: Path = TEMP_FILES_PATH / "tensorflow/models/research/slim"

    sys.path.append(str(NEW_SLIM_PATH))
    sys.path.append(str(TEMP_FILES_PATH / "tensorflow/models/research"))

    from object_detection.utils import label_map_util
    from object_detection.utils import ops as utils_ops
    from object_detection.utils import visualization_utils

    PYTHON_ADDITIONAL_ENV: str = ":".join(
        {str(NEW_SLIM_PATH), str(TEMP_FILES_PATH / "tensorflow/models/research")}
    )
except Exception as e:
    print(f"=== {e}")


def convert_pascal_voc_xml_to_df_and_get(
    xml_directory: Union[Path, str],
    optional_column_list: list = [],
) -> pandas.DataFrame:
    """
    Reference
        https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html


    Args:
        xml_directory (Union[Path, str]): The path containing the .xml files
        optional_column_set (set, optional): when it is not empty, input column will be added to return value.
            current available element: {"area"}
            Defaults to [].

    Returns:
        [type]: [description]
    """
    available_optional_column_list: list = ["area"]
    assert all(
        [x in ["", *available_optional_column_list] for x in optional_column_list]
    )

    xml_directory = Path(xml_directory)
    xml_list: list = []
    for xml_file in xml_directory.glob("*.xml"):
        tree = ET.parse(f"{xml_file}")
        root = tree.getroot()
        filename = root.find("filename").text
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        for member in root.findall("object"):
            bounding_box = member.find("bndbox")
            value = (
                filename,
                width,
                height,
                member.find("name").text,
                int(float(bounding_box.find("xmin").text)),
                int(float(bounding_box.find("ymin").text)),
                int(float(bounding_box.find("xmax").text)),
                int(float(bounding_box.find("ymax").text)),
            )
            xml_list.append(value)

    column_name: list[str] = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    xml_df: pandas.DataFrame = pandas.DataFrame(xml_list, columns=column_name)
    if "area" in optional_column_list:
        xml_df.loc[:, "area"] = (xml_df.loc[:, "xmax"] - xml_df.loc[:, "xmin"]) * (
            xml_df.loc[:, "ymax"] - xml_df.loc[:, "ymin"]
        )

    return xml_df


def refine_pascal_voc_xml_df_and_get(
    xml_df: pandas.DataFrame,
    optional_list_to_be_filtered: list[str] = [],
) -> pandas.DataFrame:
    """
    It use .groupby(["filename"] and aggregate item_list_to_be_filtered using lambda x: list(x).

    default filter column is ["class", "xmin", "ymin", "xmax", "ymax"].

    Args:
        xml_df (pandas.DataFrame): [description]
        optional_list_to_be_filtered (list[str], optional): [description]. Defaults to [].

    Returns:
        pandas.DataFrame: refined_xml_df
    """
    true_item_list_to_be_filtered = [
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        *optional_list_to_be_filtered,
    ]
    return (
        xml_df.filter(items=["filename", *true_item_list_to_be_filtered])
        .groupby(["filename"], as_index=False)
        .aggregate({item: lambda x: list(x) for item in true_item_list_to_be_filtered})
    )


def change_xml_filename_as_true_reference_name(
    input_directory: Union[Path, str]
) -> None:
    """
    it may be required when you use convert_xml_to_tfrecord.py because it brings text in <filename> tag.

    Args:
        input_directory (Union[Path, str]): [description]
    """
    print("===== start function change_xml_filename_as_true_reference_name()")

    input_directory: Path = Path(input_directory)
    for xml_file in input_directory.glob("*.xml"):
        tree: ET = ET.parse(xml_file)
        root: ET.Element = tree.getroot()
        filename_e: ET.Element = root.find("filename")
        actual_file: Path = [
            x for x in xml_file.parent.glob(f"{xml_file.stem}.*") if x.suffix != ".xml"
        ][0]

        if filename_e.text != f"{actual_file.name}":
            filename_e.text = f"{actual_file.name}"
            tree.write(f"{xml_file}")

        # # test
        # print(filename_e.text)
        # print(f"{actual_file.name}")
    print("===== end function change_xml_filename_as_true_reference_name()")


def remove_EXIF_of_image_and_save(
    output_parent_directory: Path, image_list: list[Path]
) -> None:
    for image_file in image_list:
        Image.open(image_file).save(output_parent_directory / image_file.name)


def split_test_dataset_from_image_and_pascal_voc_xml(
    input_directory: Union[Path, str],
    output_directory: Union[Path, str],
    test_dataset_fraction: float,
    split_by_first_separator: Optional[str] = None,
) -> None:
    """
    Todo: remove split_customarily_train_and_test_with_annotation and unifiy into this function (require random seed)
    Todo: it require to check if existing output directory have files, but it collects test files from multiple directory so impossible.

    + About split_by_first_separator
        if this value is not None, test_dataset_fraction will be applied to each group divided into split_by_first_separator.

    Args:
        input_directory (Union[Path, str]): [description]
        output_directory (Union[Path, str]): [description]
        test_dataset_fraction (float): [description]
        split_by_first_separator (Optional[str], optional): [description]. Defaults to None.
    """

    def split_a_test_dataset_from_image_and_pascal_voc_xml(
        output_directory: Path,
        test_dataset_fraction: float,
        file_list: list[Path],
        group_name: str,
    ) -> None:
        file_list_length: int = len(file_list)
        if file_list_length % 2 != 0:
            print(
                f"=== the number of file is not even number. so, pass the {group_name}."
            )
            return

        unique_stems_length: int = int(file_list_length / 2)
        test_dataset_unique_stems_length: int = math.floor(
            unique_stems_length * test_dataset_fraction
        )

        if (
            test_dataset_unique_stems_length < 1
            or test_dataset_unique_stems_length > unique_stems_length
        ):
            print(
                f"=== test_dataset_fraction value is invalid. so, pass the {group_name}."
            )

            return

        # count = 0
        for file in file_list[
            file_list_length - test_dataset_unique_stems_length * 2 :
        ]:
            shutil.move(src=file, dst=output_directory / file.name)
            # # test
            # count+=1
        # print(count)

    input_directory: Path = Path(input_directory)
    output_directory: Path = Path(output_directory)

    file_dict_list_split_by_first_separator: dict[str, list[Path]] = {}

    if not split_by_first_separator:
        split_a_test_dataset_from_image_and_pascal_voc_xml(
            output_directory=output_directory,
            test_dataset_fraction=test_dataset_fraction,
            file_list=wbfw109.libs.utilities.iterable.get_file_list(
                input_directory=input_directory
            ),
            group_name="all files",
        )
    else:
        regex_pattern: re.Pattern = re.compile(
            f"([^{split_by_first_separator}]+{split_by_first_separator}).*"
        )
        for file in input_directory.iterdir():
            if match_object := regex_pattern.match(string=file.name):
                if (
                    match_object.group(1)
                    not in file_dict_list_split_by_first_separator.keys()
                ):
                    file_dict_list_split_by_first_separator[match_object.group(1)] = []

                file_dict_list_split_by_first_separator[match_object.group(1)].append(
                    file
                )
        for (
            key,
            file_list_split_by_first_separator,
        ) in file_dict_list_split_by_first_separator.items():
            split_a_test_dataset_from_image_and_pascal_voc_xml(
                output_directory=output_directory,
                test_dataset_fraction=test_dataset_fraction,
                file_list=file_list_split_by_first_separator,
                group_name=f"{key} files",
            )

    print(f"===== End function; {inspect.currentframe().f_code.co_name}")


@dataclasses.dataclass
class RangeOfTwoPointWithRequiredCount:
    range_list_of_two_point: list[RangeOfTwoPoint] = dataclasses.field(
        default_factory=list
    )
    required_count: Optional[int] = dataclasses.field(default_factory=int)


def create_sorted_image_files_with_pascal_voc_xml_from_CVAT_for_tensorflow_api(
    parent_of_exported_directory_from_CVAT: Union[Path, str],
    available_annotation_index_range_list_of_folder_list: list[
        RangeOfTwoPointWithRequiredCount
    ] = [],
) -> Path:
    """
    + Note
        - you may require to additional operation becuase filename of exported pascal voc xml files may have parent path.
            if that, it occurs error when use other function related to tensorflow.
            so additionally use function change_xml_filename_as_true_reference_name().
        - it expects minimal CVAT directory tree like:
            📂 <parameter: parent_of_exported_directory_from_CVAT>
                ┖─📂 <export root>
                    ┖─📂 Annotations
                        ┖─ 📂 <task name>
                    ┖─📂 JPEGImages
                        ┖─ <task name>
                ...

    e.g.
        refer to the function _augment_padding_from_CVAT_example1().

    + Reference
        - https://github.com/openvinotoolkit/cvat

    Args:
        parent_of_exported_directory (Union[Path, str]): [description]
        available_annotation_index_range_list_of_folder_list (list[RangeOfTwoPointWithLimitCount], optional): [description]. Defaults to [[]].

    Returns:
        Path: copied_root_folder.
    """

    # preprocess
    parent_of_exported_directory_from_CVAT: Path = Path(
        parent_of_exported_directory_from_CVAT
    )
    root_folder_to_be_copied: Path = Path(
        parent_of_exported_directory_from_CVAT.parent
        / f"{parent_of_exported_directory_from_CVAT.name}_sorted"
    )
    root_folder_to_be_copied.mkdir(exist_ok=True)

    if len(list(root_folder_to_be_copied.iterdir())) != 0:
        print(
            f"=== root_folder_to_be_copied is not empty. so, pass the {root_folder_to_be_copied.name}."
        )
        return

    original_folder_list_to_be_iterated: list[Path] = [
        path
        for path in parent_of_exported_directory_from_CVAT.iterdir()
        if path.is_dir()
    ]

    # process
    for original_folder, available_annotation_index_range_list_of_folder in zip(
        original_folder_list_to_be_iterated,
        available_annotation_index_range_list_of_folder_list,
    ):
        # preprocess
        image_folder: Path = original_folder / "JPEGImages"
        annotation_folder: Path = original_folder / "Annotations"
        new_directory: Path = root_folder_to_be_copied / original_folder.name
        new_directory.mkdir(exist_ok=True)
        if not image_folder.exists() or not annotation_folder.exists():
            continue
        if len(list(new_directory.iterdir())) > 0:
            print(
                f"=== already files are exists, so the function passes this folder: {original_folder}."
            )
            continue

        image_folder = list(image_folder.iterdir())[0]
        image_file_list = list(image_folder.iterdir())
        annotation_folder = list(annotation_folder.iterdir())[0]
        annotation_file_list = list(annotation_folder.iterdir())

        # process
        # image_folder
        image_file_list_to_be_copied: list[Path] = []
        annotation_file_list_to_be_copied: list[Path] = []
        current_file_count: int = 0
        for (
            available_annotation_index_range_of_folder
        ) in available_annotation_index_range_list_of_folder.range_list_of_two_point:
            for available_annotation_index in range(
                available_annotation_index_range_of_folder.start,
                available_annotation_index_range_of_folder.end + 1,
            ):
                current_file_count += 1
                image_file_list_to_be_copied.append(
                    image_file_list[available_annotation_index]
                )
                annotation_file_list_to_be_copied.append(
                    annotation_file_list[available_annotation_index]
                )
                if (
                    available_annotation_index_range_list_of_folder.required_count
                    and current_file_count
                    >= available_annotation_index_range_list_of_folder.required_count
                ):
                    break

        if (
            available_annotation_index_range_list_of_folder.required_count
            and current_file_count
            < available_annotation_index_range_list_of_folder.required_count
        ):
            print(
                f"=== file does not enough as required file count ({available_annotation_index_range_list_of_folder.required_count}) in original_folder ({original_folder}). but, just process following operations."
            )

        # save images and annotations
        remove_EXIF_of_image_and_save(
            output_parent_directory=new_directory,
            image_list=image_file_list_to_be_copied,
        )

        for annotation_file in annotation_file_list_to_be_copied:
            shutil.copy2(
                src=f"{annotation_file}", dst=f"{new_directory/annotation_file.name}"
            )

        print(f"=== a process completes to folder; {original_folder}")
    print(f"===== End function; {inspect.currentframe().f_code.co_name}")

    return root_folder_to_be_copied


def draw_rectangle_on_image_from_random_part_of_image(
    on_image_pillow: Image.Image,
    from_image_pillow: Image.Image,
    drawing_rectangle_point: list[int],
) -> None:
    """
    1. it crops from_image_pillow and pastes on_image_pillow.
      (may resize if from_image_pillow size < drawing_rectangle_point size)

    2. it draws rectangle based on drawing_rectangle point.

    ??? 3. return None. if _on_image_pillow is not applied, I must change this code to return PIL.Image.Image.
        but, in iPython it is well.

    Args:
        on_image_pillow (Image.Image): [description]
        from_image_pillow (Image.Image): [description]
        drawing_rectangle_point (list[int]): (xmin, ymin, xmax, ymax) like PIL coordinate system.
    """
    from_image_pillow_copy: Image.Image = from_image_pillow.copy()

    drawing_rectangle_size = numpy.array(
        (
            (drawing_rectangle_point[2] - drawing_rectangle_point[0]),
            (drawing_rectangle_point[3] - drawing_rectangle_point[1]),
        )
    )

    index_to_be_resized: numpy.ndarray = numpy.argwhere(
        from_image_pillow_copy.size - drawing_rectangle_size < 0
    ).flatten()

    # should_resize_background_image_pillow
    if index_to_be_resized.size > 0:
        size_to_be_reisized: list = list(from_image_pillow_copy.size)
        for i in index_to_be_resized:
            size_to_be_reisized[i] = drawing_rectangle_size[i]
        from_image_pillow_copy: Image.Image = from_image_pillow_copy.resize(
            size_to_be_reisized
        )

    chosen_from_image_start_x_to_be_cropped = random.randint(
        0,
        from_image_pillow_copy.size[0] - drawing_rectangle_size[0],
    )
    chosen_from_image_start_y_to_be_cropped = random.randint(
        0,
        from_image_pillow_copy.size[1] - drawing_rectangle_size[1],
    )

    from_image_rectangle_to_be_cropped: numpy.ndarray = numpy.array(
        (
            chosen_from_image_start_x_to_be_cropped,
            chosen_from_image_start_y_to_be_cropped,
            chosen_from_image_start_x_to_be_cropped + drawing_rectangle_size[0],
            chosen_from_image_start_y_to_be_cropped + drawing_rectangle_size[1],
        )
    )

    # crop and paste on object image
    on_image_pillow.paste(
        from_image_pillow_copy.crop(from_image_rectangle_to_be_cropped),
        box=drawing_rectangle_point,
    )


def get_image_iou(ground_truth_box, prediction_box) -> float:
    g_xmin, g_ymin, g_xmax, g_ymax = tuple(ground_truth_box)
    d_xmin, d_ymin, d_xmax, d_ymax = tuple(prediction_box)

    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)
    return intersection / float(boxAArea + boxBArea - intersection)


def get_converted_dict_from_extension_pbtxt(
    pbtxt: Union[Path, str], is_displayed_name=False, orders_in_id_and_name=False
) -> dict:
    """
    Reference
        https://stackoverflow.com/questions/11026959/writing-a-dict-to-txt-file-and-reading-it-back

    Args:
        pbtxt_path (str): [description]
        is_displayed_name (bool, optional): if True, save display_name instead of item_name.
        is_reverse (bool, optional): if True, items[item_name] = item_id. else, items[item_id] = item_name.

    Returns:
        dict: dict

    e.g.
        print(
            get_converted_dict_from_extension_pbtxt(
                "/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/my_label_map.pbtxt",
            )
        )
    """
    pbtxt = Path(pbtxt)
    item_id = None
    item_name = None
    items: dict = {}
    if is_displayed_name:
        name = " display_name:"
    else:
        name = " name:"

    with open(f"{pbtxt}", "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif name in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()
            if item_id is not None and item_name is not None:
                if orders_in_id_and_name:
                    items[item_id] = item_name
                else:
                    items[item_name] = item_id
                item_id = None
                item_name = None

    return items


def normalize_using_zscore(
    df: pandas.DataFrame, columns: pandas.Index, common_threshold: float = 3.0
) -> pandas.DataFrame:
    """
    + for example,
        - A standard score of 1.0 (= 60 deviation) or more is 15.866% *2 of the total.
        - A standard score of 2.0 (= 70 deviation) or higher is 2.275% *2 of the total.
        - A standard score of 3.0 (= 80 deviation) or higher is 0.13499% *2 of the total.
        - A standard score of 4.0 (= 90 deviation) or higher is 0.00315% *2 of the total.
        - A standard score of 5.0 (= 100 deviation) or higher is 0.00002% *2 of the total.

    Args:
        df (pandas.DataFrame): dataframe to be normalized
        columns (pandas.Index): colmun list
        common_threshold (float, optional): If the data is too small, 2.0 may be appropriate.
            Defaults to 3.0.

    Returns:
        pandas.DataFrame: normalized dataframe
    """
    # for compatibility version:
    df_zscore: pandas.DataFrame = stats.zscore(df)
    bool_list: list = []
    for index, row in df_zscore.iterrows():
        is_valid: bool = True
        for column in columns:
            if abs(row[column]) > common_threshold:
                is_valid = False
                break
        bool_list.append(is_valid)
    return df.loc[bool_list]


def get_features(correlations: pandas.DataFrame, threshold: float = 0.01) -> list:
    """get feature if correlations is euqal and greater than threshold value.

    Args:
        correlations (pandas.DataFrame): dataframe from pandas.DataFrame.corr()
        threshold (float, optional): Defaults to 0.01.

    Returns:
        list: [description]
    """
    absoulte_correlation: pandas.DataFrame = correlations.abs()
    high_correlations: pandas.DataFrame = absoulte_correlation.loc[
        absoulte_correlation > threshold
    ].index.values.tolist()
    return high_correlations


def get_coefficient_from_linear(linear_estimator: tf.estimator.LinearEstimator) -> dict:
    """get trained coefficient data.

    Args:
        linear_estimator (tf.estimator.LinearEstimator): trained linear estimator

    Returns:
        dict: pairs of attribute and value

    deprecated:
        this is used in only deprecated package "estimator"
    """
    trained_coefficient: dict = {}
    pt_attribute: re.Pattern = re.compile("(^linear/linear_model/)(.+)(/weights$)")
    for key, value in {
        key: linear_estimator.get_variable_value(key)
        for key in linear_estimator.get_variable_names()
    }.items():
        match: re.Match = pt_attribute.match(key)
        if match:
            # assert type(value)==numpy.ndarray
            trained_coefficient[match.group(2)] = value.tolist()[0]
    return trained_coefficient


def test_change_gamma_from_tf_and_save(image: Union[Path, str]) -> None:
    print("===== start function test_change_gamma_from_tf_and_save()")
    image: Path = Path(image)
    image_cv = cv2.imread(f"{image}")
    for i in range(1, 20):
        gamma: float = round(i * 0.1, 2)
        new_image: Path = image.parent / f"{image.stem}-gamma_{gamma}{image.suffix}"
        new_image_ts = keras.preprocessing.image.img_to_array(
            tf.image.adjust_gamma(image_cv, gamma)
        )
        cv2.imwrite(f"{new_image}", new_image_ts)
    print("===== end function test_change_gamma_from_tf_and_save()")


def change_gamma_from_tf_and_save(image: Union[Path, str], gamma: float) -> None:
    print("===== start function change_gamma_from_tf_and_save()")
    image: Path = Path(image)
    image_cv = cv2.imread(f"{image}")
    new_image: Path = image.parent / f"{image.stem}-gamma_{gamma}{image.suffix}"
    new_image_ts = keras.preprocessing.image.img_to_array(
        tf.image.adjust_gamma(image_cv, gamma)
    )
    cv2.imwrite(f"{new_image}", new_image_ts)
    print("===== end function change_gamma_from_tf_and_save()")


# Following functions require some upper functions ~


def superimpose_images_from_pascal_voc_xml(
    input_directory: Union[Path, str],
    background_image_directory: Union[Path, str],
    extensions_of_background_images: list[str],
    output_directory: Union[Path, str],
    should_copy_original_files_to_output: bool = False,
    drawing_shape: str = "rectangle",
    gaussian_blur: int = 15,
    iteration: Optional[int] = None,
    random_seed: int = 42,
    input_split_object_class_stem_list: list[str] = [],
    input_split_object_class_stem_suffix: str = "",
    output_object_class_stem_list_to_be_split: list[str] = [],
    output_object_class_stem_suffix: str = "",
) -> None:
    """
    + About iteration
        - It creates images to reduce the sensitivity of objects in the existing image to the surrounding background, instead of creating an image to mainly learn objects in the image.
        - so in this function background image is important than original image, base of iteration is the number of background image.
        - but in order to keep balance of the number of each class on original image for classification, you can input to iteration argument as None (default). in this case, iteration is stopped when it operates as many as the number of original image.

        - while it keeps annotation that has xml locations, it iterates the operation that superimposes a original image on a background image.
        - if a background image is different from a original background image, it automatically resizes in same size of original images only in this function.

    + About input and output directory
        - if input_split_object_class_stem_list and output_object_class_stem_list_to_be_split are not empty, output_object_class_stem_list_to_be_split must be subset of input_split_object_class_stem_list.


    e.g.
        superimpose_images_from_pascal_voc_xml(
            input_directory=Path("/mnt/c/Users/wbfw109/image_test"),
            background_image_directory=Path("/mnt/c/Users/wbfw109/image_background"),
            extensions_of_background_images=[
                "jpg",
            ],
            output_directory=Path("/mnt/c/Users/wbfw109/image_temp1"),
            should_copy_original_files_to_output=True,
            output_object_class_stem_list_to_be_split=sorted(
                list(
                    get_converted_dict_from_extension_pbtxt(
                        Path(
                            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/my_label_map.pbtxt"
                        ),
                    ).keys()
                )[:-1]
            )
        )

    Args:
        input_directory (Union[Path, str]): directory that have image and xml files
        background_image_directory (Union[Path, str]): directory that have image files
        output_directory (Union[Path, str]): [description]
        should_copy_original_files_to_output (bool, optional): [description]
            Defaults to False.
        extensions_of_background_images (list[str]): [description]
        random_seed (int, optional): [description]. Defaults to 42.
        drawing_shape: (str, optional): only select in list ["ellipse", "rectangle"].
            Defaults to "rectangle".
        gaussian_blur (int, optional): [description].
            Defaults to 15.
        iteration (int, optional): refer to the "About iteration".
            if it is none, it iterates until count is equal the number of original image.
            Defaults to None.
        random_seed (int, optional): [description].
            Defaults to 42.
        input_split_object_class_stem_list (list[str], optional): if it is not empty, it will read from each true_input_directory by object_class.
            true_input_directory: Path = input_directory / f"{object_class}{input_split_object_class_stem_suffix}"
            Defaults to [].
        input_split_object_class_stem_suffix (str, optional): if input_split_object_class_stem_list is set, it will be used.
            Defaults to "".
        output_object_class_stem_list_to_be_split (list[str], optional): if it is not empty, it will output on each true_output_directory by object_class.
            true_output_directory: Path = output_directory / f"{object_class}{output_object_class_stem_suffix}".
            Defaults to [].
        output_object_class_stem_suffix (str, optional): if output_object_class_stem_list_to_be_split is set, it will be used.
            Defaults to "".
    """

    def superimpose_images(
        generator_of_refined_xml_df: Iterable,
        output_directory: Path,
        generator_of_background_image: Iterable[Path],
        threshold_count: int,
    ):
        count: int = 0
        for background_image in generator_of_background_image:
            xml_series: pandas.Series = next(generator_of_refined_xml_df)[1]
            xml_file: Path = Path(xml_series.loc["filename"])
            xml_file = xml_file.parent / f"{xml_file.stem}.xml"

            background_image_pillow: Image.Image = Image.open(f"{background_image}")
            original_image_pillow: Image.Image = Image.open(
                f"{xml_series.loc['filename']}"
            )
            if background_image_pillow.size != original_image_pillow.size:
                background_image_pillow = background_image_pillow.resize(
                    original_image_pillow.size
                )

            # ... When translating a color image to greyscale (mode “L”),
            mask = Image.new("L", original_image_pillow.size, 0)
            draw = ImageDraw.Draw(mask)
            if drawing_shape == "rectangle":
                for i in range(len(xml_series.loc["class"])):
                    draw.rectangle(
                        (
                            xml_series.loc["xmin"][i],
                            xml_series.loc["ymin"][i],
                            xml_series.loc["xmax"][i],
                            xml_series.loc["ymax"][i],
                        ),
                        fill=255,
                    )
            elif drawing_shape == "ellipse":
                for i in range(len(xml_series.loc["class"])):
                    draw.ellipse(
                        (
                            xml_series.loc["xmin"][i],
                            xml_series.loc["ymin"][i],
                            xml_series.loc["xmax"][i],
                            xml_series.loc["ymax"][i],
                        ),
                        fill=255,
                    )

            mask_blur = mask.filter(ImageFilter.GaussianBlur(gaussian_blur))
            new_image_pillow = Image.composite(
                original_image_pillow, background_image_pillow, mask_blur
            )
            # # test
            # mask.show()
            # im.show()

            new_file_without_extension: Path = (
                wbfw109.libs.utilities.iterable.get_changed_file_without_suffix(
                    input_file=xml_series.loc["filename"],
                    output_directory=output_directory,
                    revision_list=["superimposed"],
                )
            )

            new_image_pillow.save(
                "{new_file_without_extension}{extension}".format(
                    new_file_without_extension=new_file_without_extension,
                    extension=xml_series.loc["filename"].suffix,
                )
            )
            shutil.copy2(src=f"{xml_file}", dst=f"{new_file_without_extension}.xml")

            # additional condition
            if threshold_count > 0 and ((count := count + 1) >= threshold_count):
                break

    print("===== start function superimpose_images_from_pascal_voc_xml()")

    # * preprocess
    input_directory: Path = Path(input_directory)
    background_image_directory: Path = Path(background_image_directory)
    output_directory: Path = Path(output_directory)

    output_directory.mkdir(exist_ok=True)

    assert output_directory.is_dir()
    assert input_directory.is_dir()
    assert background_image_directory.is_dir()
    assert drawing_shape in ["ellipse", "rectangle"]

    sliced_refined_xml_df_dict: dict[str, pandas.DataFrame] = {}
    if input_split_object_class_stem_list:
        for object_class in input_split_object_class_stem_list:
            true_input_directory: Path = (
                input_directory
                / f"{object_class}{input_split_object_class_stem_suffix}"
            )
            assert true_input_directory.is_dir()

            sliced_refined_xml_df_dict[object_class] = refine_pascal_voc_xml_df_and_get(
                xml_df=convert_pascal_voc_xml_to_df_and_get(
                    xml_directory=true_input_directory,
                )
            )

            sliced_refined_xml_df_dict[object_class].loc[:, "filename"] = (
                sliced_refined_xml_df_dict[object_class]
                .loc[:, "filename"]
                .apply(lambda filename: true_input_directory / filename)
            )

        if not output_object_class_stem_list_to_be_split:
            refined_xml_df: pandas.DataFrame = pandas.concat(
                sliced_refined_xml_df_dict.values()
            )

    else:
        refined_xml_df: pandas.DataFrame = refine_pascal_voc_xml_df_and_get(
            xml_df=convert_pascal_voc_xml_to_df_and_get(
                xml_directory=input_directory,
            )
        )

        if output_object_class_stem_list_to_be_split:
            for object_class in output_object_class_stem_list_to_be_split:
                sliced_refined_xml_df_dict[object_class] = refined_xml_df.loc[
                    refined_xml_df.loc[:, "filename"].str.contains(
                        f"_{object_class}_", na=False
                    )
                ].copy()

                sliced_refined_xml_df_dict[object_class].loc[:, "filename"] = (
                    sliced_refined_xml_df_dict[object_class]
                    .loc[:, "filename"]
                    .apply(lambda filename: input_directory / filename)
                )
        else:
            refined_xml_df.loc[:, "filename"] = refined_xml_df.loc[:, "filename"].apply(
                lambda filename: input_directory / filename
            )

    # * process
    if output_object_class_stem_list_to_be_split:
        for object_class in output_object_class_stem_list_to_be_split:
            true_output_directory: Path = (
                output_directory / f"{object_class}{output_object_class_stem_suffix}"
            )
            true_output_directory.mkdir(exist_ok=True)

            assert true_output_directory.is_dir()

            if iteration is None:
                # set min enough iteration
                true_iteration = sliced_refined_xml_df_dict[object_class].index.size
                threshold_count: int = true_iteration
            else:
                true_iteration = iteration
                threshold_count: int = 0
                assert isinstance(iteration, int)

            superimpose_images(
                generator_of_refined_xml_df=itertools.cycle(
                    sliced_refined_xml_df_dict[object_class].iterrows()
                ),
                output_directory=true_output_directory,
                generator_of_background_image=wbfw109.libs.utilities.iterable.generator_with_shuffle(
                    wbfw109.libs.utilities.iterable.get_file_list(
                        background_image_directory,
                        extensions=extensions_of_background_images,
                    ),
                    iteration=true_iteration,
                    random_seed=random_seed,
                ),
                threshold_count=threshold_count,
            )

            # * postprocess
            change_xml_filename_as_true_reference_name(true_output_directory)
            if should_copy_original_files_to_output:
                for image in (
                    sliced_refined_xml_df_dict[object_class].loc[:, "filename"].tolist()
                ):
                    shutil.copy2(
                        src=f"{image}", dst=f"{true_output_directory}/{image.name}"
                    )
                    shutil.copy2(
                        src=f"{image.parent}/{image.stem}.xml",
                        dst=f"{true_output_directory}/{image.stem}.xml",
                    )

    else:
        if iteration is None:
            # set min enough iteration
            true_iteration = refined_xml_df.index.size
            threshold_count: int = true_iteration
        else:
            true_iteration = iteration
            threshold_count: int = 0
            assert isinstance(iteration, int)

        superimpose_images(
            generator_of_refined_xml_df=itertools.cycle(refined_xml_df.iterrows()),
            output_directory=output_directory,
            generator_of_background_image=wbfw109.libs.utilities.iterable.generator_with_shuffle(
                wbfw109.libs.utilities.iterable.get_file_list(
                    background_image_directory,
                    extensions=extensions_of_background_images,
                ),
                iteration=true_iteration,
                random_seed=random_seed,
            ),
            threshold_count=threshold_count,
        )

        # * postprocess
        change_xml_filename_as_true_reference_name(output_directory)
        if should_copy_original_files_to_output:
            for image in refined_xml_df.loc[:, "filename"].tolist():
                shutil.copy2(src=f"{image}", dst=f"{output_directory}/{image.name}")
                shutil.copy2(
                    src=f"{image.parent}/{image.stem}.xml",
                    dst=f"{output_directory}/{image.stem}.xml",
                )

    print("===== end function superimpose_images_from_pascal_voc_xml()")


def augment_images_with_xml_from_pascal_voc_xml(
    input_directory: Union[Path, str],
    output_directory: Union[Path, str],
    augmentation_case_list: list[iaa_meta.Augmenter],
    iteration: int = 1,
    input_split_object_class_stem_list: list[str] = [],
    input_split_object_class_stem_suffix: str = "",
    output_object_class_stem_list_to_be_split: list[str] = [],
    output_object_class_stem_suffix: str = "",
) -> None:
    """
    This function not includes random_seed for iteration because iaa_meta.Augmenter includes probability setting.

    + Note
        - iaa.Sequential() can be nested. so, you can add augmentation as Sequential of augmentation or a augmentation.
            e.g.
                augmentation_test = iaa.Sequential([iaa.Sequential([augmentation], random_order=False)], random_order=False)
                augmented_image: numpy.ndarray = augmentation_test.augment_image(
                    image=original_image_cv2
                )

    + Known problem
        - Make the bounding box more tighten for the object after image rotation
            https://github.com/aleju/imgaug/issues/90
            ... This is a known problem. The bounding box augmentation is based on augmenting the corner points of the bounding boxes, then drawing a new bounding box around these corners. That can lead to badly fitting bounding boxes, especially after rotations.
            >> ... If the object occupied the corners of the original bounding box, your new bounding box needs to be bigger after the image rotates. So you must be careful of not doing too higher rotations with bounding boxes because there is not enough information for them to stay accurate. So here, we do maximum of 3 degree rotation to avoid this problem https://www.youtube.com/watch?v=0frKXR-2PBY&feature=youtu.be&t=9m14s



    + About input and output directory
        - if input_split_object_class_stem_list and output_object_class_stem_list_to_be_split are not empty, output_object_class_stem_list_to_be_split must be subset of input_split_object_class_stem_list.


    e.g.
        augment_images_with_xml_from_pascal_voc_xml(
            input_directory=Path("/mnt/c/Users/wbfw109/images_temp_2"),
            output_directory=Path("/mnt/c/Users/wbfw109/images_temp_3"),
            augmentation_case_list=augmentation_case_list,
            input_split_object_class_stem_list=["tumbler", "takeout_ice", "mug"],
            input_split_object_class_stem_suffix="_temp",
            output_object_class_stem_list_to_be_split=["tumbler", "mug"],
        )

        # and
        refer to the function:
            - _augment_with_common_augmentation()
            - _augment_padding_from_CVAT_example1()

    Args:
        input_directory (Union[Path, str]): [description]
        output_directory (Union[Path, str]): [description]
        augmentation_case_list (list[iaa_meta.Augmenter]): [description]
        iteration (int, optional): base is the number of original image.
            Defaults to 1.
        input_split_object_class_stem_list (list[str], optional): if it is not empty, it will read from each true_input_directory by object_class.
            true_input_directory: Path = input_directory / f"{object_class}{input_split_object_class_stem_suffix}"
            Defaults to [].
        input_split_object_class_stem_suffix (str, optional): if input_split_object_class_stem_list is set, it will be used.
            Defaults to "".
        output_object_class_stem_list_to_be_split (list[str], optional): if it is not empty, it will output on each true_output_directory by object_class.
            true_output_directory: Path = output_directory / f"{object_class}{output_object_class_stem_suffix}".
            Defaults to [].
        output_object_class_stem_suffix (str, optional): if output_object_class_stem_list_to_be_split is set, it will be used.
            Defaults to "".
    """

    def augment_images_with_xml(
        refined_xml_df: pandas.DataFrame,
        output_directory: Path,
        iteration: int,
    ) -> None:
        for i in range(iteration):
            for j, xml_series in refined_xml_df.iterrows():
                original_image_cv2: numpy.ndarray = cv2.imread(
                    str(xml_series.loc["filename"])
                )

                # * start create augmentation images from input iamges
                bounding_box_list: list[BoundingBox] = [
                    BoundingBox(
                        x1=xml_series.loc["xmin"][j],
                        y1=xml_series.loc["ymin"][j],
                        x2=xml_series.loc["xmax"][j],
                        y2=xml_series.loc["ymax"][j],
                        label=xml_series.loc["class"][j],
                    )
                    for j in range(len(xml_series.loc["class"]))
                ]

                bounding_boxes_on_images = BoundingBoxesOnImage(
                    bounding_box_list, original_image_cv2.shape
                )

                for augmentation in augmentation_case_list:
                    augmentation: iaa_meta.Augmenter = augmentation.to_deterministic()

                    augmented_image: numpy.ndarray = augmentation.augment_image(
                        image=original_image_cv2
                    )

                    augmented_annotation: BoundingBoxesOnImage = (
                        augmentation.augment_bounding_boxes(
                            bounding_boxes_on_images=bounding_boxes_on_images
                        )
                    )
                    new_file_without_extension: Path = (
                        wbfw109.libs.utilities.iterable.get_changed_file_without_suffix(
                            input_file=xml_series.loc["filename"],
                            output_directory=output_directory,
                            revision_list=[
                                children.__class__.__name__.lower()
                                for children in augmentation
                            ],
                        )
                    )

                    new_image_path: Path = Path(
                        "{new_file_without_extension}{extension}".format(
                            new_file_without_extension=new_file_without_extension,
                            extension=xml_series.loc["filename"].suffix,
                        )
                    )
                    cv2.imwrite(f"{new_image_path}", augmented_image)

                    # # test
                    # image_before = bounding_boxes_on_images.draw_on_image(original_image_cv2, size=2)
                    # image_after = annotations_aug.draw_on_image(images_aug, size=2)

                    # order of parameters is (path, width, height).
                    pascal_voc_xml_writer = pascal_voc_writer.Writer(
                        f"{new_image_path}", *list(reversed(augmented_image.shape[:-1]))
                    )
                    for new_bounding_box in augmented_annotation.items:
                        pascal_voc_xml_writer.addObject(
                            name=new_bounding_box.label,
                            xmin=new_bounding_box.x1_int,
                            ymin=new_bounding_box.y1_int,
                            xmax=new_bounding_box.x2_int,
                            ymax=new_bounding_box.y2_int,
                        )

                    pascal_voc_xml_writer.save(f"{new_file_without_extension}.xml")

    print("===== start function augment_images_with_xml_from_pascal_voc_xml()")

    # * preprocess
    input_directory: Path = Path(input_directory)
    output_directory: Path = Path(output_directory)
    output_directory.mkdir(exist_ok=True)

    assert output_directory.is_dir()

    sliced_refined_xml_df_dict: dict[str, pandas.DataFrame] = {}
    if input_split_object_class_stem_list:
        for object_class in input_split_object_class_stem_list:
            true_input_directory: Path = (
                input_directory
                / f"{object_class}{input_split_object_class_stem_suffix}"
            )
            assert true_input_directory.is_dir()

            sliced_refined_xml_df_dict[object_class] = refine_pascal_voc_xml_df_and_get(
                xml_df=convert_pascal_voc_xml_to_df_and_get(
                    xml_directory=true_input_directory,
                )
            )

            sliced_refined_xml_df_dict[object_class].loc[:, "filename"] = (
                sliced_refined_xml_df_dict[object_class]
                .loc[:, "filename"]
                .apply(lambda filename: true_input_directory / filename)
            )

        if not output_object_class_stem_list_to_be_split:
            refined_xml_df: pandas.DataFrame = pandas.concat(
                sliced_refined_xml_df_dict.values()
            )

    else:
        refined_xml_df: pandas.DataFrame = refine_pascal_voc_xml_df_and_get(
            xml_df=convert_pascal_voc_xml_to_df_and_get(
                xml_directory=input_directory,
            )
        )

        if output_object_class_stem_list_to_be_split:
            for object_class in output_object_class_stem_list_to_be_split:
                sliced_refined_xml_df_dict[object_class] = refined_xml_df.loc[
                    refined_xml_df.loc[:, "filename"].str.contains(
                        f"_{object_class}_", na=False
                    )
                ].copy()

                sliced_refined_xml_df_dict[object_class].loc[:, "filename"] = (
                    sliced_refined_xml_df_dict[object_class]
                    .loc[:, "filename"]
                    .apply(lambda filename: input_directory / filename)
                )
        else:
            refined_xml_df.loc[:, "filename"] = refined_xml_df.loc[:, "filename"].apply(
                lambda filename: input_directory / filename
            )

    # * process
    if output_object_class_stem_list_to_be_split:
        for object_class in output_object_class_stem_list_to_be_split:
            true_output_directory: Path = (
                output_directory / f"{object_class}{output_object_class_stem_suffix}"
            )
            true_output_directory.mkdir(exist_ok=True)
            assert true_output_directory.is_dir()

            # + postprocess
            augment_images_with_xml(
                refined_xml_df=sliced_refined_xml_df_dict[object_class],
                output_directory=true_output_directory,
                iteration=iteration,
            )
    else:
        # + postprocess
        augment_images_with_xml(
            refined_xml_df=refined_xml_df,
            output_directory=output_directory,
            iteration=iteration,
        )

    print("===== end function augment_images_with_xml_from_pascal_voc_xml()")


def noise_on_bounding_boxes_by_filling_rectangle_from_pascal_voc_xml(
    input_directory: Union[Path, str],
    background_image_directory: Union[Path, str],
    extensions_of_background_images: list[str],
    output_directory: Union[Path, str],
    class_precedence: dict[int, list[str]] = {},
    class_filter_list: list[str] = [],
    will_use_class_filter_mode_as_whitelist: bool = False,
    drawing_shape: list[str] = ["horizontal", "vertical"],
    drawing_xy_edge_offset_rate: tuple[float] = (0.15, 0.15),
    drawing_thickness_random_range: tuple[float] = (0.05, 0.15),
    iteration: int = 1,
    random_seed: int = 42,
    input_split_object_class_stem_list: list[str] = [],
    input_split_object_class_stem_suffix: str = "",
    output_object_class_stem_list_to_be_split: list[str] = [],
    output_object_class_stem_suffix: str = "",
) -> None:
    """
    it parses xml and noises horizontally or vertically filling rectangle.

    about all argument with "_rate", these are operated based on image.size

    + About class_precedence
        - objects with higher values will be pasted more in front.

        - if class_precedence value is same, automatically objects with lower area will be pasted more in front.
            the reason I wrote code like this is because small objects are more difficult to catch than large objects in normal object detection.

        - if object class is not in class_precedence, automatically other objects will have minimum and same class_precedence.

    + About drawing rectangles
        - if it can not draw rectangle because of relation between drawing thickness and offsets, it will be passed printing the cause.

    + About input and output directory
        - if input_split_object_class_stem_list and output_object_class_stem_list_to_be_split are not empty, output_object_class_stem_list_to_be_split must be subset of input_split_object_class_stem_list.


    e.g.
        noise_on_bounding_boxes_by_filling_rectangle_from_pascal_voc_xml(
            input_directory=Path("/mnt/c/Users/wbfw109/image_temp1"),
            background_image_directory=Path("/mnt/c/Users/wbfw109/image_test"),
            extensions_of_background_images=[
                "jpg",
            ],
            output_directory=Path("/mnt/c/Users/wbfw109/image_temp2"),
            class_precedence={
                3: sorted(
                    list(
                        get_converted_dict_from_extension_pbtxt(
                            Path(
                                "/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/my_label_map.pbtxt"
                            ),
                        ).keys()
                    )[:-1]
                )
            },
            input_split_object_class_stem_list=sorted(
                list(
                    get_converted_dict_from_extension_pbtxt(
                        Path(
                            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/my_label_map.pbtxt"
                        ),
                    ).keys()
                )[:-1]
            ),
            input_split_object_class_stem_suffix="_temp",
            output_object_class_stem_list_to_be_split=sorted(
                list(
                    get_converted_dict_from_extension_pbtxt(
                        Path(
                            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/my_label_map.pbtxt"
                        ),
                    ).keys()
                )[:-1]
            ),
        )

    Args:
        input_directory (Union[Path, str]): directory that have image and xml files
        background_image_directory (Union[Path, str]): directory that have image files
        output_directory (Union[Path, str]): [description]
        extensions_of_background_images (list[str]): [description]
        class_precedence (dict[int, list[str]], optional): refer to the "About class_precedence".
            Defaults to {}.
        class_filter_list (list[str], optional): if you want to not noise for these class set this.
            Defaults to [].
        will_use_class_filter_mode_as_whitelist (bool, optional): default class_filter_list is as blacklist.
            if you want to whitelist, set this as true.
            Defaults to False.
        drawing_shape (list[str], optional): [description].
            Defaults to ["horizontal", "vertical"].
        drawing_xy_edge_offset_rate (tuple[float], optional): if you want to not hide edge, set this.
            Defaults to (0.15, 0.15).
        drawing_thickness_random_range (tuple[float], optional): refer to the "About drawing rectangles".
            Defaults to (0.05, 0.15).
        iteration (int, optional): base is the number of original image.
            Defaults to 1.
        random_seed (int, optional): [description].
            Defaults to 42.
        input_split_object_class_stem_list (list[str], optional): if it is not empty, it will read from each true_input_directory by object_class.
            true_input_directory: Path = input_directory / f"{object_class}{input_split_object_class_stem_suffix}"
            Defaults to [].
        input_split_object_class_stem_suffix (str, optional): if input_split_object_class_stem_list is set, it will be used.
            Defaults to "".
        output_object_class_stem_list_to_be_split (list[str], optional): if it is not empty, it will output on each true_output_directory by object_class.
            true_output_directory: Path = output_directory / f"{object_class}{output_object_class_stem_suffix}".
            Defaults to [].
        output_object_class_stem_suffix (str, optional): if output_object_class_stem_list_to_be_split is set, it will be used.
            Defaults to "".
    """

    @dataclasses.dataclass
    class ObjectInformation:
        image_pillow: Image.Image
        pandas_series_index: int

    def get_class_precedence_reversed_kv(
        refined_xml_df: pandas.DataFrame, class_precedence: dict[int, list[str]]
    ) -> dict[str, int]:
        class_precedence_reversed_kv: dict[str, int] = {}
        object_class_set_in_xml: set = set(
            itertools.chain.from_iterable(refined_xml_df.loc[:, "class"].tolist())
        )
        other_object_class_precedence: int = 1
        if class_precedence:
            other_object_class_precedence = (
                min(class_precedence.keys()) - other_object_class_precedence
            )
            for key, object_class_list in class_precedence.items():
                for object_class in object_class_list:
                    class_precedence_reversed_kv[object_class] = key

        for class_to_be_added in list(
            object_class_set_in_xml - set(class_precedence_reversed_kv.keys())
        ):
            class_precedence_reversed_kv[
                class_to_be_added
            ] = other_object_class_precedence

        return class_precedence_reversed_kv

    def noise_on_bounding_boxes_by_filling_rectangle(
        refined_xml_df: pandas.DataFrame,
        output_directory: Path,
        generator_of_background_image: Iterable[Path],
        iteration: int,
        random_seed: int,
        class_precedence_reversed_kv: dict[str, int],
        class_filter_list: list[str],
        will_use_class_filter_mode_as_whitelist: bool,
        drawing_shape: list[str],
        drawing_xy_edge_offset_rate: tuple[float],
        drawing_thickness_random_range: tuple[float],
    ) -> None:
        for i in range(iteration):
            # shuffle
            refined_xml_df = refined_xml_df.sample(
                frac=1, random_state=random_seed
            ).reset_index(drop=True)

            for j, xml_series in refined_xml_df.iterrows():
                xml_file: Path = Path(xml_series.loc["filename"])
                xml_file = xml_file.parent / f"{xml_file.stem}.xml"

                background_image_pillow: Image.Image = Image.open(
                    f"{next(generator_of_background_image)}"
                )
                original_image_pillow: Image.Image = Image.open(
                    f"{xml_series.loc['filename']}"
                )

                object_information_list: list[ObjectInformation] = [
                    ObjectInformation(
                        image_pillow=original_image_pillow.crop(
                            (
                                xml_series.loc["xmin"][ii],
                                xml_series.loc["ymin"][ii],
                                xml_series.loc["xmax"][ii],
                                xml_series.loc["ymax"][ii],
                            )
                        ),
                        pandas_series_index=ii,
                    )
                    for ii in range(len(xml_series.loc["class"]))
                ]

                # set sequence to paste cropped image
                object_information_list = sorted(
                    object_information_list,
                    key=lambda object_information: (
                        class_precedence_reversed_kv[
                            xml_series.loc["class"][
                                object_information.pandas_series_index
                            ]
                        ],
                        -xml_series.loc["area"][object_information.pandas_series_index],
                    ),
                )

                # # test
                # for x in object_information_list:
                #     print(xml_series.loc["class"][x[1]])
                # print()

                # noise to each object
                for object_information in object_information_list:
                    if not will_use_class_filter_mode_as_whitelist:
                        if (
                            xml_series.loc["class"][
                                object_information.pandas_series_index
                            ]
                            in class_filter_list
                        ):
                            continue
                    else:
                        if (
                            xml_series.loc["class"][
                                object_information.pandas_series_index
                            ]
                            not in class_filter_list
                        ):
                            continue

                    object_size: numpy.ndarray = numpy.array(
                        object_information.image_pillow.size
                    )

                    # r"//" is the floor division operator
                    drawing_xy_edge_offset_rate: numpy.ndarray = numpy.array(
                        drawing_xy_edge_offset_rate
                    )
                    drawing_thickness_random_range: numpy.ndarray = numpy.array(
                        drawing_thickness_random_range
                    )

                    offset_xy_start: numpy.ndarray = (
                        object_size * drawing_xy_edge_offset_rate / 2
                    ).astype(int)
                    offset_xy_end: numpy.ndarray = object_size - offset_xy_start

                    avilable_x_min_max_thickness: numpy.ndarray = (
                        object_size[0] * drawing_thickness_random_range
                    ).astype(int)

                    avilable_y_min_max_thickness: numpy.ndarray = (
                        object_size[1] * drawing_thickness_random_range
                    ).astype(int)

                    if "horizontal" in drawing_shape:
                        remaining_max_thickness: int = (
                            offset_xy_end[1] - offset_xy_start[1]
                        )
                        # if remaining max space is greater than avilable drawing min thickness
                        if remaining_max_thickness > avilable_y_min_max_thickness[0]:
                            # if remaining max space is less than avilable drawing max thickness
                            if (
                                remaining_max_thickness
                                < avilable_y_min_max_thickness[1]
                            ):
                                avilable_y_min_max_thickness[
                                    1
                                ] = remaining_max_thickness
                            chosen_thickness_y: int = random.randint(
                                avilable_y_min_max_thickness[0],
                                avilable_y_min_max_thickness[1],
                            )

                            drawing_start_y: int = random.randint(
                                offset_xy_start[1],
                                offset_xy_end[1] - chosen_thickness_y,
                            )
                            # xmin, ymin, xmax, ymax
                            drawing_rectangle_point: numpy.ndarray = numpy.array(
                                (
                                    0,
                                    drawing_start_y,
                                    object_size[0],
                                    drawing_start_y + chosen_thickness_y,
                                )
                            )

                            draw_rectangle_on_image_from_random_part_of_image(
                                on_image_pillow=object_information.image_pillow,
                                from_image_pillow=background_image_pillow,
                                drawing_rectangle_point=drawing_rectangle_point,
                            )
                        else:
                            print(
                                "[pass] it can not horizontally noise on {object_class} in {filename} because offset_y is greater than thickness size.".format(
                                    filename=xml_series.loc["filename"],
                                    object_class=xml_series.loc["class"][
                                        object_information.pandas_series_index
                                    ],
                                )
                            )
                    if "vertical" in drawing_shape:
                        remaining_max_thickness: int = (
                            offset_xy_end[0] - offset_xy_start[0]
                        )
                        # if remaining max space is greater than avilable drawing min thickness
                        if remaining_max_thickness > avilable_x_min_max_thickness[0]:
                            # if remaining max space is less than avilable drawing max thickness
                            if (
                                remaining_max_thickness
                                < avilable_x_min_max_thickness[1]
                            ):
                                avilable_x_min_max_thickness[
                                    1
                                ] = remaining_max_thickness
                            chosen_thickness_x: int = random.randint(
                                avilable_x_min_max_thickness[0],
                                avilable_x_min_max_thickness[1],
                            )

                            drawing_start_x: int = random.randint(
                                offset_xy_start[0],
                                offset_xy_end[0] - chosen_thickness_x,
                            )
                            # xmin, ymin, xmax, ymax
                            drawing_rectangle_point: numpy.ndarray = numpy.array(
                                (
                                    drawing_start_x,
                                    0,
                                    drawing_start_x + chosen_thickness_x,
                                    object_size[1],
                                )
                            )
                            draw_rectangle_on_image_from_random_part_of_image(
                                on_image_pillow=object_information.image_pillow,
                                from_image_pillow=background_image_pillow,
                                drawing_rectangle_point=drawing_rectangle_point,
                            )
                        else:
                            print(
                                "[pass] it can not vertically noise on {object_class} in {filename} because offset_x is greater than thickness size.".format(
                                    filename=xml_series.loc["filename"],
                                    object_class=xml_series.loc["class"][
                                        object_information_list[i].pandas_series_index
                                    ],
                                )
                            )
                    # # test
                    # object_information.image_pillow.show()

                for object_information in object_information_list:
                    original_image_pillow.paste(
                        object_information.image_pillow,
                        (
                            xml_series.loc["xmin"][
                                object_information.pandas_series_index
                            ],
                            xml_series.loc["ymin"][
                                object_information.pandas_series_index
                            ],
                            xml_series.loc["xmax"][
                                object_information.pandas_series_index
                            ],
                            xml_series.loc["ymax"][
                                object_information.pandas_series_index
                            ],
                        ),
                    )

                # # test
                # print(xml_file)
                # original_image_pillow.show()

                new_file_without_extension: Path = (
                    wbfw109.libs.utilities.iterable.get_changed_file_without_suffix(
                        input_file=xml_series.loc["filename"],
                        output_directory=output_directory,
                        revision_list=["noised"],
                    )
                )

                original_image_pillow.save(
                    "{new_file_without_extension}{extension}".format(
                        new_file_without_extension=new_file_without_extension,
                        extension=xml_series.loc["filename"].suffix,
                    )
                )
                shutil.copy2(src=f"{xml_file}", dst=f"{new_file_without_extension}.xml")

    print(
        "===== start function noise_on_bounding_boxes_by_filling_rectangle_from_pascal_voc_xml()"
    )

    # * preprocess
    input_directory: Path = Path(input_directory)
    background_image_directory: Path = Path(background_image_directory)
    output_directory: Path = Path(output_directory)
    output_directory.mkdir(exist_ok=True)

    assert output_directory.is_dir()
    assert input_directory.is_dir()
    assert background_image_directory.is_dir()
    assert all([x in ["horizontal", "vertical"] for x in drawing_shape])
    assert (
        all([x > 0 and x < 1 for x in drawing_xy_edge_offset_rate])
        and len(drawing_xy_edge_offset_rate) == 2
    )
    assert (
        all([x > 0 and x < 1 for x in drawing_thickness_random_range])
        and len(drawing_thickness_random_range) == 2
        and drawing_thickness_random_range[0] <= drawing_thickness_random_range[1]
    )

    generator_of_background_image: Iterable = itertools.cycle(
        wbfw109.libs.utilities.iterable.get_file_list(
            background_image_directory, extensions=extensions_of_background_images
        )
    )

    sliced_refined_xml_df_dict: dict[str, pandas.DataFrame] = {}
    if input_split_object_class_stem_list:
        for object_class in input_split_object_class_stem_list:
            true_input_directory: Path = (
                input_directory
                / f"{object_class}{input_split_object_class_stem_suffix}"
            )
            assert true_input_directory.is_dir()

            sliced_refined_xml_df_dict[object_class] = refine_pascal_voc_xml_df_and_get(
                xml_df=convert_pascal_voc_xml_to_df_and_get(
                    xml_directory=true_input_directory,
                    optional_column_list=["area"],
                ),
                optional_list_to_be_filtered=["area"],
            )

            sliced_refined_xml_df_dict[object_class].loc[:, "filename"] = (
                sliced_refined_xml_df_dict[object_class]
                .loc[:, "filename"]
                .apply(lambda filename: true_input_directory / filename)
            )

        if not output_object_class_stem_list_to_be_split:
            refined_xml_df: pandas.DataFrame = pandas.concat(
                sliced_refined_xml_df_dict.values()
            )

    else:
        refined_xml_df: pandas.DataFrame = refine_pascal_voc_xml_df_and_get(
            xml_df=convert_pascal_voc_xml_to_df_and_get(
                xml_directory=input_directory,
                optional_column_list=["area"],
            ),
            optional_list_to_be_filtered=["area"],
        )

        if output_object_class_stem_list_to_be_split:
            for object_class in output_object_class_stem_list_to_be_split:
                sliced_refined_xml_df_dict[object_class] = refined_xml_df.loc[
                    refined_xml_df.loc[:, "filename"].str.contains(
                        f"_{object_class}_", na=False
                    )
                ].copy()

                sliced_refined_xml_df_dict[object_class].loc[:, "filename"] = (
                    sliced_refined_xml_df_dict[object_class]
                    .loc[:, "filename"]
                    .apply(lambda filename: input_directory / filename)
                )
        else:
            refined_xml_df.loc[:, "filename"] = refined_xml_df.loc[:, "filename"].apply(
                lambda filename: input_directory / filename
            )

    # * process
    if output_object_class_stem_list_to_be_split:
        for object_class in output_object_class_stem_list_to_be_split:
            true_output_directory: Path = (
                output_directory / f"{object_class}{output_object_class_stem_suffix}"
            )
            true_output_directory.mkdir(exist_ok=True)
            assert true_output_directory.is_dir()

            noise_on_bounding_boxes_by_filling_rectangle(
                refined_xml_df=sliced_refined_xml_df_dict[object_class],
                output_directory=true_output_directory,
                generator_of_background_image=generator_of_background_image,
                class_precedence_reversed_kv=get_class_precedence_reversed_kv(
                    refined_xml_df=sliced_refined_xml_df_dict[object_class],
                    class_precedence=class_precedence,
                ),
                class_filter_list=class_filter_list,
                will_use_class_filter_mode_as_whitelist=will_use_class_filter_mode_as_whitelist,
                drawing_shape=drawing_shape,
                drawing_xy_edge_offset_rate=drawing_xy_edge_offset_rate,
                drawing_thickness_random_range=drawing_thickness_random_range,
                iteration=iteration,
                random_seed=random_seed,
            )

            # * postprocess
            change_xml_filename_as_true_reference_name(true_output_directory)
    else:
        noise_on_bounding_boxes_by_filling_rectangle(
            refined_xml_df=refined_xml_df,
            output_directory=output_directory,
            generator_of_background_image=generator_of_background_image,
            class_precedence_reversed_kv=get_class_precedence_reversed_kv(
                refined_xml_df=refined_xml_df, class_precedence=class_precedence
            ),
            class_filter_list=class_filter_list,
            will_use_class_filter_mode_as_whitelist=will_use_class_filter_mode_as_whitelist,
            drawing_shape=drawing_shape,
            drawing_xy_edge_offset_rate=drawing_xy_edge_offset_rate,
            drawing_thickness_random_range=drawing_thickness_random_range,
            iteration=iteration,
            random_seed=random_seed,
        )

        # * postprocess
        change_xml_filename_as_true_reference_name(output_directory)

    print(
        "===== end function noise_on_bounding_boxes_by_filling_rectangle_from_pascal_voc_xml()"
    )


"""
Following functions are one command for batch operation used in actual work.
    - it is require some upper functions
    - if you want to reuse these functions, must update arguments before use.

"""


def _augment_with_common_augmentation():
    initial_input_directory: Path = Path(
        "/mnt/c/Users/wbfw109/MyDrive/shared_resource/images/train"
    )
    input_and_output_directory: list[Path] = [
        Path("/mnt/c/Users/wbfw109/images_train_temp1"),
        Path("/mnt/c/Users/wbfw109/images_train_temp2"),
        Path("/mnt/c/Users/wbfw109/images_train_true"),
    ]
    current_input_and_output_index: int = 0

    background_image_directory = Path(
        "/mnt/c/Users/wbfw109/images_natural_scene/cps2014.ppm"
    )
    required_object_class_list = sorted(
        list(
            get_converted_dict_from_extension_pbtxt(
                Path(
                    "/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/my_label_map.pbtxt"
                ),
            ).keys()
        )[:-1]
    )

    superimpose_images_from_pascal_voc_xml(
        input_directory=initial_input_directory,
        background_image_directory=background_image_directory,
        extensions_of_background_images=[
            "jpg",
        ],
        output_directory=input_and_output_directory[current_input_and_output_index],
        should_copy_original_files_to_output=True,
        output_object_class_stem_list_to_be_split=required_object_class_list,
    )

    measurement_of_combinations = (
        wbfw109.libs.utilities.iterable.MeasurementOfCombinations(
            original_data_bytes=38936576
        )
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

    # create augmentation_case_list
    augmentation_case_list: list = [
        iaa.Sequential(available_case, random_order=False)
        for available_case in measurement_of_combinations.get_available_case_list()
    ]
    print(f"available augmentation size = {len(augmentation_case_list)}")

    augment_images_with_xml_from_pascal_voc_xml(
        input_directory=input_and_output_directory[current_input_and_output_index],
        output_directory=input_and_output_directory[current_input_and_output_index + 1],
        augmentation_case_list=augmentation_case_list,
        input_split_object_class_stem_list=required_object_class_list,
        output_object_class_stem_list_to_be_split=required_object_class_list,
    )
    current_input_and_output_index += 1

    noise_on_bounding_boxes_by_filling_rectangle_from_pascal_voc_xml(
        input_directory=input_and_output_directory[current_input_and_output_index],
        background_image_directory=background_image_directory,
        extensions_of_background_images=[
            "jpg",
        ],
        output_directory=input_and_output_directory[current_input_and_output_index + 1],
        class_precedence={
            3: required_object_class_list,
        },
        input_split_object_class_stem_list=required_object_class_list,
        output_object_class_stem_list_to_be_split=required_object_class_list,
    )

    # convert xml to tfrecord.
    for directory in list(
        input_and_output_directory[len(input_and_output_directory) - 1].glob("*")
    ):
        convert_xml_to_tfrecord(xml_dir=directory, output_prefix="train_")


def _augment_padding_from_CVAT_example1():
    """
    + Note
        - it expects minimal CVAT directory tree like:

          📂 <parameter: parent_of_exported_directory_from_CVAT>
              ┖─📂 <export root>
                  ┖─📂 Annotations
                      ┖─ 📂 <task name>
                  ┖─📂 JPEGImages
                      ┖─ <task name>
            ...
    """
    parent_of_exported_directory: Path = Path("/mnt/c/users/wbfw109/image_tasks")
    available_annotation_index_range_list_of_folder_list: list[
        RangeOfTwoPointWithRequiredCount
    ] = [
        RangeOfTwoPointWithRequiredCount(
            [RangeOfTwoPoint(start=0, end=945)], required_count=None
        ),
        RangeOfTwoPointWithRequiredCount(
            [RangeOfTwoPoint(start=0, end=51), RangeOfTwoPoint(start=150, end=297)],
            required_count=200,
        ),
        RangeOfTwoPointWithRequiredCount(
            [RangeOfTwoPoint(start=218, end=431)], required_count=200
        ),
        RangeOfTwoPointWithRequiredCount(
            [RangeOfTwoPoint(start=0, end=206), RangeOfTwoPoint(start=973, end=1153)],
            required_count=None,
        ),
        RangeOfTwoPointWithRequiredCount(
            [RangeOfTwoPoint(start=0, end=227)], required_count=200
        ),
        RangeOfTwoPointWithRequiredCount(
            [RangeOfTwoPoint(start=0, end=199)], required_count=200
        ),
    ]
    copied_root_directory: Path = create_sorted_image_files_with_pascal_voc_xml_from_CVAT_for_tensorflow_api(
        parent_of_exported_directory_from_CVAT=parent_of_exported_directory,
        available_annotation_index_range_list_of_folder_list=available_annotation_index_range_list_of_folder_list,
    )

    for copied_folder in copied_root_directory.iterdir():
        change_xml_filename_as_true_reference_name(input_directory=copied_folder)

    root_test_dataset_folder: Path = (
        copied_root_directory.parent / f"{copied_root_directory.name}_test_dataset"
    )
    root_test_dataset_folder.mkdir(exist_ok=True)
    augmented_root_directory: Path = (
        copied_root_directory.parent / f"{copied_root_directory.name}_true"
    )
    augmented_root_directory.mkdir(exist_ok=True)
    for sub_directory in copied_root_directory.iterdir():
        split_test_dataset_from_image_and_pascal_voc_xml(
            input_directory=sub_directory,
            output_directory=root_test_dataset_folder,
            test_dataset_fraction=0.3,
        )

    measurement_of_combinations = (
        wbfw109.libs.utilities.iterable.MeasurementOfCombinations(
            original_data_bytes=31457280
        )
    )

    # add cases
    image_padding_augmentation_list: list = [
        iaa.Pad(
            percent=(0, 0.35),
            pad_mode="constant",
            pad_cval=(0, 255),
            sample_independently=False,
        )
    ]

    measurement_of_combinations.create_case_for_consecutive_k(
        key="tensorflow_api_padding_augmentation_list",
        iterator=image_padding_augmentation_list,
        end_k=1,
    )

    # create augmentation_case_list
    augmentation_case_list: list = [
        iaa.Sequential(available_case, random_order=False)
        for available_case in measurement_of_combinations.get_available_case_list(
            should_include_empty_case=False
        )
    ]
    print(f"available augmentation size = {len(augmentation_case_list)}")

    for copied_folder in copied_root_directory.iterdir():
        augment_images_with_xml_from_pascal_voc_xml(
            input_directory=copied_folder,
            output_directory=augmented_root_directory / copied_folder.name,
            augmentation_case_list=augmentation_case_list,
        )

    for sub_directory in augmented_root_directory.iterdir():
        convert_xml_to_tfrecord(xml_dir=sub_directory, output_prefix="train_")

    convert_xml_to_tfrecord(xml_dir=root_test_dataset_folder, output_prefix="test_")


def _augment_padding_from_CVAT_example2():
    """
    + Note
        - it expects minimal CVAT directory tree like:

          📂 <parameter: parent_of_exported_directory_from_CVAT>
              ┖─📂 <export root>
                  ┖─📂 Annotations
                      ┖─ 📂 <task name>
                  ┖─📂 JPEGImages
                      ┖─ <task name>
            ...
    """
    parent_of_exported_directory_from_CVAT: Path = Path(
        "/mnt/c/users/wbfw109/task_training_unknown_list"
    )
    available_annotation_index_range_list_of_folder_list: list[
        RangeOfTwoPointWithRequiredCount
    ] = [
        RangeOfTwoPointWithRequiredCount(
            [RangeOfTwoPoint(start=0, end=199)], required_count=None
        ),
    ]
    copied_root_directory: Path = create_sorted_image_files_with_pascal_voc_xml_from_CVAT_for_tensorflow_api(
        parent_of_exported_directory_from_CVAT=parent_of_exported_directory_from_CVAT,
        available_annotation_index_range_list_of_folder_list=available_annotation_index_range_list_of_folder_list,
    )

    for copied_folder in copied_root_directory.iterdir():
        change_xml_filename_as_true_reference_name(input_directory=copied_folder)

    root_test_dataset_folder: Path = (
        copied_root_directory.parent / f"{copied_root_directory.name}_test_dataset"
    )
    root_test_dataset_folder.mkdir(exist_ok=True)

    augmented_root_directory: Path = (
        copied_root_directory.parent / f"{copied_root_directory.name}_true"
    )
    augmented_root_directory.mkdir(exist_ok=True)

    for sub_directory in copied_root_directory.iterdir():
        split_test_dataset_from_image_and_pascal_voc_xml(
            input_directory=sub_directory,
            output_directory=root_test_dataset_folder,
            test_dataset_fraction=0.3,
        )

    measurement_of_combinations = (
        wbfw109.libs.utilities.iterable.MeasurementOfCombinations(
            original_data_bytes=163956
        )
    )

    # add cases
    image_padding_augmentation_list: list = [
        iaa.Pad(
            percent=(0, 0.50),
            pad_mode="constant",
            pad_cval=(0, 255),
            sample_independently=False,
        ),
    ]
    measurement_of_combinations.create_case_for_consecutive_k(
        key="image_padding_augmentation",
        iterator=image_padding_augmentation_list,
        end_k=1,
    )

    base_angle = 90
    image_angle_augmentation_list: list = [
        iaa.Affine(rotate=(base_angle * -1, base_angle), mode="edge", fit_output=True),
    ]
    measurement_of_combinations.create_case_for_consecutive_k(
        key="image_angle_augmentation",
        iterator=image_angle_augmentation_list,
        end_k=1,
    )

    # create augmentation_case_list
    augmentation_case_list: list = [
        iaa.Sequential(available_case, random_order=False)
        for available_case in measurement_of_combinations.get_available_case_list(
            should_include_empty_case=True
        )
    ]
    print(f"available augmentation size = {len(augmentation_case_list)}")

    for copied_folder in copied_root_directory.iterdir():
        augment_images_with_xml_from_pascal_voc_xml(
            input_directory=copied_folder,
            output_directory=augmented_root_directory / copied_folder.name,
            augmentation_case_list=augmentation_case_list,
        )

    for sub_directory in augmented_root_directory.iterdir():
        convert_xml_to_tfrecord(xml_dir=sub_directory, output_prefix="train_")

    convert_xml_to_tfrecord(xml_dir=root_test_dataset_folder, output_prefix="test_")


def _augment_padding_from_pascal_voc_dataset():
    original_directory: Path = Path("/mnt/c/users/wbfw109/image_task_211019_AreaSample")
    no_exif_original_directory: Path = (
        original_directory.parent / f"{original_directory.name}_no_exif"
    )
    no_exif_original_directory.mkdir(exist_ok=True)

    augmented_root_directory: Path = (
        no_exif_original_directory.parent / f"{no_exif_original_directory.name}_true"
    )
    augmented_root_directory.mkdir(exist_ok=True)

    root_test_directory: Path = (
        no_exif_original_directory.parent
        / f"{no_exif_original_directory.name}_test_dataset"
    )
    root_test_directory.mkdir(exist_ok=True)

    remove_EXIF_of_image_and_save(
        output_parent_directory=no_exif_original_directory,
        image_list=wbfw109.libs.utilities.iterable.get_file_list(
            input_directory=original_directory, extensions=["jpg"]
        ),
    )
    for xml_file in wbfw109.libs.utilities.iterable.get_file_list(
        input_directory=original_directory, extensions=["xml"]
    ):
        shutil.copy2(f"{xml_file}", f"{no_exif_original_directory / xml_file.name}")

    split_test_dataset_from_image_and_pascal_voc_xml(
        input_directory=no_exif_original_directory,
        output_directory=root_test_directory,
        test_dataset_fraction=0.3,
        split_by_first_separator="_",
    )

    measurement_of_combinations = (
        wbfw109.libs.utilities.iterable.MeasurementOfCombinations(
            original_data_bytes=31457280
        )
    )

    # add cases
    image_padding_augmentation_list: list = [
        iaa.Pad(
            percent=(0, 0.35),
            pad_mode="constant",
            pad_cval=(0, 255),
            sample_independently=False,
        )
    ]

    measurement_of_combinations.create_case_for_consecutive_k(
        key="tensorflow_api_padding_augmentation_list",
        iterator=image_padding_augmentation_list,
        end_k=1,
    )

    # create augmentation_case_list
    augmentation_case_list: list = [
        iaa.Sequential(available_case, random_order=False)
        for available_case in measurement_of_combinations.get_available_case_list(
            should_include_empty_case=False
        )
    ]
    print(f"available augmentation size = {len(augmentation_case_list)}")

    augment_images_with_xml_from_pascal_voc_xml(
        input_directory=no_exif_original_directory,
        output_directory=augmented_root_directory,
        augmentation_case_list=augmentation_case_list,
    )

    convert_xml_to_tfrecord(xml_dir=augmented_root_directory, output_prefix="train_")

    convert_xml_to_tfrecord(xml_dir=root_test_directory, output_prefix="test_")


"""
Following functions are one command from existing file by other authors.
    - if you want to reuse these functions, must update arguments before use.
    - it is require some upper functions and locally installed library: object_detection
"""


def convert_xml_to_tfrecord(xml_dir: Union[Path, str], output_prefix: str = "") -> None:
    """
    Note:
        - Require to run in environment tf2 because this is written in that environment
        - Require to change command according to in your file.
        - you must cast as float type before casting as int type code on convert_xml_to_tfrecord.py file for specific conditions.

    Warning:
        Do not change env='PYTHONPATH' key name!
    """
    print("===== start function convert_xml_to_tfrecord()")

    xml_dir: Path = Path(xml_dir)
    p = subprocess.Popen(
        [
            sys.executable,
            "/home/wbfw109/study-core/study-flask/flask_apps/mysite/tutorial/tensorflow/for_tensorflow/git_object_detection_api/scripts/convert_xml_to_tfrecord.py",
            "--xml_dir",
            f"{xml_dir}",
            "--labels_path",
            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/my_label_map.pbtxt",
            "--output_path",
            f"/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/{output_prefix}{xml_dir.name}.record",
            # "--csv_path",
            # "/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/train_with_natural_scene_2.csv",
        ],
        # cwd=CONFIG_CLASS.PROJECT_ABSOLUTE_ROOT_DIRECTORY,
        env={"PYTHONPATH": PYTHON_ADDITIONAL_ENV},
        text=True,
        bufsize=1,
        stdin=subprocess.PIPE,
        stderr=sys.stderr,
        stdout=sys.stdout,
    )
    p.wait()
    print("===== end function convert_xml_to_tfrecord()")


"""
Following functions are one command for batch operation used in actual work.
    - it is require some upper functions and locally installed library: object_detection
    - if you want to reuse these functions, must update arguments before use.
"""


def _infer_from_model_and_write_pascal_voc_xml(
    model_directory: Union[Path, str],
    image_to_be_inferred_directory: Union[Path, str],
    creates_image_file: bool = False,
    creates_pascal_voc_file: bool = True,
):
    """
    set PATH_TO_LABELS
    """

    model_directory: Path = Path(model_directory)
    image_to_be_inferred_directory: Path = Path(image_to_be_inferred_directory)
    assert all([model_directory.is_dir(), image_to_be_inferred_directory.is_dir()])

    freeze_saved_model_directory: Path = model_directory / "freeze" / "saved_model"
    # tflite_graph_file: Path = model_directory / "tflite" / "tflite_graph.pb"

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS: Path = model_directory / "my_label_map.pbtxt"
    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS, use_display_name=False
    )

    print("Loading model…", end="")
    start_time = time.time()
    detection_model = tf.saved_model.load(str(freeze_saved_model_directory))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done! Took {} seconds".format(elapsed_time))

    # # check model
    # model.signatures['serving_default'].inputs
    # model.signatures['serving_default'].output_dtypes
    # model.signatures['serving_default'].output_shapes

    def run_inference_for_single_image(model, image):
        image = numpy.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        model_fn = model.signatures["serving_default"]
        output_dict = model_fn(input_tensor)
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop("num_detections"))
        output_dict = {
            key: value[0, :num_detections].numpy() for key, value in output_dict.items()
        }
        output_dict["num_detections"] = num_detections
        # # test
        # print(list(output_dict))
        # > ['detection_multiclass_scores', 'detection_scores', 'raw_detection_scores', 'detection_boxes', 'detection_classes', 'raw_detection_boxes', 'num_detections']

        # detection_classes should be ints.
        output_dict["detection_classes"] = output_dict["detection_classes"].astype(
            numpy.int64
        )

        # Handle models with masks:
        if "detection_masks" in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict["detection_masks"],
                output_dict["detection_boxes"],
                image.shape[0],
                image.shape[1],
            )
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
            output_dict["detection_masks_reframed"] = detection_masks_reframed.numpy()

        # print(len(output_dict["detection_boxes"]))
        # tf.image.non_max_suppression_with_scores()

        return output_dict

    DetectionBox = collections.namedtuple(
        "DetectionBox", ["coordinate_list", "category", "score"]
    )

    def get_detection_box_list(
        image,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        instance_boundaries=None,
        keypoints=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=20,
        min_score_thresh=0.1,
        agnostic_mode=False,
        line_thickness=4,
        groundtruth_box_visualization_color="black",
        skip_scores=False,
        skip_labels=False,
    ) -> list[DetectionBox]:
        """
        + Reference
            - https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/issues/69
                Extracting coordinates instead of drawing a box

        """

        # Create a display string (and color) for every box location, group any boxes
        # that correspond to the same location.
        box_to_display_str_map = collections.defaultdict(list)
        box_to_color_map = collections.defaultdict(str)
        box_to_instance_masks_map = {}
        box_to_instance_boundaries_map = {}
        box_to_score_map = {}
        box_to_keypoints_map = collections.defaultdict(list)
        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                if instance_masks is not None:
                    box_to_instance_masks_map[box] = instance_masks[i]
                if instance_boundaries is not None:
                    box_to_instance_boundaries_map[box] = instance_boundaries[i]
                if keypoints is not None:
                    box_to_keypoints_map[box].extend(keypoints[i])
                if scores is None:
                    box_to_color_map[box] = groundtruth_box_visualization_color
                else:
                    display_str = ""
                    if not skip_labels:
                        if not agnostic_mode:
                            if classes[i] in category_index.keys():
                                class_name = category_index[classes[i]]["name"]
                            else:
                                class_name = "N/A"
                            display_str = str(class_name)
                    if not skip_scores:
                        if not display_str:
                            display_str = "{}%".format(int(100 * scores[i]))
                        else:
                            display_str = "{}: {}%".format(
                                display_str, int(100 * scores[i])
                            )
                    box_to_display_str_map[box].append(display_str)
                    box_to_score_map[box] = scores[i]

                    if agnostic_mode:
                        box_to_color_map[box] = "DarkOrange"
                    else:
                        box_to_color_map[box] = visualization_utils.STANDARD_COLORS[
                            classes[i] % len(visualization_utils.STANDARD_COLORS)
                        ]
        # Draw all boxes onto image.
        detection_box_list: list[DetectionBox] = []
        counter_for = 0
        for box, color in box_to_color_map.items():
            ymin, xmin, ymax, xmax = box
            height, width, channels = image.shape
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            detection_box_list.append(
                DetectionBox(
                    coordinate_list=[xmin, ymin, xmax, ymax],
                    category=box_to_display_str_map[box][0],
                    score=(box_to_score_map[box] * 100),
                )
            )
            counter_for = counter_for + 1

        return detection_box_list

    def show_inference(
        model,
        image_file: Path,
        result_directory: Path,
        creates_image_file: bool,
        creates_pascal_voc_file: bool,
    ):
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_pillow: Image.Image = Image.open(image_file)

        image_np = numpy.array(image_pillow)
        # # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)

        # # Visualization of the results of a detection.
        detection_box_list: list[DetectionBox] = get_detection_box_list(
            image_np,
            output_dict["detection_boxes"],
            output_dict["detection_classes"],
            output_dict["detection_scores"],
            category_index,
            use_normalized_coordinates=True,
            line_thickness=1,
            min_score_thresh=0.01,
            skip_scores=True,
        )

        new_image_file: Path = result_directory / image_file.name
        if creates_image_file:
            image_pillow.save(new_image_file)

        if creates_pascal_voc_file:
            new_pascal_voc_xml_file: Path = (
                new_image_file.parent / f"{new_image_file.stem}.xml"
            )

            pascal_voc_xml_writer = pascal_voc_writer.Writer(
                f"{image_file}", *image_pillow.size
            )
            for detection_box in detection_box_list:
                pascal_voc_xml_writer.addObject(
                    name=detection_box.category,
                    xmin=detection_box.coordinate_list[0],
                    ymin=detection_box.coordinate_list[1],
                    xmax=detection_box.coordinate_list[2],
                    ymax=detection_box.coordinate_list[3],
                )
            pascal_voc_xml_writer.save(f"{new_pascal_voc_xml_file}")
            print(new_pascal_voc_xml_file)
            pass
        print("===== end function visualize_inference()")

    off_set_start_directory_count = 1
    off_set_start_image_count = 1

    for directory_count, material_sub_directory in enumerate(
        image_to_be_inferred_directory.iterdir(), start=1
    ):
        image_file_list: list[Path] = sorted(list(material_sub_directory.glob("*.jpg")))

        pascal_voc_xml_to_be_written_directory: Path = Path(
            f"{image_to_be_inferred_directory.parent}/{image_to_be_inferred_directory.name}_inferred_annotations"
        )
        pascal_voc_xml_to_be_written_directory.mkdir(exist_ok=True)

        for image_count, image_file in enumerate(image_file_list, start=1):
            if (
                directory_count < off_set_start_directory_count
                or image_count < off_set_start_image_count
            ):
                continue
            show_inference(
                model=detection_model,
                result_directory=pascal_voc_xml_to_be_written_directory,
                image_file=image_file,
                creates_image_file=creates_image_file,
                creates_pascal_voc_file=creates_pascal_voc_file,
            )
            print(f"{directory_count} . {image_count} done for {image_file.name}")
            print("================\n")
            # if image_count > 5:
            # break
        # break
    print("============= end ")
