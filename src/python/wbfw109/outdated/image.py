"""
⚠️ deprecated. require to reconstruct file

This library not includes tensorflow library.
about library using tensorflow, refer to the machine_learning.py file.
"""
import re
from pathlib import Path
from typing import Union

import cv2
import wbfw109.libs.utilities.iterable
from PIL import Image, ImageOps


def save_images_with_no_exif(
    input_directory: Union[Path, str],
    output_directory: Union[Path, str],
    extensions: list[str],
    should_rotate_based_on_exif: bool = False,
) -> None:
    """
    ??? Why do saved files get smaller in size?

    e.g.
        save_images_with_no_exif(
            input_directory="/mnt/c/Users/wbfw109/test_exif",
            output_directory="/mnt/c/Users/wbfw109/test_new_exif",
            extensions=[
                "jpg",
            ],
        )
    """
    # prepare
    input_directory: Path = Path(input_directory)
    output_directory: Path = Path(output_directory)
    assert input_directory.is_dir()
    assert output_directory.is_dir()
    assert input_directory != output_directory

    # preprocess
    input_image_list: list[Path] = wbfw109.libs.utilities.iterable.get_file_list(
        input_directory, extensions=extensions
    )

    # process
    if should_rotate_based_on_exif:
        for file_path in input_image_list:
            ImageOps.exif_transpose(Image.open(file_path)).save(
                output_directory / file_path.name
            )
    else:
        for file_path in input_image_list:
            Image.open(file_path).save(output_directory / file_path.name)

    print("completes save_images_with_no_exif")


def resize_and_convert_extension(
    image_directory: Union[Path, str],
    width: int,
    height: int,
    extension_a: str,
    extension_b: str,
) -> None:
    """This function uses cv2 insetad of PIL because cv2 canread all .ppm file.

    Args:
        width (int): [description]
        height (int): [description]
        extension_a (str): [description]
        extension_b (str): [description]
    """
    image_directory: Path = Path(image_directory)
    DOWNLOADED_IMAGE_LIST: list[Path] = sorted(
        list(image_directory.glob(f"*.{extension_a}"))
    )
    for image_file in DOWNLOADED_IMAGE_LIST:
        new_file: Path = image_file.parent / f"{image_file.stem}.{extension_b}"
        img = cv2.imread(f"{image_file}")
        resized_img = cv2.resize(img, dsize=(width, height))
        cv2.imwrite(f"{new_file}", resized_img)
