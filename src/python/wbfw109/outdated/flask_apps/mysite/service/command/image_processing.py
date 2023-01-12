from mysite.config import CONFIG_CLASS
from mysite.service.video import LocalImages
from logging import error
from datetime import datetime
from pathlib import PurePath, Path
import argparse
import numpy
import cv2

if __name__ == "__main__":
    CONFIG_CLASS.ensure_folders_exist()

    parser = argparse.ArgumentParser(
        description="perform some operations for image processing.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v", "--verbosity", help="increase output verbosity", action="count", default=0
    )
    group.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument(
        "--image-dir",
        help="directory where source images exist. \ndefault directory is $home/image_processing/source",
        default=str(CONFIG_CLASS.IMAGE_PROCESSING_SOURCE_PATH),
    )
    parser.add_argument(
        "--result-dir",
        help="directory to be stored as some operations result. \ndefault directory is $home/image_processing/result",
        default=str(CONFIG_CLASS.IMAGE_PROCESSING_RESULT_PATH),
    )
    required_group = parser.add_argument_group("required named arguments")
    required_group.add_argument(
        "-m",
        "--mode",
        help="image processsing mode.",
        required=True,
        nargs="+",
        choices=[
            "crop",
            "resize",
            "grayscale",
            "rotation",
            "merge",
            "blend",
            "hstack",
            "vstack",
        ],
    )

    args = parser.parse_args()

    print(args)

    path_image_dir: Path = Path(args.image_dir)
    if not path_image_dir.exists() or not path_image_dir.is_dir():
        raise Exception("PathError: Source Directory Not Found")
    path_result_dir: Path = Path(args.result_dir)
    if not path_result_dir.exists() or not path_result_dir.is_dir():
        raise Exception("PathError: Result Directory Not Found")

    # instance
    local_images = LocalImages(path_image_dir)
    local_images.operate(args.mode)
