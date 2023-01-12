#%%
#

import IPython
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output
import requests
from PIL import Image
import numpy as np
from IPython import get_ipython
from pathlib import Path
import sys

TEMP_FILES_PATH: Path = Path.home() / ".local_files"
NEW_SLIM_PATH: Path = TEMP_FILES_PATH / "tensorflow/models/research/slim"

sys.path.append(str(NEW_SLIM_PATH))
sys.path.append(str(TEMP_FILES_PATH / "tensorflow/models/research"))

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

url2 = r"https://upload.wikimedia.org/wikipedia/en/9/99/Kitesurfing_in_Noordwijk2.JPG"
url = r"https://i2.wp.com/gyogamman.com/wp-content/uploads/2020/04/dog-3753706_1280_Pixabay%EB%A1%9C%EB%B6%80%ED%84%B0-%EC%9E%85%EC%88%98%EB%90%9C-monicore%EB%8B%98%EC%9D%98-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%9E%85%EB%8B%88%EB%8B%A4..jpg"
"""
??? 여러번 요청하면 일정시간 밴 먹는듯? reponse 가 안나옴. 
- Interrupt Jupyter Kenel 을 눌러주고 (Jupyter 정지버튼)
    다음 셀에서 사용하자.
"""
response = requests.get(url2, stream=True)
response
print("asdf")

# %%
#
from io import BytesIO

if response.status_code == 200:
    print("asfd")
    my_image = Image.open(BytesIO(response.content))
    my_image.show()

# This code is mainly from object detection API jupyter notebook and object detection API preprocessor.py
from object_detection.core import preprocessor
import functools, os
from object_detection import inputs
from object_detection.core import standard_fields as fields
from matplotlib import pyplot as mp
from matplotlib import pyplot as plt
import tensorflow as tf
from PIL import Image

# This is needed to display the images.
# %matplotlib inline
get_ipython().run_line_magic("matplotlib", "inline")


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return (
        np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.float32)
    )  # .astype(np.uint8)


number_of_repeats = 1  # lot of augmentations have probabilities < 1 will not happen if repeated only once.

image_np = load_image_into_numpy_array(my_image)
# UNCOMMENT THIS CODE TO USE YOUR PICTURE
# IMAGE = "" # add image here
# image2 = Image.open(IMAGE)

save_to_disk = False
directory = "visualize_augmentation"
preprocessing_list = [
    None,
    (preprocessor.random_horizontal_flip, {}),
    (preprocessor.random_vertical_flip, {}),
    (preprocessor.random_rotation90, {}),
    (
        preprocessor.random_pixel_value_scale,
        {},
    ),  # slightly changes the values of pixels
    (preprocessor.random_image_scale, {}),
    (preprocessor.random_rgb_to_gray, {}),
    (preprocessor.random_adjust_brightness, {}),
    (preprocessor.random_adjust_contrast, {}),
    (preprocessor.random_adjust_hue, {}),
    (preprocessor.random_adjust_saturation, {}),
    (preprocessor.random_distort_color, {}),  # very strong augmentation
    # (preprocessor.random_jitter_boxes, {}),
    (preprocessor.random_crop_image, {}),
    (preprocessor.random_pad_image, {}),  # changes the pixel values
    # (preprocessor.random_absolute_pad_image, {}),
    (preprocessor.random_crop_pad_image, {}),
    # (preprocessor.random_crop_to_aspect_ratio, {}),
    (preprocessor.random_pad_to_aspect_ratio, {}),
    (preprocessor.random_black_patches, {}),
    # (preprocessor.random_resize_method, {}),
    (preprocessor.resize_to_min_dimension, {}),
    (preprocessor.scale_boxes_to_pixel_coordinates, {}),
    # (preprocessor.subtract_channel_mean, {}),
    # (preprocessor.random_self_concat_image, {}),
    (preprocessor.ssd_random_crop, {}),
    (preprocessor.ssd_random_crop_pad, {}),
    # (preprocessor.ssd_random_crop_fixed_aspect_ratio, {}),
    (preprocessor.ssd_random_crop_pad_fixed_aspect_ratio, {}),
    # (preprocessor.convert_class_logits_to_softmax, {}),
    #
]
for preprocessing_technique in preprocessing_list:
    for i in range(number_of_repeats):

        tf.reset_default_graph()
        # tf.compat.v1.reset_default_graph()
        if preprocessing_technique is not None:
            print(str(preprocessing_technique[0].__name__))
        else:
            print("Image without augmentation: ")
        if preprocessing_technique is not None:
            data_augmentation_options = [preprocessing_technique]
        else:
            data_augmentation_options = []
        data_augmentation_fn = functools.partial(
            inputs.augment_input_data,
            data_augmentation_options=data_augmentation_options,
        )
        tensor_dict = {
            fields.InputDataFields.image: tf.constant(image_np.astype(np.float32)),
            fields.InputDataFields.groundtruth_boxes: tf.constant(
                np.array([[0.5, 0.5, 1.0, 1.0]], np.float32)
            ),
            fields.InputDataFields.groundtruth_classes: tf.constant(
                np.array([1.0], np.float32)
            ),
        }
        augmented_tensor_dict = data_augmentation_fn(tensor_dict=tensor_dict)
        with tf.Session() as sess:
            augmented_tensor_dict_out = sess.run(augmented_tensor_dict)
        plt.figure()
        plt.imshow(augmented_tensor_dict_out[fields.InputDataFields.image].astype(int))
        plt.show()

        if save_to_disk:
            plt.imshow(
                augmented_tensor_dict_out[fields.InputDataFields.image].astype(int)
            )
            if not os.path.exists(directory):
                os.makedirs(directory)
            if preprocessing_technique is not None:
                mp.savefig(
                    directory
                    + "/augmentation_"
                    + str(preprocessing_technique[0].__name__)
                    + "_"
                    + str(i)
                    + ".png",
                    dpi=300,
                    bbox_inches="tight",
                )
            else:
                mp.savefig(
                    directory + "/no_augmentation.png", dpi=300, bbox_inches="tight"
                )
        plt.close("all")
