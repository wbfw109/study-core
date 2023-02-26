# %%
"""
This library was written to collect the cumbersome individual tasks and run at a same time. 
it includes commands, raw codes.

???
    환경변수를 설정해서 subprocess 를 사용하려면 shell=False  sys.executable 을 사용해야 한다. 그렇지 않으면 다음과 같은 오류 발생
        /bin/sh: 1: python: not found

## # tensorboard
tensorboard --logdir "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training"

## # tfrecord viewer
cd /mnt/c/Users/wbfw109/Downloads/repository/test_labeling/git_clone/tfrecord-viewer/
python tfviewer.py /mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/test.record
"""

is_ipython_mode = False
if is_ipython_mode:
    import IPython
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.display import clear_output, display

    InteractiveShell.ast_node_interactivity = "all"

import subprocess
import sys
from pathlib import Path

import imgaug.augmenters as iaa
import tensorflow as tf
from mysite.config import CONFIG_CLASS
from wbfw109.libs.utilities import iterable, machine_learning

TEMP_FILES_PATH: Path = Path.home() / ".local_files"
NEW_SLIM_PATH: Path = TEMP_FILES_PATH / "tensorflow/models/research/slim"

sys.path.append(str(NEW_SLIM_PATH))
sys.path.append(str(TEMP_FILES_PATH / "tensorflow/models/research"))

import time

import numpy
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
from PIL import Image

# IPython.get_ipython().run_line_magic("matplotlib", "inline")

PYTHON_ADDITIONAL_ENV: str = ":".join(
    {str(NEW_SLIM_PATH), str(TEMP_FILES_PATH / "tensorflow/models/research")}
)
print(PYTHON_ADDITIONAL_ENV)


def export_inference_graph() -> None:
    """
    Note:
        Require to run in environment tf1 if your model is tf1 from tf zoo
        Require to change command according to in your file

    Warning:
        Do not change env='PYTHONPATH' key name!
    """
    print("===== start function export_inference_graph()")

    p = subprocess.Popen(
        [
            sys.executable,
            Path.home()
            / ".local_files/tensorflow/models/research/object_detection/export_inference_graph.py",
            "--input_type",
            "image_tensor",
            "--pipeline_config_path",
            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_padding_50_angle_90_map_900_recall_72/pipline_mine_colab.config",
            "--trained_checkpoint_prefix",
            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_padding_50_angle_90_map_900_recall_72/model.ckpt-87142",
            "--output_directory",
            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_padding_50_angle_90_map_900_recall_72/freeze",
        ],
        cwd=CONFIG_CLASS.PROJECT_ABSOLUTE_ROOT_DIRECTORY,
        env={"PYTHONPATH": PYTHON_ADDITIONAL_ENV},
        text=True,
        bufsize=1,
        stdin=subprocess.PIPE,
        stderr=sys.stderr,
        stdout=sys.stdout,
    )
    p.wait()
    print("===== end function export_inference_graph()")


def infer_detections(input_tf_record_path_list: list[Path]) -> None:
    """
    it create tfrecord file from frozen_inference_graph

    Note:
        Require to run in environment tf1 if your model is tf1 from tf zoo
        Require to change command according to in your file

    it is possible to accept multiple tf_record. refer to the that code.py

    Warning:
        Do not change env='PYTHONPATH' key name!
    """
    print("===== start function infer_detections()")
    input_tf_record_string_join_paths: str = ",".join(
        [
            f"{input_tf_record_path}"
            for input_tf_record_path in input_tf_record_path_list
        ]
    )

    p = subprocess.Popen(
        [
            sys.executable,
            Path.home()
            / ".local_files/tensorflow/models/research/object_detection/inference/infer_detections.py",
            "--input_tfrecord_paths",
            input_tf_record_string_join_paths,
            "--output_tfrecord_path",
            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_padding_50_angle_90_map_900_recall_72/freeze/_test_collection_prediction.record",
            "--inference_graph",
            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_padding_50_angle_90_map_900_recall_72/freeze/frozen_inference_graph.pb",
        ],
        cwd=CONFIG_CLASS.PROJECT_ABSOLUTE_ROOT_DIRECTORY,
        env={"PYTHONPATH": PYTHON_ADDITIONAL_ENV},
        text=True,
        bufsize=1,
        stdin=subprocess.PIPE,
        stderr=sys.stderr,
        stdout=sys.stdout,
    )
    p.wait()
    print("===== end function infer_detections()")


def visualize_inference() -> None:
    """
    Todo: 정리

    it saves images which objects inference is drawn in.
    """
    print("===== start function visualize_inference()")

    model_directory: Path = Path(
        "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_padding_50_angle_90_map_900_recall_72"
    )
    freeze_saved_model_directory: Path = model_directory / "freeze" / "saved_model"

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS: Path = model_directory / "my_label_map.pbtxt"
    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS, use_display_name=False
    )

    test_image_directory: Path = (
        model_directory.parent / "images" / "image_tasks_sorted_test_dataset"
    )
    inference_result_directory: Path = (
        freeze_saved_model_directory.parent / f"{test_image_directory.stem}_inference"
    )
    inference_result_directory.mkdir(exist_ok=True)

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

    def show_inference(model, image_path: Path):
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = numpy.array(Image.open(image_path))
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict["detection_boxes"],
            output_dict["detection_classes"],
            output_dict["detection_scores"],
            category_index,
            instance_masks=output_dict.get("detection_masks_reframed", None),
            use_normalized_coordinates=True,
            line_thickness=1,
            # # added. set confidence threshold
            min_score_thresh=0.15,
        )
        Image.fromarray(image_np).save(inference_result_directory / image_path.name)

    # object_classes = [
    #     'mug',
    #     'takeout_hot',
    #     'takeout_ice',
    #     'tumbler',
    #     'bagel',
    #     'cheese_cake',
    #     'croissant',
    #     'fruit_cake',
    #     'macaron',
    #     'muffin',
    #     'scone',
    #     'square_sandwich',
    #     'tiramisu',
    #     'triangle_sandwich'
    #     ]

    # for object_class in object_classes:
    #     TEST_IMAGE_FILE_LIST: list[Path] = sorted(
    #         list(test_image_directory.glob(f"*{object_class}*.jpg"))
    #     )
    #     for c, image_path in enumerate(TEST_IMAGE_FILE_LIST, start=1):
    #         show_inference(detection_model, image_path)
    #         print(f"{c} done for {object_class}")
    TEST_IMAGE_FILE_LIST: Path = iterable.get_file_list(
        "/mnt/c/users/wbfw109/image_tasks_sorted_test_dataset", extensions=["jpg"]
    )
    for test_image in TEST_IMAGE_FILE_LIST:
        show_inference(detection_model, test_image)
        # break
    # show_inference(detection_model, Path("/mnt/c/Users/wbfw109/MyDrive/shared_resource/images/test/cam02_fd_cheese_cake_02_00.jpg"))
    # print(f"cheese... done.")
    print("===== end function visualize_inference()")


def export_tflite_ssd_graph() -> None:
    """
    Note:
        Require to run in environment tf1 if your model is tf1 from tf zoo
        Require to change command according to in your file.
        If model tensorflow version is tf1, after using this function you require to use convert_tflite_graph_to_tflite_from_tf_v1_ssd_models

    Warning:
        Do not change env='PYTHONPATH' key name!
    """
    print("===== start function export_tflite_ssd_graph()")
    p = subprocess.Popen(
        [
            sys.executable,
            Path.home()
            / ".local_files/tensorflow/models/research/object_detection/export_tflite_ssd_graph.py",
            "--input_type",
            "image_tensor",
            "--max_detections",
            "100",
            "--pipeline_config_path",
            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_padding_50_angle_90_map_900_recall_72/pipline_mine_colab.config",
            "--trained_checkpoint_prefix",
            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_padding_50_angle_90_map_900_recall_72/model.ckpt-87142",
            "--output_directory",
            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_padding_50_angle_90_map_900_recall_72/tflite",
        ],
        cwd=CONFIG_CLASS.PROJECT_ABSOLUTE_ROOT_DIRECTORY,
        env={"PYTHONPATH": PYTHON_ADDITIONAL_ENV},
        text=True,
        bufsize=1,
        stdin=subprocess.PIPE,
        stderr=sys.stderr,
        stdout=sys.stdout,
    )
    p.wait()
    print("===== end function export_tflite_ssd_graph()")


def infer_tflite() -> None:
    """
    # Todo
    """
    # model_path = str(tflite_graph_file.parent / "model_quantized.tflite")
    # is_quantized = "quantized" in model_path.lower()

    # model_interpreter = tf.lite.Interpreter(model_path=model_path)
    # model_interpreter.allocate_tensors()
    # input_id = model_interpreter.get_input_details()[0]["index"]
    # output_det = model_interpreter.get_output_details()
    # output_id_list = [output_det[x]["index"] for x in range(4)]

    # images: list[Path] = [
    #     Path(
    #         "/mnt/c/Users/wbfw109/MyDrive/shared_resource/images/test/cam01_fd_muffin_02_60.jpg"
    #     )
    # ]

    # def get_mobilenet_input(f, out_size=(300, 300), is_quant=True):
    #     image_np = numpy.array(Image.open(f).resize(out_size))
    #     if not (is_quant):
    #         image_np = image_np.astype(numpy.float32) / 128 - 1
    #     return numpy.array([image_np])

    # def print_output(input_files, result):
    #     boxes, classes, scores, num_det = result

    #     for i, fname in enumerate(input_files):
    #         n_object = int(num_det[i])

    #         print("{} - found objects:".format(fname))
    #         for j in range(n_object):
    #             class_id = int(classes[i][j]) + 1
    #             label = labels_ids[class_id]
    #             score = scores[i][j]
    #             if score < 0.5:
    #                 continue
    #             box = boxes[i][j]
    #             print("  ", class_id, label, score, box)

    # for image in images:
    #     img = get_mobilenet_input(str(image), is_quant=is_quantized)
    #     model_interpreter.set_tensor(input_id, img)
    #     model_interpreter.invoke()
    #     boxes = model_interpreter.get_tensor(output_id_list[0])
    #     classes = model_interpreter.get_tensor(output_id_list[1])
    #     scores = model_interpreter.get_tensor(output_id_list[2])
    #     num_det = model_interpreter.get_tensor(output_id_list[3])
    #     print_output([str(image)], [boxes, classes, scores, num_det])
    pass


def convert_tflite_graph_to_tflite_from_tf_v1_ssd_models(
    tflite_graph_file: Path = Path.home(), is_quantized=False
) -> None:
    """

    assert output_file path == tflite_graph_file.parent / "model.tflite"

    Reference
        https://gist.github.com/apivovarov/eff80275d0f72e4582c105921919b852

    Args:
        tflite_graph_file (Path): [description]
        is_quantized (bool, optional): [description]. Defaults to False.

    e.g.
        model_directory: Path = Path("/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_padding_50_angle_90_map_900_recall_72")
        tflite_graph_file: Path = model_directory / "tflite" / "tflite_graph.pb"
        convert_tflite_graph_to_tflite_from_tf_v1_ssd_models(
            tflite_graph_file=tflite_graph_file, is_quantized=True
        )
    """
    model_directory: Path = Path(
        "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_padding_50_angle_90_map_900_recall_72"
    )
    tflite_graph_file: Path = model_directory / "tflite" / "tflite_graph.pb"
    is_quantized = True

    print("===== start function convert_tflite_graph_to_tflite_from_tf_v1_ssd_models()")
    model_suffix: list[str] = []
    if is_quantized:
        model_suffix.append("_quantized")

    output_file: Path = tflite_graph_file.parent / (
        "model" + "".join(model_suffix) + ".tflite"
    )

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        str(tflite_graph_file),
        input_arrays=["normalized_input_image_tensor"],
        output_arrays=[
            "TFLite_Detection_PostProcess",
            "TFLite_Detection_PostProcess:1",
            "TFLite_Detection_PostProcess:2",
            "TFLite_Detection_PostProcess:3",
        ],
        input_shapes={"normalized_input_image_tensor": [1, 300, 300, 3]},
    )
    converter.inference_type = (
        tf.compat.v1.lite.constants.QUANTIZED_UINT8
        if is_quantized
        else tf.compat.v1.lite.constants.FLOAT
    )
    converter.allow_custom_ops = True
    converter.quantized_input_stats = (
        {"normalized_input_image_tensor": (128.0, 127.0)} if is_quantized else None
    )

    tflite_model = converter.convert()

    # Save the model.
    with open(str(output_file), "wb") as f:
        f.write(tflite_model)
    print("===== end function convert_tflite_graph_to_tflite_from_tf_v1_ssd_models()")


if __name__ == "__main__":
    # # 2. get confusion matrix and visualize_inference
    # # Require to run in environment tf1 if your model is tf1 from tf zoo
    # infer_detections(
    #     input_tf_record_path_list=[
    #         Path("/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/test_image_tasks_sorted_test_dataset.record"),
    #         Path("/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/test_image_task_211019_AreaSample_no_exif_test_dataset.record"),
    #         Path("/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/test_task_training_unknown_list_sorted_test_dataset.record")
    #     ])

    # visualize_inference()
    # # get tflite file
    model_directory: Path = Path(
        "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_padding_50_angle_90_map_900_recall_72"
    )
    freeze_saved_model_dir: Path = model_directory / "freeze" / "saved_model"
    tflite_graph_file: Path = model_directory / "tflite" / "tflite_graph.pb"
    convert_tflite_graph_to_tflite_from_tf_v1_ssd_models(
        tflite_graph_file=tflite_graph_file, is_quantized=True
    )

    pass
