"""
Note:
    This is dedicated for tensorflow 1 version
    This is revision versino from me. refer to "## added"
    Set your IOU_THRESHOLD = 0.5, CONFIDENCE_THRESHOLD = 0.7
    
- confusion matrix 에 row, column 이 1개씩 추가됨. 아예 정답을 다른 것으로 예측한 것이 아니라 아예 분류하지 못한 것들.
- ipython_mode = True 를 하면 SystemExit 이 마지막에 발생하는데 상관이 없다.

Reference
    https://github.com/svpino/tf_object_detection_cm/blob/master/confusion_matrix.py
    https://sujithsoppa.medium.com/object-detection-model-using-end-to-end-custom-development-with-tensorflow-2-73d438fe3cb4
        여기에 있는 것을 따라하면 값이 NaN 이 나오고 precision, recall 이 잘못 나오는 버그가 있다.
"""
## added
is_ipython_mode = False

import math
import os
import subprocess
import sys
from pathlib import Path

from starter.config import CONFIG_CLASS

TEMP_FILES_PATH: Path = Path.home() / ".local_files"
NEW_SLIM_PATH: Path = TEMP_FILES_PATH / "tensorflow/models/research/slim"

sys.path.append(str(NEW_SLIM_PATH))
sys.path.append(str(TEMP_FILES_PATH / "tensorflow/models/research"))

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
import tensorflow as tf
from object_detection.core import standard_fields
from object_detection.metrics import tf_example_parser
from object_detection.utils import label_map_util

flags = tf.app.flags

flags.DEFINE_string("label_map", None, "Path to the label map")
flags.DEFINE_string("detections_record", None, "Path to the detections record file")
flags.DEFINE_string("output_path", None, "Path to the output the results in a csv.")
flags.DEFINE_string("confdience_threshold", None, "confdience_threshold")

FLAGS = flags.FLAGS

IOU_THRESHOLD = 0.5
# set min_score
if not FLAGS.confdience_threshold:
    CONFIDENCE_THRESHOLD = 0.5
else:
    CONFIDENCE_THRESHOLD = float(FLAGS.confdience_threshold)


def get_image_iou(groundtruth_box, detection_box):
    g_ymin, g_xmin, g_ymax, g_xmax = tuple(groundtruth_box.tolist())
    d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box.tolist())

    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)


def get_confusion_matrix_using_object_detection(detections_record, categories):
    record_iterator = tf.python_io.tf_record_iterator(path=detections_record)
    data_parser = tf_example_parser.TfExampleDetectionAndGTParser()

    confusion_matrix = numpy.zeros(shape=(len(categories) + 1, len(categories) + 1))

    image_index = 0
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        decoded_dict = data_parser.parse(example)

        image_index += 1

        if decoded_dict:
            groundtruth_boxes = decoded_dict[
                standard_fields.InputDataFields.groundtruth_boxes
            ]
            groundtruth_classes = decoded_dict[
                standard_fields.InputDataFields.groundtruth_classes
            ]

            detection_scores = decoded_dict[
                standard_fields.DetectionResultFields.detection_scores
            ]
            detection_classes = decoded_dict[
                standard_fields.DetectionResultFields.detection_classes
            ][detection_scores >= CONFIDENCE_THRESHOLD]
            detection_boxes = decoded_dict[
                standard_fields.DetectionResultFields.detection_boxes
            ][detection_scores >= CONFIDENCE_THRESHOLD]

            matches = []

            if image_index % 100 == 0:
                print("Processed %d images" % (image_index))

            for i in range(len(groundtruth_boxes)):
                for j in range(len(detection_boxes)):
                    iou = get_image_iou(groundtruth_boxes[i], detection_boxes[j])

                    if iou > IOU_THRESHOLD:
                        matches.append([i, j, iou])

            matches = numpy.array(matches)
            if matches.shape[0] > 0:
                # Sort list of matches by descending IOU so we can remove duplicate detections
                # while keeping the highest IOU entry.
                matches = matches[matches[:, 2].argsort()[::-1][: len(matches)]]

                # Remove duplicate detections from the list.
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]

                # Sort the list again by descending IOU. Removing duplicates doesn't preserve
                # our previous sort.
                matches = matches[matches[:, 2].argsort()[::-1][: len(matches)]]

                # Remove duplicate ground truths from the list.
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]

            for i in range(len(groundtruth_boxes)):
                if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
                    confusion_matrix[groundtruth_classes[i] - 1][
                        detection_classes[int(matches[matches[:, 0] == i, 1][0])] - 1
                    ] += 1
                else:
                    confusion_matrix[groundtruth_classes[i] - 1][
                        confusion_matrix.shape[1] - 1
                    ] += 1

            for i in range(len(detection_boxes)):
                if matches.shape[0] > 0 and matches[matches[:, 1] == i].shape[0] == 0:
                    confusion_matrix[confusion_matrix.shape[0] - 1][
                        detection_classes[i] - 1
                    ] += 1
        else:
            print("Skipped image %d" % (image_index))

    print("Processed %d images" % (image_index))

    return confusion_matrix


def export_human_readable_confusion_matrix(confusion_matrix, categories, output_path):
    print("\nConfusion Matrix:")
    ## added
    temp_savefile_path: Path = Path(output_path)
    savefig_file: str = str(
        temp_savefile_path.parent / f"{temp_savefile_path.stem}.png"
    )
    class_names = [str(Path(x["name"]).name) for x in categories]
    array = confusion_matrix.astype(int)
    df_cm = pandas.DataFrame(
        array, range(len(class_names) + 1), range(len(class_names) + 1)
    )
    df_cm = df_cm / df_cm.sum(axis=1)[:, numpy.newaxis]
    # plt.figure(figsize=(10,7))
    fig, ax = plt.subplots(figsize=(20, 20))
    seaborn.heatmap(
        df_cm,
        annot=True,
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="YlGnBu",
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    if is_ipython_mode:
        plt.show()
    else:
        print(confusion_matrix, "\n")
    plt.savefig(savefig_file)

    results = []

    for i in range(len(categories)):
        id = categories[i]["id"] - 1
        name = categories[i]["name"]

        total_target = numpy.sum(confusion_matrix[id, :])
        total_predicted = numpy.sum(confusion_matrix[:, id])

        precision = float(confusion_matrix[id, id] / total_predicted)
        recall = float(confusion_matrix[id, id] / total_target)

        # print('precision_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, precision))
        # print('recall_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, recall))

        ## added
        if (
            not math.isnan(precision)
            and precision != 0
            and not math.isnan(recall)
            and recall != 0
        ):
            f1_score: str = "%.6f" % float(2 / (precision**-1 + recall**-1))
        else:
            f1_score: str = "not number"

        results.append(
            {
                "category": name,
                "precision_@{}IOU".format(IOU_THRESHOLD): precision,
                "recall_@{}IOU".format(IOU_THRESHOLD): recall,
                "f1_score_@{}IOU".format(IOU_THRESHOLD): f1_score,
            }
        )

    df = pandas.DataFrame(results)
    print(df)
    df.to_csv(output_path)


def main(argv):
    ## added start
    if is_ipython_mode:
        FLAGS.detections_record = "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_temp/freeze/test_prediction.record"
        FLAGS.label_map = "/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/my_label_map.pbtxt"
        FLAGS.output_path = "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_temp/freeze/confusion_matrix.csv"
        import IPython
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.display import clear_output, display

        InteractiveShell.ast_node_interactivity = "all"
    ## added end

    del argv
    required_flags = ["detections_record", "label_map", "output_path"]
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError("Flag --{} is required".format(flag_name))

    label_map = label_map_util.load_labelmap(FLAGS.label_map)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=100, use_display_name=True
    )

    confusion_matrix = get_confusion_matrix_using_object_detection(
        FLAGS.detections_record, categories
    )

    ## added start
    temp_output_path: Path = Path(FLAGS.output_path)
    FLAGS.output_path = (
        temp_output_path.parent
        / f"{temp_output_path.stem}-ms_{CONFIDENCE_THRESHOLD}{temp_output_path.suffix}"
    )
    ## added end

    export_human_readable_confusion_matrix(
        confusion_matrix, categories, FLAGS.output_path
    )


if __name__ == "__main__":
    tf.app.run(main)
