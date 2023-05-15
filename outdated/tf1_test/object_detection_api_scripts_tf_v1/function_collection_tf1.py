# %%
"""
This library was written to collect the cumbersome individual tasks and run at a same time. 
it includes commands, raw codes.
???
    환경변수를 설정해서 subprocess 를 사용하려면 shell=False  sys.executable 을 사용해야 한다. 그렇지 않으면 다음과 같은 오류 발생
        /bin/sh: 1: python: not found

"""
is_ipython_mode = True
if is_ipython_mode:
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.ast_node_interactivity = "all"

from starter.config import CONFIG_CLASS
from pathlib import Path
import sys
import subprocess

TEMP_FILES_PATH: Path = Path.home() / ".local_files"
NEW_SLIM_PATH: Path = TEMP_FILES_PATH / "tensorflow/models/research/slim"

sys.path.append(str(NEW_SLIM_PATH))
sys.path.append(str(TEMP_FILES_PATH / "tensorflow/models/research"))
PYTHON_ADDITIONAL_ENV = ":".join(
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


def export_confusion_matrix_tf1(confdience_threshold=0.5) -> None:
    """
    Note:
        Require to change command according to in your file.

    Warning:
        Do not change env='PYTHONPATH' key name!
    """
    print("===== start function export_confusion_matrix_tf1()")
    p = subprocess.Popen(
        [
            sys.executable,
            "starter/object_detection_api_scripts_tf_v1/export_confusion_matrix_tf1.py",
            "--detections_record",
            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_padding_50_angle_90_map_900_recall_72/freeze/_test_collection_prediction.record",
            "--label_map",
            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/annotations/my_label_map.pbtxt",
            "--output_path",
            "/mnt/c/Users/wbfw109/MyDrive/shared_resource/training_padding_50_angle_90_map_900_recall_72/freeze/confusion_matrix.csv",
            "--confdience_threshold",
            f"{confdience_threshold}",
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
    print("===== end function export_confusion_matrix_tf1()")


def export_tflite_ssd_graph() -> None:
    """
    Note:
        Require to run in environment tf1 if your model is tf1 from tf zoo
        Require to change command according to in your file.
        If model tensorflow version is tf1, after using this function you require to use convert_tflite_graph_to_tflite_from_tf_v1_ssd_models from utils_ml.py

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


if __name__ == "__main__":
    ## 1. export graph
    # export_inference_graph()

    ## 3. export confusion matrix
    # for x in range(1, 9):
    #     export_confusion_matrix_tf1(str(round(x / 10, 2)))
    # 학습 threshold 를 낮춰야하나-
    ## get tflite file
    export_tflite_ssd_graph()

    pass
