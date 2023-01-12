import IPython
from IPython.core.interactiveshell import InteractiveShell
from enum import Enum
from pathlib import Path
import sys
import subprocess
import ast

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"


# * setting
# ** common environment setting for each python version
class EnvironmentLocation(Enum):
    LOCAL = 1
    LOCAL_WITH_DOCKER = 2
    GOOGLE_COLAB = 11


environment_location: EnvironmentLocation = EnvironmentLocation.LOCAL
# tesnorflow_version: str = 1 | 2
tensorflow_version_as_integer: int = 2

# + setting - fixed variables
installed_packages: list = [
    pip_list_as_json["name"]
    for pip_list_as_json in ast.literal_eval(
        subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format", "json"],
            capture_output=True,
            text=True,
        ).stdout
    )
]

# + setting - tensorfow version
if tensorflow_version_as_integer == 2:
    import tensorflow as tf
    import tensorflow.keras as keras
elif tensorflow_version_as_integer == 1:
    # python <= 3.7
    if environment_location == EnvironmentLocation.GOOGLE_COLAB:
        IPython.get_ipython().run_line_magic("tensorflow_version", "1.x")
    import tensorflow as tf

    # tf.compat.v1.enable_eager_execution()
    import keras

tesnorflow_version: str = tf.__version__

# ** create temp_files folder and download required files
TEMP_FILES_PATH: Path = Path.home() / ".local_files"
TEMP_FILES_PATH.mkdir(exist_ok=True)
TEMP_FILES_TENSORFLOW_PATH: Path = TEMP_FILES_PATH / "tensorflow"
TEMP_FILES_TENSORFLOW_MOELS_PATH: Path = TEMP_FILES_TENSORFLOW_PATH / "models"
TEMP_FILES_TENSORFLOW_MOELS_RESEARCH_PATH: Path = (
    TEMP_FILES_TENSORFLOW_MOELS_PATH / "research"
)
TEMP_FILES_TENSORFLOW_PATH.mkdir(exist_ok=True)
TEMP_FILES_TENSORFLOW_MOELS_PATH.mkdir(exist_ok=True)
if not TEMP_FILES_TENSORFLOW_MOELS_RESEARCH_PATH.exists():
    subprocess.run(
        f"git clone https://github.com/tensorflow/models.git {TEMP_FILES_TENSORFLOW_MOELS_PATH}",
        capture_output=False,
        text=True,
        shell=True,
    )

if "keras-adabound" not in installed_packages:
    if environment_location == EnvironmentLocation.GOOGLE_COLAB:
        print(
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "keras-adabound"],
                capture_output=True,
                text=True,
            ).stdout
        )
    else:
        # if environment_location == LOCAL-LIKE Environment:
        # subprocess.run("pipenv install keras-adabound")
        pass

# + install library for object detction on tensorflow for each environment
if "object-detection" not in installed_packages:
    # create .py files in research/object_detection/protos directory
    subprocess.run(
        "".join(
            [
                f'protoc {str(TEMP_FILES_TENSORFLOW_MOELS_RESEARCH_PATH/"object_detection/protos")}/*.proto',
                f" --python_out={str(TEMP_FILES_TENSORFLOW_MOELS_RESEARCH_PATH)}",
                f" --proto_path={str(TEMP_FILES_TENSORFLOW_MOELS_RESEARCH_PATH)}",
            ]
        ),
        capture_output=False,
        text=True,
        shell=True,
    )

    # --proto_path={str(TENSORFLOW_MODELS_RESEARCH_PATH)}"
    if environment_location == EnvironmentLocation.GOOGLE_COLAB:
        print(
            subprocess.run(
                "python -m pip install {setup_location}".format(
                    setup_location=str(
                        TEMP_FILES_TENSORFLOW_MOELS_RESEARCH_PATH
                        / f"object_detection/packages/tf{tensorflow_version_as_integer}"
                    )
                ),
                capture_output=True,
                text=True,
                shell=True,
            ).stdout
        )

    elif environment_location == EnvironmentLocation.LOCAL_WITH_DOCKER:
        subprocess.run(
            "".join(
                [
                    f"docker build -f {TEMP_FILES_TENSORFLOW_MOELS_RESEARCH_PATH}/object_detection/dockerfiles/tf{tensorflow_version_as_integer}/Dockerfile -t od .",
                    "; docker run -it od",
                ]
            ),
            shell=True,
        )
    else:
        # if environment_location == EnvironmentLocation.LOCAL:
        print(
            subprocess.run(
                "python -m pip install {setup_location}".format(
                    setup_location=str(
                        TEMP_FILES_TENSORFLOW_MOELS_RESEARCH_PATH
                        / f"object_detection/packages/tf{tensorflow_version_as_integer}"
                    )
                ),
                capture_output=True,
                text=True,
                shell=True,
            ).stdout
        )
        # subprocess.run(
        #     "pipenv install {setup_location}".format(
        #         setup_location=str(
        #             TEMP_FILES_TENSORFLOW_MOELS_RESEARCH_PATH
        #             / f"object_detection/packages/tf{tensorflow_version_as_integer}"
        #         )
        #     ),
        #     shell=True,
        # )

# %%
#
