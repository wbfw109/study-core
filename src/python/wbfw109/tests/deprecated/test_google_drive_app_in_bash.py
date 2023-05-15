#%%
import subprocess
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell

# ** setting
# allow multiple print
InteractiveShell.ast_node_interactivity = "all"
"""
Note: To use Windows' Google Drive app in WSL, you need to use a symbolic link or replace all " " character with "\ " of Path string
- The name of the My Drive folder in "/mnt/c/Users/wbfw109/MyDrive/" in Windows cannot be changed.
    and, on Linux it doesn't matter because the mount will be in the form "/content/drive/MyDrive/". 
"""
# *** use a symbolic link
GOOGLE_DRIVE_APP_PATH = Path("/mnt/c/Users/wbfw109/MyDrive")
GOOGLE_DRIVE_APP_SYMBOLIC_LINK_Path: Path = Path.home() / "my_google_drive_app"
if not GOOGLE_DRIVE_APP_SYMBOLIC_LINK_Path.exists():
    result = subprocess.Popen(
        f"ln -s '{str(GOOGLE_DRIVE_APP_PATH)}' {str(GOOGLE_DRIVE_APP_SYMBOLIC_LINK_Path)}",
        shell=True,
        stdout=subprocess.PIPE,
    ).communicate()
    print("create symbolic link for Your Google Drive App")
GOOGLE_DRIVE_APP_SYMBOLIC_LINK_Path

# get string using call_root_classes()
call_root_classes(
    f"ls {str(GOOGLE_DRIVE_APP_SYMBOLIC_LINK_Path)}/Colab_Notebooks/tensorflow/models/research/object_detection/protos/*.proto",
    shell=True,
)

# *** replace all " " character with "\ " of Path string
# get bytes using subprocess.check_output()
subprocess.check_output(
    "ls /mnt/g/My\ Drive/Colab_Notebooks/tensorflow/models/research/object_detection/protos/*.proto",
    shell=True,
)


def get_raw_path_to_use_bash(path: Path) -> str:
    return f"{str(path)}".replace(" ", "\ ")


TENSORFLOW_MODELS_RESEARCH_PATH: Path = Path(
    "/mnt/c/Users/wbfw109/MyDrive/Colab_Notebooks/tensorflow/models/research"
)

subprocess.check_output(
    f"ls {get_raw_path_to_use_bash(TENSORFLOW_MODELS_RESEARCH_PATH)}/object_detection/protos/*.proto",
    shell=True,
)

# ** deprecated experiement
# 인수를 나누어 사용하는 함수를 사용g, 띄어쓰기는 인식이 되나 bash 에 의해서 eval 되지 않아서 파일이나 디렉토리 없다는 에러가 발생한다.
result = subprocess.Popen(
    [
        "eval" "ls",
        "/mnt/c/Users/wbfw109/MyDrive/Colab_Notebooks/tensorflow/models/research/object_detection/protos/*.proto",
    ],
    shell=True,
    stdout=subprocess.PIPE,
).communicate()

# 인수를 함께 사용하는 함수를 사용하면, 띄어쓰기가 된 /mnt/c/Users/wbfw109/MyDrive/Colab_Notebooks/t 부분이 두 개의 경로로 나뉘어 해석되어 오류가 발생한다. 따옴표를 붙여서 묶어주면 이전 오류는 발생하지는 않으나 *.proto" 부분에서 또 다시 오류가 발생한다.
subprocess.check_output(
    'eval "ls /mnt/c/Users/wbfw109/MyDrive/Colab_Notebooks/tensorflow/models/research/object_detection/protos/*.proto"',
    shell=True,
)
