import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


# Writtin at 📅 2024-09-15 20:26:01
def create_temp_json_file(
    data: Any, prefix: str = "cheat_sheet-cpp", suffix: str = ".json"
) -> Path:
    """
    Create a temporary JSON file with a specified prefix and suffix.

    Parameters:
        data (dict): The data to write to the JSON file.
        prefix (str): The prefix for the temporary file name (default is 'cheat_sheet-cpp').
        suffix (str): The suffix for the temporary file name (default is '.json').

    Returns:
        Path: The full path to the temporary file as a Path object.

    Example:
        >>> data = {"key": "value"}
        >>> temp_json_file = create_temp_json_file(data)
        >>> print(temp_json_file)
        PosixPath('/tmp/cheat_sheet-cpp12345.json')
    """
    # Create a temporary file with the specified prefix and suffix
    with tempfile.NamedTemporaryFile(
        delete=False, prefix=prefix, suffix=suffix, mode="w", encoding="utf-8"
    ) as temp_file:
        json.dump(data, temp_file, indent=2)
        temp_file_name = temp_file.name

    return Path(temp_file_name)


def create_temp_str_file(data: str, prefix: str = "", suffix: str = ".txt") -> Path:
    """
    Create a temporary file with a specified prefix and suffix to store a string.

    Parameters:
        data (str): The string data to write to the file.
        prefix (str): The prefix for the temporary file name (default is 'cheat_sheet-cpp').
        suffix (str): The suffix for the temporary file name (default is '.txt').

    Returns:
        Path: The full path to the temporary file as a Path object.

    Example:
        >>> data = "This is some string data."
        >>> temp_str_file = create_temp_str_file(data)
        >>> print(temp_str_file)
        PosixPath('/tmp/cheat_sheet-cpp12345.txt')
    """
    # Create a temporary file with the specified prefix and suffix
    with tempfile.NamedTemporaryFile(
        delete=False, prefix=prefix, suffix=suffix, mode="w", encoding="utf-8"
    ) as temp_file:
        temp_file.write(data)  # Write the string data to the file
        temp_file_name = temp_file.name

    return Path(temp_file_name)


# Writtin at 📅 2024-09-15 20:26:01
def open_file_in_vscode(file_path: Path) -> None:
    """
    Open a JSON file in VS Code using the 'code' command.

    Parameters:
        file_path (Path): Path to the JSON file to be opened as a Path object.

    Example:
        >>> temp_json_file = Path("/tmp/cheat_sheet-cpp12345.json")
        >>> open_json_in_vscode(temp_json_file)
        Opened /tmp/cheat_sheet-cpp12345.json in VS Code.
    """
    try:
        # Convert Path to str for subprocess
        subprocess.run(["code", str(file_path)], check=True)
        print(f"Opened {file_path} in VS Code.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to open {file_path} in VS Code: {e}")
