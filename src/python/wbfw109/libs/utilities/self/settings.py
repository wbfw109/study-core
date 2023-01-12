"""Package that related with project settings value"""
import json
from pathlib import Path
from typing import LiteralString

import tomllib
from wbfw109.libs.typing import (  # type: ignore
    ReferenceJsonAlgorithmDetailType,
    ReferenceJsonAlgorithmMetaType,
    ReferenceTomlType,
)


class GlobalConfig:
    PROJECT_TOML_PATH: Path = Path("ref/ref.toml")
    ALGORITHMS_JSON_META_PATH: Path = Path("ref/algorithms_meta_ref.json")
    ALGORITHMS_JSON_PATH: Path = Path("ref/algorithms_ref.json")


class GlobalConfigAlgorithms:
    INTERACTIVE_TESTING_FILE_NAME: LiteralString = "interactive_testing_tool.py"


def get_project_toml_data() -> ReferenceTomlType:
    with GlobalConfig.PROJECT_TOML_PATH.open("rb") as f:
        project_toml_data: ReferenceTomlType = tomllib.load(f)  # type: ignore
    return project_toml_data


def get_algorithms_json_meta_data() -> ReferenceJsonAlgorithmMetaType:
    """Used to crawl in batch by company and competition that exist in the meta file"""
    with GlobalConfig.ALGORITHMS_JSON_META_PATH.open("rb") as f:
        algorithms_json_meta_data: ReferenceJsonAlgorithmMetaType = json.load(f)
    return algorithms_json_meta_data


def get_algorithms_json_data() -> dict[
    str, dict[str, list[ReferenceJsonAlgorithmDetailType]]
]:
    """
    Returns:
        dict[ str, dict[ str, list[ReferenceJsonAlgorithmDetailType] ] ]
        : dict[ <company>, dict[ <competition_title>, list[ReferenceJsonAlgorithmDetailType] ] ]
    """
    with GlobalConfig.ALGORITHMS_JSON_PATH.open("rb") as f:
        algorithms_json_data: dict[
            str, dict[str, list[ReferenceJsonAlgorithmDetailType]]
        ] = json.load(f)
    return algorithms_json_data
