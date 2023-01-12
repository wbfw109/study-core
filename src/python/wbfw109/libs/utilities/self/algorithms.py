"""
This is a library for crawling and parent class of src/wbfw109/algorithms.
# ToDo: migration to msgspec or msgpack? if larger
"""
import dataclasses
import functools
import itertools
import json
import pprint  # type: ignore
import time
import unittest
from abc import ABC, abstractmethod
from http import HTTPStatus
from http.client import HTTPResponse
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Final, Generator, Generic, Optional, TypeVar
from urllib.parse import ParseResult, urlparse
from urllib.request import urlopen
from zipfile import ZipFile

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.wait import WebDriverWait
from wbfw109.libs.objects.object import (  # type: ignore
    get_default_value_map,
    initialize_not_constant_fields,
)
from wbfw109.libs.string import rename_to_snake_case_with_replace  # type: ignore
from wbfw109.libs.typing import (  # type: ignore
    DST,
    ReferenceJsonAlgorithmDetailType,
    ReferenceJsonAlgorithmMetaType,
    ReferenceTomlType,
)
from wbfw109.libs.utilities.self.settings import (  # type: ignore
    GlobalConfig,
    GlobalConfigAlgorithms,
    get_algorithms_json_data,
    get_project_toml_data,
)


def get_algorithm_target_dir(
    project_settings: ReferenceTomlType,
    algorithm_meta: ReferenceJsonAlgorithmMetaType,
    algorithm_detail_data: ReferenceJsonAlgorithmDetailType,
) -> Path:
    return Path(
        "/".join(
            [
                project_settings["project"]["resources_root"],
                *[
                    algorithm_meta[path_name]
                    for path_name in project_settings["algorithms"]["parent_path_order"]
                ],
                *[
                    algorithm_detail_data[path_name]
                    for path_name in project_settings["algorithms"][
                        "problem_path_order"
                    ]
                ],
            ]
        )
    )


def synchronize_algorithms_resources(
    algorithm_meta: ReferenceJsonAlgorithmMetaType = ReferenceJsonAlgorithmMetaType(
        company="google", contest="code_jam"
    ),
    *,
    crawled_data: Optional[list[ReferenceJsonAlgorithmDetailType]] = None,
) -> None:
    """
    if crawled_data exists, update that and synchronize.
    """
    project_settings = get_project_toml_data()
    algorithms_json_data = get_algorithms_json_data()

    # validate
    if algorithm_meta["company"] not in algorithms_json_data.keys():
        algorithms_json_data[algorithm_meta["company"]] = {
            algorithm_meta["contest"]: []
        }

    algorithm_detail_data_list = algorithms_json_data[algorithm_meta["company"]][
        algorithm_meta["contest"]
    ]

    if crawled_data:
        # for operation to remove duplicated elements
        for algorithm_detail_data in algorithm_detail_data_list:
            algorithm_detail_data["is_in_resources"] = False
        for algorithm_detail_data in crawled_data:
            algorithm_detail_data["is_in_resources"] = False

        algorithm_detail_data_list.extend(crawled_data)
        algorithm_detail_data_list = list(
            {
                frozenset(item.items()): item for item in algorithm_detail_data_list
            }.values()
        )
        # python is call by sharing. so require re-assignment code.
        algorithms_json_data[algorithm_meta["company"]][
            algorithm_meta["contest"]
        ] = algorithm_detail_data_list

    for algorithm_detail_data in algorithm_detail_data_list:
        target_dir = get_algorithm_target_dir(
            project_settings, algorithm_meta, algorithm_detail_data
        )
        if target_dir.exists() and list(target_dir.iterdir()):
            algorithm_detail_data["is_in_resources"] = True
        else:
            algorithm_detail_data["is_in_resources"] = False

    with GlobalConfig.ALGORITHMS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(algorithms_json_data, f)


def download_algorithms_resources(
    algorithm_meta: ReferenceJsonAlgorithmMetaType = ReferenceJsonAlgorithmMetaType(
        company="google", contest="code_jam"
    ),
    *,
    only_season: Optional[str] = "2022",
) -> None:
    """
    if only_season not exists, download all season.
    """
    project_settings = get_project_toml_data()
    algorithms_json_data = get_algorithms_json_data()
    is_modified: bool = False
    algorithm_detail_data_list = algorithms_json_data[algorithm_meta["company"]][
        algorithm_meta["contest"]
    ]

    target_dir: Path = Path()
    try:
        for algorithm_detail_data in algorithm_detail_data_list:
            if only_season and only_season != algorithm_detail_data["season"]:
                continue
            target_dir = get_algorithm_target_dir(
                project_settings, algorithm_meta, algorithm_detail_data
            )
            if (
                target_dir.exists()
                or algorithm_detail_data["is_in_resources"]
                or algorithm_detail_data["dataset_hyperlink"] == ""
            ):
                continue

            # check HTTPResponse and validate
            parse_result: ParseResult = urlparse(
                algorithm_detail_data["dataset_hyperlink"]
            )
            parsed_url_path: Path = Path(parse_result.path)
            if not parse_result.scheme.startswith("http"):
                raise Exception("Request scheme is not starts with 'http'")
            response: HTTPResponse = urlopen(  # nosec
                algorithm_detail_data["dataset_hyperlink"]
            )
            if response.status != HTTPStatus.OK:
                print(f"[Pass] {parsed_url_path}. response.status is not HTTPStatus.OK")
                continue
            target_dir.mkdir(parents=True, exist_ok=True)

            # download
            if parsed_url_path.suffix == ".zip":
                ZipFile(BytesIO(response.read())).extractall(target_dir)
            elif parsed_url_path.suffix.startswith(".py"):
                with (
                    target_dir / GlobalConfigAlgorithms.INTERACTIVE_TESTING_FILE_NAME
                ).open("wb") as binary_file:
                    binary_file.write(response.read())
            algorithm_detail_data["is_in_resources"] = True
            is_modified = True
            print(f"download complete: {target_dir}")
    except Exception as e:
        print(e)
        print(f"[Error] Stop download: {target_dir}")
    finally:
        if is_modified:
            with GlobalConfig.ALGORITHMS_JSON_PATH.open("w", encoding="utf-8") as f:
                json.dump(algorithms_json_data, f)


@dataclasses.dataclass
class CodeJamCommonExtraction:
    text: str = dataclasses.field(default="")
    hyperlink: str = dataclasses.field(default="")


def crawl_algorithms_code_jam(
    *,
    only_season: Optional[str] = "2022",
) -> None:
    """
    if only_season not exists, navigate all season.

    📰 before 2021, test set link does not exists... so.. valid season are 2022, 2021.
    and some valid exists until 2018.
    The interactive runner was changed after the 2019 contest.


    - dataset_hyperlink
        when copy get link "?dl=1" parameter, you will get only test data set.
        but if you truncate "?dl=1", you can get also sample data set.
        in the case it is python file, just you can get text code.
        It may be none, value will be ""

    """
    # Title: preprocess
    # Setup chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # type: ignore    # ensure GUI off
    chrome_options.add_argument("--no-sandbox")  # type: ignore   # in GUI is off
    # set path to chrome diver as per your configuration
    webdriver_service = Service(f"{Path.home()}/chromedriver/stable/chromedriver")
    # choose Chrome Browser
    web_driver: WebDriver = webdriver.Chrome(
        service=webdriver_service, options=chrome_options
    )

    project_settings = get_project_toml_data()
    algorithm_meta: ReferenceJsonAlgorithmMetaType = ReferenceJsonAlgorithmMetaType(
        company="google", contest="code_jam"
    )

    find_competition_elements: Callable[
        [WebDriver, str], list[WebElement]
    ] = lambda web_driver, xpath: web_driver.find_elements(By.XPATH, xpath)
    find_competition_element: Callable[
        [WebDriver, str], WebElement
    ] = lambda web_driver, xpath: web_driver.find_element(By.XPATH, xpath)
    ROOT_LINK: str = "https://codingcompetitions.withgoogle.com"
    problem_detail_info_list: list[ReferenceJsonAlgorithmDetailType] = []

    try:
        # step 1; navigate code_jam season in archive
        web_driver.get("https://codingcompetitions.withgoogle.com/codejam/archive")
        SEASON_XPATH: str = '//a[starts-with(@id,"archive-card-")]'
        season_element_list: list[WebElement] = WebDriverWait(web_driver, timeout=4).until(functools.partial(find_competition_elements, xpath=SEASON_XPATH))  # type: ignore
        season_common_extraction_list: list[CodeJamCommonExtraction] = []
        for e in season_element_list:
            season_common_extraction_list.append(
                CodeJamCommonExtraction(
                    text=e.find_element(By.XPATH, ".//div[contains(@class, 'card-body')]/div[1]/p[1]").get_attribute("textContent").strip().replace("Code Jam ", ""),  # type: ignore
                    hyperlink=e.get_attribute("href"),  # type: ignore
                )
            )

        # step 2; navigate code_jam round (from 2022 season)
        for season_info in season_common_extraction_list:
            if only_season and only_season != season_info.text:
                continue

            web_driver.get(season_info.hyperlink)
            ROUND_XPATH: str = '//a[starts-with(@id,"archive-view-cta-") and contains(@class, "mobile")]'
            round_common_extraction_list: list[CodeJamCommonExtraction] = []
            round_element_list: list[WebElement] = WebDriverWait(web_driver, timeout=4).until(functools.partial(find_competition_elements, xpath=ROUND_XPATH))  # type: ignore
            for e in round_element_list:
                round_common_extraction_list.append(
                    CodeJamCommonExtraction(
                        text=e.find_element(By.XPATH, "./preceding-sibling::span").get_attribute("textContent").strip().replace(" Round 2022", ""),  # type: ignore
                        hyperlink=e.get_attribute("href"),  # type: ignore
                    )
                )
            # step 3; navigate code_jam problem (from qualification round)
            for round_info in reversed(round_common_extraction_list):
                web_driver.get(round_info.hyperlink)
                current_relative_link: str = round_info.hyperlink.replace(ROOT_LINK, "")
                PROBLEM_XPATH: str = (
                    f'//a[starts-with(@href,"{current_relative_link}")]'
                )
                problem_common_extraction_list: list[CodeJamCommonExtraction] = []
                problem_element_list: list[WebElement] = WebDriverWait(web_driver, timeout=4).until(functools.partial(find_competition_elements, xpath=PROBLEM_XPATH))  # type: ignore
                for e in problem_element_list:
                    problem_common_extraction_list.append(
                        CodeJamCommonExtraction(
                            text=e.find_element(By.XPATH, "./preceding-sibling::p").get_attribute("textContent").strip(),  # type: ignore
                            hyperlink=e.get_attribute("href") + "#analysis",  # type: ignore
                        )
                    )
                # step 4; navigate code_jam test dataset of problem
                for problem_info in problem_common_extraction_list:
                    web_driver.get(problem_info.hyperlink)
                    TEST_DATASET_XPATH: str = (
                        '//div[@class="test-data-download-header-download-button"]/a'
                    )
                    LOCAL_TESTING_TOOL_XPATH: str = (
                        '//a[contains(@download, "testing_tool")]'
                    )
                    PROBLEM_TAB_ELEMENT_XPATH: str = '//span[contains(@class, "mdc-tab__text-label") and contains(text(), "Problem")]/../..'
                    test_dataset_hyperlink: str = ""
                    complete_msg_prefix: str = "crawling complete: "
                    try:
                        # find test dataset.zip
                        test_dataset_element: WebElement = WebDriverWait(web_driver, timeout=4).until(functools.partial(find_competition_element, xpath=TEST_DATASET_XPATH))  # type: ignore
                        test_dataset_hyperlink = test_dataset_element.get_attribute("href").replace("?dl=1", "")  # type: ignore
                    except TimeoutException:
                        try:
                            # find local testing tool .py
                            problem_tab_element: WebElement = WebDriverWait(web_driver, timeout=2).until(functools.partial(find_competition_element, xpath=PROBLEM_TAB_ELEMENT_XPATH))  # type: ignore
                            problem_tab_element.click()
                            local_testing_tool_element: WebElement = WebDriverWait(web_driver, timeout=2).until(functools.partial(find_competition_element, xpath=LOCAL_TESTING_TOOL_XPATH))  # type: ignore
                            test_dataset_hyperlink = local_testing_tool_element.get_attribute("href").replace("?dl=1", "")  # type: ignore
                        except TimeoutException:
                            complete_msg_prefix = "[No Dataset] "

                    algorithm_detail_data: ReferenceJsonAlgorithmDetailType = (
                        ReferenceJsonAlgorithmDetailType(
                            season=rename_to_snake_case_with_replace(season_info.text),
                            round=rename_to_snake_case_with_replace(round_info.text),
                            name=rename_to_snake_case_with_replace(problem_info.text),
                            dataset_hyperlink=test_dataset_hyperlink,
                            is_in_resources=False,
                        )
                    )
                    problem_detail_info_list.append(algorithm_detail_data)
                    target_dir = get_algorithm_target_dir(
                        project_settings, algorithm_meta, algorithm_detail_data
                    )
                    print(f"{complete_msg_prefix}{target_dir}")
    except Exception as e:
        print(e)
        print("[Error] Stop Crawling")
    finally:
        web_driver.quit()
        synchronize_algorithms_resources(
            algorithm_meta, crawled_data=problem_detail_info_list
        )


@dataclasses.dataclass
class TestDataset:
    input_list: list[Path] = dataclasses.field(default_factory=list)
    output_list: list[Path] = dataclasses.field(default_factory=list)


def get_algorithms_resources_path(file_path: str) -> Path:
    """
    📝 Note that you must pass <file_path> as __file__ in root caller
    it's not working in interactive jupyter file.

    Returns:
        Path: resources_path of <file_path>
    """
    project_toml_data = get_project_toml_data()
    return Path(project_toml_data["project"]["resources_root"]) / Path(
        file_path
    ).relative_to(Path.cwd()).with_suffix("").relative_to(
        "/".join(
            [
                project_toml_data["project"]["src_root"],
                project_toml_data["algorithms"]["python_packages_path"],
            ]
        )
    )


def get_test_dataset(
    target_path: Path,
    name: str,
    *,
    only_sample: bool = True,
    should_not_empty: bool = True,
) -> TestDataset:
    """
    📝 Note that test_dataset includes sample_dataset
    if only_sample is False, returns all test dataset including sample.

    Args:
        target_parent_path (Path): parent path of problem name.
        name (str): problem name.
        only_sample (bool, optional): if the value is False, returns all test dataset including sample. Defaults to True.
        should_not_empty (bool, optional): if the value is True, check wether files are valid or not.
            if empty, raise Exceptions. Defaults to True.

    Returns:
        TestDataset: test dataset
    """
    resource_full_path: Path = target_path / name
    test_dataset = TestDataset()
    distinguish_input_and_output: Callable[[TestDataset, Path], None] = (
        lambda test_dataset, target_path: test_dataset.input_list.append(target_path)
        if "input" in target_path.name
        else test_dataset.output_list.append(target_path)
    )

    test_dataset_path_list: list[Path] = sorted(resource_full_path.glob("**/*.txt"))

    if only_sample:
        test_dataset_path_list = list(
            itertools.compress(
                test_dataset_path_list,
                [True if "sample" in x.name else False for x in test_dataset_path_list],
            )
        )

    # to locate sample order in first.
    for target_path in test_dataset_path_list:
        if "sample" in target_path.name:
            distinguish_input_and_output(test_dataset, target_path)
    if not only_sample:
        for target_path in test_dataset_path_list:
            if "sample" not in target_path.name:
                distinguish_input_and_output(test_dataset, target_path)
    if should_not_empty:
        if not test_dataset.input_list or not test_dataset.output_list:
            raise Exception("No TestDataset files.")
        elif len(test_dataset.input_list) != len(test_dataset.input_list):
            raise Exception("TestDataset not have pairs")
    return test_dataset


class AbstractProblemSolution(ABC, Generic[DST]):
    """Root class of ProblemSolution"""

    @abstractmethod
    def __init__(
        self,
        dst_bundles_list: list[list[DST]],
        test_case_number: int = 0,
        OUTPUT_DELIMITER: str = "Case #",
        BUNDLE_NUMBER_IN_ONE_CASE: int = 1,
        PROBLEM_RESOURCE_NAME: str = "problem",
        RUN_FILE_PATH: str = "__file__",
    ) -> None:
        ...

    @property
    @abstractmethod
    def dst_bundles_list(self) -> list[list[DST]]:
        ...

    @dst_bundles_list.setter
    @abstractmethod
    def dst_bundles_list(self, value: list[list[DST]]) -> None:
        ...

    @property
    @abstractmethod
    def test_case_number(self) -> int:
        ...

    @test_case_number.setter
    @abstractmethod
    def test_case_number(self, value: int) -> None:
        ...

    # if test_case is None input from in keyboard (standard input), else input from sample file
    @abstractmethod
    def input_case(self, test_case: Optional[str] = None) -> None:
        ...

    @abstractmethod
    def input_case_bundle(self, get_contents: Callable[[], str]) -> list[DST]:
        """📝 require override"""

    @staticmethod
    @abstractmethod
    def get_solution(dst_bundles: list[DST]) -> str:
        """📝 require override. <solution method name>
        end character of end line must be "\n" """

    @abstractmethod
    def output_case(self) -> list[str]:
        ...

    @abstractmethod
    def judge_acceptance(
        self, my_output: str, file_output: str, *, bundles_index: int
    ) -> None:
        """📝[Optional] override. default is AssertEqual"""

    @classmethod
    @abstractmethod
    def test_dataset(cls, /, *, test_level: int) -> None:
        ...

    @classmethod
    @abstractmethod
    def main(cls, /, *, test_level: int = 0) -> None:
        ...


class GoogleCodeJamProblemSolution(AbstractProblemSolution[DST]):
    """📝~ Note that some function must be override in subclasses.

    When it subclassing to prevent problem name starts with number, convert start string to english.

    - Required:
        input_case_bundle, get_solution
    - Optional:
        judge_acceptance
    """

    def __init__(
        self,
        test_case_number: int = 0,
        OUTPUT_DELIMITER: str = "Case #",
        BUNDLE_NUMBER_IN_ONE_CASE: int = 1,
        PROBLEM_RESOURCE_NAME: str = "problem",
        RUN_FILE_PATH: str = "__file__",
    ) -> None:
        self._dst_bundles_list: list[list[DST]] = []
        self._test_case_number: int = test_case_number
        self.OUTPUT_DELIMITER: Final[str] = OUTPUT_DELIMITER
        self.BUNDLE_NUMBER_IN_ONE_CASE: Final[int] = BUNDLE_NUMBER_IN_ONE_CASE
        self.PROBLEM_RESOURCE_NAME: Final[str] = PROBLEM_RESOURCE_NAME
        self.RUN_FILE_PATH: Final[str] = RUN_FILE_PATH

        self.DEFAULT_VALUE_MAP: Final[dict[str, Any]] = get_default_value_map()

    @property
    def dst_bundles_list(self) -> list[list[DST]]:
        return self._dst_bundles_list

    @dst_bundles_list.setter
    def dst_bundles_list(self, value: list[list[DST]]) -> None:
        self._dst_bundles_list = value

    @property
    def test_case_number(self) -> int:
        return self._test_case_number

    @test_case_number.setter
    def test_case_number(self, value: int) -> None:
        self._test_case_number = value

    # if test_case is None input from in keyboard (standard input), else input from sample file
    def input_case(self, test_case: Optional[str] = None) -> None:
        # define get_contents by whether test_case exists.
        if test_case:
            contents_generator: Generator[str, None, None] = (
                line for line in test_case.split("\n")
            )

            def get_contents() -> str:
                return next(contents_generator)

        else:
            get_contents = input

        self.test_case_number = int(get_contents())
        for _ in range(self.test_case_number):
            self.dst_bundles_list.append(self.input_case_bundle(get_contents))

    def input_case_bundle(self, get_contents: Callable[[], str]) -> list[DST]:
        """📝 Require override"""
        dst_list: list[DST] = []
        for _ in range(self.BUNDLE_NUMBER_IN_ONE_CASE):
            ...
            # one_input_length: int = 4
            # dst_list.append(
            #     DST( # type: ignore
            #         *map(
            #             int,
            #             get_contents().split(" ")[:one_input_length],
            #         )
            #     )
            # )
        return dst_list

    @staticmethod
    def get_solution(dst_bundles: list[DST]) -> str:
        """📝 Require override. <solution method name>

        Note that "\\n" character automatically will be added end of output. so, not add manually new line character."""
        return ""

    def output_case(self) -> list[str]:
        return [
            f"{self.OUTPUT_DELIMITER}{count}:" + self.get_solution(dst_bundles) + "\n"
            for count, dst_bundles in enumerate(self.dst_bundles_list, start=1)
        ]

    def judge_acceptance(
        self, my_output: str, file_output: str, *, bundles_index: int
    ) -> None:
        """📝[Optional] override. default is AssertEqual"""
        unittest.TestCase().assertEqual(my_output, file_output)

    @classmethod
    def test_dataset(cls, /, *, test_level: int) -> None:
        problem_solution = cls()
        test_dataset: TestDataset = get_test_dataset(
            target_path=get_algorithms_resources_path(problem_solution.RUN_FILE_PATH),
            name=problem_solution.PROBLEM_RESOURCE_NAME,
            only_sample=not bool(test_level - 1),
            should_not_empty=True,
        )
        start_time: float = time.time()
        for input_path, output_path in zip(
            test_dataset.input_list, test_dataset.output_list
        ):
            with input_path.open("r") as f:
                problem_solution.input_case(f.read())
            with output_path.open("r") as f:
                file_full_output = [
                    problem_solution.OUTPUT_DELIMITER + e
                    for e in f.read().split(problem_solution.OUTPUT_DELIMITER)[1:]
                ]
            print(f"validating one dataset...: {input_path.name}, {output_path.name}")
            my_full_output: list[str] = problem_solution.output_case()
            for i, (my_output, file_output) in enumerate(
                zip(my_full_output, file_full_output)
            ):
                problem_solution.judge_acceptance(
                    my_output, file_output, bundles_index=i
                )
            initialize_not_constant_fields(
                problem_solution, default_map=problem_solution.DEFAULT_VALUE_MAP
            )

        print(f"elapsed time: {time.time()- start_time}.")

    @classmethod
    def main(cls, /, *, test_level: int = 0) -> None:
        """
        test_level
        0: not test. from input standard input (keyboard).
        1: only test sample dataset.
        2: all test dataset including sample.
        """
        if test_level > 0:
            cls.test_dataset(test_level=test_level)
        else:
            problem_solution = cls()
            problem_solution.input_case()
            print(problem_solution.output_case())
