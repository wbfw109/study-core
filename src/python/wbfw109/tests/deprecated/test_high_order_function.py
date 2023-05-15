# %%
from __future__ import annotations

import logging
import os

from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

from collections.abc import Callable
from typing import Callable, ParamSpec, TypeVar

from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.wait import WebDriverWait


# Title: method descriptors, but recommend use with  @functools.wraps(func)
def decorator(f):
    def new_function():
        print("Extra Functionality")
        f()

    return new_function


@decorator
def initial_function():
    print("Initial Functionality")


initial_function()
# Note that if decorator first parameter is cls, it also can be accepted on Class.
# refer to built-in dataclass definition in library

# Title: in case of logging
T = TypeVar("T")
P = ParamSpec("P")


logging.basicConfig(level=logging.INFO)


def add_logging(f: Callable[P, T]) -> Callable[P, T]:
    """A type-safe decorator to add logging to a function."""

    def inner(*args: P.args, **kwargs: P.kwargs) -> T:
        logging.info(f"{f.__name__} was called")
        return f(*args, **kwargs)

    return inner


@add_logging
def add_two(x: float, y: float) -> float:
    """Add two numbers together."""
    return x + y


add_two(1, 3)


# Title: in Lambda function (deprecated)
## Setup chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ensure GUI is off
chrome_options.add_argument("--no-sandbox")
# Set path to chromedriver as per your configuration
homedir = os.path.expanduser("~")
webdriver_service = Service(f"{homedir}/chromedriver/stable/chromedriver")

# Choose Chrome Browser
web_driver: WebDriver = webdriver.Chrome(
    service=webdriver_service, options=chrome_options
)
code_jam_archive_link: str = "https://codingcompetitions.withgoogle.com/codejam/archive"
web_driver.get(code_jam_archive_link)


code_jam_start_year: int = 2008
code_jam_xpath_list: list[str] = [
    f'//*[@id="archive-card-{x}"]'
    for x in range(code_jam_start_year, datetime.date.today().year + 1)
]
find_competition_elements: Callable[
    [WebDriver, str], WebElement
] = lambda web_driver, xpath: web_driver.find_element(By.XPATH, xpath)

els: list[WebElement] = [
    WebDriverWait(web_driver, timeout=3).until(functools.partial(find_competition_elements, xpath=code_jam_xpath))  # type: ignore
    for code_jam_xpath in code_jam_xpath_list
]
