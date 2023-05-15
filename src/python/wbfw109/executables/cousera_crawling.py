# %%
from __future__ import annotations

import itertools
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# %doctest_mode

# from google.protobuf.internal import api_implementation
# print(api_implementation.Type())


# %% # title: Declaration and Crawled tab

import logging
import urllib.parse
from datetime import datetime
from typing import Literal, Optional

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from wbfw109.libs.path import get_project_root

crawling_range_mode: Optional[str] = "auto"  # [None, "auto", "re-download_warning_log"]
# crawling_range.start_dt = datetime.fromisoformat("2022-11-24 23:25:03+09:00")
# crawling_range.end_dt = datetime.fromisoformat("2022-11-24 23:25:05+09:00")
logging_file: Path = get_project_root() / "logs/cousera_crawling.log"
logging.basicConfig(
    filename=logging_file,
    level=logging.INFO,  # DEBUG level is not logged into the file.
)
log = logging.getLogger(__name__)

crawled_file: Path = get_project_root() / "rsrc/cousera_crawling.txt"

# e.g.: query_url = "https://www.coursera.org/search?query=Security&productTypeDescription=Professional%20Certificates&subtitleLanguage=English&language=English&sortBy=NEW"
chrome_options = Options()
# chrome_options.add_argument("--incognito")
# 123.0.6312.58 chromdriver-linux ; https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json
chrome_service = Service(str(get_project_root() / "rsrc/chromedriver"))

web_driver = webdriver.Chrome(options=chrome_options, service=chrome_service)
web_driver.set_page_load_timeout(10)
timeout_retry_count: int = 10
wait = WebDriverWait(web_driver, 10)


class SearchingEndException(Exception):
    """It is occurred When
    - no longer clickable page moving button exists.
    - datetime of post not meets the specified range.
    """


# %% # title: run Web
base_search_url = "https://www.coursera.org/search?"
params = {
    "query": "Business Intelligence",
    "productTypeDescription": "Professional Certificates",
    "subtitleLanguage": "English",
    "language": "English",
    "sortBy": "NEW",
}  # note: my required query = [Security or Cloud, Business Intelligence]
query_url = base_search_url + urllib.parse.urlencode(
    params, quote_via=urllib.parse.quote
)

print(f"query text: {params['query']}")

web_driver.get(query_url)
web_driver.switch_to.window(web_driver.window_handles[0])


# %%
# ??? Ubuntu Chrome browser's behavior is different with Windows Chrome. ??? suddenly re changed to button types..
next_btn_xpath = '//*[@id="rendered-content"]/div/div/main/div[2]/div/div/div/div/div[2]/div[5]/div/nav/ul/li[last()]/button'
first_page_posts_xpath = (
    r"//*[contains(@id, 'cds-react-aria-') and contains(@id, '-product-card-title')]"
)
post_title_xpath = (
    '//*[@id="rendered-content"]/div/main/section[2]/div/div/div[1]/div[1]/section/h1'
)
post_level_xpath = '//*[@id="rendered-content"]/div/main/section[2]/div/div/div[2]/div/div/section/div[2]/div[2]/div[1]'
post_script_text_xpath = "/html/body/script[1]"


def get_scrolled_posts_xpath(
    position: int = 0, relational_opr: Literal[">", "<", ">=", "<=", "="] = ">"
) -> str:
    """infinite_scroll_posts_xpath
    e.g. /html/body/div[2]/div/div/main/div[3]/div/div/div/div/div[2]/ul[2]/li[position() > 0]//*[contains(@id, 'cds-react-aria-') and contains(@id, '-product-card-title')]
    """
    return (
        rf"/html/body/div[2]/div/div/main/div[3]/div/div/div/div/div[2]/ul[2]/li[position() {relational_opr} {position}]"
        + first_page_posts_xpath
    )


def crawl_post(web_driver: WebDriver, elem: WebElement):
    """Run a post"""

    crawled_data: list[str] = []
    try:
        post_href: str = elem.get_attribute("href")
        print(post_href)

        web_driver.switch_to.new_window("tab")
        web_driver.get(post_href)

        # load all contents by scroll. Note that too fast scroll can not load items.
        web_driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        post_title = web_driver.find_element(By.XPATH, post_title_xpath).text
        crawled_data.append(
            "".join(
                [
                    "⚓ ",
                    post_title,
                    " ; ",
                    post_href,
                ]
            )
        )

        post_level = web_driver.find_element(By.XPATH, post_level_xpath).text
        crawled_data.append("".join(["  🎚️ Level: ", post_level.replace(" level", "")]))

        script_elem = wait.until(
            EC.presence_of_all_elements_located((By.XPATH, post_script_text_xpath))
        )[0]
        script_text: str = script_elem.get_attribute("innerHTML")
        start_idx: int = script_text.find('"launchedAt":') + len('"launchedAt":')
        end_idx: int = script_text.find(",", start_idx)
        post_upload_unix_dt: int = int(script_text[start_idx:end_idx])
        post_upload_dt = datetime.fromtimestamp(
            post_upload_unix_dt / 1000
        ).isoformat()  # ignore millisecond
        crawled_data.append("".join(["  📅 Launched At: ", post_upload_dt]))
        crawled_data.append("")

        # write file
        with crawled_file.open("a+") as f:
            f.write("\n".join(crawled_data))
    except NoSuchElementException as e:
        raise NoSuchElementException from e
    except Exception as e:
        print(f"An error occured: {e}")
    finally:
        # close a post
        web_driver.close()
        web_driver.switch_to.window(web_driver.window_handles[0])


try:
    counter = itertools.count(1)

    # loop initial posts
    elems = wait.until(
        EC.presence_of_all_elements_located((By.XPATH, first_page_posts_xpath))
    )

    for elem in elems:
        print(f"running: {next(counter)}th: ", sep="")
        crawl_post(web_driver, elem)

    # if page types
    try:
        while True:
            web_driver.find_element(By.XPATH, next_btn_xpath).click()

            elems = wait.until(
                EC.presence_of_all_elements_located((By.XPATH, first_page_posts_xpath))
            )
            for elem in elems:
                print(f"running: {next(counter)}th: ", sep="")
                crawl_post(web_driver, elem)
    except NoSuchElementException as exc:
        raise SearchingEndException from exc

    #! 뭔가 이상하다. infinite scrolle 방식으로 로드 될 때도 있꼬, Show More 버튼 방식 또는 페이지 방식으로 로드될 때도 있음
    #! Inprivate 브라우저를 로드하거나 그냥 사용하거나 바꿔가면서 페이지 방식으로 로드 될 떄.. 이 코드를 불러오자..
    #! 거기다 infinite scroll 방식은 분명 DOM 상에는 잘 불러와져서 F12 에서 XPATH 는 잘 찾는데, 셀레니움은 찾지 못함.
    # # if infinite scrolled types
    # scrolled_post_count = 0
    # while True:
    #     # load all contents by scroll. Note that scroll to 0 can not load items.
    #     web_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
    #     web_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
    #     web_driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    #     time.sleep(3)

    #     # TODO: if no "next" button, break page loop .. can not load...
    #     try:
    #         # ? that all codntions of wait.until can not recognize dynamic loaded content.
    #         # check wehther items loaded
    #         wait.until(
    #             EC.element_to_be_clickable((By.XPATH, get_scrolled_posts_xpath()))
    #         )

    #         elems: list[WebElement] = web_driver.find_elements(
    #             By.XPATH, get_scrolled_posts_xpath()
    #         )
    #         for elem in elems:
    #             print(f"running: {next(counter)}th: ", sep="")
    #             crawl_post(web_driver, elem)

    #         scrolled_post_count += len(elems)
    #     except NoSuchElementException as exc:
    #         raise SearchingEndException from exc
except SearchingEndException:
    print("exit")

# %%
