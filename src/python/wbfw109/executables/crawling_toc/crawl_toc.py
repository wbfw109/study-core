"""
Written at 📅 2024-09-28 19:20:16
🌪️ TODO:
    - crawl with dl, dt, dd in Python (https://docs.python.org/3/using/cmdline.html#cmdoption-c)
        add TableOfContentsType.<??> or .. add when MAIN_CONTENT_AREA_TOC
    - MSDN crawling; dynamic call html element by click. (not one-time. multiple-times click is required for new recieved html elements.)
    - OpenVINO ; https://docs.openvino.ai/2024/index.html

This script extracts Table of Contents (ToC) from various documentation websites using Playwright.
It supports multiple websites and provides a generalized approach to parsing ToC based on URL patterns.

Supported sites:
- Python (https://docs.python.org)
- Yocto Project (https://docs.yoctoproject.org)
- Numpy (https://numpy.org)
- Pandas (https://pandas.pydata.org)
- OpenCV (https://docs.opencv.org)
- PyTorch (https://pytorch.org)
- TensorFlow (https://www.tensorflow.org)

Usage:
    You can pass the URL of the documentation page and specify the type of ToC (Main content or Navigation).
    The appropriate ToC selectors are determined based on the URL.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urljoin

from playwright.async_api import ElementHandle, async_playwright
from wbfw109.libs.crawling.playwright_utils import extract_direct_text, get_tag_name
from wbfw109.libs.file import create_temp_str_file, open_file_in_vscode

INDENT_BASE_UNIT = "  "


class TableOfContentsType(Enum):
    MAIN_CONTENT_AREA_TOC = 1
    SECTION_NAVIGIATON_TOC = 2


@dataclass
class ToCSelectors:
    """
    Class to store selectors for extracting the Table of Contents (ToC) and related elements.

    Attributes:
        toc_selector (str): Selector for the ToC container.
        toc_body_selector (str): Selector for the body of the ToC inside the container.
        title_query (str): Selector for extracting the title element (default: "h1").
    """

    toc_selector: str
    toc_body_selector: str
    title_query: str = "h1"


@dataclass(frozen=True)
class NextSiblingAdjustment:
    """
    Data class to store the adjustment applied to the next sibling element during DFS traversal.

    Attributes:
        additional_indent (int): The additional indentation to apply to the next sibling element.
            For example, if the current tag is <p>, after DFS traversal, the next sibling of
            the current tag will have an additional_indent of 2, unless the next sibling is
            also a <p> tag, in which case the indentation will remain unchanged.
        continue_dfs (bool): A flag indicating whether DFS should continue for the next sibling element (optional).
    """

    # Additional indentation to apply to the next sibling element
    additional_indent: int


def get_toc_selectors(
    url: str, tos_type: TableOfContentsType
) -> Optional[ToCSelectors]:
    """
        Returns an object of ToCSelectors containing the appropriate selectors based on the URL and ToC type.

        Parameters:
            url (str): The URL to determine which site's ToC to extract.
            tos_type (TableOfContentsType): The type of ToC to extract.
    class="bd-toc-nav page-toc"
        Returns:
            ToCSelectors: An object containing selectors for ToC container, body, and title query.
    """
    if "docs.python.org" in url:
        if tos_type == TableOfContentsType.MAIN_CONTENT_AREA_TOC:
            # 🛍️ e.g. https://docs.python.org/3/reference/expressions.html
            return ToCSelectors(
                toc_selector="div[class='sphinxsidebarwrapper'] > div > ul",
                toc_body_selector="",
                title_query=":scope > h1",
            )
        elif tos_type == TableOfContentsType.SECTION_NAVIGIATON_TOC:
            # 📝 It is only works for each root category ursl from https://docs.python.org/3/index.html
            # 🛍️ e.g. https://docs.python.org/3/reference/index.html, https://docs.python.org/3/library/index.html
            return ToCSelectors(
                toc_selector="div[role='main'][class='body'] > section",
                toc_body_selector=":scope > div[class='toctree-wrapper compound']",
                title_query=":scope > h1",
            )
    elif "docs.yoctoproject.org" in url:
        if tos_type == TableOfContentsType.MAIN_CONTENT_AREA_TOC:
            return ToCSelectors(
                toc_selector="div[role='main'][class='document'] > div > section",
                toc_body_selector=":scope > div[class='toctree-wrapper compound']",
                title_query=":scope > h1",
            )
        elif tos_type == TableOfContentsType.SECTION_NAVIGIATON_TOC:
            # 📝 recommend to run script in https://docs.yoctoproject.org/index.html
            return ToCSelectors(
                toc_selector="div[class='wy-menu wy-menu-vertical']",
                toc_body_selector="",
                title_query=":scope > h1",
            )
    elif "numpy.org" in url:
        if tos_type == TableOfContentsType.MAIN_CONTENT_AREA_TOC:
            return ToCSelectors(
                toc_selector="body main[id='main-content']",
                toc_body_selector="nav[class='bd-toc-nav page-toc']",
                title_query="article[class='bd-article'] > section > h1",
            )
        elif tos_type == TableOfContentsType.SECTION_NAVIGIATON_TOC:
            return ToCSelectors(
                toc_selector="body",
                toc_body_selector=":scope div[class='bd-toc-item navbar-nav']",
                title_query=":scope > header li[class='nav-item current active'] > a",
            )
    elif "pandas.pydata.org" in url:
        if tos_type == TableOfContentsType.MAIN_CONTENT_AREA_TOC:
            return ToCSelectors(
                toc_selector="body main[id='main-content']",
                toc_body_selector="nav[class='bd-toc-nav page-toc']",
                title_query="article[class='bd-article'] > section > h1",
            )
        elif tos_type == TableOfContentsType.SECTION_NAVIGIATON_TOC:
            return ToCSelectors(
                toc_selector="body",
                toc_body_selector=":scope div[class='bd-toc-item navbar-nav']",
                title_query=":scope > nav[class~='bd-header'] li[class='nav-item current active'] > a",
            )
    elif "docs.opencv.org" in url:
        if tos_type == TableOfContentsType.MAIN_CONTENT_AREA_TOC:
            return ToCSelectors(
                toc_selector="body > div:not([id])",
                toc_body_selector=":scope > div[class='contents']",
                title_query=":scope div[class='title']",
            )
        elif tos_type == TableOfContentsType.SECTION_NAVIGIATON_TOC:
            return None
    elif "pytorch.org" in url:
        if tos_type == TableOfContentsType.MAIN_CONTENT_AREA_TOC:
            return ToCSelectors(
                toc_selector="div[id='pytorch-side-scroll-right']",
                toc_body_selector="",
                title_query=":scope > h1",
            )
        elif tos_type == TableOfContentsType.SECTION_NAVIGIATON_TOC:
            return ToCSelectors(
                toc_selector="div[id='pytorch-documentation']",
                toc_body_selector="",
                title_query=":scope > h1",
            )
    elif "tensorflow.org" in url:
        if tos_type == TableOfContentsType.MAIN_CONTENT_AREA_TOC:
            return ToCSelectors(
                toc_selector="main[class='devsite-main-content']",
                toc_body_selector="devsite-toc[class='devsite-nav devsite-toc']",
                title_query=":scope > devsite-content > article> h1",
            )
        elif tos_type == TableOfContentsType.SECTION_NAVIGIATON_TOC:
            return ToCSelectors(
                toc_selector="ul[class~='devsite-nav-list'][menu='_book']",
                toc_body_selector="",
                title_query=":scope > h1",
            )

    # Default case for unsupported URLs
    return None


# DFS function to parse <ul>, <li>, or <p> elements and extract links
async def parse_toc_elements(
    tag: ElementHandle,
    base_url: str,
    result_list: list[str],
    current_indent: str = INDENT_BASE_UNIT,
) -> NextSiblingAdjustment:
    """
    Recursively parse <ul>, <li>, or <p> elements and extract links from a Table of Contents (ToC).

    This function performs a Depth-First Search (DFS) to traverse the nested structure of a
    Table of Contents. It collects all links (<a> tags) from the elements (<ul>, <li>, or <p>),
    formats them with hierarchical indentation, and appends them to the result list.

    The indentation increases as we dive deeper into the nested structure, and additional
    indentation is applied to the next sibling after encountering a <p> tag in the same DOM level.

    Parameters:
        tag (ElementHandle): The current HTML element to start parsing from.
        base_url (str): The base URL to resolve relative links.
        result_list (list[str]): A list to store the formatted output with links.
        current_indent (str): The current indentation level to apply to the hierarchy.

    Returns:
        NextSiblingAdjustment: An object indicating whether additional indentation is needed
        for the next sibling element.
    """
    next_indent = current_indent
    href = full_href_format = ""
    anchor_symbol = "#"

    # Identify the tag type (e.g., <p>, <ul>, <li>) and handle text extraction
    tag_name = await get_tag_name(tag)
    prefix_or_item_header = ""
    if tag_name in ["div", "p"]:
        span_element = await tag.query_selector(":scope > span")
        if span_element:
            prefix_or_item_header = await extract_direct_text(span_element)
    else:
        prefix_or_item_header = await extract_direct_text(tag)
        # # History: from MSDN crawling
        # if not prefix_or_item_header:
        #     span_element = await tag.query_selector(":scope > span")
        #     if span_element:
        #         prefix_or_item_header = await extract_direct_text(span_element)

    text = prefix_or_item_header

    # Initialize the next indentation level
    a_tag = await tag.query_selector(":scope > a")

    # If an <a> tag is found, extract its href and text
    if a_tag:
        href = await a_tag.get_attribute("href")
        if href is None or href == "#":
            href = ""
        a_tag_text = await a_tag.inner_text()
        text = (
            f"{a_tag_text} ({prefix_or_item_header})"
            if prefix_or_item_header
            else a_tag_text
        )

        # Resolve the href against the base URL and prepare for formatting
        if href:
            full_href = urljoin(base_url, href)
            full_href_format = f" ; {full_href}"
            anchor_symbol = "#️⃣" if href.find("#") != -1 else "⚓"

    # Find child elements for further recursive processing (<ul>, <li>, <p>)
    child_elements = await tag.query_selector_all(
        ":scope > div, :scope > p, :scope > ul, :scope > li"
    )

    if text:
        # Append the current item to the result list with the current indentation
        result_list.append(f"{current_indent}{anchor_symbol} {text}{full_href_format}")
        next_indent += INDENT_BASE_UNIT  # Increase the indent for child elements

    # Track whether additional indentation is required after encountering a <p> tag
    indent_offset: int = 0
    for child in child_elements:
        # Pre-processing: determine the tag name of the child element
        child_tag_name = await get_tag_name(child)
        child_indent = next_indent

        # Apply extra indentation only after a <p> tag at the same DOM level
        if child_tag_name not in ["div", "p"]:
            child_indent += INDENT_BASE_UNIT * indent_offset

        # Recursively process child elements
        next_sibling_adjustment = await parse_toc_elements(
            child, base_url, result_list, child_indent
        )

        # Post-processing: after encountering the first <p> tag, apply extra indentation to siblings
        if indent_offset == 0 and next_sibling_adjustment.additional_indent != 0:
            indent_offset = 1  # Set the additional indent flag for non-<p> siblings

    # If the current element is a <p> tag with text, return an indicator for extra indentation
    if tag_name in ["div", "p"] and text:
        return NextSiblingAdjustment(additional_indent=1)
    else:
        return NextSiblingAdjustment(additional_indent=0)


# Main function to initiate parsing
async def extract_toc(
    url: str, tos_type: TableOfContentsType = TableOfContentsType.MAIN_CONTENT_AREA_TOC
) -> str:
    """
    Extract and format Pandas ToC from a given URL using Playwright.

    Parameters:
        url (str): The URL of the Pandas documentation page to parse.
        bar_type (str): Determines which bar to parse, "side_bar" or "main_bar".

    Returns:
        str: A formatted string containing the extracted links, with indentation indicating
             the hierarchy of the content.
    """
    print(f"Start parse {url}")

    async with async_playwright() as p:
        ### Initialize result list
        result_list: list[str] = []

        # Launch Microsoft Edge
        browser = await p.chromium.launch(headless=True, channel="msedge")
        page = await browser.new_page()

        # Navigate to the Pandas page
        await page.goto(url)

        # show visible elements
        await page.evaluate(
            """
            document.querySelectorAll('ul').forEach(function(ul) {
                ul.classList.add('visible');
            });
            """
        )
        ## Parse browser
        # Get ToCSelectors object based on the URL and ToC type
        toc_selectors = get_toc_selectors(url, tos_type)
        if not toc_selectors:
            return "the corresponding domain name in the URL is not supported or the specified TOC does not exist in the URL! 📅 2024-09-28 18:48:34"

        if not toc_selectors.toc_selector:
            print(f"Table of Contents (ToC) for {tos_type} not found!")
            return ""

        toc_container = await page.query_selector(toc_selectors.toc_selector)
        if not toc_container:
            print(f"Table of Contents (ToC) for {tos_type} not found!")
            return ""

        # Search whether title exists or not on the page using title_query.
        toc_string = ""
        title_element = await toc_container.query_selector(toc_selectors.title_query)
        if title_element:
            # Extract the title text and URL
            title_text = await extract_direct_text(title_element)
            root_url = page.url
            toc_string = f"⚓ {title_text} ; {root_url}"
        if toc_string:
            result_list.append(toc_string)

        # If toc_body_selector is different, perform an internal query; otherwise use toc_container directly
        if toc_selectors.toc_body_selector:
            toc_body = await toc_container.query_selector(
                toc_selectors.toc_body_selector
            )
        else:
            toc_body = toc_container  # toc_container is directly used as toc_body
        if not toc_body:
            print("Table of Contents (ToC) Body not found!")
            return ""

        ## Start parsing tos body
        next_indent = "  " if toc_string else ""
        await parse_toc_elements(toc_body, url, result_list, current_indent=next_indent)

        await browser.close()

        # Join the result list into a formatted string
        return "\n".join(result_list)


if __name__ == "__main__":
    # Title: Python
    result = asyncio.run(
        extract_toc(
            url="https://docs.python.org/3/reference/expressions.html",
            # url="https://docs.python.org/3/reference/index.html",
            tos_type=TableOfContentsType.MAIN_CONTENT_AREA_TOC,
            # tos_type=TableOfContentsType.SECTION_NAVIGIATON_TOC,
        )
    )
    # # Title: Yocto Project
    # result = asyncio.run(
    #     extract_toc(
    #         # url="https://docs.yoctoproject.org/index.html",
    #         url="https://docs.yoctoproject.org/overview-manual/index.html",
    #         tos_type=TableOfContentsType.MAIN_CONTENT_AREA_TOC,
    #         # tos_type=TableOfContentsType.SECTION_NAVIGIATON_TOC,
    #     )
    # )
    # Title: OpenCV
    # result = asyncio.run(
    #     extract_toc(
    #         url="https://pandas.pydata.org/docs/user_guide/io.html",
    #         # url="https://numpy.org/doc/stable/user/quickstart.html",
    #         # tos_type=TableOfContentsType.MAIN_CONTENT_AREA_TOC,
    #         tos_type=TableOfContentsType.SECTION_NAVIGIATON_TOC,
    #     )
    # )

    # # Title: pytorch
    # result = main_content_area_toc = asyncio.run(
    #     extract_toc(
    #         url="https://pytorch.org/docs/stable/notes/cuda.html",
    #         tos_type=TableOfContentsType.MAIN_CONTENT_AREA_TOC,
    #     )
    # )

    # result = asyncio.run(
    #     extract_toc(
    #         url="https://pytorch.org/docs/stable/index.html",
    #         tos_type=TableOfContentsType.SECTION_NAVIGIATON_TOC,
    #     )
    # )

    # # Title: tensorflow
    # result = asyncio.run(
    #     extract_toc(
    #         url="https://www.tensorflow.org/tutorials/keras/regression",
    #         # tos_type=TableOfContentsType.MAIN_CONTENT_AREA_TOC,
    #         tos_type=TableOfContentsType.SECTION_NAVIGIATON_TOC,
    #     )
    # )

    temp_str_file = create_temp_str_file(result, prefix="")
    open_file_in_vscode(temp_str_file)
