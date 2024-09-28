"""
Written at 📅 2024-09-28 17:53:14
📝 It is last recent crawling code.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urljoin

from playwright.async_api import ElementHandle, async_playwright
from wbfw109.libs.crawling.playwright_utils import extract_direct_text, get_tag_name

INDENT_BASE_UNIT = "  "


class TableOfContentsType(Enum):
    MAIN_CONTENT_AREA_TOC = 1
    DOCUMENT_NAVIGIATON_TOC = 2


@dataclass(frozen=True)
class SiblingImpact:
    """
    Data class to store the impact on the next sibling element during DFS traversal.

    Attributes:
        additional_indent (int): The additional indentation to apply to the next sibling element.
            For example, if the current tag is <p>, after DFS traversal, the next sibling of
            the current tag will have an additional_indent of 2, unless the next sibling is
            also a <p> tag, in which case the indentation will remain unchanged.
        continue_dfs (bool): A flag indicating whether DFS should continue for the next sibling element (optional).
    """

    # Additional indentation to apply to the next sibling element
    additional_indent: int


# DFS function to parse <ul>, <li>, or <p> elements and extract links
async def parse_toc_elements_pytorch(
    tag: ElementHandle,
    base_url: str,
    result_list: list[str],
    current_indent: str = INDENT_BASE_UNIT,
) -> SiblingImpact:
    """
    Recursively parse <ul>, <li>, or <p> elements and extract links from a Table of Contents (ToC).

    This function performs a Depth-First Search (DFS) to traverse the nested structure of a
    Table of Contents in Pandas documentation. It collects all links (<a> tags) from the
    elements (<ul>, <li>, or <p>), formats them with hierarchical indentation, and appends them to the result list.

    The indentation increases as we dive deeper into the nested structure, and additional
    indentation is applied to siblings after encountering a <p> tag in the same DOM level.

    Parameters:
        tag (ElementHandle): The current HTML element to start parsing from.
        base_url (str): The base URL to resolve relative links.
        result_list (list[str]): A list to store the formatted output with links.
        current_indent (str): The current indentation level to apply to the hierarchy.

    Returns:
        SiblingImpact: An object indicating if additional indentation is needed for siblings.
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
        sibling_impact = await parse_toc_elements_pytorch(
            child, base_url, result_list, child_indent
        )

        # Post-processing: after encountering the first <p> tag, apply extra indentation to siblings
        if indent_offset == 0 and sibling_impact.additional_indent != 0:
            indent_offset = 1  # Set the additional indent flag for non-<p> siblings

    # If the current element is a <p> tag with text, return an indicator for extra indentation
    if tag_name in ["div", "p"] and text:
        return SiblingImpact(additional_indent=1)
    else:
        return SiblingImpact(additional_indent=0)


# Main function to initiate parsing
async def extract_toc_pytorch(
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
        if tos_type == TableOfContentsType.MAIN_CONTENT_AREA_TOC:
            toc_selector = 'div[id="pytorch-side-scroll-right"]'
        elif tos_type == TableOfContentsType.DOCUMENT_NAVIGIATON_TOC:
            toc_selector = 'div[id="pytorch-documentation"]'
        else:
            return ""

        toc_container = await page.query_selector(toc_selector)
        if not toc_container:
            print(f"Table of Contents (ToC) for {tos_type} not found!")
            return ""

        # Search whether title exists or not on the page.
        toc_string = ""
        title_element = await toc_container.query_selector("h1")
        if title_element:
            # Extract the title text and URL
            title_text = await extract_direct_text(title_element)
            root_url = page.url
            toc_string = f"⚓ {title_text} ; {root_url}\n"
        if toc_string:
            result_list.append(toc_string)

        # Differentiate between direct use or further query inside toc_container
        if tos_type == TableOfContentsType.MAIN_CONTENT_AREA_TOC:
            toc_body_selector = ""
        elif tos_type == TableOfContentsType.DOCUMENT_NAVIGIATON_TOC:
            toc_body_selector = ""
        else:
            return ""

        # If toc_body_selector is different, perform an internal query; otherwise use toc_container directly
        if toc_body_selector:
            toc_body = await toc_container.query_selector(toc_body_selector)
        else:
            toc_body = toc_container  # toc_container is directly used as toc_body
        if not toc_body:
            print("Table of Contents (ToC) Body not found!")
            return ""

        ## Start parsing tos body
        next_indent = "  " if toc_string else ""
        await parse_toc_elements_pytorch(
            toc_body, url, result_list, current_indent=next_indent
        )

        await browser.close()

        # Join the result list into a formatted string
        return "\n".join(result_list)


if __name__ == "__main__":
    # print("MAIN_CONTENT_AREA_TOC:")
    main_content_area_toc = asyncio.run(
        extract_toc_pytorch(
            url="https://pytorch.org/docs/stable/notes/cuda.html",
            tos_type=TableOfContentsType.MAIN_CONTENT_AREA_TOC,
        )
    )
    print(main_content_area_toc)

    # print("\n\n\nDOCUMENT_NAVIGIATON_TOC:")
    # document_navigation_toc = asyncio.run(
    #     extract_toc_pytorch(
    #         url="https://pytorch.org/docs/stable/index.html",
    #         tos_type=TableOfContentsType.DOCUMENT_NAVIGIATON_TOC,
    #     )
    # )
    # print(document_navigation_toc)
