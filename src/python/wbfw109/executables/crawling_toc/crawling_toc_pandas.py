import asyncio
from dataclasses import dataclass
from urllib.parse import urljoin

from playwright.async_api import ElementHandle, async_playwright
from wbfw109.libs.crawling.playwright_utils import extract_direct_text, get_tag_name


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
async def parse_toc_elements_pandas(
    tag: ElementHandle,
    base_url: str,
    result_list: list[str],
    current_indent: str = "  ",
) -> SiblingImpact:
    """
    Recursively parse <ul>, <li>, or <p> elements and extract links from Pandas ToC.

    This function performs a DFS to traverse through the nested elements,
    collecting all links from <a> tags. The extracted links are appended to the result_list
    in a hierarchical format with increasing indentation for deeper levels.

    Parameters:
        tag (ElementHandle): The current HTML element to start parsing from.
        base_url (str): The base URL to prepend to relative links.
        result_list (list[str]): A list to store the formatted link entries.
        current_indent (str): The current level of indentation for the hierarchy.

    Returns:
        list[str]: The updated list of links with hierarchical indentation.
    """

    # Find the <a> tag directly under the <li>
    tag_name = await get_tag_name(tag)
    if tag_name == "p":
        prefix_or_item_header = await tag.inner_text()
    else:
        prefix_or_item_header = await extract_direct_text(tag)

    next_indent = current_indent
    a_tag = await tag.query_selector(":scope > a")
    href = full_href_format = ""
    text = prefix_or_item_header
    anchor_symbol = "#"
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

        # Use urljoin to resolve the relative URL against the base URL
        if href:
            full_href = urljoin(base_url, href)
            # Dynamically build the full href
            full_href_format = f" ; {full_href}"
            anchor_symbol = "#️⃣" if href.find("#") != -1 else "⚓"

    # Recursively search child <ul>, <li> elements and increase indentation level
    child_elements = await tag.query_selector_all(
        ":scope > p, :scope > ul, :scope > li"
    )

    # # Add result if it's not the last node in DFS
    # if not child_elements and not href:
    #     return SiblingImpact(0)

    if text:
        # Add the result in the format with indentation
        result_list.append(f"{current_indent}{anchor_symbol} {text}{full_href_format}")
        next_indent += "  "

    # whether <p> tag acts as parent tree of next siblings which not <p> tag in same DOM parent level.
    indent_offset: int = 0
    for child in child_elements:
        ## pre-process
        child_tag_name = await get_tag_name(child)
        child_indent = ""

        if tag_name == "p" and text:
            child_indent = next_indent
        else:
            child_indent = next_indent + "  " * indent_offset

        ## process
        sibling_impact = await parse_toc_elements_pandas(
            child, base_url, result_list, child_indent
        )

        ## pro-process
        # After first encountering a <p> tag in same DOM parent level.
        if indent_offset == 0 and sibling_impact.additional_indent != 0:
            # it maintains additional indent until in same DOm parent level except for <p> tag; refer to "pre-process" part
            indent_offset = 2
    if tag_name == "p" and text:
        return SiblingImpact(additional_indent=2)
    else:
        return SiblingImpact(additional_indent=0)


# Main function to initiate parsing
async def extract_toc_pandas(url: str, bar_type: str = "side_bar") -> str:
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
        browser = await p.chromium.launch(headless=True)
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
        # Get the root element (title) of the page, if exists
        title_element = await page.query_selector('div[class="title"]')
        toc_string = ""
        if title_element:
            # Extract the title text and URL
            title_text = await title_element.inner_text()
            root_url = page.url
            toc_string = f"⚓ {title_text} ; {root_url}\n"
        # else:
        #     print("Title not found, but continuing with ToC parsing...")

        # Determine the correct ToC container based on bar_type
        toc_container_selector = (
            'div[class="sidebar-primary-item"] > nav > div'
            if bar_type == "side_bar"
            else 'div[class="sidebar-secondary-item"] > nav > ul'
        )

        toc_container = await page.query_selector(toc_container_selector)
        if not toc_container:
            print(f"Table of Contents (ToC) for {bar_type} not found!")
            return toc_string

        # Initialize result list
        result_list: list[str] = []
        if toc_string:
            result_list.append(toc_string)

        # Get all <li> elements from the ToC
        toc_items = await toc_container.query_selector_all(
            ":scope > p, :scope > li, :scope > ul"
        )

        next_indent = "  " if toc_string else ""
        for item in toc_items:

            await parse_toc_elements_pandas(
                item, url, result_list, current_indent=next_indent
            )

        await browser.close()

        # Join the result list into a formatted string
        return toc_string + "\n".join(result_list)


if __name__ == "__main__":
    # Example Pandas page URL
    # url = "https://pandas.pydata.org/docs/user_guide/io.html"
    url = "https://numpy.org/doc/stable/user/quickstart.html"

    # Parse the side bar
    print("Parsing Side Bar:")
    result_side_bar = asyncio.run(extract_toc_pandas(url, bar_type="side_bar"))
    print(result_side_bar)
    print("\n\n\n")

    # Parse the main bar
    # print("\nParsing Main Bar:")
    # result_main_bar = asyncio.run(extract_toc_pandas(url, bar_type="main_bar"))
    # print(result_main_bar)
