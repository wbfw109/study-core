import asyncio
from urllib.parse import urljoin

from playwright.async_api import ElementHandle, async_playwright
from wbfw109.libs.crawling.playwright_utils import extract_direct_text


# DFS function to parse <ul>, <li>, or <p> elements and extract links
async def parse_toc_elements_opencv(
    tag: ElementHandle,
    base_url: str,
    result_list: list[str],
    current_indent: str = "  ",
) -> list[str]:
    """
    Recursively parse <ul>, <li>, or <p> elements and extract links from OpenCV ToC.

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

    # Find the <a> tag directly under the <li> or <p> (not recursive)
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
        a_tag_text = await extract_direct_text(a_tag)
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

    # Recursively search child <ul>, <li>, or <p> elements and increase indentation level
    child_elements = await tag.query_selector_all(
        ":scope > ul, :scope > li, :scope > p"
    )

    # add result when it is not last node of a branch in DFS.
    if not child_elements and not href:
        return result_list

    if text:
        # Add the result in the format with indentation
        result_list.append(f"{current_indent}{anchor_symbol} {text}{full_href_format}")
        next_indent += "  "
    for child in child_elements:
        await parse_toc_elements_opencv(child, base_url, result_list, next_indent)

    return result_list


# Main function to initiate parsing
async def extract_opencv_toc(url: str) -> str:
    """
    Extract and format OpenCV ToC from a given URL using Playwright.

    Parameters:
        url (str): The URL of the OpenCV page to parse.

    Returns:
        str: A formatted string containing the extracted links, with indentation indicating
             the hierarchy of the content.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True
        )  # Change headless to False if you want to see the browser
        page = await browser.new_page()

        # Navigate to the OpenCV page
        await page.goto(url)

        # Get the root element (title) of the page
        title_element = await page.query_selector('div[class="title"]')
        if not title_element:
            print("Title not found!")
            return ""

        # Extract the title text and URL

        title_text = await title_element.inner_text()
        root_url = page.url
        toc_string = f"⚓ {title_text} ; {root_url}\n"

        # Get the ToC container
        toc_container = await page.query_selector('[class="contents"]')
        if not toc_container:
            print("Table of Contents (ToC) not found!")
            return toc_string

        # Get all <li>, <p> elements from the ToC
        toc_items = await toc_container.query_selector_all(
            ":scope > div.textblock > ul > li"
        )

        # Initialize result list
        result_list: list[str] = []
        for item in toc_items:
            await parse_toc_elements_opencv(
                item, "https://docs.opencv.org/5.x/", result_list
            )

        await browser.close()

        # Join the result list into a formatted string
        return toc_string + "\n".join(result_list)


if __name__ == "__main__":
    # Example OpenCV page URL
    # url = "https://docs.opencv.org/5.x/d6/d00/tutorial_py_root.html"
    url = "https://docs.opencv.org/5.x/index.html"
    result = asyncio.run(extract_opencv_toc(url))
    print(result)
