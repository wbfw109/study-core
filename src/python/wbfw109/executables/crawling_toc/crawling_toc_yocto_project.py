"""
Written at 📅 2024-09-26 01:26:45
"""

import asyncio
from urllib.parse import urljoin

from playwright.async_api import ElementHandle, async_playwright
from wbfw109.libs.crawling.playwright_utils import extract_direct_text


# DFS function to parse <ul>, <li>, <p>, and <a> elements from Yocto Project ToC
async def parse_toc_elements_yocto_side_bar(
    tag: ElementHandle,
    base_url: str,
    result_list: list[str],
    current_indent: str = "  ",
) -> list[str]:
    """
    Recursively parse <ul>, <li>, <p>, or <a> elements and extract links from Yocto Project ToC.

    Parameters:
        tag (ElementHandle): The current HTML element to start parsing from.
        base_url (str): The base URL to prepend to relative links.
        result_list (list[str]): A list to store the formatted link entries.
        current_indent (str): The current level of indentation for the hierarchy.

    Returns:
        list[str]: The updated list of links with hierarchical indentation.
    """

    # Get the <p> tag's text or <a> tag text
    item_header = await extract_direct_text(tag)
    next_indent = current_indent
    a_tag = await tag.query_selector(":scope > a")
    href = full_href_format = ""
    text = item_header
    anchor_symbol = "#"

    if a_tag:
        href = await a_tag.get_attribute("href")
        if href is None or href == "#":
            href = ""
        a_tag_text = await extract_direct_text(a_tag)
        text = f"{a_tag_text} ({item_header})" if item_header else a_tag_text

        # Resolve relative URLs using urljoin
        if href:
            full_href = urljoin(base_url, href)
            full_href_format = f" ; {full_href}"
            anchor_symbol = "#️⃣" if href.find("#") != -1 else "⚓"

    # Search for child elements (<ul>, <li>, <p>)
    child_elements = await tag.query_selector_all(
        ":scope > ul, :scope > li, :scope > p"
    )

    if not child_elements and not href:
        return result_list

    if text:
        # Append the result with indentation
        result_list.append(f"{current_indent}{anchor_symbol} {text}{full_href_format}")
        next_indent += "  "

    for child in child_elements:
        await parse_toc_elements_yocto_side_bar(
            child, base_url, result_list, next_indent
        )

    return result_list


# Main function to initiate parsing for Yocto Project ToC
async def extract_toc_yocto_side_bar(url: str) -> str:
    """
    Extract and format Yocto Project ToC from a given URL using Playwright.

    Parameters:
        url (str): The URL of the Yocto Project page to parse.

    Returns:
        str: A formatted string containing the extracted links, with indentation indicating
             the hierarchy of the content.
    """
    async with async_playwright() as p:
        # Launch Microsoft Edge
        browser = await p.chromium.launch(headless=True, channel="msedge")
        page = await browser.new_page()

        # Navigate to the Yocto Project page
        await page.goto(url)

        # Find the div with a class that contains 'wy-menu'
        toc_container = await page.query_selector('div[class*="wy-menu"]')
        if not toc_container:
            print("Table of Contents (ToC) not found!")
            return ""

        # Initialize result list
        result_list: list[str] = []
        toc_items = await toc_container.query_selector_all(":scope > *")

        current_indent = ""
        for index in range(0, len(toc_items), 2):
            p_element = toc_items[index]
            ul_element = toc_items[index + 1]

            # Extract <p> text and append to result
            p_text = await p_element.inner_text()
            result_list.append(f"{current_indent}# {p_text}")

            # Parse <ul> under the current <p> for nested ToC items
            ul_items = await ul_element.query_selector_all(":scope > li")
            for ul_item in ul_items:
                await parse_toc_elements_yocto_side_bar(
                    ul_item, url, result_list, current_indent + "  "
                )

        await browser.close()

        # Return the formatted result
        return "\n".join(result_list)


# if __name__ == "__main__":
#     # Example Yocto Project page URL
#     url = "https://docs.yoctoproject.org/brief-yoctoprojectqs/index.html"
#     result = asyncio.run(extract_toc_yocto_side_bar(url))
#     print(result)


from playwright.async_api import ElementHandle


# DFS function to parse <ul>, <li>, or <a> elements and extract links
async def parse_toc_elements_yocto_manual(
    tag: ElementHandle,
    base_url: str,
    result_list: list[str],
    current_indent: str = "  ",
) -> list[str]:
    """
    Recursively parse <ul>, <li>, or <a> elements and extract links from Yocto Project ToC.

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
    # Find the <a> tag directly under the <li> or <p>
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
            # Build the full href with formatting
            full_href_format = f" ; {full_href}"
            anchor_symbol = "#️⃣" if href.find("#") != -1 else "⚓"

    # Recursively search child <ul>, <li> elements and increase indentation level
    child_elements = await tag.query_selector_all(":scope > ul, :scope > li")

    # Add result if it's not the last node in DFS
    if not child_elements and not href:
        return result_list

    if text:
        # Add the result in the format with indentation
        result_list.append(f"{current_indent}{anchor_symbol} {text}{full_href_format}")
        next_indent += "  "

    for child in child_elements:
        await parse_toc_elements_yocto_manual(child, base_url, result_list, next_indent)

    return result_list


# Main function to initiate parsing for Yocto Project ToC
async def extract_toc_yocto_manual(url: str) -> str:
    """
    Extract and format Yocto Project ToC from a given URL using Playwright.

    Parameters:
        url (str): The URL of the Yocto Project main manual page to parse.

    Returns:
        str: A formatted string containing the extracted links, with indentation indicating
             the hierarchy of the content.
    """
    async with async_playwright() as p:
        # Launch Microsoft Edge
        browser = await p.chromium.launch(headless=True, channel="msedge")
        page = await browser.new_page()

        # Navigate to the Yocto Project page
        await page.goto(url)

        # Locate the root element (<div itemprop="articleBody">/<section>)
        root_element = await page.query_selector(
            'div[itemprop="articleBody"] > section'
        )
        if not root_element:
            print("Root element not found!")
            return ""

        # Extract the title from <h1> and append it to the results
        result_list: list[str] = []
        h1_element = await root_element.query_selector("h1")
        if h1_element:
            title_text = await extract_direct_text(h1_element)
            result_list.append(f"⚓ {title_text} ; {url}")

        # Find the div with class="toctree-wrapper compound"
        toc_container = await root_element.query_selector(
            'div[class*="toctree-wrapper compound"]'
        )
        if not toc_container:
            print("ToC container not found!")
            return "\n".join(result_list)

        # Get all <ul>/<li> elements from the ToC container
        toc_items = await toc_container.query_selector_all(":scope > ul, :scope > li")

        # Parse each <li> element and extract the text and href
        current_indent = "  "
        for toc_item in toc_items:
            await parse_toc_elements_yocto_manual(
                toc_item, url, result_list, current_indent
            )

        await browser.close()

        # Return the formatted result
        return "\n".join(result_list)


if __name__ == "__main__":
    # Example Yocto Project main manual page URL
    url = "https://docs.yoctoproject.org/overview-manual/index.html"
    result = asyncio.run(extract_toc_yocto_manual(url))
    print(result)
