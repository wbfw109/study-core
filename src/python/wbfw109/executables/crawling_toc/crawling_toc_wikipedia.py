"""
Written at 📅 2024-09-24 04:27:38
https://en.wikipedia.org/wiki/Systems_development_life_cycle

TODO: crawling query 2: document.querySelectorAll('table[class*="sidebar"]')
TODO: crawling query 3: document.querySelectorAll('table[role="navigation"]')

TODO: It seems that Sidebar table may not have consistent structure. It required to be checked for sidebar tables in other documentations. it may be similar with structure of crawling query 3'

crawling query 3
    /html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/div[52]/table/tbody/tr[3]/td[1]/table/tbody/tr[3]/td/div/ul/li[1]/a
    /html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/div[52]/table/tbody/tr[3]/td[1]/table/tbody/tr[1]/td/div/ul/li[1]/a
    /html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/div[52]/table/tbody/tr[4]/td/div/ul/li[1]/a

"""

import asyncio

from playwright.async_api import ElementHandle, async_playwright


# DFS function to parse <ul> or <li> elements and extract links
# DFS function to parse <ul> or <li> elements and extract links
async def parse_toc_elements_id_vector_toc(
    tag: ElementHandle, base_url: str, result_list: list[str], current_indent: str = ""
) -> list[str]:
    """
    Recursively parse <ul> or <li> elements and extract links from Wikipedia ToC.

    This function performs a DFS to traverse through the nested <ul> and <li> elements,
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

    # Find the <a> tag directly under the <li> (not recursive)
    a_tag = await tag.query_selector(":scope > a")

    if a_tag:
        href = await a_tag.get_attribute("href")
        text_element = await a_tag.query_selector(".vector-toc-text")

        # Initialize text variable to avoid unbound variable error
        text = ""

        if text_element:
            # Get all text, including child nodes
            full_text = await text_element.inner_text()
            # Exclude the text in <span class="vector-toc-numb">
            num_element = await text_element.query_selector(".vector-toc-numb")
            if num_element:
                num_text = await num_element.inner_text()
                text = full_text.replace(
                    num_text, ""
                ).strip()  # Remove the number and extra spaces
            else:
                text = full_text.strip()  # If no number exists, just use the full text

        # Check if href exists and text is not empty
        if href == "#" or (href is None and not text):
            return result_list

        # Dynamically build the full href
        full_href_format = f" ; {base_url}{href}" if href else ""

        # Dynamically set anchor_symbol
        anchor_symbol = "#"
        if href:
            anchor_symbol = "#️⃣" if href.find("#") != -1 else "⚓"

        # Append the formatted result
        result_list.append(f"{current_indent}{anchor_symbol} {text}{full_href_format}")

    # Recursively search child <ul> or <li> elements and increase indentation level
    child_elements = await tag.query_selector_all(":scope > ul > li")
    for child in child_elements:
        await parse_toc_elements_id_vector_toc(
            child, base_url, result_list, current_indent + "  "
        )

    return result_list


# Main function to initiate parsing
async def extract_wikipedia_toc(url: str) -> str:
    """
    Extract and format Wikipedia ToC from a given URL using Playwright.

    Parameters:
        url (str): The URL of the Wikipedia page to parse.

    Returns:
        str: A formatted string containing the extracted links, with indentation indicating
             the hierarchy of the content.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        # Navigate to the Wikipedia page
        await page.goto(url)

        # Wait for the table of contents (ToC) to load
        toc_container = await page.query_selector('div[id="vector-toc"]')
        if not toc_container:
            print("Table of Contents (ToC) not found!")
            return ""

        # Get all <li> elements from the ToC
        toc_items = await toc_container.query_selector_all(
            "ul.vector-toc-contents > li"
        )

        # Initialize result list
        result_list: list[str] = []
        for item in toc_items:
            await parse_toc_elements_id_vector_toc(
                item, "https://en.wikipedia.org", result_list
            )

        await browser.close()

        return "\n".join(result_list)


if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Software_engineering"
    result = asyncio.run(extract_wikipedia_toc(url))
    print(result)
