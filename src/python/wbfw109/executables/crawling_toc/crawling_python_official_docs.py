"""
Written at 📅 2024-09-24 04:27:38
"""

import asyncio

from playwright.async_api import ElementHandle, async_playwright
from wbfw109.libs.file import create_temp_str_file, open_file_in_vscode


# DFS function to parse <ul> or <li> elements and extract links
async def parse_toc_elements(
    tag: ElementHandle,
    base_url: str = "https://docs.python.org/3/library/",
    result_list: list[str] = [],
    parent_indent: str = "",
) -> list[str]:
    """
    Recursively parse <ul> or <li> elements and extract links from Sphinx-based documentation.

    This function performs a Depth-First Search (DFS) to traverse through the nested
    <ul> and <li> elements in Sphinx-generated HTML documents, collecting all links
    from <a> tags. The extracted links are appended to the result_list in a hierarchical
    format with increasing indentation for deeper levels.

    This function can be used not only with Python's official documentation but with
    any Sphinx-generated documentation that follows a similar structure.

    Parameters:
        tag (ElementHandle): The current HTML element to start parsing from. Typically a <ul> or <li> element.
        base_url (str): The base URL to prepend to relative links (default is Python's documentation base URL).
        result_list (list[str]): A list to store the formatted link entries. Indentation indicates the hierarchy.
        parent_indent (str): The current level of indentation for the hierarchy (used for recursive calls).

    Returns:
        list[str]: The updated list of links with hierarchical indentation.

    Example:
        >>> result_list = []
        >>> await parse_toc_elements(some_tag, "https://docs.python.org/3/library/", result_list)
        >>> print(result_list)
        ['⚓ Section 1 ; https://docs.python.org/3/library/section1', '  ⚓ Subsection 1.1 ; https://docs.python.org/3/library/subsection1.1']
    """
    current_indent = parent_indent

    # Find the <a> tag among the direct children (use :scope to limit to direct children only)
    a_tag = await tag.query_selector(":scope > a")

    # If an <a> tag is found, extract the text and href
    if a_tag:
        href = await a_tag.get_attribute("href")
        text = await a_tag.inner_text()

        # Check if href exists, otherwise skip this element
        if href is None and not text:
            return result_list

        # Build the full href
        full_href_format = f" ; {base_url}{href}" if href else ""
        # Increase indent by 2 spaces per level
        current_indent += "  " if result_list else ""
        # set anchor_symbol
        anchor_symbol: str = "#"
        if href is not None:
            anchor_symbol: str = "#️⃣" if href.find("#") != -1 else "⚓"
        result_list.append(f"{current_indent}{anchor_symbol} {text}{full_href_format}")

    # Traverse the direct children of the current tag and perform DFS on <ul> or <li> tags
    child_elements = await tag.query_selector_all(":scope > ul, :scope > li")
    for child in child_elements:
        await parse_toc_elements(child, base_url, result_list, current_indent)

    return result_list


# Main function to initiate parsing
async def extract_links_from_page(url: str, parse_sphinx_toc: bool = False) -> str:
    """
    Extract and format links from a Sphinx-based documentation page.

    This function loads a webpage (typically Sphinx-based documentation) using Playwright,
    extracts relevant links, and formats them in a hierarchical manner. It can parse either
    the Sphinx table of contents or general page content depending on the `parse_sphinx_toc` flag.

    Parameters:
        url (str): The URL of the webpage to parse.
        parse_sphinx_toc (bool): If True, the function will specifically target the Sphinx
                                 sidebar table of contents (default is False).

    Returns:
        str: A formatted string containing the extracted links, with indentation indicating
             the hierarchy of the content.

    Example:
        >>> result = await extract_links_from_page("https://docs.python.org/3/library/index.html", parse_sphinx_toc=True)
        >>> print(result)
        ⚓ 6. Expressions ; https://docs.python.org/3/library/expressions.html
          ⚓ 6.1. Arithmetic conversions ; https://docs.python.org/3/library/arithmetic-conversions.html
    """
    async with async_playwright() as p:
        # Launch Edge browser in headful mode
        browser = await p.chromium.launch(headless=False, channel="msedge")
        page = await browser.new_page()

        # Navigate to the URL
        await page.goto(url)

        # Wait for the DOM to be fully loaded
        await page.wait_for_load_state("domcontentloaded")

        result_list: list[str] = []

        if parse_sphinx_toc:
            # Find the Sphinx sidebar <div class="sphinxsidebarwrapper">
            sidebar_wrapper = await page.query_selector("div.sphinxsidebarwrapper")
            if not sidebar_wrapper:
                print("Sidebar not found!")
                return ""

            # Start parsing from the first <ul> within the sidebar
            toc_ul = await sidebar_wrapper.query_selector("ul")
            if toc_ul:
                await parse_toc_elements(toc_ul, result_list=result_list)
            else:
                print("No <ul> found in sidebar!")
        else:
            # Find the <div class="body" role="main"> element for normal page parsing
            main_div = await page.query_selector("div.body[role='main']")
            if not main_div:
                print("Main div not found!")
                return ""

            # Find the first <section> tag that is a direct child of <div class="body" role="main">
            section_tag = await main_div.query_selector("section")
            if not section_tag:
                print("Section tag not found!")
                return ""

            # Parse the <h1> tag for the page title
            h1_tag = await section_tag.query_selector("h1")
            if h1_tag:
                title_text = await h1_tag.inner_text()
                result_list.append(f"⚓ {title_text} ; {url}")

            # Parse the <div class="toctree-wrapper compound">
            toc_wrapper = await page.query_selector("div.toctree-wrapper.compound")
            if toc_wrapper:
                await parse_toc_elements(toc_wrapper, result_list=result_list)
            else:
                print("TOC wrapper not found!")

        await browser.close()
        return "\n".join(result_list)


if __name__ == "__main__":
    # result = asyncio.run(
    #     extract_links_from_page(
    #         "https://docs.python.org/3/reference/expressions.html",
    #         parse_sphinx_toc=True,
    #     )
    # )
    # result = asyncio.run(
    #     extract_links_from_page(
    #         "https://docs.python.org/3/library/index.html",
    #         parse_sphinx_toc=False,
    #     )
    # )
    result = asyncio.run(
        extract_links_from_page(
            "https://docs.python.org/3/reference/index.html",
            parse_sphinx_toc=False,
        )
    )

    temp_str_file = create_temp_str_file(result, prefix="python_docs")
    open_file_in_vscode(temp_str_file)
