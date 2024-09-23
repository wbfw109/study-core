import asyncio
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell
from playwright.async_api import async_playwright

InteractiveShell.ast_node_interactivity = "all"

# Set the image directory
ai_models_dir: Path = Path.home() / "ai_models"
ai_models_dir.mkdir(parents=True, exist_ok=True)
images_dir: Path = Path.home() / "crawling" / "images"
images_dir.mkdir(parents=True, exist_ok=True)
upscalling_output_dir = images_dir / "upscaling"
upscalling_output_dir.mkdir(parents=True, exist_ok=True)


# Function to manage indent levels and ensure no more than 2 spaces difference
def adjust_indent_level(current_indent, result_list):
    if result_list:
        # Check the last item's indent level
        last_item_indent = result_list[-1].count(" ")
        # Ensure the current indent is no more than 2 spaces deeper than the last item
        if current_indent - last_item_indent > 2:
            return last_item_indent + 2
    return current_indent


# DFS function to parse <ul> or <li> elements and extract links
async def parse_toc_elements(
    tag,
    base_url: str = "https://docs.python.org/3/library/",
    result_list: list[str] = [],
    parent_indent: str = "",
    visited=set(),
) -> list[str]:
    current_indent = parent_indent

    # Find the <a> tag among the direct children
    a_tag = await tag.query_selector("a")

    # If an <a> tag is found, extract the text and href
    if a_tag:
        href = await a_tag.get_attribute("href")
        text = await a_tag.inner_text()

        # Create a unique identifier to avoid duplicates
        identifier = (text, href)

        # Check if this link has already been processed
        if identifier not in visited:
            visited.add(identifier)
            full_href = base_url + href

            # Adjust the indent level to avoid too much difference
            adjusted_indent = adjust_indent_level(len(current_indent), result_list)
            current_indent = " " * adjusted_indent  # Set the indent level

            result_list.append(f"{current_indent}⚓ {text} ; {full_href}")

    # Traverse the children of the current tag and perform DFS on <ul> or <li> tags
    child_elements = await tag.query_selector_all(":scope > ul, :scope > li")
    for child in child_elements:
        await parse_toc_elements(child, base_url, result_list, current_indent, visited)

    return result_list


# Main function to initiate parsing
async def extract_links_from_page(url: str, parse_sphinx_toc: bool = False) -> str:
    async with async_playwright() as p:
        # Launch Edge browser in headful mode
        browser = await p.chromium.launch(headless=False, channel="msedge")
        page = await browser.new_page()

        # Navigate to the URL
        await page.goto(url)

        # Wait for the DOM to be fully loaded
        await page.wait_for_load_state("domcontentloaded")

        result_list: list[str] = []
        visited = set()  # Set to track visited links and avoid duplicates

        if parse_sphinx_toc:
            # Find the Sphinx sidebar <div class="sphinxsidebarwrapper">
            sidebar_wrapper = await page.query_selector("div.sphinxsidebarwrapper")
            if not sidebar_wrapper:
                print("Sidebar not found!")
                return ""

            # Start parsing from the first <ul> within the sidebar
            toc_ul = await sidebar_wrapper.query_selector(":scope > ul")
            if toc_ul:
                await parse_toc_elements(
                    toc_ul, result_list=result_list, visited=visited
                )
            else:
                print("No <ul> found in sidebar!")
        else:
            # Find the <div class="body" role="main"> element for normal page parsing
            main_div = await page.query_selector("div.body[role='main']")
            if not main_div:
                print("Main div not found!")
                return ""

            # Find the first <section> tag that is a direct child of <div class="body" role="main">
            section_tag = await main_div.query_selector(":scope > section")
            if not section_tag:
                print("Section tag not found!")
                return ""

            # Parse the <h1> tag for the page title
            h1_tag = await section_tag.query_selector(":scope > h1")
            if h1_tag:
                title_text = await h1_tag.inner_text()
                result_list.append(f"⚓ {title_text} ; {url}")

            # Parse the <div class="toctree-wrapper compound">
            toc_wrapper = await main_div.query_selector(
                ":scope > div.toctree-wrapper.compound"
            )
            if toc_wrapper:
                await parse_toc_elements(
                    toc_wrapper, result_list=result_list, visited=visited
                )
            else:
                print("TOC wrapper not found!")

        await browser.close()
        return "\n".join(result_list)


if __name__ == "__main__":
    result = asyncio.run(
        extract_links_from_page(
            "https://docs.python.org/3/reference/expressions.html",
            parse_sphinx_toc=True,
        )
    )
    print(result)
