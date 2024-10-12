# %%
"""
Wirrten at 📅 2024-10-13 04:30:08
"""

from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %%
# Title: crawl steam workshop
import asyncio
import re

from playwright.async_api import Page, async_playwright


async def extract_steam_workshop_titles(page: Page) -> list[str]:
    """
    Extracts titles and URLs from the Steam Workshop page and removes brackets with specific language tags.

    Args:
        page (Page): The current browser page.

    Returns:
        list[str]: A list of formatted titles and URLs in the format "⚓ title ; url".
    """
    elements = await page.query_selector_all(
        "div[class='workshopBrowseItems'] > div > a[class='item_link']"
    )
    result_list: list[str] = []

    # Regular expression pattern: Remove 【...】, [...], 『...』 pairs if they contain EN, CN, JP, or KR
    pattern = re.compile(r"[【\[\『].*?\b(?:EN|CN|JP|KR)\b.*?[】\]\』]")

    for element in elements:
        title = await element.inner_text()
        href = await element.get_attribute("href")

        if title and href:
            # Remove any part that matches the pattern
            cleaned_title = re.sub(pattern, "", title).strip()
            result_list.append(f"⚓ {cleaned_title} ; {href}")

    return result_list


async def go_to_next_page(page: Page) -> bool:
    """
    Navigates to the next page by clicking the second `pagebtn` element, if available.

    Args:
        page (Page): The current browser page.

    Returns:
        bool: True if navigation to the next page was successful, False if the second pagebtn is disabled.
    """
    # Find the second child element with the 'pagebtn' class using full selector syntax
    next_page_button = await page.query_selector(
        "div[class='workshopBrowsePagingControls'] > *[class~='pagebtn']:last-child"
    )

    if next_page_button:
        # Check if the button has the 'disabled' class
        is_disabled = await next_page_button.get_attribute("class")
        if is_disabled and "disabled" in is_disabled:
            return False

        # Check if the element has an href attribute (for anchor tags or similar)
        href = await next_page_button.get_attribute("href")
        if href:
            # Navigate to the next page using href
            await page.goto(href)
            return True

        # If no href, attempt to click the element
        await next_page_button.click()
        return True

    return False


async def crawl_steam_workshop(url: str) -> None:
    """
    Crawls the Steam Workshop, extracting titles and URLs from multiple pages until the second `pagebtn` becomes disabled.

    Args:
        url (str): The URL of the Steam Workshop page.
    """
    async with async_playwright() as p:
        # Launch Microsoft Edge
        browser = await p.chromium.launch(headless=True, channel="msedge")
        page = await browser.new_page()
        await page.goto(url)

        all_results: list[str] = []

        # Repeat until the second page button is disabled
        while True:
            # Extract titles and URLs from the current page
            current_page_results = await extract_steam_workshop_titles(page)
            all_results.extend(current_page_results)

            # Try to go to the next page
            if not await go_to_next_page(page):
                break

        # Sort and print the results
        sorted_results = sorted(all_results)
        for item in sorted_results:
            print(item)

        await browser.close()


if __name__ == "__main__":
    # Replace with the actual URL of the Steam Workshop page
    url = "https://steamcommunity.com/workshop/browse/?appid=1188930&searchtext=kr&childpublishedfileid=0&browsesort=trend&section=readytouseitems&created_date_range_filter_start=0&created_date_range_filter_end=0&updated_date_range_filter_start=0&updated_date_range_filter_end=0&actualsort=trend&p=1"
    asyncio.run(crawl_steam_workshop(url))

# %%
# Title: for chronoark "Workshop Mod KR Localization" mod patch ; https://steamcommunity.com/sharedfiles/filedetails/?id=3343188695&searchtext=local
import shutil
from pathlib import Path

# Set up paths
chrono_ark_steam_app_path: Path = Path(
    "/mnt/c/Program Files/Steam/steamapps/workshop/content/1188930"
)
workshop_mod_kr_localization_path: Path = (
    chrono_ark_steam_app_path / "3343188695" / "Localization"
)

# Dictionary of workshop items
workshop_items: dict[str, str] = {
    "Clyu": "3338047948",
    "AliceToho": "3285393554",
    "Lumia": "3294493213",
    "Kogasa": "3292385904",
    "Clyne": "3266293043",
    "Reimu": "3340884556",
    "Sanae": "3337971709",
    "Chiyo": "3299933546",
    "Meiring": "3340670336",
    "Dorchi": "3333102302",
    "EnchantedArk": "3219754941",
}


# Function to copy localization files
def copy_localization_files(
    workshop_items: dict[str, str],
    chrono_ark_steam_app_path: Path,
    workshop_mod_kr_localization_path: Path,
) -> None:
    for name, workshop_id in workshop_items.items():
        source_path: Path = workshop_mod_kr_localization_path / name / "LangDataDB.csv"
        destination_path: Path = (
            chrono_ark_steam_app_path / workshop_id / "Localization" / "LangDataDB.csv"
        )

        # Copy only if destination_path exists
        if destination_path.exists():
            try:
                shutil.copy(source_path, destination_path)
                print(f"{name} localization file copied successfully.")
            except FileNotFoundError:
                print(f"Source file for {name} not found.")
            except Exception as e:
                print(f"Error copying localization file for {name}: {e}")
        else:
            print(f"Destination path for {name} does not exist.")


# Execute the function
copy_localization_files(
    workshop_items, chrono_ark_steam_app_path, workshop_mod_kr_localization_path
)
