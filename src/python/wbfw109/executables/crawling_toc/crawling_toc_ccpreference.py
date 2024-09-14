from typing import Optional, Union

import requests
from bs4 import BeautifulSoup, NavigableString, Tag


def fetch_html(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def parse_element(
    element: Union[Tag, NavigableString],
    indent_level: int = 0,
    seen_titles: Optional[set[str]] = None,
    pending_titles: Optional[dict[str, str]] = None,
    parent_has_content: bool = False,
) -> str:
    """
    Recursively parses HTML elements and formats them while avoiding duplicate titles and links.

    - Uses Depth First Search (DFS) to process child elements before siblings.
    - Uses `pending_titles` to store titles without links and updates them when links are found.
    """
    result = []

    if seen_titles is None:
        seen_titles = set()
    if pending_titles is None:
        pending_titles = {}

    indent = " " * indent_level

    if isinstance(element, NavigableString):
        return ""

    name = element.get_text(strip=True)

    has_content = False

    if element.name == "a":
        href = element.get("href")
        if href:
            full_link = f"https://en.cppreference.com{href}"
            if name not in seen_titles:
                # Handle pending titles and attach the found link
                if name in pending_titles:
                    result.append(f"{indent}⚓ {pending_titles[name]} ; {full_link}")
                    pending_titles.pop(name)
                else:
                    result.append(f"{indent}⚓ {name} ; {full_link}")
                seen_titles.add(name)
                has_content = True

    elif element.name in ["b", "p", "tt"]:
        if name not in seen_titles and name not in pending_titles:
            pending_titles[name] = name
            result.append(f"{indent}# {name}")
            has_content = True

    # Recursively parse child elements
    for child in element.children:
        if isinstance(child, (Tag, NavigableString)):
            child_content = parse_element(
                child,
                indent_level + 2 if has_content else indent_level,
                seen_titles,
                pending_titles,
                has_content,
            )
            if child_content:
                result.append(child_content)

    return "\n".join(result)


def extract_title(soup: BeautifulSoup) -> Optional[str]:
    title_element = soup.find("h1", {"id": "firstHeading"})
    if title_element:
        return title_element.get_text(strip=True)
    return None


def write_to_file(content: str, file_path: str) -> None:
    with open(file_path, "w") as f:
        f.write(content)


def process_url(url: str, file_path: str) -> None:
    html_content = fetch_html(url)
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract the title and add it at the top of the result
    title = extract_title(soup)
    result = ""
    if title:
        result += f"⚓ {title} ; {url}\n"

    # Parse the table content
    table = soup.find("table", {"class": "mainpagetable"})
    if table:
        parsed_content = parse_element(table, indent_level=2)
        result += parsed_content
        result += "\n" * 3  # Adds 3 newline characters after each section

    write_to_file(result, file_path)

    import os

    os.system(f"code {file_path}")


def main() -> None:
    urls = ["https://en.cppreference.com/w/cpp", "https://en.cppreference.com/w/c"]

    for url in urls:
        file_name = url.split("/")[-1] + "_toc.txt"
        file_path = f"/tmp/{file_name}"
        process_url(url, file_path)


if __name__ == "__main__":
    main()
