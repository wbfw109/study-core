from typing import Optional, Union

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag


def format_toc_msdn(xml_input: str) -> None:
    """Depth-First Search (DFS) 📅 2024-09-02 05:13:55"""
    soup: BeautifulSoup = BeautifulSoup(xml_input, "html.parser")

    def process_element(element: Union[Tag, NavigableString], depth: int = 0) -> None:
        indentation: str = "  " * depth

        if isinstance(element, Tag):
            href: Optional[str] = None
            if element.name == "a":
                # Handle href that could be either a str or a list[str]
                href_value = element.get("href")
                if isinstance(href_value, list):
                    href = href_value[0]  # If it's a list, take the first element
                elif isinstance(href_value, str):
                    href = href_value

                if href:
                    text: str = element.get_text(strip=True)
                    print(f"{indentation}⚓ {text} ; {href}")

            elif element.name == "span" and element.get_text(strip=True):
                text: str = element.get_text(strip=True)
                print(f"\n\n{indentation}# {text}")

            # Process children elements recursively         ### 🚣 recursive=False
            children: list[Tag] = element.find_all(["li", "a", "span"], recursive=False)
            for child in children:
                ## There are no nested li elements in Table of Contents in MSDN.
                if child.name == "li":
                    nested_a = child.find("a", recursive=False)
                    if isinstance(nested_a, Tag):
                        process_element(nested_a, depth)

                    nested_span = child.find("span", recursive=False)
                    if isinstance(nested_span, Tag):
                        process_element(nested_span, depth)

                    nested_ul = child.find("ul", recursive=False)
                    if isinstance(nested_ul, Tag):
                        process_element(nested_ul, depth + 1)
                elif child.name in ["a", "span"]:
                    process_element(child, depth)

    # Find the top-level 'ul' by filtering all 'ul' elements for the correct class
    top_level_ul: Optional[Tag] = next(
        (
            ul
            for ul in soup.find_all("ul")
            if isinstance(ul, Tag)
            and "table-of-contents" in (ul.get("class") or [])  ##### 💡
        ),
        None,
    )

    if top_level_ul:
        process_element(top_level_ul)
    else:
        print("Top-level <ul> with class 'table-of-contents' not found.")


# 📝 Read XML input from your hardcoded input
if __name__ == "__main__":
    xml_input = """"""
    format_toc_msdn(xml_input)
