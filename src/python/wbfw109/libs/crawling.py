"""deprecated 📅 2024-09-24 03:09:58
Not use BeatifulSoup. use playwright
"""

from typing import Optional

import requests
from bs4 import BeautifulSoup, Tag
from bs4.element import Tag


# Fetch the web page content
def fetch_web_page(url: str, headers: dict[str, str]) -> str:
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Set encoding to utf-8 to prevent character issues
    response.encoding = "utf-8"
    return response.text


# Parse HTML content into a BeautifulSoup object
def parse_html(html_content: str) -> BeautifulSoup:
    # Use lxml parser for better encoding handling
    return BeautifulSoup(html_content, "lxml")


# Title:  ensures Types


# Define a function to convert the 'class' attribute to a list
def get_class_list(tag: Tag) -> list[str]:
    """
    Retrieve the 'class' attribute from a Tag and convert it to a list of strings.

    Parameters:
        tag (Tag): A BeautifulSoup Tag object.

    Returns:
        list[str]: A list of class names. Returns an empty list if no classes are present.

    Writtin at 📅 2024-09-15 20:27:54
    """
    class_attr = tag.get("class")

    if isinstance(class_attr, str):
        return [class_attr]
    elif isinstance(class_attr, list):
        return class_attr
    else:
        return []


def get_attribute_as_list(tag: Tag, attribute: str) -> Optional[list[str]]:
    """
    Retrieve an attribute from a Tag and ensure the result is a list of strings.

    Parameters:
        tag (Tag): A BeautifulSoup Tag object.
        attribute (str): The attribute name to retrieve.

    Returns:
        Optional[list[str]]: A list of strings if the attribute is found, or None if the attribute is not present.
    """
    attr_value = tag.get(attribute)

    # If the attribute is a string, convert it to a single-item list
    if isinstance(attr_value, str):
        return [attr_value]
    # If the attribute is already a list of strings, return it
    elif isinstance(attr_value, list):
        return attr_value
    # If the attribute does not exist or is None, return None
    return None
