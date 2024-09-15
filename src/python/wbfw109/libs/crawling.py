from typing import Optional, Union

from bs4.element import Tag


# Writtin at 📅 2024-09-15 20:27:54
# Define a function to convert the 'class' attribute to a list
def get_class_list(tag: Tag) -> list[str]:
    """
    Retrieve the 'class' attribute from a Tag and convert it to a list of strings.

    Parameters:
        tag (Tag): A BeautifulSoup Tag object.

    Returns:
        list[str]: A list of class names. Returns an empty list if no classes are present.
    """
    class_attr: Optional[Union[str, list[str]]] = tag.get("class")

    if isinstance(class_attr, str):
        return [class_attr]
    elif isinstance(class_attr, list):
        return class_attr
    else:
        return []
