"""Written at 📅 2024-09-25 23:54:31
Not use BeatifulSoup. use playwright
"""

from playwright.async_api import ElementHandle


async def extract_direct_text(tag: ElementHandle) -> str:
    """
    Extracts the direct (non-recursive) text content from an HTML element.

    This function uses Playwright's `evaluate()` to execute JavaScript in the browser context
    and retrieve the text content from the immediate element, without including text from any
    child elements or nested structures. If there is no direct text in the element, it returns
    an empty string.

    Args:
        tag (ElementHandle): The handle of the element from which to extract the direct text content.

    Returns:
        str: The trimmed direct text content of the element if it exists, otherwise an empty string.

    Example:
        # Assuming 'tag' is an ElementHandle representing a DOM element
        text = await extract_direct_text(tag)
        print(text)  # Outputs the direct text content, excluding text from child elements
    """
    return await tag.evaluate(
        '(element) => Array.from(element.childNodes).filter(node => node.nodeType === Node.TEXT_NODE).map(node => node.textContent.trim()).join("")'
    )


async def get_tag_name(tag: ElementHandle) -> str:
    """
    Retrieves the tag name (e.g., 'a', 'div', 'p') of the specified HTML element.

    Args:
        tag (ElementHandle): The handle of the element whose tag name is to be extracted.

    Returns:
        str: The tag name of the element in lowercase.
    """
    return await tag.evaluate("(element) => element.tagName.toLowerCase()")
