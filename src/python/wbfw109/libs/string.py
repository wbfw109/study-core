"""String manipulation"""

# title: run for pytest %shell> pytest --doctest-modules src/python/wbfw109/libs/string.py -v

import re
import unicodedata


# Writtin at 📅 2024-09-15 23:52:35
def camel_or_pascal_to_snake(value: str):
    """
    Converts a CamelCase or PascalCase string to snake_case.

    The function operates by finding positions in the string where an uppercase
    letter follows a lowercase letter or another uppercase letter (except the first
    character of the string). It inserts an underscore (`_`) before each of these
    uppercase letters to separate them, and then converts the entire string to
    lowercase to produce a valid snake_case output.

    ❔ Mechanism:
    - The regular expression pattern `(?<!^)(?=[A-Z])` consists of two parts:
      1. `(?<!^)`: A negative lookbehind that ensures the match is not at the start
         of the string. This prevents an underscore from being added before the
         first character, even if it is uppercase.
      2. `(?=[A-Z])`: A positive lookahead that looks for positions in the string
         where the next character is an uppercase letter.
    - When such a position is found (before each uppercase letter that is not the
      first character), the `re.sub()` function inserts an underscore (`_`) at that
      position.
    - Finally, the resulting string is converted to lowercase to match the snake_case
      format.

    Example:
        >>> camel_or_pascal_to_snake("Composable_Range_Views_Ranges_C++20/23")
        'composable_range_views_ranges_cpp20_23'
        >>> camel_or_pascal_to_snake("camelCaseExample")
        'camel_case_example'

    Args:
        name (str): The input string in CamelCase or PascalCase format.

    Returns:
        str: The converted string in snake_case format.
    """
    snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    return snake_case


# Writtin at 📅 2024-09-15 20:09:05
def replace_special_chars(value: str) -> str:
    """
    Replace special characters with predefined replacements.

    Parameters:
        value (str): The original string.

    Returns:
        str: The string with special characters replaced.
    """
    manual_replacements = {
        "++": "pp",
        "#": "sharp",
        "&": "and",
        "/": "_",  # Replace slashes with underscores
        "@": "at",  # Replace @ with "at"
        ":": "_",  # Remove colons
        ";": "",  # Remove semicolons
        "(": "",  # Remove parentheses
        ")": "",
        "[": "",
        "]": "",
        "<": "_",
        ">": "_",
        " ": "_",  # Spaces should be turned into underscores
        ".": "",  # Remove periods
        ",": "",  # Remove commas
    }

    # Replace all occurrences based on the manual_replacements dictionary
    for key, replacement in manual_replacements.items():
        value = value.replace(key, replacement)

    return value


# Writtin at 📅 2024-09-15 20:09:05
def slugify(value: str, allow_unicode: bool = False) -> str:
    """
    Convert a string into a slug format, allowing for manual replacements
    of special characters.

    This function replaces symbols like `++` with more user-friendly equivalents
    (e.g., `pp`), removes unwanted characters, and transforms the string into
    a format suitable for file names or URLs.

    Parameters:
        value (str): The string to slugify.
        allow_unicode (bool): If True, allow unicode characters.

    Returns:
        str: The slugified string.

    Example:
        >>> slugify("Composable_Range_Views_Ranges_C++20/23")
        'composable_range_views_ranges_cpp20_23'
    """
    # First, apply manual replacements
    value = replace_special_chars(value)

    # Normalize the value (convert to ASCII if needed)
    value = str(value)
    if allow_unicode:
        # NFKC (Normalization Form Compatibility Composition)
        value = unicodedata.normalize("NFKC", value)
    else:
        # NFKD (Normalization Form Compatibility Decomposition)

        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )

    # Replace any remaining unwanted characters
    value = re.sub(r"[^\w\s-]", "", camel_or_pascal_to_snake(value))

    # Replace spaces and hyphens with underscores, and strip leading/trailing underscores
    value = re.sub(r"[-\s_]+", "_", value).strip("_")

    return value


def rename_to_snake_case(
    string: str, delimiter_alias: str = r"-. _/", conjunction: str = "_"
) -> str:
    """If <delimiter_alias> are existing before uppercase, it just replace uppercase with lowercase while do nothing to <delimiter_alias>.

    Args:
        string (str): target string
        delimiter_alias (str, optional): Note that the value must not be zero length.
        conjunction (str, optional): Defaults to "_".

    Returns:
        str: renamed string
    """

    return conjunction.join(
        re.sub(
            rf"\s*([{delimiter_alias}])_([A-Z]+)\s*",
            r"\1\2",
            re.sub(r"(?<!^)(?=[A-Z])", conjunction, string),
        ).split()
    ).lower()


def rename_to_snake_case_with_replace(
    string: str, delimiter_alias: str = r"-.?!,;:\"'", conjunction: str = "_"
) -> str:
    """It replaces <delimiter_alias> with <conjunction>

    Args:
        string (str): target string
        delimiter_alias (str, optional): 📝 Note that the value must not be zero length.
        conjunction (str, optional): Defaults to "_".

    Returns:
        str: renamed string
    """
    return conjunction.join(
        re.sub(
            rf"{conjunction}\s*([A-Z]+)\s*",
            r" \1",
            re.sub(f"([{delimiter_alias}])", conjunction, string),
        ).split()
    ).lower()
