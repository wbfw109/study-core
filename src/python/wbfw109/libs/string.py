"""String manipulation"""
import re


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
