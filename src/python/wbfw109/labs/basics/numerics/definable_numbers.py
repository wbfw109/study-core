"""This file includes `Positional notation` category."""
# %%
from __future__ import annotations

import math
import random

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    VisualizationManager,
    VisualizationRoot,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


class DigitSum(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["name", "eval", "print"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "-"

    # TODO: rule of nines


## Rationale Number
class DecimalRepresentation(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["function", "eval", "print"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "-"

    def has_repeating_decimal(self, /, numerator: int, denominator: int) -> bool:
        """Check if a fraction (numerator/denominator) has a repeating decimal."""
        denominator //= math.gcd(numerator, denominator)

        # Factor out all the 2s and 5s from the denominator
        for x in (2, 5):
            while True:
                quotient, remainder = divmod(denominator, x)
                if remainder == 0:
                    denominator = quotient
                else:
                    break

        # If the remaining denominator is 1, the division results in a finite floating point
        return denominator != 1

    @classmethod
    def test_case(cls):
        decimal_representation: DecimalRepresentation = cls()
        denominator: int = random.randint(3, 50)
        numerator: int = random.randint(2, denominator + 25)
        has_repeating_decimal: bool = decimal_representation.has_repeating_decimal(
            numerator, denominator
        )

        decimal_representation.append_line_into_df_in_wrap(
            [
                "Has repeating decimal",
                "retval, (numerator, denominator)",
                f"{has_repeating_decimal}, ({numerator}, {denominator})",
            ]
        )

        decimal_representation.visualize()


class PositionalNotation(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["function", "eval", "print"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "-"

    def convert_decimal_to_base(self, num: int, base: int) -> str:
        BASE_MAP: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result: list[str] = []
        while num > 0:
            quotient, remainder = divmod(num, base)
            result.append(BASE_MAP[remainder])
            num = quotient
        return "".join(reversed(result))

    def convert_base_to_decimal(self, num_str: str, base: int) -> int:
        return int(num_str, base)

    def convert_decimal_to_one_based_number(self, num: int, base_map: str) -> str:
        """🔍"""
        base: int = len(base_map)
        result: list[str] = []
        while num > 0:
            num -= 1
            quotient, remainder = divmod(num, base)
            result.append(base_map[remainder])
            num = quotient
        return "".join(reversed(result))

    @classmethod
    def test_case(cls):
        positional_notation: PositionalNotation = cls()

        num = 255
        base = 7
        converted_num: str = positional_notation.convert_decimal_to_base(num, base)
        reconverted: int = positional_notation.convert_base_to_decimal(
            converted_num, base
        )
        assert reconverted == num
        positional_notation.append_line_into_df_in_wrap(
            [
                "Decimal to 2, 8, 16 base",
                "( 11, bin(11), oct(11), hex(11) )",
                (11, bin(11), oct(11), hex(11)),
            ]
        )

        positional_notation.visualize()


# %%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [PositionalNotation]
    VisualizationManager.call_root_classes(only_class_list=only_class_list)
