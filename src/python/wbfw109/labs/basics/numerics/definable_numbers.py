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
            columns=["name", "eval", "print"],
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


# %%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [DecimalRepresentation]
    VisualizationManager.call_root_classes(only_class_list=only_class_list)
