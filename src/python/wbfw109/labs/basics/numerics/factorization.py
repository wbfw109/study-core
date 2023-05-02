# %%
from __future__ import annotations

import random
from collections import Counter

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    VisualizationManager,
    VisualizationRoot,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


# TODO: (Quadratic sieve, Lenstra elliptic-curve factorization) for large numbers.
# 🔍 Does decision problem version of this have weakly polynomial time?
class IntegerFactorization(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["function", "eval", "print"],
            has_df_lock=False,
            should_highlight=True,
        )
        self.df_caption = [
            "⚙️ Complexity of 'Find divisors by square root' method",
            "  - Time: O(sqrt(n)); pseudo polynomial",
            "  - Space: O(n)",
        ]

    def __str__(self) -> str:
        return "-"

    def find_divisors_by_square_root(self, num: int) -> list[int]:
        divisors: list[int] = []
        for i in range(1, int(num**0.5) + 1):
            quotient, remainder = divmod(num, i)
            if remainder == 0:
                divisors.append(i)
                # calculate counterpart of the divisor
                if i != quotient:  # To avoid duplicated factors like 5*5
                    divisors.append(quotient)
        return divisors

    def has_composite_number(self, num: int) -> bool:
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return True
        return False

    def decompose_primes(self, num: int) -> Counter[int]:
        prime_factors: Counter[int] = Counter()
        num_temp: int = num
        for i in range(2, int(num**0.5) + 1):
            while True:
                quotient, remainder = divmod(num_temp, i)
                if remainder != 0:
                    break
                else:
                    num_temp = quotient
                    prime_factors[i] += 1
        else:
            if num_temp != 1:
                prime_factors[num_temp] += 1

        return prime_factors

    @classmethod
    def test_case(cls):
        integer_factorization: IntegerFactorization = cls()

        num: int = random.randint(24, 28)
        divisors = integer_factorization.find_divisors_by_square_root(num)
        integer_factorization.append_line_into_df_in_wrap(
            [
                "Find divisors by square root",
                "num, divisors",
                f"{num}, {divisors}",
            ]
        )

        num: int = random.randint(10, 100)
        is_composite_number = integer_factorization.has_composite_number(num)
        integer_factorization.append_line_into_df_in_wrap(
            [
                "Is composite number?",
                "num, retval",
                f"{num}, {is_composite_number}",
            ]
        )

        num: int = random.randint(10, 100)
        prime_factors = integer_factorization.decompose_primes(num)
        integer_factorization.append_line_into_df_in_wrap(
            [
                "Decompose primes",
                "num, prime_factors",
                f"{num}, {prime_factors}",
            ]
        )

        integer_factorization.visualize()


# %%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [IntegerFactorization]
    VisualizationManager.call_root_classes(only_class_list=only_class_list)
