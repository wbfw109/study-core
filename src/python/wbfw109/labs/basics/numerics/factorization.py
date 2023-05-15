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


# TODO: sieve of atkin
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
        return any(num % i == 0 for i in range(2, int(num**0.5) + 1))

    def decompose_primes(self, num: int) -> Counter[int]:
        prime_factors: Counter[int] = Counter()
        num_temp: int = num
        for i in range(2, int(num**0.5) + 1):
            while True:
                quotient, remainder = divmod(num_temp, i)
                if remainder == 0:
                    num_temp = quotient
                    prime_factors[i] += 1
                else:
                    break
        else:
            if num_temp != 1:
                prime_factors[num_temp] += 1

        return prime_factors

    def get_primes_by_using_sieve_of_eratosthenes(self, max_range: int) -> list[int]:
        """
        Time Complexity: O(n log (log n))
            🔍 proof is complex.
        Space Complexity: O(n)
        """
        # <sieve> denotes i-th index (number) is prime or composite.
        sieve: list[bool] = [True] * (max_range + 1)
        sieve[0] = False
        sieve[1] = False
        for i in range(2, int(max_range**0.5) + 1):
            if sieve[i]:
                # Arithmetic sequence loop with a common difference equal to the prime number
                # , starting from the square of that prime number and up to the <max_range>.
                for j in range(i**2, max_range + 1, i):
                    sieve[j] = False
        return [i for i in range(max_range + 1) if sieve[i]]

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
        num: int = random.randint(10, 100)
        count_of_primes_to_num: int = len(
            integer_factorization.get_primes_by_using_sieve_of_eratosthenes(num)
        )
        integer_factorization.append_line_into_df_in_wrap(
            [
                "Count of primes to num",
                "num, count_of_primes_to_num",
                f"{num}, {count_of_primes_to_num}",
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
