# %%
from __future__ import annotations

import itertools
import random
import re

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    VisualizationManager,
    VisualizationRoot,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


class Palindrome(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["function", "eval", "print"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "-"

    def check_palindrome(self, word: str) -> bool:
        """
        🚣 When you call len(word) multiple times, Python has to recompute the length of the string each time, which can add unnecessary overhead
        , especially if the function is being called multiple times in a loop or a critical part of your code.

        range(half_len)
            when len(string) = 4:  2 (i=0 to 1)
            when len(string) = 5:  2 (i=0 to 1)
        range(word_len - 1, half_len - (not word_len & 1), -1)
            when len(string) = 4:  2 (i=3 to 2) -> range(3, 1, -1)
            when len(string) = 5:  2 (i=4 to 3) -> range(4, 2, -1)

            🚣 But because zip() automatically stops if one of iterables is exhausted
                , to use zip directly is faster than others.
                # 1.708735390973743  vs.  2.1405712419946212 ; (25.27% faster when `word = "abbbbabbba"`)
        """
        return all(
            (
                left == right
                for _, left, right in zip(range(len(word) // 2), word, reversed(word))
            )
        )

    @classmethod
    def test_case(cls):
        palindrome: Palindrome = cls()

        word: str = "".join(random.choices("ab", k=random.randint(2, 3)))
        is_palindrome = palindrome.check_palindrome(word)
        palindrome.append_line_into_df_in_wrap(
            [
                "Check palindrome",
                "word, is_palindrome",
                f"'{word}', {is_palindrome}",
            ]
        )
        palindrome.visualize()


class RepeatedSubstring(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["function", "eval", "print"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "-"

    def get_max_period_shortest_naive(self, word: str) -> tuple[str, int]:
        """🔍 Does more Efficient algorithm exists when string size <= 1,000,000? variant of LCS?

        Time complexity: O(n^2)
            - O(n^2) from given regular expression r"(.+?)(?=\1)".
                it captures non-greedy repeated substrings.
                The regex engine may need to perform multiple backtracks to find all the matches.
        Space complexity: O(1)
        """
        pattern = re.compile(r"(.+?)(?=\1)")
        shortest: str = word[0]
        max_repeated: int = 0
        # <consecutive> = pattern.findall(word)
        for substring, g in itertools.groupby(pattern.findall(word)):
            size = sum((1 for _ in g))
            if size > max_repeated:
                max_repeated = size
                shortest = substring
            if size == max_repeated:
                if len(substring) < len(shortest):
                    shortest = substring
        return (shortest, max_repeated + 1)

    @classmethod
    def test_case(cls):
        repeated_substring: RepeatedSubstring = cls()

        word = "dabcdbcdbcdd bcdbcd dcd bb"  # ['bcd', 'bcd', 'd', 'bcd', 'b']
        max_period_shortest_naive = repeated_substring.get_max_period_shortest_naive(
            word
        )
        repeated_substring.append_line_into_df_in_wrap(
            [
                "Max period Shortest repeating substring",
                "word, max_period_shortest_naive",
                f"'{word}', {max_period_shortest_naive}",
            ]
        )

        repeated_substring.visualize()


# %%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [RepeatedSubstring]
    VisualizationManager.call_root_classes(only_class_list=only_class_list)
