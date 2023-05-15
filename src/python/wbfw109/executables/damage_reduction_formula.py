# %%
from __future__ import annotations

from typing import Callable, Iterable

from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# %doctest_mode

# from google.protobuf.internal import api_implementation
# print(api_implementation.Type())


# %% # title: Declaration and Crawled tab

# from source https://imgur.com/of3URIu
# https://www.reddit.com/r/MinecraftDungeons/comments/1bsib6o/got_a_question_for_yall_about_damage_reduction/

armored_damage_reductions = [90, 15]
non_armored_damage_reductions = [
    40,
]

sum_x: Callable[[int], float] = lambda x: x / (100 - x)
result1 = 1 / (sum(map(sum_x, armored_damage_reductions)) + 1)


def mul_elements(y: Iterable[int]) -> float:
    total: float = 1.0
    for x in y:
        total *= (100 - x) / 100
    return total


result2 = mul_elements(non_armored_damage_reductions)

result = (1 - (result1 * result2)) * 100
print(f"Damage reduction as {result}%")

# %%
