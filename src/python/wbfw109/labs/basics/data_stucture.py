# pylint: disable=C3002,R1721
# References: https://docs.python.org/3/library/stdtypes.html
# %%
from __future__ import annotations

import dis
import sys
import timeit
from array import array
from collections import ChainMap, defaultdict
from enum import Enum
from queue import Queue
from typing import Any

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    VisualizationManager,
    VisualizationRoot,
    display_data_frame_with_my_settings,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


# Title: Numeric Types ~
class IntDT(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["operation", "result"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "Primitive type: Integer"

    @staticmethod
    def test_binary_state_technique():
        visualization = VisualizationRoot(
            columns=["name", "eval", "print"],
            has_df_lock=False,
            should_highlight=True,
            header_string="Binary state technique in graph",
        )

        visualization.append_line_into_df_in_wrap(
            [
                "Ternary operator",
                "conditional_expression ::=  or_test ['if' or_test 'else' expression]",
            ]
        )
        visualization.append_line_into_df_in_wrap()
        visualization.append_line_into_df_in_wrap(
            [
                "Is odd or Is even?",
                "n & 1, not n & 1",
            ]
        )
        visualization.append_line_into_df_in_wrap(
            [
                "Two state indexing",
                "arr[n & 1].append(x)",
            ]
        )
        visualization.append_line_into_df_in_wrap()
        visualization.append_line_into_df_in_wrap(
            [
                "(X := 2^n) representation",
                "1 << n",
            ]
        )
        visualization.append_line_into_df_in_wrap(
            [
                "(X := divided as 2^n) representation",
                "X >> n",
            ]
        )

        display_data_frame_with_my_settings(
            visualization.df, caption=visualization.df_caption
        )

    @staticmethod
    def test_mask_technique_in_graph():
        visualization = VisualizationRoot(
            columns=["name", "eval", "print"],
            has_df_lock=False,
            should_highlight=True,
            header_string="Mask technique in graph",
        )
        visualization.df_caption = [
            "⚠️ This technique is valid when:",
            "  - vertices are Integer equal or greater than 0.",
            "  - vertex 0 is not used (🚣 if that is used, ** operation must be used instead of bitwise operation.).",
        ]

        n: int = 4
        visualization.append_line_into_df_in_wrap(["n (vertices)", "n = 4"])
        visualization.append_line_into_df_in_wrap(
            [
                "routes (bitwise)",
                "[format(x, 'b') for x in range(1, 1 << n - 1)]",
                [format(x, "b") for x in range(1, 1 << n - 1)],
            ]
        )
        visualization.append_line_into_df_in_wrap(
            [
                "routes (actual)",
                "list(range(1, 1 << n - 1))",
                list(range(1, 1 << n - 1)),
            ]
        )
        visualization.append_line_into_df_in_wrap(
            [
                "vertices (bitwise)",
                "[format(1 << v - 1, 'b') for v in range(1, n)]",
                [format(1 << v - 1, "b") for v in range(1, n)],
            ]
        )
        visualization.append_line_into_df_in_wrap(
            ["vertices (actual)", "list(range(1, n))", list(range(1, n))]
        )
        visualization.append_line_into_df_in_wrap()
        visualization.append_line_into_df_in_wrap(
            ["❔ How to check <vertex> in <route>", "1 << v - 1 & route != 0"]
        )
        visualization.append_line_into_df_in_wrap(
            [
                "❔ Traversed Vertices in <route>",
                "[v for v in range(1, n) if 1 << (v - 1) & route]",
            ]
        )
        visualization.append_line_into_df_in_wrap(
            ["❔ How to exclude a <vertex> in <route>", "route & ~(1 << v - 1)"]
        )

        display_data_frame_with_my_settings(
            visualization.df, caption=visualization.df_caption
        )

    @classmethod
    def test_case(cls):
        # int_dt: IntDT = cls()

        # int_dt.visualize()
        IntDT.test_binary_state_technique()
        IntDT.test_mask_technique_in_graph()


# Title: Sequence Types ~
class SequenceDT(VisualizationRoot):
    """Sequence Types - list, tuple, range, str, bytes, bytearray, memoryview"""

    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["operation", "result"],
            has_df_lock=False,
            should_highlight=True,
        )
        self.df_caption = [
            "⚙️ about Slicing",
            "  - If i or j is negative, the index is relative to the end of sequence s: len(s) + i or len(s) + j is substituted. 🚣 But note that -0 is still 0.",
        ]

    def __str__(self) -> str:
        return "-"

    @staticmethod
    def test_negative_index_technique():
        visualization = VisualizationRoot(
            columns=["function", "eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
            header_string="Negative index technique",
        )
        visualization.df_caption = [
            "⚙️ [Slicing]",
            "  - If i or j is negative, the index is relative to the end of sequence s: len(s) + i or len(s) + j is substituted. 🚣 But note that -0 is still 0.",
            "",
            "⚙️ 💡 [Optimized Memoization in Modular Operation with Negative Indexing and Bijection]",
            "  - [Motivation] to complement aspect where cost of division and modular operation is expensive.",
            "  - [Condition] ⚠️ This technique is valid when (0 ≤  (a, nn)  < n)",
        ]

        word = "0123456789"
        s, e = 0, 7
        visualization.append_line_into_df_in_wrap(
            [
                "Negative step",
                "(word[e:s:-1], word[e: None if s==0 else e: -1])",
                (word[e:s:-1], word[e : None if s == 0 else e : -1]),
                "# word = '0123456789'; s, e = 0, 7",
            ]
        )

        n: int = 5
        a = 2
        remainders = list(range(n))
        visualization.append_line_into_df_in_wrap(
            [
                "Optimized Memoization in Modular Operation",
                "(n, remainders, a)",
                (n, remainders, a),
            ]
        )
        visualization.append_line_into_df_in_wrap(
            [
                "Optimized Memoization in Modular Operation",
                "all((remainders[(a + nn) % n] == remainders[(a + nn) - n] for nn in range(n)))",
                all(
                    (
                        remainders[(a + nn) % n] == remainders[(a + nn) - n]
                        for nn in range(n)
                    )
                ),
            ]
        )

        display_data_frame_with_my_settings(
            visualization.df, caption=visualization.df_caption
        )

    @classmethod
    def test_case(cls):
        sequence_dt: SequenceDT = cls()

        ## common_sequence_operation
        sequence_dt.append_line_into_df_in_wrap(["s + t", "concatenation"])
        sequence_dt.append_line_into_df_in_wrap(["s * n  or  n * s", "multiplication"])
        sequence_dt.append_line_into_df_in_wrap(["s[i]", "i-th item of s, origin 0"])
        sequence_dt.append_line_into_df_in_wrap(["s[i:j]", "slice of s from i to j"])
        sequence_dt.append_line_into_df_in_wrap(
            ["s[i:j:k]", "slice of s from i to j with step k"]
        )
        sequence_dt.append_line_into_df_in_wrap()
        sequence_dt.append_line_into_df_in_wrap(
            [
                "s.index(x[, i[, j]])",
                "index of the first occurrence of x in s (at or after index i and before index j)",
            ]
        )
        sequence_dt.append_line_into_df_in_wrap(
            ["s.count(x)", "total number of occurrences of x in s"]
        )
        sequence_dt.append_line_into_df_in_wrap(
            [
                "( min(s), max(s) )",
                "( smallest item of s, largest item of s )",
            ]
        )

        sequence_dt.visualize()
        SequenceDT.test_negative_index_technique()


class TupleDT(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "ADT: Finite sequence type; ImmutableSequence"

    @classmethod
    def test_case(cls):
        tuple_dt: TupleDT = cls()

        word = "word"
        tuple_dt.append_line_into_df_in_wrap(["", "", "# word = 'word'"])
        tuple_dt.append_line_into_df_in_wrap(["(word,)", (word,)])
        tuple_dt.append_line_into_df_in_wrap(["(word)", (word)])
        tuple_dt.append_line_into_df_in_wrap(["tuple(word)", tuple(word)])
        tuple_dt.visualize()


class ListDT(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["operation", "result"],
            has_df_lock=False,
            should_highlight=True,
        )
        self.df_caption = ["It can be used as Stack (LIFO)"]

    def __str__(self) -> str:
        return "ADT: Finite sequence type; MutableSequence"

    def profile_list_whose_elements_are_integers_in_arithmetic_progression(
        self, /, *, only_conclusion: bool
    ) -> str:
        """
        [Object memory size]
        - 'list(range(1, n+1, 2))'    ; 🥇 Win
            - Since the range object is generated in a lazy manner and range object has a known length, Python can preallocate the 🚣 exact amount of memory required for the list when converting the range object to a list.
        - '[i for i in range(1, n + 1, 2)]' ; 🥈
            - When using list comprehensions, Python may 🚣 over-allocate memory to improve the appending performance
                , meaning it reserves more memory than actually needed with the anticipation of potential list growth
                , and then trims the excess capacity once the list comprehension is completed.
        ----
        - ➡️ The difference in memory usage is not related to the object's header, as both lists have the same type and the header information will be similar.
            - Instead, the difference is in the payload, which is the actual data stored in the list.
        """
        if only_conclusion:
            return "\n".join(
                [
                    "⌛ [Profile conclusion: list whose elements are integers in arithmetic progression] 🔪 Object memory size",
                    "  `list(range(1, n+1, 2))` occupies less memory than `[i for i in range(1, n + 1, 2)]`.",
                ]
            )

        n = 1000000
        result1 = (lambda n: list(range(1, n + 1, 2)))(n)
        result2 = (lambda n: [i for i in range(1, n + 1, 2)])(n)

        # output: 4000056 bytes  <  4167352 bytes
        print(sys.getsizeof(result1) < sys.getsizeof(result2))  # True
        return ""

    def profile_list_multiplication_and_list_comprehension(
        self, /, *, only_conclusion: bool
    ) -> str:
        """
        [Speed]
        - List multiplication    ; 🥇 Win
            - list multiplication is a single operation that's implemented at the C level in the Python interpreter.
                - This operation is very efficient because it can allocate memory for the entire list at once and then fill this memory with references to the same object.
            - Since the range object is generated in a lazy manner and range object has a known length, Python can preallocate the 🚣 exact amount of memory required for the list when converting the range object to a list.
        - List comprehension ; 🥈
            - list comprehension involves a loop that creates a new integer and appends it to the list on each iteration.
                - This involves more Python bytecode operations.
        ----
        - ➡️ we can see that List multiplication involves fewer and simpler operations than List comprehension, which corresponds to its faster execution speed,
            , by using the dis module to disassemble the bytecode of these methods

        """
        if only_conclusion:
            return "\n".join(
                [
                    "⌛ [Profile conclusion: list multiplication and list comprehension] 🔪 Speed",
                    "  list multiplication is faster than list comprehension (in the case where value of elements are same).",
                    "  ⚠️ but Note that list multiplication refer same object (so cases where elements are Mutable may not be valid.).",
                ]
            )

        ITERATION: int = 1000000
        range_obj = range(ITERATION)

        def method1():
            [0] * ITERATION  # type: ignore

        def method2():
            [0 for _ in range_obj]

        print("===== List multiplication =====")
        dis.dis(method1)
        print("\n===== List comprehension =====")
        dis.dis(method2)

        results = [timeit.timeit(x, number=100) for x in (method1, method2)]
        print(f"\n\nList multiplication: {results[0]}")  # 0.32677s
        print(f"List comprehension: {results[1]}")  # 2.91321s

        return ""

    def profile_list_consecutive_pop_and_del_at_end(
        self, /, *, only_conclusion: bool
    ) -> str:
        """
        [Speed]
        - List del    ; 🥇 Win
            - all elements from the end of the list are deleted in one go, without any need for shifting, which makes it faster.
        - List consecutive pop ; 🥈
            - pop() in loop deletes the last element one by one.

        """
        if only_conclusion:
            return "\n".join(
                [
                    "⌛ [Profile conclusion: list consecutive pop and List del at end] 🔪 Speed",
                    "  List del is faster than list consecutive pop at end.",
                ]
            )
        arr = list(range(100000))
        unit = 4
        range_obj = range(unit)

        def method1():
            arr2 = arr.copy()
            while arr2:
                del arr2[-unit:]

        def method2():
            arr2 = arr.copy()
            while arr2:
                for _ in range_obj:
                    arr2.pop()

        results = [timeit.timeit(x, number=100) for x in (method1, method2)]
        print(f"List del: {results[0]}")  # 0.23572s
        print(f"List consecutive pop: {results[1]}")  # 0.40983s

        return ""

    @classmethod
    def test_case(cls):
        list_dt: ListDT = cls()
        list_dt.append_line_into_df_in_wrap(
            [
                "s.append(x)",
                "appends x to the end of the sequence (🚣 same as s[len(s):len(s)] = [x])",
            ]
        )
        list_dt.append_line_into_df_in_wrap(
            [
                "s.insert(i, x)",
                "inserts x into s at the index given by i (🚣 same as s[i:i] = [x])",
            ]
        )
        list_dt.append_line_into_df_in_wrap(
            [
                "s.extend(t) or s += t",
                "extends s with the contents of t (🚣 for the most part the same as s[len(s):len(s)] = t)",
            ]
        )
        list_dt.append_line_into_df_in_wrap(
            [
                "s[i:j:k] = t",
                "the elements of s[i:j:k] are replaced by those of t",
            ]
        )
        list_dt.append_line_into_df_in_wrap(
            [
                "s *= n",
                "updates s with its contents repeated n times",
            ]
        )
        list_dt.append_line_into_df_in_wrap()

        list_dt.append_line_into_df_in_wrap(
            [
                "s.reverse()",
                "reverses the items of s in place",
            ]
        )

        print(
            list_dt.profile_list_whose_elements_are_integers_in_arithmetic_progression(
                only_conclusion=True
            )
        )
        print(
            list_dt.profile_list_multiplication_and_list_comprehension(
                only_conclusion=True
            )
        )
        print(list_dt.profile_list_consecutive_pop_and_del_at_end(only_conclusion=True))
        list_dt.visualize()


# Title: Mapping Types ~
class DictDT(VisualizationRoot):
    """References
    - SF: PEP 584 – Add Union Operators To dict ; https://peps.python.org/pep-0584/"""

    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["function", "eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )
        self.df_caption = [
            "⚙️ collections.Chainmap 🆚 Union operator",
            "  - A ChainMap groups multiple dicts or other mappings together to create a single, updatable 🚣 view",
            "  - Dict union will return a 🚣 new dict consisting of the left operand merged with the right operand, each of which must be a dict (or an instance of a dict subclass).",
            "    - The augmented assignment version operates 🚣 in-place; 🛍️ e.g. dict_1 |= dict_2",
        ]

    def __str__(self) -> str:
        return "ADT: Mapping type"

    class Snippets:
        @staticmethod
        def update_value_as_min_value_without_min_func() -> None:
            d: dict[Any, Any] = {}
            key, new_value = "A", 10
            if d.get(key, sys.maxsize) > new_value:
                d[key] = new_value

    @classmethod
    def test_case(cls):
        dict_dt: DictDT = cls()

        dict_1 = {"A": 1, "cheese": 10}
        dict_2 = {"cheese": "breakfast", "B": "No"}
        dict_dt.append_line_into_df_in_wrap(
            [
                "Union To dict",
                "",
                "",
                "# dict_1 = {'A': 1, 'cheese': 10}; dict_2 = {'cheese': 'breakfast', 'B': 'No'}",
            ]
        )
        dict_dt.append_line_into_df_in_wrap(
            ["Union To dict", "dict_1 | dict_2", dict_1 | dict_2, "key: Last seen wins"]
        )
        dict_dt.append_line_into_df_in_wrap(
            ["Union To dict", "dict_2 | dict_1", dict_2 | dict_1, "key: Last seen wins"]
        )
        dict_dt.append_line_into_df_in_wrap(
            [
                "Union To dict",
                "ChainMap(dict_1, dict_2)",
                ChainMap(dict_1, dict_2),  # type: ignore
                "key: First seen wins",
            ]
        )
        dict_dt.append_line_into_df_in_wrap(
            [
                "Union To dict",
                "ChainMap(dict_2, dict_1)",
                ChainMap(dict_2, dict_1),  # type: ignore
                "key: First seen wins",
            ]
        )

        dict_dt.append_line_into_df_in_wrap()
        dict_dt.append_line_into_df_in_wrap(
            [
                "defaultdict",
                "defaultdict(lambda: '<missing>', [('A', 1), ('B', 2)])['unknown']",
                defaultdict(lambda: "<missing>", [("A", 1), ("B", 2)])["unknown"],  # type: ignore
            ]
        )

        dict_dt.visualize()


# %%

if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = [ListDT]
    VisualizationManager.call_root_classes(only_class_list=only_class_list)


# ObjectTypes.AbstractDataTypes().test_queue_operations()

# dequeue 랑 토폴로지, https://docs.python.org/3/library/graphlib.html 이거 하고 알고리즘 풀기.
# https://docs.python.org/3/library/collections.html#collections.deque
# Critical path method
# import heapq
# Disjoint-set for Kruskal's algorithm
# heap 이 동적으로 생성되고 삭제되는 데이터에서 이 데이터들끼리 비교가 필요할 때 사용하면 좋다. 그리디 알고리즘에서 자주 사용된다.
# heapq 는 작업할떄마다 자동으로 힙정렬되므로 사용하기 좋다.


# TODO: wrap in same category tuple, list -> to sequence type
class ObjectTypes:
    """
    m-ary tree, b tree, B+ tree

    single linked list
    https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes
    TODO:
    https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues
    from collections import deque
        extendleft(iterable)
            Extend the left side of the deque by appending elements from iterable. Note, the series of left appends results in reversing the order of elements in the iterable argument.
    """

    class PrimitiveTypes:
        """
        TODO: ... detail all method...
        """

        class Color(Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        def __init__(self) -> None:
            self.boolean: bool = True
            self.floating_point: float = 0.0
            self.enumerated: ObjectTypes.PrimitiveTypes.Color = self.Color.RED

    class CompositeTypes:
        """
        TODO: ... detail all method...
        """

        def __init__(self) -> None:
            self.string: str = ""
            self.array: array[float] = array("d", [1.0, 2.0, 3.14])

    class AbstractDataTypes:
        """
        TODO: ... detail all method...
        """

        def __init__(self) -> None:
            self.set: set[Any] = set()
            self.dict: dict[Any, Any] = {}
            self.queue: Queue[Any] = Queue()  # FIFO

        def test_queue_operations(self) -> None:
            self.queue.put(1)
            self.queue.put(3)
            self.queue.put(5)
            print(
                "queue: FIFO (put, get)",
                [self.queue.get() for _ in range(self.queue.qsize())],
            )
