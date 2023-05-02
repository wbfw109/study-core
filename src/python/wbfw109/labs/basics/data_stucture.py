# pylint: disable=C3002,R1721
# %%
from __future__ import annotations

import sys
from array import array
from enum import Enum
from queue import Queue
from typing import Any

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import VisualizationRoot  # type: ignore

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode
# %%


class ObjectTypes:
    """Default types"""

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
            self.integer: int = 0
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
            self.tuple_: tuple = ()  # type:ignore
            self.set: set[Any] = set()
            self.list: list[Any] = []
            self.stack: list[Any] = self.list.copy()  # LIFO
            self.dict: dict[Any, Any] = {}
            self.queue: Queue[Any] = Queue()  # FIFO

        def test_stack_operations(self) -> None:
            self.stack.extend([1, 3, 5, 7])
            self.stack.append(10)
            print(
                "stack: LIFO (push, pop)",
                [self.stack.pop() for _ in range(len(self.stack))],
            )

        def test_queue_operations(self) -> None:
            self.queue.put(1)
            self.queue.put(3)
            self.queue.put(5)
            print(
                "queue: FIFO (put, get)",
                [self.queue.get() for _ in range(self.queue.qsize())],
            )


class ListDT(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "-"

    def profile_list_of_integers_in_step(self) -> None:
        """

        [Object memory size]
        -----
        - 'list(range(1, n+1, 2))'    ; 🪟 Win
            Since the range object is generated in a lazy manner and range object has a known length, Python can preallocate the 🚣 exact amount of memory required for the list when converting the range object to a list.
        - '[i for i in range(1, n + 1, 2)]'
            When using list comprehensions, Python might 🚣 over-allocate memory for the list to improve performance when appending elements.
            This means that the list comprehension may reserve more memory than actually needed, anticipating that the list might grow in size later.
            Python might allocate more memory than necessary and then trim the excess capacity when the list comprehension is finished.
        ➡️ The difference in memory usage is not related to the object's header, as both lists have the same type and the header information will be similar.
            Instead, the difference is in the payload, which is the actual data stored in the list.
        """
        n = 1000000
        result1 = (lambda n: list(range(1, n + 1, 2)))(n)
        result2 = (lambda n: [i for i in range(1, n + 1, 2)])(n)

        # output: 4000056 bytes  <  4167352 bytes
        print(sys.getsizeof(result1) < sys.getsizeof(result2))  # True

    @classmethod
    def test_case(cls):
        list_dt: ListDT = cls()
        list_dt.profile_list_of_integers_in_step()

        # dict_data_type.visualize()


class DictDT(VisualizationRoot):
    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self,
            columns=["eval", "print", "note"],
            has_df_lock=False,
            should_highlight=True,
        )

    def __str__(self) -> str:
        return "-"

    @classmethod
    def test_case(cls):
        dict_dt: DictDT = cls()

        ## Snippet: key update as min value without using not using
        d: dict[Any, Any] = {}
        key, new_value = "A", 10
        if d.get(key, sys.maxsize) > new_value:
            d[key] = new_value

        # dict_data_type.visualize()


# ObjectTypes.AbstractDataTypes().test_queue_operations()

# dequeue 랑 토폴로지, https://docs.python.org/3/library/graphlib.html 이거 하고 알고리즘 풀기.
# https://docs.python.org/3/library/collections.html#collections.deque
# Critical path method
# import heapq
# Disjoint-set for Kruskal's algorithm
# [[1, 2, 3]] * 2, [1, 2, 3] * 2     // * (repetition operation)
# xx = "Cursor2"
# tuple(xx)  vs  (xx,)
# [1, 2, 3] + [4]   -> concatenation operation: create new list
# [1, 2, 3].append(4) -> in-place update
# x = [i for i in range(5)]
# x[100:3]
# x[3 : 1000]
# x[-1:2]
# Also valid:   list(map(sum, zip(card_stack_list[::2], card_stack_list[1::2])))

# heap 이 동적으로 생성되고 삭제되는 데이터에서 이 데이터들끼리 비교가 필요할 때 사용하면 좋다. 그리디 알고리즘에서 자주 사용된다.
# heapq 는 작업할떄마다 자동으로 힙정렬되므로 사용하기 좋다.
# sequence: 반복이 가능하고 순서가 중요한 객체의 열거된 모음. -> Subsequence


def data_structure():
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


# %%
# if __name__ == "__main__" or VisualizationManager.central_control_state:
#     if VisualizationManager.central_control_state:
#         # Do not change this.
#         only_class_list = []
#     else:
#         only_class_list = [DictDT]
#     VisualizationManager.call_root_classes(only_class_list=only_class_list)
