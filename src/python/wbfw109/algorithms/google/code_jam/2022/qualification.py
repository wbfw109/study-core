"""refer to algorithms/__init__.py for algorithm Guide"""
from __future__ import annotations

import dataclasses
import itertools
import pprint  # type: ignore
import unittest
from typing import Callable, Final, NamedTuple, Optional

from wbfw109.libs.utilities.self.algorithms import (  # type: ignore
    GoogleCodeJamProblemSolution,
)


class Matrix2DShape(NamedTuple):
    row: int
    column: int


class PunchedCards(GoogleCodeJamProblemSolution[Matrix2DShape]):
    """
    Punched Cards (11pts)
    Type: Array
    Complexity (Time, Space): (O(N), O(1))
    https://codingcompetitions.withgoogle.com/codejam/round/0000000000876ff1/0000000000a4621b
    """

    INITIAL_ODD_LINE_UNIT: str = "+-"
    INITIAL_ODD_LINE_LAST_STR: str = INITIAL_ODD_LINE_UNIT[0]
    INITIAL_EVEN_LINE_UNIT: str = "|."
    INITIAL_EVEN_LINE_LAST_STR: str = INITIAL_EVEN_LINE_UNIT[0]

    def __init__(
        self,
        test_case_number: int = 0,
        OUTPUT_DELIMITER: str = "Case #",
        BUNDLE_NUMBER_IN_ONE_CASE: int = 1,
        PROBLEM_RESOURCE_NAME: str = "punched_cards",
        RUN_FILE_PATH: str = __file__,
    ) -> None:
        super().__init__(
            test_case_number,
            OUTPUT_DELIMITER,
            BUNDLE_NUMBER_IN_ONE_CASE,
            PROBLEM_RESOURCE_NAME,
            RUN_FILE_PATH,
        )

    def input_case_bundle(self, get_contents: Callable[[], str]) -> list[Matrix2DShape]:
        matrix_2d_shape_list: list[Matrix2DShape] = []
        for _ in range(self.BUNDLE_NUMBER_IN_ONE_CASE):
            one_input_length: int = 2
            matrix_2d_shape_list.append(
                Matrix2DShape(*map(int, get_contents().split(" ")[:one_input_length]))
            )
        return matrix_2d_shape_list

    @staticmethod
    def get_solution(dst_bundles: list[Matrix2DShape]) -> str:
        """to punched cards"""
        matrix_2d_shape: Matrix2DShape = dst_bundles[0]
        odd_line: str = (
            PunchedCards.INITIAL_ODD_LINE_UNIT * matrix_2d_shape.column
            + PunchedCards.INITIAL_ODD_LINE_LAST_STR
        )
        even_line: str = (
            PunchedCards.INITIAL_EVEN_LINE_UNIT * matrix_2d_shape.column
            + PunchedCards.INITIAL_EVEN_LINE_LAST_STR
        )
        first_line: str = ".." + odd_line[2:]
        second_line: str = "." + even_line[1:]
        remained_two_lines: str = (
            "\n".join([even_line, odd_line, ""]) * (matrix_2d_shape.row - 1)
        )[:-1]
        return "\n".join(["", first_line, second_line, odd_line, remained_two_lines])


class ThreeDPrinterCartridge(NamedTuple):
    """the number of units of ink"""

    cyan: int
    magenta: int
    yellow: int
    black: int


class ThreeDPrinting(GoogleCodeJamProblemSolution[ThreeDPrinterCartridge]):
    """
    3D Printing (13pts)
    Type: Math
    Complexity (Time, Space): (O(1), O(1))
    https://codingcompetitions.withgoogle.com/codejam/round/0000000000876ff1/0000000000a4672b
    note: There is no indication of whether the picture is possible or impossible, which can lead to confusion, so read the text rather than the picture.

    - Multiple correct solutions
    """

    REQUIRED_EXACT_INK_AMOUNT: Final[int] = 10**6

    def __init__(
        self,
        test_case_number: int = 0,
        OUTPUT_DELIMITER: str = "Case #",
        BUNDLE_NUMBER_IN_ONE_CASE: int = 3,
        PROBLEM_RESOURCE_NAME: str = "3d_printing",
        RUN_FILE_PATH: str = __file__,
    ) -> None:
        super().__init__(
            test_case_number,
            OUTPUT_DELIMITER,
            BUNDLE_NUMBER_IN_ONE_CASE,
            PROBLEM_RESOURCE_NAME,
            RUN_FILE_PATH,
        )

    def input_case_bundle(
        self, get_contents: Callable[[], str]
    ) -> list[ThreeDPrinterCartridge]:
        three_d_printer_cartridge_list: list[ThreeDPrinterCartridge] = []
        for _ in range(self.BUNDLE_NUMBER_IN_ONE_CASE):
            one_input_length: int = 4
            three_d_printer_cartridge_list.append(
                ThreeDPrinterCartridge(
                    *map(
                        int,
                        get_contents().split(" ")[:one_input_length],
                    )
                )
            )
        return three_d_printer_cartridge_list

    @staticmethod
    def get_solution(
        dst_bundles: list[ThreeDPrinterCartridge],
    ) -> str:
        """get valid and available cartridge units"""
        result_msg: str = " IMPOSSIBLE"
        required_amount_by_color: list[int] = []
        each_min_amount_by_color: list[int] = [
            min(amount_by_color) for amount_by_color in zip(*dst_bundles)
        ]

        if sum(each_min_amount_by_color) >= ThreeDPrinting.REQUIRED_EXACT_INK_AMOUNT:
            remained_ink_amount: int = ThreeDPrinting.REQUIRED_EXACT_INK_AMOUNT
            for min_amount_by_color in each_min_amount_by_color:
                used_ink_amount: int = min(min_amount_by_color, remained_ink_amount)
                required_amount_by_color.append(used_ink_amount)
                remained_ink_amount -= used_ink_amount
            result_msg = " " + " ".join(map(str, required_amount_by_color))
        return result_msg

    def judge_acceptance(
        self, my_output: str, file_output: str, *, bundles_index: int
    ) -> None:
        """????[Optional] override. default is AssertEqual"""
        current_bundles: list[ThreeDPrinterCartridge] = self.dst_bundles_list[
            bundles_index
        ]
        if "IMPOSSIBLE" in file_output:
            unittest.TestCase().assertEqual(my_output, file_output)
        else:
            min_cartridge_units_list = [
                min(cartridge) for cartridge in zip(*current_bundles)
            ]
            actual_my_output_list = list(map(int, my_output.split()[2:]))
            unittest.TestCase().assertTrue(
                actual_my_output_list <= min_cartridge_units_list
                and sum(actual_my_output_list)
                == ThreeDPrinting.REQUIRED_EXACT_INK_AMOUNT
            )


class KFacetedDice(NamedTuple):
    dices: list[int]
    most_long_straight: list[int]


class D1000000(GoogleCodeJamProblemSolution[KFacetedDice]):
    """
    d1000000 (9pts, 11pts)
    Type: Sort
    Complexity (Time, Space): (ONlogN, O(1))
    https://codingcompetitions.withgoogle.com/codejam/round/0000000000876ff1/0000000000a46471
    """

    def __init__(
        self,
        test_case_number: int = 0,
        OUTPUT_DELIMITER: str = "Case #",
        BUNDLE_NUMBER_IN_ONE_CASE: int = 1,
        PROBLEM_RESOURCE_NAME: str = "d1000000",
        RUN_FILE_PATH: str = __file__,
    ) -> None:
        super().__init__(
            test_case_number,
            OUTPUT_DELIMITER,
            BUNDLE_NUMBER_IN_ONE_CASE,
            PROBLEM_RESOURCE_NAME,
            RUN_FILE_PATH,
        )

    def input_case_bundle(self, get_contents: Callable[[], str]) -> list[KFacetedDice]:
        k_faceted_dice_list: list[KFacetedDice] = []
        for _ in range(self.BUNDLE_NUMBER_IN_ONE_CASE):
            one_input_length: int = int(get_contents())
            k_faceted_dice_list.append(
                KFacetedDice(
                    dices=sorted(
                        map(
                            int,
                            get_contents().split(" ")[:one_input_length],
                        )
                    ),
                    most_long_straight=[],
                )
            )
        return k_faceted_dice_list

    @staticmethod
    def get_solution(
        dst_bundles: list[KFacetedDice],
    ) -> str:
        """
        Given K-Faceted Dice can be 1 ~ K (natural number), but can not be greater than K + 1
        Then starting from minimal number (1) is reasonable to make Straight.
        """
        result_msg: str = ""
        k_faceted_dice: KFacetedDice = dst_bundles[0]
        next_straight_number: int = 1
        for k in k_faceted_dice.dices:
            if next_straight_number <= k:
                k_faceted_dice.most_long_straight.append(next_straight_number)
                next_straight_number += 1
        result_msg = " " + str(len(k_faceted_dice.most_long_straight))
        return result_msg


@dataclasses.dataclass
class ChainReactionModule:
    fun_factor: int
    # chainable_module_index_list includes its index
    chainable_module_index_list: list[int]
    index_of_pointing_module: Optional[int] = dataclasses.field(default=None)
    is_initiator: bool = dataclasses.field(default=True)
    is_reacted: bool = dataclasses.field(default=False)


class ChainReaction(GoogleCodeJamProblemSolution[ChainReactionModule]):
    """
    ???? Chain Reactions (10pts, 12pts, 5pts)
    Type:
    Complexity (Time, Space): (, )
    https://codingcompetitions.withgoogle.com/codejam/round/0000000000876ff1/0000000000a45ef7
    """

    def __init__(
        self,
        test_case_number: int = 0,
        OUTPUT_DELIMITER: str = "Case #",
        BUNDLE_NUMBER_IN_ONE_CASE: int = 1,
        PROBLEM_RESOURCE_NAME: str = "chain_reactions",
        RUN_FILE_PATH: str = __file__,
    ) -> None:
        super().__init__(
            test_case_number,
            OUTPUT_DELIMITER,
            BUNDLE_NUMBER_IN_ONE_CASE,
            PROBLEM_RESOURCE_NAME,
            RUN_FILE_PATH,
        )

    def input_case_bundle(
        self, get_contents: Callable[[], str]
    ) -> list[ChainReactionModule]:
        module_list: list[ChainReactionModule] = []
        for _ in range(self.BUNDLE_NUMBER_IN_ONE_CASE):
            one_input_length: int = int(get_contents())
            fun_factor_list: list[int] = list(
                map(int, get_contents().split(" ")[:one_input_length])
            )
            index_of_pointing_module: list[Optional[int]] = list(
                map(
                    lambda x: None if x == 0 else x - 1,
                    map(int, get_contents().split(" ")[:one_input_length]),
                )
            )
            for j in range(one_input_length):
                module_list.append(
                    ChainReactionModule(
                        fun_factor=fun_factor_list[j],
                        index_of_pointing_module=index_of_pointing_module[j],
                        chainable_module_index_list=[j],
                    )
                )
        return module_list

    @staticmethod
    def get_solution(dst_bundles: list[ChainReactionModule]) -> str:
        """
        Given Each module may point at one other module with a lower index.
        Then reacted action must start from bigger index to lower index
        Then Following algorithm is reasonable to get maximum overall fun Wile.
            1 search module that is initiator from index end
            2 compare opportunity cost (max fun factor of remainder) except all module_pointing_itself_list >=2 from initiator
            3 and bang
            2 checking not is_reacted in branch of module_pointing_itself_list

        # directed unidirectional Multiply linked list

        ??????: ???????????? ???????????? ?????? ????????? ????????? ?????? ???????????? ?????? ?????? ?????? ???????????? ??? ?????? fun factor ??? ????????? ?????? ?????? ??????????????? ??????.
        ?????? ?????????, ????????? ?????? ?????? ??? ????????? ?????? ????????????????????? ??????.
        ??????????????? index??? ?????? initiator??? ???????????? ??? initiator ???  ??????..

        (?????? ??????, ?????? ??????)??? ????????????, ?????? ?????? ????????? ?????? ??????????????? ?????? ????????? ???????????? ??????????????????.
        ???????????? docstring ??? ?????? ????????? ???????????????
        ????????? ????????? ??? ???????????? ??? ????????? ????????? ???

        module = chain reaction module
        ????????? ??? ??? ?????? ??? ?????? ?????????
        - ????????? ????????? ??? ??????
            ????????? ?????? ?????? chain reaction ????????? ?????? ??? ??????
        - ????????? ??????
            ????????? ????????? ????????? ????????? chain reaction ????????? ?????? ??? ??????
        - abyss ??? ???????????? ?????? ??????????????? ????????? ????????? ????????? ???????
        = ???????????? ??? chain reaction ????????? ?????? ?????? ?????? ?????? ??????.
            ????????? ??????????????? ????????? ????????? ??? ?????? ????????? ????????? ?????????? ??????.

            30
        10  50  100  ->
                20
        20-> 100, 10->50, 30->30
        20-> 100, 30->50, 10->10
        30-> 100, 10->10, 20->20
        10-> 100, 30->30, 20->20

        initiator ??? ????????? ????????? ?????? ?????? ????????? # itertools.permutations(iterable)
        ??? ???????????? chain reaction ??? ?????????. ???, ?????? ?????? ???????????? ?????? ?????? is_reacted = False ??? ???????????? ????????????.
        ????????? ??? chain reaction ??? ????????? ?????? ????????? ????????? ??????????????? ???????????? ????????? ?????? ???????????? ????????????, ???????????? ???????????? ???????????? ????????? ???????????? ?????? ??? ?????????.
        """
        result_msg: str = ""
        initiator_list: list[ChainReactionModule] = []
        for module in dst_bundles:
            if module.index_of_pointing_module is not None:
                dst_bundles[module.index_of_pointing_module].is_initiator = False
        for module in dst_bundles:
            if module.is_initiator:
                initiator_list.append(module)

        # for optimization
        for initiator in initiator_list:
            current_module_index: Optional[int] = initiator.index_of_pointing_module
            while current_module_index is not None:
                initiator.chainable_module_index_list.append(current_module_index)
                current_module_index = dst_bundles[  # type:ignore
                    current_module_index
                ].index_of_pointing_module

        # group by module pointing abyss
        best_overall_sum_fun_wile: int = 0
        for _, group in itertools.groupby(
            initiator_list, lambda x: x.chainable_module_index_list[-1]
        ):
            initiator_list_groupby: list[ChainReactionModule] = list(group)
            sum_fun_wile_list_groupby_permutation: list[int] = []
            # for initialize is_reacted for each permutation-loop
            reactable_module_index_list: list[list[int]] = []
            for initiator in initiator_list_groupby:
                reactable_module_index_list.append(
                    initiator.chainable_module_index_list
                )
            reactable_module_index_set: set[int] = set(*reactable_module_index_list)

            if len(initiator_list_groupby) == 1:
                best_overall_sum_fun_wile += initiator_list_groupby[0].fun_factor
                continue
            for p in itertools.permutations(initiator_list_groupby):
                sum_fun_wile: int = 0
                # get fun wile
                for initiator in p:
                    fun_factor_list: list[int] = []
                    for i in initiator.chainable_module_index_list:
                        if dst_bundles[i].is_reacted:
                            break
                        else:
                            dst_bundles[i].is_reacted = True
                            fun_factor_list.append(dst_bundles[i].fun_factor)
                    if fun_factor_list:
                        sum_fun_wile += max(fun_factor_list)
                sum_fun_wile_list_groupby_permutation.append(sum_fun_wile)

                # post process
                for i in reactable_module_index_set:
                    dst_bundles[i].is_reacted = False
            best_overall_sum_fun_wile += max(sum_fun_wile_list_groupby_permutation)
        result_msg = str(best_overall_sum_fun_wile)
        return result_msg


# ChainReaction.main(True)
