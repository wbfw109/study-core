# %%
#
"""

정규식 전환.. 문제풀이 (not python 3.9)
    : .* =
        to
    =

"""
import logging
import os

import IPython
from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

import collections
import dataclasses
import datetime
import itertools
import math
import random
import re
import shutil
import string
import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Tuple, Type, Union

import numpy
import pandas
from PIL import Image, ImageDraw, ImageFilter


# %%
def get_champion_the_number_of_hit_list_by_target_in_disadvantageous_condition(
    the_number_of_arrow_left: int, challenger_the_number_of_hit_list: list[int]
):
    """
    if champion's the_number_of_hit each target equal or greater than challenger's the_number_of_hit,
        challenger take a score.

    it returns the case with as many as possible number of lowest points hit in order to win champion to win by the maximum difference between the challenger and the challenger.

    if champion can not win (if champion will draw or lose), return -1.

    winner will get score for each score target.
        e.g.
            get_champion_the_number_of_hit_list_by_target_in_disadvantageous_condition(
                the_number_of_arrow_left=6,
                challenger_the_number_of_hit_list=[3, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0],
            )
            # if result of champion_the_number_of_hit_list==[0, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            assert sum_of_challenger_hit_score == 11 and sum_of_champion_hit_score == 35

    e.g.
        get_champion_the_number_of_hit_list_by_target_in_disadvantageous_condition(
            the_number_of_arrow_left=5,
            challenger_the_number_of_hit_list=[2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        )
        get_champion_the_number_of_hit_list_by_target_in_disadvantageous_condition(
            the_number_of_arrow_left=6,
            challenger_the_number_of_hit_list=[3, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0],
        )

        get_champion_the_number_of_hit_list_by_target_in_disadvantageous_condition(
            the_number_of_arrow_left=1,
            challenger_the_number_of_hit_list=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        get_champion_the_number_of_hit_list_by_target_in_disadvantageous_condition(
            the_number_of_arrow_left=9,
            challenger_the_number_of_hit_list=[0, 0, 1, 2, 0, 1, 1, 1, 1, 1, 1],
        )

        get_champion_the_number_of_hit_list_by_target_in_disadvantageous_condition(
            the_number_of_arrow_left=10,
            challenger_the_number_of_hit_list=[0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 3],
        )
    """
    MIN_SCORE: int = 0
    MAX_SCORE: int = 10
    index_score_dict: dict[int, int] = {
        i: 10 - i for i in range(MIN_SCORE, MAX_SCORE + 1)
    }

    def get_sum_score(
        challenger_the_number_of_hit_list: list[int],
        champion_the_number_of_hit_list: list[int],
    ) -> tuple[int]:
        assert (
            len(challenger_the_number_of_hit_list)
            == len(champion_the_number_of_hit_list)
            == (MAX_SCORE - MIN_SCORE) + 1
        )

        sum_of_challenger_hit_score: int = 0
        sum_of_champion_hit_score: int = 0
        for index, hit_list in enumerate(
            zip(challenger_the_number_of_hit_list, champion_the_number_of_hit_list),
            start=0,
        ):
            # preprocess
            if hit_list[0] == hit_list[1] == 0:
                continue

            if hit_list[0] >= hit_list[1]:
                sum_of_challenger_hit_score += index_score_dict[index]
            else:
                sum_of_champion_hit_score += index_score_dict[index]

        return sum_of_challenger_hit_score, sum_of_champion_hit_score

    def get_weighted_index_list(
        challenger_the_number_of_hit_list: list[int],
    ) -> list[float]:
        assert len(challenger_the_number_of_hit_list) == (MAX_SCORE - MIN_SCORE) + 1
        index_weight_dict: dict[int, float] = {}
        champion_hit_to_get_score = [
            the_number_of_hit + 1
            for the_number_of_hit in challenger_the_number_of_hit_list
        ]
        for index, the_number_of_hit in enumerate(challenger_the_number_of_hit_list):
            # score can be taken away. so dividend is different for each case.
            if the_number_of_hit > 0:
                index_weight_dict[index] = (
                    index_score_dict[index] * 2 / champion_hit_to_get_score[index]
                )
            else:
                index_weight_dict[index] = index_score_dict[index] / (
                    challenger_the_number_of_hit_list[index] + 1
                )

        # sort by two key for constraints
        return [
            index
            for index, weight in sorted(
                index_weight_dict.items(),
                key=lambda kv: (kv[1], kv[0]),
                reverse=True,
            )
        ]

    champion_the_number_of_hit_list_with_max_score_target: list[int] = [
        0 for _ in range(len(challenger_the_number_of_hit_list))
    ]
    weighted_index_list: list[int] = get_weighted_index_list(
        challenger_the_number_of_hit_list=challenger_the_number_of_hit_list
    )

    remainder_of_champion_hit: int = the_number_of_arrow_left

    for weighted_index in weighted_index_list:
        hit_cost: int = challenger_the_number_of_hit_list[weighted_index] + 1
        if hit_cost > remainder_of_champion_hit:
            continue
        elif remainder_of_champion_hit < 1:
            break

        champion_the_number_of_hit_list_with_max_score_target[weighted_index] = hit_cost
        remainder_of_champion_hit -= hit_cost

    # postprocess for constraints
    if remainder_of_champion_hit > 0:
        champion_the_number_of_hit_list_with_max_score_target[
            len(champion_the_number_of_hit_list_with_max_score_target) - 1
        ] += remainder_of_champion_hit
        remainder_of_champion_hit = 0

    sum_of_challenger_hit_score, sum_of_champion_hit_score = get_sum_score(
        challenger_the_number_of_hit_list=challenger_the_number_of_hit_list,
        champion_the_number_of_hit_list=champion_the_number_of_hit_list_with_max_score_target,
    )
    # if champion can not win challenger
    if sum_of_challenger_hit_score >= sum_of_champion_hit_score:
        return -1
    else:
        return champion_the_number_of_hit_list_with_max_score_target


base_notation: str = string.digits + string.ascii_uppercase


class InOut(Enum):
    IN = 0
    OUT = 1


@dataclasses.dataclass
class TimeInOut:
    time: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime(datetime.MINYEAR, 1, 1, 0, 0, 0, 0)
    )
    log: InOut = dataclasses.field(default_factory=lambda: InOut["IN"])


def get_fee_list(fees: list[int], records: list[str]):
    """

    e.g.
        get_fee_list(
            [180, 5000, 10, 600],
            [
                "05:34 5961 IN",
                "06:00 0000 IN",
                "06:34 0000 OUT",
                "07:59 5961 OUT",
                "07:59 0148 IN",
                "18:59 0000 IN",
                "19:09 0148 OUT",
                "22:59 5961 IN",
                "23:00 5961 OUT",
            ],
        )

        get_fee_list(
            [120, 0, 60, 591],
            [
                "16:00 3961 IN",
                "16:00 0202 IN",
                "18:00 3961 OUT",
                "18:00 0202 OUT",
                "23:58 3961 IN",
            ],
        )
        get_fee_list([1, 461, 1, 10], ["00:00 1234 IN"])

    Args:
        fees (list[int]): [description]
        records (list[str]): [description]

    Returns:
        [type]: [description]
    """
    basic_time_minute, basic_fee, unit_time_minute, unit_fee = fees
    basic_time_second: int = basic_time_minute * 60
    unit_time_second: int = unit_time_minute * 60
    unique_car_number = set()
    car_record_dict: dict[str, list[TimeInOut]] = {}
    result_fee_dict: dict[str, int] = {}

    for one_record in records:
        unique_car_number.add(one_record.split(" ")[1])
    for unique_car in unique_car_number:
        car_record_dict[unique_car] = []
        result_fee_dict[unique_car] = 0

    for one_record in records:
        that_time, car_number_str, in_out_log = one_record.split(" ")
        car_record_dict[car_number_str].append(
            TimeInOut(datetime.datetime.strptime(that_time, "%H:%M"), InOut[in_out_log])
        )

    for car_number, car_record_list in car_record_dict.items():
        # ensure length have 2*x
        if car_record_list[-1:][0].log == InOut.IN:
            car_record_list.append(
                TimeInOut(datetime.datetime.strptime("23:59", "%H:%M"), InOut["OUT"])
            )
        sum_of_parking_duration = 0
        for i in range(0, len(car_record_list), 2):
            sum_of_parking_duration += (
                car_record_list[i + 1].time - car_record_list[i].time
            ).seconds

        if sum_of_parking_duration <= basic_time_second:
            result_fee_dict[car_number] = basic_fee
        else:
            result_fee_dict[car_number] = (
                basic_fee
                + math.ceil(
                    (sum_of_parking_duration - basic_time_second) / unit_time_second
                )
                * unit_fee
            )

    return [x[1] for x in sorted(result_fee_dict.items(), key=lambda kv: kv[0])]


def get_reported_mail_count(id_list: list[str], report: list[str], k: int) -> list[int]:
    """
    If the person being reported has been suspended, the person who reported at least once will be notified of how many people have been suspended.

    Duplicate reports will be processed once.

    e.g.
        get_reported_mail_count(
            ["muzi", "frodo", "apeach", "neo"],
            ["muzi frodo", "apeach frodo", "frodo neo", "muzi neo", "apeach muzi"],
            2,
        )
        get_reported_mail_count(
            ["con", "ryan"], ["ryan con", "ryan con", "ryan con", "ryan con"], 3
        )

    Args:
        id_list (list[str]): [description]
        report (list[str]): [description]
        k (int): base of suspending

    Returns:
        list[int]: Returns the values in order according to id_list.
    """
    report_counter: collections.Counter = collections.Counter()
    reported_counter: collections.Counter = collections.Counter()

    id_reported_dict: dict[str, list] = {}
    all_id_set: set = set()
    # get all unique id list
    for one_id in id_list:
        all_id_set.add(one_id)
    for one_report in report:
        reporter, reported = one_report.split(" ")
        all_id_set.add(reporter)

    # initialize
    for unique_id in all_id_set:
        id_reported_dict[unique_id] = []

    # ensure unique
    for one_report in report:
        reporter, reported = one_report.split(" ")
        if report_counter[(reporter, reported)] != 1:
            report_counter[(reporter, reported)] += 1
            id_reported_dict[reporter].append(reported)

    # sum reported count
    for reporter, reported in report_counter:
        reported_counter[reported] += 1

    answer_id_reported_dict: dict[str, int] = {}
    for one_id in id_list:
        stop_count: int = 0
        for each_reported in id_reported_dict[one_id]:
            if reported_counter[each_reported] >= k:
                stop_count += 1

        answer_id_reported_dict[one_id] = stop_count

    return list(answer_id_reported_dict.values())


def convert_base10_to_n_base(number: int, base: int) -> str:
    """
    e.g.
        number_example: int = 25
        base_example: int = 2
        x = convert_base10_to_n_base(number=number_example, base=base_example)
        assert int(x, base_example) == number_example

    Args:
        number (int): [description]
        base (int): [description]

    Returns:
        str: [description]
    """
    assert base >= 2 and base <= len(base_notation)
    result_list: list[str] = []

    while True:
        quotient, remainder = divmod(number, base)
        result_list.append(base_notation[remainder])
        number = quotient

        if quotient == 0:
            break

    return "".join(reversed(result_list))


def get_primary_key_list(number_list: list[int]) -> list[int]:
    result_list = []
    for number in number_list:

        # preprocess
        if number < 2:
            continue
        elif number < 4:
            result_list.append(number)
            continue

        # process
        is_primary_nubmer: bool = True

        for i in range(2, math.floor(math.sqrt(number)) + 1):
            quotient, remainder = divmod(number, i)
            if remainder == 0:
                is_primary_nubmer = False
                break

        if is_primary_nubmer:
            result_list.append(number)
    return result_list


def get_primary_key_list_count(number, base) -> list[int]:
    """

    e.g.
        get_primary_key_list_count(200, 3)

        get_primary_key_list_count(437674, 3)
        get_primary_key_list_count(110011, 10)

    Args:
        number ([type]): [description]
        base ([type]): [description]

    Returns:
        list[int]: [description]
    """
    return len(
        get_primary_key_list(
            number_list=map(
                int, filter(None, convert_base10_to_n_base(number, base).split("0"))
            )
        )
    )
