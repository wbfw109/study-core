"""⚠️ deprecated. require to reconstruct file"""
import datetime
import json
import random
import time
from typing import List, Tuple

from wbfw109.libs.typing import DatetimeRange, TimeEdgePair, TimeRange


def get_elapsed_seconds_from_min_time(a_time: datetime.time) -> float:
    return (
        datetime.datetime.combine(datetime.date.min, a_time) - datetime.datetime.min
    ).total_seconds()


def get_time_pair_in_ascending_order(
    a_time: datetime.time, b_time: datetime.time
) -> Tuple[datetime.time, datetime.time]:
    if get_elapsed_seconds_from_min_time(a_time) > get_elapsed_seconds_from_min_time(
        b_time
    ):
        return (b_time, a_time)
    else:
        return (a_time, b_time)


def get_verified_time_pair_whether_in_ascending_order(
    start_time: datetime.time, end_time: datetime.time
) -> bool:
    """check dateimte.time object is in ascending order.

    Args:
        start_time (datetime.time): [description]
        end_time (datetime.time): [description]

    Returns:
        bool: if in ascending order including same edge value, return True. elif in descending order, return False
    """
    if get_elapsed_seconds_from_min_time(
        start_time
    ) > get_elapsed_seconds_from_min_time(end_time):
        return False
    else:
        return True


def get_time_intersection(
    base_start_time: datetime.time,
    base_end_time: datetime.time,
    time_pair_list_to_be_united: List[TimeEdgePair],
) -> List[TimeRange]:
    """Each parameter edge range are included, not excluded.

    + Assume that (x1, x2), (y1, y2). x is base time, y is hour filter.  e.g. (04:40, 20:20), (05:00, 20:00)
        - if x1.hour >= y1, intersection_start_time is x1. else, is y1.
        - if x2.hour <= y2, intersection_end_time is x2. else, is y2.

    + Args:
        - base_start_time (datetime.time): start_time to be filtered
        - base_end_time (datetime.time): end_time to be filtered
        - time_pair_list_to_be_united (List[Tuple[datetime.time, datetime.time]]): filter by time pair list.
            - Each time pair will be united and calculate intersection with base times.
            - e.g. [(datetime.time(1, 5, 0), datetime.time(4, 10, 5)), (datetime.time(11, 14, 0), datetime.time(15, 18, 30))]
            - later elements not means to include the hour range. so, e.g. hour (5, 14) range means 5:00 ~ 14:00, not 14:59.

    Returns:
        List[Tuple[datetime.time, float]]: float value is total seconds from datetime.time
    """
    # verify
    if not get_verified_time_pair_whether_in_ascending_order(
        base_start_time, base_end_time
    ):
        raise Exception(
            "base_start_time value must be equal or less than base_end_time value."
        )
    for time_pair in time_pair_list_to_be_united:
        if len(time_pair) != 2:
            raise TypeError("Each datetime.time object is must be a pair. (2 argument)")
        if not get_verified_time_pair_whether_in_ascending_order(
            time_pair[0], time_pair[1]
        ):
            raise Exception(
                "First time value must be equal or less than second time value in time pairs."
            )

    # prepare
    time_range_list: List[TimeRange] = []
    base_datetime: datetime.datetime = datetime.datetime.min

    # process
    for time_pair in time_pair_list_to_be_united:
        base_start_elapsed_seconds: float = get_elapsed_seconds_from_min_time(
            base_start_time
        )
        base_end_elapsed_seconds: float = get_elapsed_seconds_from_min_time(
            base_end_time
        )
        filter_start_elapsed_seconds: float = get_elapsed_seconds_from_min_time(
            time_pair[0]
        )
        filter_end_elapsed_seconds: float = get_elapsed_seconds_from_min_time(
            time_pair[1]
        )

        if base_start_elapsed_seconds >= filter_start_elapsed_seconds:
            intersection_start_elapsed_seconds: float = base_start_elapsed_seconds
        else:
            intersection_start_elapsed_seconds: float = filter_start_elapsed_seconds
        if base_end_elapsed_seconds <= filter_end_elapsed_seconds:
            intersection_end_elapsed_seconds: float = base_end_elapsed_seconds
        else:
            intersection_end_elapsed_seconds: float = filter_end_elapsed_seconds

        available_range_seconds: float = (
            intersection_end_elapsed_seconds - intersection_start_elapsed_seconds
        )
        if available_range_seconds >= 0:
            time_range_list.append(
                (
                    (
                        datetime.datetime.min
                        + datetime.timedelta(seconds=intersection_start_elapsed_seconds)
                    ).time(),
                    available_range_seconds,
                )
            )

    return time_range_list


def get_datetime_filterd_by_time_range(
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
    time_pair_list_to_be_united: List[TimeEdgePair],
) -> List[DatetimeRange]:
    """Each parameter edge range are included, not excluded.

    Args:
        start_datetime (datetime.datetime): [description]
        end_datetime (datetime.datetime): [description]
        time_pair_list_to_be_united (List[Tuple[datetime.time, datetime.time]]): filter by time pair list.

    Returns:
        List[Tuple[datetime.datetime, float]: [description]
    """
    ymd_start_datetime: datetime.datetime = datetime.datetime.combine(
        date=start_datetime.date(), time=datetime.time.min
    )
    ymd_end_datetime: datetime.datetime = datetime.datetime.combine(
        date=end_datetime.date(), time=datetime.time.min
    )

    datetime_list: List[DatetimeRange] = []

    # process
    if ymd_start_datetime == ymd_end_datetime:
        if time_range_list := get_time_intersection(
            base_start_time=start_datetime.time(),
            base_end_time=end_datetime.time(),
            time_pair_list_to_be_united=time_pair_list_to_be_united,
        ):
            for time_range in time_range_list:
                datetime_list.append(
                    (
                        datetime.datetime.combine(start_datetime.date(), time_range[0]),
                        time_range[1],
                    )
                )
    else:
        for day in [
            ymd_start_datetime + datetime.timedelta(days=i)
            for i in range((ymd_end_datetime - ymd_start_datetime).days + 1)
        ]:
            if day == ymd_start_datetime:
                if time_range_list := get_time_intersection(
                    base_start_time=start_datetime.time(),
                    base_end_time=datetime.time.max,
                    time_pair_list_to_be_united=time_pair_list_to_be_united,
                ):
                    for time_range in time_range_list:
                        datetime_list.append(
                            (
                                datetime.datetime.combine(
                                    start_datetime.date(), time_range[0]
                                ),
                                time_range[1],
                            )
                        )
            elif day == ymd_end_datetime:
                if time_range_list := get_time_intersection(
                    base_start_time=datetime.time.min,
                    base_end_time=end_datetime.time(),
                    time_pair_list_to_be_united=time_pair_list_to_be_united,
                ):
                    for time_range in time_range_list:
                        datetime_list.append(
                            (
                                datetime.datetime.combine(
                                    end_datetime.date(), time_range[0]
                                ),
                                time_range[1],
                            )
                        )
            else:
                if time_range_list := get_time_intersection(
                    base_start_time=datetime.time.min,
                    base_end_time=datetime.time.max,
                    time_pair_list_to_be_united=time_pair_list_to_be_united,
                ):
                    for time_range in time_range_list:
                        datetime_list.append(
                            (
                                datetime.datetime.combine(day.date(), time_range[0]),
                                time_range[1],
                            )
                        )

    return datetime_list


def get_not_holidays_in_korea(
    start_datetime: datetime.datetime, end_datetime: datetime.datetime
) -> list:
    from wbfw109.config import CONFIG_CLASS

    # prepare
    one_day = datetime.timedelta(days=1)
    with open(
        str(CONFIG_CLASS.HOLIDAYS_IN_SOUTH_KOREA_PATH), "r", encoding="utf-8"
    ) as file:
        holidays_in_south_korea: list = json.load(file)

    not_holidays = []
    required_holidays_in_south_korea: dict = {}
    for holidays in holidays_in_south_korea:
        if (
            holidays["year"] >= start_datetime.year
            and holidays["year"] <= end_datetime.year
        ):
            # if year is valid
            holidays_as_datetime: list = []
            for holiday in holidays["holidays"]:
                holidays_as_datetime.append(
                    datetime.datetime.strptime(
                        f"{holiday['day']}/{holiday['month']}/{holidays['year']}",
                        "%d/%m/%Y",
                    )
                )
            required_holidays_in_south_korea[holidays["year"]] = holidays_as_datetime

    next_date = start_datetime

    # process
    while next_date <= end_datetime:
        if next_date.isoweekday() <= 5:
            is_holiday = False
            for holiday in required_holidays_in_south_korea[next_date.year]:
                if next_date == holiday:
                    is_holiday = True
                    break

            if not is_holiday:
                not_holidays.append(next_date)

        # postprocess
        next_date += one_day

    return not_holidays


def get_random_between_datetimes(
    start_datetime: datetime.datetime, end_datetime: datetime.datetime
) -> datetime:
    time_delta: datetime.timedelta = end_datetime - start_datetime
    return start_datetime + datetime.timedelta(
        seconds=random.randint(0, time_delta.seconds)
    )


def get_converted_unixtime_from_datatime(timestamp: datetime.datetime) -> float:
    return time.mktime(timestamp.timetuple())


def get_converted_datetime_from_unixtime(timestamp: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(timestamp)
