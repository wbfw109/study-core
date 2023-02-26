"""Command for company. This file must be not imported by other files for nammespace

Since the DBMS is different, just put the value as it is without dropping or alter the table. 
    > https://docs.sqlalchemy.org/en/14/core/type_basics.html#generic-types

CiSessions
    > source
        SELECT sf_employee.employee_id, sf_employee.employee_number, sf_employee.name, sf_employee.work_id, sf_employee_work.work_level
        FROM sf_employee
        INNER JOIN sf_employee_work
        ON sf_employee.work_id = sf_employee_work.work_id;

    > check result data
        SELECT to_timestamp(timestamp)::date AS date, ip_address, COUNT(*)
        FROM ci_sessions
        WHERE ip_address = '192.168.0.2'
        GROUP BY date, ip_address
        ORDER BY date

        SELECT ip_address, COUNT(*)
        FROM ci_sessions
        GROUP BY ip_address
        ORDER BY ip_address;

FacilitiyNotification
    > python backend/service/command/creating_log.py -f facility_notification

    > check result data
        SELECT facilities, COUNT(*)
        FROM "facility_notification"
        GROUP BY facilities

        SELECT *
        FROM "facility_notification"
        WHERE facilities = 1 AND notification_type = 2
        ORDER BY noti_datetime
        
        SELECT facilities, extract(hour from noti_datetime) as hours, COUNT(*)
        FROM "facility_notification"
        WHERE notification_type = 2
        GROUP BY facilities, hours
        ORDER BY facilities

    > test data
        DELETE FROM "facility_notification"
        WHERE True;

        ALTER SEQUENCE facility_notification_id_seq RESTART WITH 1;

"""
import argparse
import datetime
import itertools

# from datetime.datetime import datetime.datetime, time
import random
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generator, Tuple

from mysite.config import CONFIG_CLASS
from mysite.database import Base, db_session
from sqlalchemy import BigInteger, Column, DateTime, Integer, SmallInteger, String, Text
from wbfw109.libs.typing import DatetimeRange
from wbfw109.libs.utilities.datetime import (
    get_converted_unixtime_from_datatime,
    get_datetime_filterd_by_time_range,
    get_not_holidays_in_korea,
    get_random_between_datetimes,
)
from wbfw109.libs.utilities.iterable import get_sliced_list, get_unique_id_list

id_length_for_employees: int = 40


class employees:
    def __init__(self, raw_data) -> None:
        class employee:
            def __init__(self, raw_datum) -> None:
                self.id: int = raw_datum[0]
                self.employee_number: int = raw_datum[1]
                self.name: str = raw_datum[2]
                self.position: str = raw_datum[3]
                self.work_id: int = raw_datum[4]
                self.work_level: int = raw_datum[5]
                self.ip_address: str = raw_datum[6]

        self.data: list = []
        for element in [employee(raw_datum) for raw_datum in raw_data]:
            self.data.append({"position": element.position, "information": element})


@dataclasses.dataclass
class CiSessions(Base):
    __tablename__ = "ci_sessions"
    id: str = Column(
        String(id_length_for_employees), primary_key=True, autoincrement=False
    )
    ip_address: str = Column(String(16), unique=False, nullable=False)
    timestamp: int = Column(Integer, unique=False, nullable=False)
    data: str = Column(Text, unique=False, nullable=False)

    def __init__(self, id: str, ip_address: str, timestamp: int, data: str):
        self.id = id
        self.ip_address = ip_address
        self.timestamp = timestamp
        self.data = data

    def __repr__(self):
        return f"<Data {self.data!r}>"

    @staticmethod
    def create_data(
        start_datetime: datetime.datetime, end_datetime: datetime.datetime
    ) -> None:
        """create data from raw_data

        To do:
            read from csv file. not raw data
        """
        # * prepare Start
        manager_group: list = ["총괄", "소장"]
        worker_group: list = ["실장", "팀장", "기사"]
        my_data: list = [
            [1, "one0000", "김현우", "총괄", 1, 10, "192.168.0.4"],
            [3, "one0002", "나경석", "실장", 4, 1, "192.168.0.2"],
            [2, "one0001", "민재홍", "소장", 4, 1, "192.168.0.6"],
            [4, "one0003", "정나영", "팀장", 4, 1, "192.168.0.3"],
            [5, "one0004", "박설화", "팀장", 4, 1, "192.168.0.5"],
            [6, "one0005", "고가형", "기사", 4, 1, "192.168.0.7"],
        ]
        my_employees = employees(my_data)
        not_holidays_in_korea: list[datetime.datetime] = get_not_holidays_in_korea(
            start_datetime=start_datetime, end_datetime=end_datetime
        )
        timestamp_slicer: list = []

        # add dynamic attribute to employee instance: working_datetimes
        for employee_data in my_employees.data:
            connection_times: set = set()

            # add time as new working day
            if employee_data["position"] in manager_group:
                # exclude private holidays
                actual_working_day_manager_group: list = sorted(
                    set(not_holidays_in_korea)
                    - set(
                        random.choices(not_holidays_in_korea, k=random.randint(15, 25))
                    )
                )

                for working_day in actual_working_day_manager_group:
                    # not add millisecond beacuase latter it will be converted to unix time
                    connection_times.add(
                        get_random_between_datetimes(
                            datetime.datetime.combine(
                                working_day.date(), datetime.time(9, 0, 0)
                            ),
                            datetime.datetime.combine(
                                working_day.date(), datetime.time(9, 30, 0)
                            ),
                        )
                    )
                    connection_times.add(
                        get_random_between_datetimes(
                            datetime.datetime.combine(
                                working_day.date(), datetime.time(17, 0, 0)
                            ),
                            datetime.datetime.combine(
                                working_day.date(), datetime.time(17, 30, 0)
                            ),
                        )
                    )

            elif employee_data["position"] in worker_group:
                actual_working_day_worker_group: list = sorted(
                    set(not_holidays_in_korea)
                    - set(
                        random.choices(not_holidays_in_korea, k=random.randint(12, 21))
                    )
                )

                for working_day in actual_working_day_worker_group:
                    connection_times.add(
                        get_random_between_datetimes(
                            datetime.datetime.combine(
                                working_day.date(), datetime.time(9, 0, 0)
                            ),
                            datetime.datetime.combine(
                                working_day.date(), datetime.time(9, 30, 0)
                            ),
                        )
                    )

                    for _ in range(1):
                        if 1 == random.randint(0, 1):
                            connection_times.add(
                                get_random_between_datetimes(
                                    datetime.datetime.combine(
                                        working_day.date(), datetime.time(10, 0, 0)
                                    ),
                                    datetime.datetime.combine(
                                        working_day.date(), datetime.time(11, 30, 0)
                                    ),
                                )
                            )
                    for _ in range(4):
                        if 1 == random.randint(0, 1):
                            connection_times.add(
                                get_random_between_datetimes(
                                    datetime.datetime.combine(
                                        working_day.date(), datetime.time(13, 0, 0)
                                    ),
                                    datetime.datetime.combine(
                                        working_day.date(), datetime.time(17, 40, 0)
                                    ),
                                )
                            )
            employee_data["information"].working_datetimes = sorted(connection_times)
            timestamp_slicer.append(len(connection_times))

        ids: list = get_sliced_list(
            list_to_be_sliced=get_unique_id_list(
                existing_ids=set(
                    [result[0] for result in db_session.query(CiSessions.id).all()]
                ),
                additional_count=sum(timestamp_slicer),
                get_function=lambda: secrets.token_hex(
                    int(id_length_for_employees / 2)
                ),
            ),
            slicer=timestamp_slicer,
        )

        # apply form according to requirements to add data
        ci_sessions_to_be_added: list = []

        for i, employee_data in enumerate(my_employees.data):
            for j, _ in enumerate(
                range(len(employee_data["information"].working_datetimes))
            ):
                this_unixtime: int = int(
                    get_converted_unixtime_from_datatime(
                        employee_data["information"].working_datetimes[j]
                    )
                )
                ci_sessions_to_be_added.append(
                    CiSessions(
                        id=ids[i][j],
                        ip_address=employee_data["information"].ip_address,
                        timestamp=this_unixtime,
                        data=";".join(
                            [
                                f"__ci_last_regenerate|i:{this_unixtime}",
                                f'employee_id|s:1:"{employee_data["information"].id}"',
                                f'employee_number|s:7:"{employee_data["information"].employee_number}"',
                                f'name|s:9:"{employee_data["information"].name}"',
                                f'work_id|s:1:"{employee_data["information"].work_id}"',
                                f'work_level|s:2:"{employee_data["information"].work_level}"',
                            ]
                        ),
                    )
                )
        # * prepare End

        # * test data
        # DELETE FROM ci_sessions

        # user_file = CONFIG_CLASS.RESOURCE_FOLDER / 'test.txt'
        # if not Path(user_file).exists():
        #     with open(
        #         user_file, "w", encoding="utf-8"
        #     ) as txt_file:
        #         pass

        # with open(
        #     str(user_file), "w+", encoding="utf-8"
        # ) as txt_file:
        #     for x in ci_sessions_to_be_added:
        #         txt_file.write(
        #             f'{x.data}\n'
        #             # f'{x.id}, {x.ip_address}, {x.timestamp}\n'
        #         )

        # * process Start
        # db_session.add_all(ci_sessions_to_be_added)
        # db_session.commit()
        # * process End


stop_seconds_min_offset = datetime.timedelta(seconds=43200)
stop_seconds_max_offset = datetime.timedelta(seconds=108000)
resume_seconds_min_offset = datetime.timedelta(seconds=21600)
resume_seconds_max_offset = datetime.timedelta(seconds=43200)


class NotificationType(Enum):
    STOP = 1
    START = 2


@dataclasses.dataclass
class FacilitiyNotification(Base):
    __tablename__ = "facility_notification"
    id: int = Column(BigInteger, primary_key=True, autoincrement=True)
    create_datetime: datetime.datetime = Column(DateTime, unique=False, nullable=False)
    img_path: str = Column(String(255), unique=False, nullable=True)
    noti_datetime: datetime.datetime = Column(DateTime, unique=False, nullable=False)
    notification_type: int = Column(
        SmallInteger, unique=False, nullable=False, default=0
    )
    facilities: int = Column(BigInteger, unique=False, nullable=True)
    workplace: str = Column(String(255), unique=False, nullable=True)

    def __init__(
        self,
        id: int,
        create_datetime: datetime.datetime,
        img_path: str,
        noti_datetime: datetime.datetime,
        notification_type: int,
        facilities: int,
        workplace: str,
    ):
        self.id: int = id
        self.create_datetime: datetime.datetime = create_datetime
        self.img_path: str = img_path
        self.noti_datetime: datetime.datetime = noti_datetime
        self.notification_type: int = notification_type
        self.facilities: int = facilities
        self.workplace: str = workplace

    @staticmethod
    def _get_random_toggled_datetime(
        facility_state_to_be_setten: NotificationType,
        previous_datetime: datetime.datetime,
    ) -> datetime.datetime:
        """Offset is not taken into account in this function.

        Args:
            facility_state_to_be_setten (NotificationType): facility_state_to_be_setten
            previous_datetime (datetime.datetime): based on datetime

        Returns:
            datetime.datetime: toggled_datetime

        Todo:
            multiprocessing?
        """
        if facility_state_to_be_setten == NotificationType.STOP:
            # randomly returns a time after between 12 and 30 hours from the previous time.
            # auto exit system
            return previous_datetime + datetime.timedelta(
                seconds=random.randint(
                    int(stop_seconds_min_offset.total_seconds()),
                    int(stop_seconds_max_offset.total_seconds()),
                )
            )
        elif facility_state_to_be_setten == NotificationType.START:
            # randomly returns a time after between 6 and 12 hours from the previous time
            # it require person (09:00 ~ 18:00)
            min_next_datetime: datetime.datetime = (
                previous_datetime
                + datetime.timedelta(
                    seconds=int(resume_seconds_min_offset.total_seconds())
                )
            )
            max_next_datetime: datetime.datetime = (
                previous_datetime
                + datetime.timedelta(
                    seconds=int(resume_seconds_max_offset.total_seconds())
                )
            )

            is_delayed: bool = False
            while True:
                if available_ranges := get_datetime_filterd_by_time_range(
                    min_next_datetime,
                    max_next_datetime,
                    [(datetime.time(9, 0, 0), datetime.time(18, 0, 0))],
                ):
                    if is_delayed:
                        return available_ranges[0][0] + datetime.timedelta(
                            seconds=random.randint(0, int(available_ranges[0][1] / 10))
                        )
                    else:
                        choiced_range: DatetimeRange = random.choice(available_ranges)
                        return choiced_range[0] + datetime.timedelta(
                            seconds=random.randint(0, int(choiced_range[1]))
                        )
                else:
                    max_next_datetime += datetime.timedelta(days=1)
                    is_delayed = True

    @staticmethod
    def create_data(
        start_datetime: datetime.datetime, end_datetime: datetime.datetime
    ) -> None:
        """create data from raw_data

        Args:
            start_datetime (datetime.datetime): included.
            end_datetime (datetime.datetime): excluded.

        To do:
            multiprocessing?
            colmun notification_type convert to enum, not int
        """
        # * prepare Start
        facility_list: list = list(range(9, 25))
        facilitiy_notification_list: list = []
        workplace_code = "2HTZYX"

        for facility in facility_list:
            # ** set first facility state
            first_facility_state_to_be_setten: NotificationType = NotificationType(
                random.randint(1, 2)
            )
            first_datetime: datetime.datetime = (
                FacilitiyNotification._get_random_toggled_datetime(
                    facility_state_to_be_setten=first_facility_state_to_be_setten,
                    previous_datetime=start_datetime,
                )
            )

            if first_facility_state_to_be_setten == NotificationType.STOP:
                # apply random offset at only first when NotificationType.STOP
                first_datetime -= datetime.timedelta(
                    seconds=random.randint(
                        0, int(stop_seconds_min_offset.total_seconds())
                    )
                )
                next_cycle_generator: Generator = reversed(NotificationType)
                facilitiy_notification_list.append(
                    FacilitiyNotification(
                        id=None,
                        create_datetime=first_datetime,
                        img_path=None,
                        noti_datetime=first_datetime,
                        notification_type=first_facility_state_to_be_setten.value,
                        facilities=facility,
                        workplace=workplace_code,
                    )
                )
            elif first_facility_state_to_be_setten == NotificationType.START:
                next_cycle_generator: Generator = NotificationType
            else:
                raise Exception("invalid value of first_facility_state")

            # apply form according to requirements to add data

            # ** add until end_datetime
            bufferd_datetime: datetime.datetime = first_datetime
            for facility_state in itertools.cycle(next_cycle_generator):
                bufferd_datetime = FacilitiyNotification._get_random_toggled_datetime(
                    facility_state_to_be_setten=facility_state,
                    previous_datetime=bufferd_datetime,
                )

                if bufferd_datetime > end_datetime:
                    break

                facilitiy_notification_list.append(
                    FacilitiyNotification(
                        id=None,
                        create_datetime=bufferd_datetime,
                        img_path=None,
                        noti_datetime=bufferd_datetime,
                        notification_type=facility_state.value,
                        facilities=facility,
                        workplace=workplace_code,
                    )
                )
        # * prepare End

        # * test data
        for xx in [
            x
            for x in sorted(facilitiy_notification_list, key=lambda x: x.noti_datetime)
        ]:
            if (
                xx.facilities == 1
                and NotificationType(xx.notification_type) == NotificationType.START
            ):
                print(
                    f"{NotificationType(xx.notification_type)}, {xx.noti_datetime}, {xx.facilities}"
                )

        # * process Start
        db_session.add_all(
            sorted(facilitiy_notification_list, key=lambda x: x.noti_datetime)
        )
        db_session.commit()
        # * process End


if __name__ == "__main__":
    # * parser Start
    parser = argparse.ArgumentParser(
        description="create random logs",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v", "--verbosity", help="increase output verbosity", action="count", default=0
    )
    group.add_argument("-q", "--quiet", action="store_true")

    required_group = parser.add_argument_group("required named arguments")
    required_group.add_argument(
        "-f",
        "--function",
        help="function. image processsing mode.",
        required=True,
        nargs="+",
        choices=["ci_sessions", "facility_notification"],
    )

    args = parser.parse_args()

    start = time.time()
    for arg in set(args.function):
        if arg == "ci_sessions":
            CiSessions.create_data(
                start_datetime=datetime.datetime(2020, 1, 1),
                end_datetime=datetime.datetime(2021, 6, 27),
            )
        elif arg == "facility_notification":
            FacilitiyNotification.create_data(
                start_datetime=datetime.datetime(2021, 4, 19),
                end_datetime=datetime.datetime(2021, 6, 28, 16, 0, 0),
            )

    end = time.time()
    print(f"time taken: {end-start}")
