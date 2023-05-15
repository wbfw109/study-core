# This Cell is for common declaration, and Next Cells are for specified algorithms.
# comply my structure for writting algorithms like Next Cell:

# Note
# 1. if you are using Type hint, but python version is less than 3.9, Generic type hint is required instead of type object.
#   it applies some web site: https://programmers.co.kr/
"""
math.isclose, Decimal import statistics
built-in @ Matrix Multiplication 은 numpy 에서만 사용가능한가
sorted multiple key: https://hello-bryan.tistory.com/43

import quopri
import textwrap.shorten(), textwrap.wrap(), textwrap.fill(long_text, width=70)
import struct
import heapq, heapq.heapify(data), heapq.nsmallest(3, data))
import bisect 점수에 따른 학점 구하기
import curses 터미널 프로그램
import lzma, bzip2, gzip, zip, tar, zlib
pickle 객체를 파일로 저장, import copyreg 로 객체 변경에 따른 오류 방지?, 딕셔너리를 파일로 저장하려면? ― shelve??
https://docs.python.org/3/library/csv.html
import linecache
import tempfile with test function? import dircmp, fileinput (여러개의 파일 한 번에 읽기)
import configparser
암호화.. hashlib, hmac (메시지 변조 확인), secrets
operator.itemgetter
conversion:    import quopri, binascii

Threading.. import concurrent.futures, async await
import sched
문장분석 shlex
import xmlrpc? import htt.server import html.escape
socketserver + nonblocking>?
crawling? nntplib
email poplib, IMAP4


datetime
datetime.fromtimestamp(float(split_line[0])).strftime("%Y-%m-%d %H:%M:%S")

# MAth: Permutation, Cartesian product, Discrete Fourier transform, Multiplication algorithm

달력에 특정한 달에 해당하는 일수 구하기
ACM-ICPC, Codeforce, 정보 올림피아드

"""
# %%
# 📝 Last checking before coding test

# instead "with open" context, Use "with Path.open" context.
# if multiple version problems, use (Enum | IntEnum), for statement, Structural Pattern Matching


def itertools_test():
    import itertools

    pprint.pprint(
        [(key, list(group)) for key, group in itertools.groupby("AAAABBBCCD")]
    )


def datetime_test():
    import calendar

    pprint.pprint([calendar.isleap(year) for year in range(2020, 2025)])


def conversion_test():
    import base64

    encoded = base64.b64encode(b"data to be encoded")
    decoded_data = base64.b64decode(encoded)


def string_test():
    import string

    pprint.pprint([string.ascii_letters, string.digits])


@dataclasses.dataclass
class DataClassTimeLog:
    time: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime(datetime.MINYEAR, 1, 1, 0, 0, 0, 0)
    )
    log: list[Any] = dataclasses.field(default_factory=list)


import graphlib

graph = {"D": {"B", "C"}, "C": {"A"}, "B": {"A"}}
ts = graphlib.TopologicalSorter(graph)
tuple(ts.static_order())
# input_test

ts = graphlib.TopologicalSorter()
ts.add(3, 2, 1)
ts.add(1, 0)
print([*ts.static_order()])


ts2 = graphlib.TopologicalSorter()
ts2.add(1, 0)
ts2.add(3, 2, 1)
print([*ts2.static_order()])

# 제출 전 가장 위에 import 모두 제거하기

# %%
# graph # import graphlib.TopologicalSorter 위상 정렬(topological sorting)은 유향 그래프의 꼭짓점(vertex)을 변의 방향을 거스르지 않도록 나열하는 것을 의미한다.
# Note that: divide and conquer, dynamic programming, Recurrence relation, greedy

# %%


class PositionalSystemsByBase:
    """
    기록 필요
    built-in format()
    https://docs.python.org/3/library/string.html#format-specification-mini-language
    """

    @staticmethod
    def get_other_based_number_from_ten_based_number(
        ten_based_number: int, other_based: Literal[2, 8, 16]
    ) -> str:
        match other_based:
            case 2:
                return format(ten_based_number, "b")
            case 8:
                return format(ten_based_number, "o")
            case 16:
                return format(ten_based_number, "X")
            case _:
                return None

    @staticmethod
    def get_ten_based_number_from_other_based_number(
        other_based_number: str, other_based: Literal[2, 8, 16]
    ):
        match other_based:
            case 2:
                return int("0b" + other_based_number, 2)
            case 8:
                return int("0o" + other_based_number, 8)
            case 16:
                return int("0x" + other_based_number, 16)
            case _:
                return None


def input_test():
    a, b = map(str, input().split())
    c: list[int] = list(map(int, input().split()))
    d = [list(map(int, input().split())) for _ in range(3)]
