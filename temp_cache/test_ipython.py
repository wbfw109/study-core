# %%
from __future__ import annotations

from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# %doctest_mode
cd ~/repo/intel-edge-academy-6/.vscode/ && code intel-edge-academy-6.code-workspace

# from google.protobuf.internal import api_implementation
# print(api_implementation.Type())
# %%
import cv2
# code $HOME/repo/intel-edge-academy-6/.vscode/intel-edge-academy-6.code-workspace
# Open a connection to the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Set the desired frame width and height
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 📍 Set the video format to MJPG. if not set, >> [ WARN:0@10.334] global cap_v4l.cpp:1136 tryIoctl VIDEOIO(V4L2:/dev/video0): select() timeout.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"YUYV"))
if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Video stream opened successfully.")
    # Loop to continuously capture frames from the webcam
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret will be True
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the resulting frame
        cv2.imshow("Webcam Video", frame)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


# %%
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")


# %%

print("abcde")


# %%
while True:
    # 프레임 단위로 비디오 읽기
    ret, frame = cap.read()

    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 프레임을 창에 표시
    cv2.imshow("Video Test", frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()

# %%
# import sys

# # 직사각형 행렬 중 일부 정사각형 행렬만 대각 대칭행렬 만들기.
# arr = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
# for x in arr:
#     x
# x1x2 = (1, 2)
# y1y2 = (2, 3)
# offset_i = x1x2[0]
# offset_j = y1y2[0]
# for i in range(x1x2[0], x1x2[1] + 1):
#     for j in range(y1y2[0] + 1, y1y2[1] + 1):
#         vi = i - offset_i
#         vj = j - offset_j
#         if vj > vi:
#             x = vj + offset_i
#             y = vi + offset_j
#             arr[i][j], arr[x][y] = arr[x][y], arr[i][j]
#             # print(x, y)
# for x in arr:
#     x
# #  피보나치 수, 세그먼트 트리, 플로이드, 다익스트라, 자카드 유사도
# # 그래프 매칭 문제: 3번은 트리가 주어지고, 루트에서부터 각 자식 노드로 바이러스가 전파될 때 적절한 절단점(cut vertex)을 찾고 최소 감염 노드의 수를 구하는 문제였는데, 문제에서 입력 조건이 노드의 최대 개수가 50개였기 때문에 적절한 방법으로 백 트래킹 한 후 모든 경우를 구하고, bfs로 시뮬레이션하면 최소 감염 노드의 수를 구할 수 있습니다!, N-back 문제
# # %%
# i = 10
# k = 3
# q, r = divmod(i, k)
# q
# r


# # %%
# import datetime as dt

# # datetime.weekday 기록.
# from dateutil.parser import parse

# # string parse time
# a = dt.datetime.strptime("2017-01-02 14:44", "%Y-%m-%d %H:%M")
# b = dt.datetime.strptime("2017-01-02 18:44", "%Y-%m-%d %H:%M")

# (a - b).total_seconds()
# (a - b).seconds
# parse("2017-01-02")
# parse("6/7/2016")  # month/days/year

# # string format time
# dt.datetime.strftime


# # %%
# import sys

# input_ = sys.stdin.readline
# # 문제 선정하기 ; https://devth-preview.goorm.io/exam/53763/%EC%BD%94%EB%94%A9-%ED%85%8C%EC%8A%A4%ED%8A%B8-%EC%9D%91%EC%8B%9C-%ED%99%98%EA%B2%BD-%EC%B2%B4%ED%97%98%ED%95%98%EA%B8%B0/quiz/1
# input_()
# if len(set(map(int, input_().split()))) > 3:
#     print("YES")
# else:
#     print("NO")


# # 근묵자흑 ; https://devth-preview.goorm.io/exam/53763/%EC%BD%94%EB%94%A9-%ED%85%8C%EC%8A%A4%ED%8A%B8-%EC%9D%91%EC%8B%9C-%ED%99%98%EA%B2%BD-%EC%B2%B4%ED%97%98%ED%95%98%EA%B8%B0/quiz/2?_ga=2.22476799.531264434.1598839678-1290480941.1598839678
# import math

# n, k = map(int, input_().split())
# input_()
# print(math.ceil((n - 1) / (k - 1)))
# # %%
# # https://khu98.tistory.com/66

# import sys

# input_ = sys.stdin.readline
# result = []
# for _ in range(int(input_())):
#     n, m = map(int, input_().split())
#     print(min(n // 5, (n + m) // 12))


# # 직사각형 한 점의 좌표
# from collections import Counter


# def solution(v):
#     result = []
#     for vv in zip(*v):
#         result.append(next(key for key, value in Counter(vv).items() if value == 1))
#     return result


# # %%
# # 코테 4번과 비슷한듯? https://www.acmicpc.net/problem/17471
# # dictionary (0, 0): 1, (0, 1): 2, (0, 2): 3, (1, 0): 4 ...  마스킹으로

# # 맵 객체 자체는 sorted가 되지만 map 객체의 컨테이너는 sorted 가 안된다.
# sorted([*map(int, "1 22 3".split()), *map(int, "1 22 3".split())])

# # LCA ; https://www.acmicpc.net/problem/3830
# # 다익스트라??? ; https://www.acmicpc.net/problem/13303
# # https://www.acmicpc.net/problem/15713
# # LCA, 이진 검색 트리, 피보나치 트리, 이항 트리, 다익스트라, 세그먼트 트리 하고 다시 알고리즘 재개..

# # LIS ; sequence[i] 보다 작은,  길이 l 에서 끝나는 인덱스의 값들 중  가지고 가장 큰 값의 길이 l 구하기


# import re
# from datetime import datetime, timedelta

# # 프로그래머스 효율성 테스트에서 break 후 return 하는 것보다 바로 return 할떄 효율성 테스트에서 만점 받음.
# pattern = re.compile(r"[a-z]")
# a = " a  b   c"
# a.split(" ")

# # https://www.acmicpc.net/problem/1648

# # Tessellation


# # %%

# ## 구간합 문제; 배열에서 i 부터 j 까지 인덱스에 있는 값을 구할 때 접두사 합 배열 P 사용하자.

# n = 5
# data = [10, 20, 30, 40, 50]

# prefix = [0]
# sum_value = 0

# for d in data:  # O(N)
#     sum_value += d
#     prefix.append(sum_value)

# # 쿼리가 주어짐 -> 2~3번째까지 구간의 합 (인덱스가 아님)
# left = 2
# right = 3

# result = prefix[right] - prefix[left - 1]
# print(result)
# 10e3  # 10000. not 1000.


# # %%
# players = ["mumu", "soe", "poe", "kai", "mine"]
# callings = ["kai", "kai", "mine", "mine"]


# def method1():
#     ranks = {name: i for i, name in enumerate(players)}
#     for name in callings:
#         rank = ranks[name]
#         pre_rank = rank - 1


# def method2():
#     ranks = {name: i for i, name in enumerate(players)}
#     for name in callings:
#         pre_rank, rank = ranks[name] - 1, ranks[name]


# timeit.timeit(method1, number=100000)  # 0.104612 s
# timeit.timeit(method2, number=100000)  # 0.116712 s
# # %%

# range_obj = range(100000)


# def method1():
#     """The reason method1 is slower than method2 is due to the additional overhead introduced by the generator expression in method1.
#     This overhead includes:
#         Creating the generator object.
#         Entering and exiting the generator context each time it yields a value.
#         Handling the StopIteration exception when the generator is exhausted.
#     """
#     for i in (i for i in range_obj if i & 1 == 1):
#         x = i


# def method2():
#     for i in range_obj:
#         if i & 1 == 1:
#             x = i


# timeit.timeit(method1, number=100)  # 0.58332s
# timeit.timeit(method2, number=100)  # 0.38724s


# # %%
# n = 5000


# def method1():
#     """When n is large enough, the accumulated cost of the extra addition operation in method1 could outweigh the one-time cost of the division operation in method2, resulting in method2 being faster overall."""
#     for _ in range(0, n, 2):
#         pass


# def method2():
#     for _ in range(n // 2):
#         pass


# # when n = 1000000001, 9.76544s and 10.79032s.
# # when n = 5000, 4.48339e-05s and 4.80699e-05s.
# timeit.timeit(method1, number=1)
# timeit.timeit(method2, number=1)


# # %%
# ## unpack receives iterable
# s = "1" * 10
# x = [*s]
# x
# from collections import deque

# s = "1" * 1000


# def method1():
#     """Python lists are implemented as dynamic arrays in the backend.
#     This means that they preallocate additional space for new elements.
#     When you add elements to a list using the * operator, Python can take advantage of this preallocated space and insert all elements at once, which is quite efficient.

#     The * operator, also known as the unpacking operator, is used in Python to unpack collections into separate elements.
#     When you use this operator within a list, it essentially 'unpacks' the elements of the collection you're referencing and places them individually into the new list.
#     """
#     x: list[str] = ["intercept", "x", *s]


# def method2():
#     """
#     deque.extend() method will have to individually append each character from s to the deque.
#     Since deques are implemented as doubly-linked lists in Python, this individual addition requires changing pointers for each new element added, which incurs additional overhead, thus taking longer time.
#     """
#     x = deque(["intercept", "x"])
#     x.extend(s)


# timeit.timeit(method1)  # 5.04185 s
# timeit.timeit(method2)  # 6.69755 s


# # %%
# ## profile required
# # https://www.geeksforgeeks.org/merge-two-sorted-arrays-python-using-heapq/
# def solution(r1, r2):
#     """
#     # x > 0 경우의 점들 개수 *4
#     # 100만개를 다 테스트해볼 수는 없음.
#     x=1, y=r1 ~ r2-1 까지가 x=1일떄 개수.
#     x=2, y=r2-1
#     4분면 나누고, 4분면 안에서도 절반 나누면.
#     a + b = r2 (a, b >= 0)


#     count 3 + 5 + 7 ..
#     r1    1   2
#     a = 1
#     a = 3, d = 2
#     ; initial_term a+2*(r1-1)
#     the number of terms = r2-r1

#     등차수열 솔루션은 틀린듯.
#         initial_term = 3+2*(r1-1)
#         n = r2-r1
#         return (n*(2*initial_term+(n-1)*2)//2 - n + 1)*4

#     """
#     initial_term = 3 + 2 * (r1 - 1)
#     n = r2 - r1
#     return (n * (2 * initial_term + (n - 1) * 2) // 2 - n + 1) * 4


# def solution(sequence, k):
#     """- `k는 항상 sequence의 부분 수열로 만들 수 있는 값입니다.`
#     - `이때 수열의 인덱스는 0부터 시작합니다.`

#     - `1 ≤ sequence의 원소 ≤ 1,000`
#     - `sequence는 비내림차순으로 정렬되어 있습니다.`
#     n^2 솔루션이 안된다. 각 sequence[i] 부터 뒤에꺼까지 부분합으로 k 보다 크기 바로 전까지 최소 pair 수 를 만들고 합이 안맞으면 패스.
#     """
#     n = len(sequence)
#     pa = [0]  # prefix array
#     for x in sequence:
#         pa.append(x + pa[-1])

#     for pair_num in range(1, n + 1):
#         for i in range(n - pair_num + 1):
#             x = pa[i + pair_num] - pa[i]
#             if x == k:
#                 return [i, i + pair_num - 1]
#             elif x > k:
#                 break
#     return [-1, -1]


# # [1, 2, 3].index(3, -1) #..


# # %%
# def solution(n):
#     dp: list[int] = [1, 2]  # Corrected initialization
#     for i in range(2, n):
#         dp[i % 2] = (
#             dp[0] + dp[1]
#         ) % 1000000007  # add modulo operation for avoiding overflow
#     return dp[(n - 1) % 2]


# solution(4)

# ## any((_ for _ in range(0)))# valid
# # %%


# def echo_round() -> Generator[int, float, str]:
#     sent = yield 0
#     while sent >= 0:
#         sent = yield round(sent)
#     return "Done"


# x = echo_round()
# next(x)
# x.send(10)
# # https://docs.python.org/3/library/exceptions.html#bltin-exceptions
# # When a generator or coroutine function returns, a new StopIteration instance is raised, and the value returned by the function is used as the value parameter to the constructor of the exception.


# # %%

# ## It can be possible. unpacking operator. *
# a: list[int] = [0, *(i for i in range(10))]

# s = "0123456789"
# s.rindex("4", 3, 9)
# s.rfind("2", -1, 3)

# s[:-1]

# # %%
# ## i = 0
# # for i in range(i, 5):
# #     pass
# # i
# # for i in range(i, 5):
# #     print("ok")

# ## list(((i, x) for i, x in enumerate(range(10, 15)) if i != 3))  # [(0, 10), (1, 11), (2, 12), (4, 14)]

# ## Generator technique
# import itertools

# x = [False, False, True, False, True, False, False, True, False, True, False, True]
# nth_true = 3  # Change this to get the nth true value

# nth_true_value = next(
#     itertools.islice((i for i, x in enumerate(x) if x), nth_true - 1, None), None
# )

# print(nth_true_value)

# # %%


# import math

# n = 500


# def method1():
#     unit = math.factorial(n)  # order_unit
#     for i in range(n, 0, -1):
#         unit //= i


# def method2():
#     for i in range(n - 1, -1, -1):
#         unit = math.factorial(i)  # order_unit


# timeit.timeit(method1, number=100)  # 5.04185s
# timeit.timeit(method2, number=100)  # 6.69755s
# # %%


# def method1():
#     a, b, c = list(range(discs, 0, -1)), [1, 5, 4, 3, 6, 3], [1, 2, 3, 4, 6, 3]


# def method2():
#     a = list(range(discs, 0, -1))
#     b = [1, 5, 4, 3, 6, 3]
#     c = [1, 2, 3, 4, 6, 3]


# timeit.timeit(method1)  # 0.66182s
# timeit.timeit(method2)  # 0.63321s


# # %%


# import itertools
# import operator

# n = 10000


# def method1():
#     """it is implemented in C: itertools.accumulate, operator.mul."""
#     factorial = list(itertools.accumulate(range(2, n), operator.mul))


# def method2():
#     factorial = list(range(2, n))
#     for i in range(1, n - 2):
#         factorial[i] *= factorial[i - 1]


# timeit.timeit(method1, number=100)  # 4.59333s
# timeit.timeit(method2, number=100)  # 5.09988s

# # %%

# import functools

# n = 1000


# def method1():
#     @functools.lru_cache(maxsize=None)
#     def fibonacci(n: int) -> int:
#         return n if n <= 1 else fibonacci(n - 1) + fibonacci(n - 2)

#     return fibonacci(n)


# def method2():
#     dp: list[int] = [0, 1]
#     for i in range(2, n + 1):
#         dp[i & 1] = dp[0] + dp[1]
#     return dp[n & 1]


# timeit.timeit(method1, number=1)  # 0.00224s
# timeit.timeit(method2, number=1)  # 9.55379e-05s


# # %%

# disks = 3
# move_num = itertools.count(1)


# def solve_tower_of_hanoi_recursively(disks: int, source: str, spare: str, target: str):
#     if disks == 1:
#         print(
#             f"[{next(move_num)}] Move disk 1 from peg {source} to peg {target}. not used peg is {spare}."
#         )
#         return
#     # move disks-1 from source to spare.
#     solve_tower_of_hanoi_recursively(disks - 1, source, target, spare)
#     # move the disk from source to target.
#     print(
#         f"[{next(move_num)}] Move disk {disks} from peg {source} to peg {target}. not used peg is {spare}."
#     )
#     # re-move (disks-1 moved to spare from source) to target.
#     solve_tower_of_hanoi_recursively(disks - 1, spare, source, target)


# # We are referring source as <from>, spare as <r>, and target as <target<
# solve_tower_of_hanoi_recursively(disks, "f", "r", "t")

# # %%
# from collections import deque


# # TODO: Binary solution with non-recursive implementation
# def solve_tower_of_hanoi(disks: int):
#     """
#     Time complexity: O(2^n)
#     Space complexity: O(n)

#     Implementation by using regularities
#     when n is odd-th move, (`if move & 1:`)
#         It always moves disk 1 from peg {source} to peg {target} (refer to Recursive solution).
#         - Every odd move involves the smallest disk.
#         - traversed <source>, <target> pegs pattern:
#             - <source> peg pattern: (A->B->C -> <cycle...>)
#             - <target> peg pattern: (B->C->A -> <cycle...>)
#             so, when even-th move end, I ran `peg.rotate(-1)`; Rotate peg for next odd ordering.
#     when n is even-th move, ...

#     Wiki 설명의 "move disk 0." 는 이동하지 않는다. 로 해석하면 될듯.
#     """

#     moves: int = 2**disks - 1
#     ## For debugging: following two lines
#     f, r, t = list(range(disks, 0, -1)), [], []
#     pegs: deque[list[int]] = deque([f, t, r]) if disks & 1 else deque([f, r, t])
#     peg_names: deque[Any] = (
#         deque(["f", "t", "r"]) if disks & 1 else deque(["f", "r", "t"])
#     )

#     # same as `while len(c) != disks:`
#     for move in range(1, moves + 1):
#         if move & 1:
#             # move disk from source to spare peg or from spare to target peg
#             pegs[1].append(pegs[0].pop())  # move. Smallest disk now on peg[1]
#             print(
#                 f"{move}. Move disk {pegs[1][-1]} from {peg_names[0]} to {peg_names[1]}"
#             )
#         else:
#             # move when only possible move
#             if pegs[0] and (not pegs[2] or pegs[0][-1] < pegs[2][-1]):
#                 source, destination = pegs[0], pegs[2]
#                 source_name, destination_name = peg_names[0], peg_names[2]
#             else:
#                 source, destination = pegs[2], pegs[0]
#                 source_name, destination_name = peg_names[2], peg_names[0]

#             destination.append(source.pop())
#             print(
#                 f"{move}. Move disk {destination[-1]} from {source_name} to {destination_name}"
#             )
#             pegs.rotate(-1)
#             peg_names.rotate(-1)

#         print(f, r, t, sep="\n", end="\n\n")

#     print(f"minimum moves: {moves}")


# solve_tower_of_hanoi(disks=4)


# # %%


# # task 안되는 오류?
# # TODO No-three-in-line problem
# def solution_12952(n: int) -> int:
#     """N-Queen ; https://school.programmers.co.kr/learn/courses/30/lessons/12952
#     deploy n Queen in n*n matrix.
#     백트래킹을.. 비재귀적으로 효율적으로 하는 방법이 있나?
#     N queen 겹치는지 확인할때 row major, 탐색이면 매 행마다 한개씩만 등록하고 북, 북동, 북서 방향으로만 체크하면 됨. 근데 북쪽 방향은 이전 행들의 컬럼 위치를 저장하여 같은지 확인하는게 더 빠르다.
#     ㅡ 평가; 추가했을때 리스트의 길이가 n이면 count +=1. 모든 경우에 대해 헌재 행의 컬럼을 저장하고 (pop), 컬럼인덱스가 7인지 확인하고 7까지에 대해 유효한지 체크.
#     이후 이를 row i= 0 까지 수행.
#     Row가 0일때 column 7에 도달했으면 break 하고 결과 반환. 그냥 처음 for 문 안에  while (True) 돌리면 될듯?
#     아니다 while 문으로 통일해야할듯.

#     - 연속된 두 행의 퀸의 컬럼 위치는 2 이상 차이난다.
#     - row-major order 에서 각 배치가능한 컬럼 위치 (추가적으로 북동쪽, 북서쪽만 탐색하면 되므로 더 효율적임).
#         각 행에 generator 넣으면 될 듯 한데. backtracking 을 위한 루프는 어떻게 구성?
#     가능한 컬럼 위치를 미리 계산해놓기?
#     1.
#     * - - - -
#     . . - - -
#     . - . - -
#     . - - . -
#     . - - - .

#     2.
#     * - - - -
#     . . * - -
#     . . . . -
#     . - . . .
#     . - . - .

#     3 ~ 4
#     * - - - -
#     . . * - -
#     . . . . *
#     . * . . .
#     . . . - .

#     5.
#     * - - - -
#     . . * - -
#     . . . . *
#     . * . . .
#     . . . * .


#     4*4 (edge cases)
#     * - - -         * - - -
#     . . - *         . . * -
#     . * . .         . . . .
#     . . . .         . - . .

#     answer (two cases):
#     - * - -         - - * -
#     - - - *         * - - -
#     * - - -         - - - *
#     - - * -         - * - -

#     TODO:
#         Symmetry 특징이라서 절만만 계산하면 되는듯 사실?


#     각 라인에서 유효한지 확인하면서 배치해도, 안되는 경우의수가 있음.


#     탐색 개수가 알려져 있음.

#     Key points
#         - propagate trace
#             - count to be able to propagate trace is known.
#         - backtracking triggers
#             - StopIteration
#             - i == n-1; answer found

#     가능하면, 이전에 했던것부터 -1.
#     아랫줄만 전파. 데 이렇게하면, 각 리스트에서
#     end 시점을 어떻게?

#     valid_columns_by_row 가 list 가 필요없을거같은데 -

#     """
#     # Naive solution
#     from typing import Generator

#     def get_valid_columns_gen(i: int) -> Generator[int, None, None]:
#         yield from (j for j in range(n) if traces[i][j] == 0)

#     traces: list[list[int]] = [[0] * n for _ in range(n)]
#     backtracking_by_row: list[list[tuple[int, int]]] = []
#     valid_columns_by_row: list[Generator[int, None, None]] = [
#         get_valid_columns_gen(i) for i in range(n)
#     ]
#     answer: int = 0
#     i = 0
#     columns = []
#     while True:
#         # explore valid Queen points. last row can not propagate trace. so, range(..., n-1)
#         while i < n:
#             try:
#                 j = next(valid_columns_by_row[i])
#                 columns.append(j)
#                 movable_range: list[tuple[int, int]] = []

#                 ## Search with a direction
#                 nmi = n - i  # n minus i
#                 valid_count = [nmi - 1, min(nmi, n - j) - 1, min(nmi - 1, j)]
#                 for di, d in enumerate([(1, 0), (1, 1), (1, -1)]):
#                     p = (i, j)
#                     for _ in range(valid_count[di]):
#                         p = p[0] + d[0], p[1] + d[1]
#                         movable_range.append(p)
#                         traces[p[0]][p[1]] += 1
#                 backtracking_by_row.append(movable_range)
#             except StopIteration:
#                 if i == 0:
#                     return answer
#                 # backtracking
#                 columns.pop()

#                 for p in backtracking_by_row[-1]:
#                     traces[p[0]][p[1]] -= 1
#                 backtracking_by_row.pop()
#                 valid_columns_by_row[i] = get_valid_columns_gen(i)
#                 i -= 1
#             else:
#                 i += 1
#         else:
#             answer += 1
#             print(columns)
#             columns.pop()
#             backtracking_by_row.pop()
#             i -= 1


# solution_12952(4)
# # %%

# n = 10


# def n_queens(
#     n: int, i: int, a: list[int], b: list[int], c: list[int]
# ) -> Generator[list[int], None, None]:
#     """
#     a := columns' index.
#     b := used to check top-right to bottom-left diagonal.
#     c := used to check top-left to bottom-right diagonal.

#     (two positions are on the same diagonal if and only if they have the same i+j or i-j values).

#     This happens because when you're considering the cells near the top of the board, the i - j value is negative for cells on the right side of a ↘️ diagonal and positive for those on the left. Therefore, it appears that c is preventing placement on ↘️ diagonals for the top part of the board.
#     However, for the majority of the board, the c list does indeed track ↙️ diagonals (where the i - j value is constant), and the b list tracks ↘️ diagonals (where the i + j value is constant).
#     This seeming contradiction at the top of the board results from the fact that i - j is not a perfect identifier for ↙️ diagonals because it produces the same result for cells on ↘️ diagonals near the top of the board. Despite this, i - j is still used to track ↙️ diagonals because it works for the majority of cells on the board.

#     to use set() instead of list is faster if only the number of cases are required.
#     b, c 는 set() 가능할 듯? 어차피 a만 반환함ㄴ 되서.
#     """
#     if i < n:
#         for j in range(n):
#             if j not in a and i + j not in b and i - j not in c:
#                 yield from n_queens(n, i + 1, a + [j], b + [i + j], c + [i - j])
#     else:
#         yield a


# timeit.timeit(lambda: n_queens(n, 0, [], [], []), number=100)

# (4).bit_length()

# # %%


# a = Counter([1, 3, 5, 3, 3, 4])
# b = Counter([1, 3, 5, 3])
# a + b
# (a + b)
# min(a, b)

# # %%
# # New in version 3.10: Rich comparison operations were added. 자카드 유사도

# # %%
# # set 속도 차이.. add 두번과 update 한번.
# # pop_set.add(j)
# # pop_set.add(jp1)


# # %%

# from collections import OrderedDict

# x = OrderedDict().fromkeys([1, 2, 3])
# x
# # x.move_to_end(3)
# # x
# # x.popitem()
# # x
# # x.popitem(False)
# # x
# from collections import deque


# def solution(cacheSize, cities):
#     dq = deque(maxlen=cacheSize)
#     run_time = 0
#     for city in cities:
#         city = city.lower()
#         if city not in dq:  # cache miss
#             dq.append(city)
#             run_time += 5
#         else:  # cache hit
#             dq.remove(city)
#             dq.append(city)
#             run_time += 1

#     return run_time


# # %%

# msg = "ABCDE"
# # x+1 .. +2.. 에 대하 ㄴ것과 만든 string 에서 하나씩 추가하는 것 시간차이?
# msg[1:x]

# # %%


# a = [["", 123], ["abc", 123, "abc"]]
# a.sort()
# a

# # %%

# # arr.append("abc"[4:]) # result = ['']

# # %%


# def gen():
#     yield from reversed(range(10))


# x = gen()
# next(x)
# next(x)
# next(x)
# next(x)


# # %%
# x = [True] * 5
# x_gen = ((i, y) for i, y in enumerate(x) if y)
# next(x_gen)
# x[2] = False
# next(x_gen)
# next(x_gen)
# next(x_gen)

# # >>> a = [1,2,3]
# # >>> a[5:] = [2,3]
# # >>> a
# # [1, 2, 3, 2, 3]

# # >>> a= [1,2,3]
# # >>> a[-1:] = [1]
# # >>> a
# # [1, 2, 1]
# # %
# # %%

# [(i, j) for i in range(3) for j in range(4, 5)]
# # %%
# from tensorflow.python.client import device_lib

# device_lib.list_local_devices()

# # 으악 백업
