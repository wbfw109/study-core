from __future__ import annotations

import time
import unittest
from pprint import pprint
from typing import Iterator, Optional

# 중량 제한 ; https://www.acmicpc.net/problem/1939
# 공유기 설치 ; https://www.acmicpc.net/problem/2110
# 개똥벌레 ; https://www.acmicpc.net/problem/3020

# data structure 업데이트.. visualization root 포함되게
# escape_marble_2 구슬만 움직이도록 해서 다시-


def hang_balloons_to_teams(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/4716

    우선순위가 주어짐. 힙 문제?
    기회비용 비교?


    """
    pass


def weigh_weights_on_the_scales(input_lines: Optional[Iterator[str]] = None) -> str:
    """https://www.acmicpc.net/problem/2437
    첫째 줄에 주어진 추들로 측정할 수 없는 양의 정수 무게 중 최솟값을 출력한다.

    정렬하고, 에라토스테네스의 체처럼 하면 될듯 보임. 해당 숫자가 나오면 = True 하고

    포인터는 1 인덱스부터 (정수 1을 의미. 0은 안씀.)
    저울추가 동일한 무게를 가질 수도 있음.

    정수 count
    1 3 3 4 6 7 8 9 0
    hash[1] = True
    if hash[count] is False:
        count 가 `
        # 이전것들을 이용해서 조합한다.
    hash[
    hash[1] = True
    if hash[count] is False:
        # 이전것들을 이용해서 조합한다.

    현재 루프 안에서 몇개를 조합해서 다해야할지, 다음 독립변수를 변경하고 그대로 조합하명 이전까지의 것을 포함시키는지흘 확인해야한다.
    다른 알고리즘에서도 필수적으로 들어가지만 정렬 알고라즘에서는 보통 두 개 이상의 포인터를 직접 제어해야해서 어려워진다.
    한번에 다 해보려고하지말고 2개씩 해본다. 점화삭 세워보기?

    메모리 초과가 난다.. set 사용할떄 찾은거는 pop 해주자..?

    sorted 보다는 sort가 더빠르다. (in-place sorting vs create new sorted array)

    다음 수 업데이트하는 트리거? count 값의 bool 이 False 일떄?
    if array[next_i] <= count:
        update -

    if not hash_map[count]:
        returns
    다음 저울추 무게 l이 k보다 작거나 같다면 다음 저울추로 (1~l)~(k+l)의 무게를 모두 만들 수 있다.

    루프마다 현재 루프의 array[i] 무게추가 포함된 무게추의 조합을 만들고, 이를 그대로 다음 루프에서 이용하는 방식.
    근데 이렇게하면 중복?이 생기려나 상관없나
    1 2 3 5 6
    Hard....
    어떤 조건이라면 이전에 추가한 잴 수 있는 무게들 루프를 거치지 않고 구할 수 있는가?

    - 1 짜리 추가 없다면 -> [X] 1

    - 만들 수 있는 추가 1 ~ X 추까지 만들 수 있고 X 가 다음 추는 Y 의 크기만큼 됨.
    # 다음거가 만들 수있는 추에 포함되거나 원래 추의 크기 +1 이하라면
        1 2 4 9
        -> 1    -> 2,3       -> 4,5,6,7  -> [X] 9
        만들 수 있는 추는 X+Y.
        i+=1
    중복이 있는 경우
        1 1 3 9
        -> 1    -> 2,3       -> 4,5,6,7  -> [X] 9
        만들 수 있는 추는 X+Y.
        i+=1
    미분의 특징?.. 수학으로 나타낼 수 있나

    이건 귀납법...


    1 2 3 5 20 weights
    --- a Loop ---
    1-st: 1.         hash_map[1] = True
    2-rd: 2. 2+1    hash_map[2, 3] = True
    3-th: -
    4-th: 3. 3+1, 3+2, 3+(2+1)
    5-th: -
    6-th: -
    7-th: 5. 5+1, 5+2, 5+(2+1), 5+3, 5+(3+1), 5+(3+2), 5+(3+2+1)
    8-th: -
    9-th: -
    10-th: -
    11-th: end

    이전까지 추가한 측정가능한 무게만큼 루프르 돌고 자신을 추가하는 구현에서ㅡ
    set 를 해시 테이블로서 사용하려고 하니까 메모리초과난다.
    루프를 돌아야 하는 횟수가 Sum(n) = (1+ Sum(n-1)) 이라서.. 기하급수적으로 늘어난다.
    Sum(0) = 1
    (1)+(1+1)+(1+3)+(1+7)
    """

    import sys

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = sys.stdin.readline

    # Title: input
    # condition (1 ≤ N < 1000)
    n: int = int(input_())
    # condition (1 ≤ each weight of weights ≤ 10^6)
    # negative integer is acid solution, positive integer is alkaline solution.
    weights: list[int] = list(map(int, input_().split()))
    # condition (1 ≤ weights to be measured ≤ n)
    not_found_weight: int = 1
    measurable_weight: int = 0

    # Title: solve
    weights.sort()
    for weight in weights:
        if weight <= measurable_weight + 1:
            measurable_weight += weight
        else:
            not_found_weight = measurable_weight + 1
            break
    else:
        not_found_weight = measurable_weight + 1

    # Title: output
    print(not_found_weight)
    return str(not_found_weight)


def test_weigh_weights_on_the_scales() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "7",
                "3 1 6 2 7 30 1",
            ],
            ["21"],
        ],
        [
            [
                "5",
                "1 2 3 4 5",
            ],
            ["16"],
        ],
    ]:
        start_time = time.time()
        test_case.assertEqual(
            weigh_weights_on_the_scales(iter(input_lines)), output_lines[0]
        )
        print(f"elapsed time: {time.time() - start_time}")


test_weigh_weights_on_the_scales()
