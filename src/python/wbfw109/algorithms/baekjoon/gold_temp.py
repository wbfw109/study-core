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
    """https://www.acmicpc.net/problem/4716"""
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

    sum (generator) 보다 이미 있는 리스트라면 인덱스로 접근해서 직접 더하는게 더 빠른가?
    ㅇㅇ 맞다.. 유의미한 차이를 보인다.. 통과 못햇음.. generator 도 결국 그만큼 순환해서 생성해야 하므로..

    sorted 보다는 sort가 더빠르다. (in-place sorting vs create new sorted array)

    """
    pass


def test_weigh_weights_on_the_scales() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            [
                "7",
                "3 1 6 2 7 30 1",
            ],
            ["21"],
        ]
    ]:
        start_time = time.time()
        test_case.assertEqual(
            weigh_weights_on_the_scales(iter(input_lines)), output_lines[0]
        )
        print(f"elapsed time: {time.time() - start_time}")


test_weigh_weights_on_the_scales()
