"""Purpose is to solve in 10 minutes
- 💤: 문제 이해 제대로 하기
    "문제 설명"에 모든 제한사항이 함게 주어지지 않는 경우가 있어서, 완벽히 이해하기 어려울 수 있다.
    에매한 경우, 명확하게 이해할 수 있도록 "예시" 가 주어지므로 꼭 함께 보도록 한다.
- 🧠: (신박한 풀이, 수학식을 활용한 풀이, Benchmark)
- 💦: built-in functions, grammar

"""
# Regex to find problems in which one of upper emojis was used; """[^\uAC00-\uD7A3\d\w]\s


def solution_181188(targets: list[list[int]]) -> int:
    """💤 요격 시스템 ; https://school.programmers.co.kr/learn/courses/30/lessons/181188
    Tag: Greedy, Sweep line algorithm

    Other solution
        count: int = 0
        targets.sort()
        i: int = 0
        targets_len: int = len(targets)
        while i < targets_len:
            maxx = targets[i][1]
            j = i + 1  # j is next target pointer.
            while j < targets_len:
                if targets[j][0] < maxx:
                    maxx = min(maxx, targets[j][1]) ## it is not required.
                    j += 1
                else:
                    break
            i = j  # set next pointer
            count += 1
        return count

    Consideration
        요격미사일의 최소개수도 중요하긴 한데, 모든 폭격 미사일을 맞춰야 하기 때문에 그리디 접근방식 필요할듯 보임.

    Debugging
        1, 4    # 1
        3, 7    # 1
        4, 5    # 2
        4, 8    # 2
        5, 12   # 2
        10, 14  # 2
        11, 13  # 3
    """
    count: int = 0
    targets.sort(key=lambda x: x[0])
    max_endpoint: int = 0
    for x1, x2 in targets:
        if x1 >= max_endpoint:
            count += 1
            max_endpoint = x2
    return count


## TODO ~
def solution_12900(n: int) -> int:
    """2 * n 타일링 ; https://school.programmers.co.kr/learn/courses/30/lessons/12900
    Tag: Dynamic programming
    """
    count: int = 0
    return count


def solution_12899(n: int) -> str:
    """🧠 124 나라의 숫자 ; https://school.programmers.co.kr/learn/courses/30/lessons/12899
    Tag: Math (Base Conversion)

    - 😠 int 를 반환하는 것이 아니라 str 을 반환하는 것이다.
    Other Implementation
        1. [TimeOut] itertools.product
        2. O(n) Solution
            # Purpose: count(k) < n <= count(k+1)
            #     count(max_digit(d))
            #         d=0: 0
            #         d=1: 3
            #         d=2: 3*3
            #         d=3: 3*3*3
            count = 0
            for d in range(1, n + 1):
                max_d = 3**d
                count += max_d
                if n <= count:
                    result: list[str] = []
                    order = n - (count - max_d)  # inner_order
                    for dd in range(d - 1, 0, -1):
                        q, r = divmod(order, 3**dd)
                        result.append("124"[q - 1] if r == 0 else "124"[q])
                        order = r
                        if dd == 1:
                            result.append("124"[r - 1])
                            break
                    else:
                        result.append("124"[order - 1])

                    return "".join(result)
        Debugging
            7 (21) ; divmod(7, 3) == 2, 1
                instead -> divmod(7-1, 3) == 2, 0     -> divmod(2-1, 3) == 0, 1
            124
            141
            142
            144    1, 0 = divmod(9, 9). 0, 0 = divmod(0, 3).

            211
            212
            214    1, 3 = divmod(12, 9). 1, 0 = divmod(3, 3)
    """
    ## O(log n) solution
    num = "124"
    answer: list[str] = []
    while n > 0:
        n -= 1  # in order to match directly Zero-based index of string "124" to result of divmod().
        q, r = divmod(n, 3)
        answer.append(num[r])
        n = q
    return "".join(reversed(answer))


def solution_1844(maps: list[list[int]]) -> int:
    """게임 맵 최단거리 ; https://school.programmers.co.kr/learn/courses/30/lessons/1844
    Tag: BFS
    """
    from collections import deque

    n, m = len(maps), len(maps[0])
    turn = 1  ## important
    dq: list[deque[tuple[int, int]]] = [deque([(0, 0)]), deque()]
    maps[0][0] = -1  ## is_visited
    p = 0
    target = n - 1, m - 1
    if target != (0, 0):
        turn += 1
    while dq[p]:
        x, y = dq[p].popleft()
        for nx, ny in (
            (x + dx, y + dy) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ):
            if 0 <= nx < n and 0 <= ny < m and maps[nx][ny] == 1:
                if target == (nx, ny):
                    return turn
                maps[nx][ny] = -1
                dq[p ^ 1].append((nx, ny))

        if len(dq[p]) == 0:
            if dq[p ^ 1]:
                p ^= 1
                turn += 1
            else:
                turn = -1
                break

    return -1
