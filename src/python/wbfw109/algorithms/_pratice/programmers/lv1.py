"""Purpose is to solve in 4 minutes
- 💤: 문제 이해 제대로 하기
    "문제 설명"에 모든 제한사항이 함게 주어지지 않는 경우가 있어서, 완벽히 이해하기 어려울 수 있다.
    에매한 경우, 명확하게 이해할 수 있도록 "예시" 가 주어지므로 꼭 함께 보도록 한다.
- 🧠: (신박한 풀이, 수학식을 활용한 풀이, Benchmark)
- 💦: built-in functions, grammar

"""
# Regex to find problems in which one of upper emojis was used; """[^\uAC00-\uD7A3\d\w]+\s


def solution_178871() -> None:
    """달리기 경주 ; https://school.programmers.co.kr/learn/courses/30/lessons/178871"""


def solution_176963() -> None:
    """추억 점수 ; https://school.programmers.co.kr/learn/courses/30/lessons/176963"""


def solution_172928() -> None:
    """공원 산책 ; https://school.programmers.co.kr/learn/courses/30/lessons/172928"""


def solution_161990() -> None:
    """바탕화면 정리 ; https://school.programmers.co.kr/learn/courses/30/lessons/161990"""


def solution_161989() -> None:
    """덧칠하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/161989"""


def solution_160586() -> None:
    """대충 만든 자판 ; https://school.programmers.co.kr/learn/courses/30/lessons/160586"""


def solution_159994() -> None:
    """카드 뭉치 ; https://school.programmers.co.kr/learn/courses/30/lessons/159994"""


def solution_155652() -> None:
    """둘만의 암호 ; https://school.programmers.co.kr/learn/courses/30/lessons/155652"""


def solution_150370() -> None:
    """개인정보 수집 유효기간 ; https://school.programmers.co.kr/learn/courses/30/lessons/lv150370"""


def solution_147355() -> None:
    """크기가 작은 부분 문자열 ; https://school.programmers.co.kr/learn/courses/30/lessons/147355"""


def solution_142086() -> None:
    """가장 가까운 같은 글자 ; https://school.programmers.co.kr/learn/courses/30/lessons/142086"""


def solution_140108() -> None:
    """문자열 나누기 ; https://school.programmers.co.kr/learn/courses/30/lessons/140108"""


def solution_138477() -> None:
    """명예의 전당 (1) ; https://school.programmers.co.kr/learn/courses/30/lessons/138477"""


def solution_136798() -> None:
    """기사단원의 무기 ; https://school.programmers.co.kr/learn/courses/30/lessons/136798"""


def solution_135808(k: int, m: int, score: list[int]) -> int:
    """🧠 과일 장수 ; https://school.programmers.co.kr/learn/courses/30/lessons/135808

    Other solution
        # 🆚 Current implementation, this solution; These have similar elapsed time in Test cases.
        # Time complexity: O(n)
        # Space complexity: O(k)
        answer: int = 0
        counts: list[int] = [0] * (k + 1)
        for score_ in score:
            counts[score_] += 1

        previous_r: int = 0
        for i in range(k, 0, -1):
            q, r = divmod(counts[i] + previous_r, m)
            answer += i * m * q
            previous_r = r
        return answer
    """
    # Time Complexity: O(n log n)
    return sum(sorted(score)[len(score) % m :: m]) * m


def solution_134240(food: list[int]) -> str:
    """💤 푸드 파이트 대회 ; https://school.programmers.co.kr/learn/courses/30/lessons/134240
    - 2 ≤ food의 길이 ≤ 9
    """
    left = "".join((str(i) * (food[i] // 2) for i in range(1, len(food))))
    return "".join((left, "0", "".join(reversed(left))))


# TODO: --
def solution_133502(ingredient: list[int]) -> None:
    """햄버거 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/133502"""
    from collections import deque

    dq = [deque() for _ in range(4)]
    for i, x in enumerate(ingredient):
        dq[x].append(i)
    while True:
        if len(dq[1]) >= 2 and dq[1]:
            pass

    # return min(counts[1]//2, counts[2], counts[3])


def solution_133499(babbling: list[str]) -> int:
    """🧠 옹알이 (2) ; https://school.programmers.co.kr/learn/courses/30/lessons/133499

    - `len(word.strip())==0` is slightly faster than `if not word.split():`
    """
    valid_str: list[str] = ["aya", "ye", "woo", "ma"]
    not_valid_str: list[str] = [s * 2 for s in valid_str]
    count: int = 0
    for word in babbling:
        if any((word.find(s) >= 0 for s in not_valid_str)):
            continue
        # prevent separate words from attaching each others by replacing <s> with " " instead of "".
        for s in valid_str:
            word = word.replace(s, " ")
        if len(word.strip()) == 0:
            count += 1
    return count


def solution_132267(a: int, b: int, n: int) -> int:
    """🧠🔍 콜라 문제 ; https://school.programmers.co.kr/learn/courses/30/lessons/132267

    Consideration
        - The payback is not in 10% percent unit unlike "치킨 쿠폰" problem of lv0.
        - condition: ( 1 ≤ b < a ≤ n ≤ 1,000,000 )

    Other solutions:
        answer = 0
        while n >= a:
            q, r = divmod(n, a)
            new_cola = q*b
            answer+=new_cola
            n = new_cola+r
        return answer

    Implementation
        give=3, get=1
        -----
        n=7
        n = 7 + (-3+1) = 5
        n = 5 + (-3+1) = 3
        n = 3 + (-3+1) = 1
        7 / 2 == 3.5
        -----
        n=6
        n = 6 + (-3+1) = 4
        n = 4 + (-3+1) = 2
        6 / 2 == 3.0

        While repeating n - (a-b) - (a-b) ..., if the result value is less than a, do not proceed further.
        ❓ to do that, n must be changed in the exchange system. It can be thought as firstly <a> is consumed, but <b> is not received.
        ; (n - b) // (a - b) * b
    """
    return (n - b) // (a - b) * b


def solution_131705(number: list[int]) -> int:
    """🧠 삼총사 ; https://school.programmers.co.kr/learn/courses/30/lessons/131705
    Tag: Math (3SUM variant)
    Other solutions
        - Brute-force solution (slower than current implementation)
            from itertools import combinations
            return sum(1 if sum(comb) == 0 else 0 for comb in combinations(number,3))
    """
    import math

    n: int = len(number)
    count: int = 0
    number.sort()
    for i in range(n - 2):
        j = i + 1
        k = n - 1
        while j < k:
            total = number[i] + number[j] + number[k]
            if total < 0:
                j += 1
            elif total > 0:
                k -= 1
            else:
                if number[j] == number[k]:
                    count += math.comb(k - j + 1, 2)
                    break
                else:
                    left, right = j, k
                    while left + 1 < right and number[left + 1] == number[j]:
                        left += 1
                    while right - 1 > left and number[right - 1] == number[k]:
                        right -= 1
                    count += (left - j + 1) * (k - right + 1)
                    j = left + 1
                    k = right - 1
    return count


def solution_131128(X: str, Y: str) -> str:
    """🧠 숫자 짝꿍 ; https://school.programmers.co.kr/learn/courses/30/lessons/131128
    🔍 Why str.count() solution faster than Counter() solution?
        str.count() solution iterates than 2 times, even if Counter() convert all characters to int()
        , so I think prior solution is slower than latter solution. but not.

    Counter() solution
        from collections import Counter
        counters: list[Counter[str]] = [Counter(X), Counter(Y)]
        result: str = "".join((
            num_s * counters[0][num_s]
            if counters[0][num_s] < counters[1][num_s]
            else num_s * counters[1][num_s]
            for num_s in "9876543210"
        ))
        if result:
            return "0" if result[0] == "0" else result
        else:
            return "-1"
    """
    result: str = "".join(
        (num_s * min(X.count(num_s), Y.count(num_s))) for num_s in "9876543210"
    )
    if result:
        return "0" if result[0] == "0" else result
    else:
        return "-1"


def solution_118666(survey: list[str], choices: list[int]) -> str:
    """성격 유형 검사하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/118666
    Clues
        - `단, 하나의 지표에서 각 성격 유형 점수가 같으면, 두 성격 유형 중 사전 순으로 빠른 성격 유형을 검사자의 성격 유형이라고 판단합니다.`
            given string "RTCFJMAN" is already sorted alphabetically.
    """
    scores: dict[str, int] = {s: 0 for s in "RTCFJMAN"}
    for (a, b), choice in zip(survey, choices):
        diff: int = 4 - choice
        score: int = abs(diff)
        scores[a if diff > 0 else b] += score
    return "".join(
        (a if scores[a] >= scores[b] else b for a, b in ("RT", "CF", "JM", "AN"))
    )


def solution_92334(id_list: list[str], report: list[str], k: int) -> list[int]:
    """💤 신고 결과 받기 ; https://school.programmers.co.kr/learn/courses/30/lessons/92334

    Clues
        - `한 유저를 여러 번 신고할 수도 있지만, 동일한 유저에 대한 신고 횟수는 1회로 처리됩니다.`
            it denotes that hashable type (set or dict) may be required.
        - `k번 이상 신고된 유저는 게시판 이용이 정지되며, 해당 유저를 신고한 모든 유저에게 정지 사실을 메일로 발송합니다.`
            it denotes that hashable type I have to use is dictionary type.; `reported: dict[int, set[int]] = defaultdict(set)`
        - `return 하는 배열은 id_list에 담긴 id 순서대로 각 유저가 받은 결과 메일 수를 담으면 됩니다.`
            it denotes that given names must be converted to id numbers.; map_ = {id_: i for i, id_ in enumerate(id_list)}
            and the list whose elements denotes reported count may be required.; `result: list[int] = [0] * len(id_list)`
    """
    from collections import defaultdict

    map_ = {id_: i for i, id_ in enumerate(id_list)}
    reported: dict[int, set[int]] = defaultdict(set)
    for r in report:
        a, b = r.split()
        reported[map_[b]].add(map_[a])

    result: list[int] = [0] * len(id_list)
    for reporters in reported.values():
        if len(reporters) >= k:
            for reporter in reporters:
                result[reporter] += 1
    return result


def solution_87389(n: int) -> int:
    """💤💦 나머지가 1이 되는 수 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/87389
    - `3 ≤ n ≤ 1,000,000`
    """
    return min((i for i in range(2, int(n**0.5) + 1) if n % i == 1), default=n - 1)


def solution_86491(sizes: list[list[int]]) -> int:
    """🧠 최소직사각형 ; https://school.programmers.co.kr/learn/courses/30/lessons/86491
    Definition
        n := size of <sizes>
    Time compelxity: O(n)
        - n from one loop
    Space compelxity: O(1)

    Axiom
        When x + y := (rectangle perimeter)/2,  as (diff:= abs(x-y)) is decreased, width is increased.
        🛍️ e.g. if x + y = 10,
            1*9 = 9,  2*8 = 16,  ... 5*5 = 25
    Proof
        1. x + y = perimeter/2;    y = perimeter/2 - x
        2. x * y = -x^2 + x*perimeter/2 = width
            It is Quadratic function in which the parabola opens downwards.
            Through the differential equation, the point at which the slope becomes zero (Maximum width) can be found.
            namely, -2x + perimeter/2 = 0;  x = perimeter/4;  # it means the case rectangle is square.

    In given definition of problems unless x == y, The length of one side can be either the maximum or minimum value.

    Same solution:
        return max((x if x > y else y for x, y in sizes)) * max((x if x < y else y for x, y in sizes))
    """
    max_shorter_side, max_longer_side = 0, 0

    for x, y in sizes:
        # set (shorter_side, longer_side)
        if x < y:
            shorter_side, longer_side = x, y
        else:
            shorter_side, longer_side = y, x

        # set (max_shorter_side, max_longer_side)
        if max_shorter_side < shorter_side:
            max_shorter_side = shorter_side
        if max_longer_side < longer_side:
            max_longer_side = longer_side

    return max_shorter_side * max_longer_side


def solution_86051(numbers: list[int]) -> int:
    """없는 숫자 더하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/86051"""
    return sum(set(range(10)).difference(numbers))


def solution_82612(price: int, money: int, count: int) -> int:
    """부족한 금액 계산하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/82612"""
    return diff if (diff := count * (count + 1) * price // 2 - money) > 0 else 0


def solution_81301(s: str) -> int:
    """💤 숫자 문자열과 영단어 ; https://school.programmers.co.kr/learn/courses/30/lessons/81301
    Clues
        - `s가 "zero" 또는 "0"으로 시작하는 경우는 주어지지 않습니다.`
    """
    map_: dict[str, str] = {
        word: str(num)
        for word, num in zip(
            (
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            ),
            range(10),
        )
    }
    for word in map_:
        s = s.replace(word, map_[word])
    return int(s)


def solution_77884(left: int, right: int) -> int:
    """🧠 약수의 개수와 덧셈 ; https://school.programmers.co.kr/learn/courses/30/lessons/77884"""
    return sum(
        (
            -num if float(num**0.5).is_integer() else num
            for num in range(left, right + 1)
        )
    )


def solution_77484(lottos: list[int], win_nums: list[int]) -> list[int]:
    """로또의 최고 순위와 최저 순위 ; https://school.programmers.co.kr/learn/courses/30/lessons/77484"""
    ranks: list[int] = [6, 6, 5, 4, 3, 2, 1]
    min_answer_count = len(set(lottos).intersection(win_nums))
    max_answer_count = min_answer_count + lottos.count(0)  # "zero" are wildcards.
    return [ranks[max_answer_count], ranks[min_answer_count]]


def solution_76501(absolutes: list[int], signs: list[bool]) -> int:
    """음양 더하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/76501"""
    return sum((num if sign else -num for num, sign in zip(absolutes, signs)))


def solution_72410(new_id: str) -> str:
    """🧠 신규 아이디 추천 ; https://school.programmers.co.kr/learn/courses/30/lessons/72410
    Definition
        n := <new_id>'s length.
    Time complexity: O(n)
        - n from `validate with while simultaneously creating recommendation string`
        - 15 ~ 30 from `Did it pass validation?`
    Space complexity: O(1)
        - 8 from `stack`
        - 7 from `dots`

    Consideration
        💡 How to merge processes of validation and recommendation to string in one iteration?
            - step (5, 7) and 6 is exclusive. so order will 4 -> (5 or 6).
            - step 4 and 6 have a same process.
            - validation can be done by comparing <new_id> and recommendation id
                , except for condition (<new_id> length < 3)
    """
    stack: list[str] = []
    count = 0
    last_dot: int = 0  # dot num
    ## validate with while simultaneously creating recommendation to string.
    for s in new_id:
        if s.isalpha():
            count += 1
            stack.append(s.lower())
        elif s.isdigit() or s in "-_":
            count += 1
            stack.append(s)
        elif s == ".":
            # if dot is first, or dots are consecutive.
            if last_dot == count:
                continue
            else:
                count += 1
                last_dot = count
                stack.append(".")
        else:
            continue

        if count == 15:
            break
    # if stack size is less than 15, or last character is ".".
    if stack and stack[-1] == ".":
        stack.pop()

    # Did it pass validation?
    if "".join(stack) != new_id or len(new_id) < 3:
        if not stack:
            stack.append("a")
        if (new_len := len(stack)) <= 2:
            stack += [stack[-1]] * (3 - new_len)
        return "".join(stack)
    else:
        return new_id


def solution_70128(a: list[int], b: list[int]) -> int:
    """내적 ; https://school.programmers.co.kr/learn/courses/30/lessons/70128"""
    return sum((aa * bb for aa, bb in zip(a, b)))


def solution_68935(n: int) -> int:
    """3진법 뒤집기 ; https://school.programmers.co.kr/learn/courses/30/lessons/68935"""
    result: list[str] = []
    while n > 0:
        q, r = divmod(n, 3)
        result.append(str(r))
        n = q
    return int("".join(result), 3)


def solution_68644(numbers: list[int]) -> list[int]:
    """두 개 뽑아서 더하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/68644"""
    from itertools import combinations

    return sorted(set((sum(comb) for comb in combinations(numbers, 2))))


def solution_67256(numbers: list[int], hand: str) -> str:  # type: ignore
    """키패드 누르기 ; https://school.programmers.co.kr/learn/courses/30/lessons/67256
    Consideration
        thumbs' distance method is Taxicab distance.
        when press the numer button, thumb's location is changed as the number's location.
    """
    num_map = {
        1: (0, 0),
        2: (0, 1),
        3: (0, 2),
        4: (1, 0),
        5: (1, 1),
        6: (1, 2),
        7: (2, 0),
        8: (2, 1),
        9: (2, 2),
        0: (3, 1),
    }
    hand: str = hand[0].upper()
    left: tuple[int, int] = (3, 0)  # left hand thumb's point
    right: tuple[int, int] = (3, 2)  # right hand thumb's point
    result: list[str] = []
    for number in numbers:
        if number in (1, 4, 7):
            left = num_map[number]
            result.append("L")
        elif number in (3, 6, 9):
            right = num_map[number]
            result.append("R")
        else:
            left_diff = abs(num_map[number][0] - left[0]) + abs(
                num_map[number][1] - left[1]
            )
            right_diff = abs(num_map[number][0] - right[0]) + abs(
                num_map[number][1] - right[1]
            )
            if left_diff == right_diff:
                if hand == "L":
                    left = num_map[number]
                else:
                    right = num_map[number]
                result.append(hand)
            elif left_diff > right_diff:
                right = num_map[number]
                result.append("R")
            else:
                left = num_map[number]
                result.append("L")

    return "".join(result)


def solution_64061(board: list[list[int]], moves: list[int]) -> int:
    """💤 크레인 인형뽑기 게임 ; https://school.programmers.co.kr/learn/courses/30/lessons/64061
    Time complexity: O(n*m + moves)
        - n*m from `create stacks by line`
        - <moves> from `simulate`
    Space complexity: O(n*m)
        - maximum number of elements of <lines> and <stack> is n*m.

    Consideration
        💡 I must predict What is board's row and column from given Example. given description not mentions that.
        use Property of claw crane game; stack
    """
    import itertools
    from typing import Callable

    ## create stacks by line
    has_valid_element: Callable[[int], bool] = lambda x: x > 0
    lines: list[list[int]] = [
        list(itertools.takewhile(has_valid_element, reversed(column)))
        for column in zip(*board)
    ]

    # simulate
    stack: list[int] = []
    answer: int = 0
    for move in moves:
        move -= 1
        if lines[move]:
            x = lines[move].pop()
            if stack and stack[-1] == x:
                stack.pop()
                answer += 2
            else:
                stack.append(x)

    return answer


def solution_42889(N: int, stages: list[int]) -> list[int]:
    """💤 실패율 ; https://school.programmers.co.kr/learn/courses/30/lessons/42889
    Time complexity: O((N log N) + stages)
        - 2N from `stops = [0] * (N + 2)`, `create failure_rate`
        - (N log N) from `failure_rate.sort`
        - <stages> from `stops[stage] += 1`

    Space Complexity: O(N)
        - (N+2) from `stops = [0] * (N + 2)`
        - N from `failure_rate`

    Clues
        `만약 실패율이 같은 스테이지가 있다면 작은 번호의 스테이지가 먼저 오도록 하면 된다.`
            ; size of outputs is <n> and it should be sorted according to some criterion.
        `단, N + 1 은 마지막 스테이지(N 번째 스테이지) 까지 클리어 한 사용자를 나타낸다.`
            ; stops = [0] * (N + 2)    # instead of  `stops = [0] * (N + 1)`.

    Consideration
        The number of players reaching a stage has the property of 🚣 decreasing monotonic function as the stage increases.
    """
    # 0 index will not be used.
    stops = [0] * (N + 2)
    for stage in stages:
        stops[stage] += 1

    # create failure_rate
    total: int = len(stages)  # ; the number of people stopped at a stage; denominator
    failure_rate: list[tuple[int, float]] = []
    for stage in range(1, N + 1):
        if total != 0:
            failure_rate.append((stage, stops[stage] / total))
            total -= stops[stage]
        else:
            failure_rate.append((stage, 0))

    # make result
    # 💡 property of Stable sorting; failure_rate.sort() key does not have to be `key=lambda x: (-x[1], x[0])`   # (-failure_rate, stage).
    failure_rate.sort(key=lambda x: -x[1])
    return [stage for stage, _ in failure_rate]


def solution_42862(n: int, lost: list[int], reserve: list[int]) -> int:
    """💤 체육복 ; https://school.programmers.co.kr/learn/courses/30/lessons/42862

    Time complexity: O(n + lost + reserve)
        - 2*n, 1*lost, 1*reserve from loops.
    Space complexity: O(n)
        - n+1 from `create spares`

    Clues
        - `여벌 체육복을 가져온 학생이 체육복을 도난당했을 수 있습니다. 이때 이 학생은 체육복을 하나만 도난당했다고 가정하며, 남은 체육복이 하나이기에 다른 학생에게는 체육복을 빌려줄 수 없습니다.`
    Consideration
        - It seems that implemenation by using set() is valid, but it requires more operations.
    """
    # 0 index will not be used. so Note that index can be n
    ## create spares
    spares: list[int] = [0] * (n + 1)
    for num in lost:
        spares[num] -= 1
    for num in reserve:
        spares[num] += 1

    answer: int = n
    nn: int = n + 1
    for i in range(1, n + 1):
        if spares[i] == -1:
            if spares[(ii := i - 1)] == 1:
                spares[ii] -= 1
            elif (ii := i + 1) < nn and spares[ii] == 1:
                spares[ii] -= 1
            else:
                answer -= 1
    return answer


def solution_42840(answers: list[int]) -> list[int]:
    """모의고사 ; https://school.programmers.co.kr/learn/courses/30/lessons/42840
    Time complexity: O(answers)
        - 3*answer from `compare answers and guesses`
        - 3*log(3) from sort in `rank`, 2*3 from loop
    """
    import itertools

    one = itertools.cycle((1, 2, 3, 4, 5))
    two = itertools.cycle((2, 1, 2, 3, 2, 4, 2, 5))
    three = itertools.cycle((3, 3, 1, 1, 2, 2, 4, 4, 5, 5))

    # compare answers and guesses
    sums = [
        sum((1 if next(guesses) == answer else 0 for answer in answers))
        for guesses in (one, two, three)
    ]

    # rank
    s_sums = sorted(sums, reverse=True)
    rank = {}
    for i, score in enumerate(s_sums, start=1):
        if score not in rank:
            rank[score] = i

    return [i for i, score in enumerate(sums, start=1) if rank[score] == 1]


def solution_42748(array: list[int], commands: list[list[int]]) -> list[int]:
    return [sorted(array[i - 1 : j])[k - 1] for i, j, k in commands]


def solution_42576(participant: list[str], completion: list[str]) -> str:
    """완주하지 못한 선수 ; https://school.programmers.co.kr/learn/courses/30/lessons/42576"""
    from collections import Counter

    pc: Counter[str] = Counter(participant)
    for c in completion:
        pc[c] -= 1
    return pc.most_common(1)[0][0]


def solution_17682(dartResult: str) -> int:
    """💤 [1차] 다트 게임 ; https://school.programmers.co.kr/learn/courses/30/lessons/17682
    Clues
        - `옵션으로 스타상(*) , 아차상(#)이 존재하며 스타상(*) 당첨 시 해당 점수와 바로 전에 얻은 점수를 각 2배로 만든다. 아차상(#) 당첨 시 해당 점수는 마이너스된다.`
    """
    dart_len: int = len(dartResult)
    i: int = 0
    terms: list[int] = []
    while i < dart_len:
        number: int = 0
        exponent: int = 1
        symbol: int = 1
        multiplier: int = 1

        # number
        if dartResult[i : i + 2].isdigit():  # Note that not [i:i+1].
            number = int(dartResult[i : i + 2])
            i += 2
        else:
            number = int(dartResult[i])
            i += 1

        # bonus
        if dartResult[i] == "D":
            exponent = 2
        elif dartResult[i] == "T":
            exponent = 3
        i += 1

        # option
        if i < dart_len and not dartResult[i].isdigit():
            if dartResult[i] == "#":
                symbol = -1
            else:
                multiplier = 2
                if terms:
                    terms[-1] *= 2
            i += 1

        terms.append(number**exponent * multiplier * symbol)

    return sum(terms)


def solution_17681(n: int, arr1: list[int], arr2: list[int]) -> list[str]:
    """💦 [1차] 비밀지도 ; https://school.programmers.co.kr/learn/courses/30/lessons/17681"""
    return [
        "".join(("#" if x == "1" else " " for x in map_)).rjust(n, " ")
        for map_ in (format(row1 | row2, "b") for row1, row2 in zip(arr1, arr2))
    ]


def solution_12982(d: list[int], budget: int) -> int:
    """예산 ; https://school.programmers.co.kr/learn/courses/30/lessons/12982"""
    d.sort()
    total: int = 0
    i: int = 0
    for i, dd in enumerate(d):
        total += dd
        if total > budget:
            return i
    else:
        return i + 1


def solution_12977(nums: list[int]) -> int:
    """💤 소수 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12977
    Think whether the problem can be solved Dynamic programming or Brute force.
        given <nums> is not recursive and arithmetic sequence.
    """
    from itertools import combinations

    # sieve from 0 to 1000+999+998
    MAX_SUM = 2997
    sieve = [True] * (MAX_SUM + 1)
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if sieve[i]:
            for j in range(i**2, MAX_SUM + 1, i):
                sieve[j] = False
    return sum((1 if sieve[sum(comb)] else 0 for comb in combinations(nums, 3)))


def solution_12969() -> None:
    """💤 직사각형 별찍기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12969
    Same solution: print("\n".join(("*"*a for _ in range(b))))      ; but slower than current implementation.
    """
    a, b = map(int, input().split())
    print(("*" * a + "\n") * b)


def solution_12954(x: int, n: int) -> list[int]:
    """x만큼 간격이 있는 n개의 숫자 ; https://school.programmers.co.kr/learn/courses/30/lessons/12954"""
    return [x + x * i for i in range(n)]


def solution_12950(arr1: list[list[int]], arr2: list[list[int]]) -> list[list[int]]:
    """행렬의 덧셈 ; https://school.programmers.co.kr/learn/courses/30/lessons/12950
    - 😠 주어진 행렬의 차원이 몇인지 나오지 않는다."""
    return [[e1 + e2 for e1, e2 in zip(row1, row2)] for row1, row2 in zip(arr1, arr2)]


def solution_12948(phone_number: str) -> str:
    """💤 핸드폰 번호 가리기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12948
    - index"""
    hide_num_len = len(phone_number) - 4
    return "*" * hide_num_len + phone_number[hide_num_len:]


def solution_12947(x: int) -> bool:
    """하샤드 수 ; https://school.programmers.co.kr/learn/courses/30/lessons/12947"""
    return float(x / sum(map(int, str(x)))).is_integer()


def solution_12944(arr: list[int]) -> float:
    """평균 구하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12944"""
    return sum(arr) / len(arr)


def solution_12943(num: int) -> int:
    """💤 콜라츠 추측 ; https://school.programmers.co.kr/learn/courses/30/lessons/12943
    - 예외 처리
    """
    if num == 1:
        return 0
    for i in range(1, 501):
        num = num * 3 + 1 if num & 1 else num // 2
        if num == 1:
            return i
    else:
        return -1


def solution_12940(n: int, m: int) -> list[int]:
    """최대공약수와 최소공배수 ; https://school.programmers.co.kr/learn/courses/30/lessons/12940"""
    import math

    gcd = math.gcd(n, m)
    return [gcd, n * m // gcd]


def solution_12937(num: int) -> str:
    """짝수와 홀수 ; https://school.programmers.co.kr/learn/courses/30/lessons/12937"""
    return "Odd" if num & 1 else "Even"


def solution_12935(arr: list[int]) -> list[int]:
    """제일 작은 수 제거하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12935
    - 😠 작은 '수'를 제거. '수들' 이 아님.
    """
    arr.pop(min((i for i in range(len(arr))), key=lambda i: arr[i]))
    return arr if arr else [-1]


def solution_12934(n: int) -> int:
    """💦 정수 제곱근 판별 ; https://school.programmers.co.kr/learn/courses/30/lessons/12934"""
    return (x + 1) ** 2 if (x := float(n**0.5)).is_integer() else -1  # type: ignore


def solution_12933(n: int) -> int:
    """정수 내림차순으로 배치하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12933"""
    return int("".join((sorted(str(n), reverse=True))))


def solution_12932(n: int) -> list[int]:
    """자연수 뒤집어 배열로 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12932"""
    return [int(s) for s in reversed(str(n))]


def solution_12931(n: int) -> int:
    """자릿수 더하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12931"""
    return sum((int(s) for s in str(n)))


def solution_12930(s: str) -> str:
    """💤 이상한 문자 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12930
    Clues
        - `각 단어는 하나 이상의 공백문자로 구분되어 있습니다.`
    """
    answer: list[str] = []
    is_odd: int = 0
    for ss in s:
        if ss.isalpha():
            answer.append(ss.lower() if is_odd else ss.upper())
            is_odd ^= 1
        else:
            answer.append(ss)
            is_odd = 0
    return "".join(answer)


def solution_12928(n: int) -> int:
    """약수의 합 ; https://school.programmers.co.kr/learn/courses/30/lessons/12928"""
    divisors: list[int] = []
    for i in range(1, int(n**0.5) + 1):
        q, r = divmod(n, i)
        if r == 0:
            divisors.append(i)
            if q != i:
                divisors.append(q)
    return sum(divisors)


def solution_12926(s: str, n: int) -> str:
    """시저 암호 ; https://school.programmers.co.kr/learn/courses/30/lessons/12926"""
    import string

    answer: list[str] = []
    for ss in s:
        if ss.isupper():
            answer.append(string.ascii_uppercase[(ord(ss) - 65 + n) % 26])
        elif ss.islower():
            answer.append(string.ascii_lowercase[(ord(ss) - 97 + n) % 26])
        else:
            answer.append(ss)
    return "".join(answer)


def solution_12925(s: str) -> int:
    """문자열을 정수로 바꾸기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12925"""
    return int(s)


def solution_12922(n: int) -> str:
    """수박수박수박수박수박수? ; https://school.programmers.co.kr/learn/courses/30/lessons/12922"""
    return "수박" * (n // 2) + "수" * (n & 1)


def solution_12921(n: int) -> int:
    """🧠 소수 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12921
    Normal solution
        answer = 0
        for num in range(2, n+1):
            for i in range(2, int(num**0.5)+1):
                if num % i == 0:
                    break
            else:
                answer += 1
        return answer
    """
    sieve: list[bool] = [True] * (n + 1)
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i**2, n + 1, i):
                sieve[j] = False
    return sum(sieve)


def solution_12919(seoul: list[str]) -> str:
    """서울에서 김서방 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12919"""
    return f"김서방은 {seoul.index('Kim')}에 있다"


def solution_12918(s: str) -> int:
    """문자열 다루기 기본 ; https://school.programmers.co.kr/learn/courses/30/lessons/12918"""
    return s.isdigit() and len(s) in (4, 6)


def solution_12917(s: str) -> str:
    """💦 문자열 내림차순으로 배치하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12917

    Lambda is not be necessary because string comparison is based on ord().
    """
    return "".join(sorted(s, reverse=True))


def solution_12916(s: str) -> bool:
    """문자열 내 p와 y의 개수 ; https://school.programmers.co.kr/learn/courses/30/lessons/12916"""
    s = s.lower()
    return s.count("p") == s.count("y")


def solution_12915(strings: list[str], n: int) -> list[str]:
    """문자열 내 마음대로 정렬하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12915
    - 인덱스 1의 문자가 같은 문자열이 여럿 일 경우, 사전순으로 앞선 문자열이 앞쪽에 위치합니다.
    """
    return sorted(strings, key=lambda s: (s[n], s))


def solution_12912(a: int, b: int) -> int:
    """두 정수 사이의 합 ; https://school.programmers.co.kr/learn/courses/30/lessons/12912"""
    return sum((range(a, b + 1))) if b >= a else sum((range(b, a + 1)))


def solution_12910(arr: list[int], divisor: int) -> list[int]:
    """나누어 떨어지는 숫자 배열 ; https://school.programmers.co.kr/learn/courses/30/lessons/12910"""
    return answer if (answer := sorted((x for x in arr if x % divisor == 0))) else [-1]


def solution_12906(arr: list[int]) -> list[int]:
    """같은 숫자는 싫어 ; https://school.programmers.co.kr/learn/courses/30/lessons/12906
    Same solution:
        stack: list[int] = []
        for i, x in enumerate(arr):
            while stack and stack[-1] == x:
                stack.pop()
            stack.append(x)
        return stack
    """
    stack: list[int] = [arr[0]]
    stack.extend((arr[i] for i in range(1, len(arr)) if stack[-1] != arr[i]))
    return stack


def solution_12903(s: str) -> str:
    """💤 가운데 글자 가져오기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12903"""
    half = len(s) // 2
    return s[half] if len(s) & 1 else s[half - 1 : half + 1]


def solution_12901(a: int, b: int) -> str:
    """💦 2016년 ; https://school.programmers.co.kr/learn/courses/30/lessons/12901"""
    from datetime import datetime

    return ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][
        datetime(year=2016, month=a, day=b).weekday()
    ]


def solution(nums: list[int]) -> int:
    """💤 폰켓몬 ; https://school.programmers.co.kr/learn/courses/30/lessons/1845"""
    return min(len(set(nums)), len(nums) // 2)
