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
        targets.sort() # 🚣 to sort with all key is not required.
        i: int = 0
        targets_len: int = len(targets)
        while i < targets_len:
            maxx = targets[i][1]
            j = i + 1  # j is next target pointer.
            while j < targets_len:
                if targets[j][0] < maxx:
                    maxx = min(maxx, targets[j][1]) ## 🚣 it is not required.
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


def solution_181187(r1: int, r2: int) -> int:
    """🧠 두 원 사이의 정수 쌍 ; https://school.programmers.co.kr/learn/courses/30/lessons/181187
    partial sum?

    """
    import math

    rr2 = r2**2
    rr1 = r1**2
    answer: int = 0
    for x in range(1, r2 + 1):
        xx: int = x**2
        y_max = int((rr2 - xx) ** 0.5)
        y_min = 0 if x >= r1 else math.ceil((rr1 - xx) ** 0.5)
        answer += y_max - y_min + 1
    return answer * 4


def solution_178870() -> None:
    """연속된 부분 수열의 합 ; https://school.programmers.co.kr/learn/courses/30/lessons/178870
    sort+partial sum (two pointer)?
    """


def solution_176962() -> None:
    """과제 진행하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/176962"""


def solution_172927() -> None:
    """광물 캐기 ; https://school.programmers.co.kr/learn/courses/30/lessons/172927"""


def solution_169199() -> None:
    """리코쳇 로봇 ; https://school.programmers.co.kr/learn/courses/30/lessons/169199"""


def solution_169198() -> None:
    """당구 연습 ; https://school.programmers.co.kr/learn/courses/30/lessons/169198"""


def solution_160585() -> None:
    """혼자서 하는 틱택토 ; https://school.programmers.co.kr/learn/courses/30/lessons/160585"""


def solution_159993() -> None:
    """미로 탈출 ; https://school.programmers.co.kr/learn/courses/30/lessons/159993"""


def solution_155651() -> None:
    """호텔 대실 ; https://school.programmers.co.kr/learn/courses/30/lessons/155651"""


def solution_154540() -> None:
    """무인도 여행 ; https://school.programmers.co.kr/learn/courses/30/lessons/154540"""


def solution_154539() -> None:
    """뒤에 있는 큰 수 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/154539"""


def solution_154538() -> None:
    """숫자 변환하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/154538"""


def solution_152996() -> None:
    """시소 짝꿍 ; https://school.programmers.co.kr/learn/courses/30/lessons/152996"""


def solution_150369() -> None:
    """택배 배달과 수거하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/150369"""


def solution_150368() -> None:
    """이모티콘 할인행사 ; https://school.programmers.co.kr/learn/courses/30/lessons/150368"""


def solution_148653() -> None:
    """마법의 엘리베이터 ; https://school.programmers.co.kr/learn/courses/30/lessons/148653"""


def solution_148652() -> None:
    """유사 칸토어 비트열 ; https://school.programmers.co.kr/learn/courses/30/lessons/148652"""


def solution_147354(
    data: list[list[int]], col: int, row_begin: int, row_end: int
) -> int:
    """💤 테이블 해시 함수 ; https://school.programmers.co.kr/learn/courses/30/lessons/147354
    Tag: Math (Base conversion)"""
    if col > 1:
        data.sort(key=lambda x: (x[col - 1], -x[0]))
    else:
        data.sort(key=lambda x: x[0])

    result: int = 0
    for i in range(row_begin, row_end + 1):
        result ^= sum((x % i for x in data[i - 1]))
    return result


def solution_142085(n: int, k: int, enemy: list[int]) -> int:
    """💦 디펜스 게임 ; https://school.programmers.co.kr/learn/courses/30/lessons/142085
    - `남은 병사의 수보다 현재 라운드의 적의 수가 더 많으면 게임이 종료됩니다.`

    Other solution
        [Timeout] Dynamic programming solution
            dp[round][k]; when used number of free pass ticket is k in round <k>, maximum number of remained soldiers.
                dp[round][k] := max(dp[round-1][k] - enemey, dp[round-1][k-1])
            - if max(dp[round]) < 0, game end.
    """
    import heapq

    hq: list[int] = enemy[:k]
    heapq.heapify(hq)
    for stage in range(k, len(enemy)):
        n -= (
            heapq.heapreplace(hq, enemy[stage])
            if hq[0] < enemy[stage]
            else enemy[stage]
        )
        if n < 0:
            return stage  # offset value for one-based stage. so not +1.
    return len(enemy)


def solution_140107(k: int, d: int) -> int:
    """🧠 점 찍기 ; https://school.programmers.co.kr/learn/courses/30/lessons/140107
    Tag: Math (Partial sum)

    Other Solution
        # <x> and <maxy> are inclusive range.
        dd: int = d**2
        maxy: int = d - d % k
        answer: int = 0
        for x in range(0, d + 1, k):
            while x**2 + maxy**2 > dd:
                maxy -= k
            answer += maxy // k + 1
        return answer
    """
    dd: int = d**2
    return sum(int((dd - x**2) ** 0.5) // k + 1 for x in range(0, d + 1, k))


def solution_138476() -> None:
    """귤 고르기 ; https://school.programmers.co.kr/learn/courses/30/lessons/138476"""


def solution_135807() -> None:
    """숫자 카드 나누기 ; https://school.programmers.co.kr/learn/courses/30/lessons/135807"""


def solution_134239() -> None:
    """우박수열 정적분 ; https://school.programmers.co.kr/learn/courses/30/lessons/134239"""


def solution_132265() -> None:
    """롤케이크 자르기 ; https://school.programmers.co.kr/learn/courses/30/lessons/132265"""


def solution_131704() -> None:
    """택배상자 ; https://school.programmers.co.kr/learn/courses/30/lessons/131704"""


def solution_131701() -> None:
    """연속 부분 수열 합의 개수 ; https://school.programmers.co.kr/learn/courses/30/lessons/131701"""


def solution_131130() -> None:
    """혼자 놀기의 달인 ; https://school.programmers.co.kr/learn/courses/30/lessons/131130"""


def solution_131127() -> None:
    """할인 행사 ; https://school.programmers.co.kr/learn/courses/30/lessons/131127"""


def solution_118667() -> None:
    """두 큐 합 같게 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/118667"""


def solution_92342() -> None:
    """양궁대회 ; https://school.programmers.co.kr/learn/courses/30/lessons/92342"""


def solution_92341() -> None:
    """주차 요금 계산 ; https://school.programmers.co.kr/learn/courses/30/lessons/92341"""


def solution_92335() -> None:
    """k진수에서 소수 개수 구하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/92335"""


def solution_87946() -> None:
    """피로도 ; https://school.programmers.co.kr/learn/courses/30/lessons/87946"""


def solution_87390() -> None:
    """n^2 배열 자르기 ; https://school.programmers.co.kr/learn/courses/30/lessons/87390"""


def solution_87377() -> None:
    """교점에 별 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/87377"""


def solution_86971() -> None:
    """전력망을 둘로 나누기 ; https://school.programmers.co.kr/learn/courses/30/lessons/86971"""


def solution_86052() -> None:
    """빛의 경로 사이클 ; https://school.programmers.co.kr/learn/courses/30/lessons/86052"""


def solution_84512() -> None:
    """모음사전 ; https://school.programmers.co.kr/learn/courses/30/lessons/84512"""


def solution_81302() -> None:
    """거리두기 확인하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/81302"""


def solution_77885() -> None:
    """2개 이하로 다른 비트 ; https://school.programmers.co.kr/learn/courses/30/lessons/77885"""


def solution_77485() -> None:
    """행렬 테두리 회전하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/77485"""


def solution_76502() -> None:
    """괄호 회전하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/76502"""


def solution_72412() -> None:
    """순위 검색 ; https://school.programmers.co.kr/learn/courses/30/lessons/72412"""


def solution_72411() -> None:
    """메뉴 리뉴얼 ; https://school.programmers.co.kr/learn/courses/30/lessons/72411"""


def solution_70129() -> None:
    """이진 변환 반복하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/70129"""


def solution_68936() -> None:
    """쿼드압축 후 개수 세기 ; https://school.programmers.co.kr/learn/courses/30/lessons/68936"""


def solution_68645() -> None:
    """삼각 달팽이 ; https://school.programmers.co.kr/learn/courses/30/lessons/68645"""


def solution_67257() -> None:
    """수식 최대화 ; https://school.programmers.co.kr/learn/courses/30/lessons/67257"""


def solution_64065() -> None:
    """튜플 ; https://school.programmers.co.kr/learn/courses/30/lessons/64065"""


def solution_62048() -> None:
    """멀쩡한 사각형 ; https://school.programmers.co.kr/learn/courses/30/lessons/62048"""


def solution_60058() -> None:
    """괄호 변환 ; https://school.programmers.co.kr/learn/courses/30/lessons/60058"""


def solution_60057() -> None:
    """문자열 압축 ; https://school.programmers.co.kr/learn/courses/30/lessons/60057"""


def solution_49994() -> None:
    """방문 길이 ; https://school.programmers.co.kr/learn/courses/30/lessons/49994"""


def solution_49993() -> None:
    """스킬트리 ; https://school.programmers.co.kr/learn/courses/30/lessons/49993"""


def solution_43165() -> None:
    """타겟 넘버 ; https://school.programmers.co.kr/learn/courses/30/lessons/43165"""


def solution_42890() -> None:
    """후보키 ; https://school.programmers.co.kr/learn/courses/30/lessons/42890"""


def solution_42888() -> None:
    """오픈채팅방 ; https://school.programmers.co.kr/learn/courses/30/lessons/42888"""


def solution_42885() -> None:
    """구명보트 ; https://school.programmers.co.kr/learn/courses/30/lessons/42885"""


def solution_42883() -> None:
    """큰 수 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/42883"""


def solution_42860() -> None:
    """조이스틱 ; https://school.programmers.co.kr/learn/courses/30/lessons/42860"""


def solution_42842() -> None:
    """카펫 ; https://school.programmers.co.kr/learn/courses/30/lessons/42842"""


def solution_42839() -> None:
    """소수 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/42839"""


def solution_42747() -> None:
    """H-Index ; https://school.programmers.co.kr/learn/courses/30/lessons/42747"""


def solution_42746() -> None:
    """가장 큰 수 ; https://school.programmers.co.kr/learn/courses/30/lessons/42746"""


def solution_42626() -> None:
    """더 맵게 ; https://school.programmers.co.kr/learn/courses/30/lessons/42626"""


def solution_42587() -> None:
    """프로세스 ; https://school.programmers.co.kr/learn/courses/30/lessons/42587"""


def solution_42586() -> None:
    """기능개발 ; https://school.programmers.co.kr/learn/courses/30/lessons/42586"""


def solution_42584() -> None:
    """주식가격 ; https://school.programmers.co.kr/learn/courses/30/lessons/42584"""


def solution_42583() -> None:
    """다리를 지나는 트럭 ; https://school.programmers.co.kr/learn/courses/30/lessons/42583"""


def solution_42578() -> None:
    """의상 ; https://school.programmers.co.kr/learn/courses/30/lessons/42578"""


def solution_42577() -> None:
    """전화번호 목록 ; https://school.programmers.co.kr/learn/courses/30/lessons/42577
    //// 이거 전까지만 하고, 정리+필요한 이론 공부하고 다시 진행.
    """


def solution_17687() -> None:
    """[3차] n진수 게임 ; https://school.programmers.co.kr/learn/courses/30/lessons/17687"""


def solution_17686(files: list[str]) -> None:
    """[3차] 파일명 정렬 ; https://school.programmers.co.kr/learn/courses/30/lessons/17686
    Tag: Sequence sort (natural sort)
    """


def solution_17684(msg: str) -> list[int]:
    """[3차] 압축 ; https://school.programmers.co.kr/learn/courses/30/lessons/17684"""
    import itertools
    import string

    d: dict[str, int] = {c: i for i, c in enumerate(string.ascii_uppercase, start=1)}
    i: int = 0
    count_gen = itertools.count(len(string.ascii_uppercase) + 1)
    length = len(msg)
    answer: list[int] = []
    while i < length:
        x = msg[i]
        for i in range(i + 1, length):
            if (y := x + msg[i]) in d:
                x = y
            else:
                i -= 1
                d[y] = next(count_gen)
                break
        answer.append(d[x])
        i += 1
    return answer


def solution_17683(m: str, musicinfos: list[str]) -> str:
    """[3차] 방금그곡 ; https://school.programmers.co.kr/learn/courses/30/lessons/17683

    - Consideration
        - `네오가 기억한 멜로디와 악보에 사용되는 음은 C, C#, D, D#, E, F, F#, G, G#, A, A#, B 12개이다.`
        - `조건이 일치하는 음악이 여러 개일 때에는 라디오에서 재생된 시간이 제일 긴 음악 제목을 반환한다. 재생된 시간도 같을 경우 먼저 입력된 음악 제목을 반환한다.`
        - Example: 12:00 ~ 12:14 (exclusive)
    """
    from itertools import cycle

    for sharp, temp in [
        ("C#", "c"),
        ("D#", "d"),
        ("F#", "f"),
        ("G#", "g"),
        ("A#", "a"),
    ]:
        m = m.replace(sharp, temp)

    # comparisons: tuple[length, title]
    comparisons: list[tuple[int, str]] = []
    for info in musicinfos:
        time1, time2, title, melody = info.split(",")
        for sharp, temp in [
            ("C#", "c"),
            ("D#", "d"),
            ("F#", "f"),
            ("G#", "g"),
            ("A#", "a"),
        ]:
            melody = melody.replace(sharp, temp)

        delta = (int(time2[:2]) - int(time1[:2])) * 60 + int(time2[3:]) - int(time1[3:])
        x = cycle(melody)
        total_melody = "".join((next(x) for _ in range(delta)))
        if total_melody.find(m) >= 0:
            comparisons.append((len(total_melody), title))
    found_i = max(
        (i for i in range(len(comparisons))),
        key=lambda i: comparisons[i][0],
        default=-1,
    )
    return "(None)" if found_i == -1 else comparisons[found_i][1]


def solution_17680(cacheSize: int, cities: list[str]) -> int:
    """💦 [1차] 캐시 ; https://school.programmers.co.kr/learn/courses/30/lessons/17680
    Consideration
        - `cacheSize는 정수이며, 범위는 0 ≦ cacheSize ≦ 30 이다.`
    """
    from collections import OrderedDict

    if cacheSize == 0:
        return 5 * len(cities)

    cities = [city.lower() for city in cities]
    cache: OrderedDict[str, None] = OrderedDict()
    cost: int = 0
    count = i = 0
    for i, city in enumerate(cities):
        if city in cache:
            cost += 1
            cache.move_to_end(city)
        else:
            cost += 5
            cache[city] = None
            count += 1

        if count == cacheSize:
            break

    for i in range(i + 1, len(cities)):
        if cities[i] in cache:
            cost += 1
            cache.move_to_end(cities[i])
        else:
            cost += 5
            cache.popitem(last=False)
            cache[cities[i]] = None
    return cost


def solution_17679(m: int, n: int, board: list[str]) -> int:
    """🔍💤 [1차] 프렌즈4블록 ; https://school.programmers.co.kr/learn/courses/30/lessons/17679
    stack 사용해서
        영향을 미치는 stack[i] 에만 계산
        영향을 미치는 유효한 stack[i][start_j] ~ stack[i][end_j] 까지만 계산.
    11번 테스트케이스 실패.
        마지막 요소에 대한 iteration 문제인듯한데
    next_pop_set 이 한 번 더 실행되는듯? iterate 를 모두 하지 않았을 떄
    """
    stacks: list[list[str]] = [list(reversed(x)) for x in zip(*board)]
    is_popped = True
    nm1 = n - 1
    i_and_start_j = ((i, 0) for i in range(n))

    while is_popped:
        is_popped = False
        pop_set: set[int] = set()
        next_pop_set: set[int] = set()
        # <first_popped_j> by i. it is used to to bring not popped elements to the first popped index of the stack, and used to create next <i_and_start_j>.
        first_popped_j: list[int] = [m] * n
        i = 0
        for i, start_j in i_and_start_j:
            ## another branch
            if i == nm1:
                continue

            ## a branch. starts with Memoization
            ip1 = i + 1  # i plus 1
            im1 = i - 1

            # if stacks[i-1] is popped, <pop_set> will be inherited from <next_pop_set>.
            pop_set = next_pop_set if first_popped_j[im1] < m else set()
            next_pop_set = set()
            e_to_be_moved: list[str] = []

            # set <end_j>
            end_j = min(len(stacks[i]), len(stacks[ip1])) - 1
            is_first_popped = True
            for j in range(start_j, end_j):
                jp1: int = j + 1  # j plus 1
                if stacks[i][j] == stacks[ip1][j] == stacks[i][jp1] == stacks[ip1][jp1]:
                    x = [j, jp1]
                    pop_set.update(x)
                    next_pop_set.update(x)
                    if is_first_popped:
                        first_popped_j[ip1] = j
                        if first_popped_j[i] > j:
                            first_popped_j[i] = first_popped_j[im1] = j
                        is_first_popped = False

                # Statement to bring not popped elements to the first popped index of the stack.
                if j > first_popped_j[i] and j not in pop_set:
                    e_to_be_moved.append(stacks[i][j])
            else:
                # Statement to bring not popped elements to the first popped index of the stack.
                if end_j > first_popped_j[i]:
                    if end_j in pop_set:
                        e_to_be_moved.extend(stacks[i][end_j + 1 :])
                    else:
                        e_to_be_moved.extend(stacks[i][end_j:])

            # a <i> iteration ends.
            if pop_set:
                is_popped = True
                # move not popped elements to forward of stack if is_greater_than_first_pop_j
                stacks[i][first_popped_j[i] :] = e_to_be_moved
        else:
            if next_pop_set:
                last_i = nm1 if i == nm1 else i + 1
                first_popped_j[last_i] = first_popped_j[last_i - 1]
                stacks[last_i][first_popped_j[last_i] :] = [
                    stacks[last_i][j]
                    for j in range(first_popped_j[last_i] + 2, len(stacks[last_i]))
                    if j not in next_pop_set
                ]

        # fix when j is 0 when a <i> iteration ends.
        if first_popped_j[nm1 - 1] == m:
            first_popped_j[nm1] = m
        # create Generator <i_and_start_j>
        i_and_start_j = (
            (i, x - 1) if x > 1 else (i, 0)
            for i, x in enumerate(first_popped_j)
            if x < m
        )

    return n * m - sum((len(stacks[i]) for i in range(n)))


def solution_17677(str1: str, str2: str) -> int:
    """💦 [1차] 뉴스 클러스터링 ; https://school.programmers.co.kr/learn/courses/30/lessons/17677
    - 😠 what is return value when value of A union B is 0 (divisor is 0)?
    """
    from collections import Counter

    str1 = str1.lower()
    str2 = str2.lower()

    a = Counter((x for i in range(len(str1) - 1) if (x := str1[i : i + 2]).isalpha()))
    b = Counter((x for i in range(len(str2) - 1) if (x := str2[i : i + 2]).isalpha()))
    x = sum((a & b).values())
    y = sum((a | b).values())
    return 65536 if y == 0 else int(x / y * 65536)


def solution_12985(n: int, a: int, b: int) -> int:
    """🧠 예상 대진표 ; https://school.programmers.co.kr/learn/courses/30/lessons/12985
    Other solution
        import math
        if a > b:
            a, b = b, a

        for i in range(int(math.log2(n)), 0, -1):
            n >>= 1
            if b > n:
                if a <= n:
                    return i
                else:
                    b-=n
                    a-=n
        return 1

    Implementation
        upper implementation finds a branch in which <a>, <b> are divided as 2^k.
        This process is equivalent to operate XOR to Most significant bits of zero-based <a> and <b>; the result have <x> value.
        the answer is value (log_2 x); bit_length.
    """
    return ((a - 1) ^ (b - 1)).bit_length()


def solution_12981(n: int, words: list[str]) -> list[int]:
    """💤 영어 끝말잇기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12981
    Tag: Math (Base Conversion)

    Consideration
        - `한 글자인 단어는 인정되지 않습니다.`
            solved by `단어의 길이는 2 이상 50 이하입니다.`
        - `앞사람이 말한 단어의 마지막 문자로 시작하는 단어를 말해야 합니다.`
        - `이전에 등장했던 단어는 사용할 수 없습니다.`
    """
    words_set: set[str] = set([words[0]])
    for i in range(1, len(words)):
        if words[i][0] != words[i - 1][-1] or words[i] in words_set:
            q, r = divmod(i, n)
            return [r + 1, q + 1]
        else:
            words_set.add(words[i])
    return [0, 0]


def solution_12980(n: int) -> int:
    """🧠 점프와 순간 이동 ; https://school.programmers.co.kr/learn/courses/30/lessons/12980
    Clues
        - `(현재까지 온 거리) x 2 에 해당하는 위치로 순간이동`, `순간이동을 하면 건전지 사용량이 줄지 않지만, 앞으로 K 칸을 점프하면 K 만큼의 건전지 사용량이 듭니다.`
            ; It can be thought as n is sum of 2^{subset}, which the <subset> is a subset of set of strictly positive integer.
            It means that values not included in <subset> denotes "0" bit, otherwise "1" bits.

    Other solution
        answer: int = 0
        while n > 0:
            q, r = divmod(n, 2)
            answer += r
            n = q
        return answer
    """
    return n.bit_count()


def solution_12978(N: int, road: list[list[int]], K: int) -> int:
    """💦 배달 ; https://school.programmers.co.kr/learn/courses/30/lessons/12978
    Tag: Graph search

    Consideration
        - `1번 마을에 있는 음식점이 K 이하의 시간에 배달이 가능한 마을의 개수를 return 하면 됩니다.`
            - `(1 ≤ a, b ≤ N, a != b)`
            - `두 마을 a, b를 연결하는 도로는 여러 개가 있을 수 있습니다.`
            - `임의의 두 마을간에 항상 이동 가능한 경로가 존재합니다.`
    """
    import heapq

    MAX_DISTANCE: int = 500001
    # 0 index will not be used.
    graph_metric: list[dict[int, int]] = [{} for _ in range(N + 1)]
    for a, b, distance in road:
        if distance < graph_metric[a].get(b, MAX_DISTANCE):
            graph_metric[a][b] = distance
            graph_metric[b][a] = distance

    # hq: (total_distance, <a>); minimum distance from "1" village to <a> village.
    hq: list[tuple[int, int]] = [(0, 1)]
    # 0 index will not be used.
    traces: list[bool] = [False] * (N + 1)
    while hq:
        total_distance, a = heapq.heappop(hq)
        if traces[a]:
            continue
        traces[a] = True

        for b, distance in graph_metric[a].items():
            if not traces[b] and (new_total_distance := total_distance + distance) <= K:
                heapq.heappush(hq, (new_total_distance, b))
    return sum(traces)


def solution_12973(s: list[int]) -> int:
    """💤 짝지어 제거하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12973"""
    stack: list[int] = []
    for x in s:
        if stack and stack[-1] == x:
            stack.pop()
        else:
            stack.append(x)
    return 0 if stack else 1


def solution_12953(arr: list[int]) -> int:
    """N개의 최소공배수 ; https://school.programmers.co.kr/learn/courses/30/lessons/12953
    Tag: Math (Euclidean algorithm)
    - `math.gcd(*integers); Changed in version 3.9: Added support for an arbitrary number of arguments. Formerly, only two arguments were supported.`
    """
    import math

    lcm: int = arr[0]
    for i in range(1, len(arr)):
        lcm = lcm * arr[i] // math.gcd(lcm, arr[i])
    return lcm


def solution_12952(n: int) -> int:
    """🧠 N-Queen ; https://school.programmers.co.kr/learn/courses/30/lessons/12952"""
    from typing import Generator

    def n_queens(
        n: int, i: int, a: list[int], b: list[int], c: list[int]
    ) -> Generator[int, None, None]:
        if i < n:
            for j in range(n):
                if j not in a and i + j not in b and i - j not in c:
                    yield from n_queens(n, i + 1, a + [j], b + [i + j], c + [i - j])
        else:
            yield 1

    return sum(n_queens(n, 0, [], [], []))


def solution_12951(s: str) -> str:
    """💦 JadenCase 문자열 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12951"""
    return " ".join((word.capitalize() for word in s.split(" ")))


def solution_12949(arr1: list[list[int]], arr2: list[list[int]]) -> list[list[int]]:
    """💤 행렬의 곱셈 ; https://school.programmers.co.kr/learn/courses/30/lessons/12949"""
    import operator

    arr2_temp: list[tuple[int]] = list(zip(*arr2))  # 🚣 For cache locality
    return [
        [sum(map(operator.mul, row, column)) for column in arr2_temp] for row in arr1
    ]


def solution_12946(n: int) -> list[list[int]]:
    """🧠 하노이의 탑 ; https://school.programmers.co.kr/learn/courses/30/lessons/12946"""
    from collections import deque

    moves: int = 2**n - 1
    a, b, c = list(range(n, 0, -1)), [], []
    pegs: deque[list[int]] = deque([a, c, b]) if n & 1 else deque([a, b, c])
    peg_names: deque[int] = deque([1, 3, 2]) if n & 1 else deque([1, 2, 3])
    result: list[list[int]] = []
    for move in range(1, moves + 1):
        if move & 1:
            pegs[1].append(pegs[0].pop())
            result.append([peg_names[0], peg_names[1]])
        else:
            if pegs[0] and (not pegs[2] or pegs[0][-1] < pegs[2][-1]):
                src, dest = pegs[0], pegs[2]
                src_name, dest_name = peg_names[0], peg_names[2]
            else:
                src, dest = pegs[2], pegs[0]
                src_name, dest_name = peg_names[2], peg_names[0]

            dest.append(src.pop())
            result.append([src_name, dest_name])

            pegs.rotate(-1)
            peg_names.rotate(-1)

    return result


def solution_12945(n: int) -> int:
    """피보나치 수 ; https://school.programmers.co.kr/learn/courses/30/lessons/12945
    Tag: Dynamic programming"""
    dp: list[int] = [0, 1]
    for i in range(2, n + 1):
        dp[i & 1] = dp[0] + dp[1]
    return dp[n & 1]


def solution_12941(A: list[int], B: list[int]) -> int:
    """🧠 최솟값 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12941
    Proof (Induction)
        Variables
            A = [1, 2]; set(natural number) [a1, a2] (sorted in ascendant order)
            B = [4, 5]; set(natural number) [b1, b2] (sorted in ascendant order)
        Axioms
            1. a1 <= a2
            2. b1 <= b2
        Process
            If we multiply to each side of a1 <= a2 by (b2-b1), we get:
                a1*(b2-b1) <= a2*(b2-b1) (Equation 1)
                ; a1*b2 - a1*b1 <= a2*b2 - a2*b1
            and if we add to each side of Equation 1 by (a1*b1 + a2*b1), we get:
                ; a1*b2 + a2*b1 <= a1*b1 + a2*b2
        Result
            a1*b2 + a2*b1 <= a1*b1 + a2*b2
    """
    A.sort()
    B.sort(reverse=True)
    return sum((a * b for a, b in zip(A, B)))


def solution_12939(s: str) -> str:
    """최댓값과 최솟값 ; https://school.programmers.co.kr/learn/courses/30/lessons/12939"""
    import sys

    min_val: int = sys.maxsize
    max_val: int = -sys.maxsize
    for x in s.split():
        xx: int = int(x)
        if xx < min_val:
            min_val = xx
        if xx > max_val:
            max_val = xx
    return f"{min_val} {max_val}"


def solution_12936(n: int, k: int) -> list[int]:
    """🧠 줄 서는 방법 ; https://school.programmers.co.kr/learn/courses/30/lessons/12936
    Tag: Math (Base conversion; Top-down flow)

    Other solution
        1. itertools.islice() solution
            is_not_used = [True]*(n+1)
            result: list[int] = []
            k -= 1
            unit = math.factorial(n) # order_unit
            for i in range(n, 1, -1):
                unit //= i
                multiplier, r = divmod(k, unit)
                found_num = next(islice((i for i, x in enumerate(is_not_used) if x), multiplier+1, None))
                is_not_used[found_num] = False
                result.append(found_num)
                k = r
            else:
                result.append(next(islice((i for i, x in enumerate(is_not_used) if x), 1, None)))
            return result
        2. <TimeOut>
            - itertools.permutations
            - Efficiency test case 2
                for i in range(n-1, 0, -1):
                    q, k = divmod(k, unit)
                    result.append(nums.pop(q))
                    unit //= i
                ➡️
                for i in range(n, 1, -1):
                    unit //= i
                    q, k = divmod(k, unit)
                    result.append(nums.pop(q))
        3. 🔍 deque rotate, pop solution
    """
    import itertools
    import operator

    result: list[int] = []
    nums: list[int] = list(range(1, n + 1))
    k -= 1
    units = list(itertools.accumulate(range(1, n), operator.mul))  # order_unit
    for unit in reversed(units):
        q, k = divmod(k, unit)
        result.append(nums.pop(q))
    else:
        result.append(nums[0])
    return result


def solution_12924(n: int) -> int:
    """💤 숫자의 표현 ; https://school.programmers.co.kr/learn/courses/30/lessons/12924
    Tag
        Math (Partition problem)
    Clues
        - `연속한 자연수들로 표현 하는 방법`; Partial sum with pointers

    Time complexity: O(n); O(2n)
    Space complexity: O(1)

    Debugging
        1 + 2 + 3 + 4 + 5 = 15
        4 + 5 + 6 = 15
        7 + 8 = 15
        15 = 15
    """
    i = j = 1  # i is inclusive, j is exclusive in a range.
    partial_sum = answer = 0
    for i in range(1, n + 1):
        while partial_sum < n:
            partial_sum += j
            j += 1
        if partial_sum == n:
            answer += 1
        partial_sum -= i
    return answer


def solution_12923(begin: int, end: int) -> list[int]:
    """💤 숫자 블록 ; https://school.programmers.co.kr/learn/courses/30/lessons/12923
    Tag: Math

    Consideration
        - `그렙시는 길이가 1,000,000,000인 도로에 1부터 10,000,000까지의 숫자가 적힌 블록들을 이용해 위의 규칙대로 모두 설치 했습니다.`
        - `1 ≤ begin ≤ end ≤ 1,000,000,000`
    """
    result: list[int] = []
    MAX_BLOCK_NUM_RANGE: float = 10e6
    for num in range(begin, end + 1):  # condition: end - begin ≤ 5,000
        last_divisor: int = 1
        for i in range(2, int(num**0.5) + 1):
            q, r = divmod(num, i)
            if r == 0:
                if q <= MAX_BLOCK_NUM_RANGE:
                    result.append(q)
                    break
                else:
                    last_divisor = i
        else:
            result.append(last_divisor)

    if begin == 1:  # edge case
        result[0] = 0

    return result


def solution_12914(n: int) -> int:
    """멀리 뛰기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12914
    - It is similiar problem with 2*n domino tiling.

    Time complexity: O(n)
    Space complexity: O(1)
        from Sliding Window approach

    Debugging
        dp[0] = 1.
        dp[1] = 1
        dp[2] = 2
        dp[3] = 1+2
    """
    dp = [2, 1]
    for i in range(3, n + 1):
        dp[i & 1] = (dp[0] + dp[1]) % 1234567
    return dp[n & 1]


def solution_12913(land: list[list[int]]) -> int:
    """💤 땅따먹기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12913
    Tag: Dynamic programming

    Time complexity: O(n); the number of lines
    Space complexity: O(1)
        from Sliding Window approach
    """
    # max_score when player stepped on column (0, 1, 2, 3) in a line.
    max_score: list[int] = [0] * 4
    for line in land:
        max_score = [
            line[i] + max((x for j, x in enumerate(max_score) if j != i))
            for i in (0, 1, 2, 3)
        ]
    return max(max_score)


def solution_12911(n: int) -> int:
    """💤 다음 큰 숫자 ; https://school.programmers.co.kr/learn/courses/30/lessons/12911
    Time complexity: O(log n)
    Space complexity: O(log n)

    Implementation
        - p1; index of first found "1" from end index to zero index.
        - p2; index of first found "0" from <p1>-1 to zero index
        - <result>
            - `bin_repr[: 0 if p2 <= 0 else p2]`; previous substring before p2.
            - `"1"`; location of increased "1"
            - `"0" * (len(bin_repr) - p1)`; fill "0"s for next appeared right justified "1"s.
            - `"1" * (p1 - p2 - 1)`; right justify "1"s between p1, p2 (inclusive) except for previously added "1".
    Debugging
        -----
        1001100
          * *   ; find p1, p2.
        1010100 ; increase "1" bit at p2+1 index.
        1010001 ; rjustify right bits of p2.

        -----
        1111111 (Edge case)
        *
        01111111
        10111111

    """
    # condition: 1 ≤ n ≤ 1,000,000
    bin_repr: str = format(n, "b")
    p1 = bin_repr.rindex("1")
    p2 = bin_repr.rfind("0", 0, p1)
    return int(
        "".join(
            (
                bin_repr[: 0 if p2 <= 0 else p2],
                "1",
                "0" * (len(bin_repr) - p1),
                "1" * (p1 - p2 - 1),
            )
        ),
        2,
    )


def solution_12909(s: str) -> bool:
    """올바른 괄호 ; https://school.programmers.co.kr/learn/courses/30/lessons/12909"""
    offset: int = 0
    for parenthesis in s:
        if (offset := offset + 1 if parenthesis == "(" else offset - 1) < 0:
            return False
    return offset == 0


def solution_12905(board: list[list[int]]) -> int:
    """🧠 가장 큰 정사각형 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12905
    Tag: Dynamic programming

    In-place implementation
        - board[i][j] := side length to create largest square by using from board[0][0] to board[i][j].
            It uses mechanism that four partially overlapping 2*2 squares are required in order to compose 3*3 square, and so on.

    Debugging
        from
        0  1  1  0  0
        1  1  1  1  1
        1  1  1  1  1
        0  0  1  1  1

        to
        0  1  1  0  0
        1  1  2  1  1
        1  2  2  2  2
        0  0  1  2  3
    """
    n, m = len(board), len(board[0])

    if n == 1 or m == 1:
        from itertools import chain

        return int(any((True for x in chain.from_iterable(board) if x == 1)))

    side_len: int = 0
    range_m = range(1, m)
    for i in range(1, n):
        for j in range_m:
            if board[i][j] == 1:
                board[i][j] = (
                    min(board[i - 1][j - 1], board[i - 1][j], board[i][j - 1]) + 1
                )
                if board[i][j] > side_len:
                    side_len = board[i][j]
    return side_len**2


def solution_12902(n: int) -> int:
    """🧠🔍 3 x n 타일링 ; https://school.programmers.co.kr/learn/courses/30/lessons/12902
    Tag: Dynamic programming
        - Domino tiling

    Time Complexity: O(n)
    Space complexity: O(1)
        from Sliding Window approach

    Recurrence Relation
        if <n> is odd, dp[n] == 0,
        if <n> is even, dp[n] = dp[n-2]*3 + 2 + (dp[n-4]*2 + dp[n-6]*2 ... dp[2]*2)
            dp[n-2]*3 ; (previous cases)*dp[2]
            +2 ; the new shapes that didn't exist before.
            (dp[n-4]*2 + dp[n-6]*2 ... dp[2]*2) ; permutations with expanded area and the previous new shapes that didn't exist before.

    Other solution
        if n & 1:
            return 0
        dp, new_perm_count = 3, 0
        for _ in range(4, n + 1, 2):
            previous_dp = dp
            dp = dp * 3 + new_perm_count + 2
            new_perm_count = new_perm_count + previous_dp * 2
        return dp % 1000000007

    Current implemenation (🔍 proof); https://oeis.org/A001835
    """
    if n & 1:
        return 0
    term = preceding_term = 1
    for _ in range(0, n, 2):  # condition (1 ≤ nn ≤ 5,000)
        term, preceding_term = (4 * term - preceding_term) % 1000000007, term
    return term


def solution_12900(n: int) -> int:
    """2 * n 타일링 ; https://school.programmers.co.kr/learn/courses/30/lessons/12900
    Tag: Dynamic programming
        - Domino tiling

    Time Complexity: O(n)
    Space complexity: O(1)
        from Sliding Window approach

    Recurrence Relation
        f(n) = f(n-1) + f(n-2); number of cases that tiling 2*n rectangle with 2*1 or 1*2 tiles.
            - `f(n-1)`; number of previous cases adding one 2*1 tile.
            - `f(n-2)`; number of previous cases adding two 1*2 tile.
    """
    dp: list[int] = [2, 1]
    for i in range(3, n + 1):
        dp[i & 1] = (dp[0] + dp[1]) % 1000000007
    return dp[n & 1]


def solution_12899(n: int) -> str:
    """124 나라의 숫자 ; https://school.programmers.co.kr/learn/courses/30/lessons/12899
    Tag: Math (Base Conversion)

    - 😠 int 를 반환하는 것이 아니라 str 을 반환하는 것이다.
    Other Implementation
        1. <TimeOut>: itertools.product
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
