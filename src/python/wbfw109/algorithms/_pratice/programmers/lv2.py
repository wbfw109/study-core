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


def solution_181187() -> None:
    """두 원 사이의 정수 쌍 ; https://school.programmers.co.kr/learn/courses/30/lessons/181187"""


def solution_178870() -> None:
    """연속된 부분 수열의 합 ; https://school.programmers.co.kr/learn/courses/30/lessons/178870"""


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


def solution_147354() -> None:
    """테이블 해시 함수 ; https://school.programmers.co.kr/learn/courses/30/lessons/147354"""


def solution_142085() -> None:
    """디펜스 게임 ; https://school.programmers.co.kr/learn/courses/30/lessons/142085"""


def solution_140107() -> None:
    """점 찍기 ; https://school.programmers.co.kr/learn/courses/30/lessons/140107"""


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
    """전화번호 목록 ; https://school.programmers.co.kr/learn/courses/30/lessons/42577"""


def solution_17687() -> None:
    """[3차] n진수 게임 ; https://school.programmers.co.kr/learn/courses/30/lessons/17687"""


def solution_17686() -> None:
    """[3차] 파일명 정렬 ; https://school.programmers.co.kr/learn/courses/30/lessons/17686"""


def solution_17684() -> None:
    """[3차] 압축 ; https://school.programmers.co.kr/learn/courses/30/lessons/17684"""


def solution_17683() -> None:
    """[3차] 방금그곡 ; https://school.programmers.co.kr/learn/courses/30/lessons/17683"""


def solution_17680() -> None:
    """[1차] 캐시 ; https://school.programmers.co.kr/learn/courses/30/lessons/17680"""


def solution_17679() -> None:
    """[1차] 프렌즈4블록 ; https://school.programmers.co.kr/learn/courses/30/lessons/17679"""


def solution_17677() -> None:
    """[1차] 뉴스 클러스터링 ; https://school.programmers.co.kr/learn/courses/30/lessons/17677"""


def solution_12985() -> None:
    """예상 대진표 ; https://school.programmers.co.kr/learn/courses/30/lessons/12985"""


def solution_12981() -> None:
    """영어 끝말잇기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12981"""


def solution_12980() -> None:
    """점프와 순간 이동 ; https://school.programmers.co.kr/learn/courses/30/lessons/12980"""


def solution_12978() -> None:
    """배달 ; https://school.programmers.co.kr/learn/courses/30/lessons/12978"""


def solution_12973() -> None:
    """짝지어 제거하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12973"""


def solution_12953() -> None:
    """N개의 최소공배수 ; https://school.programmers.co.kr/learn/courses/30/lessons/12953"""


def solution_12952() -> None:
    """N-Queen ; https://school.programmers.co.kr/learn/courses/30/lessons/12952"""


def solution_12951() -> None:
    """JadenCase 문자열 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12951"""


def solution_12949() -> None:
    """행렬의 곱셈 ; https://school.programmers.co.kr/learn/courses/30/lessons/12949"""


def solution_12946() -> None:
    """하노이의 탑 ; https://school.programmers.co.kr/learn/courses/30/lessons/12946"""


def solution_12945() -> None:
    """피보나치 수 ; https://school.programmers.co.kr/learn/courses/30/lessons/12945"""


def solution_12941() -> None:
    """최솟값 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12941"""


def solution_12939() -> None:
    """최댓값과 최솟값 ; https://school.programmers.co.kr/learn/courses/30/lessons/12939"""


def solution_12936() -> None:
    """줄 서는 방법 ; https://school.programmers.co.kr/learn/courses/30/lessons/12936"""


def solution_12924() -> None:
    """숫자의 표현 ; https://school.programmers.co.kr/learn/courses/30/lessons/12924"""


def solution_12923() -> None:
    """숫자 블록 ; https://school.programmers.co.kr/learn/courses/30/lessons/12923"""


def solution_12914() -> None:
    """멀리 뛰기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12914"""


def solution_12913() -> None:
    """땅따먹기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12913"""


def solution_12911(n: int) -> int:
    """다음 큰 숫자 ; https://school.programmers.co.kr/learn/courses/30/lessons/12911"""
    bin_repr: list[str] = ["0", *list(format(n, "b"))]


def solution_12909(s: str) -> bool:
    """올바른 괄호 ; https://school.programmers.co.kr/learn/courses/30/lessons/12909"""
    offset: int = 0
    for parenthesis in s:
        if (offset := offset + 1 if parenthesis == "(" else offset - 1) < 0:
            return False
    return offset == 0


# TODO: description
def solution_12905(board: list[list[int]]) -> int:
    """🧠 가장 큰 정사각형 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/12905
    Tag: Dynamic programming

    Recurrence Relation
        board[i][j] := side length to create largest square by using from board[0][0] to board[i][j].

    Debugging
        0  1  1  0  0
        1  1  1  1  1
        1  1  1  1  1
        0  0  1  1  1
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
