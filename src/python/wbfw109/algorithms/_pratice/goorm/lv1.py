"""Purpose is to solve in 5 minutes
- 💤: 문제 이해 제대로 하기 (구현 과정 문제)
    "문제 설명"에 모든 제한사항이 함게 주어지지 않는 경우가 있어서, 완벽히 이해하기 어려울 수 있다.
    에매한 경우, 명확하게 이해할 수 있도록 "예시" 가 주어지므로 꼭 함께 보도록 한다.
- 🧠: (신박한 풀이, 수학식을 활용한 풀이)
- 💦: built-in functions, grammar
- 😠: 문제 설명에 모순이 있는 것
- 🌔: Python 이 아닌 다른 언어로만 풀 수있는 것.
"""


def solution_167844() -> None:
    """배열 ; https://level.goorm.io/exam/167844/%EA%B0%9C%EB%85%90-%EB%B0%B0%EC%97%B4/quiz/1"""
    result: list[str] = []
    a = list(range(3, 0, -1))

    result.append(str(a))
    a.sort()
    result.append(str(a))
    result.append(str(a[-1]))
    result.append(str(list("34512").index("2")))
    print("\n".join(result))


def solution_167846() -> None:
    """배열 (숫자 야구 게임) ; https://level.goorm.io/exam/167846/%EB%AC%B8%EC%A0%9C-%EB%B0%B0%EC%97%B4/quiz/1"""
    answer: list[int] = [3, 2, 8, 1]
    output: list[str] = []
    for _ in range(5):
        user_input = list(map(int, input().split()))
        result: list[str] = []
        for j in range(4):
            try:
                loc: int = answer.index(user_input[j])
            except ValueError:
                result.append("Fail")
            else:
                result.append(("Strike" if loc == j else "Ball"))
        output.append(str(result))
    print("\n".join(output))


def solution_175880() -> None:
    """💦 큰 팩토리얼 ; https://level.goorm.io/exam/175880/%ED%81%B0-%ED%8C%A9%ED%86%A0%EB%A6%AC%EC%96%BC/quiz/1"""
    import math

    print(math.factorial(int(input())) % 1000000007)


def solution_174702() -> None:
    """A + B ; https://level.goorm.io/exam/174702/a-b/quiz/1"""
    print(sum(map(int, input().split())))


def solution_174704() -> None:
    """💦 A + B (2) ; https://level.goorm.io/exam/174704/a-b-2/quiz/1"""
    print(format(sum(map(float, input().split())), ".6f"))


def solution_174805() -> None:
    """💤 숫자 제거 배열 ; https://level.goorm.io/exam/174805/%EC%88%AB%EC%9E%90-%EC%A0%9C%EA%B1%B0-%EB%B0%B0/quiz/1"""
    _, k = input().split()
    print(sum((0 if k in x else 1 for x in input().split())))


def solution_174732() -> None:
    """💦 대소문자 바꾸기 https://level.goorm.io/exam/174732/%EB%8C%80%EC%86%8C%EB%AC%B8%EC%9E%90-%EB%B0%94%EA%BE%B8%EA%B8%B0/quiz/1"""
    input()
    print(input().swapcase())


def solution_173089() -> None:
    """정수의 길이 ; https://level.goorm.io/exam/173089/%EC%A0%95%EC%88%98%EC%9D%98-%EA%B8%B8%EC%9D%B4/quiz/1"""
    print(len(str(input())))


def solution_171280() -> None:
    """💤 성적표 ; https://level.goorm.io/exam/171280/%EC%84%B1%EC%A0%81%ED%91%9C/quiz/1"""
    import sys

    input_ = sys.stdin.readline
    n, m = map(int, input_().split())
    sums: list[int] = [0] * (m + 1)
    counts: list[int] = [0] * (m + 1)
    for c_i, s_i in (map(int, input_().split()) for _ in range(n)):
        sums[c_i] += s_i
        counts[c_i] += 1

    best_i: int = max(
        (i for i in range(m + 1) if counts[i] > 0), key=lambda i: sums[i] / counts[i]
    )
    print(best_i)


def solution_49088() -> None:
    """💤 의좋은 형제 ; https://level.goorm.io/exam/49088/%EC%9D%98%EC%A2%8B%EC%9D%80-%ED%98%95%EC%A0%9C/quiz/1
    Tag: Masking, Bisection method

    - "통째로 넘겨준다.." 의 의미가 자신의 모든 식량을 모두 준다는 의미가 아님. 예시를 함께 보자.
    """
    sums: list[int] = list(map(int, input().split()))
    ap: int = 0
    bp: int = 1
    for _ in range(int(input())):
        t: int = sums[ap] // 2
        if sums[ap] & 1:
            t += 1
        sums[bp] += t
        sums[ap] -= t
        ap ^= 1
        bp ^= 1

    print(sums[0], sums[1])


def solution_167341() -> None:
    """💦 별찍기 ; https://level.goorm.io/exam/167341/%EB%B3%84%EC%B0%8D%EA%B8%B0/quiz/1"""
    n = int(input())
    print("\n".join((("*" * i).rjust(n) for i in range(1, n + 1))))


def solution_43059() -> None:
    """파도 센서 ; https://level.goorm.io/exam/43059/%ED%8C%8C%EB%8F%84-%EC%84%BC%EC%84%9C/quiz/1"""
    x, y, r = map(int, input().split())
    sensors: list[list[int]] = [list(map(int, input().split())) for _ in range(5)]
    answer: int = -1
    comp: list[float] = [((x - sx) ** 2 + (y - sy) ** 2) ** 0.5 for sx, sy in sensors]

    min_i: int = min((i for i in range(len(comp))), key=lambda i: comp[i])
    if comp[min_i] <= r:
        answer = min_i + 1

    print(answer)


def solution_159664() -> None:
    """합격자 찾기 ; https://level.goorm.io/exam/159664/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EB%A8%BC%EB%8D%B0%EC%9D%B4-%ED%95%A9%EA%B2%A9%EC%9E%90-%EC%B0%BE%EA%B8%B0/quiz/1"""
    import sys

    input_ = sys.stdin.readline
    result: list[str] = []
    for _ in range(int(input_())):
        n: int = int(input_())
        scores: list[int] = list(map(int, input_().split()))
        aver: float = sum(scores) / n

        passers_count: int = sum((1 if x >= aver else 0 for x in scores))

        result.append(f"{passers_count}/{n}")

    print("\n".join(result))


def solution_167336() -> None:
    """💤 최장 맨해튼 거리 ; https://level.goorm.io/exam/167336/%EC%B1%8C%EB%A6%B0%EC%A7%80-%EC%B5%9C%EC%9E%A5-%EB%A7%A8%ED%95%B4%ED%8A%BC-%EA%B1%B0%EB%A6%AC/quiz/1
    동일 문제: ID 159183: [알고리즘먼데이] 최장 맨해튼 거리
    """
    import itertools

    print(
        max(
            (
                abs(a - b) + abs(c - d)
                for a, b, c, d in itertools.permutations(map(int, input().split()), 4)
            )
        )
    )


def solution_171192() -> None:
    """💤 절약 ; https://level.goorm.io/exam/171192/%EC%A0%88%EC%95%BD/quiz/1"""
    import sys

    input_ = sys.stdin.readline
    answer: str = "success"
    balance: int = 0
    for _ in range(int(input_())):
        command, money = input_().split()
        money = int(money)
        if command[0] == "i":
            balance += money
        else:
            if balance < money:
                answer = "fail"
                break
            else:
                balance -= money
    print(answer)


def solution_51358() -> None:
    """🧠 재원 넘버 ; https://level.goorm.io/exam/51358/%EC%9E%AC%EC%9B%90-%EB%84%98%EB%B2%84/quiz/1
    Tag: Geometric sequence

    sum( len( itertools.product([2,3,6], repeat=1 to n) ) )

    A := total = 3*(1+3+3^2+... 3^(n-1))
    B := 3*total = 3*(3+... 3^n)
    B - A = 2*total = 3*(3^n-1)
    total = 3*(3^n-1) / 2
    """
    print(3 * (3 ** int(input()) - 1) // 2)


def solution_171953() -> None:
    """대문자 만들기 ; https://level.goorm.io/exam/171953/%EB%8C%80%EB%AC%B8%EC%9E%90-%EB%A7%8C%EB%93%A4%EA%B8%B0/quiz/1"""
    input()
    print(input().upper())


def solution_150332() -> None:
    """구름 크기 측정하기 ; https://level.goorm.io/exam/150332/%ED%98%84%EB%8C%80%EB%AA%A8%EB%B9%84%EC%8A%A4-%EB%AA%A8%EC%9D%98%ED%85%8C%EC%8A%A4%ED%8A%B8-%EA%B5%AC%EB%A6%84-%ED%81%AC%EA%B8%B0-%EC%B8%A1%EC%A0%95%ED%95%98%EA%B8%B0/quiz/1"""
    a, b = map(int, input().split())
    print(a * b)


def solution_49053() -> None:
    """💤💦 앵무새 꼬꼬 ; https://level.goorm.io/exam/49053/%EC%95%B5%EB%AC%B4%EC%83%88-%EA%BC%AC%EA%BC%AC/quiz/1

    - 주어진 입력에 대문자가 포함될 수 있는 것을 잘 캐치해야함.
    """
    import re

    pattern = re.compile(r"[aeiouAEIOU]")
    parrot_words: list[str] = []
    for _ in range(int(input())):
        parrot_word = "".join(re.findall(pattern, input()))
        if parrot_word == "":
            parrot_word = "???"
        parrot_words.append(parrot_word)

    print("\n".join(parrot_words))


def solution_51353() -> None:
    """💤 뱀이 지나간 자리 ; https://level.goorm.io/exam/51353/%EB%B1%80%EC%9D%B4-%EC%A7%80%EB%82%98%EA%B0%84-%EC%9E%90%EB%A6%AC/quiz/1
    Tag: Masking
    """
    n, m = map(int, input().split())
    pm: int = m - 1
    map_: list[str] = []
    p = 0
    available_lines = ["." * pm + "#", "#" + "." * pm, "#" * m]
    for i in range(1, n + 1):
        if i & 1:
            map_.append(available_lines[2])
        else:
            map_.append(available_lines[p])
            p ^= 1
    print("\n".join(map_))


def solution_49086() -> None:
    """정사각형의 개수 ; https://level.goorm.io/exam/49086/%EC%A0%95%EC%82%AC%EA%B0%81%ED%98%95%EC%9D%98-%EA%B0%9C%EC%88%98/quiz/1"""
    print(sum((i**2 for i in range(1, int(input()) + 1))))


def solution_49095() -> None:
    """💤 고장난 컴퓨터 ; https://level.goorm.io/exam/49095/%EA%B3%A0%EC%9E%A5%EB%82%9C-%EC%BB%B4%ED%93%A8%ED%84%B0/quiz/1

    Purpose to calculate: interval of pairs of typings
    Purpose to answer: remained the number of typings
    """
    n, c = map(int, input().split())
    timings: list[int] = list(map(int, input().split()))
    is_all_complete: bool = True
    i: int = n
    for i in range(n - 1, 0, -1):
        if timings[i] - timings[i - 1] > c:
            is_all_complete = False
            break
    print(n - i + is_all_complete)


def solution_49087() -> None:
    """🧠 여름의 대삼각형 ; https://level.goorm.io/exam/49087/%EC%97%AC%EB%A6%84%EC%9D%98-%EB%8C%80%EC%82%BC%EA%B0%81%ED%98%95/quiz/1
    Ways to check if three points are collinear
    1. Slope Method
    2. Area Method (Determinant of Homogeneous coordinates with 1)
        Area = 0.5 * |(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))|
        if Area is zero, three points are collinear.

    - 😠 만약 삼각형이 만들어지지 않는다면 0을 출력하라고 되어있는데 실제로는 0.00까지 출력해야 한다.
        "출력" 란의 설명과 "예제"란의 출력이 다르다.
    """
    x, y = zip(*(tuple(map(int, input().split())) for _ in range(3)))
    area: float = 0.5 * abs(
        x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])  # type: ignore
    )
    print(format(round(area, 3), ".2f"))


def solution_170832() -> None:
    """다이어트 ; https://level.goorm.io/exam/170832/%EB%8B%A4%EC%9D%B4%EC%96%B4%ED%8A%B8/quiz/1"""
    import sys

    input_ = sys.stdin.readline
    w, n = map(int, input_().split())
    for _ in range(n):
        calories, steps = map(int, input_().split())
        if calories < steps:
            if w > 10:
                w -= 1
        elif calories > steps:
            if w < 80:
                w += 1

    print(w)


def solution_171306() -> None:
    """💦 거꾸로 수 비교 ; https://level.goorm.io/exam/171306/%EA%B1%B0%EA%BE%B8%EB%A1%9C-%EC%88%98-%EB%B9%84%EA%B5%90/quiz/1

    - int 로 변환하지 않고 값을 문자열 상태로 비교하는 방법은 입력이 모두 부호 포함 같은 길이일 때만 가능하다.
    - str.join() 은 iterable 로 받을 수 있다.
    """
    a, b = input().split()

    print((a if int("".join(reversed(a))) > int("".join(reversed(b))) else b))


def solution_170830() -> None:
    """부등호 표시하기 ; https://level.goorm.io/exam/170830/%EB%B6%80%EB%93%B1%ED%98%B8-%ED%91%9C%EC%8B%9C%ED%95%98%EA%B8%B0/quiz/1"""
    abc = tuple(map(int, input().split()))
    result: list[str] = []
    for i in range(len(abc) - 1):
        result.append(str(abc[i]))
        if abc[i] < abc[i + 1]:
            result.append("<")
        elif abc[i] == abc[i + 1]:
            result.append("==")
        else:
            result.append(">")
    else:
        result.append(str(abc[-1]))

    print("".join(result))


def solution_170788() -> None:
    """두 정수 더하기 ; https://level.goorm.io/exam/170788/%EB%91%90-%EC%A0%95%EC%88%98-%EB%8D%94%ED%95%98%EA%B8%B0/quiz/1"""
    print(sum(map(int, input().split())))


def solution_170813() -> None:
    """두 실수 더하기 ; https://level.goorm.io/exam/170813/%EB%91%90-%EC%8B%A4%EC%88%98-%EB%8D%94%ED%95%98%EA%B8%B0/quiz/1"""
    print(format(sum(map(float, input().split())), ".6f"))


def solution_171918() -> None:
    """배수 삭제 ; https://level.goorm.io/exam/171918/%EB%B0%B0%EC%88%98-%EC%82%AD%EC%A0%9C/quiz/1"""
    _, k = map(int, input().split())
    print(sum((0 if x % k == 0 else x for x in map(int, input().split()))))


def solution_167852() -> None:
    """2차원 배열 ; https://level.goorm.io/exam/167852/%EA%B0%9C%EB%85%90-2%EC%B0%A8%EC%9B%90-%EB%B0%B0%EC%97%B4/quiz/1"""
    matrix: list[list[int]] = [[0] * 4 for _ in range(4)]
    result: list[str] = []
    for row in matrix:
        result.append(str(row))
    result.append("")

    for i in range(4):
        matrix[i][2] = 4
    matrix[3][0] = 2
    for row in matrix:
        result.append(str(row))
    result.append("")

    for i in range(4):
        matrix[3][i] = 5
    for row in matrix:
        result.append(str(row))

    result.append((str(list(range(1, 4))) + "\n") * 3)

    print("\n".join(result))


def solution_167841() -> None:
    """💤 반복과 조건 ; https://level.goorm.io/exam/167841/%EB%AC%B8%EC%A0%9C-%EB%B0%98%EB%B3%B5%EA%B3%BC-%EC%A1%B0%EA%B1%B4/quiz/1

    - 대칭성 있는 모든 부분 확인하기.
    """
    n: int = int(input())
    num_map: list[list[str]] = [["0"] * n for _ in range(n)]
    result: list[str] = []
    for x, i in enumerate(range(n), start=1):
        xs = str(x)
        num_map[i][i] = xs
        num_map[i][-i - 1] = xs
        result.append("".join(num_map[i]))
    print("\n".join(result))


def solution_167835() -> None:
    """반복문 ; https://level.goorm.io/exam/167835/%EB%B0%98%EB%B3%B5%EB%AC%B8/quiz/1"""
    print(("*" * 5 + "\n") * 5)


def solution_167837() -> None:
    """반복문 ; https://level.goorm.io/exam/167837/%EB%B0%98%EB%B3%B5%EB%AC%B8/quiz/1"""
    print("\n".join(("*" * i for i in range(1, int(input()) + 1))))


def solution_167840() -> None:
    """반복과 조건 ; https://level.goorm.io/exam/167840/%EA%B0%9C%EB%85%90-%EB%B0%98%EB%B3%B5%EA%B3%BC-%EC%A1%B0%EA%B1%B4/quiz/1"""
    n = int(input())
    result: list[str] = []
    num_map: list[list[str]] = [["0"] * n for _ in range(n)]
    for x, i in enumerate(range(n), start=1):
        xs = str(x)
        for j in range(i, n, x):
            num_map[i][j] = xs

        result.append("".join(num_map[i]))

    print("\n".join(result))


def solution_171960() -> None:
    """반복과 조건 ; https://level.goorm.io/exam/171960/%EC%88%AB%EC%9E%90-%EC%A0%9C%EA%B1%B0-%EC%A0%95%EB%A0%AC/quiz/1"""
    input()
    print(" ".join(map(str, sorted(set(map(int, input().split()))))))


def solution_167343() -> None:
    """[챌린지] 점수 계산하기 ; https://level.goorm.io/exam/167343/%EC%B1%8C%EB%A6%B0%EC%A7%80-%EC%A0%90%EC%88%98-%EA%B3%84%EC%82%B0%ED%95%98%EA%B8%B0/quiz/1"""
    input()
    ox: str = input()
    score: int = 0
    bonus: int = 1
    for char in ox:
        if char == "O":
            score += bonus
            bonus += 1
        else:
            bonus = 1
    print(score)


def solution_159697() -> None:
    """달달함이 넘쳐흘러 ; https://level.goorm.io/exam/159697/%EB%8B%AC%EB%8B%AC%ED%95%A8%EC%9D%B4-%EB%84%98%EC%B3%90%ED%9D%98%EB%9F%AC/quiz/1"""
    a: tuple[int, int, int] = tuple(map(int, input().split()))
    c: tuple[int, int, int] = tuple(map(int, input().split()))
    print(c[0] - a[2], c[1] // a[1], c[2] - a[0])


def solution_159665() -> None:
    """💦 [알고리즘먼데이] 철자 분리 집합 ; https://level.goorm.io/exam/159665/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EB%A8%BC%EB%8D%B0%EC%9D%B4-%EC%B2%A0%EC%9E%90-%EB%B6%84%EB%A6%AC-%EC%A7%91%ED%95%A9/quiz/1
    Tag: Regular expression
    """
    import re

    input()
    print(sum((1 for _ in re.finditer(r"((.)\2*)", input()))))


def solution_159667() -> None:
    """💤 [알고리즘먼데이] 출석부 ; https://level.goorm.io/exam/159667/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EB%A8%BC%EB%8D%B0%EC%9D%B4-%EC%B6%9C%EC%84%9D%EB%B6%80/quiz/1

    - 인덱싱 할 때 k 의 범위를 잘 보자."""
    import sys

    input_ = sys.stdin.readline
    n, k = map(int, input_().split())
    print(" ".join(sorted((input_().split()) for _ in range(n))[k - 1]))


def solution_147452(t: int, case: list[list[int]]) -> int:
    """💤 [함수형] 행복은 성적순이 아니잖아요. ; https://level.goorm.io/exam/147452/%ED%95%A8%EC%88%98%ED%98%95-%ED%96%89%EB%B3%B5%EC%9D%80-%EC%84%B1%EC%A0%81%EC%88%9C%EC%9D%B4-%EC%95%84%EB%8B%88%EC%9E%96%EC%95%84%EC%9A%94/quiz/1
    동일 문제: ID 147448: [기본] 행복은 성적순이 아니잖아요.

    - "백분위보다 높다" 라는 표현과 등수에 대한 크기 비교가 어떻게 되야하는지 잘 생각해보자."""
    is_all_a_plus: bool = True
    for i in range(t):
        l, s, n, _, m, *vs = case[i]
        if s / l >= n / 100 or any((v_i <= m for v_i in vs)):
            is_all_a_plus = False
            break
    return int(is_all_a_plus)


def solution_159696() -> None:
    """스타후르츠 ; https://level.goorm.io/exam/159696/%EC%8A%A4%ED%83%80%ED%9B%84%EB%A5%B4%EC%B8%A0/quiz/1

    - 방정식 세워보기.
        1 + t*x <= N
        x <= (n-1)/t
    """
    n, t, c, p = map(int, input().split())
    print((n - 1) // t * c * p)


def solution_159698() -> None:
    """운동장 한 바퀴 ; https://level.goorm.io/exam/159698/%EC%9A%B4%EB%8F%99%EC%9E%A5-%ED%95%9C-%EB%B0%94%ED%80%B4/quiz/1"""
    d1: int = int(input())
    d2: int = int(input())

    print(format(2 * (d1 + 3.141592 * d2), ".6f"))


def solution_159695() -> None:
    """💤 별 찍기-12 ; https://level.goorm.io/exam/159695/%EB%B3%84-%EC%B0%8D%EA%B8%B0-12/quiz/1"""
    n = int(input())
    print("\n".join((("*" * (n - abs(x))).rjust(n) for x in range(-(n - 1), n))))


def solution_159694() -> None:
    """🧠 이산수학 과제 ; https://level.goorm.io/exam/159694/%EC%9D%B4%EC%82%B0%EC%88%98%ED%95%99-%EA%B3%BC%EC%A0%9C/quiz/1
    🔍 다시. inner_order = k - (step * (step + 1) // 2 - step)  대신 inner_order = k - (step * (step - 1) // 2)

    Tag: Parameteric search

    step i
        1: 1/1
        2: 2/1 1/2
        3: 3/1 2/2 1/3

    order(n) := 1+2+3+... n = n*(n+1)/2
        if order(n) <  k-th order  <= order(n+1), k-th order is in (n+1) step.
        so Purpose is to find maximum n-th step to satisfy ( order(n) <  k-th order )


    """

    k: int = int(input())
    lo, hi = 1, 1000
    step = 0
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if mid * (mid + 1) // 2 < k:
            lo = mid + 1
        else:
            hi = mid - 1
    else:
        step = lo
    inner_order = k - (step * (step + 1) // 2 - step)
    a = step - (inner_order - 1)
    b = inner_order
    print(a, b)


def solution_159693() -> None:
    """💤 저작권 ; https://level.goorm.io/exam/159693/%EC%A0%80%EC%9E%91%EA%B6%8C/quiz/1

    - 올림의 성질 이해
    """
    a, l = map(int, input().split())
    print(a * (l - 1) + 1)


def solution_159692() -> None:
    """🧠 아즈텍 피라미드 ; https://level.goorm.io/exam/159692/%EC%95%84%EC%A6%88%ED%85%8D-%ED%94%BC%EB%9D%BC%EB%AF%B8%EB%93%9C/quiz/1
    🔍 등차수열 썻었나 그떄 ㅇㅇ..
        1
      1 2 1
    1 2 3 2 1
      1 2 1
        1

    - sum_blocks(n) := (1) + (1+3+1) + (1+3+5+3+1) + (...+2*n-1+...)
        ; each step i = i*(1+(2*i-1))/2 * 2 - 2*i-1
        if sum_blocks(n) <=  k-th block  < sum_blocks(n+1), height is n.
        so Purpose is to find maximum n-th step to satisfy ( sum_blocks(n)  <=  k-th block )
    """
    k = int(input())
    used_b = 0
    i = 0
    while used_b <= k:
        i += 1
        l = 2 * i - 1
        used_b += i * (1 + l) - l
    print(i - 1)


def solution_159691() -> None:
    """접시 안의 원 ; https://level.goorm.io/exam/159691/%EC%A0%91%EC%8B%9C-%EC%95%88%EC%9D%98-%EC%9B%90/quiz/1

    - 원의 접선은 접정을 지나는 반지름에 수직.
        (2/t)**2 + b**2 = a**2
    """
    print(round((int(input()) / 2) ** 2))


def solution_159177() -> None:
    """💦 [알고리즘먼데이] 경로의 개수 ; https://level.goorm.io/exam/159177/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EB%A8%BC%EB%8D%B0%EC%9D%B4-%EA%B2%BD%EB%A1%9C%EC%9D%98-%EA%B0%9C%EC%88%98/quiz/1"""
    import functools

    input()
    print(functools.reduce(lambda x, y: x * y, map(int, input().split())))


def solution_159181() -> None:
    """[알고리즘먼데이] 동명이인 ; https://level.goorm.io/exam/159181/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EB%A8%BC%EB%8D%B0%EC%9D%B4-%EB%8F%99%EB%AA%85%EC%9D%B4%EC%9D%B8/quiz/1"""
    import sys

    input_ = sys.stdin.readline
    n, s = input_().split()
    print(sum((1 if s in x else 0 for x in (input_() for _ in range(int(n))))))


def solution_151664() -> None:
    """정의를 정확하게 아는 것이 중요하지 ; https://level.goorm.io/exam/151664/%EC%A0%95%EC%9D%98%EB%A5%BC-%EC%A0%95%ED%99%95%ED%95%98%EA%B2%8C-%EC%95%84%EB%8A%94-%EA%B2%83%EC%9D%B4-%EC%A4%91%EC%9A%94%ED%95%98%EC%A7%80/quiz/1
    - 😠: 주어진 값이 모두 양수이고 피타고라스 정리를 만족해야 하기 때문에 2 번째 예시는 cosine(θ) 만 가능하다.
    """
    import math

    adjacent, hypotenuse, b = map(int, input().split())
    output: list[list[str]] = [["justice", "cosinehamsu"], ["sinehamsu", "definition"]]

    # check for sine(θ)
    i = False
    opposite = b
    if hypotenuse**2 - opposite**2 == adjacent**2:
        i = True

    # check for cosine(θ)
    j = False
    if (temp := hypotenuse**2 - b**2) > 0:
        opposite = math.sqrt(temp)
        if opposite < hypotenuse and b == adjacent:
            j = True

    print(output[i][j])


def solution_48137() -> None:
    """🔍💤 [KOI 2016] 장애물 경기 ; https://level.goorm.io/exam/48137/%EC%9E%A5%EC%95%A0%EB%AC%BC-%EA%B2%BD%EA%B8%B0/quiz/1

    https://m.blog.naver.com/jqkt15/222003684586 흠...
        다익스트라+세그먼트 트리로 된다고는 한다. 이거 사용 안하고 하는방법잇나?

    end-0 은 모두 똑같이 필요하며 현재 y의 위치와 y 축으로 이동한 누적거리를 저장하면 된다.
        동일 y 라도 거리가 가장 적은거만 저장하면 된다.
    dictionary solution:
        5번 실패, 16번, 20번에서 Timeout
        비효율적인가?
        할 목록이 y 좌표별로 정렬되어 있다면, 모든 new_y_to_dicts .items() 를 돌지 않고
        , 장애물과 겹치는 일부 구간에 대해서만 삭제 후, 장애물의 양 끝점에 대해 거리가 가장 작은 것만 한 번에 new_y_to_dicts 에 넣으면 될 듯.
        이를 위해서 이진 검색 트리가 필요해보임.
    삽입삭제 N^2+ 이진 검색 트리 NlogN?
    장애물 별로 x
    """
    import sys

    input_ = sys.stdin.readline
    print_ = sys.stdout.write
    n = int(input_())
    sy, end = map(int, input_().split())
    obstacles: list[tuple[int, int, int]] = sorted(
        (tuple(map(int, input_().split())) for _ in range(n)), key=lambda x: x[0]
    )

    # Runner's y point, moved distance except for distance to x-axis.
    y_to_dist: dict[int, int] = {sy: 0}
    for _, yl, yh in obstacles:
        new_y_to_dict: dict[int, int] = {}
        for y, dist in y_to_dist.items():
            if yl < y < yh:
                new_yl_dist = dist + y - yl
                new_yh_dist = dist + yh - y
                if new_y_to_dict.get(yl, sys.maxsize) > new_yl_dist:
                    new_y_to_dict[yl] = new_yl_dist
                if new_y_to_dict.get(yh, sys.maxsize) > new_yh_dist:
                    new_y_to_dict[yh] = new_yh_dist
            else:
                new_y_to_dict[y] = dist
        y_to_dist = new_y_to_dict

    ranks = sorted(y_to_dist.items(), key=lambda x: [x[1], x[0]])
    min_d = ranks[0][1]
    # parametric search for custom bisect_right
    lo, hi = 0, len(ranks) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if ranks[mid][1] == min_d:
            lo = mid + 1
        else:
            hi = mid - 1
    else:
        count = lo
    print_(
        f"{min_d + end}\n{count} {' '.join((str(ranks[i][0]) for i in range(count)))}\n"
    )


solution_48137()
