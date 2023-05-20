"""Purpose is to solve in 4 minutes
- 💤: 문제 이해 제대로 하기
    "문제 설명"에 모든 제한사항이 함게 주어지지 않는 경우가 있어서, 완벽히 이해하기 어려울 수 있다.
    에매한 경우, 명확하게 이해할 수 있도록 "예시" 가 주어지므로 꼭 함께 보도록 한다.
- 🧠: (신박한 풀이, 수학식을 활용한 풀이, Benchmark)
- 💦: built-in functions, grammar

Almost problems is related with String and Index manipluation.
Almost solutions of problems have 1 ~ 5 lines.
"""
# Regex to find problems in which one of upper emojis was used; """[^\uAC00-\uD7A3\d\w]+\s


def solution_181952() -> None:
    """문자열 출력하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181952"""
    print(input())


def solution_181951() -> None:
    """a와 b 출력하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181951"""
    a, b = map(int, input().split())
    print(f"a = {a}\nb = {b}")


def solution_181950() -> None:
    """문자열 반복해서 출력하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181950"""
    a, b = input().split()
    print(a * int(b))


def solution_181949() -> None:
    """대소문자 바꿔서 출력하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181949"""
    print(input().swapcase())


def solution_181948() -> None:
    """특수문자 출력하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181948"""
    print(r"""!@#$%^&*(\'"<>?:;""")


def solution_181947() -> None:
    """덧셈식 출력하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181947"""
    a, b = map(int, input().split())
    print(f"{a} + {b} = {a+b}")


def solution_181946() -> None:
    """문자열 붙여서 출력하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181946"""
    print(input().replace(" ", ""))


def solution_181945() -> None:
    """💤 문자열 돌리기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181945"""
    print("\n".join((s for s in input())))


def solution_181944() -> None:
    """홀짝 구분하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181944"""
    print(f"{n} is odd" if ((n := int(input())) & 1) else f"{n} is even")


def solution_181943(my_string: str, overwrite_string: str, s: int) -> str:
    """🧠 문자열 겹쳐쓰기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181943"""
    return my_string[:s] + overwrite_string + my_string[s + len(overwrite_string) :]


def solution_181942(str1: str, str2: str) -> str:
    """문자열 섞기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181942"""
    return "".join((s1 + s2 for s1, s2 in zip(str1, str2)))


def solution_181941(arr: list[str]) -> str:
    """문자 리스트를 문자열로 변환하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181941"""
    return "".join(arr)


def solution_181940(my_string: str, k: int) -> str:
    """문자열 곱하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181940"""
    return my_string * k


def solution_181939(a: int, b: int) -> int:
    """💤 더 크게 합치기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181939
    - 첫째자리만 고려하면 안된다. e.g. when a=89, b=8.
    """
    return x if (x := int(f"{a}{b}")) > (y := int(f"{b}{a}")) else y


def solution_181938(a: int, b: int) -> int:
    """두 수의 연산값 비교하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181938"""
    return x if (x := int(f"{a}{b}")) > (y := 2 * a * b) else y


def solution_181937(num: int, n: int) -> int:
    """n의 배수 ; https://school.programmers.co.kr/learn/courses/30/lessons/181937"""
    return int(num % n == 0)


def solution_181936(number: int, n: int, m: int) -> int:
    """💤 공배수 ; https://school.programmers.co.kr/learn/courses/30/lessons/181936
    - 최소 공배수를 구하는 문제가 아님."""
    return int(number % n == 0 and number % m == 0)


def solution_181935(n: int) -> int:
    """홀짝에 따라 다른 값 반환하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181935"""
    return (
        sum(range(1, n + 1, 2)) if n & 1 else sum((x**2 for x in range(2, n + 1, 2)))
    )


def solution_181934(ineq: str, eq: str, n: int, m: int) -> int:
    """조건 문자열 ; https://school.programmers.co.kr/learn/courses/30/lessons/181934"""
    return int(
        eval(f"{n} {ineq}{eq} {m}".replace("!", ""))  # pylint: disable=W0123 # nosec
    )


def solution_181933(a: int, b: int, flag: bool) -> int:
    """flag에 따라 다른 값 반환하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181933"""
    return a + b if flag else a - b


def solution_181932(code: str) -> str:
    """코드 처리하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181932"""
    ret: list[str] = []
    mode: int = 0
    for i, x in enumerate(code):
        if x == "1":
            mode ^= 1
        else:
            is_odd = i & 1
            if (mode == 0 and not is_odd) or (mode == 1 and is_odd):
                ret.append(x)
    result = "".join(ret)
    return result if result else "EMPTY"


def solution_181931(a: int, d: int, included: list[bool]) -> int:
    """🧠 등차수열의 특정한 항만 더하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181931

    Benchmark: Solution 1 (faster) > 2 > 3
        Solution 1: return sum(a + i * d for i, x in enumerate(included) if x)
            Win; iterates once and uses generator so that it reduces memory footprint and increases performance.
        Solution 2: x = [i for i in range(len(included)) if included[i]];   return len(x)*a+sum(x)*d
        Solution 3: return a*sum(1 for x in included if x) + sum((i for i in range(len(included)) if included[i]))*d
    """
    return sum(a + i * d for i, x in enumerate(included) if x)


def solution_181930(a: int, b: int, c: int) -> int:
    """💤 주사위 게임 2 ; https://school.programmers.co.kr/learn/courses/30/lessons/181930

    🚣 Benchmark
        It can be solved by "Counter object from collections".
        but using set is faster than using Counter because of "import" overhead.
    """
    set_ = set((a, b, c))
    if len(set_) == 1:
        return (a + b + c) * (a**2 + b**2 + c**2) * (a**3 + b**3 + c**3)
    elif len(set_) == 2:
        return (a + b + c) * (a**2 + b**2 + c**2)
    else:
        return a + b + c


def solution_181929(num_list: list[int]) -> int:
    """원소들의 곱과 합 ; https://school.programmers.co.kr/learn/courses/30/lessons/181929"""
    import math

    return int(math.prod(num_list) < sum(num_list) ** 2)


def solution_181928(num_list: list[int]) -> int:
    """💤 이어 붙인 수 ; https://school.programmers.co.kr/learn/courses/30/lessons/181928"""
    sums: list[list[str]] = [[], []]
    for num in num_list:
        sums[num & 1].append(str(num))
    return sum(map(int, map("".join, sums)))


def solution_181927(num_list: list[int]) -> list[int]:
    """마지막 두 원소 ; https://school.programmers.co.kr/learn/courses/30/lessons/181927"""
    num_list.append(x if (x := num_list[-1] - num_list[-2]) > 0 else num_list[-1] * 2)
    return num_list


def solution_181926(n: int, control: str) -> int:
    """🧠 수 조작하기 1 ; https://school.programmers.co.kr/learn/courses/30/lessons/181926

    Implementation not using Counter object; second solution
        map_ = {"w": 1, "s": -1, "d": 10, "a": -10}
        return sum((map_[k] for k in control)) + n

    🚣 Benchmark; solution using Counter object is faster than not using one.
        because Counter object is highly optimized for counting elements in a collection
        , and second solution lookup operations are performed as many <n> in normal dictionary
        , first solution is generally faster than second solution.
    """
    # first solution
    from collections import Counter

    map_ = {"w": 1, "s": -1, "d": 10, "a": -10}
    return sum((map_[k] * v for k, v in Counter(control).items())) + n


def solution_181925(numLog: list[int]) -> str:
    """수 조작하기 2 ; https://school.programmers.co.kr/learn/courses/30/lessons/181925"""
    map_: dict[int, str] = {1: "w", -1: "s", 10: "d", -10: "a"}
    return "".join((map_[numLog[i + 1] - numLog[i]] for i in range(len(numLog) - 1)))


def solution_181924(arr: list[int], queries: list[list[int]]) -> list[int]:
    """수열과 구간 쿼리 3 ; https://school.programmers.co.kr/learn/courses/30/lessons/181924"""
    for i, j in queries:
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def solution_181923(arr: list[int], queries: list[list[int]]) -> list[int]:
    """수열과 구간 쿼리 2 ; https://school.programmers.co.kr/learn/courses/30/lessons/181923"""
    result: list[int] = []
    # eary stopping solution
    for s, e, k in queries:
        kk = k + 1
        temp: list[int] = []
        for i in range(s, e + 1):
            if arr[i] == kk:
                result.append(kk)
                break
            elif arr[i] > kk:
                temp.append(arr[i])
        else:
            result.append(min(temp) if temp else -1)
    return result


def solution_181922(arr: list[int], queries: list[list[int]]) -> list[int]:
    """💤 수열과 구간 쿼리 4 ; https://school.programmers.co.kr/learn/courses/30/lessons/181922"""
    for s, e, k in queries:
        for i in range(s + k - r if k != 0 and (r := s % k) != 0 else s, e + 1, k):
            arr[i] += 1
    return arr


def solution_181921(l: int, r: int) -> list[int]:
    """💤 배열 만들기 2 ; https://school.programmers.co.kr/learn/courses/30/lessons/181921

    - itertools.product 의 특성상 sort를 할 필요가 없다.
    """
    import itertools

    return (
        x
        if (
            x := list(
                filter(
                    lambda x: l <= x <= r,
                    (
                        int("".join(digits))
                        for digits in itertools.product("05", repeat=len(str(r)))
                    ),
                )
            )
        )
        else [-1]
    )


def solution_181920(start: int, end: int) -> list[int]:
    """카운트 업 ; https://school.programmers.co.kr/learn/courses/30/lessons/181920"""
    return list(range(start, end + 1))


def solution_181919(n: int) -> list[int]:
    """콜라츠 수열 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181919"""
    stack = [n]
    while stack[-1] != 1:
        stack.append(3 * stack[-1] + 1 if stack[-1] & 1 else stack[-1] // 2)
    return stack


def solution_181918(arr: list[int]) -> list[int]:
    """💤 배열 만들기 4 ; https://school.programmers.co.kr/learn/courses/30/lessons/181918"""
    stack: list[int] = []
    for i in range(len(arr)):
        while stack and stack[-1] >= arr[i]:
            stack.pop()
        stack.append(arr[i])
    return stack


def solution_181917(x1: bool, x2: bool, x3: bool, x4: bool) -> bool:
    """간단한 논리 연산 ; https://school.programmers.co.kr/learn/courses/30/lessons/181917"""
    return (x1 or x2) and (x3 or x4)


def solution_181916(a: int, b: int, c: int, d: int) -> int:
    """💤 주사위 게임 3 ; https://school.programmers.co.kr/learn/courses/30/lessons/181916"""
    from collections import Counter

    counter = Counter((a, b, c, d))
    if len(counter) == 1:  # 4
        return 1111 * a
    elif len(counter) == 2:  # (2,2) (3,1)
        p, q = counter.most_common(2)
        if p[1] == 3:
            return (10 * p[0] + q[0]) ** 2
        else:
            return (p[0] + q[0]) * abs(p[0] - q[0])
    elif len(counter) == 3:  # (1, 1, 2)
        _, q, r = counter.most_common(3)
        return q[0] * r[0]
    else:
        return min(counter)


def solution_181915(my_string: str, index_list: list[int]) -> str:
    """글자 이어 붙여 문자열 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181915"""
    return "".join((my_string[i] for i in index_list))


def solution_181914(number: str) -> int:
    """9로 나눈 나머지 ; https://school.programmers.co.kr/learn/courses/30/lessons/181914"""
    return sum((int(s) for s in number)) % 9


def solution_181913(my_string: str, queries: list[list[int]]) -> str:
    """💦 문자열 여러 번 뒤집기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181913"""
    for s, e in queries:
        my_string = "".join(
            (
                my_string[:s],
                my_string[e : None if s == 0 else s - 1 : -1],
                my_string[e + 1 :],
            )
        )
    return my_string


def solution_181912(int_strs: list[str], k: int, s: int, l: int) -> list[int]:
    """배열 만들기 5 ; https://school.programmers.co.kr/learn/courses/30/lessons/181912"""
    return [x for x in (int(int_str[s : s + l]) for int_str in int_strs) if x > k]


def solution_181911(my_strings: list[str], parts: list[list[int]]) -> str:
    """부분 문자열 이어 붙여 문자열 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181911"""
    return "".join(
        (my_string[part[0] : part[1] + 1] for my_string, part in zip(my_strings, parts))
    )


def solution_181910(my_string: str, n: int) -> str:
    """문자열의 뒤의 n글자 ; https://school.programmers.co.kr/learn/courses/30/lessons/181910"""
    return my_string[-n:]


def solution_181909(my_string: str) -> list[str]:
    """접미사 배열 ; https://school.programmers.co.kr/learn/courses/30/lessons/181909"""
    return sorted((my_string[i:] for i in range(len(my_string))))


def solution_181908(my_string: str, is_suffix: str) -> int:
    """접미사인지 확인하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181908
    Same solution:  int(my_string[-len(is_suffix):] == is_suffix)
    """
    return int(my_string.endswith(is_suffix))


def solution_181907(my_string: str, n: int) -> str:
    """문자열의 앞의 n글자 ; https://school.programmers.co.kr/learn/courses/30/lessons/181907"""
    return my_string[:n]


def solution_181906(my_string: str, is_prefix: str) -> int:
    """💦 접두사인지 확인하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181906
    Same solution:  return int(my_string[: len(is_prefix)] == is_prefix)
    """
    return int(my_string.startswith(is_prefix))


def solution_181905(my_string: str, s: int, e: int) -> str:
    """문자열 뒤집기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181905"""
    return (
        my_string[:s]
        + "".join((my_string[i] for i in range(e, s - 1, -1)))
        + my_string[e + 1 :]
    )


def solution_181904(my_string: str, m: int, c: int) -> str:
    """💤 세로 읽기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181904
    Same solution:  return "".join((my_string[i] for i in range(c-1, len(my_string), m)))
    """
    return my_string[c - 1 :: m]


def solution_181903(q: int, r: int, code: str) -> str:
    """qr code ; https://school.programmers.co.kr/learn/courses/30/lessons/181903
    Same solution:  return "".join((code[c] for c in range(r, len(code), q)))"""
    return code[r::q]


def solution_181902(my_string: str) -> list[int]:
    """💤 문자 개수 세기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181902
    - map 을 만들 필요가 없다."""
    # ord("a") = 97, ord("A") = 65, ord("Z") = 90. ord numbers are not consecutive.
    counts: list[int] = [0] * 52
    for s in my_string:
        if s.isupper():
            counts[ord(s) - 65] += 1
        else:
            counts[ord(s) - 71] += 1  # -97+26
    return counts


def solution_181901(n: int, k: int) -> list[int]:
    """배열 만들기 1 ; https://school.programmers.co.kr/learn/courses/30/lessons/181901"""
    return list(range(k, n + 1, k))


def solution_181900(my_string: str, indices: list[int]) -> str:
    """글자 지우기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181900"""
    compression = [True] * len(my_string)
    for i in indices:
        compression[i] = False
    return "".join((x for i, x in enumerate(my_string) if compression[i]))


def solution_181899(start: int, end: int) -> list[int]:
    """카운트 다운 ; https://school.programmers.co.kr/learn/courses/30/lessons/181899"""
    return list(range(start, end - 1, -1))


def solution_181898(arr: list[int], idx: int) -> int:
    """가까운 1 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181898
    - 😠 "idx보다 크면서": 테스트케이스들은 idx 보다 같거나 크면서에 해당되는 경우에만 정답처리됨.
    """
    try:
        x = next((i for i in range(idx, len(arr)) if arr[i] == 1))
    except StopIteration:
        return -1
    else:
        return x


def solution_181897(n: int, slicer: list[int], num_list: list[int]) -> list[int]:
    """리스트 자르기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181897"""
    a, b, c = slicer
    if n == 1:
        return num_list[0 : b + 1]
    elif n == 2:
        return num_list[a:]
    elif n == 3:
        return num_list[a : b + 1]
    else:
        return num_list[a : b + 1 : c]


def solution_181896(num_list: list[int]) -> int:
    """첫 번째로 나오는 음수 ; https://school.programmers.co.kr/learn/courses/30/lessons/181896"""
    try:
        i = next((i for i, n in enumerate(num_list) if n < 0))
    except StopIteration:
        return -1
    else:
        return i


def solution_181895(arr: list[int], intervals: list[list[int]]) -> list[int]:
    """배열 만들기 3 ; https://school.programmers.co.kr/learn/courses/30/lessons/181895"""
    [[x1, x2], [x3, x4]] = intervals
    return arr[x1 : x2 + 1] + arr[x3 : x4 + 1]


def solution_181894(arr: list[int]) -> list[int]:
    """💦💤 2의 영역 ; https://school.programmers.co.kr/learn/courses/30/lessons/181894

    🚣 If "arr_index(2)" does not cause an error, "next(<Generator expression>)" does not causes StopIteration because Generator expression generates at least one element.
    """
    try:
        a = arr.index(2)
        b = len(arr) - 1 - next(i for i, x in enumerate(reversed(arr)) if x == 2)
    except ValueError:
        return [-1]
    else:
        return arr[a : b + 1]


def solution_181893(arr: list[int], query: list[int]) -> list[int]:
    """배열 조각하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181893"""
    is_even: int = 0
    for qi in query:
        if is_even := is_even ^ 1:
            arr = arr[: qi + 1]
        else:
            arr = arr[qi:]
    return arr


def solution_181892(num_list: list[int], n: int) -> list[int]:
    """n 번째 원소부터 ; https://school.programmers.co.kr/learn/courses/30/lessons/181892"""
    return num_list[n - 1 :]


def solution_181891(num_list: list[int], n: int) -> list[int]:
    """💤 순서 바꾸기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181891"""
    return num_list[n:] + num_list[:n]


def solution_181890(str_list: list[str]) -> list[str]:
    """왼쪽 오른쪽 ; https://school.programmers.co.kr/learn/courses/30/lessons/181890"""
    for i, s in enumerate(str_list):
        if s == "l":
            return str_list[:i]
        elif s == "r":
            return str_list[i + 1 :]
    return []


def solution_181889(num_list: list[int], n: int) -> list[int]:
    """n 번째 원소까지 ; https://school.programmers.co.kr/learn/courses/30/lessons/181889"""
    return num_list[:n]


def solution_181888(num_list: list[int], n: int) -> list[int]:
    """n개 간격의 원소들 ; https://school.programmers.co.kr/learn/courses/30/lessons/181888"""
    return num_list[::n]


def solution_181887(num_list: list[int]) -> int:
    """홀수 vs 짝수 ; https://school.programmers.co.kr/learn/courses/30/lessons/181887"""
    return max(
        sum((num_list[i] for i in range(0, len(num_list), 2))),
        sum((num_list[i] for i in range(1, len(num_list), 2))),
    )


def solution_181886(names: list[str]) -> list[str]:
    """💦 5명씩 ; https://school.programmers.co.kr/learn/courses/30/lessons/181886

    Another solution
        import math
        return [names[i*5] for i in range(math.ceil(len(names)/5))]
    """
    return names[::5]


def solution_181885(todo_list: list[str], finished: list[bool]) -> list[str]:
    """할 일 목록 ; https://school.programmers.co.kr/learn/courses/30/lessons/181885"""
    return [x for x, is_ok in zip(todo_list, finished) if not is_ok]


def solution_181884(numbers: list[int], n: int) -> int:
    """💤 n보다 커질 때까지 더하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181884"""
    total: int = 0
    for num in numbers:
        total += num
        if total > n:
            break
    return total


def solution_181883(arr: list[int], queries: list[list[int]]) -> list[int]:
    """수열과 구간 쿼리 1 ; https://school.programmers.co.kr/learn/courses/30/lessons/181883"""
    for s, e in queries:
        for i in range(s, e + 1):
            arr[i] += 1
    return arr


def solution_181882(arr: list[int]) -> list[int]:
    """조건에 맞게 수열 변환하기 1 ; https://school.programmers.co.kr/learn/courses/30/lessons/181882"""
    for i, num in enumerate(arr):
        is_odd = num & 1
        if num >= 50 and not is_odd:
            arr[i] = num // 2
        elif num < 50 and is_odd:
            arr[i] = num * 2
    return arr


def solution_181881(arr: list[int]) -> int:
    """💤 조건에 맞게 수열 변환하기 2 ; https://school.programmers.co.kr/learn/courses/30/lessons/181881

    - Note that minimum <x> is 0 (given array) in formula arr(x) == arr(x + 1).
    """
    count: int = 0
    while True:
        new_arr: list[int] = []
        for num in arr:
            is_odd = num & 1
            if num >= 50 and not is_odd:
                new_arr.append(num // 2)
            elif num < 50 and is_odd:
                new_arr.append(num * 2 + 1)
            else:
                new_arr.append(num)

        if new_arr == arr:
            break
        else:
            arr = new_arr
            count += 1
    return count


def solution_181880(num_list: list[int]) -> int:
    """💤 1로 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181880
    - 😠 "정수" 가 아니라 "양의 정수"임."""
    # -3 denotes subtracting first "0b" and last "1" bit.
    return sum(len(bin(num)) - 3 for num in num_list)


def solution_181879(num_list: list[int]) -> int:
    """길이에 따른 연산 ; https://school.programmers.co.kr/learn/courses/30/lessons/181879"""
    import math

    return sum(num_list) if len(num_list) >= 11 else math.prod(num_list)


def solution_181878(my_str: str, pat: str) -> int:
    """원하는 문자열 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181878"""
    return 1 if my_str.lower().find(pat.lower()) >= 0 else 0


def solution_181877(my_str: str) -> str:
    """대문자로 바꾸기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181877"""
    return my_str.upper()


def solution_181876(my_str: str) -> str:
    """소문자로 바꾸기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181876"""
    return my_str.lower()


def solution_181875(str_arr: str) -> list[str]:
    """배열에서 문자열 대소문자 변환하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181875"""
    return [s.upper() if i & 1 else s.lower() for i, s in enumerate(str_arr)]


def solution_181874(my_string: str) -> str:
    """💤 A 강조하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181874"""
    return my_string.lower().replace("a", "A")


def solution_181873(my_string: str, alp: str) -> str:
    """특정한 문자를 대문자로 바꾸기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181873"""
    return "".join((s.upper() if s == alp else s for s in my_string))


def solution_181872(my_str: str, pat: str) -> str:
    """🧠 특정 문자열로 끝나는 가장 긴 부분 문자열 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181872
    the soultion is same:
        return myString[:len(myString)-"".join(reversed(myString)).find("".join(reversed(pat)))]
    """
    return my_str[: my_str.rindex(pat) + len(pat)]


def solution_181871(my_str: str, pat: str) -> int:
    """문자열이 몇 번 등장하는지 세기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181871
    - 😠 "등장하는 횟수" 가 이미 발견된 문자의 일부를 포함하는 경우도 포함한다.
    """
    i = 0
    count = 0
    while True:
        i = my_str.find(pat, i)
        if i >= 0:
            count += 1
            i += 1
        else:
            break

    return count


def solution_181870(str_arr: str) -> list[str]:
    """ad 제거하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181870"""
    return [s for s in str_arr if "ad" not in s]


def solution_181869(my_string: str) -> list[str]:
    """공백으로 구분하기 1 ; https://school.programmers.co.kr/learn/courses/30/lessons/181869"""
    return my_string.split()


def solution_181868(my_string: str) -> list[str]:
    """공백으로 구분하기 2 ; https://school.programmers.co.kr/learn/courses/30/lessons/181868"""
    return my_string.split()


def solution_181867(my_str: str) -> list[int]:
    """x 사이의 개수 ; https://school.programmers.co.kr/learn/courses/30/lessons/181867"""
    return list(map(len, my_str.split("x")))


def solution_181866(my_str: str) -> list[str]:
    """💤 문자열 잘라서 정렬하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181866

    - "단, 빈 문자열은 반환할 배열에 넣지 않습니다."
    """
    return sorted((my_str.replace("x", " ").split()))


def solution_181865(binomial: str) -> int:
    """간단한 식 계산하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181865"""
    return eval(binomial)  # pylint: disable=W0123 # nosec


def solution_181864(my_str: str, pat: str) -> int:
    """문자열 바꿔서 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181864"""
    return (
        1 if "".join(("B" if s == "A" else "A" for s in my_str)).find(pat) >= 0 else 0
    )


def solution_181863(rny_string: str) -> str:
    """rny_string ; https://school.programmers.co.kr/learn/courses/30/lessons/181863"""
    return rny_string.replace("m", "rn")


def solution_181862(my_str: str) -> list[str]:
    """세 개의 구분자 ; https://school.programmers.co.kr/learn/courses/30/lessons/181862"""
    return (
        x
        if (x := "".join((" " if s in "abc" else s for s in my_str)).split())
        else ["EMPTY"]
    )


def solution_181861(arr: list[int]) -> list[int]:
    """배열의 원소만큼 추가하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181861"""
    import itertools

    return list(itertools.chain.from_iterable(([x] * x for x in arr)))


def solution_181860(arr: list[int], flag: list[bool]) -> list[int]:
    """💤 빈 배열에 추가, 삭제하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181860

    - 😠 "flag[i] 가 false라면 X에서 마지막 arr[i]개의 원소를 제거한 뒤 X를 return 하는 solution 함수를 작성해 주세요."
        예시를 보면, false 가 나온다고 즉시 return 하는 것이 아니다.
    """
    stack: list[int] = []
    for i, x in enumerate(arr):
        if flag[i]:
            stack.extend([x] * (x * 2))
        else:
            for _ in range(x):
                stack.pop()
    return stack


def solution_181859(arr: list[int]) -> list[int]:
    """💤 배열 만들기 6 ; https://school.programmers.co.kr/learn/courses/30/lessons/181859"""
    stack: list[int] = []
    for x in arr:
        # you don't need to use "not stack and (stack and stack[-1] != x)""
        if not stack or stack[-1] != x:
            stack.append(x)
        else:
            stack.pop()
    return stack if stack else [-1]


def solution_181858(arr: list[int], k: int) -> list[int]:
    """💤 무작위로 K개의 수 뽑기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181858"""
    x = list(dict.fromkeys(arr))
    x.extend([-1] * (k - len(x)))
    return x[:k]


def solution_181857(arr: list[int]) -> list[int]:
    """💤 배열의 길이를 2의 거듭제곱으로 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181857"""
    prod: int = 1
    if len(arr) > 1:
        for _ in range(len(arr)):
            prod <<= 1
            if prod >= len(arr):
                arr.extend([0] * (prod - len(arr)))
                break
    return arr


def solution_181856(arr1: list[int], arr2: list[int]) -> int:
    """배열 비교하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181856"""
    x, y = (len(arr1), sum(arr1)), (len(arr2), sum(arr2))
    if x > y:
        return 1
    elif x == y:
        return 0
    else:
        return -1


def solution_181855(strArr: list[str]) -> int:
    """💤 문자열 묶기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181855
    - 그룹의 크기 == 그룹 안에 있는 원소들의 개수. 가장 개수가 많은 그룹의 문자열의 길이가 아님.
    """
    from collections import Counter

    return Counter((len(x) for x in strArr)).most_common(1)[0][1]


def solution_181854(arr: list[int], n: int) -> list[int]:
    """배열의 길이에 따라 다른 연산하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181854"""
    return (
        [arr[i] if i & 1 else arr[i] + n for i in range(len(arr))]
        if len(arr) & 1
        else [arr[i] + n if i & 1 else arr[i] for i in range(len(arr))]
    )


def solution_181853(num_list: list[int]) -> list[int]:
    """뒤에서 5등까지 ; https://school.programmers.co.kr/learn/courses/30/lessons/181853"""
    return sorted(num_list)[:5]


def solution_181852(num_list: list[int]) -> list[int]:
    """뒤에서 5등 위로 ; https://school.programmers.co.kr/learn/courses/30/lessons/181852"""
    return sorted(num_list)[5:]


def solution_181851(rank: list[int], attendance: list[bool]) -> int:
    """💤 전국 대회 선발 고사 ; https://school.programmers.co.kr/learn/courses/30/lessons/181851"""
    ranks = sorted(((r, i) for i, r in enumerate(rank) if attendance[i]))
    return 10000 * ranks[0][1] + 100 * ranks[1][1] + ranks[2][1]


def solution_181850(flo: float) -> int:
    """정수 부분 ; https://school.programmers.co.kr/learn/courses/30/lessons/181850"""
    return int(flo)


def solution_181849(num_str: str) -> int:
    """문자열 정수의 합 ; https://school.programmers.co.kr/learn/courses/30/lessons/181849"""
    return sum((int(num_s) for num_s in num_str))


def solution_181848(n_str: str) -> int:
    """문자열을 정수로 변환하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181848"""
    return int(n_str)


def solution_181847(n_str: str) -> str:
    """0 떼기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181847"""
    return str(int(n_str))


def solution_181846(a: str, b: str) -> str:
    """두 수의 합 ; https://school.programmers.co.kr/learn/courses/30/lessons/181846"""
    return str(int(a) + int(b))


def solution_181845(n: int) -> str:
    """문자열로 변환 ; https://school.programmers.co.kr/learn/courses/30/lessons/181845"""
    return str(n)


def solution_181844(arr: list[int], delete_list: list[int]) -> list[int]:
    """배열의 원소 삭제하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181844"""
    return [x for x in arr if x not in delete_list]


def solution_181843(my_string: str, target: str) -> int:
    """부분 문자열인지 확인하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181843"""
    return int(target in my_string)


def solution_181842(str1: str, str2: str) -> int:
    """부분 문자열 ; https://school.programmers.co.kr/learn/courses/30/lessons/181842"""
    return int(str1 in str2)


def solution_181841(str_list: list[str], ex: str) -> str:
    """꼬리 문자열 ; https://school.programmers.co.kr/learn/courses/30/lessons/181841"""
    return "".join((s for s in str_list if ex not in s))


def solution_181840(num_list: list[int], n: int) -> int:
    """정수 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181840"""
    return int(any((num == n for num in num_list)))


def solution_181839(a: int, b: int) -> int:
    """주사위 게임 1 ; https://school.programmers.co.kr/learn/courses/30/lessons/181839"""
    if (x := (a & 1, b & 1)) == (True, True):
        return a**2 + b**2
    elif x == (False, False):
        return abs(a - b)
    else:
        return 2 * (a + b)


def solution_181838(date1: list[int], date2: list[int]) -> int:
    """날짜 비교하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181838"""
    return int(date1 < date2)


def solution_181837(order: list[str]) -> int:
    """커피 심부름 ; https://school.programmers.co.kr/learn/courses/30/lessons/181837"""
    return sum(
        (
            4500
            if purchase.find("americano") >= 0 or purchase.find("anything") >= 0
            else 5000
            for purchase in order
        )
    )


def solution_181836(picture: list[str], k: int) -> list[str]:
    """💤 그림 확대 ; https://school.programmers.co.kr/learn/courses/30/lessons/181836"""
    import itertools

    return list(
        itertools.chain.from_iterable(
            (["".join((s * k for s in row))] * k for row in picture)
        )
    )


def solution_181835(arr: list[int], k: int) -> list[int]:
    """조건에 맞게 수열 변환하기 3 ; https://school.programmers.co.kr/learn/courses/30/lessons/181835"""
    return list(map(lambda x: x * k, arr)) if k & 1 else list(map(lambda x: x + k, arr))


def solution_181834(my_string: str) -> str:
    """l로 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181834"""
    return "".join(("l" if s < "l" else s for s in my_string))


def solution_181833(n: int) -> list[list[int]]:
    """특별한 이차원 배열 1 ; https://school.programmers.co.kr/learn/courses/30/lessons/181833"""
    # symmetric matrix
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1
    return matrix


def solution_181832(n: int) -> list[list[int]]:
    """🧠 정수를 나선형으로 배치하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181832"""
    # spiral pattern
    matrix = [[0] * n for _ in range(n)]
    x, y = 0, 0
    dx, dy = 0, 1
    for num in range(1, n**2 + 1):
        matrix[x][y] = num
        if (
            (new_x := x + dx) >= n
            or (new_y := y + dy) >= n
            or matrix[new_x][new_y] != 0  # it is run when new_y == -1
        ):
            dx, dy = dy, -dx
            x, y = x + dx, y + dy
        else:
            x, y = new_x, new_y
    return matrix


def solution_181831(arr: list[list[int]]) -> int:
    """💤 특별한 이차원 배열 2 ; https://school.programmers.co.kr/learn/courses/30/lessons/181831
    - <is_symmetry: bool> 이 필요가 없다."""
    for i in range(1, len(arr)):
        for j in range(0, i):
            if arr[i][j] != arr[j][i]:
                return 0
    return 1


def solution_181830(arr: list[list[int]]) -> list[list[int]]:
    """정사각형으로 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181830"""
    if (diff := len(arr) - len(arr[0])) > 0:
        for row in arr:
            row.extend([0] * diff)
    elif (diff := len(arr[0]) - len(arr)) > 0:
        arr.extend([[0] * len(arr[0])] * diff)
    return arr


def solution_181829(board: list[list[int]], k: int) -> int:
    """이차원 배열 대각선 순회하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/181829"""
    return sum(
        (
            board[i][j]
            for i in range(len(board))
            for j in range(len(board[0]))
            if i + j <= k
        )
    )


def solution_120956(babbling: list[str]) -> int:
    """💤 옹알이 ; https://school.programmers.co.kr/learn/courses/30/lessons/120956"""
    from itertools import permutations

    speakable_words: list[str] = ["aya", "ye", "woo", "ma"]
    words_pm: list[str] = []
    for i in range(1, len(speakable_words) + 1):
        words_pm.extend(
            ("".join(list(perm)) for perm in permutations(speakable_words, i))
        )

    return sum((1 for b in babbling if b in words_pm))


def solution_120924(common: list[int]) -> int:
    """다음에 올 숫자 ; https://school.programmers.co.kr/learn/courses/30/lessons/120924"""
    answer: int = 0
    x: int = common[1] - common[0]
    y: int = common[-1] - common[-2]

    if x != y:
        answer = common[-1] * (common[-1] // common[-2])
    else:
        answer = common[-1] + x
    return answer


def solution_120923(num: int, total: int) -> list[int]:
    """🧠 연속된 수의 합 ; https://school.programmers.co.kr/learn/courses/30/lessons/120923
    from Arithmetic sequence, 공차가 1인 어떤 등차수열의 합을 total, 첫 항을 a 라 하면,
        total = a + (a+1) + (a+2) ... (a + num-1)
            # value to be returned: [a + (a+1) + (a+2) ... (a + num-1)]
        total = num*a + (0+1+2+3+...num-1)
        total = num*a + num*(num-1)/2
        a = (total - num*(num-1)/2) / num
    It can be solved by using endpoint's two pointer (more complex than that.)
    """
    a: int = (total - num * (num - 1) // 2) // num
    return list(range(a, a + num))


def solution_120922(M: int, N: int) -> int:
    """종이 자르기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120922"""
    return (M - 1) + (M * (N - 1))


def solution_120921(A: str, B: str) -> int:
    """🧠 문자열 밀기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120921
    # for i in range(len(A)):
    #     if A[-i:] + A[:-i] == B:
    #         return i
    """
    return (B * 2).find(A)


def solution_120913(my_str: str, n: int) -> list[str]:
    """💤 잘라서 배열로 저장하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120913"""
    import math

    return [my_str[i * n : (i + 1) * n] for i in range(math.ceil(len(my_str) / n))]


def solution_120912(array: list[int]) -> int:
    """💦 7의 개수 ; https://school.programmers.co.kr/learn/courses/30/lessons/120912"""
    return str(array).count("7")


def solution_120911(my_string: str) -> str:
    """문자열 정렬하기 (2) ; https://school.programmers.co.kr/learn/courses/30/lessons/120911"""
    return "".join(sorted(my_string.lower()))


def solution_120910(n: int, t: int) -> int:
    """세균 증식 ; https://school.programmers.co.kr/learn/courses/30/lessons/120910"""
    return n * 2**t


def solution_120909(n: int) -> int:
    """💦 제곱수 판별하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120909"""
    return 1 if float(n**0.5).is_integer() else 2


def solution_120908(str1: str, str2: str) -> int:
    """문자열 안에 문자열 ; https://school.programmers.co.kr/learn/courses/30/lessons/120908"""
    return 1 if str1.find(str2) >= 0 else 2


def solution_120907(quiz: list[str]) -> list[str]:
    """💤 OX퀴즈 ; https://school.programmers.co.kr/learn/courses/30/lessons/120907"""
    return [
        "O" if x else "X" for x in map(eval, map(lambda x: x.replace("=", "=="), quiz))
    ]


def solution_120906(n: int) -> int:
    """자릿수 더하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120906"""
    return sum(map(int, str(n)))


def solution_120905(n: int, numlist: list[int]):
    """💤 n의 배수 고르기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120905

    - De Morgan's laws ; n 의 배수가 아닌 수들을 제거한 배열 구하기 = n 의 배수인 수 구하기
    """
    return [num for num in numlist if num % n == 0]


def solution_120904(num: int, k: int):
    """숫자 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120904"""
    return x + 1 if (x := str(num).find(str(k))) >= 0 else x


def solution_120903(s1: list[str], s2: list[str]) -> int:
    """배열의 유사도 ; https://school.programmers.co.kr/learn/courses/30/lessons/120903"""
    return len(set(s1).intersection(s2))


def solution_120902(my_string: str) -> int:
    """문자열 계산하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120902"""
    return eval(my_string)  # pylint: disable=W0123 # nosec


def solution_120899(array: list[int]) -> list[int]:
    """가장 큰 수 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120899"""
    max_i = max((i for i in range(len(array))), key=lambda i: array[i])
    return [array[max_i], max_i]


def solution_120898(message: str) -> int:
    """편지 ; https://school.programmers.co.kr/learn/courses/30/lessons/120898
    - 공백도 "글자" 에 포함되는데 2cm 인지 예시 확인하기.
    """
    return len(message) * 2


def solution_120897(n: int) -> list[int]:
    """약수 구하기 https://school.programmers.co.kr/learn/courses/30/lessons/120897"""
    divisors: list[int] = []
    for i in range(1, int(n**0.5) + 1):
        quotient, remainder = divmod(n, i)
        if remainder == 0:
            divisors.append(i)
            # calculate counterpart of the divisor
            if i != quotient:  # To avoid duplicate factors like 5*5
                divisors.append(quotient)
    return sorted(divisors)


def solution_120896(s: str) -> str:
    """💤 한 번만 등장한 문자 ; https://school.programmers.co.kr/learn/courses/30/lessons/120896

    - 정규식 써서 한 번만 등장한 문자를 모두 찾기는 어렵다. 연속된 문자열을 찾는게 아니라 전체 개수를 찾아야 하기 때문.
    """
    from collections import Counter

    return "".join(
        (
            kv[0]
            for kv in sorted(
                filter(lambda kv: kv[1] == 1, Counter(s).items()), key=lambda kv: kv[0]
            )
        )
    )


def solution_120895(my_string: str, num1: int, num2: int) -> str:
    """인덱스 바꾸기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120895"""
    my_str_l = list(my_string)
    my_str_l[num1], my_str_l[num2] = my_str_l[num2], my_str_l[num1]
    return "".join(my_str_l)


def solution_120894(numbers: str) -> int:
    """💤 영어가 싫어요 ; https://school.programmers.co.kr/learn/courses/30/lessons/120894

    Solution using Regex expression (it is slower than current solution, especially for very longer input string because of regex's overhead)
        import re

        num_str = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        pattern = re.compile("|".join(num_str))
        num_map = {s: str(i) for s, i in zip(num_str, range(10))}
        return int("".join((num_map[x.group()] for x in re.finditer(pattern, numbers))))
    """
    for num, eng in enumerate(
        ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ):
        numbers = numbers.replace(eng, str(num))
    return int(numbers)


def solution_120893(my_string: str) -> str:
    """대소문자와 소문자 ; https://school.programmers.co.kr/learn/courses/30/lessons/120893"""
    return my_string.swapcase()


def solution_120892(cipher: str, code: int) -> str:
    """💤 암호 해독 ; https://school.programmers.co.kr/learn/courses/30/lessons/120892
    Same solution:  return "".join((cipher[i * code - 1] for i in range(1, len(cipher) // code + 1)))
    """
    return "".join(cipher[code - 1 :: code])


def solution_120891(order: str) -> int:
    """369게임 ; https://school.programmers.co.kr/learn/courses/30/lessons/120891"""
    return sum((1 if x in "369" else 0 for x in str(order)))


def solution_120890(array: list[int], n: int) -> int:
    """💤 가까운 수 ; https://school.programmers.co.kr/learn/courses/30/lessons/120890"""
    min_diff: int = abs(n - array[0])
    cloest: int = array[0]
    for i in range(1, len(array)):
        if (diff := abs(n - array[i])) < min_diff:
            min_diff = diff
            cloest = array[i]
        elif diff == min_diff and array[i] < cloest:
            cloest = array[i]

    return cloest


def solution_120889(sides: list[int]) -> int:
    """삼각형의 완성조건 (1) ; https://school.programmers.co.kr/learn/courses/30/lessons/120889"""
    sides.sort()
    return 1 if sides[2] < sides[0] + sides[1] else 2


def solution_120888(my_string: str) -> str:
    """💦 중복된 문자 제거 ; https://school.programmers.co.kr/learn/courses/30/lessons/120888"""
    return "".join(dict.fromkeys(my_string))


def solution_120887(i: int, j: int, k: int):
    """k의 개수 ; https://school.programmers.co.kr/learn/courses/30/lessons/120887"""
    return sum((str(x).count(str(k)) for x in range(i, j + 1)))


def solution_120886(before: str, after: str) -> int:
    """A로 B 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120886"""
    from collections import Counter

    return int(Counter(before) == Counter(after))


def solution_120885(bin1: str, bin2: str) -> str:
    """💦 이진수 더하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120885"""
    return format(int(bin1, 2) + int(bin2, 2), "b")


def solution_120884(chicken: int) -> int:
    """🧠 치킨 쿠폰 ; https://school.programmers.co.kr/learn/courses/30/lessons/120884

    - Multiplier is not Fraction(1, 9). floating point is different according to the number of initial chicken.
    """
    return int(chicken * float(f"0.{'1'*len(str(chicken))}"))


def solution_120883(id_pw: list[str], db: list[list[str]]) -> str:
    """💦 로그인 성공? ; https://school.programmers.co.kr/learn/courses/30/lessons/120883"""
    if db_pw := dict(db).get(id_pw[0]):
        return "login" if db_pw == id_pw[1] else "wrong pw"
    else:
        return "fail"


def solution_120882(score: list[list[int]]) -> list[int]:
    """등수 매기기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120882"""
    scores: list[int] = [x[0] + x[1] for x in score]
    sorted_scores = sorted(scores, reverse=True)
    ranks = {}
    for i, num in enumerate(sorted_scores, start=1):
        if num not in ranks:
            ranks[num] = i
    return [ranks[score] for score in scores]


def solution_120880(numlist: list[int], n: int) -> list[int]:
    """특이한 정렬 ; https://school.programmers.co.kr/learn/courses/30/lessons/120880"""
    return sorted(numlist, key=lambda num: (abs(n - num), -num))


def solution_120878(a: int, b: int) -> int:
    """유한소수 판별하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120878"""
    import math

    b //= math.gcd(a, b)
    # Factor out all the 2s and 5s from the denominator
    for x in (2, 5):
        while True:
            quotient, remainder = divmod(b, x)
            if remainder == 0:
                b = quotient
            else:
                break

    # If the remaining denominator is 1, the division results in a finite floating point
    return 1 if b == 1 else 2


def solution_120876(lines: list[list[int]]) -> int:
    """💤 겹치는 선분의 길이 ; https://school.programmers.co.kr/learn/courses/30/lessons/120876"""
    ## General version of this problem
    points = [False] * 200
    for i, line in enumerate(lines):
        for j in range(i + 1, len(lines)):
            a = line[0] if line[0] >= lines[j][0] else lines[j][0]
            b = line[1] if line[1] <= lines[j][1] else lines[j][1]

            for x in range(a, b):
                points[x] = True

    return sum((1 for p in points if p))


def solution_120875(dots: list[list[int]]) -> int:  # type: ignore
    """💤 평행 ; https://school.programmers.co.kr/learn/courses/30/lessons/120875"""
    ## General version of this problem
    import itertools

    dots: list[tuple[int, int]] = list(map(tuple, dots))
    # generate_pairs
    all_combs = list(itertools.combinations(dots, 2))
    valid_combs: list[tuple[tuple[tuple[int, int], ...], ...]] = []

    for i in range(len(all_combs)):
        for j in range(i + 1, len(all_combs)):
            pair1 = all_combs[i]
            pair2 = all_combs[j]

            # Check if the pairs have no common elements
            if not set(pair1).intersection(pair2):
                valid_combs.append((pair1, pair2))

    is_parallel: bool = False
    for p12, p34 in valid_combs:
        if (p12[1][1] - p12[0][1]) * (p34[1][0] - p34[0][0]) == (
            p34[1][1] - p34[0][1]
        ) * (p12[1][0] - p12[0][0]):
            is_parallel = True
            break
    return int(is_parallel)


def solution_120871(n: int) -> int:
    """💤 저주의 숫자 3 ; https://school.programmers.co.kr/learn/courses/30/lessons/120871"""
    # find n-th number in "3x town" radix
    i_3x = 0
    for _ in range(n):
        i_3x += 1
        while i_3x % 3 == 0 or "3" in str(i_3x):
            i_3x += 1
    return i_3x


def solution_120869(spell: str, dic: list[str]) -> int:
    """외계어 사전 ; https://school.programmers.co.kr/learn/courses/30/lessons/120869"""
    from collections import Counter

    c = Counter(spell)
    return 1 if any((c == Counter(word) for word in dic)) else 2


def solution_120868(sides: list[int]) -> int:
    """🧠 삼각형의 완성조건 (2) ; https://school.programmers.co.kr/learn/courses/30/lessons/120868
    if b > a:
        - a+b > c and c > b;    range(b+1, a+b);    a+b-(b+1) == a-1
        - c > b-a and c <= b;   range(b-a+1, b+1);  b+1-(b-a+1) == a
            🚣 c can be b, and Two cases have exclusive range.
    = 2a - 1
    """
    return sides[sides[1] < sides[0]] * 2 - 1


def solution_120866(board: list[list[int]]) -> int:
    """💤 안전지대 ; https://school.programmers.co.kr/learn/courses/30/lessons/120866"""
    import itertools

    n = len(board)
    for i in range(n):
        for j in range(n):
            if board[i][j] == 1:
                board[i][j] = 2
                for nx, ny in (
                    (i + dx, j + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                ):
                    if 0 <= nx < n and 0 <= ny < n and board[nx][ny] == 0:
                        board[nx][ny] = 2

    return sum((0 if x > 0 else 1 for x in itertools.chain.from_iterable(board)))


def solution_120864(my_string: str) -> int:
    """💤 숨어있는 숫자의 덧셈 (2) ; https://school.programmers.co.kr/learn/courses/30/lessons/120864"""
    return sum(
        map(int, "".join((s if s.isdigit() else " " for s in my_string)).split())
    )


def solution_120863(polynomial: str) -> str:
    """💤 다항식 더하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120863"""
    x_coefficient: int = 0
    constant: int = 0
    for term in polynomial.split(" + "):
        if term[-1] == "x":
            x_coefficient += 1 if term[0] == "x" else int(term[:-1])

        else:
            constant += int(term)

    result: list[str] = []
    if x_coefficient > 1:
        result.append(f"{x_coefficient}x")
    elif x_coefficient == 1:
        result.append("x")
    if constant > 0:
        result.append(str(constant))

    return " + ".join(result)


def solution_120862(numbers: list[int]) -> int:
    """최댓값 만들기 (2) ; https://school.programmers.co.kr/learn/courses/30/lessons/120862"""
    numbers.sort()
    return max(numbers[0] * numbers[1], numbers[-1] * numbers[-2])


def solution_120861(keyinput: list[str], board: list[int]) -> list[int]:
    """캐릭터의 좌표 ; https://school.programmers.co.kr/learn/courses/30/lessons/120861"""
    key_map = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
    # board: n (left and right), m (up and down); odd
    n, m = board[0] // 2, board[1] // 2
    p: list[int] = [0, 0]
    for d in (key_map[key] for key in keyinput):
        new_p: list[int] = [p[0] + d[0], p[1] + d[1]]
        if -n <= new_p[0] <= n and -m <= new_p[1] <= m:
            p = new_p
    return p


def solution_120860(dots: list[list[int]]) -> int:
    """💤 직사각형 넓이 구하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120860"""
    max_p, min_p = max(dots), min(dots)
    return (max_p[0] - min_p[0]) * (max_p[1] - min_p[1])


def solution_120854(strlist: list[str]) -> list[int]:
    """배열 원소의 길이 ; https://school.programmers.co.kr/learn/courses/30/lessons/120854"""
    return list(map(len, strlist))


def solution_120853(s: str) -> int:
    """컨트롤 제트 ; https://school.programmers.co.kr/learn/courses/30/lessons/120853"""
    total: int = 0
    is_z: bool = False
    for ss in reversed(s.split()):
        if ss == "Z":
            is_z = True
        else:
            if is_z:
                is_z = False
            else:
                total += int(ss)

    return total


def solution_120852(n: int) -> list[int]:
    """소인수분해 ; https://school.programmers.co.kr/learn/courses/30/lessons/120852"""
    divisors: set[int] = set()
    for i in range(2, int(n**0.5) + 1):
        while True:
            q, r = divmod(n, i)
            if r != 0:
                break
            else:
                n = q
                divisors.add(i)
    else:
        if n != 1:
            divisors.add(n)

    return sorted(divisors)


def solution_120851(my_string: str) -> int:
    """💤 숨어있는 숫자의 덧셈 (1) ; https://school.programmers.co.kr/learn/courses/30/lessons/120851"""
    return sum((int(s) if s.isdigit() else 0 for s in my_string))


def solution_120850(my_string: str) -> list[int]:
    """문자열 정렬하기 (1) ; https://school.programmers.co.kr/learn/courses/30/lessons/120850"""
    return sorted((int(s) for s in my_string if s.isdigit()))


def solution_120849(my_string: str) -> str:
    """모음 제거 ; https://school.programmers.co.kr/learn/courses/30/lessons/120849"""
    return "".join((s for s in my_string if s not in "aeiou"))


def solution_120848(n: int) -> int:
    """💤 팩토리얼 ; https://school.programmers.co.kr/learn/courses/30/lessons/120848

    - product of factorial is less than or equal to n
    """
    prod: int = 1
    for i in range(1, n + 1):
        prod *= i
        if prod > n:
            return i - 1
    return n


def solution_120847(numbers: list[int]) -> int:
    """최댓값 만들기(1) ; https://school.programmers.co.kr/learn/courses/30/lessons/120847"""
    numbers.sort()
    return numbers[-1] * numbers[-2]


def solution_120846(n: int) -> int:
    """💤 합성수 찾기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120846

    - 1 과 자기 자신으로는 무조건 나누어지기에 이를 제외하고 나누어 지는 하나의 값만 찾으면 된다.
        전체 약수의 개수를 세기 위한 변수를 만들 필요가 없다.
    """
    return sum(
        (
            any(num % i == 0 for i in range(2, int(num**0.5) + 1))
            for num in range(4, n + 1)
        )
    )


def solution_120845(box: list[int], n: int) -> int:
    """주사위의 개수 ; https://school.programmers.co.kr/learn/courses/30/lessons/120845"""
    import math

    return math.prod((d // n for d in box))


def solution_120844(numbers: list[int], direction: str) -> list[int]:
    """배열로 회전시키기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120844"""
    if direction == "right":
        numbers.insert(0, numbers.pop())
    else:
        numbers.append(numbers.pop(0))
    return numbers


def solution_120843(numbers: list[int], k: int) -> int:
    """공 던지기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120843"""
    return numbers[(2 * (k - 1) + 1) % len(numbers) - 1]


def solution_120842(num_list: list[int], n: int) -> list[list[int]]:
    """2차원으로 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120842"""
    return [num_list[i * n : (i + 1) * n] for i in range(len(num_list) // n)]


def solution_120841(dot: list[int]) -> int:
    """점의 위치 구하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120841"""
    return [[3, 2], [4, 1]][dot[0] > 0][dot[1] > 0]


def solution_120840(balls: int, share: int) -> int:
    """구슬을 나누는 경우의 수 ; https://school.programmers.co.kr/learn/courses/30/lessons/120840"""
    import math

    return math.comb(balls, share)


def solution_120839(rsp: str) -> str:
    """가위 바위 보 ; https://school.programmers.co.kr/learn/courses/30/lessons/120839"""
    win_map = {"2": "0", "0": "5", "5": "2"}
    return "".join((win_map[x] for x in rsp))


def solution_120838(letter: str) -> str:
    """모스 부호 (1) ; https://school.programmers.co.kr/learn/courses/30/lessons/120838"""
    # before running, copy and paste characters map from the problem site.
    morse: dict[str, str] = {}
    return "".join((morse[l] for l in letter.split()))


def solution_120837(hp: int) -> int:
    """개미 군단 ; https://school.programmers.co.kr/learn/courses/30/lessons/120837"""
    count = 0
    for atk_power in (5, 3, 1):
        q, hp = divmod(hp, atk_power)
        count += q
        if hp == 0:
            break
    return count


def solution_120836(n: int) -> int:
    """순서쌍의 개수 ; https://school.programmers.co.kr/learn/courses/30/lessons/120836"""
    count: int = 0
    for i in range(1, int(n**0.5) + 1):
        q, r = divmod(n, i)
        if r == 0:
            if q != i:
                count += 2
            else:
                count += 1
    return count


def solution_120835(emergency: list[int]) -> list[int]:
    """진료순서 정하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120835"""
    sorted_emergency = sorted(emergency, reverse=True)
    ranks = {}
    for i, x in enumerate(sorted_emergency, start=1):
        ranks[x] = i
    return [ranks[x] for x in emergency]


def solution_120834(age: int) -> str:
    """외계행성의 나이 ; https://school.programmers.co.kr/learn/courses/30/lessons/120834"""
    num_map = {str(i): s for i, s in enumerate("abcdefghij")}
    return "".join((num_map[s] for s in str(age)))


def solution_120833(numbers: list[int], num1: int, num2: int) -> list[int]:
    """배열 자르기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120833"""
    return numbers[num1 : num2 + 1]


def solution_120831(n: int) -> int:
    """짝수의 합 ; https://school.programmers.co.kr/learn/courses/30/lessons/120831"""
    return sum((x for x in range(0, n + 1, 2)))


def solution_120830(n: int, k: int) -> int:
    """💤 양꼬치 ; https://school.programmers.co.kr/learn/courses/30/lessons/120830"""
    return 12000 * n + 2000 * (k - n // 10)


def solution_120829(angle: int) -> int:
    """각도기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120829"""
    q, r = divmod(angle, 90)
    return q * 2 + (r > 0)


def solution_120826(my_string: str, letter: str) -> str:
    """특정 문자 제거하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120826"""
    return "".join((s for s in my_string if s != letter))


def solution_120825(my_string: str, n: int) -> str:
    """문자 반복 출력하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120825"""
    return "".join((s * n for s in my_string))


def solution_120824(num_list: list[int]) -> list[int]:
    """💤 짝수 홀수 개수 ; https://school.programmers.co.kr/learn/courses/30/lessons/120824"""
    even_odd_counts = [0, 0]
    for num in num_list:
        even_odd_counts[num & 1] += 1
    return even_odd_counts


def solution_120823() -> None:
    """직각삼각형 출력하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120823"""
    print("\n".join(("*" * i for i in range(1, int(input()) + 1))))


def solution_120822(my_string: str) -> str:
    """문자열 뒤집기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120822"""
    return "".join(reversed(my_string))


def solution_120821(num_list: list[int]) -> list[int]:
    """배열 뒤집기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120821"""
    return list(reversed(num_list))


def solution_120820(age: int) -> int:
    """나이 출력 ; https://school.programmers.co.kr/learn/courses/30/lessons/120820"""
    return 2022 - (age - 1)


def solution_120819(money: int) -> list[int]:
    """아이스 아메리카노 ; https://school.programmers.co.kr/learn/courses/30/lessons/120819"""
    return list(divmod(money, 5500))


def solution_120818(price: int) -> int | None:
    """💤 옷가게 할인 받기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120818"""
    discount_rates: dict[int, float] = {500000: 0.8, 300000: 0.9, 100000: 0.95, 0: 1}
    for standard_price, discount_rate in discount_rates.items():
        if price >= standard_price:
            return int(price * discount_rate)


def solution_120817(numbers: list[int]) -> float:
    """배열의 평균값 ; https://school.programmers.co.kr/learn/courses/30/lessons/120817"""
    return sum(numbers) / len(numbers)


def solution_120816(slice: int, n: int) -> int:
    """피자 나눠 먹기 (3) ; https://school.programmers.co.kr/learn/courses/30/lessons/120816"""
    import math

    return math.ceil(n / slice)


def solution_120815(n: int) -> int:
    """💤 피자 나눠 먹기 (2) ; https://school.programmers.co.kr/learn/courses/30/lessons/120815"""
    import math

    return n * 6 // math.gcd(n, 6) // 6


def solution_120814(n: int) -> int:
    """피자 나눠 먹기 (1) ; https://school.programmers.co.kr/learn/courses/30/lessons/120814"""
    import math

    return math.ceil(n / 7)


def solution_120813(n: int) -> list[int]:
    """짝수는 싫어요 ; https://school.programmers.co.kr/learn/courses/30/lessons/120813"""
    return list(range(1, n + 1, 2))


def solution_120812(array: list[int]) -> int:
    """💤 최빈값 구하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120812"""
    from collections import Counter

    c = Counter(array)
    if len(c) > 1:  # it is not "len(array) > 1".
        a, b = c.most_common(2)
        return a[0] if a[1] != b[1] else -1
    else:
        return array[0]


def solution_120811(array: list[int]) -> int:
    """중앙값 구하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120811"""
    return sorted(array)[len(array) // 2]


def solution_120810(num1: int, num2: int) -> int:
    """나머지 구하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120810"""
    return num1 % num2


def solution_120809(numbers: list[int]) -> list[int]:
    """배열 두 배 만들기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120809"""
    return [x * 2 for x in numbers]


def solution_120808(numer1: int, denom1: int, numer2: int, denom2: int) -> list[int]:
    """분수의 덧셈 ; https://school.programmers.co.kr/learn/courses/30/lessons/120808"""
    from fractions import Fraction

    result = Fraction(numer1, denom1) + Fraction(numer2, denom2)
    return [result.numerator, result.denominator]


def solution_120807(num1: int, num2: int) -> int:
    """숫자 비교하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120807"""
    return 1 if num1 == num2 else -1


def solution_120806(num1: int, num2: int) -> int:
    """두 수의 나눗셈 ; https://school.programmers.co.kr/learn/courses/30/lessons/120806"""
    return int(num1 / num2 * 1000)


def solution_120805(num1: int, num2: int) -> int:
    """몫 구하기 ; https://school.programmers.co.kr/learn/courses/30/lessons/120805"""
    return num1 // num2


def solution_120804(num1: int, num2: int) -> int:
    """두 수의 곱 ; https://school.programmers.co.kr/learn/courses/30/lessons/120804"""
    return num1 * num2


def solution_120803(num1: int, num2: int) -> int:
    """두 수의 차 ; https://school.programmers.co.kr/learn/courses/30/lessons/120803"""
    return num1 - num2


def solution_120802(num1: int, num2: int) -> int:
    """두 수의 합 ; https://school.programmers.co.kr/learn/courses/30/lessons/120802"""
    return num1 + num2


def solution_120585(array: list[int], height: int) -> int:
    """머쓱이보다 키 큰 사람 ; https://school.programmers.co.kr/learn/courses/30/lessons/120585"""
    return sum((1 if x > height else 0 for x in array))


def solution_120583(array: list[int], n: int) -> int:
    """중복된 숫자 개수 ; https://school.programmers.co.kr/learn/courses/30/lessons/120583"""
    return sum((1 if x == n else 0 for x in array))
