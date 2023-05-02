"""Purpose is to solve in 4 minutes
- 💤: 문제 이해 제대로 하기
    "문제 설명"에 모든 제한사항이 함게 주어지지 않는 경우가 있어서, 완벽히 이해하기 어려울 수 있다.
    에매한 경우, 명확하게 이해할 수 있도록 "예시" 가 주어지므로 꼭 함께 보도록 한다.
- 🧠: (신박한 풀이, 수학식을 활용한 풀이, Benchmark)
- 💦: built-in functions, grammar

"""
# Regex to find problems in which one of upper emojis was used; """[^\uAC00-\uD7A3\d\w]\s


# TODO: write additioanl descriptions this problems.
def solution_133499(babbling: list[str]) -> int:
    """🧠 옹알이 (2) ; https://school.programmers.co.kr/learn/courses/30/lessons/133499

    - `len(word.strip())==0` is slightly faster than `if not word.split():`
    - `for s in valid_str:  word = word.replace(s, " ")` is slightly faster than
        `if len(functools.reduce(lambda w, s: w.replace(s, " "), valid_str, word).strip()) == 0:   count += 1`
        , because of function call overhead.

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
