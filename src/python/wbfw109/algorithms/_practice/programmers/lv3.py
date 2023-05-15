"""Purpose is to solve in 10 minutes
- 💤: 문제 이해 제대로 하기
    "문제 설명"에 모든 제한사항이 함게 주어지지 않는 경우가 있어서, 완벽히 이해하기 어려울 수 있다.
    에매한 경우, 명확하게 이해할 수 있도록 "예시" 가 주어지므로 꼭 함께 보도록 한다.
- 🧠: (신박한 풀이, 수학식을 활용한 풀이, Benchmark)
- 💦: built-in functions, grammar

"""
# Regex to find problems in which one of upper emojis was used; """[^\uAC00-\uD7A3\d\w]\s


def solution_12927(n: int, works: list[int]) -> int:
    """야근 지수 ; https://school.programmers.co.kr/learn/courses/30/lessons/12927
    Tag: Data sturctures
    """
    from collections import Counter

    counter = Counter(works)
    for i in range(max(works), 0, -1):
        count = counter[i]
        x = count if count <= n else n

        if count == x:
            del counter[i]
        else:
            counter[i] -= x
        n -= x
        counter[i - 1] += x

        if n == 0:
            break
    return sum((k**2 * v for k, v in counter.items()))
