def nesting_depth() -> None:
    # test cases
    for case in range(1, int(input()) + 1):
        # real input
        stringOfDigits: str = str(input())
        newStringOfDigits: str = []
        addedString: list = []

        # if i == 0
        addedString.append("(" * int(stringOfDigits[0]))
        # if i != 0 and i != len(stringOfDigits)-1
        for i in range(0, len(stringOfDigits) - 1, 1):
            former: int = int(stringOfDigits[i])
            latter: int = int(stringOfDigits[i + 1])
            absValue: int = abs(former - latter)
            if former > latter:
                addedString.append(")" * absValue)
            elif former == latter:
                addedString.append("")
            else:
                addedString.append("(" * absValue)
        # if i == len(stringOfDigits)-1
        addedString.append(")" * int(stringOfDigits[len(stringOfDigits) - 1]))

        # sumString
        for i in range(len(stringOfDigits)):
            newStringOfDigits += addedString[i]
            newStringOfDigits += stringOfDigits[i]
        newStringOfDigits += addedString[len(stringOfDigits)]

        print("Case #{}: {}".format(case, "".join(newStringOfDigits)))

        # test values for 5 testcase: "0000", "101", "1110000", "1", "1524434"

        # # Test Set 2
        # x =  "".join([int(x) * "(" + x + ")" * int(x) for x in str(input())])
        # for _ in range(9):
        #     x = x.replace(")(", "")
        # print("Case #{}: {}".format(case, x))


nesting_depth()
"""
Qualification Round
TestCase 1
    We can use the following trick to simplify the implementation: prepend and append one extra 0 to S. Then the implementation is just replacing 01 with 0(1 and 10 with 1)0, which can be written in one line of code in some programming languages. Don't forget to remove the extra 0s from the end of the resulting string!
        print(
            "Case #{}: {}".format(
                case,
                ("0" + "".join([x for x in str(input())]) + "0")
                .replace("01", "0(1")
                .replace("10", "1)0")[1:-1],
            )
        )
TestCase 2
    1. 하나의 배열을 더 이용하기; 알고리즘 추측
        a. 측면이 엔드포인트인가?
            b. 측면의 값이 같은가?
                측면의 자리수와 함께 하나로 취급
                ca. 측면의 값이 더 작은가?
                    큰 값 - 작은 값 만큼 큰 값과 작은 값 사이에 큰 값의 괄호를 닫도록 적용.
                cb.측면의 값이 더 큰가?
                    큰 값 - 작은 값 만큼, 큰 값과 작은 값 사이에 큰 값의 괄호를 닫도록 적용.
        - 좌우에 따라 달라지지 않음.
        - a  0  b  1  c  2  d  1  와 같이 양 끝과 숫자 사이에 포인터를 넣는다.
            (n-1) 개 만큼 비교하고 양 끝점을 합해서 총 (n+1) 개의 포인터의 문자열을 더해야 한다. range 는 마지막 값은 exclusive 라는 것에 주의.
    2. ★ An inefficient? but fun solution
        The problem can be solved using only string replacements. First, replace each digit D with D (s, then the digit itself, then D )s. Then eliminate all instances of )(, collapsing the string each time, until there are no more to remove.

"""
