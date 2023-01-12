def parenting_partnering_returns() -> None:
    # test cases
    for case in range(1, int(input()) + 1):
        # C: cameron, J: Jamie. Prioritize job into Cameron
        cameron: int = 0
        jamie: int = 0
        plan: list = []

        # real input
        nString: int = int(input())
        plan = [tuple(map(int, input().split())) for _ in range(nString)]

        ## test values
        # nString = 3
        # plan = [(360, 480), (420, 540), (600, 660)]

        # nString = 3
        # plan = [(0, 1440), (1, 3), (2, 4)]

        # nString = 5
        # plan = [(99, 150), (1, 100), (100, 301), (2, 5), (150, 250)]

        # nString = 6
        # plan = [(99, 150), (1, 100), (300, 500), (400, 600), (150, 250), (150, 250)]

        # additional variable
        planToPerson: dict = {}
        result: list = []
        sortedPlan: dict = {}
        # map data
        for i in range(len(plan)):
            sortedPlan[i] = plan[i]
        sortedPlan: list = sorted(sortedPlan.items(), key=lambda kv: kv[1])

        # allocate
        for i in sortedPlan:
            if i[1][0] >= cameron:
                cameron = i[1][1]
                planToPerson[i[0]] = "C"
            else:
                if i[1][0] >= jamie:
                    jamie = i[1][1]
                    planToPerson[i[0]] = "J"
                else:
                    result.append("IMPOSSIBLE")
                    break

        if result != ["IMPOSSIBLE"]:
            for i in range(nString):
                result.append(planToPerson[i])

        print("Case #{}: {}".format(case, "".join(result)))


parenting_partnering_returns()
"""
Qualification Round
TestCase
    1. 시작시간 순서대로 정렬이라고 하기에는 정렬 안된 상태에 대한 각 작업에 할당을 해야함.
        정렬된 값과 이전의 위치를 비교해서 위치 인덱스를 매핑한 데이터를 딕셔너리에 놓고 알고리즘 작성. 똑같은 개수만큼 작업자 (J or S) 를 나타내는 리스트 하나를 작성하여 그 값을 되돌려줌. collections.OrderedDict 가 편하다.
            - 작업 끝시간과 다음 작업 시작시간을 비교하여 시간이 되면 다음 작업의 끝 시간을 넣는다.

    2. 순서
        C 에게 작업을 우선적으로 할당;

    - from queue import Queue 를 활용하기에는 할당된 값이 무엇인지 확인해야 하기 때문에 list 를 스택처럼 사용하자. 어차피 이전 값을 확인하고 최대 1개까지만 넣을 것이다.
"""
