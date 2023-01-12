def vestigium() -> None:
    # test cases
    for case in range(1, int(input()) + 1):
        overlapRow: int = 0
        overlapColumn: int = 0
        trace: int = 0

        # real input
        matrixSize: int = int(input())
        matrix: list = [list(map(int, input().split())) for _ in range(matrixSize)]

        # ## test cases input
        # matrixSize = 4
        # matrix = [
        #     [1, 2, 3, 4],
        #     [2, 1, 4, 3],
        #     [3, 4, 1, 2],
        #     [4, 3, 2, 1],
        # ]
        # matrixSize = 4
        # matrix = [
        #     [2, 2, 2, 2],
        #     [2, 3, 2, 3],
        #     [2, 2, 2, 3],
        #     [2, 2, 2, 2],
        # ]
        # matrixSize = 3
        # matrix = [
        #     [2, 1, 3],
        #     [1, 3, 2],
        #     [1, 2, 3],
        # ]

        for i in range(matrixSize):
            trace += matrix[i][i]
            if len(set(matrix[i])) != matrixSize:
                overlapRow += 1
            if len(set([matrix[x][i] for x in range(matrixSize)])) != matrixSize:
                overlapColumn += 1

        print("Case #{}: {} {} {}".format(case, trace, overlapRow, overlapColumn))


vestigium()
"""
Qualification Round
TestCase
    1. 중복된 값을 찾는 방법
        a. ★ Set 함수로 중복된 값이 제거된 원소의 개수를 찾아 비교하기
        b. 배열을 정렬하여 다음 값 -1 == 이전 값 인지 확인하기
        c. 원소의 값을 하나하나 비교하기
"""
