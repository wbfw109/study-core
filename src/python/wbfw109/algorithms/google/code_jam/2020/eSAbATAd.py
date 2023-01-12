import sys
import collections
import math


class ESAbATAd:
    MAX_QUERIES: int = 150
    REQUIRED_COUNT_FOR_GUESS: int = 4
    reversed_bit: dict = {"0": "1", "1": "0"}

    def __init__(self, bits: int) -> None:
        self.bits: int = bits
        self.position_bits: dict = {}
        self.state: list = []
        self.result: list = []
        self.is_asymmetry_bits: bool = False
        self.asymmetry_bit_number: int = 0
        self.asymmetry_bit: str = ""
        self.is_symmetry_bits: bool = False
        self.symmetry_bit_number: int = 0
        self.symmetry_bit: str = ""

    def query(self, bit: int) -> str:
        print(bit, flush=True)
        response: str = input()
        return response

    def fluctuate(self, start_apply_index: int, bits_pairs: list) -> list:
        state_number: dict = collections.Counter(self.state[start_apply_index:])
        state_c: int = (state_number["cr"] + state_number["c"]) % 2
        state_r: int = (state_number["cr"] + state_number["r"]) % 2

        new_bit_pairs: list = []
        for bits in bits_pairs:
            new_bits: list = bits
            for index in range(len(new_bits)):
                new_bits[index] = str((int(new_bits[index]) + state_c) % 2)
            if state_r == 1:
                new_bits[0], new_bits[1] = new_bits[1], new_bits[0]
            new_bit_pairs.extend(new_bits)

        return new_bit_pairs

    def guess_state(self) -> str:
        # c: complementation, r: reversal, cr: reversal plus complementation, etc: nothing or unknown
        new_asymmetry_bit: str = self.query(self.asymmetry_bit_number)
        new_symmetry_bit: str = self.query(self.symmetry_bit_number)
        new_asymmetry_not_bit: str = ESAbATAd.reversed_bit[new_asymmetry_bit]
        new_symmetry_not_bit: str = ESAbATAd.reversed_bit[new_symmetry_bit]
        if (
            new_asymmetry_not_bit == self.asymmetry_bit
            and new_symmetry_not_bit == self.symmetry_bit
        ):
            self.asymmetry_bit = new_asymmetry_bit
            self.symmetry_bit = new_symmetry_bit
            return "c"
        elif (
            new_asymmetry_not_bit != self.asymmetry_bit
            and new_symmetry_not_bit == self.symmetry_bit
        ):
            self.symmetry_bit = new_symmetry_bit
            return "cr"
        elif (
            new_asymmetry_not_bit == self.asymmetry_bit
            and new_symmetry_not_bit != self.symmetry_bit
        ):
            self.asymmetry_bit = new_asymmetry_bit
            return "r"
        else:
            # if new_asymmetry_not_bit != bit_not_asymmetry_bit and not_ != bit_not_symmetry_bit:
            return "x"

    def guess_state_only_asymmetry(self) -> str:
        new_asymmetry_not_bit: str = self.query(self.asymmetry_bit_number)
        # waste one query
        self.query(self.asymmetry_bit_number)
        bit_not_asymmetry_bit: str = ESAbATAd.reversed_bit[new_asymmetry_not_bit]
        if new_asymmetry_not_bit == bit_not_asymmetry_bit:
            # "c" or "r"
            self.asymmetry_bit = new_asymmetry_not_bit
            return "c"
        else:
            # if new_asymmetry_not_bit != bit_not_asymmetry_bit:
            # "cr" or "etc". etc 가 아무작업도 하지 않으므로 더 이득이다.
            return "x"

    def guess_state_only_symmetry(self) -> str:
        not_: str = self.query(self.symmetry_bit_number)
        # waste one query
        self.query(self.symmetry_bit_number)
        bit_not_symmetry_bit: str = ESAbATAd.reversed_bit[not_]
        if not_ == bit_not_symmetry_bit:
            # "c" or "cr". c 가 한 번만 작업하므로 더 이득이다.
            self.symmetry_bit = not_
            return "c"
        else:
            # if not_ != bit_not_symmetry_bit:
            # "r" or "etc"
            return "x"

    def set_bit_state(self, number_a: int, number_b: int) -> None:
        if not self.is_asymmetry_bits:
            if self.position_bits[number_a] != self.position_bits[number_b]:
                self.asymmetry_bit_number = number_a
                self.asymmetry_bit = self.position_bits[number_a]
                self.is_asymmetry_bits = True
        if not self.is_symmetry_bits:
            if self.position_bits[number_a] == self.position_bits[number_b]:
                self.symmetry_bit_number = number_a
                self.symmetry_bit = self.position_bits[number_a]
                self.is_symmetry_bits = True

    def guess_answer(self):
        is_end_dict: bool = False
        # first 10 queries
        new_bit_pairs: list = self.fluctuate(
            0,
            [
                [
                    self.position_bits[position],
                    self.position_bits[self.bits - position + 1],
                ]
                for position in range(1, ESAbATAd.REQUIRED_COUNT_FOR_GUESS + 2)
            ],
        )
        # update dict
        for number in range(1, ESAbATAd.REQUIRED_COUNT_FOR_GUESS + 2):
            start_number: int = number
            symmetry_number: int = self.bits - start_number + 1
            self.position_bits[start_number] = new_bit_pairs[number * 2 - 2]
            self.position_bits[symmetry_number] = new_bit_pairs[number * 2 - 1]

        # rest of queries
        for loop_index in range(int(ESAbATAd.MAX_QUERIES / 10) - 1):
            temp_list: list = []
            for number in range(1, ESAbATAd.REQUIRED_COUNT_FOR_GUESS + 1):
                start_number: int = (
                    ESAbATAd.REQUIRED_COUNT_FOR_GUESS
                    + 1
                    + loop_index * ESAbATAd.REQUIRED_COUNT_FOR_GUESS
                    + number
                )
                symmetry_number: int = self.bits - start_number + 1
                if start_number > self.bits / 2:
                    break
                temp_list.append(
                    [
                        self.position_bits[start_number],
                        self.position_bits[symmetry_number],
                    ]
                )

            new_bit_pairs: list = self.fluctuate(loop_index + 1, temp_list)

            # update dict
            for number in range(1, ESAbATAd.REQUIRED_COUNT_FOR_GUESS + 1):
                start_number: int = (
                    ESAbATAd.REQUIRED_COUNT_FOR_GUESS
                    + 1
                    + loop_index * ESAbATAd.REQUIRED_COUNT_FOR_GUESS
                    + number
                )
                symmetry_number: int = self.bits - start_number + 1
                if start_number > self.bits / 2:
                    is_end_dict = True
                    break
                self.position_bits[start_number] = new_bit_pairs[number * 2 - 2]
                self.position_bits[symmetry_number] = new_bit_pairs[number * 2 - 1]

            if is_end_dict:
                break

        for number in range(1, len(self.position_bits) + 1):
            self.result.append(self.position_bits[number])

        print("".join(self.result), flush=True)
        if input() == "N":
            sys.exit()

    def run(self):
        is_end_position: bool = False
        # first 10 queries
        for number in range(1, ESAbATAd.REQUIRED_COUNT_FOR_GUESS + 2):
            start_number: int = number
            symmetry_number: int = self.bits - start_number + 1
            self.position_bits[start_number] = self.query(start_number)
            self.position_bits[symmetry_number] = self.query(symmetry_number)
            self.set_bit_state(start_number, symmetry_number)

        # rest of queries
        for loop_index in range(int(ESAbATAd.MAX_QUERIES / 10) - 1):
            # 10*x + 1  ~  10*x + 2 th bit
            if self.is_asymmetry_bits and self.is_symmetry_bits:
                self.state.append(self.guess_state())
            elif not self.is_asymmetry_bits and self.is_symmetry_bits:
                self.state.append(self.guess_state_only_asymmetry())
            else:
                # if self.is_asymmetry_bits and not self.is_symmetry_bits:
                self.state.append(self.guess_state_only_symmetry())

            #! Input position is out of range. trace 방법을 알아야 함...
            #! 문법오류보다 로직오류가 진짜 찾기 힘듬;;; input 이니까 query 어딘가에서 오류 발생.. N이 홀수일때 발생하는 문제인가?
            #! Trace 기능이 있는건가? 다른 파이썬 파일로부터 input이 무엇인지 trace
            # 10*x + 3  ~  10*x + 10 th bit
            for number in range(1, ESAbATAd.REQUIRED_COUNT_FOR_GUESS + 1):
                if not is_end_position:
                    start_number: int = (
                        ESAbATAd.REQUIRED_COUNT_FOR_GUESS
                        + 1
                        + loop_index * ESAbATAd.REQUIRED_COUNT_FOR_GUESS
                        + number
                    )
                    symmetry_number: int = self.bits - start_number + 1
                    # if invalid position (index out of range) before follwing queries
                    if start_number > int(self.bits / 2):
                        is_end_position = True
                        # waste queries
                        self.query(1)
                        self.query(1)
                        continue

                    self.position_bits[start_number] = self.query(start_number)
                    self.position_bits[symmetry_number] = self.query(symmetry_number)
                    self.set_bit_state(start_number, symmetry_number)
                else:
                    # waste queries
                    self.query(1)
                    self.query(1)

        self.guess_answer()


testCase, bits = map(int, input().split())
for case in range(testCase):
    eSAbATAd = ESAbATAd(bits)
    eSAbATAd.run()
"""
Qualification Round
    Interactive problem
        $ps: python .\src\algorithm\codejam\interactive_runner.py python .\src\algorithm\codejam\2020\eSAbATAd_testing_tool.py 0 -- python .\src\algorithm\codejam\2020\eSAbATAd.py
    .vscode//lanch.json
        "configurations": [
            {
            "name": "Python: esAbAtAd",
            "type": "python",
            "request": "launch",
            "program": ".\\src\\algorithm\\codejam\\interactive_runner.py",
            "console": "integratedTerminal",
            "args": ["python", ".\\src\\algorithm\\codejam\\2020\\eSAbATAd_testing_tool.py", "0", "--", "python", "${file}"] 
            }
        ]
TestCase
    내가 쿼리를 하고, 상대가 그 값을 보내주면서, 내가 10x + 1 쿼리마다 (x>=1: int) 마다 a ~ d 중 하나의 사항으로 값들이 바뀌고
    , 150번째 쿼리가 끝나고 모든 비트의 값을 추측해서 보내줘야함.
    Output
        a. 25% of the time, the array is complemented: every 0 becomes a 1, and vice versa.
        b. 25% of the time, the array is reversed: the first bit swaps with the last bit, the second bit swaps with the second-to-last bit, and so on.
        c. 25% of the time, both of the things above (complementation and reversal) happen to the array. (Notice that the order in which they happen does not matter.)
        d. 25% of the time, nothing happens to the array.
        Your program outputs one line containing a string of B characters, each of which is 0 or 1, representing the bits currently stored in the array (which will not necessarily match the bits that were initially present!)
    Problem
        하나의 P 번째 비트에 대해 len(P-i) 비트와 대칭일 경우, b, c, d 중에서 무엇이 일어낫는지 비교할 수 없다.
            1001
                a. 0110
                b. 1001
                c. 1001
                d. 1001
        하나의 P 번째 비트에 대해 len(P-i) 비트와 비대칭인 경우, c, d 중에 무엇이 일어낫는지 알 수 있지만 a, b 를 비교할 수 없다.
            1011
                a. 0100
                b. 1101
                c. 0010
                d. 1011
        그래서 비대칭, 대칭인 값을 하나씩 사용하여 값을 비교하여 찾아야 한다. 첫 비교를 시작하면 무조건 비대칭/대칭인 비트가 하나씩은 존재하므로 분기를 3개 만든다.
        하지만 x 번째 까지 쿼리했을 때 무조건 비대칭, 대칭인 값 둘 모두가 존재한다고 볼 수 없다. 때문에 어떤 상태가 적용되었는지 선택할 수 없지만 어떤 것을 적용해도 상관이 없다는 뜻이라는 것을 알아야 한다. 하지만 이를 위한 if 문 분기를 만들긴 해야 한다.
        //
        결과값에는 모든 값의 변동 이후 값을 출력하므로, 10 개 이후 8 개마다 해당 루프의 수만큼 중첩해서 적용한다.
        //
        10번째 이후에는 비교를 위한 쿼리 횟수 2번이 추가적으로 필요하여 14번의 변동상태를 확인을 해야 한다.
        총 8*15 + 4 (117) 개의 위치를 알 수 있다. bit 수가 더 작아 다른 위치를 파악할 필요가 없게 되면 고정된 비교할 2개의 위치만 파악한다.
        * 각 테스트 케이스의 고정된 bits 수는 limit 에 나오지는 않았지만 Test Sets 에 최소값이 10 에 짝수값이므로 10 미만에 대해 제한을 만들 필요는 없는듯 하다.
        150 번째 쿼리 이전에 쿼리를 종료하고 정답을 맞추는 기능은 없기 때문에 10 + 8 + 8.. 중간에 index out of range 가 되면 한 쌍의 쿼리의 남은 한번도 똑같이 하고, dict 에서 값을 제거하고 continue 한다.
        //
        dict 값에 각 위치의 값을 저장하고 나중에 양 끝에서 5번째 비트마다 0 번의 연산을 한 값을 연산해서 MAX_QUERIES 까지 합친다.
        single integer P between 1 and B 로 query 하면 single character 로 응답한다. result 에 저장할 때에는 str 값으로 저장하고 나중에 이를 "".join( ~ ) 한다. 
        //
//
https://stackoverflow.com/questions/15608229/what-does-prints-flush-do

바이너리 변수 사용하기
    처음에 리터럴 값을 0b 를 붙여 바이너리 형태로 저장하거나, bin() 는 str 로 변환하므로 변수를 int( ~ , 2) 형태로 저장해야 비트연산을 할 수 있다.
        - 이전 값과 변화된 값을 비교하기 위해서 다음의 방법을 사용할 수 없다. 컴퓨터는 2의 보수 변환 방식을 취하기 때문에 not 연산을 해봤자이다.
            문자열을 2진수 int로 바꾸고 이를 not 연산한다. 이후 bin( ~ ) 로 문자열 2진수로 만들고 3번째 인덱스이상을 슬라이싱한다. 0이상의 정수는 2의 보수를 취하면 모두 음수가 되기 때문에 2번재부터가 아닌 3번째부터이다.
                a = "1"
                x = bin(~int(a, 2))[3:]
            결국 dict 를 사용하여 {"0": "1", 1": "0"} 바꾸는 방법이 가장 좋다.
            dict를 넣은 변경해본 a 비트와 변경된 b 비트 값을 비교하면 어떤 연산이 적용되었는지 알 수 있다.
"""
