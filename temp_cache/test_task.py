# written in Python 3.11.1 environment with only built-in library

from __future__ import annotations

import logging
import random
import time
import unittest
from pprint import pprint
from typing import Iterator, Optional


def exchange_product(
    input_lines: Optional[Iterator[str]] = None, is_debugging: bool = False
) -> str:
    """https://sharetreats.notion.site/5ada2a56cf9c43ad8eca806ed129a260

    1. 상품 코드는 9개의 숫자 문자열로 구성된 총 20개를 개발자가 임의로 제공합니다.
    상품 코드는 문자열 Array 또는 파일이든 어떠한 형태로 제공이 되어도 관계 없습니다.

    2. 발생할 수 있는 상황들에 대한 테스트 케이스가 필요합니다.
    각 케이스 별 고객의 입력과 개발자가 예상하는 결과를 이용하여 모든 테스트를 통과해야 합니다.
    Requirements from Business team
        - product code := "\\d{9}"
        - store code := "[A-Za-z ]+"
        - Command :=
            CHECK (상품 교환여부 확인)
            HELP (사용법 안내)
            CLAIM (상품 교환)

    Consideration
        - `상품 교환을 할 때는 어떤 상점에서 교환하였는지 상점 코드를 알아야 합니다.`
            - (상점 코드, 상품 코드)의 유효성 확인 필요.
        - `2. 상품 코드는 10개가 준비되면 고객에게 10개까지만 제공됩니다.`
            ➡️ 고객은 고객에게 제공된 10개 이외의 상품 코드에 대한 권한이 없다고 가정하고 구현한다.
        - Command
            - Command 는 대/소문자 여부를 가리지 않음.
            - 사용자가 유효한 Command 를 지정했는지 확인 필요.
            - 사용자가 유효한 Command 이후, 필수적인 인수를 입력했는지 확인 필요.

    ❓ Collisions of Requirements
        상품 코드 개수
            - `2. 상품 코드는 10개가 준비되면 고객에게 10개까지만 제공됩니다.` from `[ 비지니스 팀 요구사항 ]`
                ??? "10개가 준비되면" ???
            - `1. 상품 코드는 9개의 숫자 문자열로 구성된 총 20개를 개발자가 임의로 제공합니다. ` from `[ 개발팀 요구사항 ]`

    ❔ 상점 코드에 대한 내

    TODO: 상점 코드 안내, 코드는 A~Z,a~z 까지의 대,소 영문자만 사용이 가능하며 6문자로 이루어져 있습니다.
        - 고객이 상품 교환을 요구하면 가능한지 여부와 교환 결과를 안내해 주세요.
        발생할 수 있는 상황 ? 엣지 케이스가 뭐가 있지 -> 디버깅 모드 시, 상품 코드 랜덤이 아니라 직접 조정.
    """

    if input_lines:
        input_ = lambda: next(input_lines)
    else:
        input_ = input

    ## PRODUCT CODE
    PRODUCT_CODE_FORM: str = "0123456789"
    PRODUCT_CODE_SIZE: int = 9
    PRODUCT_CODE_COUNT: int = 20
    # product_code_pool: dict[product_code, store_code in which it used ]
    product_code_pool: dict[str, str] = {}
    while len(product_code_pool) < PRODUCT_CODE_COUNT:
        product_code_pool[
            "".join(random.choice(PRODUCT_CODE_FORM) for _ in range(PRODUCT_CODE_SIZE))
        ] = ""

    ## CUSTOMER CODE
    CUSTOMER_COUPON_COUNT: int = 10
    customer_coupon: list[str] = [
        x for x, _ in zip(product_code_pool, range(CUSTOMER_COUPON_COUNT))
    ]

    ## HELP MESSAGE
    def print_help_message() -> None:
        print(
            "\n".join(
                [
                    "===== 사용 가능한 입력 형식 =====",
                    "CHECK [상품코드]",
                    "HELP",
                    "CLAIM <상점 코드> [상품 코드]",
                    "",
                    "===== 사용자가 소유한 상품 코드 =====",
                    " ".join(
                        (
                            f"{x} ({product_code_pool[x]})"
                            if product_code_pool[x]
                            else x
                            for x in customer_coupon
                        )
                    ),
                ]
            )
        )
        if is_debugging:
            print(
                "\n".join(
                    [
                        "",
                        "===== 전체 상품 코드 =====",
                        " ".join((f"{k} ({v})" for k, v in product_code_pool.items())),
                    ]
                )
            )

    ## process ~
    def check_permission(
        customer_coupon: list[str], /, *, product_codes: list[str]
    ) -> list[str]:
        """Check whether customer have own coupon codes.

        Returns:
            list[str]: valid <product_codes> after checking customer's access permission
        """
        valid_product_codes: list[str] = []
        for product_code in product_codes:
            if product_code not in customer_coupon:
                # Use lazy % formatting in logging functions Pylint(W1201:logging-not-lazy)
                logging.warning("[client] ⚠️ 상품 코드 %s 에 대한 접근 권한이 없습니다.", product_code)
            else:
                valid_product_codes.append(product_code)
        return valid_product_codes

    def process_product_codes(
        customer_coupon: list[str],
        product_codes: list[str],
        store_code: Optional[str] = None,
    ):
        """
        If <store_code> exists, run function <claim_customer_coupons> otherwise, run function <check_product_codes>

        Args:
            customer_coupon (list[str]): _description_
            product_codes (list[str]): _description_
            store_code (Optional[str], optional): Defaults to None.
        """
        product_codes = check_permission(customer_coupon, product_codes=product_codes)
        for product_code in product_codes:
            if product_code in product_code_pool:
                if product_code_pool[product_code]:
                    print("이미 사용된 상품 코드입니다.")
                else:
                    if store_code is None:
                        print("사용 가능한 상품 코드입니다.")
                    else:
                        product_code_pool[product_code] = store_code
                        print("사용되었습니다.")
            else:
                logging.warning("[client] ⚠️ 유효하지 않은 접근입니다")

    def check_product_codes(customer_coupon: list[str], product_codes: list[str]):
        process_product_codes(customer_coupon, product_codes)

    def claim_customer_coupons(
        customer_coupon: list[str], product_codes: list[str], store_code: str
    ):
        process_product_codes(customer_coupon, product_codes, store_code)

    try:
        while True:
            command, *arguments = input_().split()
            command = command.upper()
            if command == "CHECK":
                product_codes = arguments
                check_product_codes(customer_coupon, product_codes=product_codes)
            elif command == "CLAIM":
                store_code = arguments[0]
                product_codes = arguments[1:]
                claim_customer_coupons(
                    customer_coupon, product_codes=product_codes, store_code=store_code
                )
            else:
                print_help_message()
    except (KeyboardInterrupt, StopIteration):
        pass
    return ""


def test_exchange_product() -> None:
    test_case = unittest.TestCase()
    for input_lines, output_lines in [
        [
            ["Check 132 122", "HELP", "ClaiM ABcde 1231"],
            [""],
        ]
    ]:
        start_time = time.time()
        test_case.assertEqual(
            exchange_product(iter(input_lines), is_debugging=True),
            "\n".join(output_lines),
        )
        print(f"elapsed time: {time.time() - start_time}")


test_exchange_product()
# exchange_product(is_debugging=True)
