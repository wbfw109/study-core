# contents of wallet.py
from __future__ import annotations
import dataclasses


@dataclasses.dataclass
class Wallet:
    verified: bool

    amount_eur: int
    amount_usd: int
    amount_gbp: int
    amount_jpy: int


# @dataclasses.dataclass
# class Wallet:
#     _meta: Wallet = dataclasses.field(default=None)
#     verified: bool = dataclasses.field(default=None)

#     amount_eur: int = dataclasses.field(default=None)
#     amount_usd: int = dataclasses.field(default=None)
#     amount_gbp: int = dataclasses.field(default=None)
#     amount_jpy: int = dataclasses.field(default=None)

#     def __post_init__(self):
#         self._meta = type(Wallet)
