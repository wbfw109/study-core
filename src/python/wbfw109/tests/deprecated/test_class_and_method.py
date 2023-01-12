# %%
from __future__ import annotations

import logging
import os
from typing import Any

import IPython
from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

#%%
# Title: # cls variable
class AAA:
    value = 3

    @classmethod
    def foo(cls):
        cls.value = 5


class BBB(AAA):
    @classmethod
    def foo(cls):
        cls.value = 10


AAA.value
AAA.foo()
AAA.value
BBB.value


#%%
# Title: # class inheritance with class variable

AHI = 1


class Config(object):
    DEBUG = False
    TESTING = False
    SECRET_KEY = b"a\x16s$c{'@\r\xf1[~x\xd85\xf2"
    SESSION_COOKIE_SECURE = True

    @staticmethod
    def foo():
        print(Config.DEBUG)

    @classmethod
    def bar(cls):
        print(cls.DEBUG)


class DevelopmentConfig(Config):
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    TEMPLATES_AUTO_RELOAD = True


if AHI == 1:
    BCA = DevelopmentConfig
    BCA.foo()
    print()
    BCA.bar()


print("\n\nclass variable would be inherited.")

#%%


#%%
# Title: # call by sharing
def f(a_list: list[Any]):
    a_list = a_list + [1]


m = []
f(m)


def f2(a_list: list[Any]):
    a_list.append(1)


m2 = []
f2(m2)
print(m2)
