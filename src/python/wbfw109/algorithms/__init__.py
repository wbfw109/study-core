"""
migrations guides from python >= 3.9 to python 3.7  (tested by pyenv 3.7.15)
    - changed and added contents
        from typing import Literal, Final: python >= 3.8
        Type hint of nested list: python >= 3.9  (error: object is not subscriptable)
        Assign Expressions(:=): python >= 3.8
    - but ㅑf possible, running immediately in my computer is effective rather than cloud by downloading test dataset with sample.
    The worst part is when you pass the sample test but fail the test case.
    - use VS code extension: replacerules.rulesets": {"Remove Type hint"} ...

* for Faster coding
    - Debugging: ipython, exit(), pprint, assert statement
    - not use @property, exception because of time save
"""
