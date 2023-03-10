"""
migrations guides from python >= 3.9 to python 3.7  (tested by pyenv 3.7.15)
    - changed and added contents
        from typing import Literal, Final: python >= 3.8
        Type hint of nested list: python >= 3.9  (error: object is not subscriptable)
        Assign Expressions(:=): python >= 3.8
    - but If possible, running immediately in my computer is effective rather than cloud by downloading test dataset with sample.
    The worst part is when you pass the sample test but fail the test case.
    - use VS code extension: replacerules.rulesets": {"Remove Type hint"} ...


* Complexity
    affected including: sorted()
* for Faster coding
    - Debugging: ipython, exit(), pprint, assert statement
    - not use @property, exception because of time save
    - use (dataclasses | NamedTuple)
    
        
๐ฐ~
* Tip
    - ๋ณด๊ณ ์ ๊ฐ๋ฅํ๋ฉด ์์ ๋จ์๋ถํฐ ๋ณ์ํํ๊ณ  ๋ ํฐ ๋จ์๊ฐ ์์ผ๋ฉด ๋ณ์ํํ์.
    - ๊ธฐ์ค์ ์ ์ก๊ณ , ๊ฐ๋ฅํ ์๊ณ ๋ฆฌ์ฆ๋ค ์ค ๊ธฐํ๋น์ฉ์ ๋น๊ตํ๋ค.
interactive ๋ช๊ฐ ๋ฌธb์  ํ์ด๋ณด๊ณ , ProblemSolution ์ถ์ํํด๋์ค ์์๋ฐ์์ ์์ฑ ํ์
โ ์๊ณ ๋ฆฌ์ฆ..๋ํ๋ง๊ณ  ๋ฐ๋ก ์๋ฃ๊ตฌ์กฐ๋ง๋ค์ด์ ํ์คํธํ ๊ฒ.. ์๋ฃ๊ตฌ์กฐ ์ด๋ก ๋ณ๋ก ํจํค์ง ์ ๋ฆฌ? number_theory, graph_theory ๋ฌธ์  ๋ณด๊ณ  ์๋ฃ๊ตฌ์กฐ ๋ญ์ธ์ง ๋ ์ฌ๋ ค์ผํจ.
asyncio ํ์คํธ์ฝ๋. ๋ญ๋? ๋ฆฌ์กํธ ํ์์คํฌ๋ฆฝํธ?
๋น๋๊ธฐ ํจ์ - ์ฝ๋ฐฑ. ๋น๋๊ธฐ ํจ์๊ฐ ๋ฆฌํด๊ฐ์ ๋ฐ์ ์์๋ ๋ฐ์์ค๋ ๊ฐ.? ์ด๋ผ๊ณ  ์ ์ํ๋ฉด ๋๋.
Flask, Django ํ  ๋ logging ์ฌ์ฉํ๊ธฐ.
์ด ํ๋ก์ ํธ ์์์ Flask Django ๋ฅผ ๊ณต์กดํ  ์ ์๋? ์ด๊ฑฐ ํด๋ณด๊ณ  CI / CD ์ ์ฉํด๋ณด๊ธฐ
๐ ๋ง์ดํฌ๋ก์๋น์ค ๋จ์? ์ฟ ๋ฒ๋คํฐ์ค ์ฐ๊ณ , Docker ๋ ๊ต์ฒดํด์ผํ๋?
2022 oci docker vs Buildah, Podman, Skopeo`
๐ Flask, Django ์ฉ๋ ์ ํ๊ธฐ
    Flask <-> AI ํต์ .
    Django doesnโt allow individual processes to handle multiple requests simultaneously.
    Developers have to come up with ways to make individual processes handle multiple requests
    - Flask ์์ ํด์ผํ๋, Django ์ ๋ด์ฅ๋์ด์๋ ๊ฒ์๋ค ํด์ผ ํ๋๋ฐ ๊ทธ ์ฒ๋ฆฌํ  ๋ชฉ๋ก์ ๋ชจ๋ฅด๊ธฐ ๋๋ฌธ์ Django ๋ถํฐ ํ๋๊ฒ ๋ซ๋?
    
https://stateofjs.com/en-us/

๋๋ถ๋ถ์ AI ๋ชจ๋ธ์ ํ๋ผ์คํฌ์ ์ด์ธ๋ฆฐ๋ค๊ณ  ํ๋ค.
TpeSCript ํ๋ก์ ํธ?

pylint https://google.github.io/styleguide/pyguide.html ํ์ํ ๊น?
FastAPI?


"""
