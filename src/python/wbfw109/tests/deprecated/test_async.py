"""
https://www.python.org/dev/peps/pep-0492/#await-expression
    https://www.python.org/dev/peps/pep-0492/#examples-of-await-expressions
https://www.python.org/dev/peps/pep-0492/#asynchronous-context-managers-and-async-with
https://www.python.org/dev/peps/pep-0492/#asynchronous-iterators-and-async-for

https://www.python.org/dev/peps/pep-3148/#future-objects
"""

from time import time
from urllib.request import Request, urlopen
import asyncio

urls = [
    "https://www.google.co.kr/search?q=" + i
    for i in ["apple", "pear", "grape", "pineapple", "orange", "strawberry"]
]
"""
blocking function
    e.g. urlopen, response.read
네이티브 코루틴 안에서 블로킹 I/O 함수를 실행하려면 이벤트 루프의 run_in_executor 함수를 사용하여 다른 스레드에서 병렬로 실행시켜야 한다.
"""


async def fetch(url):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})  # UA가 없으면 403 에러 발생
    response = await loop.run_in_executor(None, urlopen, request)
    page = await loop.run_in_executor(None, response.read)
    return len(page)


async def main():
    futures = [asyncio.ensure_future(fetch(url)) for url in urls]
    # 태스크(퓨처) 객체를 리스트로 만듦
    result = await asyncio.gather(*futures)  # 결과를 한꺼번에 가져옴
    print(result)


begin = time()
loop = asyncio.get_event_loop()  # 이벤트 루프를 얻음
loop.run_until_complete(main())  # main이 끝날 때까지 기다림
loop.close()  # 이벤트 루프를 닫음
end = time()
print("실행 시간: {0:.3f}초".format(end - begin))
