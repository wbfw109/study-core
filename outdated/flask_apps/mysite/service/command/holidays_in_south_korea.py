"""
https://en.wikipedia.org/wiki/Lunar_New_Year
https://en.wikipedia.org/wiki/Korean_New_Year

https://www.naver.com/ is not found Temporary Holiday. so I used https://www.timeanddate.com/holidays/south-korea/.
    but limited from 2000 to 2030 in this url.
"""
from mysite.config import CONFIG_CLASS
import asyncio
import json
import time
import requests
import bs4
from functools import partial
from datetime import datetime
import queue

START_DATE = 2020
END_DATE = 2021

holidays_queue = queue.Queue()


async def append_year_to_json(year: int):
    global holidays_queue
    # preapre
    appended_data: dict = {"year": year}
    holidays: list = []
    url: str = f"https://www.timeanddate.com/holidays/south-korea/{year}?hol=1"

    # preprocess
    loop = asyncio.get_event_loop()
    request = partial(requests.get, url, headers={"Accept-Language": "en-US"})
    response: requests.Response = await loop.run_in_executor(None, request)

    # process
    text = bs4.BeautifulSoup(response.text, "html.parser")
    table_data: bs4.BeautifulSoup = text.find("table", {"id": "holidays-table"})
    th_list: list = [th.text.strip() for th in table_data.find("thead").find_all("th")]

    for tr in table_data.find("tbody"):
        if tr.text == "":
            continue
        temp_date: list = tr.find("th").text.split()
        temp_day = tr.find("td").text
        # %b : Month as locale’s abbreviated name.
        holiday_datetime: datetime = datetime.strptime(
            f"{temp_day}/{temp_date[0]}/{temp_date[1]}", "%A/%d/%b"
        )
        holidays.append(
            {
                "day_of_the_week": holiday_datetime.strftime("%A"),
                "day": holiday_datetime.day,
                "month": holiday_datetime.month,
            }
        )

    appended_data["holidays"] = holidays
    holidays_queue.put(appended_data)


async def main():
    if not CONFIG_CLASS.HOLIDAYS_IN_SOUTH_KOREA_PATH.exists():
        with open(
            str(CONFIG_CLASS.HOLIDAYS_IN_SOUTH_KOREA_PATH), "w", encoding="utf-8"
        ) as file:
            json.dump([], file, ensure_ascii=False, indent=2)

    with open(
        str(CONFIG_CLASS.HOLIDAYS_IN_SOUTH_KOREA_PATH), "r", encoding="utf-8"
    ) as file:
        data: list = json.load(file)
        years: list = [year for year in range(START_DATE, END_DATE + 1)]
        existing_years: list = [item["year"] for item in data]
        required_years: list = list(set(years) - set(existing_years))

    futures = [
        asyncio.ensure_future(append_year_to_json(year)) for year in required_years
    ]

    await asyncio.gather(*futures)


async def put_data():
    global holidays_queue
    with open(
        str(CONFIG_CLASS.HOLIDAYS_IN_SOUTH_KOREA_PATH), "r", encoding="utf-8"
    ) as file:
        data: list = json.load(file)
    while not holidays_queue.empty():
        data.append(holidays_queue.get())
    with open(
        str(CONFIG_CLASS.HOLIDAYS_IN_SOUTH_KOREA_PATH), "w", encoding="utf-8"
    ) as file:
        json.dump(data, file, indent=2)


if __name__ == "__main__":
    start = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    asyncio.run(put_data())
    end = time.time()
    print(f"time taken: {end-start}")
