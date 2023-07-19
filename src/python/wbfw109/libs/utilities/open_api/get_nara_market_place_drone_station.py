"""📝 Before use, set <service_key> and <should_download_attachments> variable !!!
It create <nara_market_place-drone_station>/** for documents.

- <notice_num>s which start with have "EA" few information.
- ❓ the download can be stuck according to time. (by the www.data.go.kr OpenAPI policy?)
- 📝 <one_data_frame.to_excel> requires <openpyxl> package. (It is not required to import this. install only.)
    - .csv file must be loaded by process "Data - From Text/CSV" in Excel to prevent from Encoding corruption.

Background
    - 나라장터의 검색 시스템이 "통합검색"인 경우와 "입찰공고검색" 인 경우의 결과가 다르게 나온다.
        통합검색은 "A B" 로 필드를 채워 검색을 하면 띄어쓰기 기준으로 AND 가 적용되어 두 단어 모두 제목에 존재하는 공고가 우선적으로 나오는 반면
        , 입찰공고검색은 "A B" 로 필드를 채워 검색을 하면 "AB" 가 연속적으로 쓰여진 제목만 검색이 된다.
        때문에 파싱의 루트는 "통합검색" 에서 시작되어야 한다. 하지만 "https://www.g2b.go.kr/pt/" 이후의 맥락은 크롤링이 안된다고 한다.
        ➡️ Microsoft Power automate 으로 "입찰번호" 들만 얻는다.
    - 공공 데이터포탈의 "조달청_입찰공고정보" OpenAPI 중 "입찰공고목록 정보에 대한 e발주 첨부파일정보조회" 라는 Get method 는 있지만 "첨부 파일" 에 대한 OpenAPI 는 없다.
        나라장터의 "통합검색" 을 사용한다해도, 각 입찰공고 페이지를 클릭한 후, "[첨부 파일 ]" 과 "[첨부 파일 (e-발주시스템)]" 에 있는 파일을 개별적으로 다운로드해야 하나.
        내부적으로는 각각 `function dtl_fileDownload(fileSeq, fileName), function eeOrderAttachFileDownload(isMobileYn, eeOrderAttachFileNo, eeOrderAttachFileSeq)` 으로 되어 있다.
        ➡️ OpenAPI 를 사용해 물품/용역별로 입찰공고정보를 호출하고, 각각에 대해 E_FILE 정보를 얻는다. 물품/용역 정보는 응답으로부터 알 수 있다.
    - 동일한 이름, 수요기관의 입찰공고라도 유찰됨으로서 동일한 내용의 입찰 공고가 있을 수 있다. 이러한 것들은 가장 최근 것을 제외한 나머지 공고는 제외해야 한다.
        20230130553  >  [20230117566, 20230110878]
        20220436567  >  [20220402053]
        20210811154  >  [20210727302]

OpenAPI columns
    bidNtceNo: 입찰공고번호
    rgstTyNm: 입찰공고 등록이름
    rgstDt: 입찰공고 게시 날짜
    dminsttNm: 수요 기관
    presmptPrce: 추정 가격 (사업금액 - 부가세)
    bidNtceDtlUrl: 공고 hyperlink
    srvceDivNm: 용역 구분 (["일반용역", "기술용역", ""])
    purchsObjPrdctList: 구매대상 물품ㅣ스트
    (optional)
        prearngPrceDcsnMthdNm: 예가 방법
        sucsfbidMthdNm? cntrctCnclsMthdNm?: 계약 방법
Note
    - prdctSpecNm: 구매대상물품 세부품명 규격은 대상물품은 세트 중 첫번째 제품의 규격만 보내주는 듯 함.
        😠 규격서.hwp 에 나오지 않는 경우, 공고에 수동으로 들어가서 개별적으로 확인 필요.
    - 한국전자통신연구원 공고는 정보가 없음.
"""

# %%
from __future__ import annotations

import json
import ssl
import urllib.request
from http.client import HTTPResponse
from pathlib import Path
from typing import Any

import pandas as pd
from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# %doctest_mode

## settings
working_dir: Path = Path.cwd() / "nara_market_place-drone_station"
working_dir.mkdir(exist_ok=True)
should_download_attachments = False
service_key: str = r""

# %%
# ??? prevent from SSLCertVerificationError
ssl._create_default_https_context = ssl._create_unverified_context

service_notice_nums: list[str] = [
    "20220709655",
    "20210830104",
    "20210811154",
    "EA20212073",
    "EA20211545",
    "EA20201531",
    "20200429229",
]
thing_notice_nums: list[str] = [
    "20230604105",
    "20230601047",
    "20230130553",
    "20221032695",
    "20220909043",
    "20220606725",
    "20220436567",
    "20211211295",
    "20201132933",
]
GET_SERVICE_METHOD_NAME = "getBidPblancListInfoServc01"
GET_THING_METHOD_NAME = "getBidPblancListInfoThng01"
GET_E_FILE_METHOD_NAME: str = "getBidPblancListInfoEorderAtchFileInfo01"
data_frames: list[pd.DataFrame] = []
response: HTTPResponse

## use OpenAPI (GET methods)
for notice_nums, method_name in [
    (service_notice_nums, GET_SERVICE_METHOD_NAME),
    (thing_notice_nums, GET_THING_METHOD_NAME),
]:
    for notice_num in notice_nums:
        print(f"----- notice_num: {notice_num} -----")
        sub_working_dir: Path = working_dir / notice_num
        sub_working_dir.mkdir(exist_ok=True)

        url: str = f"http://apis.data.go.kr/1230000/BidPublicInfoService04/{method_name}?numOfRows=5&pageNo=1&ServiceKey={service_key}&inqryDiv=2&bidNtceNo={notice_num}&type=json"
        with urllib.request.urlopen(url) as response:
            data = response.read().decode("utf-8")
            json_obj = json.loads(data)
        target_obj: dict[str, Any] = json_obj["response"]["body"]["items"][0]
        # If attachments exist, download
        if should_download_attachments:
            for i in range(1, 11):
                doc_url: str = target_obj[f"ntceSpecDocUrl{i}"]
                if not doc_url:
                    break
                doc_name: str = target_obj[f"ntceSpecFileNm{i}"]
                file_path: Path = sub_working_dir / doc_name

                with urllib.request.urlopen(doc_url) as response:
                    content = response.read()
                with file_path.open("wb") as download:
                    download.write(content)
        ## optional data about E_FILES
        additional_url: str = rf"http://apis.data.go.kr/1230000/BidPublicInfoService04/{GET_E_FILE_METHOD_NAME}?numOfRows=5&pageNo=1&ServiceKey={service_key}&inqryDiv=2&bidNtceNo={notice_num}&type=json"
        with urllib.request.urlopen(additional_url) as response:
            data = response.read().decode("utf-8")
            additional_json_obj = json.loads(data)
        if "items" in additional_json_obj["response"]["body"]:
            target_additional_obj: list[dict[str, Any]] = additional_json_obj[
                "response"
            ]["body"]["items"]
            for i, doc_info in enumerate(target_additional_obj, start=1):
                target_obj.update({f"{k}{i}": v for k, v in doc_info.items()})
                if should_download_attachments:
                    # If attachments exist, download
                    doc_url: str = doc_info["eorderAtchFileUrl"]
                    doc_name: str = doc_info["eorderAtchFileNm"]
                    file_path: Path = sub_working_dir / doc_name

                    with urllib.request.urlopen(doc_url) as response:
                        content = response.read()
                    with file_path.open("wb") as download:
                        download.write(content)

        ## add to list <data_frames>
        data_frames.append(pd.DataFrame(json_obj["response"]["body"]["items"]))

## create one data_frame from multiple data_frame
# one_data_frame = pd.concat(data_frames)  # type: ignore
one_data_frame = pd.DataFrame()
one_data_frame.to_excel(
    working_dir / "one_data_frame.xslx", sheet_name="notices", index=False
)

print("===== End Open API calls =====")
