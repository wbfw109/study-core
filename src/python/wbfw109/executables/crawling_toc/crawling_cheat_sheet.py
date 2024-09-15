# %%
# Written in 📅 2024-09-15 05:06:05
# Enable all outputs in the Jupyter notebook environment
import json
from datetime import datetime
from pathlib import Path
from typing import Union

import requests
from bs4 import BeautifulSoup, Tag
from bs4.element import Tag
from IPython.core.interactiveshell import InteractiveShell
from requests.compat import urljoin
from wbfw109.libs.crawling import get_class_list
from wbfw109.libs.file import create_temp_json_file, open_json_in_vscode
from wbfw109.libs.string import slugify

InteractiveShell.ast_node_interactivity = "all"

# %%
from typing import Any

# Set the image directory
images_dir: Path = Path.home() / "crawling" / "images"
images_dir.mkdir(parents=True, exist_ok=True)


def download_image(image_url: str, file_name: str, images_dir: Path) -> None:
    """
    Downloads an image from a URL and saves it to the specified directory.

    Parameters:
        image_url (str): The URL of the image to download.
        file_name (str): The name of the file to save the image as.
        images_dir (Path): The directory where the image will be saved.
    """
    response = requests.get(image_url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses

    # Set the full path to save the image
    image_path: Path = images_dir / file_name

    # Write the image content to the file
    with image_path.open("wb") as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)

    print(f"Downloaded {file_name} to {image_path}")


def parse_json_and_download_images(json_file_path: Path, images_dir: Path) -> None:
    """
    Parses a JSON file to download images based on the stored data.

    Parameters:
        json_file_path (Path): Path to the JSON file that contains image info.
        images_dir (Path): The directory where images will be downloaded.
    """
    # Load existing data from JSON file
    with json_file_path.open("r", encoding="utf-8") as f:
        picture_info: list[dict[str, Any]] = json.load(f)

    # Iterate over each entry and download images
    for item in picture_info:
        file_name: str = item["file_name"]
        image_url: str = item["image_url"]

        # Download the image
        download_image(image_url, file_name, images_dir)


# Example usage:
# Assuming you already have the JSON file generated from the fetch step
json_file_path: Path = Path.home() / "crawling" / "cheat_sheet-cpp.json"

# Parse JSON and download images
parse_json_and_download_images(json_file_path, images_dir)


# %%
from pathlib import Path

# Set the base name for file naming
base_name: str = slugify("cheat_sheet-cpp")

# Load existing data from a temporary JSON file
json_file_path = create_temp_json_file({}, prefix="cheat_sheet-cpp", suffix=".json")

# Check if the file already contains data
try:
    with json_file_path.open("r", encoding="utf-8") as f:
        picture_info: list[dict[str, Union[str, datetime]]] = json.load(f)
        if not isinstance(picture_info, list):
            picture_info = []
except FileNotFoundError:
    picture_info: list[dict[str, Union[str, datetime]]] = []

# Create a set of existing file names to avoid duplicates
existing_file_names: set[str] = set(item["file_name"] for item in picture_info)

# Fetch the web page content using requests
url: str = "https://hackingcpp.com/cpp/cheat_sheets.html"
base_url = "https://hackingcpp.com"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

response = requests.get(url, headers=headers)
response.raise_for_status()
html_content: str = response.text

# Parse the HTML content with BeautifulSoup
soup: BeautifulSoup = BeautifulSoup(html_content, "html.parser")

# Find the main section with class 'main plain'
section: Tag = soup.find("section", class_="main plain")


# %%
def process_image_tags(
    root_tag: Tag,
    base_name: str,
    base_url: str,
    category_1: str,
    category_2: str,
    category_3: str,
    existing_file_names: set[str],
    picture_info: list[dict[str, Union[str, datetime]]],
    description: str = "",
) -> None:
    img_tags = root_tag.find_all("img")
    description = ""
    # for hn_tag in root_tag.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
    #     hn_tag: Tag
    #     hn_tag.extract()
    description = root_tag.get_text(separator=" ", strip=True)

    for img_tag in img_tags:
        img_src = img_tag.get("src") or img_tag.get("data-src")

        if img_src:
            file_path = Path(img_src)
            file_stem, file_suffix = file_path.stem, file_path.suffix
            file_stem = slugify(file_stem.replace("-", "_"))
            file_suffix = file_suffix.lower()
            file_name = file_stem + file_suffix

            # Check if the file is an image
            if file_suffix in [".png", ".svg", ".webp", ".jpg", ".jpeg", ".gif"]:
                full_file_name = "-".join(
                    [base_name, category_1, category_2, category_3, file_name]
                )
                # if category_1 == "language_rules_and_mechanisms":
                print(full_file_name)
                image_url = urljoin(base_url, img_src)

                if full_file_name not in existing_file_names:
                    fetch_datetime = datetime.now().isoformat()
                    picture_info.append(
                        {
                            "file_name": full_file_name,
                            "image_url": image_url,
                            "fetch_datetime": fetch_datetime,
                            "description": description,
                        }
                    )
                    existing_file_names.add(full_file_name)


def section_process_single_tag(
    tag: Tag,
    base_name: str,
    base_url: str,
    category_1: str,
    existing_file_names: set[str],
    picture_info: list[dict[str, Union[str, datetime]]],
) -> None:
    """
    Processes a section tag with a single div, handling cases where an h3 or h4 tag exists.
    """
    # case 1-2
    h3_tag = tag.find("h3", recursive=False)
    if h3_tag:
        category_2 = slugify(h3_tag.get_text(separator=" ", strip=True))
        process_image_tags(
            root_tag=tag,
            base_name=base_name,
            base_url=base_url,
            category_1=category_1,
            category_2=category_2,
            category_3="",
            existing_file_names=existing_file_names,
            picture_info=picture_info,
        )
        return

    # case 1-2: iterate through child elements
    for child in (child for child in tag.children if isinstance(child, Tag)):
        section_process_children_tags(
            child, base_name, base_url, category_1, existing_file_names, picture_info
        )


def section_process_children_tags(
    root_tag: Tag,
    base_name: str,
    base_url: str,
    category_1: str,
    existing_file_names: set[str],
    picture_info: list[dict[str, Union[str, datetime]]],
) -> None:
    """
    Handles processing of child tags when there is no h3 tag, handling h4 tags or further nested tags.
    """
    # case 1-2-1
    h3_tag = root_tag.find("h3", recursive=False)
    if h3_tag:
        category_2 = slugify(h3_tag.get_text(separator=" ", strip=True))
        process_image_tags(
            root_tag=root_tag,
            base_name=base_name,
            base_url=base_url,
            category_1=category_1,
            category_2=category_2,
            category_3="",
            existing_file_names=existing_file_names,
            picture_info=picture_info,
        )
        return

    # case 1-2-2
    h4_tag = root_tag.find("h4", recursive=False)
    if h4_tag:
        category_3 = slugify(h4_tag.get_text(separator=" ", strip=True))
        process_image_tags(
            root_tag=root_tag,
            base_name=base_name,
            base_url=base_url,
            category_1=category_1,
            category_2="",
            category_3=category_3,
            existing_file_names=existing_file_names,
            picture_info=picture_info,
        )
        return

    # case 1-2-3
    for child2 in (child2 for child2 in root_tag.children if isinstance(child2, Tag)):
        h3_tag = child2.find("h3", recursive=False)
        if h3_tag:
            category_2 = slugify(h3_tag.get_text(separator=" ", strip=True))
            process_image_tags(
                root_tag=child2,
                base_name=base_name,
                base_url=base_url,
                category_1=category_1,
                category_2=category_2,
                category_3="",
                existing_file_names=existing_file_names,
                picture_info=picture_info,
            )


def section_process_nested_div_tags(
    content_div_tag: Tag,
    base_name: str,
    base_url: str,
    category_1: str,
    existing_file_names: set[str],
    picture_info: list[dict[str, Union[str, datetime]]],
) -> None:
    """
    Processes nested div tags inside a section and handles nested h3 tags for section case 2-2.
    """
    nest_sub_tags = content_div_tag.find_all("div", recursive=False)
    section_j = 0
    nest_sub_tag_len = len(nest_sub_tags)

    while section_j < nest_sub_tag_len:
        nest_temp_div_tag = nest_sub_tags[section_j]

        h3_tag = nest_temp_div_tag.find("h3", recursive=False)
        category_2 = ""
        # case 2-1
        if h3_tag:
            category_2 = slugify(h3_tag.get_text(separator=" ", strip=True))
            section_j += 1
            if section_j >= nest_sub_tag_len:
                break

            nest_temp_div_tag = nest_sub_tags[section_j]
            if nest_temp_div_tag.name != "div":
                section_j += 1
                continue

            process_image_tags(
                root_tag=nest_temp_div_tag,
                base_name=base_name,
                base_url=base_url,
                category_1=category_1,
                category_2=category_2,
                category_3="",
                existing_file_names=existing_file_names,
                picture_info=picture_info,
            )
        section_j += 1


def section_process_multiple_tags(
    sub_tags: list[Tag],
    base_name: str,
    base_url: str,
    category_1: str,
    existing_file_names: set[str],
    picture_info: list[dict[str, Union[str, datetime]]],
) -> None:
    """
    Processes multiple div tags inside a section, handling nested divs and determining categories for images.
    """
    section_i = 0
    sub_tag_len = len(sub_tags)

    while section_i < sub_tag_len:
        temp_div_tag = sub_tags[section_i]

        h3_tag = temp_div_tag.find("h3", recursive=False)
        # case 2-1
        if h3_tag:
            category_2 = slugify(h3_tag.get_text(separator=" ", strip=True))
            section_i += 1
            if section_i >= sub_tag_len:
                break

            process_image_tags(
                root_tag=sub_tags[section_i],
                base_name=base_name,
                base_url=base_url,
                category_1=category_1,
                category_2=category_2,
                category_3="",
                existing_file_names=existing_file_names,
                picture_info=picture_info,
            )
        else:
            # case 2-2: if there's a nested div structure
            content_div_tag = temp_div_tag.find("div", recursive=False)
            if isinstance(content_div_tag, Tag):
                section_process_nested_div_tags(
                    content_div_tag,
                    base_name,
                    base_url,
                    category_1,
                    existing_file_names,
                    picture_info,
                )

        section_i += 1


def section_process_section_part(
    root_tag: Tag,
    base_name: str,
    base_url: str,
    category_1: str,
    existing_file_names: set[str],
    picture_info: list[dict[str, Union[str, datetime]]],
) -> None:
    """
    The main function that determines whether to process a single div or multiple divs in a section tag.
    Processes a section of HTML tags, handling single or multiple divs and nested structures
        to extract and download images.

    Flow of function calls:

    process_section_part
        ├──> process_single_tag [If len(sub_tags) == 1]
        │       ├──> process_image_tags [If h3 tag found in single div]
        │       └──> process_children_tags [If no h3 tag found]
        │               ├──> process_image_tags [If h3 tag found in children]
        │               └──> process_image_tags [If h4 tag found in children]
        └──> process_multiple_tags [If len(sub_tags) > 1]
                ├──> process_image_tags [If h3 tag found in divs]
                └──> process_nested_div_tags [If nested div structure found]
                        ├──> process_image_tags [If h3 tag found in nested divs]
                        └──> process_image_tags [If nested div structure continues]

    """
    sub_tags = root_tag.find_all("div", recursive=False)

    if len(sub_tags) == 1:
        section_process_single_tag(
            sub_tags[0],
            base_name,
            base_url,
            category_1,
            existing_file_names,
            picture_info,
        )
    else:
        section_process_multiple_tags(
            sub_tags, base_name, base_url, category_1, existing_file_names, picture_info
        )


# %%
# Create a list of Tag objects from the children of the section
existing_file_names = set()
picture_info = []
children: list[Tag] = [child for child in section.children if isinstance(child, Tag)]

section_i: int = 0
while section_i < len(children):
    ######### pair: (h2, div)
    h2_tag: Tag = children[section_i]
    if h2_tag.name != "h2":
        section_i += 1
        continue

    # Extract category_1 from the h2 tag
    category_1 = slugify(h2_tag.get_text(separator=" ", strip=True))
    print("===== category 1:", category_1)

    section_i += 1
    if section_i >= len(children):
        break

    div_tag: Tag = children[section_i]
    class_list: list[str] = get_class_list(div_tag)
    if div_tag.name != "div" or "content" not in class_list:
        section_i += 1
        continue

    # Title: (h2-div) --->
    # Process pairs of divs within the content div
    sub_children: list[Tag] = [
        child for child in div_tag.children if isinstance(child, Tag)
    ]
    section_j: int = 0

    while section_j < len(sub_children):
        ######### pair: (div, div)  // h3, div
        div_or_section_1 = sub_children[section_j]

        class_list = get_class_list(div_or_section_1)

        # if "panel-fold" not in class_list:
        if div_or_section_1.name not in ["section", "div"]:
            section_j += 1
            continue

        if div_or_section_1.name == "section":
            section_process_section_part(
                root_tag=div_or_section_1,
                base_name=base_name,
                base_url=base_url,
                category_1=category_1,
                existing_file_names=existing_file_names,
                picture_info=picture_info,
            )
            section_j += 1
            continue

        # Extract category_2 from the h3 tag within div1
        h3_tag = div_or_section_1.find("h3", recursive=False)
        if h3_tag:
            category_2 = slugify(h3_tag.get_text(separator=" ", strip=True))
            section_j += 1
            if section_j >= len(sub_children):
                break
            div_or_section_2: Tag = sub_children[section_j]
        else:
            category_2: str = ""
            div_or_section_2 = div_or_section_1

        sub_divs: list[Tag] = [
            child for child in div_or_section_2.children if isinstance(child, Tag)
        ]
        section_k: int = 0

        while section_k < len(sub_divs):
            ######### pair: (div, div)  // [h4], div
            div3: Tag = sub_divs[section_k]

            ### 🎠 case of 5
            if not h3_tag:
                h3_tag = div3.find("h3", recursive=False)
                if h3_tag:
                    category_2 = slugify(h3_tag.get_text(separator=" ", strip=True))
                    process_image_tags(
                        root_tag=div3,
                        base_name=base_name,
                        base_url=base_url,
                        category_1=category_1,
                        category_2=category_2,
                        category_3="",
                        existing_file_names=existing_file_names,
                        picture_info=picture_info,
                    )
                    section_k += 1
                    continue

            h4_tag = div3.find("h4", recursive=False)
            if h4_tag:
                ### 🎠 case of 2
                category_3_text = h4_tag.get_text(separator=" ", strip=True)
                category_3 = slugify(category_3_text)
                section_k += 1
                if section_k >= len(sub_divs):
                    break
                div4: Tag = sub_divs[section_k]
            else:
                ### 🎠 case of 1
                category_3 = ""
                div4 = div3

            # Process images within div4
            process_image_tags(
                root_tag=div4,
                base_name=base_name,
                base_url=base_url,
                category_1=category_1,
                category_2=category_2,
                category_3=category_3,
                existing_file_names=existing_file_names,
                picture_info=picture_info,
            )

            section_k += 1

        section_j += 1

    section_i += 1

# Write the updated picture_info to the temporary JSON file
# with json_file_path.open("w", encoding="utf-8") as f:
#     json.dump(picture_info, f, ensure_ascii=False, indent=4)


# %%


# Create a temporary JSON file with "cheat_sheet-cpp" in the file name
temp_json_file = create_temp_json_file(data)

# Open the JSON file in VS Code
open_json_in_vscode(temp_json_file)

# %%
from bs4 import BeautifulSoup

# 테스트할 HTML
html_content = """
<section>
  <div class="panel-fold panel-fold-header" open="">
    <h3 class="nav-none"><code>std::</code> Sequence Containers</h3>
  </div>
  <div class="panel bg-filled vcompact">
    <div class="block">
      <a class="img hcentered" href="cpp/std/sequence_containers.png">
        <img
          alt="standard library sequence containers overview"
          loading="lazy"
          src="cpp/std/sequence_containers_crop.png"
          style="max-width: 100%; width: 1000px"
          title="click for fullscreen view"
        />
      </a>
    </div>
    <div class="panel fit-content">
      <ul class="resources">
        <li class="cat-article">
          <a href="cpp/std/sequence_containers.html"
            >Standard Library Sequence Containers Overview</a>
        </li>
      </ul>
    </div>
  </div>
</section>
"""

# BeautifulSoup 객체 생성
soup = BeautifulSoup(html_content, "html.parser")

# root_tag로 section을 기준으로 설정
root_tag = soup.find("section")

# 테스트 1: recursive=False로 바로 하위 div만 검색
direct_divs = root_tag.find_all("div", recursive=False)

# 테스트 2: recursive=True로 모든 자손 div 태그 검색
all_divs = root_tag.find_all("div", recursive=True)

# %%
# 출력 결과 확인
print("=== direct divs (recursive=False) ===")
for div in direct_divs:
    print(div.prettify())

# %%
print("\n=== all divs (recursive=True) ===")
for div in all_divs:
    print(div.prettify())
