# %%
# Written in 📅 2024-09-15 05:06:05
# Enable all outputs in the Jupyter notebook environment
import json
from datetime import datetime
from pathlib import Path
from typing import Union

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from IPython.core.interactiveshell import InteractiveShell
from requests.compat import urljoin
from wbfw109.libs.crawling import get_class_list
from wbfw109.libs.file import create_temp_json_file, open_json_in_vscode
from wbfw109.libs.string import slugify

InteractiveShell.ast_node_interactivity = "all"
# %%

# Set the base name for file naming
base_name: str = slugify("cheat_sheet-cpp")

# Load existing data from a temporary JSON file
json_file_path = create_temp_json_file({}, prefix="cheat_sheet-cpp", suffix=".json")

# Check if the file already contains data
try:
    with json_file_path.open("r", encoding="utf-8") as f:
        picture_info: list[dict[str, Union[str, datetime]]] = json.load(f)
        if not isinstance(picture_info, list):
            # In case the JSON file has incorrect data format (like dict), reset to an empty list
            picture_info = []
except FileNotFoundError:
    picture_info: list[dict[str, Union[str, datetime]]] = []

# Proceed with the rest of the code...

# Create a set of existing file names to avoid duplicates
existing_file_names: set[str] = set(item["file_name"] for item in picture_info)

# Fetch the web page content using requests
url: str = "https://hackingcpp.com/cpp/cheat_sheets.html"
base_url = "https://hackingcpp.com"


headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

response = requests.get(url, headers=headers)
response.raise_for_status()  # Raise an exception for HTTP errors
html_content: str = response.text

# Parse the HTML content with BeautifulSoup
soup: BeautifulSoup = BeautifulSoup(html_content, "html.parser")

# Find the main section with class 'main plain'
section: Tag = soup.find("section", class_="main plain")

# Create a list of Tag objects from the children of the section
children: list[Tag] = [child for child in section.children if isinstance(child, Tag)]
# %%
i: int = 0
while i < len(children):
    h2_tag: Tag = children[i]
    if h2_tag.name != "h2":
        i += 1
        continue

    ### Extract category_1 from the h2 tag
    category_1_text = h2_tag.get_text(separator=" ", strip=True)
    category_1 = slugify(category_1_text)

    ### Move to the next sibling
    i += 1
    if i >= len(children):
        break

    div_tag: Tag = children[i]
    class_list: list[str] = get_class_list(div_tag)
    if div_tag.name != "div" or "content" not in class_list:
        i += 1
        continue

    # Process pairs of divs within the content div
    sub_children: list[Tag] = [
        child for child in div_tag.children if isinstance(child, Tag)
    ]
    j: int = 0

    while j < len(sub_children):
        div_or_section_1: Tag = sub_children[j]
        class_list = get_class_list(div_or_section_1)
        ###
        if "panel-fold" not in class_list:
            j += 1
            continue

        # Extract category_2 from the h3 tag within div1
        h3_tag = div_or_section_1.find("h3")

        if h3_tag:
            category_2_text = h3_tag.get_text(separator=" ", strip=True)
            category_2 = slugify(category_2_text)
        else:
            category_2: str = "Unknown"
        ###
        j += 1
        if j >= len(sub_children):
            break

        div_or_section_2: Tag = sub_children[j]

        # Find the image tag within div2
        img_tag = div_or_section_2.find("img")
        if isinstance(img_tag, Tag):
            img_src = img_tag.get("src")
            if not img_src:
                img_src = img_tag.get("data-src")

            if img_src:
                file_path: Path = Path(img_src)
                file_stem, file_suffix = file_path.stem, file_path.suffix
                file_stem = slugify(file_stem.replace("-", "_"))
                file_suffix = file_suffix.lower()
                file_name = file_stem + file_suffix

                # Check if the file is an image
                if file_suffix in [".png", ".svg", ".webp", ".jpg", ".jpeg", ".gif"]:
                    # Construct the full file name and image URL
                    full_file_name: str = "-".join(
                        [base_name, category_1, category_2, file_name]
                    )
                    print(full_file_name)
                    image_url: str = urljoin(base_url, img_src)

                    # Avoid duplicates
                    if full_file_name not in existing_file_names:
                        fetch_datetime: str = datetime.now().isoformat()
                        picture_info.append(
                            {
                                "file_name": full_file_name,
                                "image_url": image_url,
                                "fetch_datetime": fetch_datetime,
                            }
                        )
                        existing_file_names.add(full_file_name)
        j += 1

    i += 1

# Write the updated picture_info to the temporary JSON file
with json_file_path.open("w", encoding="utf-8") as f:
    json.dump(picture_info, f, ensure_ascii=False, indent=4)

# %%
# Open the JSON file in VS Code for review
open_json_in_vscode(json_file_path)


# %%

# Image downloading code (if needed)
# Set the directory to save images
images_dir: Path = Path.home() / "crawling" / "images"
images_dir.mkdir(parents=True, exist_ok=True)

for item in picture_info:
    file_path: Path = images_dir / item["file_name"]
    if not file_path.exists():
        response = requests.get(item["image_url"], headers=headers)
        if response.status_code == 200:
            file_path.write_bytes(response.content)
            print(f"Downloaded {item['file_name']}")
        else:
            print(f"Failed to download {item['file_name']}")
    else:
        print(f"Already exists: {item['file_name']}")


# %%

# Create a temporary JSON file with "cheat_sheet-cpp" in the file name
temp_json_file = create_temp_json_file(data)

# Open the JSON file in VS Code
open_json_in_vscode(temp_json_file)

# %%
import json
from pathlib import Path
from typing import Any

import requests

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
import json
from datetime import datetime
from pathlib import Path
from typing import Union

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from requests.compat import urljoin
from wbfw109.libs.crawling import get_class_list
from wbfw109.libs.file import create_temp_json_file
from wbfw109.libs.string import slugify

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

# Create a list of Tag objects from the children of the section
children: list[Tag] = [child for child in section.children if isinstance(child, Tag)]
section_i: int = 0
while section_i < len(children):
    ######### pair: (h2, div)
    h2_tag: Tag = children[section_i]
    if h2_tag.name != "h2":
        section_i += 1
        continue

    # Extract category_1 from the h2 tag
    category_1_text = h2_tag.get_text(separator=" ", strip=True)
    category_1 = slugify(category_1_text)

    section_i += 1
    if section_i >= len(children):
        break

    div_tag: Tag = children[section_i]
    class_list: list[str] = get_class_list(div_tag)
    if div_tag.name != "div" or "content" not in class_list:
        section_i += 1
        continue
    # Process pairs of divs within the content div
    sub_children: list[Tag] = [
        child for child in div_tag.children if isinstance(child, Tag)
    ]
    section_j: int = 0

    while section_j < len(sub_children):
        ######### pair: (div, div)  // h3, div
        div_or_section_1: Tag = sub_children[section_j]
        class_list = get_class_list(div_or_section_1)

        if "panel-fold" not in class_list:
            section_j += 1
            continue

        # Extract category_2 from the h3 tag within div1
        h3_tag = div_or_section_1.find("h3")
        if h3_tag:
            category_2_text = h3_tag.get_text(separator=" ", strip=True)
            category_2 = slugify(category_2_text)
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
            h4_tag = div3.find("h4")
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
            img_tags = div4.find_all("img")
            for img_tag in img_tags:
                img_src = img_tag.get("src") or img_tag.get("data-src")

                if img_src:
                    file_path: Path = Path(img_src)
                    file_stem, file_suffix = file_path.stem, file_path.suffix
                    file_stem = slugify(file_stem.replace("-", "_"))
                    file_suffix = file_suffix.lower()
                    file_name = file_stem + file_suffix

                    # Check if the file is an image
                    if file_suffix in [
                        ".png",
                        ".svg",
                        ".webp",
                        ".jpg",
                        ".jpeg",
                        ".gif",
                    ]:
                        full_file_name: str = "-".join(
                            [base_name, category_1, category_2, category_3, file_name]
                        )
                        print(full_file_name)
                        image_url: str = urljoin(base_url, img_src)

                        if full_file_name not in existing_file_names:
                            fetch_datetime: str = datetime.now().isoformat()
                            picture_info.append(
                                {
                                    "file_name": full_file_name,
                                    "image_url": image_url,
                                    "fetch_datetime": fetch_datetime,
                                }
                            )
                            existing_file_names.add(full_file_name)

            section_k += 1

        section_j += 1

    section_i += 1

# Write the updated picture_info to the temporary JSON file
with json_file_path.open("w", encoding="utf-8") as f:
    json.dump(picture_info, f, ensure_ascii=False, indent=4)

# %%
