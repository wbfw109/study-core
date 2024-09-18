# %%
# Written in 📅 2024-09-15 05:06:05
# Enable all outputs in the Jupyter notebook environment
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import cairosvg
import requests
from bs4 import BeautifulSoup, Tag
from bs4.element import Tag
from IPython.core.interactiveshell import InteractiveShell
from PIL import Image
from requests.compat import urljoin
from wbfw109.libs.crawling import get_class_list
from wbfw109.libs.file import create_temp_json_file, open_json_in_vscode
from wbfw109.libs.string import slugify
# from wbfw109.libs.ai_models import ModelManager, RealESRGANPlugin


InteractiveShell.ast_node_interactivity = "all"

# Set the image directory
ai_models_dir: Path = Path.home() / "ai_models"
ai_models_dir.mkdir(parents=True, exist_ok=True)
images_dir: Path = Path.home() / "crawling" / "images"
images_dir.mkdir(parents=True, exist_ok=True)
upscalling_output_dir = images_dir / "upscaling"
upscalling_output_dir.mkdir(parents=True, exist_ok=True)


# %%
# 1. Utility Functions


# Function to set the base name for file naming
def get_base_name(name: str) -> str:
    """
    Slugifies and returns the base name for file naming.

    Parameters:
        name (str): The name to be slugified.

    Returns:
        str: The slugified base name.
    """
    return slugify(name)


# Function to load existing data from a JSON file
def load_json_data(json_file_path: Path) -> list[dict[str, Union[str, datetime]]]:
    """
    Loads existing data from a JSON file, if available.

    Parameters:
        json_file_path (Path): The path to the JSON file.

    Returns:
        list[dict[str, Union[str, datetime]]]: A list of dictionaries containing picture information.
    """
    try:
        with json_file_path.open("r", encoding="utf-8") as f:
            picture_info = json.load(f)
            if not isinstance(picture_info, list):
                return []
            return picture_info  # type: ignore
    except FileNotFoundError:
        return []


# Function to create a set of existing file names to avoid duplicates
def get_existing_file_names(
    picture_info: list[dict[str, Union[str, datetime]]]
) -> set[str]:
    """
    Creates a set of existing file names from the picture information.

    Parameters:
        picture_info (list[dict[str, Union[str, datetime]]]): A list of dictionaries with picture info.

    Returns:
        set[str]: A set of file names (strings) to avoid duplicates.
    """
    return {
        item["file_name"] for item in picture_info if isinstance(item["file_name"], str)
    }


# Function to fetch the web page content
def fetch_web_page(url: str, headers: dict[str, str]) -> str:
    """
    Fetches the content of a web page.

    Parameters:
        url (str): The URL of the web page.
        headers (dict[str, str]): Headers to be used in the HTTP request.

    Returns:
        str: The HTML content of the web page.
    """
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text


# Function to parse the HTML content using BeautifulSoup
def parse_html(html_content: str) -> BeautifulSoup:
    """
    Parses HTML content into a BeautifulSoup object.

    Parameters:
        html_content (str): The HTML content to be parsed.

    Returns:
        BeautifulSoup: A BeautifulSoup object representing the parsed HTML.
    """
    return BeautifulSoup(html_content, "html.parser")


# Function to find the main section with a specific class
def find_main_section(soup: BeautifulSoup, class_name: str) -> Tag | None:
    """
    Finds the main section of an HTML document by class name.

    Parameters:
        soup (BeautifulSoup): The BeautifulSoup object of the parsed HTML.
        class_name (str): The class name of the section to find.

    Returns:
        Tag | None: The section tag if found, else None.
    """
    section = soup.find("section", class_=class_name)
    if isinstance(section, Tag):
        return section
    return None  # or raise an error if necessary


# 2. Processing HTML Tags


# Function to process a tag and extract information (non-section version)
def process_tag_structure(
    base_name: str,
    base_url: str,
    section: Tag,
    existing_file_names: set[str],
    picture_info: list[dict[str, Union[str, datetime]]],
) -> None:
    """
    Processes an HTML tag to extract information from non-section tags.

    Parameters:
        base_name (str): The base name for file naming.
        base_url (str): The base URL for image sources.
        section (Tag): The tag to be processed.
        existing_file_names (set[str]): Set of existing file names to avoid duplicates.
        picture_info (list[dict[str, Union[str, datetime]]]): List of dictionaries containing picture info.
    """
    children: list[Tag] = [
        child for child in section.children if isinstance(child, Tag)
    ]

    section_i = 0
    while section_i < len(children):
        h2_tag = children[section_i]
        if h2_tag.name != "h2":
            section_i += 1
            continue

        # Extract category_1 from the h2 tag
        category_1 = slugify(h2_tag.get_text(separator=" ", strip=True))
        # print(f"===== category 1: {category_1}")

        section_i += 1
        if section_i >= len(children):
            break

        div_tag = children[section_i]
        class_list = get_class_list(div_tag)
        if div_tag.name != "div" or "content" not in class_list:
            section_i += 1
            continue

        # Process pairs of divs within the content div
        sub_children: list[Tag] = [
            child for child in div_tag.children if isinstance(child, Tag)
        ]
        process_sub_children(
            sub_children,
            base_name,
            base_url,
            category_1,
            existing_file_names,
            picture_info,
        )

        section_i += 1


# Function to process sub-children tags
def process_sub_children(
    sub_children: list[Tag],
    base_name: str,
    base_url: str,
    category_1: str,
    existing_file_names: set[str],
    picture_info: list[dict[str, Union[str, datetime]]],
) -> None:
    """
    Processes sub-children tags to extract information and categories.

    Parameters:
        sub_children (list[Tag]): A list of sub-tags within a div.
        base_name (str): The base name for file naming.
        base_url (str): The base URL for image sources.
        category_1 (str): The first category of images.
        existing_file_names (set[str]): Set of existing file names to avoid duplicates.
        picture_info (list[dict[str, Union[str, datetime]]]): List of dictionaries containing picture info.
    """
    section_j = 0
    while section_j < len(sub_children):
        div_or_section_1 = sub_children[section_j]

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
        process_sub_divs(
            sub_divs,
            base_name,
            base_url,
            category_1,
            category_2,
            existing_file_names,
            picture_info,
        )
        section_j += 1


# Function to process sub-divs and handle nested structures
def process_sub_divs(
    sub_divs: list[Tag],
    base_name: str,
    base_url: str,
    category_1: str,
    category_2: str,
    existing_file_names: set[str],
    picture_info: list[dict[str, Union[str, datetime]]],
) -> None:
    """
    Processes sub-div tags to extract and download images, handling nested structures.

    Parameters:
        sub_divs (list[Tag]): A list of sub-tags within a div.
        base_name (str): The base name for file naming.
        base_url (str): The base URL for image sources.
        category_1 (str): The first category of images.
        category_2 (str): The second category of images.
        existing_file_names (set[str]): Set of existing file names to avoid duplicates.
        picture_info (list[dict[str, Union[str, datetime]]]): List of dictionaries containing picture info.
    """
    section_k = 0
    while section_k < len(sub_divs):
        div3 = sub_divs[section_k]

        h3_tag = div3.find("h3", recursive=False)
        if h3_tag:
            category_2 = slugify(h3_tag.get_text(separator=" ", strip=True))
            process_image_tags(
                div3,
                base_name,
                base_url,
                category_1,
                category_2,
                "",
                existing_file_names,
                picture_info,
            )
            section_k += 1
            continue

        h4_tag = div3.find("h4", recursive=False)
        if h4_tag:
            category_3 = slugify(h4_tag.get_text(separator=" ", strip=True))
            section_k += 1
            if section_k >= len(sub_divs):
                break
            div4: Tag = sub_divs[section_k]
        else:
            category_3 = ""
            div4 = div3

        process_image_tags(
            div4,
            base_name,
            base_url,
            category_1,
            category_2,
            category_3,
            existing_file_names,
            picture_info,
        )
        section_k += 1


# 3. Main Processing Logic


# Main function to process a section tag and extract image information
def section_process_section_part(
    root_tag: Tag,
    base_name: str,
    base_url: str,
    category_1: str,
    existing_file_names: set[str],
    picture_info: list[dict[str, Union[str, datetime]]],
) -> None:
    """
    Processes a section of HTML tags, handling single or multiple divs and nested structures to extract and download images.

    **Flow of function calls**:

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
                      └──> proce
    Parameters:
        root_tag (Tag): The root section tag to process.
        base_name (str): The base name for file naming.
        base_url (str): The base URL for image sources.
        category_1 (str): The first category of images.
        existing_file_names (set[str]): Set of existing file names to avoid duplicates.
        picture_info (list[dict[str, Union[str, datetime]]]): List of dictionaries containing picture info.
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


# Function to process a single section tag
def section_process_single_tag(
    tag: Tag,
    base_name: str,
    base_url: str,
    category_1: str,
    existing_file_names: set[str],
    picture_info: list[dict[str, Union[str, datetime]]],
) -> None:
    """
    Processes a single div within a section and extracts images based on its content.

    Parameters:
        tag (Tag): The div tag to process.
        base_name (str): The base name for file naming.
        base_url (str): The base URL for image sources.
        category_1 (str): The first category of images.
        existing_file_names (set[str]): Set of existing file names to avoid duplicates.
        picture_info (list[dict[str, Union[str, datetime]]]): List of dictionaries containing picture info.
    """
    h3_tag = tag.find("h3", recursive=False)
    if h3_tag:
        category_2 = slugify(h3_tag.get_text(separator=" ", strip=True))
        process_image_tags(
            tag,
            base_name,
            base_url,
            category_1,
            category_2,
            "",
            existing_file_names,
            picture_info,
        )
        return

    for child in (child for child in tag.children if isinstance(child, Tag)):
        section_process_children_tags(
            child, base_name, base_url, category_1, existing_file_names, picture_info
        )


# Function to process child tags within a section
def section_process_children_tags(
    root_tag: Tag,
    base_name: str,
    base_url: str,
    category_1: str,
    existing_file_names: set[str],
    picture_info: list[dict[str, Union[str, datetime]]],
) -> None:
    """
    Processes child tags within a section and extracts images based on h3 and h4 tags.

    Parameters:
        root_tag (Tag): The root tag containing child elements.
        base_name (str): The base name for file naming.
        base_url (str): The base URL for image sources.
        category_1 (str): The first category of images.
        existing_file_names (set[str]): Set of existing file names to avoid duplicates.
        picture_info (list[dict[str, Union[str, datetime]]]): List of dictionaries containing picture info.
    """
    h3_tag = root_tag.find("h3", recursive=False)
    if h3_tag:
        category_2 = slugify(h3_tag.get_text(separator=" ", strip=True))
        process_image_tags(
            root_tag,
            base_name,
            base_url,
            category_1,
            category_2,
            "",
            existing_file_names,
            picture_info,
        )
        return

    h4_tag = root_tag.find("h4", recursive=False)
    if h4_tag:
        category_3 = slugify(h4_tag.get_text(separator=" ", strip=True))
        process_image_tags(
            root_tag,
            base_name,
            base_url,
            category_1,
            "",
            category_3,
            existing_file_names,
            picture_info,
        )
        return

    for child2 in (child2 for child2 in root_tag.children if isinstance(child2, Tag)):
        h3_tag = child2.find("h3", recursive=False)
        if h3_tag:
            category_2 = slugify(h3_tag.get_text(separator=" ", strip=True))
            process_image_tags(
                child2,
                base_name,
                base_url,
                category_1,
                category_2,
                "",
                existing_file_names,
                picture_info,
            )


# Function to process nested div tags in a section
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

    Parameters:
        content_div_tag (Tag): The div tag containing nested div elements.
        base_name (str): The base name for file naming.
        base_url (str): The base URL for image sources.
        category_1 (str): The first category of images.
        existing_file_names (set[str]): Set of existing file names to avoid duplicates.
        picture_info (list[dict[str, Union[str, datetime]]]): List of dictionaries containing picture info.
    """
    nest_sub_tags = content_div_tag.find_all("div", recursive=False)
    section_j = 0
    nest_sub_tag_len = len(nest_sub_tags)

    while section_j < nest_sub_tag_len:
        nest_temp_div_tag = nest_sub_tags[section_j]

        h3_tag = nest_temp_div_tag.find("h3", recursive=False)
        category_2 = ""
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
                nest_temp_div_tag,
                base_name,
                base_url,
                category_1,
                category_2,
                "",
                existing_file_names,
                picture_info,
            )
        section_j += 1


# Function to process multiple div tags in a section
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

    Parameters:
        sub_tags (list[Tag]): A list of div tags to process.
        base_name (str): The base name for file naming.
        base_url (str): The base URL for image sources.
        category_1 (str): The first category of images.
        existing_file_names (set[str]): Set of existing file names to avoid duplicates.
        picture_info (list[dict[str, Union[str, datetime]]]): List of dictionaries containing picture info.
    """
    section_i = 0
    sub_tag_len = len(sub_tags)

    while section_i < sub_tag_len:
        temp_div_tag = sub_tags[section_i]

        h3_tag = temp_div_tag.find("h3", recursive=False)
        if h3_tag:
            category_2 = slugify(h3_tag.get_text(separator=" ", strip=True))
            section_i += 1
            if section_i >= sub_tag_len:
                break

            process_image_tags(
                sub_tags[section_i],
                base_name,
                base_url,
                category_1,
                category_2,
                "",
                existing_file_names,
                picture_info,
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


# 4. Helper Functions


# Function to process image tags and extract image information
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
    """
    Processes image tags within a root tag, extracts image URLs, and stores information.

    Parameters:
        root_tag (Tag): The root tag containing image elements.
        base_name (str): The base name for file naming.
        base_url (str): The base URL for image sources.
        category_1 (str): The first category of images.
        category_2 (str): The second category of images.
        category_3 (str): The third category of images.
        existing_file_names (set[str]): Set of existing file names to avoid duplicates.
        picture_info (list[dict[str, Union[str, datetime]]]): List of dictionaries containing picture info.
        description (str): Description of the image (optional).
    """
    img_tags = root_tag.find_all("img")
    description = root_tag.get_text(separator=" ", strip=True)

    for img_tag in img_tags:
        img_src = img_tag.get("src") or img_tag.get("data-src")
        if img_src:
            file_path = Path(img_src)
            file_stem, file_suffix = file_path.stem, file_path.suffix
            file_stem = slugify(file_stem.replace("-", "_"))
            file_suffix = file_suffix.lower()
            file_name = file_stem + file_suffix

            if file_suffix in [".png", ".svg", ".webp", ".jpg", ".jpeg", ".gif"]:
                full_file_name = "-".join(
                    [base_name, category_1, category_2, category_3, file_name]
                )
                # print(full_file_name)
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


# Title: download by parsing .json
from pathlib import Path

from PIL import PngImagePlugin


def download_image(
    image_url: str, file_name: str, images_dir: Path, description: str
) -> None:
    """
    Downloads an image from the specified URL, saves it, and adds a description to its EXIF data.

    Parameters:
        image_url (str): The URL of the image to download.
        file_name (str): The name of the file to save the image as.
        images_dir (Path): The directory where the image will be saved.
        description (str): The description to add to the image's EXIF data.
    """
    response = requests.get(image_url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses

    # Set the full path to save the image
    image_path: Path = images_dir / file_name

    # Write the image content to the file
    with image_path.open("wb") as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)

    # Open the image using PIL to modify EXIF
    with Image.open(image_path) as img:
        # Create a dictionary for EXIF data
        exif_dict = img.info.get("exif", b"")

        # Check if the image is in a format that supports EXIF
        if img.format in ["JPEG", "TIFF"]:
            exif_data = img.getexif()
            # Set the ImageDescription (EXIF tag 270) with the description
            exif_data[270] = description

            # Save the image with the updated EXIF data
            img.save(image_path, exif=exif_data)

        elif img.format == "PNG":
            # For PNG, we can use PngImagePlugin to add a description
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("Description", description)
            img.save(image_path, "PNG", pnginfo=metadata)

    print(
        f"Downloaded {file_name} to {image_path} with EXIF description: {description}"
    )


def parse_json_and_download_images(json_file_path: Path, images_dir: Path) -> None:
    """
    Parses the JSON file and downloads the images based on the URLs inside the file,
    adding descriptions from the JSON file to the images' EXIF data.

    Parameters:
        json_file_path (Path): The path to the JSON file that contains the image info.
        images_dir (Path): The directory where images will be downloaded.
    """
    # Load the JSON file
    with json_file_path.open("r", encoding="utf-8") as f:
        picture_info: list[dict[str, str]] = json.load(f)

    # Iterate over each entry and download the images
    for item in picture_info:
        file_name: str = item["file_name"]
        image_url: str = item["image_url"]
        description: str = item.get("description", "No description")

        # Download the image and add the description to the EXIF
        download_image(image_url, file_name, images_dir, description)


# Example usage:
# json_file_path = Path("path_to_json_file")
# images_dir = Path("path_to_images_dir")
# parse_json_and_download_images(json_file_path, images_dir)


## converto svg to png with white background
def convert_svg_to_png_with_background(
    svg_file_path: Path,
    output_dir: Path,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Converts an SVG file to PNG format with a background color (removes transparency).

    Parameters:
        svg_file_path (Path): Path to the SVG file to be converted.
        output_dir (Path): The directory where the converted PNG file will be saved.
        background_color (Tuple[int, int, int]): The RGB color for the background (default: white).
    """
    # Create the PNG file path
    png_file_path = output_dir / (svg_file_path.stem + ".png")

    # Convert SVG to PNG using CairoSVG, preserving transparency initially
    cairosvg.svg2png(url=str(svg_file_path), write_to=str(png_file_path))

    # Load the PNG into PIL to manipulate the background
    with Image.open(png_file_path) as img:
        if img.mode == "RGBA":
            # Create a background image (default white)
            background = Image.new("RGB", img.size, background_color)
            # Paste the image on the background (alpha composite)
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            # Save the final PNG with the specified background color
            background.save(png_file_path, "PNG")

    print(
        f"Converted {svg_file_path} to {png_file_path} with background color {background_color}"
    )


def convert_all_svgs_in_directory(
    input_dir: Path,
    output_dir: Path,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Converts all SVG files in the input directory to PNG format with a white background.

    Parameters:
        input_dir (Path): Directory containing the SVG files.
        output_dir (Path): Directory where the converted PNG files will be saved.
        background_color (tuple): The RGB color for the background (default: white).

    Example:
        # Set the input directory where SVG images are located
        input_directory = Path.home() / "crawling" / "images_copy"

        # Set the output directory where PNG images will be saved
        output_directory = Path.home() / "crawling" / "converted_images"

        # Convert all SVG files in the input directory to PNG
        convert_all_svgs_in_directory(input_directory, output_directory)
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through all SVG files in the input directory
    for svg_file in input_dir.glob("*.svg"):
        convert_svg_to_png_with_background(svg_file, output_dir, background_color)


# Title:  main~
# Example of how to use these functions
def fetch_data() -> Path:
    base_name = get_base_name("cheat_sheet-cpp")

    # Load existing data from the temporary JSON file
    json_file_path = create_temp_json_file({}, prefix="cheat_sheet-cpp", suffix=".json")
    picture_info = load_json_data(json_file_path)
    existing_file_names = get_existing_file_names(picture_info)

    # Fetch the web page content
    url = "https://hackingcpp.com/cpp/cheat_sheets.html"
    base_url = "https://hackingcpp.com"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    html_content = fetch_web_page(url, headers)

    # Parse HTML content and find the main section
    soup = parse_html(html_content)
    section = find_main_section(soup, "main plain")
    if not section:
        return Path()

    # Process the section and extract images
    process_tag_structure(
        base_name, base_url, section, existing_file_names, picture_info
    )

    temp_json_file: Path = create_temp_json_file(
        picture_info, prefix="cheat_sheet-cpp", suffix=".json"
    )

    # Open the JSON file in VS Code
    open_json_in_vscode(temp_json_file)

    return temp_json_file


if __name__ == "__main__":
    ## 1. Fetch data from web site
    # temp_json_file = fetch_data()

    # Set the input directory where SVG images are located
    input_directory = Path.home() / "crawling" / "images_copy"
    # Set the output directory where PNG images will be saved
    output_directory = Path.home() / "crawling" / "converted_images"
    ## 2. Convert all SVG files in the input directory to PNG
    # convert_all_svgs_in_directory(input_directory, output_directory)


# %%


# Write the updated picture_info to the temporary JSON file
# with json_file_path.open("w", encoding="utf-8") as f:
#     json.dump(picture_info, f, ensure_ascii=False, indent=4)


# %%

from pathlib import Path

import pillow_avif  # noqa: F401
from waifu2x_ncnn_py import Waifu2x

STANDARD_SIZE_FOR_QHD = 1920  # 3840/2

from pathlib import Path



# def convert_png_to_avif(png_image_path: Path, avif_image_path: Path) -> None:
#     """
#     Converts a PNG image to AVIF format using Pillow with AVIF plugin support, handling color and transparency issues.

#     Parameters:
#         png_image_path (Path): Path to the input PNG image.
#         avif_image_path (Path): Path to save the AVIF image.
#     """
#     # Load the PNG image
#     img = Image.open(png_image_path)

#     # Ensure the image is in RGB mode before conversion
#     # if img.mode != "RGB":
#     #     img = img.convert("RGB")


#     # Convert and save as AVIF
#     img.save(avif_image_path, format="PNG", quality=95)
#     print(f"Converted {png_image_path} to {avif_image_path}")
def upscale_image_recursively(
    input_image_path: Path,
    output_image_path: Path,
    waifu2x_instance,
    scale_factor: int = 2,
) -> None:
    """
    Recursively upscales the input image by a factor of 2 using waifu2x until either the width or height
    is equal to or greater than STANDARD_SIZE_FOR_QHD. Saves the final image as PNG.

    Parameters:
        input_image_path (Path): Path to the input image.
        output_image_path (Path): Path to save the upscaled PNG image.
        waifu2x_instance (Waifu2x): An instance of the Waifu2x class for processing.
        scale_factor (int): The factor by which to upscale (default: 2).
    """
    with Image.open(input_image_path) as img:
        # Get the original image mode and size
        original_mode = img.mode
        width, height = img.size

        # Recursively upscale the image if necessary
        while width < STANDARD_SIZE_FOR_QHD and height < STANDARD_SIZE_FOR_QHD:
            print(f"Upscaling: Current size = ({width}, {height})")

            # Use Waifu2x instance to upscale the image
            img = waifu2x_instance.process_pil(img)

            # Update image size
            width, height = img.size

        # Save the final upscaled image, preserving the original color mode
        img.save(output_image_path, format="PNG", quality=95)
        print(f"Final upscaled image saved at {output_image_path}")


def upscale_and_convert_directory(
    input_dir: Path, output_dir: Path, waifu2x_instance, scale_factor: int = 2
) -> None:
    """
    Upscales all PNG images in the input directory.

    Parameters:
        input_dir (Path): Directory containing the PNG images.
        output_dir (Path): Directory where the upscaled images will be saved.
        waifu2x_instance (Waifu2x): An instance of the Waifu2x class for processing.
        scale_factor (int): The factor by which to upscale (default: 2).
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through all PNG files in the input directory
    for input_image_path in input_dir.glob("*.png"):
        output_png_path = (
            output_dir / input_image_path.name
        )  # Save as PNG in output directory

        # Step 1: Recursively upscale the image until it meets the QHD size requirements
        upscale_image_recursively(
            input_image_path, output_png_path, waifu2x_instance, scale_factor
        )


# Example usage:
input_directory = Path.home() / "crawling" / "converted_images"
output_directory = upscalling_output_dir


# Create a Waifu2x instance
waifu2x_instance = Waifu2x(
    gpuid=0, scale=2, noise=3
)  # Use GPU, scale by 2, and apply noise reduction

# Upscale and convert all PNG images in the input directory
upscale_and_convert_directory(input_directory, output_directory, waifu2x_instance)
#! sudo apt install -y libomp5 libvulkan-dev

# %%
# python inference_realesrgan.py -n RealESRGAN_x4plus -i input_folder --outscale 4
# /home/wbfw109/crawling/converted_images
# python inference_realesrgan.py -n RealESRGAN_x4plus -i /home/wbfw109/crawling/converted_images_test_sample --outscale 4

poetry 에서 Real-ESRGAN 을 설치하고 이미지의 크기를 scaling 하기 위한 방법.
 STANDARD_SIZE_FOR_QHD = 1920  # 3840/2
으로 높이나 너비가 1920 이상이 될 때가지 *2 를 반복하는 또는 한번에 그 크기보다 크도록 *4나 *8.. 이렇게 되도록 코드를 짜줄 수 있어? 후자가 더 효율적일 것 같아. 이 케이스에 대해 짜줘.

#%%
import subprocess

def upscale_image_cli(input_image_path: str, output_image_path: str, scale: int = 4):
    """
    Python에서 Real-ESRGAN CLI 명령어를 사용하여 이미지를 업스케일하는 함수.
    
    Args:
        input_image_path (str): 입력 이미지 파일 경로.
        output_image_path (str): 출력 이미지 파일 경로.
        scale (int): 업스케일 배율 (기본값은 4배).
    """
    # CLI 명령어 구성
    command = [
        "python", "inference_realesrgan.py",  # Real-ESRGAN inference 스크립트
        "-i", input_image_path,               # 입력 이미지 경로
        "-o", output_image_path,              # 출력 이미지 경로
        "--scale", str(scale)                 # 업스케일 배율
    ]
    
    # 명령어 실행
    result = subprocess.run(command, capture_output=True, text=True)
    
    # 실행 결과 출력 (성공/실패 여부와 메시지)
    if result.returncode == 0:
        print(f"Successfully upscaled the image and saved to {output_image_path}")
    else:
        print(f"Error in upscaling: {result.stderr}")

# 사용 예시
my_input_file: str = "~/crawling/converted_images_test_sample/cheat_sheet_cpp-terminology-function_contracts--function_contracts_crop.png"
my_output_file: str = "~/crawling/converted_images_test_sample/cheat_sheet_cpp-terminology-function_contracts--function_contracts_crop-tttt.png"
upscale_image_cli(my_input_file, my_output_file, scale=4)


# %%
if __name__ == "__main__":
    # Create a general model manager
    model_manager = ModelManager()

    # RealESRGAN plug-in
    real_esrgan_plugin = RealESRGANPlugin(model_manager)

    # List all RealESRGAN models with their file paths
    print("Available RealESRGAN models with their paths:")
    print(real_esrgan_plugin.list_real_esrgan_models())

    # Download all RealESRGAN models
    # real_esrgan_plugin.download_real_esrgan_models(overwrite="ignore")

    # List all model paths across all sources
    print("All registered model paths:")
    print(model_manager.list_model_paths())

    # Access the manager and register more plugins if needed
    manager = real_esrgan_plugin.get_manager()

#%%
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

STANDARD_SIZE_FOR_QHD = 1920  # Target size

def compute_scale_factor(image_width: int, image_height: int, target_size: int = STANDARD_SIZE_FOR_QHD) -> int:
    """
    Compute the scale factor based on the input image dimensions to ensure both dimensions meet or exceed the target size.
    This function calculates the minimum power of 2 scaling factor (e.g., 2, 4, 8) to efficiently upscale the image.
    """
    max_dim = max(image_width, image_height)
    scale_factor = 1

    # Calculate the smallest scale factor (power of 2) that makes the max dimension >= target size
    while max_dim * scale_factor < target_size:
        scale_factor *= 2

    return scale_factor

def upscale_image(input_image_path: str, output_image_path: str, model_path: str, target_size: int = STANDARD_SIZE_FOR_QHD):
    """
    Upscale an image using the Real-ESRGAN model until both dimensions meet or exceed the target size.

    This function utilizes Real-ESRGAN to upscale an image by calculating the necessary scale factor 
    (in powers of 2) and applying the upscaling through the RealESRGANer class.

    Difference Between RRDBNet and RealESRGANer:
    -------------------------------------------
    - RRDBNet is the deep learning model (Residual-in-Residual Dense Block Network) used for super-resolution tasks. 
      It processes the image data during upscaling and serves as the core convolutional neural network (CNN) 
      responsible for the actual super-resolution.
    
    - RealESRGANer is the inference wrapper around RRDBNet. It handles the upscaling process by integrating 
      additional functionalities such as tiling (for large images), padding, and performance optimizations 
      like half-precision (FP16). It abstracts the complexity of RRDBNet and provides a high-level interface 
      for efficient image upscaling, making the process more memory-efficient and user-friendly.

    Args:
        input_image_path (str): Path to the input image file.
        output_image_path (str): Path to save the upscaled image.
        model_path (str): Path to the Real-ESRGAN model (.pth file).
        target_size (int): The target size for either width or height (default is 1920, suitable for QHD).

    Returns:
        None: The function saves the upscaled image to the specified output path.

    Example usage:
    --------------
    >>> model_path = "path_to_your_model/realesr-general-x4v3.pth"
    >>> input_image = "input_image.jpg"
    >>> output_image = "output_image.png"
    >>> upscale_image(input_image, output_image, model_path, target_size=1920)
    """
    # Load the Real-ESRGAN model (RRDBNet architecture)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # Initialize the upsampler (RealESRGANer)
    upsampler = RealESRGANer(
        scale=4,  # The model is trained for 4x upscaling
        model_path=model_path,
        model=model,
        tile=0,  # You can use tiling if you run out of memory, 0 disables it
        tile_pad=10,
        pre_pad=0,
        half=True  # Use FP16 for faster and more memory-efficient inference
    )
    
    # Read the input image
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    
    # Compute the scale factor needed to meet the target size
    scale_factor = compute_scale_factor(width, height, target_size)

    print(f"Original dimensions: {width}x{height}")
    print(f"Computed scale factor: {scale_factor}")

    # Decompose the scale_factor into powers of 2 and upscale accordingly
    while scale_factor > 1:
        if scale_factor >= 4:
            img, _ = upsampler.enhance(img, outscale=4)
            scale_factor //= 4  # Reduce the scale factor by 4
        elif scale_factor == 2:
            img, _ = upsampler.enhance(img, outscale=2)
            scale_factor //= 2  # Reduce the scale factor by 2

    # Save the upscaled image
    cv2.imwrite(output_image_path, img)
    print(f"Upscaled image saved to {output_image_path}")


# # Example usage
model_path = "path_to_your_model/realesr-general-x4v3.pth"  # Change to the correct model path
# input_image_path = "input_image.jpg"
# output_image_path = "output_image.png"

# upscale_image(input_image_path, output_image_path, model_path, target_size=STANDARD_SIZE_FOR_QHD)
#%%
import subprocess
import shutil
from pathlib import Path
from typing import Optional
import zipfile
import tempfile
import os


class ModelManager:
    def __init__(self, model_dir: Optional[Path] = None, executable_dir: Optional[Path] = None):
        """
        Initialize the model and executable manager.

        Args:
            model_dir (Optional[Path], optional): Directory path where models will be stored. Defaults to ~/ai_models.
            executable_dir (Optional[Path], optional): Directory where executables will be stored. Defaults to ~/ai_models/executables.
        """
        # Set model directory (default: ~/ai_models)
        if model_dir is None:
            model_dir = Path.home() / "ai_models"
        self.model_dir: Path = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Set executable directory (default: ~/ai_models/executables)
        if executable_dir is None:
            executable_dir = Path.home() / "ai_models" / "executables"
        self.executable_dir: Path = executable_dir
        self.executable_dir.mkdir(parents=True, exist_ok=True)

        # Dictionary to store models and executables
        self.models: dict[str, dict[str, dict[str, str]]] = {}
        self.executables: dict[str, dict[str, dict[str, str]]] = {}

    # ====================== MODEL MANAGEMENT =========================== #
    def register_model(self, source: str, model_name: str, model_url: str) -> None:
        """
        Register a new model from a specific source (e.g., RealESRGAN).

        Args:
            source (str): The source of the model (e.g., 'RealESRGAN').
            model_name (str): The name of the model.
            model_url (str): The URL to download the model.
        """
        model_filename = model_url.split("/")[-1]
        model_path = self.model_dir / model_filename

        if source not in self.models:
            self.models[source] = {}

        # Register model with its URL and calculated path
        self.models[source][model_name] = {
            "url": model_url,
            "path": str(model_path),  # Save path as string
        }

    def download_and_update_model_paths(self, source: str, overwrite: Optional[str] = "ignore") -> None:
        """
        Download all models from a registered source and update their paths.
        If the model already exists, only the path will be updated.

        Args:
            source (str): The source of the models to download (e.g., 'RealESRGAN').
            overwrite (Optional[str], optional): Determines what to do if the file already exists.
                                                 Options are 'ignore' (default), 'overwrite', or 'delete'.
        """
        if source not in self.models:
            raise ValueError(f"No models registered under source '{source}'.")

        for model_name in self.models[source]:
            self.download_model(source, model_name, overwrite)
    def download_model(self, source: str, model_name: str, overwrite: Optional[str] = "ignore") -> Path:
        """
        Download a specific model from a registered source and grant execute permissions.

        Args:
            source (str): The source of the model (e.g., 'RealESRGAN').
            model_name (str): The name of the model.
            overwrite (Optional[str], optional): Determines what to do if the file already exists.

        Returns:
            Path: The path where the downloaded model is saved.
        """
        if source not in self.models or model_name not in self.models[source]:
            raise ValueError(f"Invalid source '{source}' or model name '{model_name}'.")

        model_info = self.models[source][model_name]
        model_url: str = model_info["url"]
        model_path: Path = Path(model_info["path"])

        if model_path.exists() and overwrite == "ignore":
            print(f"Model '{model_name}' already exists at {model_path}. Skipping download.")
            return model_path
        elif model_path.exists() and overwrite == "delete":
            model_path.unlink()

        command = ["wget", model_url, "-O", str(model_path)]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Model '{model_name}' downloaded and saved to {model_path}")
        else:
            raise RuntimeError(f"Error downloading the model: {result.stderr}")

        # Grant execute permissions to the model file
        model_path.chmod(0o755)  # Read, write, and execute for the owner; read and execute for others

        return model_path

    def list_model_paths(self, source: Optional[str] = None) -> dict[str, str]:
        """
        List the paths of all registered models, either from a specific source or from all sources.

        Args:
            source (Optional[str], optional): The source of the models to list paths for. If None, lists all models.

        Returns:
            dict: A dictionary of model names and their paths for the specified source or all sources.
        """
        if source:
            if source not in self.models:
                raise ValueError(f"Invalid source '{source}'.")
            return {model: self.models[source][model]["path"] for model in self.models[source]}
        return {model: self.models[source_name][model]["path"] for source_name in self.models for model in self.models[source_name]}


    # ====================== EXECUTABLE MANAGEMENT =========================== #

    def register_executable(self, source: str, executable_name: str, executable_url: str) -> None:
        """
        Register a new executable from a specific source.

        Args:
            source (str): The source of the executable (e.g., 'RealESRGAN').
            executable_name (str): The name of the executable.
            executable_url (str): The URL to download the executable.
        """
        executable_filename = executable_url.split("/")[-1]
        executable_path = self.executable_dir / executable_filename

        if source not in self.executables:
            self.executables[source] = {}

        self.executables[source][executable_name] = {
            "url": executable_url,
            "path": str(executable_path),  # Initially pointing to the archive
        }

    def download_and_update_executable_paths(self, source: str, overwrite: Optional[str] = "ignore") -> None:
        """
        Download all executables from a registered source and update their paths.
        If the executable already exists, only the path will be updated.

        Args:
            source (str): The source of the executables to download (e.g., 'RealESRGAN').
            overwrite (Optional[str], optional): Determines what to do if the file already exists.
                                                 Options are 'ignore' (default), 'overwrite', or 'delete'.
        """
        if source not in self.executables:
            raise ValueError(f"No executables registered under source '{source}'.")

        for executable_name in self.executables[source]:
            self.download_and_update_executable_path(source, executable_name, overwrite)

    def download_and_update_executable_path(self, source: str, executable_name: str, overwrite: Optional[str] = "ignore") -> Path:
        """
        Download and extract a specific executable from a registered source, and update the path
        to point to the actual executable (not the archive). Also grants execute permissions.

        Args:
            source (str): The source of the executable.
            executable_name (str): The name of the executable.
            overwrite (Optional[str], optional): Determines what to do if the file already exists.

        Returns:
            Path: The path to the main executable file.
        """
        if source not in self.executables or executable_name not in self.executables[source]:
            raise ValueError(f"Invalid source '{source}' or executable name '{executable_name}'.")

        executable_info = self.executables[source][executable_name]
        executable_url: str = executable_info["url"]
        executable_archive_path: Path = Path(executable_info["path"])

        extract_dir = self.executable_dir / executable_name

        if extract_dir.exists() and overwrite == "ignore":
            print(f"Executable '{executable_name}' already exists at {extract_dir}. Skipping download.")
            main_executable = self.find_main_executable(extract_dir)
            self.executables[source][executable_name]['path'] = str(main_executable)
            return main_executable
        
        elif extract_dir.exists() and overwrite == "delete":
            shutil.rmtree(extract_dir, ignore_errors=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / executable_url.split("/")[-1]
            command = ["wget", executable_url, "-O", str(tmp_path)]
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Error downloading the executable: {result.stderr}")

            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

        main_executable = self.find_main_executable(extract_dir)
        self.executables[source][executable_name]['path'] = str(main_executable)

        # Grant execute permissions to the main executable
        main_executable.chmod(0o755)  # Read, write, and execute for the owner; read and execute for others

        return main_executable

    def find_main_executable(self, directory: Path) -> Path:
        """
        Find the main executable file in a given directory (file without an extension).

        Args:
            directory (Path): The directory to search for the main executable.

        Returns:
            Path: The path to the main executable file.
        """
        for item in directory.iterdir():
            if item.is_file() and not item.suffix:  
                return item

        raise FileNotFoundError(f"No main executable found in directory {directory}")

        raise FileNotFoundError(f"No main executable found in directory {directory}")
    def list_executable_paths(self, source: Optional[str] = None) -> dict[str, str]:
        """
        List the paths of all registered executables, either from a specific source or from all sources.

        Args:
            source (Optional[str], optional): The source of the executables to list paths for. If None, lists all executables.

        Returns:
            dict: A dictionary of executable names and their paths for the specified source or all sources.
        """
        if source:
            if source not in self.executables:
                raise ValueError(f"Invalid source '{source}'.")
            return {exe: self.executables[source][exe]["path"] for exe in self.executables[source]}
        return {exe: self.executables[source_name][exe]["path"] for source_name in self.executables for exe in self.executables[source_name]}

    # ====================== DOWNLOAD ALL (FOR BOTH MODELS AND EXECUTABLES) =========================== #

    def download_all(self, source: Optional[str] = None, overwrite: Optional[str] = "ignore") -> None:
        """
        Download all models and executables from a registered source or from all sources.

        Args:
            source (Optional[str], optional): The source of the models and executables to download (e.g., 'RealESRGAN').
                                              If None, downloads all models and executables from all sources.
            overwrite (Optional[str], optional): Determines what to do if the file already exists.
                                                 Options are 'ignore' (default), 'overwrite', or 'delete'.
        """
        # Download all models for a specific source or all sources
        if source:
            if source in self.models:
                for model_name in self.models[source]:
                    self.download_model(source, model_name, overwrite)
            if source in self.executables:
                for executable_name in self.executables[source]:
                    self.download_and_update_executable_path(source, executable_name, overwrite)
        else:
            for source_name in self.models:
                for model_name in self.models[source_name]:
                    self.download_model(source_name, model_name, overwrite)
            for source_name in self.executables:
                for executable_name in self.executables[source_name]:
                    self.download_and_update_executable_path(source_name, executable_name, overwrite)


class RealESRGANPlugin:
    """
    RealESRGAN Plugin to register and manage models and executables for RealESRGAN using the ModelManager.

    This plugin allows users to manage only RealESRGAN models and executables,
    including downloading and updating their paths.

    Example Usage:
    --------------
    >>> model_manager = ModelManager()
    >>> real_esrgan_plugin = RealESRGANPlugin(model_manager)

    # List RealESRGAN models
    >>> print(real_esrgan_plugin.list_real_esrgan_models())

    # Download and update all RealESRGAN models
    >>> real_esrgan_plugin.download_and_update_real_esrgan_model_paths(overwrite="ignore")

    # Get model path for a specific model
    >>> model_path = real_esrgan_plugin.get_model_path("realesr-general-x4v3")
    >>> print(f"Model path: {model_path}")

    # List all RealESRGAN executables
    >>> print(real_esrgan_plugin.list_real_esrgan_executables())

    # Download and update all RealESRGAN executables
    >>> real_esrgan_plugin.download_and_update_real_esrgan_executable_paths(overwrite="ignore")

    # Get executable path for a specific executable
    >>> executable_path = real_esrgan_plugin.get_executable_path("realesrgan-ncnn-vulkan")
    >>> print(f"Executable path: {executable_path}")
    """


    def __init__(self, manager: ModelManager):
        """
        Initialize the RealESRGAN plugin with the provided ModelManager instance.

        Args:
            manager (ModelManager): An instance of the ModelManager to manage models and executables.
        """
        self.manager = manager

        # Register Real-ESRGAN models
        self.manager.register_model(
            "RealESRGAN",
            "realesr-general-x4v3",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        )
        self.manager.register_model(
            "RealESRGAN",
            "realesr-general-wdn-x4v3",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        )

        # Register Real-ESRGAN executables
        self.manager.register_executable(
            "RealESRGAN",
            "realesrgan-ncnn-vulkan",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
        )
        
    # ===================== MODEL MANAGEMENT ====================== #

    def download_and_update_real_esrgan_model_paths(self, overwrite: Optional[str] = "ignore") -> None:
        """
        Download all registered RealESRGAN models and update their paths.
        If the model already exists, only the path will be updated.

        Args:
            overwrite (Optional[str], optional): Determines what to do if the file already exists.
                                                 Options are 'ignore' (default), 'overwrite', or 'delete'.
        """
        self.manager.download_and_update_model_paths("RealESRGAN", overwrite)

    def list_real_esrgan_models(self) -> dict[str, str]:
        """
        List all registered RealESRGAN models with their paths.

        Returns:
            dict[str, str]: A dictionary of RealESRGAN model names and their file paths.
        """
        return self.manager.list_model_paths("RealESRGAN")

    def get_model_path(self, model_name: str) -> str:
        """
        Retrieve the full path of a registered RealESRGAN model from the ModelManager.

        Args:
            model_name (str): The name of the model (e.g., "realesr-general-x4v3").

        Returns:
            str: The file path of the requested model.
        """
        model_paths = self.manager.list_model_paths("RealESRGAN")
        if model_name in model_paths:
            return model_paths[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found.")

    # ===================== EXECUTABLE MANAGEMENT ====================== #

    def download_and_update_real_esrgan_executable_paths(self, overwrite: Optional[str] = "ignore") -> None:
        """
        Download all registered RealESRGAN executables and update their paths.
        If the executable already exists, only the path will be updated.

        Args:
            overwrite (Optional[str], optional): Determines what to do if the file already exists.
                                                 Options are 'ignore' (default), 'overwrite', or 'delete'.
        """
        self.manager.download_and_update_executable_paths("RealESRGAN", overwrite)

    def list_real_esrgan_executables(self) -> dict[str, str]:
        """
        List all registered RealESRGAN executables with their paths.

        Returns:
            dict[str, str]: A dictionary of RealESRGAN executable names and their file paths.
        """
        return self.manager.list_executable_paths("RealESRGAN")

    def get_executable_path(self, executable_name: str) -> str:
        """
        Retrieve the full path of a registered RealESRGAN executable from the ModelManager.

        Args:
            executable_name (str): The name of the executable (e.g., "realesrgan-ncnn-vulkan").

        Returns:
            str: The file path of the requested executable.
        """
        executable_paths = self.manager.list_executable_paths("RealESRGAN")
        if executable_name in executable_paths:
            return executable_paths[executable_name]
        else:
            raise ValueError(f"Executable '{executable_name}' not found.")
#%%
if __name__ == "__main__":
    model_manager = ModelManager()
    real_esrgan_plugin = RealESRGANPlugin(model_manager)

    # List RealESRGAN models
    print(real_esrgan_plugin.list_real_esrgan_models())

    # Download and update all RealESRGAN models
    real_esrgan_plugin.download_and_update_real_esrgan_model_paths(overwrite="ignore")

    # Get model path for a specific model
    model_path = real_esrgan_plugin.get_model_path("realesr-general-x4v3")
    print(f"Model path: {model_path}")

    # List all RealESRGAN executables
    print(real_esrgan_plugin.list_real_esrgan_executables())

    # Download and update all RealESRGAN executables
    real_esrgan_plugin.download_and_update_real_esrgan_executable_paths(overwrite="ignore")

    # Get executable path for a specific executable
    executable_path = real_esrgan_plugin.get_executable_path("realesrgan-ncnn-vulkan")
    print(f"Executable path: {executable_path}")
    
# %%
## 출력 경로를 따로 만들어줘야함..
 -m /home/wbfw109/ai_models/executables/realesrgan-ncnn-vulkan/models
r

realesrgan-ncnn-vulkan -i ~/crawling/converted_images_test_sample -o ~/crawling/converted_images_test_sample_output \
-s 4 -t 128 -n realesrgan-x4plus -j 2:4:4 -f png -x -v


#%%
nvidia-smi
sudo apt update -y
sudo apt install -y libvulkan1 vulkan-utils

sudo apt install libvulkan1 vulkan-tools
vulkaninfo
