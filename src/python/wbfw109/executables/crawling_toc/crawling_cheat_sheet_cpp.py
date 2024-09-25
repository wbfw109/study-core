# TODO
"""
Written in 📅 2024-09-18 22:27:18
1. Image Processing and Crawling
   - If `img src` and `a href` share the same extension, prioritize crawling the image from `a href`.
   - Reason: `img src` might be cropped or resized, affecting image quality when using AI-based enlargement.
   - Next: Implement AI-based image enlargement, but handle potential text distortion carefully.

2. Breadcrumb Structure
   - Modify the breadcrumb structure for better readability.
   - Instead of using "_" or "-" characters, use one space between words and the "🔪" emoji to represent the hierarchy.

3. Bandiviewer Configuration
   - Lock Title Bar: Press the "`" key.
   - Settings:
     - DPI: Set to 200%.
     - Background Color: Black.
     - Border Color: Black.
   - Use "No Filter" to avoid lag from the "Sharpen" filter during zoom or movement.

4. Image Conversion and Enlargement
   - Convert downloaded SVG files to PNG with a white background.
   - Use waifu-2x to upscale all images (ensure height and width are at least 1920px).
   - Add EXIF description data from JSON to the converted images.


❌ Deprecated
There was a previous idea to create an SVG viewer that would display SVG files within a recursive folder structure.
- The viewer would show folders hierarchically, similar to HTML tags (h1, h2, h3).
- It would include a sidebar displaying the current location and the entire classification, similar to the ToS (Table of Sections) feature in MSDN.
- Breadcrumb navigation was also part of the plan.
- The goal was to automatically explore nested directories and generate the viewer without manually specifying file names, allowing zooming with + and - keys.

However, this approach was deprecated as the implementation became overly complex and drawbacks emerged.
Instead, opening `.avif` files with Bandiviewer and utilizing its built-in features was determined to be a more efficient solution.

"""

# %%
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
from wbfw109.libs.file import create_temp_json_file, open_file_in_vscode
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
    open_file_in_vscode(temp_json_file)

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
#### ⚠️ in WSL Waifu and other image scaling ai model doesn't use GPU in WSL.

# Write the updated picture_info to the temporary JSON file
# with json_file_path.open("w", encoding="utf-8") as f:
#     json.dump(picture_info, f, ensure_ascii=False, indent=4)

from pathlib import Path

import pillow_avif  # noqa: F401
from waifu2x_ncnn_py import Waifu2x

STANDARD_SIZE_FOR_QHD = 1920  # 3840/2

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


# Example usage:
input_directory = Path.home() / "crawling" / "converted_images"
output_directory = upscalling_output_dir


# Create a Waifu2x instance
waifu2x_instance = Waifu2x(
    gpuid=0, scale=2, noise=3
)  # Use GPU, scale by 2, and apply noise reduction

# Upscale and convert all PNG images in the input directory

# 사용 예시
my_input_file: str = (
    "~/crawling/converted_images_test_sample/cheat_sheet_cpp-terminology-function_contracts--function_contracts_crop.png"
)
my_output_file: str = (
    "~/crawling/converted_images_test_sample/cheat_sheet_cpp-terminology-function_contracts--function_contracts_crop-tttt.png"
)
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


# %%
"""
⚠️ Vulkan APi..  WSL can not use GPU...  by AI model inference


 -m /home/wbfw109/ai_models/executables/realesrgan-ncnn-vulkan/models
r

./realesrgan-ncnn-vulkan -i ~/crawling/converted_images_test_sample -o ~/crawling/converted_images_test_sample_output \
    -s 4 -t 128 -n realesrgan-x4plus -j 4:8:8 -f png -x -v

# sudo apt install -y libomp5 libvulkan-dev


nvidia-smi
sudo apt update -y
sudo apt install -y libvulkan1 vulkan-utils

sudo apt install libvulkan1 vulkan-tools
vulkaninfo



# https://github.com/microsoft/wslg/issues/1254
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install mesa-vulkan-drivers
sudo apt install mesa-vulkan-drivers:i386
ls /usr/share/vulkan/icd.d/
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/dzn_icd.x86_64.json
vulkaninfo
sudo apt install vulkan-tools
vcube

sudo apt update
sudo apt install nvidia-cuda-toolkit
nvcc --version
nvidia-smi
sudo nano /etc/wsl.conf // 자동 등록?
    [boot]
        systemd=true
// 컴퓨터 재부팅. not wSL.
"""


# %%
a = "cheat_sheet_cpp-standard_randomness-standard_random_distributions_cpp11--random_distributions_crop"
a.replace("-", " - ").replace("_", " ")
"Cheat Sheet Cpp - Standard Randomness - Standard Random Distributions Cpp11 - - Random Distributions Crop"
