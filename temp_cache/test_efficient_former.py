# %%
from __future__ import annotations

from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# %doctest_mode

# from google.protobuf.internal import api_implementation
# print(api_implementation.Type())
# %%
# Title: Download Data ; https://www.uni-ulm.de/in/mwt/forschung/online-datenbank/traffic-gesture-dataset/
import zipfile
from pathlib import Path

import requests

# Define constants for unit conversion
BYTES_IN_KB = 1024  # 1 KB = 1024 bytes
BYTES_IN_MB = BYTES_IN_KB * 1024  # 1 MB = 1024 KB = 1024 * 1024 bytes


# Define the base path for the dataset
base_dir: Path = Path.home() / "ml" / "dataset" / "Kern2023"


# Function to create the base directory if it doesn't exist
def create_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    print(f"Directory created or already exists: {path}")


# Function to list remote directory contents like wget does
def list_remote_directory(url: str) -> list[str]:
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to access {url}, status code: {response.status_code}")
        return []

    # Print the HTML content and list file links
    file_links: list[str] = []
    print(f"Listing contents of {url}:")
    for line in response.text.splitlines():
        if ".zip" in line:  # Looking for .zip files
            # Extract the file link from the href attribute
            start = line.find('href="') + len('href="')
            end = line.find(".zip") + len(".zip")
            file_name: str = line[start:end]
            if file_name:
                full_file_url: str = f"{url}/{file_name}"
                file_links.append(full_file_url)
                print(f"Found file: {full_file_url}")

    return file_links


# Function to download a file with progress tracking
def download_file_with_progress(url: str, download_path: Path) -> None:
    file_name: str = url.split("/")[-1]
    destination: Path = download_path / file_name
    extracted_folder: Path = (
        download_path / destination.stem
    )  # Folder name based on the zip file name

    # Check if the extracted folder already exists
    if extracted_folder.exists():
        print(f"Folder {extracted_folder} already exists. Skipping download.")
        return

    # Stream the download to allow progress tracking
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))  # Total size in bytes
    block_size = 1024  # 1 KB chunks

    print(f"Downloading {file_name} to {destination}")

    # Download in chunks and display progress
    with open(destination, "wb") as f:
        downloaded_size = 0
        for data in response.iter_content(block_size):
            downloaded_size += len(data)
            f.write(data)

            # Calculate percentage and format progress
            percent_complete = (
                downloaded_size / total_size * 100 if total_size > 0 else 0
            )
            progress_bar = f"{percent_complete:.2f}%"

            # Print progress, overwriting the previous line
            print(
                f"\rProgress: {progress_bar} of {total_size / BYTES_IN_MB:.2f} MB",
                end="",
            )

    print("\nDownload complete.")

    # Create the folder with the same name as the zip file
    extracted_folder.mkdir(parents=True, exist_ok=True)

    # Unzip the file into the newly created folder
    try:
        with zipfile.ZipFile(destination, "r") as zip_ref:
            zip_ref.extractall(extracted_folder)  # Extract into the created folder
        print(f"Successfully extracted: {destination} to {extracted_folder}")

        # Remove the zip file after successful extraction
        destination.unlink()
        print(f"Removed zip file: {destination}")
    except zipfile.BadZipFile:
        print(f"Error: {destination} is not a valid zip file. Skipping extraction.")


# Function to process and download all files from a list of URLs
def download_all_files(urls: list[str], download_path: Path) -> None:
    for url in urls:
        download_file_with_progress(url, download_path)


# Main function to process multiple URLs in batch
def main(urls: list[str], dry_run: bool = True) -> None:
    # Ensure base directory exists
    create_directory(base_dir)

    for url in urls:
        print(f"\nProcessing URL: {url}")

        # List the contents of the remote directory (like wget does)
        file_links: list[str] = list_remote_directory(url)

        # If not dry-run, download the files
        if not dry_run:
            download_all_files(file_links, base_dir)


# List of URLs of the remote folders you want to download from
urls: list[str] = [
    "https://mwt-www.e-technik.uni-ulm.de/downloads/publicData/Kern2023/gesture_dataset",
    "https://mwt-www.e-technik.uni-ulm.de/downloads/publicData/Kern2023/cont_datasets",
]

if __name__ == "__main__":
    # First, do a dry-run to see what files are available
    main(urls, dry_run=True)

    # Uncomment the following line to download files after the dry-run
    # main(urls, dry_run=False)  # Run this line to download files


# %%
