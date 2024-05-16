import os
import shutil
import logging
from zipfile import ZipFile
from typing import Optional


def zip_dir(
    directory: str, zip_file: str, include_subdirs: Optional[bool] = True
) -> None:
    """Zip all the files in the directory."""
    if not os.path.isdir(directory):
        logging.error(f"Directory {directory} does not exist.")
        return

    zip_dir = os.path.dirname(zip_file)
    if not os.path.isdir(zip_dir):
        logging.info(
            f"Directory {zip_dir} does not exist. Attempting to create the directory."
        )
        os.makedirs(zip_dir, exist_ok=True)
        logging.info(f"Directory {zip_dir} created.")

    try:
        with ZipFile(zip_file, "w") as zip_file:
            for root, _, files in os.walk(directory):
                if not include_subdirs and root != directory:
                    continue
                for file in files:
                    zip_file.write(os.path.join(root, file), file)
        if os.path.getsize(zip_file.filename) == 0:
            logging.warning(
                f"No files found in {directory}. Created zip file is empty."
            )
        else:
            logging.info(f"Zipped all files in {directory} to {zip_file.filename}")
    except Exception as e:
        logging.error(f"An error occurred while zipping the files: {e}")


def unzip_file(zip_file: str, extract_dir: str) -> None:
    """Unzip the file to the extract directory."""
    if not os.path.isfile(zip_file):
        logging.error(f"File {zip_file} does not exist.")
        return

    if not os.path.isdir(extract_dir):
        logging.info(
            f"Directory {extract_dir} does not exist. Attempting to create the directory."
        )
        os.makedirs(extract_dir, exist_ok=True)
        logging.info(f"Directory {extract_dir} created.")

    try:
        with ZipFile(zip_file, "r") as zip_file:
            zip_file.extractall(extract_dir)
        logging.info(f"Unzipped {zip_file} to {extract_dir}")
    except Exception as e:
        logging.error(f"An error occurred while unzipping the file: {e}")


def remove_if_exists(path: str) -> None:
    """Remove the file or directory if it exists."""
    if os.path.isfile(path):
        os.remove(path)
        print(f"File {path} found and deleted.")
    elif os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Directory {path} found and deleted.")
    else:
        print(f"{path} not found. Skipping deletion.")


def move_dir(source_dir: str, destination_dir: str) -> None:
    """Move the directory to a new location."""
    if os.path.isdir(source_dir):
        shutil.move(source_dir, destination_dir)
        print(f"Directory {source_dir} moved to {destination_dir}.")
    else:
        print(f"Directory {source_dir} does not exist. Skipping move.")


def read_file(file_path):
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            return file.read()
    return None


def write_file(file_path, content):
    with open(file_path, "w") as file:
        file.write(content)


def copy_dir(source_dir: str, destination_dir: str) -> None:
    """Copy the directory to a new location."""
    if os.path.isdir(source_dir):
        shutil.copytree(source_dir, destination_dir)
        print(f"Directory {source_dir} copied to {destination_dir}.")
    else:
        print(f"Directory {source_dir} does not exist. Skipping copy.")
