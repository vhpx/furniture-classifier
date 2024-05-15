import os
import sys
import logging
import importlib
import tensorflow as tf
from zipfile import ZipFile
from typing import Optional


def reload_custom_libraries():
    modules_for_refresh = [
        "utils.cache",
        "utils.constants",
        "utils.converter",
        "utils.image_process",
    ]

    # Refresh modules if they have been updated
    unfound_modules = []

    for module in modules_for_refresh:
        try:
            if module in sys.modules:
                importlib.reload(sys.modules[module])
                logging.info(f"Reloaded {module}")
            else:
                importlib.import_module(module)
                logging.info(f"Imported {module}")
        except ImportError:
            unfound_modules.append(module)
            logging.error(f"Failed to import {module}")

    # Refresh cache for unfound modules
    if unfound_modules:
        logging.error(
            f"Modules {unfound_modules} do not exist in the current environment."
        )
        logging.info("Refreshing cache for all modules...")


def zip_util_libs(
    directory: str, zip_file: str, include_subdirs: Optional[bool] = True
) -> None:
    """Zip all the files in the directory."""
    # Reload libraries before zipping
    logging.info("Reloading custom libraries...")
    reload_custom_libraries()

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


# def load_or_cache_dataset(data_dir, cache_dir):
#     if tf.io.gfile.exists(cache_dir):
#         print("Loading dataset from cache...")
#         train_ds = tf.data.Dataset.load(cache_dir)  # Load from cache
#     else:
#         print("Loading dataset from directory and caching...")
#         train_ds = tf.keras.utils.image_dataset_from_directory(
#             data_dir,
#         )
#         tf.data.Dataset.save(train_ds, cache_dir)  # Save to cache
#     return train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
