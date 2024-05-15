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
        "utils.zipper",
        "utils.constants",
        "utils.constants_colab",
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
