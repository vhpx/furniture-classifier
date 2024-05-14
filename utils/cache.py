import os
import sys
import importlib
import tensorflow as tf


def reload_custom_libraries():
    modules_for_refresh = [
        "utils.cache",
        "utils.image_process",
    ]

    # Refresh modules if they have been updated
    unfound_modules = []

    for module in modules_for_refresh:
        if module in sys.modules:
            importlib.reload(sys.modules[module])
        else:
            unfound_modules.append(module)

    # Refresh cache for unfound modules
    if unfound_modules:
        print(f"Modules {unfound_modules} do not exist in the current environment.")
        print("Refreshing cache for all modules...")


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
