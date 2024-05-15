import os
import shutil
import pickle
import pandas as pd
import imagehash
from PIL import Image
from collections import defaultdict, Counter
from tqdm.auto import tqdm
import tensorflow as tf
import logging
import random
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from utils.constants import (
    DATASET_DIR,
    CLEANED_DATASET_DIR,
    PROCESSED_DATASET_DIR,
)

logging.basicConfig(level=logging.INFO)


def get_category_styles(directory, category):
    return [
        i
        for i in os.listdir(f"{directory}/{category}")
        if os.path.isdir(os.path.join(directory, category, i))
        and i not in [".DS_Store", "README.txt"]
    ]


def get_category_image_paths(directory, category):
    styles = get_category_styles(directory, category)
    image_files = []
    with tqdm(
        total=len(styles), desc=f"Getting image path for category '{category}'"
    ) as pbar:
        for style in styles:
            path = f"{directory}/{category}/{style}"
            image_files.extend(
                [
                    f"{path}/{file}"
                    for file in os.listdir(path)
                    if file.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )
            pbar.update(1)
    return image_files


def compute_hash(path):
    with Image.open(path) as img:
        hash_value = imagehash.average_hash(img)
    return hash_value


def image_duplicate(category, image_files, style):
    hashes = {}
    unique_images = []
    style_images = [img for img in image_files if style in img]

    for image_file in tqdm(
        style_images,
        total=len(style_images),
        desc=f"Finding duplicate images in category '{category}', style '{style}'",
    ):
        try:
            image_hash = compute_hash(image_file)
        except PermissionError:
            print(f"Permission denied for file: {image_file}")
            continue

        if image_hash not in hashes:
            hashes[image_hash] = image_file
            unique_images.append(image_file)
    print(
        f"Category '{category}', Style '{style}': Found {len(style_images) - len(unique_images)} duplicate images."
    )
    return unique_images


def delete_unknown_style_images(image_list, styles):
    for image in image_list:
        if not any(style in image for style in styles):
            os.remove(image)
            print(f"Deleted {image} due to unknown style.")


def get_image_sizes(image_list, cache_file="image_sizes.pkl"):
    if os.path.isfile(cache_file):
        with open(cache_file, "rb") as f:
            image_sizes, size_counter = pickle.load(f)
        print("Loaded image sizes from cache.")
    else:
        image_sizes = []
        size_counter = Counter()
        for image in image_list:
            try:
                with Image.open(image) as img:
                    size = img.size
                    image_sizes.append(size)
                    size_counter[size] += 1
            except PermissionError:
                print(f"Permission denied for file: {image}")
        with open(cache_file, "wb") as f:
            pickle.dump((image_sizes, size_counter), f)
        print("Saved image sizes to cache.")
    return image_sizes, size_counter


def process_style_images(category, image_list, style):
    cleaned_dir = os.path.join(CLEANED_DATASET_DIR, category, style)
    cleaned_image_sizes_cache_file = f"{cleaned_dir}/cleaned_image_sizes.pkl"

    if os.path.exists(cleaned_dir) and os.listdir(cleaned_dir):
        print(
            f"Category '{category}', Style '{style}': Cleaned images are already available."
        )
        if os.path.isfile(cleaned_image_sizes_cache_file):
            with open(cleaned_image_sizes_cache_file, "rb") as f:
                cleaned_image_sizes = pickle.load(f)
            print("Loaded cleaned image sizes from cache.")
            return cleaned_image_sizes

    unique_images = image_duplicate(category, image_list, style)

    os.makedirs(cleaned_dir, exist_ok=True)
    cleaned_image_sizes = []
    for image_path in unique_images:
        try:
            shutil.copy(image_path, cleaned_dir)
            with Image.open(
                os.path.join(cleaned_dir, os.path.basename(image_path))
            ) as img:
                cleaned_image_sizes.append(img.size)
        except PermissionError:
            print(f"Permission denied for file: {image_path}")

    print(f"Category '{category}', Style '{style}': Cleaned images saved successfully.")
    with open(cleaned_image_sizes_cache_file, "wb") as f:
        pickle.dump(cleaned_image_sizes, f)
    print("Saved cleaned image sizes to cache.")
    return cleaned_image_sizes


def process_images(image_list, category, cache_dir):
    styles = get_category_styles(DATASET_DIR, category)

    delete_unknown_style_images(image_list, styles)

    image_sizes, size_counter = get_image_sizes(
        image_list, cache_file=f"{cache_dir}/{category}_image_sizes.pkl"
    )

    for size, count in size_counter.items():
        print(f"{size} pixels: ", count)

    cleaned_image_sizes = []
    for style in styles:
        cleaned_image_sizes.extend(process_style_images(category, image_list, style))

    cleaned_size_counter = Counter(cleaned_image_sizes)
    for size, count in cleaned_size_counter.items():
        print(f"Cleaned {size} pixels: ", count)

    print(f"Category '{category}': All styles processed successfully.")
    return image_sizes


def resize_images(data, category, size):
    styles = get_category_styles(DATASET_DIR, category)
    for style in styles:
        try:
            image_list = [path for path in data[category]["paths"] if style in path]
        except KeyError:
            logging.warning(
                f"Style '{style}' not found in category '{category}'. Skipping this style."
            )
            continue
        cache_file = f"{PROCESSED_DATASET_DIR}/{category}/{style}/resized_images.pkl"
        if os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                resized_images = pickle.load(f)
            logging.info(
                f"Loaded {len(resized_images)} resized images from cache for category '{category}', style '{style}'."
            )
        else:
            resized_images = []
            with tqdm(
                total=len(image_list),
                desc=f"Resizing images for '{category}/{style}'",
            ) as pbar:
                for item in image_list:
                    with Image.open(item) as img:
                        img1 = img.resize(size, resample=0)
                        img_path = os.path.split(item)
                        img_path1 = os.path.split(img_path[0])
                        img_path2 = os.path.split(img_path1[0])
                        save_path = os.path.join(
                            PROCESSED_DATASET_DIR,
                            img_path2[1],
                            img_path1[1],
                            img_path[1],
                        )
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        img1.save(save_path, "JPEG")
                        resized_images.append(save_path)
                    pbar.update(1)
            with open(cache_file, "wb") as f:
                pickle.dump(resized_images, f)
            logging.info(
                f"Saved {len(resized_images)} resized images to cache for category '{category}', style '{style}'."
            )
    return data


def augment_images(data, category):
    styles = get_category_styles(DATASET_DIR, category)
    for style in styles:
        try:
            # Change the way we access the image paths
            image_list = [path for path in data[category]["paths"] if style in path]
        except KeyError:
            print(
                f"Style '{style}' not found in category '{category}'. Skipping this style."
            )
            continue
        save_dir = f"{PROCESSED_DATASET_DIR}/{category}/{style}"
        cache_file = f"{save_dir}/augmented_images.pkl"
        if os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                augmented_images = pickle.load(f)
            print(
                f"Loaded augmented images from cache for category '{category}', style '{style}'."
            )
        else:
            augmented_images = []
            with tqdm(total=len(image_list), desc="Augmenting images") as pbar:
                for item in image_list:
                    img_path = os.path.split(item)
                    img_path1 = os.path.split(img_path[0])
                    img_path2 = os.path.split(img_path1[0])
                    save_path = os.path.join(
                        save_dir, img_path2[1], img_path1[1], img_path[1]
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    image = tf.keras.preprocessing.image.load_img(item)
                    x = tf.keras.preprocessing.image.img_to_array(image)
                    x = x.reshape((1,) + x.shape)
                    i = 0
                    for batch in tf.keras.preprocessing.image.ImageDataGenerator().flow(
                        x,
                        batch_size=1,
                        save_to_dir=os.path.dirname(save_path),
                        save_prefix="Augment",
                        save_format="jpg",
                    ):
                        i += 1
                        if i > 20:
                            break
                    augmented_images.append(save_path)
                    pbar.update(1)
            with open(cache_file, "wb") as f:
                pickle.dump(augmented_images, f)
            print(
                f"Saved augmented images to cache for category '{category}', style '{style}'."
            )
    return data
