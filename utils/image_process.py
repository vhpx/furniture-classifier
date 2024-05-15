import os
import shutil
import pickle
import pandas as pd
import imagehash
from PIL import Image
from collections import defaultdict, Counter
from tqdm.notebook import tqdm
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


def get_majority_class(data):
    """
    Identifies the majority class in the dataset and returns its name and count.

    Args:
        data (dict): Dictionary containing image paths and sizes for each category.

    Returns:
        tuple: A tuple containing the name of the majority class and its count.
    """
    class_counts = {category: len(items["paths"]) for category, items in data.items()}
    majority_class = max(class_counts, key=class_counts.get)
    return majority_class, class_counts[majority_class]


def identify_minority_classes(data, threshold_ratio=0.8):
    """
    Identifies minority classes in the dataset based on a threshold ratio compared to the majority class.

    Args:
        data (dict): Dictionary containing image paths and sizes for each category.
        threshold_ratio (float, optional): The ratio below which a class is considered a minority. Defaults to 0.8.

    Returns:
        list: A list of minority class names.
    """
    class_counts = {category: len(items["paths"]) for category, items in data.items()}
    max_count = max(class_counts.values())
    minority_classes = [
        category
        for category, count in class_counts.items()
        if count / max_count < threshold_ratio
    ]
    return minority_classes


def calculate_category_oversampling(data, minority_classes, target_count=None):
    """
    Calculates the number of images to be augmented for each category.

    Args:
        data (dict): Dictionary containing image paths and sizes for each category.
        minority_classes (list): List of minority class names.
        target_count (int, optional): Target number of images per minority class. If None, matches the majority class count.

    Returns:
        dict: Dictionary where the keys are the category names and the values are the number of images to be augmented.
    """
    if target_count is None:
        class_counts = {
            category: len(items["paths"]) for category, items in data.items()
        }
        majority_class = max(class_counts, key=class_counts.get)
        target_count = class_counts[majority_class]

    oversampling = {
        category: max(0, target_count - len(data[category]["paths"]))
        for category in minority_classes
    }

    return oversampling


def calculate_style_oversampling(data, minority_classes, target_count=None):
    """Calculates the number of images to augment for each style within each category."""
    oversampling = {}

    if target_count is None:
        all_style_counts = []
        for category in data:
            style_counts = Counter(
                os.path.dirname(path).split("/")[-1] for path in data[category]["paths"]
            )
            all_style_counts.extend(style_counts.values())
        target_count = max(all_style_counts)

    for category in minority_classes:
        style_counts = Counter(
            os.path.dirname(path).split("/")[-1] for path in data[category]["paths"]
        )
        oversampling[category] = {
            style: max(0, target_count - count) for style, count in style_counts.items()
        }

    return oversampling


def delete_augmented_files(directory):
    """Deletes all files in a directory that start with 'aug_' and returns the number of deleted files and affected directories."""
    deleted_files_count = 0
    affected_directories = set()

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("aug_"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                deleted_files_count += 1
                affected_directories.add(root)

    print(
        f"Deleted {deleted_files_count} files from {len(affected_directories)} directories."
    )
    print("Affected directories:")
    for directory in affected_directories:
        print(directory)


def augment_images(data, category, style, num_augmentations=2):
    """Augments images of a specific style within a category."""
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    image_list = [
        path
        for path in data[category]["paths"]
        if style in path and CLEANED_DATASET_DIR in path
    ]

    save_dir = f"{PROCESSED_DATASET_DIR}/{category}/{style}"
    os.makedirs(save_dir, exist_ok=True)
    augmented_images = []

    # Count the number of existing augmentations
    existing_augmentations = len(
        [name for name in os.listdir(save_dir) if name.startswith("aug_")]
    )
    num_augmentations -= existing_augmentations

    if num_augmentations > 0:
        augmentations_per_image = num_augmentations // len(image_list)
        remaining_augmentations = num_augmentations % len(image_list)

        # Calculate the starting image index
        start_image_index = existing_augmentations // (augmentations_per_image + 1)

        # Initialize tqdm progress bar with leave=False
        pbar = tqdm(
            total=num_augmentations,
            desc=f"Augmenting images for '{category}/{style}'",
            leave=False,
        )

        for i, image_path in enumerate(image_list[start_image_index:]):
            num_augmentations_for_this_image = augmentations_per_image
            if i < remaining_augmentations:
                num_augmentations_for_this_image += 1

            for _ in range(num_augmentations_for_this_image):
                img = tf.keras.preprocessing.image.load_img(image_path)
                x = tf.keras.preprocessing.image.img_to_array(img)
                x = x.reshape((1,) + x.shape)

                augmented_img = datagen.flow(x, batch_size=1)[0][0]
                augmented_img_path = os.path.join(
                    save_dir,
                    f"aug_{os.path.basename(image_path)}_{existing_augmentations}.jpg",
                )

                tf.keras.preprocessing.image.save_img(augmented_img_path, augmented_img)
                augmented_images.append(augmented_img_path)

                existing_augmentations += 1

                # Update the progress bar
                pbar.update(1)

        # Close the progress bar
        pbar.close()

    # Update the data dictionary with the augmented image paths
    data[category]["paths"].extend(augmented_images)

    return data  # Return the updated data dictionary


def oversample_minority_classes(data, minority_classes, verbose=True):
    """
    Oversamples minority classes using data augmentation to reach a target count or match the majority class.

    Args:
        data (dict): Dictionary containing image paths and sizes for each category.
        minority_classes (list): List of minority class names.
        verbose (bool, optional): Whether to display a progress bar. Defaults to True.

    Returns:
        dict: Updated data dictionary with oversampled minority classes.
    """
    cache_file = os.path.join(PROCESSED_DATASET_DIR, "oversample_cache.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    style_counts = calculate_style_oversampling(data, minority_classes)

    total_images_to_augment = sum(
        sum(counts.values()) for counts in style_counts.values()
    )

    if verbose:
        pbar = tqdm(total=total_images_to_augment, desc="Oversampling minority classes")
    else:
        pbar = None

    for category in minority_classes:
        if category not in cache:
            cache[category] = []

        for style, count in style_counts[category].items():
            if count > 0:
                data = augment_images(data, category, style, num_augmentations=count)
                cache[category].extend(data[category]["paths"])

                # Update the progress bar immediately after the augmentations are created
                if pbar is not None:
                    pbar.update(count)

    if pbar is not None:
        pbar.close()

    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

    return data
