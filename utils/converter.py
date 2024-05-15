import os
import pickle
import shutil
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import logging


def convert_to_df(directory, save_path):
    if not os.path.isdir(directory):
        print(f"Directory {directory} does not exist. Skipping...")
        return

    if os.path.isfile(save_path):
        print(f"CSV file already exists at {save_path}. Loading from cache...")
        return pd.read_csv(save_path)

    cat_list = [
        i
        for i in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, i))
        and i not in [".DS_Store", "README.txt"]
    ]

    data = []
    with tqdm(total=len(cat_list), desc="Converting to DataFrame") as pbar:
        for cat in cat_list:
            style_list = os.listdir(f"{directory}/{cat}")
            style_list = [i for i in style_list if i not in [".DS_Store", "README.txt"]]
            for style in style_list:
                category_path = f"{directory}/{cat}/{style}"
                for image_name in os.listdir(category_path):
                    full_path = f"{category_path}/{image_name}"
                    data.append([image_name, cat, style, category_path, full_path])
            pbar.update(1)
    df = pd.DataFrame(
        data, columns=["Image_Name", "Category", "Style", "Category_Path", "Full_Path"]
    )
    df.to_csv(save_path, index=False)
    return df


def prepare_data_for_training(dir_path):
    # Define paths to the data directories
    base_dir = os.path.dirname(dir_path)
    all_dir = os.path.join(base_dir, "all")
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    # Define the cache file path
    cache_file = os.path.join(base_dir, "cache.pkl")

    # If the cache file exists, load the results from it
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Clean directories if they exist
    print("Cleaning directories...")
    for directory in [train_dir, val_dir, test_dir]:
        if os.path.exists(directory):
            for subdir in os.listdir(directory):
                shutil.rmtree(os.path.join(directory, subdir), ignore_errors=True)
        else:
            logging.info(f"Directory {directory} does not exist.")
    print("Directories cleaned.")

    # Get all subdirectories in the "all" directory
    categories = [
        d for d in os.listdir(all_dir) if os.path.isdir(os.path.join(all_dir, d))
    ]

    # Initialize progress bar
    pbar = tqdm(total=len(categories))

    for category in categories:
        print(f"Splitting data for {category}...")
        category_path = os.path.join(all_dir, category)
        styles = [
            d
            for d in os.listdir(category_path)
            if os.path.isdir(os.path.join(category_path, d))
        ]

        # Initialize count variables
        train_count = 0
        val_count = 0
        test_count = 0

        for style in styles:
            style_path = os.path.join(category_path, style)
            images = [
                f
                for f in os.listdir(style_path)
                if os.path.isfile(os.path.join(style_path, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            # Check if there are images in the style
            if images:
                # Split the data into train, validation, and test sets
                train_images, test_images = train_test_split(
                    images, test_size=0.2, random_state=42
                )
                train_images, val_images = train_test_split(
                    train_images, test_size=0.25, random_state=42
                )  # 0.25 x 0.8 = 0.2

                # Create category and style directories in train, val, and test directories
                os.makedirs(os.path.join(train_dir, category, style), exist_ok=True)
                os.makedirs(os.path.join(val_dir, category, style), exist_ok=True)
                os.makedirs(os.path.join(test_dir, category, style), exist_ok=True)

                # Copy the images into the correct directories
                for image in train_images:
                    shutil.copy(
                        os.path.join(style_path, image),
                        os.path.join(train_dir, category, style),
                    )
                for image in val_images:
                    shutil.copy(
                        os.path.join(style_path, image),
                        os.path.join(val_dir, category, style),
                    )
                for image in test_images:
                    shutil.copy(
                        os.path.join(style_path, image),
                        os.path.join(test_dir, category, style),
                    )

                # Increment count variables
                train_count += len(train_images)
                val_count += len(val_images)
                test_count += len(test_images)
        print(
            f"Data split for {category} complete. Tr/Va/Te: {train_count}/{val_count}/{test_count}"
        )

        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    # Save the results to the cache file at the end of the function
    with open(cache_file, "wb") as f:
        pickle.dump((train_dir, val_dir, test_dir), f)
