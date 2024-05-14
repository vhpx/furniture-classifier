import os
import pandas as pd
import imagehash
from PIL import Image
from collections import defaultdict, Counter
from tqdm.auto import tqdm
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Set global directories
DATA_DIR = "./data"
DATASET_DIR = f"{DATA_DIR}/datasets"

CACHE_DIR = f"{DATA_DIR}/cache"
DATASET_CACHE_DIR = f"{CACHE_DIR}/datasets"

TRAIN_DATA_DIR = f"{DATASET_DIR}/raw"
CLEANED_TRAIN_DATA_DIR = f"{DATASET_DIR}/cleaned"
PROCESSED_TRAIN_DATA_DIR = f"{DATASET_DIR}/processed"

# Set path to cache visualizations and models
VISUALIZATION_DIR = f"{CACHE_DIR}/visualizations"
MODEL_DIR = f"{CACHE_DIR}/models"


def image_category(directory):
    cat_list = pd.DataFrame(os.listdir(directory), columns=["Category"])
    for i in cat_list["Category"].values:
        if i == ".DS_Store" or i == "README.txt":
            cat_list.drop(cat_list[cat_list["Category"] == i].index, inplace=True)
    cat_list.reset_index(drop=True, inplace=True)
    return cat_list["Category"].values


def image_style(directory):
    style_list = os.listdir(f"{directory}/beds")
    style_list = [i for i in style_list if i not in [".DS_Store", "README.txt"]]
    return style_list


def image_path(directory, category):
    style = image_style(directory)
    image_files = []
    with tqdm(total=len(style), desc="Getting path") as pbar:
        for st in style:
            path = f"{directory}/{category}/{st}"
            for file in os.listdir(path):
                image_files.append(f"{path}/{file}")
            pbar.update(1)
    return image_files


def compute_hash(image_path):
    with Image.open(image_path) as img:
        hash_value = imagehash.average_hash(img)
    return hash_value


def image_duplicate(image_files):
    hashes = {}
    duplicates = defaultdict(list)
    for image_file in tqdm(
        (compute_hash(image_file) for image_file in image_files),
        total=len(image_files),
        desc="Finding duplicate images",
    ):
        if image_file in hashes:
            original_file = hashes[image_file]
            duplicates[original_file].append(image_file)
        else:
            hashes[image_file] = image_file
    return duplicates


def imgSizeList(lists):
    imageSize = []
    size_counter = Counter()
    with tqdm(total=len(lists), desc="Getting image size") as pbar:
        for item in lists:
            with Image.open(item) as img:
                size = img.size
                imageSize.append(size)
                size_counter[size] += 1
            pbar.update(1)
    print("224x224 pixels: ", size_counter[(224, 224)])
    print("350x350 pixels: ", size_counter[(350, 350)])
    print(
        "Other size: ", len(lists) - size_counter[(224, 224)] - size_counter[(350, 350)]
    )
    return imageSize


def imgResize(lists, size):
    with tqdm(total=len(lists), desc="Resizing images") as pbar:
        for item in lists:
            with Image.open(item) as img:
                img1 = img.resize(size, resample=0)
                img1.save(item[1], "JPEG")
            pbar.update(1)        

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

def random_augmentation(lists):
    numImg = random(len(lists))
    with tqdm(total= numImg, desc="Augmenting images") as pbar:
        for i in numImg:
            randomImg = random.choice(lists)
            img = Image.open(randomImg)
            augmentImg = random_augmentation(img)
            plt.show(augmentImg)
            augmentImg.save(img, 'JPEG')
            pbar.update(1)

def img_dupChecks(lists):
    dupli = image_duplicate(lists)
    print("Number of duplicants: ", len(dupli))
    lists[:] = [item for item in lists if item not in dupli]
    print("Duplicants has been removed!")
