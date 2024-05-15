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
            img = Image.open(item)
            img1 = img.resize(size, resample=0)
            img1.save(item, "JPEG")
            pbar.update(1)


def img_dupChecks(lists):
    dupli = image_duplicate(lists)
    print("Number of duplicants: ", len(dupli))
    lists[:] = [item for item in lists if item not in dupli]
    print("Duplicants has been removed!")


data_process = tf.keras.preprocessing.image.ImageDataGenerator(
    fill_mode="nearest", horizontal_flip=True, vertical_flip=True, rescale=1.0 / 255
)


def img_augment(lists):
    with tqdm(total=len(lists), desc="Resizing images") as pbar:
        for item in lists:
            image = item
            save_path = os.path.split(image)
            new_save = save_path[0]
            image = tf.keras.preprocessing.image.load_img(image)
            x = tf.keras.preprocessing.image.img_to_array(image)
            x = x.reshape((1,) + x.shape)
            i = 0
            for batch in data_process.flow(
                x,
                batch_size=1,
                save_to_dir=new_save,
                save_prefix="Augment",
                save_format="jpg",
            ):
                i += 1
                if i > 20:
                    break
            pbar.update(1)
