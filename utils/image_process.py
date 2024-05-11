import os
import pandas as pd
import imagehash
from PIL import Image
import tqdm


def image_category(directory):
    cat_list = pd.DataFrame(os.listdir(directory), columns=["Category"])
    for i in cat_list["Category"].values:
        if i == ".DS_Store" or i == "README.txt":
            cat_list.drop(cat_list[cat_list["Category"] == i].index, inplace=True)
    cat_list.reset_index(drop=True, inplace=True)
    return cat_list["Category"].values


def image_style(directory):
    style_list = pd.DataFrame(os.listdir(f"{directory}/beds"), columns=["Style"])
    for i in style_list["Style"].values:
        if i == ".DS_Store" or i == "README.txt":
            style_list.drop(style_list[style_list["Style"] == i].index, inplace=True)
    style_list.reset_index(drop=True, inplace=True)
    return style_list["Style"].values


def image_path(directory, category):
    style = image_style(directory)
    image_files = []

    for st in style:
        path = f"{directory}/{category}/{st}"
        for file in os.listdir(path):
            image_files.append(f"{path}/{file}")

    return image_files


def compute_hash(image_path):
    with Image.open(image_path) as img:
        hash_value = imagehash.average_hash(img)
    return hash_value


def image_duplicate(image_files):
    hashes = {}
    # Dictionary to store duplicates
    duplicates = {}

    for image_file in image_files:
        hash_value = compute_hash(image_file)
        if hash_value in hashes:
            original_file = hashes[hash_value]
            duplicates.setdefault(original_file, []).append(image_file)
        else:
            hashes[hash_value] = image_file

    # # Print duplicates
    # for original_file, duplicate_files in duplicates.items():
    #     print(f"Original: {original_file}")
    #     print("Duplicates:")
    #     for file in duplicate_files:
    #         print(file)
    #     print()
    return duplicates


# image_duplicate(image_path())
# image_duplicate(image_path('beds'))


# Image get size function: A function use to get the size of each image in (width, height) formation with unit of pixels, add the part where the function
# will count the total of images in that size
def imgSizeList(lists):
    count1 = 0
    count2 = 0
    count3 = 0
    imageSize = []
    for item in enumerate(lists):
        img = Image.open(item[1])
        imageSize.append(img.size)
        if img.size == (224, 224):
            count1 += 1
        if img.size == (350, 350):
            count2 += 1
        if img.size != (224, 224):
            if img.size != (350, 350):
                count3 += 1
    print("224x224 pixels: ", count1)
    print("350x350 pixels: ", count2)
    print("Other size: ", count3)


def imgResize(lists, size):
    for item in enumerate(lists):
        img = Image.open(item[1])
        width, height = img.size
        img1 = img.resize(size, resample=0)
        img1.save(item[1], "JPEG")


def img_dupChecks(lists):
    count = 0
    dupli = image_duplicate(lists)
    for item in dupli:
        count += 1
    print("Number of duplicants: ", count)
    for key in lists:
        if key in dupli:
            for duplicated_key in dupli[key]:
                lists.remove(duplicated_key)
    print("Duplicants has been removed!")
