import os
import pandas as pd
import os
import imagehash
from PIL import Image



def image_category():
    cat_list = pd.DataFrame(os.listdir('./Furniture_Data'), columns=['Category'])
    for i in cat_list['Category'].values:
        if i == '.DS_Store' or i == 'README.txt':
            cat_list.drop(cat_list[cat_list['Category'] == i].index, inplace=True)
    cat_list.reset_index(drop=True, inplace=True)
    return cat_list['Category'].values

def image_style():
    style_list = pd.DataFrame(os.listdir('./Furniture_Data/beds'), columns=['Style'])
    for i in style_list['Style'].values:
        if i == '.DS_Store' or i == 'README.txt':
            style_list.drop(style_list[style_list['Style'] == i].index, inplace=True)
    style_list.reset_index(drop=True, inplace=True)
    return style_list['Style'].values


def image_path(category):
    style = image_style()
    image_files = []
    

    for st in style:
        path = f'./Furniture_Data/{category}/{st}'
        for file in os.listdir(path):
             image_files.append(f'./Furniture_Data/{category}/{st}/{file}')
    # print(image_files)
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