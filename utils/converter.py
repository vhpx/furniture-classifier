import os
import pandas as pd
from tqdm.auto import tqdm


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
