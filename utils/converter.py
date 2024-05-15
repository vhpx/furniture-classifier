import os
import pandas as pd
from tqdm.auto import tqdm
from utils.image_process import image_category


def convert_to_df(directory, save_path):
    if not os.path.isdir(directory):
        print(f"Directory {directory} does not exist. Skipping...")
        return

    if os.path.isfile(save_path):
        print(f"CSV file already exists at {save_path}. Loading from cache...")
        return pd.read_csv(save_path)

    cat_list = image_category(directory)
    data = []
    with tqdm(total=len(cat_list), desc="Converting to DataFrame") as pbar:
        for cat in cat_list:
            style_list = os.listdir(f"{directory}/{cat}")
            style_list = [i for i in style_list if i not in [".DS_Store", "README.txt"]]
            for style in style_list:
                path = f"{directory}/{cat}/{style}"
                for file in os.listdir(path):
                    data.append([file, cat, style, path])
            pbar.update(1)
    df = pd.DataFrame(data, columns=["Image", "Category", "Style", "Path"])
    df.to_csv(save_path, index=False)
    return df
