ENVIRONMENT = "LOCAL"

# Recommended to set to True if the data has changed,
# otherwise set to "False" to save time
BYPASS_CACHE = False

# Recommended to set to True if you want to refresh raw dataset
# from the source (Zip file), otherwise set to False to save time
FORCE_UNZIP = False

# Set global directories
DEFAULT_DIR = "/content" if ENVIRONMENT == "GOOGLE_COLAB" else "."

DATA_DIR = f"{DEFAULT_DIR}/data"
UTILS_DIR = f"{DEFAULT_DIR}/utils"
ROOT_DATASET_DIR = f"{DATA_DIR}/datasets"
ZIPPED_RESOURCES_DIR = f"{DATA_DIR}/zipped"

GOOGLE_DRIVE_DIR = f"{DEFAULT_DIR}/drive"
GOOGLE_DRIVE_ROOT_DIR = f"{GOOGLE_DRIVE_DIR}/MyDrive"

# Caching
CACHE_DIR = (
    f"{GOOGLE_DRIVE_ROOT_DIR}/cache"
    if ENVIRONMENT == "GOOGLE_COLAB"
    else f"{DEFAULT_DIR}/cache"
)
DATASET_CACHE_DIR = f"{CACHE_DIR}/datasets"

# Datasets
DATASET_DIR = f"{ROOT_DATASET_DIR}/raw"
CLEANED_DATASET_DIR = f"{ROOT_DATASET_DIR}/cleaned"
PROCESSED_DATASET_DIR = f"{ROOT_DATASET_DIR}/processed"

TRAIN_DATA_CSV = f"{ROOT_DATASET_DIR}/raw.csv"
CLEANED_TRAIN_DATA_CSV = f"{ROOT_DATASET_DIR}/cleaned.csv"
PROCESSED_TRAIN_DATA_CSV = f"{ROOT_DATASET_DIR}/processed.csv"

# Set path to cache visualizations and models
VISUALIZATION_DIR = f"{CACHE_DIR}/visualizations"
MODEL_DIR = (
    f"{GOOGLE_DRIVE_ROOT_DIR}/Models"
    if ENVIRONMENT == "GOOGLE_COLAB"
    else f"{CACHE_DIR}/models"
)
MACOS_DIR = f"{ROOT_DATASET_DIR}/__MACOSX"

# Zipped resources for accelerated Google Colab training
RAW_DATASET_ZIP_FILE = "Furniture_Data.zip"
PREPROCESSED_DATASETS_ZIP_FILE = "datasets.zip"
MODEL_TRAINING_DATA_ZIP_FILE = "model_training_data.zip"

RAW_DATASET_ZIP = (
    f"{GOOGLE_DRIVE_ROOT_DIR}/{RAW_DATASET_ZIP_FILE}"
    if ENVIRONMENT == "GOOGLE_COLAB"
    else f"{DEFAULT_DIR}/{RAW_DATASET_ZIP_FILE}"
)

PREPROCESSED_DATASETS_ZIP = (
    f"{GOOGLE_DRIVE_ROOT_DIR}/{PREPROCESSED_DATASETS_ZIP_FILE}"
    if ENVIRONMENT == "GOOGLE_COLAB"
    else f"{DEFAULT_DIR}/{PREPROCESSED_DATASETS_ZIP_FILE}"
)

MODEL_TRAINING_DATA_ZIP = (
    f"{GOOGLE_DRIVE_ROOT_DIR}/{MODEL_TRAINING_DATA_ZIP_FILE}"
    if ENVIRONMENT == "GOOGLE_COLAB"
    else f"{DEFAULT_DIR}/{MODEL_TRAINING_DATA_ZIP_FILE}"
)

DATASET_EXTRACT_DIR = (
    f"{DEFAULT_DIR}/Furniture_Data"
    if ENVIRONMENT == "GOOGLE_COLAB"
    else ROOT_DATASET_DIR
)

PREPROCESSED_DATASETS_EXTRACT_DIR = DATA_DIR
MODEL_TRAINING_DATA_EXTRACT_DIR = DATA_DIR
