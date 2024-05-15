# If you are running this pipeline on Google Colab,
# please update ENVIRONMENT to "GOOGLE_COLAB".
# Otherwise, please update ENVIRONMENT to "LOCAL".

ENVIRONMENT = "LOCAL"

# Recommended to set to True if the data has changed,
# otherwise set to "False" to save time
BYPASS_CACHE = False

# Recommended to set to True if you want to refresh raw dataset
# from the source (Zip file), otherwise set to False to save time
FORCE_UNZIP = False

# Set the random seed for reproducibility
RANDOM_SEED = 42

# Set the test and evaluation sizes
TEST_SIZE = 0.2
EVAL_SIZE = 0.1

# Calculate the train size
TRAIN_SIZE = 1 - TEST_SIZE - EVAL_SIZE

# Set global directories
DEFAULT_DIR = "/content" if ENVIRONMENT == "GOOGLE_COLAB" else "."

DATA_DIR = f"{DEFAULT_DIR}/data"
DATASET_DIR = f"{DATA_DIR}/datasets"

# Caching
CACHE_DIR = f"{DATA_DIR}/cache"
DATASET_CACHE_DIR = f"{CACHE_DIR}/datasets"

# Datasets
TRAIN_DATA_DIR = f"{DATASET_DIR}/raw"
CLEANED_TRAIN_DATA_DIR = f"{DATASET_DIR}/cleaned"
PROCESSED_TRAIN_DATA_DIR = f"{DATASET_DIR}/processed"

# Set path to cache visualizations and models
VISUALIZATION_DIR = f"{CACHE_DIR}/visualizations"
MODEL_DIR = f"{CACHE_DIR}/models"
MACOS_DIR = f"{DATASET_DIR}/__MACOSX"

# Path to raw dataset
GOOGLE_DRIVE_DIR = f"{DEFAULT_DIR}/drive"
GOOGLE_DRIVE_ROOT_DIR = f"{GOOGLE_DRIVE_DIR}/MyDrive"

ZIP_FILE = "Furniture_Data.zip"

DATASET_ZIP = (
    f"{GOOGLE_DRIVE_ROOT_DIR}/{ZIP_FILE}"
    if ENVIRONMENT == "GOOGLE_COLAB"
    else f"{DEFAULT_DIR}/{ZIP_FILE}"
)

DATASET_EXTRACT_DIR = (
    f"{DEFAULT_DIR}/Furniture_Data" if ENVIRONMENT == "GOOGLE_COLAB" else DATASET_DIR
)
