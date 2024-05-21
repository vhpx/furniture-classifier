import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, Sequential, models
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm.notebook import tqdm
import keras.backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
import fnmatch
import re
import csv
from PIL import Image


def find_best_model(directories):
    best_model = None
    best_accuracy = float("-inf")

    for directory in directories:
        eval_file = os.path.join(directory, "evaluation.csv")
        if not os.path.exists(eval_file):
            raise FileNotFoundError(f"evaluation.csv not found in {directory}")

        df = pd.read_csv(eval_file)
        df = df.sort_values(by="accuracy", ascending=False)

        for _, row in df.iterrows():
            model_path = row["path"]
            if os.path.exists(model_path):
                if row["accuracy"] > best_accuracy:
                    best_model = model_path
                    best_accuracy = row["accuracy"]
                break

    if best_model is None:
        raise FileNotFoundError("No valid model found")

    model = load_model(best_model, compile=False)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def evaluate_models(model_dir, test_generator, sample_fraction):
    evaluation_results = []

    # Load or create evaluation.csv
    evaluation_csv_path = os.path.join(model_dir, "evaluation.csv")
    if os.path.exists(evaluation_csv_path):
        with open(evaluation_csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for row in reader:
                if len(row) >= 6:
                    evaluation_results.append(
                        [
                            int(row[0]),
                            int(row[1]),
                            float(row[2]),
                            float(row[3]),
                            row[4],
                            float(row[5]),
                        ]
                    )
                else:
                    print(f"Skipping row due to insufficient columns: {row}")
    else:
        with open(evaluation_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["index", "epoch", "accuracy", "loss", "path", "sample_fraction"]
            )

    # Sort the model files by their epoch numbers in descending order
    model_files = sorted(
        (f for f in os.listdir(model_dir) if f.endswith(".h5")),
        key=lambda f: int(f.split("_")[1]) if f.split("_")[1].isdigit() else 0,
        reverse=True,
    )

    print(f"Evaluating {len(model_files)} models in {model_dir}...")

    # Try to load each model in turn until one is found that exists
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            model_parts = model_file.split("_")
            epoch = 0  # Default value
            if (
                len(model_parts) >= 4
                and model_parts[1].isdigit()
                and model_parts[3].replace(".", "", 1).isdigit()
            ):
                epoch = int(model_parts[1])
                accuracy = float(model_parts[3])

                # Check if the same epoch and accuracy are found in the file name
                if any(
                    result[1] == epoch
                    and round(result[2], 4) == round(accuracy, 4)
                    and result[5] == sample_fraction
                    for result in evaluation_results
                ):
                    print(
                        f"Skipping evaluation for model {model_file} as it has already been evaluated."
                    )
                    continue
            else:
                print(
                    f"Model file name {model_file} does not contain expected information. Proceeding without checks."
                )

            print(f"Loading model from {model_path}...")
            model = load_model(model_path, compile=False)
            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Evaluate the model on the test set
            print("Evaluating model...")
            test_loss, test_acc = model.evaluate(test_generator)

            new_model_file = f"epoch_{epoch}_va_{test_acc:.4f}_sf_{sample_fraction}.h5"
            new_model_path = os.path.join(model_dir, new_model_file)

            # Check if the new file name already exists
            if os.path.exists(new_model_path):
                print(
                    f"File {new_model_file} already exists. Removing duplicate file {model_file}."
                )
                os.remove(model_path)
            else:
                os.rename(model_path, new_model_path)
                print(f"Renamed model file to {new_model_file}")

            # Append the evaluation results
            evaluation_results.append(
                [
                    len(evaluation_results),
                    epoch,
                    test_acc,
                    test_loss,
                    new_model_path,
                    sample_fraction,
                ]
            )

            # Write the evaluation results to a CSV file
            print("Writing evaluation results to CSV file...")
            with open(evaluation_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(evaluation_results[-1])

    print("Done!")


def load_best_model_from_checkpoint(
    model_conf,
    initial_model,
    environment,
    root_dir,
):
    save_dir = os.path.join(root_dir, model_conf["save_dir"])
    base_model_file_path = os.path.join(save_dir, "epoch_")
    csv_logger_path = os.path.join(save_dir, "training_log.csv")

    os.makedirs(save_dir, exist_ok=True)

    model_files = os.listdir(save_dir)

    best_va = 0
    best_model_file = None
    best_epoch = 0

    for file in model_files:
        match = re.match(r"epoch_(\d+)_va_(\d+\.\d+)", file)
        if match:
            epoch = int(match.group(1))
            va = float(match.group(2))
            if va > best_va:
                best_va = va
                best_model_file = file
                best_epoch = epoch

    if not os.path.exists(csv_logger_path):
        print(
            "Warning: No training log file found. Searching for best model in directory."
        )
    else:
        log = pd.read_csv(csv_logger_path)
        epoch_val_accuracy = log[["epoch", "val_accuracy"]].values

        # Sort by validation accuracy and then by epoch number
        epoch_val_accuracy = epoch_val_accuracy[(-epoch_val_accuracy[:, 1]).argsort()]

        for epoch, val_accuracy in epoch_val_accuracy:
            # Convert epoch to integer
            epoch = int(epoch)

            # Check for all possible naming conventions
            model_file_1 = f"epoch_{epoch+1}.h5"
            model_file_2 = f"epoch_{epoch}_va_{val_accuracy:.4f}"

            if model_file_1 in model_files:
                model_path = os.path.join(save_dir, model_file_1)
            elif any(
                file.startswith(model_file_2) and file.endswith(".h5")
                for file in model_files
            ):
                model_path = os.path.join(
                    save_dir,
                    next(
                        file
                        for file in model_files
                        if file.startswith(model_file_2) and file.endswith(".h5")
                    ),
                )
            else:
                continue

            model = load_model(model_path, compile=False)

            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            print(f"Loaded model from {model_path}")
            return model, epoch, base_model_file_path, csv_logger_path

    # If no model was returned in the loop above, load the best available model
    if best_model_file:
        model_path = os.path.join(save_dir, best_model_file)
        model = load_model(model_path, compile=False)

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        print(f"Loaded model with best validation accuracy from {model_path}")
        return model, best_epoch, base_model_file_path, csv_logger_path

    print("Warning: No model file found. Using initial model.")
    return initial_model, 0, base_model_file_path, csv_logger_path


def get_best_val_accuracy(csv_logger_path):
    print(f"Checking for validation accuracy log at {csv_logger_path}...")

    if not os.path.exists(csv_logger_path):
        print(f"No log found at {csv_logger_path}.")
        return -np.inf

    with open(csv_logger_path, "r") as f:
        reader = csv.reader(f)
        log = list(reader)

    print(f"Found log with {len(log)} entries.")

    # The validation accuracy is logged in the column named 'val_accuracy'
    val_accuracy_index = log[0].index("val_accuracy")

    # Get the best validation accuracy from the log
    best_val_accuracy = max(float(row[val_accuracy_index]) for row in log[1:])
    print(f"Best validation accuracy found: {best_val_accuracy}")

    return best_val_accuracy


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(
        self,
        filepath,
        monitor="val_accuracy",
        verbose=0,
        save_best_only=False,
        mode="max",
    ):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                "Can save best model only with %s available, skipping." % self.monitor,
                RuntimeWarning,
            )
        else:
            if self.save_best_only:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print(
                            "\nEpoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"
                            % (
                                epoch + 1,
                                self.monitor,
                                self.best,
                                current,
                                self.filepath.format(epoch=epoch, val_accuracy=current),
                            )
                        )
                    self.best = current
                    self.model.save(
                        self.filepath.format(epoch=epoch, val_accuracy=current),
                        overwrite=True,
                    )
            else:
                if self.verbose > 0:
                    print(
                        "\nEpoch %05d: saving model to %s"
                        % (
                            epoch + 1,
                            self.filepath.format(epoch=epoch, val_accuracy=current),
                        )
                    )
                self.model.save(
                    self.filepath.format(epoch=epoch, val_accuracy=current),
                    overwrite=True,
                )


def train_model(
    model=None,
    model_conf=None,
    environment="LOCAL",
    start_epoch=0,
    sample_fraction=None,
    early_stopping_patience=5,
    learning_rate_patience=2,
    epochs=100,
    train_generator=None,
    val_generator=None,
    test_generator=None,
    root_dir=None,
):
    print("Starting model training...")

    # If the model is None, print an error message and return
    if model is None:
        print("No model found")
        return

    # If the model configuration is None, print an error message and return
    if model_conf is None:
        print("No model configuration found")
        return

    # If there is no sample fraction, print an error message and return
    if sample_fraction is None:
        print("No sample fraction found")
        return

    print("Loading model from checkpoint...")
    # Get cache model
    cached_model, start_epoch, base_model_file_path, csv_logger_path = (
        load_best_model_from_checkpoint(
            model_conf,
            model,
            environment,
            root_dir,
        )
    )

    # Update the model
    model = cached_model

    print("Compiling model...")
    # Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Setting up callbacks...")
    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )
    csv_logger = CSVLogger(csv_logger_path, append=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=learning_rate_patience, min_lr=0.00001
    )

    # Define the checkpoint
    checkpoint = CustomModelCheckpoint(
        filepath=f"{base_model_file_path}{{epoch}}_va_{{val_accuracy:.4f}}_sf_{sample_fraction}.h5",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
    )

    print("Computing class weights...")
    # add class weights for imbalanced dataset
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_generator.classes),
        y=train_generator.classes,
    )

    print("Getting best validation accuracy from log...")
    # Get the best validation accuracy from the log
    best_val_accuracy = get_best_val_accuracy(csv_logger_path)

    print("Starting model fit...")
    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, csv_logger, reduce_lr],
        class_weight=dict(enumerate(class_weights)),
        initial_epoch=start_epoch,
    )

    print("Evaluating model on test set...")
    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")


def normalize_path(path):
    # Convert backslashes to forward slashes and remove the leading dot and slash if present
    return path.replace("\\", "/").replace("./", "")


def get_combined_embedding(
    images, category_embedding_model, style_embedding_model, batch_size=32
):
    category_embeddings = category_embedding_model.predict(
        images, batch_size=batch_size, verbose=0
    )

    style_embeddings = style_embedding_model.predict(
        images, batch_size=batch_size, verbose=0
    )

    return np.concatenate([category_embeddings, style_embeddings], axis=-1)


def get_combined_embeddings(
    generator,
    category_embedding_model,
    style_embedding_model,
    batch_size=32,
    return_index_mapping=False,
    cache_file="combined_embeddings.npy",
):
    if os.path.exists(cache_file):
        if return_index_mapping:
            return np.load(cache_file, allow_pickle=True)
        else:
            return np.load(cache_file, allow_pickle=True)[0]

    all_embeddings = []
    image_paths = []  # Store image paths for mapping

    for batch_idx in tqdm(range(len(generator)), desc="Generating embeddings"):
        batch_images, _ = generator[batch_idx]
        batch_embeddings = get_combined_embedding(
            batch_images, category_embedding_model, style_embedding_model, batch_size
        )
        all_embeddings.extend(batch_embeddings)
        image_paths.extend(
            generator.filenames[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        )  # Get paths from generator

    # Create the mapping from image paths to embedding indices
    embedding_index_mapping = {
        normalize_path(path): i for i, path in enumerate(image_paths)
    }

    if return_index_mapping:
        np.save(cache_file, all_embeddings)  # Save embeddings separately
        np.save("embedding_index_mapping.npy", embedding_index_mapping)
        return np.array(all_embeddings), embedding_index_mapping
    else:
        result = np.array(all_embeddings)
        np.save(cache_file, result, allow_pickle=True)
        return result


def generate_kmeans_embeddings(
    embedding_dim, dataframes, model, output_dir, sample_fraction, df
):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    embeddings_file_path = os.path.join(
        output_dir, f"embeddings_sf_{sample_fraction}.csv"
    )

    if os.path.exists(embeddings_file_path):
        df_feature_vector = pd.read_csv(embeddings_file_path)
        # df_feature_vector['Style'] = ''
    else:
        embeddings = {"Image_Path": [], "Category": []}
        for i in range(embedding_dim):
            embeddings[f"Feature {i}"] = []

        total_images = len(dataframes)
        processed_images = 0

        ## Iterate over the rows of the filtered dataframe
        for i, row in dataframes.iterrows():
            processed_images += 1
            progress_percent = (processed_images / total_images) * 100
            print(
                f"Processing image {processed_images}/{total_images} - {progress_percent:.2f}% complete",
                end="\r",
            )

            embeddings["Image_Path"].append(row["Full_Path"])
            embeddings["Category"].append(row["Category"])
            with Image.open(row["Full_Path"]) as ref:
                ref = ref.resize((350, 350))
                ref_array = np.array(ref)
                ref_array = ref_array / 255.0
                ref_tensor = tf.convert_to_tensor(ref_array, dtype=tf.float32)
                ref_tensor = tf.expand_dims(ref_tensor, axis=0)

                ref_feature_vector = model.predict(ref_tensor, verbose=0)

                for j, feature in enumerate(ref_feature_vector.reshape(-1)):
                    embeddings[f"Feature {j}"].append(feature)

        df_feature_vector = pd.DataFrame(embeddings)
        df_feature_vector.to_csv(
            embeddings_file_path, index=False
        )  # Save to the specified output directory

    return df_feature_vector


# Define the classify function
def image_classification(
    image_path: str,
    model: tf.keras.Model,
    categories: list,
    verbose: bool = False,
    return_original: bool = True,
) -> tuple:
    """
    Uses a trained machine learning model to classify an image loaded from disk.

    :param image_path: Path to the image to be classified.
    :param model: Pre-loaded classifier model to be used.
    :param verbose: Verbose output.
    :param return_original: Whether to return the original image or the processed image.
    :return: The original/processed image (PIL.image) and its classification (str).
    """

    # Load the image from the given path
    im_original = Image.open(image_path)

    # Resize the image to the target size
    im_processed = im_original.resize((350, 350))

    # Convert the PIL image to a NumPy array and normalize pixel values to [0, 1]
    im_array = np.array(im_processed) / 255.0

    # Convert the NumPy array to a TensorFlow tensor and add a batch dimension
    im_tensor = tf.convert_to_tensor(im_array, dtype=tf.float32)
    im_tensor = tf.expand_dims(im_tensor, axis=0)

    # Predict the class of the processed image
    pred = model.predict(im_tensor, verbose=1 if verbose else 0)

    # Get the index of the predicted class
    pred_class_idx = tf.argmax(pred, axis=1).numpy()[0]

    # Get the label of the predicted class
    # Ensure that CLASS_LABELS is defined elsewhere in your code
    pred_class_label = categories[pred_class_idx]

    # Return the original or processed image along with the predicted class label
    if return_original:
        return im_original, pred_class_label
    else:
        return im_processed, pred_class_label


def multi_task_contrastive_loss_wrapper(batch_size):
    def multi_task_contrastive_loss(
        y_true, y_pred, margin=1.0, category_weight=0.5, style_weight=0.5
    ):
        # Initialize embedding dimensions
        CATEGORY_EMBEDDING_DIM = 128
        STYLE_EMBEDDING_DIM = 256

        y_true = K.cast(y_true, dtype="float32")  # Cast labels to float32

        # Split the predictions into category and style distances (adjust dimensions as needed)
        y_pred_category = y_pred[:, :CATEGORY_EMBEDDING_DIM]
        y_pred_style = y_pred[:, CATEGORY_EMBEDDING_DIM:]

        # Reshape to ensure correct dimensions
        y_pred_category = K.reshape(y_pred_category, (-1, 1))
        y_pred_style = K.reshape(y_pred_style, (-1, 1))

        # Calculate contrastive loss for category
        category_loss = K.mean(
            y_true[:, 0] * K.square(y_pred_category)
            + (1 - y_true[:, 0]) * K.square(K.maximum(margin - y_pred_category, 0))
        )

        # Calculate contrastive loss for style
        style_loss = K.mean(
            y_true[:, 1] * K.square(y_pred_style)
            + (1 - y_true[:, 1]) * K.square(K.maximum(margin - y_pred_style, 0))
        )

        # Ensure style_loss is zero when there are no style embeddings
        style_loss = tf.where(
            tf.equal(tf.shape(y_pred_style)[1], 0), tf.constant(0.0), style_loss
        )  # Update here

        # Combine the losses (with optional weighting)
        total_loss = category_weight * category_loss + style_weight * style_loss
        return total_loss

    return multi_task_contrastive_loss


def create_image_pairs(df, num_pairs=100):
    pairs = []
    labels = []

    unique_labels = df["Label"].unique()
    print(unique_labels)
    # i = 0
    for label in tqdm(unique_labels, desc="Creating image pairs"):
        category, style = label.split(",")

        similar_group = df[(df["Category"] == category) & (df["Style"] == style)]
        dissimilar_group = df[
            ((df["Category"] != category) | (df["Style"] != style))
            & (df["Label"] != label)
        ]

        # print(similar_group.shape, dissimilar_group.shape)

        # chk = 0
        # Positive Pairs (same category AND same style)
        for i in range(min(len(similar_group) - 1, num_pairs)):
            for j in range(i + 1, min(len(similar_group), num_pairs)):
                pairs.append(
                    (
                        similar_group.iloc[i]["Full_Path"],
                        similar_group.iloc[j]["Full_Path"],
                    )
                )
                labels.append([1, 1])
            # chk += 1

        # Negative Pairs (different category OR different style)
        # chk2 = 0
        for _ in range(min(len(similar_group), num_pairs)):
            pair = (
                np.random.choice(similar_group["Full_Path"], 1)[0],
                np.random.choice(dissimilar_group["Full_Path"], 1)[0],
            )
            pairs.append(pair)
            if (
                pair[0].split(",")[0] == pair[1].split(",")[0]
            ):  # Same category, different style
                labels.append([1, 0])
            else:  # Different category, same or different style
                labels.append([0, 0])
            # chk2 += 1
        # print('before break')
        # i += 1
        if len(pairs) >= num_pairs:
            break
    # print(i)
    # print(chk)
    # print(chk2)
    return np.array(pairs), np.array(labels)


class SiameseDataGenerator(Sequence):
    def __init__(
        self,
        image_pairs,
        labels,
        batch_size,
        embeddings,
        embedding_index_mapping,
        augmentation_params=None,
    ):
        self.image_pairs = image_pairs
        self.labels = labels
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.embedding_index_mapping = embedding_index_mapping
        if augmentation_params is not None:
            self.datagen = ImageDataGenerator(**augmentation_params)
        else:
            self.datagen = ImageDataGenerator()

    def __len__(self):
        return int(np.ceil(len(self.image_pairs) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_pairs = self.image_pairs[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_labels = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_labels = np.reshape(batch_labels, (-1, 2))

        left_indices = []
        right_indices = []
        valid_pair_indices = []  # Keep track of valid pair indices

        for i, pair in enumerate(batch_pairs):
            try:
                left_indices.append(self.embedding_index_mapping[pair[0]])
                right_indices.append(self.embedding_index_mapping[pair[1]])
                valid_pair_indices.append(i)  # Mark this pair as valid
            except KeyError as e:
                print(f"Warning: Missing embedding for image: {e}")
                continue  # Skip this pair

        # Handle the case where all pairs are invalid (empty batch)
        if not valid_pair_indices:
            # You can either return an empty batch or raise an error to stop training
            return [np.array([]), np.array([])], np.array([])
            # Or: raise ValueError("All pairs in the batch have missing embeddings.")

        # Use valid pair indices to slice embeddings and labels
        left_embeddings = self.embeddings[left_indices]
        right_embeddings = self.embeddings[right_indices]
        batch_labels = batch_labels[
            valid_pair_indices
        ]  # Update batch labels based on valid pair indices

        # print("[BF] Left Embeddings Shape:", left_embeddings.shape)
        # print("[BF] Right Embeddings Shape:", right_embeddings.shape)

        # Reshape embeddings before returning
        left_embeddings = np.reshape(left_embeddings, (-1, self.embeddings.shape[1]))
        right_embeddings = np.reshape(right_embeddings, (-1, self.embeddings.shape[1]))

        # print("[AF] Left Embeddings Shape:", left_embeddings.shape)
        # print("[AF] Right Embeddings Shape:", right_embeddings.shape)
        return [left_embeddings, right_embeddings], np.array(batch_labels)


def get_image_features(image_path, base_model):
    """
    Loads an image, preprocesses it, and extracts features using the provided base model.

    Args:
        image_path (str): Path to the image file.
        base_model: The feature extraction model (e.g., Keras model, PyTorch model).

    Returns:
        np.ndarray: Extracted feature vector.
    """

    img = Image.open(image_path)

    # Preprocessing steps
    img = img.resize((350, 350))  # Resize to a standard size (adjust as needed)
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Feature extraction
    features = base_model.predict(img_array, verbose=0)

    # Handle different feature output formats
    if len(features.shape) > 2:  # Flatten if features have multiple dimensions
        features = features.reshape(features.shape[0], -1)

    return features


def load_model_and_data(model):
    model.save("saved_model", overwrite=True)
    model.save("embeddings_extract", overwrite=True)
    fe = tf.keras.models.load_model("embeddings_extract")
    return fe


def process_image(ref_path, base_model, categories):
    with Image.open(ref_path) as ref:
        ref_processed, ref_class = image_classification(
            f"{ref_path}", base_model, return_original=False, categories=categories
        )
    return ref_processed, ref_class


def get_feature_vector(ref_processed, fe):
    ref_processed = np.squeeze(ref_processed)
    ref_feature_vector = fe.predict(tf.expand_dims(ref_processed, axis=0), verbose=0)
    ref_feature_vector = ref_feature_vector.astype(float)
    ref_feature_vector = ref_feature_vector.reshape(1, -1)
    return ref_feature_vector


def get_recommendation(embeddings, ref_class, ref_feature_vector, model):
    recommendation = embeddings[embeddings["Category"] == ref_class]
    model.fit(recommendation.drop(["Image_Path", "Category"], axis="columns").values)
    ref_cluster = model.predict(ref_feature_vector)
    ref_cluster_indices = np.where(model.labels_ == ref_cluster)[0]
    recommendation = recommendation.iloc[ref_cluster_indices]
    return recommendation


def exclude_original_image(recommendation, ref_path):
    recommendation = recommendation[recommendation["Image_Path"] != ref_path]
    return recommendation


def rank_and_recommend(recommendation, ref_feature_vector, recommendation_images):
    cosine_similarities = cosine_similarity(
        ref_feature_vector,
        recommendation.drop(["Image_Path", "Category"], axis="columns"),
    )
    sorted_ref_cluster_indices = np.argsort(-cosine_similarities.flatten())
    top_ref_cluster_indices = sorted_ref_cluster_indices[:recommendation_images]
    recommendation = recommendation.iloc[top_ref_cluster_indices]
    return recommendation


def process_image_with_labels(ref_path, base_model, categories):
    with Image.open(ref_path) as ref:
        true_label = ref_path.split("/")[-3]
        _, predicted_label = image_classification(ref_path, base_model, categories)
        ref_features = get_image_features(ref_path, base_model)
        ref_features = normalize(ref_features)
    return ref, true_label, predicted_label, ref_features


def plot_image(ax, ref, true_label, predicted_label, i):
    # Ensure the image data is a numpy array with dtype float or uint8
    if isinstance(ref, np.ndarray):
        if ref.dtype != np.float and ref.dtype != np.uint8:
            ref = ref.astype(np.float)
    else:
        ref = np.array(ref, dtype=np.float)

    ax[i][0].imshow(ref)
    ax[i][0].set_title(
        f"Ground Truth: {true_label}\nPrediction: {predicted_label}", fontsize=12
    )
    is_correct = true_label == predicted_label
    color = "green" if is_correct else "red"
    text = "CORRECT PREDICTION" if is_correct else "INCORRECT PREDICTION"
    ax[i][0].text(
        0.5,
        -0.08,
        text,
        horizontalalignment="center",
        verticalalignment="center_baseline",
        transform=ax[i][0].transAxes,
        fontsize=12,
        color=color,
        weight="bold",
    )
    ax[i][0].axis("off")


def process_and_plot_retrieved_images(ax, results, base_model, i, ref_features):
    for j, rec_path in enumerate(results[i], start=1):
        with Image.open(rec_path) as rec:
            rec_features = get_image_features(rec_path, base_model)
            rec_features = normalize(rec_features)
            similarity = cosine_similarity(ref_features, rec_features)[0][0] * 100
            similarity_text = f"{similarity:.20f}%"
            ax[i][j].imshow(rec)
            ax[i][j].text(
                0.5,
                -0.08,
                similarity_text,
                horizontalalignment="center",
                verticalalignment="center_baseline",
                transform=ax[i][j].transAxes,
                fontsize=10,
            )
            ax[i][j].axis("off")
