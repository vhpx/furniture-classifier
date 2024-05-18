import os
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Sequential, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json
import re


def load_best_model_from_checkpoint(
    model_conf,
    initial_model,
    environment,
    root_dir,
):
    save_dir = f"{root_dir}/{model_conf['save_dir']}"

    base_model_file_path = f"{save_dir}/epoch_"
    csv_logger_path = f"{save_dir}/training_log.csv"

    print(f"Checking if model directory exists at {root_dir}...")
    if environment == "LOCAL" and not os.path.exists(root_dir):
        print(f"Creating directory {root_dir}...")
        os.makedirs(root_dir)

    current_save_dir = f"{save_dir}" if environment == "LOCAL" else f"{save_dir}"

    print(f"Checking if current save directory exists at {current_save_dir}...")
    if not os.path.exists(current_save_dir):
        print(f"Creating directory {current_save_dir}...")
        os.makedirs(current_save_dir)

    print(f"Listing all files in the model directory {current_save_dir}...")
    model_files = os.listdir(current_save_dir)
    print(f"Found {len(model_files)} model files in {current_save_dir}.")

    print("Reading training log and creating a list of tuples (epoch, val_accuracy)...")
    with open(csv_logger_path, "r") as f:
        reader = csv.reader(f)
        log = list(reader)
    epoch_val_accuracy = [
        (int(row[0]), float(row[log[0].index("val_accuracy")])) for row in log[1:]
    ]

    print(
        "Sorting the list in descending order based on validation accuracy, and then epoch number..."
    )
    epoch_val_accuracy.sort(key=lambda x: (-x[1], -x[0]))

    print("Iterating over the sorted list to find the best model...")
    for epoch, val_accuracy in epoch_val_accuracy:
        model_file = f"epoch_{epoch}.h5"
        if model_file in model_files:
            print(f"Loading model from {current_save_dir}/{model_file}...")
            model = load_model(f"{current_save_dir}/{model_file}")
            print(f"Starting epoch: {epoch}")
            return model, epoch, base_model_file_path, csv_logger_path

    print("No model file found. Using initial model.")
    return initial_model, 0, base_model_file_path, csv_logger_path


def get_best_model(root_dir, search_term=None):
    best_model = None
    best_val_accuracy = -np.inf
    best_model_dir = None

    print(f"Searching for best model in {root_dir}...")

    # Walk through all directories under the root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # If a search term is provided, skip directories that do not include the search term
        if search_term is not None and search_term not in dirpath:
            continue

        print(f"Checking directory {dirpath}...")

        # Sort the model files by their epoch numbers in descending order
        model_files = sorted(
            (f for f in os.listdir(dirpath) if re.search(r"epoch_(\d+).h5", f)),
            key=lambda f: int(re.search(r"epoch_(\d+).h5", f).group(1)),
            reverse=True,
        )

        print(f"Found {len(model_files)} model files.")

        # Try to load each model in turn until one is found that exists
        for model_file in model_files:
            model_path = os.path.join(dirpath, model_file)
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}...")
                model = load_model(model_path)
                break
        else:
            # If no model file exists in the current directory, skip it
            print(f"No valid model files in {dirpath}. Skipping...")
            continue

        # Extract the epoch number from the file name and use it as the starting epoch
        start_epoch = int(re.search(r"epoch_(\d+).h5", model_file).group(1))
        print(f"Starting epoch: {start_epoch}")

        csv_logger_path = os.path.join(dirpath, "training_log.csv")

        # Get the best validation accuracy from the current directory
        val_accuracy = get_best_val_accuracy(csv_logger_path)
        print(f"Best validation accuracy in this directory: {val_accuracy}")

        # If this model is better than the current best model, update the best model and its validation accuracy
        if val_accuracy > best_val_accuracy:
            print(f"New best model found with validation accuracy: {val_accuracy}")
            best_model = model
            best_val_accuracy = val_accuracy
            best_model_dir = dirpath

    print(
        f"Best model found in {best_model_dir} with validation accuracy: {best_val_accuracy}"
    )
    return best_model, best_val_accuracy, best_model_dir


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


def train_model(
    model=None,
    model_conf=None,
    environment="LOCAL",
    start_epoch=0,
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
    checkpoint = ModelCheckpoint(
        filepath=f"{base_model_file_path}{{epoch}}.h5",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        save_freq="epoch",
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
