import os
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Sequential, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import json
import re


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

    if not os.path.exists(csv_logger_path):
        print("Warning: No training log file found. Using initial model.")
        return initial_model, 0, base_model_file_path, csv_logger_path

    log = pd.read_csv(csv_logger_path)
    epoch_val_accuracy = log[["epoch", "val_accuracy"]].values

    # Sort by validation accuracy and then by epoch number
    epoch_val_accuracy = epoch_val_accuracy[(-epoch_val_accuracy[:, 1]).argsort()]

    for epoch, val_accuracy in epoch_val_accuracy:
        # Convert epoch to integer
        epoch = int(epoch)

        # Check for both possible naming conventions
        model_file_1 = f"epoch_{epoch}.h5"
        model_file_2 = f"epoch_{epoch}_va_{val_accuracy:.4f}.h5"

        if model_file_1 in model_files:
            model_path = os.path.join(save_dir, model_file_1)
            model = load_model(model_path)
            print(f"Loaded model from {model_path}")
            return model, epoch, base_model_file_path, csv_logger_path
        elif model_file_2 in model_files:
            model_path = os.path.join(save_dir, model_file_2)
            model = load_model(model_path)
            print(f"Loaded model from {model_path}")
            return model, epoch, base_model_file_path, csv_logger_path

    print("Warning: No model file found. Using initial model.")
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
            (f for f in filenames if re.search(r"epoch_(\d+)(_va(\d+\.\d+))?.h5", f)),
            key=lambda f: int(re.search(r"epoch_(\d+)(_va(\d+\.\d+))?.h5", f).group(1)),
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

        # Extract the epoch number and validation accuracy from the file name
        match = re.search(r"epoch_(\d+)(_va(\d+\.\d+))?.h5", model_file)
        start_epoch = int(match.group(1))
        val_accuracy = (
            float(match.group(3))
            if match.group(3)
            else get_best_val_accuracy(os.path.join(dirpath, "training_log.csv"))
        )
        print(f"Starting epoch: {start_epoch}")
        print(f"Validation accuracy: {val_accuracy}")

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
    checkpoint = CustomModelCheckpoint(
        filepath=f"{base_model_file_path}{{epoch}}_va_{{val_accuracy:.4f}}.h5",
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
