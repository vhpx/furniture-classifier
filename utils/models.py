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


def load_model_from_checkpoint(
    model_conf,
    initial_model,
    environment,
    root_dir,
):
    save_dir = f"{root_dir}/{model_conf['save_dir']}"

    base_model_file_path = f"{save_dir}/epoch_"
    csv_logger_path = f"{save_dir}/training_log.csv"

    # If the model directory does not exist, create it
    if environment == "LOCAL" and not os.path.exists(root_dir):
        os.makedirs(root_dir)

    current_save_dir = f"{save_dir}" if environment == "LOCAL" else f"{save_dir}"

    # If the model directory does not exist, create it
    if not os.path.exists(current_save_dir):
        os.makedirs(current_save_dir)

    # When resuming training, list all files in the model directory
    model_files = os.listdir(current_save_dir)

    # Find the file with the highest epoch number in its name
    latest_model_file = max(
        (f for f in model_files if re.search(r"(\d+).h5", f)),
        key=lambda f: int(re.search(r"(\d+).h5", f).group(1)),
        default=None,
    )

    # Load that model if it exists
    if latest_model_file is not None:
        model = load_model(f"{current_save_dir}/{latest_model_file}")

        # Extract the epoch number from the file name and use it as the starting epoch
        start_epoch = int(re.search(r"epoch_(\d+).h5", latest_model_file).group(1))
    else:
        model = initial_model
        start_epoch = 0

    return model, start_epoch, base_model_file_path, csv_logger_path


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
    # If the model is None, print an error message and return
    if model is None:
        print("No model found")
        return

    # If the model configuration is None, print an error message and return
    if model_conf is None:
        print("No model configuration found")
        return

    # Get cache model
    cached_model, start_epoch, base_model_file_path, csv_logger_path = (
        load_model_from_checkpoint(
            model_conf,
            model,
            environment,
            root_dir,
        )
    )

    # Update the model
    model = cached_model

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

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

    # add class weights for imbalanced dataset
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_generator.classes),
        y=train_generator.classes,
    )

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, csv_logger, reduce_lr],
        class_weight=dict(enumerate(class_weights)),
        initial_epoch=start_epoch,
    )

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_generator)
