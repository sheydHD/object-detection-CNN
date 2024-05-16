import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from tkinter import filedialog, Tk
from PIL import Image
import cv2
import hashlib

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set the base directory for your dataset
base_dir = "Data\\Train"  # Change this to the path of your "Data/Train" directory

model_path = "best_object_detector_model.keras"  # Path to save/load the model


# Function to rename and load data from the directories
def rename_and_load_data(base_dir, img_size=(224, 224)):
    images = []
    labels = []
    label_dict = {}
    label_counter = 0

    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        if (
            os.path.isdir(label_dir) and label.lower() != "train"
        ):  # Only process directories in Train
            if label not in label_dict:
                label_dict[label] = label_counter
                label_counter += 1
            for idx, file in enumerate(os.listdir(label_dir)):
                file_path = os.path.join(label_dir, file)
                if os.path.isfile(file_path) and file_path.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    # Ensure unique file names
                    base_name = f"{label}{idx+1}.png"
                    new_file_path = os.path.join(label_dir, base_name)
                    if os.path.exists(new_file_path):
                        # Append unique identifier if file already exists
                        unique_id = hashlib.md5(file.encode()).hexdigest()[:6]
                        new_file_name = f"{label}{idx+1}_{unique_id}.png"
                        new_file_path = os.path.join(label_dir, new_file_name)
                    os.rename(file_path, new_file_path)
                    img = Image.open(new_file_path).convert("RGB")
                    img = img.resize(img_size)
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(label_dict[label])

    images = np.array(images)
    labels = np.array(labels)

    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    return images, labels, label_dict


# Function to create and train the model with data augmentation
def train_model(base_dir, img_size=(224, 224)):
    images, labels, label_dict = rename_and_load_data(base_dir, img_size)

    # Shuffle the data and split into training and validation sets
    images, labels = np.array(images), np.array(labels)
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )

    num_classes = len(label_dict)
    if num_classes > 2:
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
        y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
        loss = "categorical_crossentropy"
        output_activation = "softmax"
        output_units = num_classes
    else:
        y_train = np.expand_dims(y_train, axis=-1)
        y_val = np.expand_dims(y_val, axis=-1)
        loss = "binary_crossentropy"
        output_activation = "sigmoid"
        output_units = 1

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        preprocessing_function=lambda x: x / 255.0,  # Normalize pixel values
    )

    train_gen = datagen.flow(X_train, y_train, batch_size=32)
    val_gen = datagen.flow(X_val, y_val, batch_size=32)

    base_model = MobileNetV2(
        input_shape=(*img_size, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Freeze the base model initially

    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(output_units, activation=output_activation),
        ]
    )

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, min_lr=0.0001
    )
    model_checkpoint = ModelCheckpoint(
        model_path, monitor="val_accuracy", save_best_only=True, mode="max"
    )

    history = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
    )

    # Fine-tune the model
    base_model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=loss,
        metrics=["accuracy"],
    )
    history_fine_tune = model.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
    )

    print(f"Best model saved to {model_path}")

    # Print accuracy
    training_accuracy = (
        history.history["accuracy"] + history_fine_tune.history["accuracy"]
    )
    validation_accuracy = (
        history.history["val_accuracy"] + history_fine_tune.history["val_accuracy"]
    )
    for i, (train_acc, val_acc) in enumerate(
        zip(training_accuracy, validation_accuracy)
    ):
        print(
            f"Epoch {i+1}: Training Accuracy = {train_acc*100:.2f}%, Validation Accuracy = {val_acc*100:.2f}%"
        )


# Function to load the trained model
def load_trained_model():
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        clear_terminal()
        print("Model not found. Please train the model first.")
        return None


# Function to detect objects in a photo using the trained model
def detect_objects_in_photo(
    photo_path, model, label_dict, img_size=(224, 224), confidence_threshold=0.1
):
    print(f"Detecting objects in photo: {photo_path}")
    img = Image.open(photo_path).convert("RGB")  # Ensure the image is in RGB format
    img_resized = img.resize(img_size)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    if len(predictions[0]) > 1:
        predicted_label = np.argmax(predictions)
        confidence = predictions[0][predicted_label]
    else:
        predicted_label = int(predictions[0] > 0.5)
        confidence = predictions[0][0]

    detected_label = None
    if confidence >= confidence_threshold:
        for label, index in label_dict.items():
            if index == predicted_label:
                detected_label = label
                break

    if detected_label:
        clear_terminal()
        print(f"Detected object: {detected_label} (Confidence: {confidence:.2f})")

        # Draw bounding box and label on the image
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w, _ = img.shape
        cv2.rectangle(img, (0, 0), (w, h), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{detected_label} ({confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Detected Object", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return detected_label
    else:
        print("Object not recognized.")
        return None


def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    img_size = (224, 224)  # Increase image size for better performance

    # Automatically detect directories and rename files
    _, _, label_dict = rename_and_load_data(base_dir, img_size)
    clear_terminal()
    print(f"Directories detected for training: {', '.join(label_dict.keys())}")

    train = input("Do you want to train the model? (yes/no): ").strip().lower()
    if train == "yes":
        train_model(base_dir, img_size)

    detect_photo = (
        input("Do you want to upload a photo for object detection? (yes/no): ")
        .strip()
        .lower()
    )
    if detect_photo == "yes":
        root = Tk()
        root.withdraw()  # Hide the main window
        root.attributes("-topmost", True)  # Bring the file dialog to the front
        photo_path = filedialog.askopenfilename(
            title="Select a Photo", filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        print(f"Selected photo: {photo_path}")  # Debug print
        if photo_path:
            model = load_trained_model()
            if model:
                _, _, label_dict = rename_and_load_data(base_dir, img_size)
                detect_objects_in_photo(
                    photo_path, model, label_dict, img_size, confidence_threshold=0.6
                )
        else:
            print("No photo selected.")


if __name__ == "__main__":
    main()
