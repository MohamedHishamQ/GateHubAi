# train_cnn.py
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path


# Constants
CHARACTER_DIM = (30, 30)
BASE_DIR = Path(__file__).resolve().parent.parent
DIRECTORY = BASE_DIR / 'data\Data Set'
LIMIT = 2000
TEST_SIZE = 0.15
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 32


def load_dataset_cnn():
    images = []
    labels = []
    classes_dirs = os.listdir(DIRECTORY)

    for inner_dir in classes_dirs:
        count = 0
        class_path = os.path.join(DIRECTORY, inner_dir)

        for filename in os.listdir(class_path):
            if count >= LIMIT:
                break

            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, CHARACTER_DIM)
            image = image.astype('float32') / 255.0

            images.append(image)
            labels.append(inner_dir)
            count += 1

    return np.array(images), np.array(labels)


def create_cnn_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_cnn_model():
    # Load data
    print("Loading and preprocessing data...")
    X, y = load_dataset_cnn()

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Reshape images for CNN
    X = X.reshape(-1, 30, 30, 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Create data generator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    # Create and train model
    model = create_cnn_model(len(label_encoder.classes_))

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

    # Save model and label encoder
    model.save('cnn_model.h5')
    np.save('label_encoder.npy', label_encoder.classes_)

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model, label_encoder


if __name__ == "__main__":
    model, label_encoder = train_cnn_model()