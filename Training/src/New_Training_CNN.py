import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
CHARACTER_DIM = (30, 30)  # Input image dimensions
BASE_DIR = Path(__file__).resolve().parent.parent
DIRECTORY = BASE_DIR / 'data/Data Set'  # Path to dataset
LIMIT = 2000  # Maximum number of images per class to avoid imbalance
TEST_SIZE = 0.15  # Percentage of data to use for testing
RANDOM_STATE = 42  # Random seed for reproducibility
EPOCHS = 120  # Maximum number of training epochs
BATCH_SIZE = 32  # Batch size for training
PATIENCE = 20  # Early stopping patience
LEARNING_RATE = 0.001  # Initial learning rate

def load_dataset_cnn():
    """
    Load and preprocess the dataset.
    Returns:
        X: Numpy array of images.
        y: Numpy array of labels.
    """
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
            image = image.astype('float32') / 255.0  # Normalize to [0, 1]

            images.append(image)
            labels.append(inner_dir)
            count += 1

    return np.array(images), np.array(labels)

def create_cnn_model(num_classes):
    """
    Create a CNN model for character recognition.
    Args:
        num_classes: Number of output classes.
    Returns:
        model: Compiled CNN model.
    """
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 1), padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Second Conv Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Third Conv Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Fourth Conv Block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),  # Replace Flatten() with GlobalAveragePooling

        # Dense Layers
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_cnn_model():
    """
    Train the CNN model and save the best version.
    Returns:
        model: Trained CNN model.
        label_encoder: Fitted LabelEncoder for class labels.
    """
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

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    # Create and train model
    model = create_cnn_model(len(label_encoder.classes_))

    # Track the best validation accuracy
    best_val_accuracy = 0.0
    best_model = None

    # Custom training loop to save the best model
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Check if the current model is the best
        val_accuracy = history.history['val_accuracy'][-1]
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model
            print(f"New best model found with validation accuracy: {best_val_accuracy:.4f}")

        # Early stopping
        if early_stopping.stopped_epoch > 0:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Save the best model
    if best_model is not None:
        best_model.save('best_cnn_model.keras')
        print("Best model saved as 'best_cnn_model.keras'.")

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

    # Save label encoder
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