import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import inspect


# Define the UNet model
def unet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    up1 = layers.UpSampling2D(size=(2, 2))(conv3)

    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(up1)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)
    up2 = layers.UpSampling2D(size=(2, 2))(conv4)

    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(up2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def preprocess_image(image_path, target_size=(224, 224)):

    print(f"preprocess_image at script line {inspect.currentframe().f_lineno}")
    print(image_path)
    # Read the image
    image = cv2.imread(image_path)

    # Resize the image
    image = cv2.resize(image, target_size)

    # Normalize pixel values
    image = image.astype(np.float32) / 255.0

    # Perform any other preprocessing steps as needed

    return image


def preprocess_label(label_path, target_size=(224, 224)):
    # Read the label image
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    # Resize the label
    label = cv2.resize(label, target_size, interpolation=cv2.INTER_NEAREST)

    # Encode labels if necessary

    return label


# Load and preprocess data
def load_data(data_dir):
    images = []
    labels = []

    # Create labels directory if it does not exist
    labels_dir = os.path.join(data_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    # Iterate through the images directory
    for filename in os.listdir(os.path.join(data_dir, 'images')):
        # Load and preprocess image
        image_path = os.path.join(data_dir, 'images', filename)
        image = preprocess_image(image_path)  # Preprocess as needed
        images.append(image)

        print(f"Processing {filename} at script line {inspect.currentframe().f_lineno}")
        print(filename)

        # Load and preprocess corresponding label
        label_filename = filename.replace('.jpeg', '_label.png')
        label_path = os.path.join(data_dir, 'labels', label_filename)
        label = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        print(
            label_path,
            label
        )

        cv2.imwrite(label_path, label)

        print(f"preprocess_image at script line {inspect.currentframe().f_lineno}")
        print(f"Label saved as {label_path}")

        if label is None:
            print(f"preprocess_image at script line {inspect.currentframe().f_lineno}")
            print(f"Failed to read label image: {label_path}")
            print(label)
            continue

        label = preprocess_label(label_path)  # Preprocess as needed
        labels.append(label)

    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0)

    return images, labels


# Train the model
def train_model(images, labels):
    labels = np.expand_dims(labels, axis=-1)
    input_shape = images[0].shape

    model = unet(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(images, labels, batch_size=4, epochs=10, validation_split=0.2)

    return model


# Perform prediction
def predict(model, image):
    prediction = model.predict(np.expand_dims(image, axis=0))
    return prediction[0]


# Example usage
def main():
    images, labels = load_data('training-images')
    model = train_model(images, labels)

    # Example prediction
    prediction = predict(model, images[0])

    # Visualize the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(images[0], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Predicted Mask')
    plt.imshow(prediction, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
