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
    # Read the image
    image = cv2.imread(image_path)
    # image = cv2.resize(image, target_size)

    # plt.imshow(image)
    # plt.title('0 - Original Image')
    # plt.show()

    # 1) Convert RGB to HLS
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # plt.imshow(hls_image)
    # plt.title('1 - HLS Image')
    # plt.show()

    # 2) Increase colours saturation by 50%
    hls_image[:, :, 2] = np.clip(hls_image[:, :, 2] * 1.5, 0, 255)

    # plt.imshow(hls_image)
    # plt.title('2 - Increased Saturation')
    # plt.show()

    # 3) Make colors inversion
    inverted_img = cv2.bitwise_not(hls_image)

    # plt.imshow(inverted_img)
    # plt.title('3 - Inverted Image')
    # plt.show()

    # 4) Convert images into gray scale
    gray_img = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY)

    # plt.imshow(gray_img, cmap='gray')
    # plt.title('4 - Grayscale Image')
    # plt.show()

    # 5) Increase contrast (using equalizeHist for demonstration)
    contrast_img = cv2.equalizeHist(gray_img)

    # plt.imshow(contrast_img, cmap='gray')
    # plt.title('5 - Contrast Enhanced Image')
    # plt.show()

    # 6) Decrease noise level using median filter
    median_img = cv2.medianBlur(contrast_img, 5)

    # plt.imshow(median_img, cmap='gray')
    # plt.title('6 - Median Filtered Image')
    # plt.show()

    # # 7) Image normalization (normalized between 0 and 1)
    # normalized_img = cv2.normalize(median_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #
    # plt.imshow(normalized_img, cmap='gray')
    # plt.title('7 - Normalized Image')
    # plt.show()

    # 7) Threshold segmentation
    _, thresholded_img = cv2.threshold(median_img, 0.2 * 255, 255, cv2.THRESH_BINARY)

    # plt.imshow(thresholded_img, cmap='gray')
    # plt.title('7 -Thresholded Image')
    # plt.show()

    # 8)  Invert colors
    inverted_thresholded_img = cv2.bitwise_not(thresholded_img)

    plt.imshow(inverted_thresholded_img, cmap='gray')
    plt.title('8 - Inverted Thresholded Image')
    plt.show()

    return inverted_thresholded_img


def preprocess_label(label_path, target_size=(224, 224)):
    # Read the label image
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    # Resize the label
    # label = cv2.resize(label, target_size, interpolation=cv2.INTER_NEAREST)

    # Encode labels if necessary

    return label


# Load and preprocess data
def load_data(data_dir):
    images = []
    labels = []

    # Create labels directory if it does not exist
    labels_dir = os.path.join(data_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    # Create processed images directory
    processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Iterate through the images directory
    for filename in os.listdir(os.path.join(data_dir, 'images')):
        # Load and preprocess image
        image_path = os.path.join(data_dir, 'images', filename)
        image = preprocess_image(image_path)
        images.append(image)

        processed_path = os.path.join(data_dir, 'processed', filename)
        cv2.imwrite(processed_path, image)

        # Load and preprocess corresponding label
        label_filename = filename.replace('.jpeg', '_label.png')
        label = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        label_path = os.path.join(data_dir, 'labels', label_filename)
        cv2.imwrite(label_path, label)

        if label is None:
            print(f"preprocess_image at script line {inspect.currentframe().f_lineno}")
            print(f"Failed to read label image: {label_path}")
            print(label)
            continue

        label = preprocess_label(label_path)  # Preprocess as needed
        labels.append(label)

    images = np.stack(images, axis=0)
    images = np.expand_dims(images, axis=-1)  # Add an extra dimension for the channel
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
