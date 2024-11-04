import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import grayscale_to_rgb
from config import model_dir, image_h, image_w, rgb_codes

model_path = os.path.join(model_dir, "model_12000_eyes.keras")
model = tf.keras.models.load_model(model_path)

def predict_and_display_sample_image(image_path, model, alpha=0.5):
    image = cv2.imread(image_path)
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, (image_w, image_h)) / 255.0
    prediction = model.predict(np.expand_dims(image_resized, axis=0))
    predicted_mask = np.argmax(prediction[0], axis=-1)

    colored_mask = grayscale_to_rgb(predicted_mask)
    colored_mask_resized = cv2.resize(colored_mask, (original_size[1], original_size[0]))

    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask_resized, alpha, 0)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Image with Segmentation Mask")

    plt.show()

predict_and_display_sample_image("sample_image.jpg", model)
