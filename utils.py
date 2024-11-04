import os
import cv2
import numpy as np
from config import rgb_codes

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def grayscale_to_rgb(mask):
    h, w = mask.shape[:2]
    mask = mask.astype(np.int32)
    output = [rgb_codes[pixel] for pixel in mask.flatten()]
    return np.reshape(output, (h, w, 3))

def save_results(image_x, mask, pred, save_image_path):
    mask = np.expand_dims(mask, axis=-1)
    mask = grayscale_to_rgb(mask)

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred)

    line = np.ones((image_x.shape[0], 10, 3)) * 255
    cat_images = np.concatenate([image_x, line, mask, line, pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)
