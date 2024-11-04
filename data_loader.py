import os
from glob import glob
import cv2
import numpy as np
import tensorflow as tf
from config import image_h, image_w, num_classes

def load_dataset(path, limit=None):
    train_x = sorted(glob(os.path.join(path, "train", "images", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "train", "labels", "*.png")))

    valid_x = sorted(glob(os.path.join(path, "val", "images", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path, "val", "labels", "*.png")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.jpg")))
    test_y = sorted(glob(os.path.join(path, "test", "labels", "*.png")))

    if limit:
        train_x, train_y = train_x[:limit], train_y[:limit]
        valid_x, valid_y = valid_x[:limit], valid_y[:limit]
        test_x, test_y = test_x[:limit], test_y[:limit]

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image_mask(x, y):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (image_w, image_h)) / 255.0

    y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    y = cv2.resize(y, (image_w, image_h)).astype(np.int32)
    y = np.where((y == 2) | (y == 3) | (y == 4) | (y == 5), y, 0)

    return x, y

def preprocess(x, y):
    def f(x, y):
        x, y = read_image_mask(x, y)
        return x, y

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, num_classes)

    image.set_shape([image_h, image_w, 3])
    mask.set_shape([image_h, image_w, num_classes])

    return image, mask

def tf_dataset(X, Y, batch=4):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.shuffle(buffer_size=500).map(preprocess)
    ds = ds.batch(batch).prefetch(1)
    return ds
