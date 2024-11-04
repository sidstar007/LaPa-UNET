import tensorflow as tf
from data_loader import load_dataset, tf_dataset
from unet_model import build_unet
from config import model_dir, dataset_path, image_h, image_w, num_classes, batch_size
import os

model_path = os.path.join(model_dir, "sample_model_name.keras")
csv_path = os.path.join(model_dir, "sample_dataset_name.csv")

(train_x, train_y), (valid_x, valid_y), _ = load_dataset(dataset_path, limit=12000)
train_ds = tf_dataset(train_x, train_y, batch=batch_size)
valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size)

model = build_unet((image_h, image_w, 3), num_classes)
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(1e-4))

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
    tf.keras.callbacks.CSVLogger(csv_path),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
]

model.fit(train_ds, validation_data=valid_ds, epochs=5, callbacks=callbacks)
