import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json

model = tf.keras.models.load_model("models/model.h5")

with open("models/class_indices.json") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_idx = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    return idx_to_class[predicted_idx], confidence
