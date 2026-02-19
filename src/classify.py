import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import io

from src.config import *

def classify_image(data_category, file):
    try:
        model_path = "model/image_classifier.keras"
        model = keras.saving.load_model(model_path)

        # Read uploaded file into PIL image
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((img_height, img_width))

        # Convert to array
        img_arr = tf.keras.utils.img_to_array(image)
        img_bat = tf.expand_dims(img_arr, 0)

        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict)

        return {
            "class": data_category[np.argmax(score)],
            "confidence": float(np.max(score) * 100)
        }

    except Exception as e:
        return {"error": str(e)}
