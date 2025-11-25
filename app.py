import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

st.header('Image Classification Model')

# Load your trained model
model = load_model('Image_classifier.keras')

# List of class labels
data_cat = ['apple','banana','beetroot','bell pepper','cabbage','capsicum','carrot',
 'cauliflower','chilli pepper','corn','cucumber','eggplant','garlic','ginger',
 'grapes','jalepeno','kiwi','lemon','lettuce','mango','onion','orange','paprika',
 'pear','peas','pineapple','pomegranate','potato','raddish','soy beans','spinach',
 'sweetcorn','sweetpotato','tomato','turnip','watermelon']

# ---- IMAGE UPLOAD INPUT ----
uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Display image
    st.image(uploaded_image, width=250)

    # Load & preprocess
    image = Image.open(uploaded_image)
    image = image.resize((180, 180))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)   # Convert to batch

    # Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Output
    st.write("### Prediction Result")
    st.write(f"The image is of {data_cat[np.argmax(score)]}")
    st.write(f"**Confidence:** {np.max(score) * 100:.2f}%")
else:
    st.info("Please upload an image to get a prediction.")
