# Image Classification Project

This project demonstrates an end-to-end workflow for building, training,
and exporting an image classification model.\
The exported model is later used inside a **Streamlit application** to
perform real-time predictions.

## 📁 Project Structure

    .
    ├── iamge_classification.ipynb   # Notebook used for training the model
    ├── .gitignore
    ├── Image_classifier.keras       # Exported model files 
    ├── app.py                       # Streamlit application 
    ├── README.md
    ├── Image_6.JPG
    ├── Ears-corn.jpeg
    └── banana.jpeg

## 🚀 Features

-   Loads and preprocesses image datasets.
-   Trains an image classification model using a deep-learning
    framework.
-   Evaluates model performance.
-   Exports the trained model for later use.
-   Integrates with a Streamlit app to provide a simple UI for
    predictions.
```

```

## 🖥️ Streamlit App Flow

1.  User uploads an image.\
2.  Image is preprocessed to match model input shape.\
3.  Model generates predictions.\
4.  Prediction probabilities and labels are shown in the UI.

## 🚀 Run the Streamlit App

``` bash
streamlit run app.py
```

## 📦 Requirements

-   Python 3.8+
-   TensorFlow / PyTorch (depending on the model used)
-   Streamlit
-   NumPy
-   Pillow
-   Matplotlib (optional for plotting)
