from src.config import *

import tensorflow as tf

def data_load():
    try:
        #Using TensorFlow's `image_dataset_from_directory` function to create a dataset of images for training a machine learning model. 
        data_train  = tf.keras.utils.image_dataset_from_directory(
            data_train_path,
            shuffle = True,
            image_size = (img_width,img_height),
            batch_size = 32,
            validation_split = False 
        )
        data_test  =tf.keras.utils.image_dataset_from_directory(
        data_test_path,
        shuffle = False,
        image_size = (img_width,img_height),
        batch_size = 32,
        validation_split = False
        )
        data_val = tf.keras.utils.image_dataset_from_directory(
        data_validation_path,
        shuffle = False,
        image_size = (img_width,img_height),
        batch_size = 32,
        validation_split = False 
        )
        return data_train,data_test,data_val
    except Exception as e:
        print("Error loading data! ", e)
        