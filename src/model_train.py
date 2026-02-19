import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

def train_model(data_train, data_val, data_category):
    """Train the model to learn the images of the classes
    Args:
        data_train (tf.data.Dataset): trained tensor data
        data_val (tf.data.Dataset): validation tensor data
        data_category (tf.data.Dataset): classes of the trained data
    """
    early_stop = EarlyStopping(
    monitor='val_accuracy',     # what to watch
    patience=5,             # epochs to wait
    restore_best_weights=True
    )
    rescale = Sequential([
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal_and_vertical")
        layers.RandomRotation(0.2)
    ])
    model = Sequential([
        rescale,
        layers.Conv2D(16,3,padding='same',activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32,3,padding='same',activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,padding='same',activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(128),
        layers.Dense(len(data_category))
    ])
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    epochs_size = 50
    model.fit(data_train,validation_data=data_val, epochs=epochs_size, callbacks=[early_stop])
    model.save('model/Image_classifier.keras')
    