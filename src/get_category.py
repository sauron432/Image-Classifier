import tensorflow as tf
def get_category(dataset: tf.data.Dataset):
    data_category = dataset.class_names
    return data_category