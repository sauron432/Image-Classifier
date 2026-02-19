import tensorflow as tf
import json

def get_category(dataset: tf.data.Dataset):
    data_category = dataset.class_names
    
    with open("model/class_names.json", "w") as f:
        json.dump(data_category, f)
        
    return data_category