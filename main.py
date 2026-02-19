import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 2 means INFO and WARNING messages are not printed

from src.config import *
from src.data_load import data_load
from src.get_category import get_category
from src.model_train import train_model


def main():
    train,test,val = data_load()
    classes = get_category(train)
    train_model(train,val,classes)

if __name__ == "__main__":
    main()