from fastapi import FastAPI,File,UploadFile
import keras
import json
import os 

from src.config import *
from src.classify import classify_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 2 means INFO and WARNING messages are not printed
model_path = "model/image_classifier.keras"
model = keras.saving.load_model(model_path)

app = FastAPI(title="Image Classification API")

with open("model/class_names.json") as f:
    data_category = json.load(f)


@app.get("/")
def check_health():
    return {"status":"Running!!!"}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    result = classify_image(data_category, file)
    return result
    