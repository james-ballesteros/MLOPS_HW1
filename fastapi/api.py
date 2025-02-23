import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load the best registered model
MODEL_NAME = "best_air_quality_model"
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")

class PredictionInput(BaseModel):
    data: list

@app.post("/predict")
def predict(input_data: PredictionInput):
    df = pd.DataFrame(input_data.data)
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}