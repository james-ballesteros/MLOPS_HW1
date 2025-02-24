from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI(title="ML Model Inference API", version="1.0", description="API for making predictions using MLflow models")

# MLflow tracking details
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5050")
MODEL_NAME = os.getenv("MODEL_NAME", "best_air_quality_model") 

# Initialize MLflow Client
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# Load the latest production model
try:
    latest_model = next(
        iter(client.search_model_versions(f"name='{MODEL_NAME}' and current_stage='Production'")),
        None
    )
    
    if latest_model:
        model_uri = f"models:/{MODEL_NAME}/{latest_model.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Loaded Model: {MODEL_NAME}, Version: {latest_model.version}")
    else:
        print(f"⚠ No model found in Production stage for {MODEL_NAME}")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


@app.get("/")
async def root():
    return {"message": "Welcome to the ML Inference API. Use /predict to get model predictions."}


@app.post("/predict/")
async def predict(data: dict):
    """
    Make a prediction using the loaded MLflow model.

    Request:
    {
        "CO(GT)": 0.45,
        "PT08.S1(CO)": 0.94,
        "NMHC(GT)": 2.06,
        "C6H6(GT)": 0.22,
        "PT08.S2(NMHC)": 0.42,
        "NOx(GT)": -0.01,
        "PT08.S3(NOx)": 0.71,
        "NO2(GT)": 0.40,
        "PT08.S4(NO2)": 0.58,
        "PT08.S5(O3)": 0.59,
        "T": 0.07,
        "RH": 0.17,
        "AH": 0.17
    }

    Response:
    {
        "prediction": [value]
    }
    """
    if model is None:
        raise HTTPException(status_code=500, detail="No model available for predictions.")

    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        prediction = model.predict(df)

        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
