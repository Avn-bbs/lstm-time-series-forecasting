from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import os

app = FastAPI(title="LSTM Time Series Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_lstm_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class ForecastRequest(BaseModel):
    sequence: list[float]

@app.get("/")
def root():
    return {"status": "online", "model": "LSTM Time Series Forecaster"}

@app.post("/predict")
def predict(data: ForecastRequest):
    if len(data.sequence) != 24:
        raise HTTPException(status_code=400, detail="Expected a sequence of exactly 24 historical values.")

    arr = np.array(data.sequence).reshape(-1, 1)
    scaled = scaler.transform(arr).reshape(1, 24, 1)
    pred_scaled = model.predict(scaled)
    pred_actual = scaler.inverse_transform(pred_scaled)[0][0]

    return {"forecasted_value": float(pred_actual)}
