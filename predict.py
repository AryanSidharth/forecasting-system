from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Forecasting API")


@app.get("/")
def home():
    return {
        "message": "Forecasting API Running Successfully"
    }


@app.get("/predict")
def predict(state: str):

    try:

        model = joblib.load(f"models/{state}_xgb.pkl")

        predictions = []

        lag_1 = 100
        lag_7 = 95

        rolling_mean_7 = 102
        rolling_std_7 = 5

        for week in range(8):

            sample_input = pd.DataFrame({
                'lag_1': [lag_1],
                'lag_7': [lag_7],
                'rolling_mean_7': [rolling_mean_7],
                'rolling_std_7': [rolling_std_7],
                'month': [6],
                'week': [24 + week],
                'day_of_week': [2]
            })

            pred = model.predict(sample_input)[0]

            predictions.append(float(pred))

            lag_1 = pred

        return {
            "state": state,
            "8_week_forecast": predictions
        }

    except Exception as e:

        return {
            "error": str(e)
        }