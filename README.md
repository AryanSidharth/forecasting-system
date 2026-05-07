# AI Sales Forecasting Platform

## Overview

This project is a production-style end-to-end time series forecasting system developed for forecasting next 8 weeks of sales for multiple US states using historical sales data.

The system:
- Trains multiple forecasting models
- Compares model performance using RMSE
- Automatically selects the best-performing model
- Exposes predictions through a FastAPI REST API
- Provides an interactive Streamlit dashboard

---

# Models Implemented

1. SARIMA
2. Facebook Prophet
3. XGBoost
4. LSTM

---

# Feature Engineering

Implemented:
- Lag Features (t-1, t-7, t-30)
- Rolling Mean
- Rolling Standard Deviation
- Month Feature
- Day of Week Feature
- Holiday Flag
- Time-Series Validation Split

---

# Tech Stack

## Backend
- FastAPI

## Frontend
- Streamlit

## ML Libraries
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Prophet
- Statsmodels

---

# Project Structure

forecasting-system/
│
├── api/
├── data/
├── models/
├── outputs/
├── dashboard.py
├── train.py
├── requirements.txt
└── README.md

---

# API Endpoint

## Run API

```bash
uvicorn api.main:app --reload
```

API Docs:
http://127.0.0.1:8000/docs

---

# Run Dashboard

```bash
streamlit run dashboard.py
```

---

# Model Evaluation

Models were evaluated using RMSE (Root Mean Squared Error).

The system automatically selects the best-performing model for each state.

Results are stored in:
outputs/model_results.csv

---

# Dashboard Features

- Interactive Forecast Visualization
- State-wise Forecasting
- KPI Metrics
- Forecast Charts
- API Integration

---

# Future Improvements

- Docker Deployment
- Cloud Hosting
- Real-time Forecasting
- CI/CD Integration
- Database Integration

---

# Author

Aryan Sidharth