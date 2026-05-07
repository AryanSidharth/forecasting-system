import pandas as pd
import numpy as np
import warnings
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

warnings.filterwarnings("ignore")


# =========================
# LOAD DATA
# =========================

df = pd.read_excel("data/Forecasting Case- Study.xlsx")

print("Dataset Loaded Successfully")
print(df.head())


# =========================
# CLEAN COLUMN NAMES
# =========================

df.columns = df.columns.str.strip()

print("\nColumns:")
print(df.columns)


# =========================
# DATE CONVERSION
# =========================

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values('Date')


# =========================
# GROUP DATA
# =========================

df = df.groupby(['State', 'Date'])['Total'].sum().reset_index()


# =========================
# FEATURE ENGINEERING
# =========================

def create_features(data):

    data = data.copy()

    data['lag_1'] = data['Total'].shift(1)
    data['lag_7'] = data['Total'].shift(7)

    data['rolling_mean_7'] = data['Total'].rolling(7).mean()
    data['rolling_std_7'] = data['Total'].rolling(7).std()

    data['month'] = data['Date'].dt.month
    data['week'] = data['Date'].dt.isocalendar().week.astype(int)
    data['day_of_week'] = data['Date'].dt.dayofweek

    data = data.dropna()

    return data


# =========================
# EVALUATION FUNCTION
# =========================

def evaluate(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return mae, rmse, mape


# =========================
# TRAIN MODELS
# =========================

results = []

states = df['State'].unique()

print("\nStates Found:")
print(states)


for state in states:

    print(f"\nProcessing State: {state}")

    state_df = df[df['State'] == state].copy()

    state_df = state_df.sort_values('Date')

    # Weekly resampling
    state_df = (
        state_df
        .set_index('Date')
        .resample('W')
        .sum()
        .reset_index()
    )

    # Fill missing values
    state_df['Total'] = state_df['Total'].interpolate()

    # Create features
    state_df = create_features(state_df)

    # Train Validation Split
    train = state_df[:-8]
    valid = state_df[-8:]

    X_train = train.drop(columns=['Date', 'Total', 'State'])
    y_train = train['Total']

    X_valid = valid.drop(columns=['Date', 'Total', 'State'])
    y_valid = valid['Total']

    # =========================
    # XGBOOST
    # =========================

    print("Training XGBoost...")

    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5
    )

    xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_valid)

    xgb_mae, xgb_rmse, xgb_mape = evaluate(y_valid, xgb_pred)

    # =========================
    # SARIMA
    # =========================

    print("Training SARIMA...")

    sarima_model = SARIMAX(
        train['Total'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )

    sarima_fit = sarima_model.fit(disp=False)

    sarima_pred = sarima_fit.forecast(steps=8)

    sarima_mae, sarima_rmse, sarima_mape = evaluate(y_valid, sarima_pred)

    # =========================
    # PROPHET
    # =========================

    print("Training Prophet...")

    prophet_train = train[['Date', 'Total']].copy()

    prophet_train.columns = ['ds', 'y']

    prophet_model = Prophet()

    prophet_model.fit(prophet_train)

    future = prophet_model.make_future_dataframe(periods=8, freq='W')

    forecast = prophet_model.predict(future)

    prophet_pred = forecast.tail(8)['yhat'].values

    prophet_mae, prophet_rmse, prophet_mape = evaluate(
        y_valid,
        prophet_pred
    )

    # =========================
    # LSTM
    # =========================

    print("Training LSTM...")

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(state_df[['Total']])

    X_lstm = []
    y_lstm = []

    sequence_length = 8

    for i in range(sequence_length, len(scaled_data)):

        X_lstm.append(
            scaled_data[i-sequence_length:i, 0]
        )

        y_lstm.append(
            scaled_data[i, 0]
        )

    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)

    X_lstm = np.reshape(
        X_lstm,
        (X_lstm.shape[0], X_lstm.shape[1], 1)
    )

    split_index = len(X_lstm) - 8

    X_train_lstm = X_lstm[:split_index]
    y_train_lstm = y_lstm[:split_index]

    X_valid_lstm = X_lstm[split_index:]
    y_valid_lstm = y_lstm[split_index:]

    lstm_model = Sequential()

    lstm_model.add(
        LSTM(
            50,
            activation='relu',
            input_shape=(8, 1)
        )
    )

    lstm_model.add(Dense(1))

    lstm_model.compile(
        optimizer='adam',
        loss='mse'
    )

    lstm_model.fit(
        X_train_lstm,
        y_train_lstm,
        epochs=20,
        verbose=0
    )

    lstm_pred = lstm_model.predict(X_valid_lstm)

    lstm_pred = scaler.inverse_transform(lstm_pred)

    y_valid_actual = scaler.inverse_transform(
        y_valid_lstm.reshape(-1, 1)
    )

    lstm_mae, lstm_rmse, lstm_mape = evaluate(
        y_valid_actual,
        lstm_pred
    )

    # =========================
    # MODEL COMPARISON
    # =========================

    metrics = {
        'XGBoost': xgb_rmse,
        'SARIMA': sarima_rmse,
        'Prophet': prophet_rmse,
        'LSTM': lstm_rmse
    }

    best_model = min(metrics, key=metrics.get)

    print(f"Best Model for {state}: {best_model}")

    results.append({
        'State': state,
        'Best_Model': best_model,
        'XGB_RMSE': xgb_rmse,
        'SARIMA_RMSE': sarima_rmse,
        'Prophet_RMSE': prophet_rmse,
        'LSTM_RMSE': lstm_rmse
    })

    # Save best model
    if best_model == 'XGBoost':
        joblib.dump(
            xgb_model,
            f'models/{state}_xgb.pkl'
        )


# =========================
# SAVE RESULTS
# =========================

results_df = pd.DataFrame(results)

results_df.to_csv(
    'outputs/model_results.csv',
    index=False
)

print("\nTraining Completed Successfully")
print(results_df)