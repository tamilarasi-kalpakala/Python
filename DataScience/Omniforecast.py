
#######################################################################
#  Use Statistical, ML and DL models for forecasting
#######################################################################
import pandas as pd

df = pd.read_csv("atm_withdrawals.trend.csv")

# Prophet requires columns "ds" and "y"
df = df.rename(columns={"date": "ds", "withdrawals": "y"})

# print(df.head())

# Forecast using Prophet (Statistical Model)

from prophet import Prophet

forecasts = {}
atm_ids = df["atm_id"].unique()

for atm_id in atm_ids:
    # Filter by atm_id, keep only Prophet-required columns
    atm_df = df[df["atm_id"] == atm_id][["ds", "y"]]

    model = Prophet()
    model.fit(atm_df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

# Keep only relevant columns
forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
print(forecast.tail())

#######################################################################
# Forecast using Random Forest (Machine Learning Model)
#######################################################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Load CSV
df = pd.read_csv("atm_withdrawals.trend.csv")
df['date'] = pd.to_datetime(df['date'])

# Feature engineering function
def add_features(data):
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['weekday'] = data['date'].dt.weekday
    data['salary_day_flag'] = data['day'].apply(lambda x: 1 if x == 1 or x == 15 else 0)
    return data

df = add_features(df)

# List of ATMs
atm_ids = df['atm_id'].unique()
all_predictions = {}

# Forecasting horizon for future
future_days = 30

for atm_id in atm_ids:
    atm_data = df[df['atm_id'] == atm_id].copy()
    
    # Skip ATMs with insufficient data
    if atm_data.shape[0] < 5:
        print(f"Skipping {atm_id}: not enough data ({atm_data.shape[0]} rows)")
        continue
    
    # Features and target
    feature_columns = ['day', 'month', 'weekday', 'salary_day_flag']
    X = atm_data[feature_columns]
    y = atm_data['withdrawals']
    
    # Train-test split for backtesting
    test_size = 0.2
    if atm_data.shape[0]*test_size < 1:
        test_size = 1 / atm_data.shape[0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 1️⃣ Predict past withdrawals (backtesting)
    predicted_past = rf_model.predict(X_test)
    past_predictions_df = pd.DataFrame({
        'date': atm_data['date'].iloc[-len(predicted_past):].values,
        'actual_withdrawals': y_test.values,
        'predicted_withdrawals': predicted_past,
        'type': 'past'
    })
    
    # 2️⃣ Predict next 30 days (future forecast)
    last_date = atm_data['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
    future_df = pd.DataFrame({'date': future_dates})
    future_df = add_features(future_df)
    
    X_future = future_df[feature_columns]
    predicted_future = rf_model.predict(X_future)
    future_predictions_df = pd.DataFrame({
        'date': future_df['date'],
        'actual_withdrawals': np.nan,  # future actuals unknown
        'predicted_withdrawals': predicted_future,
        'type': 'future'
    })
    
    # Combine past and future predictions
    all_predictions[atm_id] = pd.concat([past_predictions_df, future_predictions_df], ignore_index=True)
    print(f"Predictions completed for {atm_id}")

# Combine all ATMs into a single CSV
combined_df = pd.concat(all_predictions.values(), keys=all_predictions.keys(), names=['atm_id'])
combined_df.to_csv("atm_random_forest_predictions_with_future.csv")
print("All past and future predictions saved to atm_random_forest_predictions_with_future.csv")





####################################################################################
# Forecast using LSTM (Deep Learning)
####################################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler  # ← This fixes your error

# Load ATM withdrawal CSV
df = pd.read_csv("atm_withdrawals.trend.csv")
df['date'] = pd.to_datetime(df['date'])

# Minimum sequence length for LSTM
seq_length = 5
future_days = 30

# Dictionary to store predictions for all ATMs
all_predictions = {}

# Function to create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# Loop over each ATM
atm_ids = df['atm_id'].unique()
for atm_id in atm_ids:
    atm_data = df[df['atm_id'] == atm_id].copy()
    
    # Skip ATM if not enough data
    if atm_data.shape[0] < seq_length + 2:
        print(f"Skipping {atm_id}: not enough data ({atm_data.shape[0]} rows)")
        continue
    
    # Use only withdrawals column
    withdrawals = atm_data[['withdrawals']].values.astype(float)
    
    # Normalize
    scaler = MinMaxScaler()
    withdrawals_scaled = scaler.fit_transform(withdrawals)
    
    # Create sequences
    X_seq, y_seq = create_sequences(withdrawals_scaled, seq_length)
    
    # Train-test split for backtesting
    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # Build LSTM model
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mse')
    
    # Train LSTM
    model_lstm.fit(X_train, y_train, epochs=50, verbose=0)
    
    # 1️⃣ Predict past withdrawals (backtesting)
    y_pred_scaled = model_lstm.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    past_predictions_df = pd.DataFrame({
        'date': atm_data['date'].iloc[-len(y_pred):].values,
        'actual_withdrawals': scaler.inverse_transform(y_test.reshape(-1,1)).flatten(),
        'predicted_withdrawals': y_pred.flatten(),
        'type': 'past'
    })
    
    # 2️⃣ Predict next 30 days
    last_sequence = withdrawals_scaled[-seq_length:]  # last available sequence
    future_predictions = []
    
    for _ in range(future_days):
        pred_scaled = model_lstm.predict(last_sequence.reshape(1, seq_length, 1))
        future_predictions.append(pred_scaled[0,0])
        last_sequence = np.vstack([last_sequence[1:], pred_scaled])
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))
    future_dates = pd.date_range(start=atm_data['date'].max() + pd.Timedelta(days=1), periods=future_days)
    future_predictions_df = pd.DataFrame({
        'date': future_dates,
        'actual_withdrawals': np.nan,
        'predicted_withdrawals': future_predictions.flatten(),
        'type': 'future'
    })
    
    # Combine past and future predictions
    all_predictions[atm_id] = pd.concat([past_predictions_df, future_predictions_df], ignore_index=True)
    print(f"Predictions completed for {atm_id}")

# Combine all ATMs into a single CSV
combined_df = pd.concat(all_predictions.values(), keys=all_predictions.keys(), names=['atm_id'])
combined_df.to_csv("atm_lstm_predictions_with_future.csv")
print("All past and future predictions saved to atm_lstm_predictions_with_future.csv")



import matplotlib.pyplot as plt

# Plot withdrawals for ATM001
df[df["atm_id"]=="ATM001"].plot(x="date", y="withdrawals", kind="line", figsize=(10,5))
plt.title("ATM001 Withdrawals Over Time")
plt.ylabel("Withdrawals")
plt.show()