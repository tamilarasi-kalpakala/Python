from prophet import Prophet
import pandas as pd

# data must have columns: ds (date), y (value) if using Prhophet directly
# Prophet does not support multiple time series in a single model, the only option is to loop through in this case
# Sample data creation for demonstration
df1 = pd.DataFrame({
    "ds": pd.date_range("2023-01-01", periods=90),
    "y": [100 + i*0.5 for i in range(90)]  # dummy trend of withdrawal amounts
})

# Load CSV into DataFrame
df = pd.read_csv("atm_withdrawals.trend.csv", parse_dates=["date"])
df = df.rename(columns={
    "date": "ds",          # Prophet wants "ds"
    "withdrawals": "y"     # Prophet wants "y"
})
forecasts = {}
atm_ids = df["atm_id"].unique()

for atm_id in atm_ids:
    # Filter by atm_id, keep only Prophet-required columns
    atm_df = df[df["atm_id"] == atm_id][["ds", "y"]]

    model = Prophet()
    model.fit(atm_df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    forecast = forecast.rename(columns={
        "yhat": f"{atm_id}_predicted",
        "yhat_lower": f"{atm_id}_lower",
        "yhat_upper": f"{atm_id}_upper"
    })

    forecasts[atm_id] = forecast[["ds", f"{atm_id}_predicted", f"{atm_id}_lower", f"{atm_id}_upper"]]

# Print the forecasts for each ATM
    for atm_id, forecast in forecasts.items():
        print(f"\n🔮 Forecast for {atm_id}:")
        print(forecast.head(10))   # print first 10 rows


# Print All Forecasts Combined into One DataFrame
combined_forecast = forecasts[list(forecasts.keys())[0]][["ds"]]  # start with dates
for atm_id, forecast in forecasts.items():
    combined_forecast = combined_forecast.merge(forecast, on="ds")

print(combined_forecast.head(10))

# Print defalut columsn if prohpet forecast output column names are not changed 
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

