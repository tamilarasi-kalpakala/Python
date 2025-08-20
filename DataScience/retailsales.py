# Install prophet if not already
# pip install prophet

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load example retail sales data
# ----------------------------
# Example data: monthly sales for 5 years
# Columns: 'ds' (date), 'y' (sales)
data = {
    'ds': pd.date_range(start='2018-01-01', periods=60, freq='MS'),
    'y': [
        200, 220, 250, 270, 300, 320, 340, 360, 380, 400, 420, 450,
        210, 230, 260, 280, 310, 330, 350, 370, 390, 410, 430, 460,
        220, 240, 270, 290, 320, 340, 360, 380, 400, 420, 440, 470,
        230, 250, 280, 300, 330, 350, 370, 390, 410, 430, 450, 480,
        240, 260, 290, 310, 340, 360, 380, 400, 420, 440, 460, 490
    ]
}
df = pd.DataFrame(data)

# ----------------------------
# 2. Fit Prophet model
# ----------------------------
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.fit(df)

# ----------------------------
# 3. Make future dataframe
# ----------------------------
future = model.make_future_dataframe(periods=12, freq='MS')  # Forecast next 12 months

# ----------------------------
# 4. Predict
# ----------------------------
forecast = model.predict(future)

# ----------------------------
# 5. Plot forecast
# ----------------------------
fig1 = model.plot(forecast)
plt.title("Retail Sales Forecast with Prophet")
plt.show()

# Optional: plot components (trend, yearly seasonality)
fig2 = model.plot_components(forecast)
plt.show()