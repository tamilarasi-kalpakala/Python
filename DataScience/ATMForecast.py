import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# -------------------------------
# Enhanced ATM Forecast + Salary/Festival Effects
# -------------------------------
# Assumptions:
# - `atm_withdrawals.csv` columns: date, atm_id, withdrawals
# - `festival_calendar.csv` optional: date, festival_name
# - If salary days aren't supplied per ATM, we assume salary is paid on the last weekday of each month.

# -------------------------------
# Utility functions
# -------------------------------

def last_weekday_of_month(year, month):
    # returns the last weekday (Mon-Fri) date of the month
    from calendar import monthrange
    last_day = monthrange(year, month)[1]
    d = pd.Timestamp(year=year, month=month, day=last_day)
    # if weekend, shift back to Friday
    if d.weekday() == 5:  # Saturday
        d -= pd.Timedelta(days=1)
    elif d.weekday() == 6:  # Sunday
        d -= pd.Timedelta(days=2)
    return d


def build_salary_days(start_date, end_date):
    rng = pd.date_range(start_date, end_date, freq='M')
    salary_days = []
    for ts in rng:
        d = last_weekday_of_month(ts.year, ts.month)
        salary_days.append(d)
    return pd.to_datetime(salary_days)

# -------------------------------
# 1. Load data
# -------------------------------
raw = pd.read_csv('atm_withdrawals.csv', parse_dates=['date'])
raw = raw.sort_values(['atm_id', 'date'])

# Optional festival csv
try:
    fest = pd.read_csv('festival_calendar.csv', parse_dates=['date'])
    festival_dates = fest['date'].dt.normalize().unique()
    festival_df = pd.DataFrame({'ds': pd.to_datetime(festival_dates), 'holiday': 'festival'})
except Exception:
    festival_df = pd.DataFrame(columns=['ds', 'holiday'])

# Build salary days for the whole data range
start_date = raw['date'].min()
end_date = raw['date'].max() + pd.Timedelta(days=60)
salary_days = build_salary_days(start_date, end_date)
salary_df = pd.DataFrame({'ds': salary_days, 'holiday': 'salary_day'})

# Combine holidays for Prophet
holidays = pd.concat([festival_df, salary_df], ignore_index=True)
holidays.drop_duplicates(subset=['ds', 'holiday'], inplace=True)

# -------------------------------
# 2. Create per-ATM forecasting dataset with regressors
# -------------------------------
# We'll create the regressors: is_salary_day, is_festival, day_of_week, is_weekend

all_atms = raw['atm_id'].unique()
forecasts = []

for atm in all_atms:
    atm_hist = raw[raw['atm_id'] == atm][['date', 'withdrawals']].rename(columns={'date': 'ds', 'withdrawals': 'y'})

    # Create full daily calendar so Prophet handles missing days
    full_idx = pd.date_range(atm_hist['ds'].min(), atm_hist['ds'].max() + pd.Timedelta(days=30), freq='D')
    df = pd.DataFrame({'ds': full_idx}).merge(atm_hist, on='ds', how='left')
    df['y'] = df['y'].fillna(0)  # optional: better imputations possible

    # regressors
    df['is_salary_day'] = df['ds'].isin(salary_days).astype(int)
    df['is_festival'] = df['ds'].isin(festival_df['ds']).astype(int)
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    # Fit Prophet with holidays + extra regressors
    m = Prophet(holidays=holidays, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_regressor('is_salary_day')
    m.add_regressor('is_festival')
    m.add_regressor('is_weekend')

    m.fit(df[['ds', 'y', 'is_salary_day', 'is_festival', 'is_weekend']])

    future = m.make_future_dataframe(periods=30)
    future['is_salary_day'] = future['ds'].isin(salary_days).astype(int)
    future['is_festival'] = future['ds'].isin(festival_df['ds']).astype(int)
    future['day_of_week'] = future['ds'].dt.dayofweek
    future['is_weekend'] = future['day_of_week'].isin([5,6]).astype(int)

    fcst = m.predict(future)
    fcst['atm_id'] = atm
    forecasts.append(fcst[['ds', 'atm_id', 'yhat', 'yhat_lower', 'yhat_upper', 'is_salary_day', 'is_festival']])

forecast_df = pd.concat(forecasts, ignore_index=True)

# -------------------------------
# 3. Feature engineering for clustering
# -------------------------------
# We'll cluster ATMs using: mean demand, weekday/weekend ratio, salary-day uplift
feat_list = []
for atm in all_atms:
    atm_fc = forecast_df[forecast_df['atm_id'] == atm]
    mean_demand = atm_fc['yhat'].mean()
    weekend_mean = atm_fc[atm_fc['ds'].dt.dayofweek.isin([5,6])]['yhat'].mean()
    weekday_mean = atm_fc[~atm_fc['ds'].dt.dayofweek.isin([5,6])]['yhat'].mean()
    weekend_ratio = (weekend_mean / (weekday_mean+1e-6))
    salary_uplift = atm_fc[atm_fc['is_salary_day']==1]['yhat'].mean() / (mean_demand+1e-6)

    feat_list.append({'atm_id': atm,
                      'mean_demand': mean_demand,
                      'weekend_ratio': weekend_ratio,
                      'salary_uplift': salary_uplift})

feat_df = pd.DataFrame(feat_list).fillna(0)

# scale and cluster
scaler = StandardScaler()
X = scaler.fit_transform(feat_df[['mean_demand', 'weekend_ratio', 'salary_uplift']])
km = KMeans(n_clusters=4, random_state=42)
feat_df['cluster'] = km.fit_predict(X)

# -------------------------------
# 4. Simple replenishment optimizer (greedy by shortage) by vehicle capacity
# -------------------------------
# Example constraints:
vehicle_capacity = 500000  # e.g., ₹500,000 cash per van
current_cash_level = 0.2  # assume ATMs currently at 20% of capacity
atm_capacity = 200000  # each ATM max cash capacity

# Build next-day shortage estimate per ATM (if predicted demand > remaining cash)
next_day = (forecast_df['ds'].min() + pd.Timedelta(days=1)).normalize()
next_day_preds = forecast_df[forecast_df['ds'] == next_day][['atm_id', 'yhat']].set_index('atm_id')
next_day_preds = next_day_preds.reindex(all_atms).fillna(0)

# remaining cash estimate = atm_capacity * current_cash_level
remaining_cash = atm_capacity * current_cash_level
shortage = (next_day_preds['yhat'] - remaining_cash).clip(lower=0)
shortage = shortage.rename('shortage')

# Greedy selection: pick ATMs with highest shortage until van capacity exhausted
shortage_df = shortage.reset_index()
shortage_df['shortage'] = shortage_df['shortage'].astype(float)
shortage_df = shortage_df.sort_values('shortage', ascending=False)

selected = []
cap_used = 0
for _, row in shortage_df.iterrows():
    need = row['shortage']
    if need <= 0:
        continue
    # if we can partially fulfill, still count as a visit
    if cap_used + min(need, atm_capacity) <= vehicle_capacity:
        selected.append({'atm_id': row['atm_id'], 'replenish_amount': min(need, atm_capacity)})
        cap_used += min(need, atm_capacity)
    else:
        # remaining capacity can be allocated partially
        remaining = vehicle_capacity - cap_used
        if remaining > 0:
            selected.append({'atm_id': row['atm_id'], 'replenish_amount': remaining})
            cap_used += remaining
        break

# selected now contains recommended replenishments for the next run

# -------------------------------
# 5. Outputs
# -------------------------------
# - forecast_df: predictions per ATM per day, including salary/festival indicators
# - feat_df: clustering + features (mean_demand, weekend_ratio, salary_uplift)
# - selected: list of ATMs and amounts to replenish under vehicle capacity constraint

# Save artifacts
forecast_df.to_csv('atm_forecasts_with_salary_fest.csv', index=False)
feat_df.to_csv('atm_clustering_features.csv', index=False)
import json
with open('replenishment_plan.json', 'w') as f:
    json.dump(selected, f, indent=2)

print('Saved forecasts, clustering features, and replenishment plan.')

# -------------------------------
# Notes & next steps
# -------------------------------
# - Replace the heuristic replenishment with an integer program if you need exact minimal visits.
# - Improve imputations for missing historical days (e.g., forward-fill or seasonal interpolation).
# - You can add ATM-specific capacities and current cash-level telemetry (if available) for more accuracy.
# - Consider a hierarchical model (e.g., global model + ATM-level adjustments) for better cold-start performance.
