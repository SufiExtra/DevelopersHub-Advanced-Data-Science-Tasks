# ============================================================
# Task 3: Energy Consumption Time Series Forecasting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
import logging

logging.getLogger("cmdstanpy").disabled = True

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# ============================================================
# 1. Load Dataset
# ============================================================

df = pd.read_csv("household_power_consumption.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# ============================================================
# 2. Parse Date and Time
# ============================================================

df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True,
    errors="coerce"
)

df = df.dropna(subset=["Datetime"])
df = df.set_index("Datetime")

# ============================================================
# 3. Convert Target Column to Numeric
# ============================================================

df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)

df["Global_active_power"] = df["Global_active_power"].ffill()

# ============================================================
# 4. Resample Data Daily
# ============================================================

daily_data = df["Global_active_power"].resample("D").mean()
daily_data = daily_data.dropna()

# Use recent 365 days for faster ARIMA and Prophet processing
daily_data = daily_data[-365:]

print("\nDaily Resampled Data:")
print(daily_data.head())

print("\nDaily Data Shape:")
print(daily_data.shape)

# ============================================================
# 5. Exploratory Data Analysis
# ============================================================

plt.figure(figsize=(12, 5))
plt.plot(daily_data)
plt.title("Daily Household Energy Consumption")
plt.xlabel("Date")
plt.ylabel("Global Active Power")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(daily_data, bins=30, kde=True)
plt.title("Distribution of Daily Energy Consumption")
plt.xlabel("Global Active Power")
plt.show()

# ============================================================
# 6. Feature Engineering
# ============================================================

data = pd.DataFrame()
data["Global_active_power"] = daily_data

data["day"] = data.index.day
data["month"] = data.index.month
data["year"] = data.index.year
data["dayofweek"] = data.index.dayofweek
data["is_weekend"] = data["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)

data["lag_1"] = data["Global_active_power"].shift(1)
data["lag_7"] = data["Global_active_power"].shift(7)
data["rolling_mean_7"] = data["Global_active_power"].rolling(window=7).mean()

data = data.dropna()

print("\nFeature Engineered Data:")
print(data.head())

# ============================================================
# 7. Train-Test Split
# ============================================================

train_size = int(len(data) * 0.8)

train = data.iloc[:train_size]
test = data.iloc[train_size:]

print("\nTrain Size:", train.shape)
print("Test Size:", test.shape)

# ============================================================
# 8. Evaluation Function
# ============================================================

def evaluate_model(actual, predicted, model_name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print(f"\n{model_name} Performance")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))

    return mae, rmse

# ============================================================
# 9. ARIMA Model
# ============================================================

arima_model = ARIMA(
    train["Global_active_power"],
    order=(2, 1, 2)
)

arima_result = arima_model.fit()

arima_pred = arima_result.forecast(steps=len(test))

arima_mae, arima_rmse = evaluate_model(
    test["Global_active_power"],
    arima_pred,
    "ARIMA"
)

plt.figure(figsize=(12, 5))
plt.plot(test.index, test["Global_active_power"], label="Actual")
plt.plot(test.index, arima_pred, label="ARIMA Forecast")
plt.title("ARIMA: Actual vs Forecasted Energy Consumption")
plt.xlabel("Date")
plt.ylabel("Global Active Power")
plt.legend()
plt.show()

# ============================================================
# 10. Prophet Model
# ============================================================

prophet_df = daily_data.reset_index()
prophet_df.columns = ["ds", "y"]

prophet_train = prophet_df.iloc[:train_size]
prophet_test = prophet_df.iloc[train_size:]

prophet_model = Prophet()
prophet_model.fit(prophet_train)

future = prophet_model.make_future_dataframe(
    periods=len(prophet_test),
    freq="D"
)

forecast = prophet_model.predict(future)

prophet_pred = forecast["yhat"].iloc[-len(prophet_test):].values

prophet_mae, prophet_rmse = evaluate_model(
    prophet_test["y"],
    prophet_pred,
    "Prophet"
)

plt.figure(figsize=(12, 5))
plt.plot(prophet_test["ds"], prophet_test["y"], label="Actual")
plt.plot(prophet_test["ds"], prophet_pred, label="Prophet Forecast")
plt.title("Prophet: Actual vs Forecasted Energy Consumption")
plt.xlabel("Date")
plt.ylabel("Global Active Power")
plt.legend()
plt.show()

# ============================================================
# 11. XGBoost Model
# ============================================================

features = [
    "day",
    "month",
    "year",
    "dayofweek",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_mean_7"
]

X_train = train[features]
y_train = train["Global_active_power"]

X_test = test[features]
y_test = test["Global_active_power"]

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_mae, xgb_rmse = evaluate_model(
    y_test,
    xgb_pred,
    "XGBoost"
)

plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, xgb_pred, label="XGBoost Forecast")
plt.title("XGBoost: Actual vs Forecasted Energy Consumption")
plt.xlabel("Date")
plt.ylabel("Global Active Power")
plt.legend()
plt.show()

# ============================================================
# 12. Model Comparison
# ============================================================

results = pd.DataFrame({
    "Model": ["ARIMA", "XGBoost", "Prophet"],
    "MAE": [arima_mae, xgb_mae, prophet_mae],
    "RMSE": [arima_rmse, xgb_rmse, prophet_rmse]
})

print("\nModel Comparison:")
print(results)

plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="RMSE", data=results)
plt.title("Model Comparison Based on RMSE")
plt.ylabel("RMSE")
plt.show()

# ============================================================
# 13. Conclusion
# ============================================================

best_model = results.sort_values(by="RMSE").iloc[0]["Model"]

print("\n================ Conclusion ================")

print(f"The best performing model based on RMSE is: {best_model}")


# In this project, household energy consumption data was analyzed and used to forecast future energy usage through time series forecasting techniques. 
# After parsing and resampling the data, time-based features such as weekdays, weekends, lag values, and rolling averages were engineered to improve predictive performance.
# Three forecasting models—ARIMA, Prophet, and XGBoost—were evaluated using MAE and RMSE metrics. 
# Among them, XGBoost achieved the best performance with the lowest prediction error, demonstrating its ability to effectively capture temporal patterns in energy consumption data. 
# Overall, this project highlights the importance of feature engineering and model comparison in developing accurate forecasting solutions for real-world energy management applications.
