⚠️ If GitHub does not preview the notebook, please open it using nbviewer.

# ⚡ Task 3: Energy Consumption Time Series Forecasting

## 📌 Project Overview

This project focuses on forecasting household energy consumption using historical power usage data. Time series forecasting techniques were applied to predict future energy usage patterns and compare the performance of multiple forecasting models.

The project demonstrates how temporal data can be transformed into meaningful insights through feature engineering, forecasting models, and performance evaluation.

---

## 🎯 Task Objective

The objective of this project is to forecast short-term household energy consumption using historical time-based patterns.

Key objectives include:

* Parse and preprocess time series data
* Resample energy consumption data into daily averages
* Engineer time-based features
* Train and compare multiple forecasting models
* Evaluate forecasting performance using MAE and RMSE
* Visualize actual versus predicted energy consumption

---

## 📂 Dataset

**Dataset:** Household Power Consumption Dataset

The dataset contains household electricity consumption measurements collected over time.

### Features

* Date
* Time
* Global Active Power
* Global Reactive Power
* Voltage
* Global Intensity
* Sub Metering 1
* Sub Metering 2
* Sub Metering 3

### Target Variable

* **Global Active Power**

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Statsmodels arima model
* Prophet
* XGBoost
* Scikit-learn

---

## 🔍 Project Approach

### 1. Data Preprocessing

* Loaded and explored the dataset
* Combined Date and Time into a single datetime column
* Converted power consumption values into numeric format
* Handled missing values
* Resampled data into daily average energy consumption

### 2. Exploratory Data Analysis

* Time series visualization
* Energy consumption distribution analysis
* Trend observation

### 3. Feature Engineering

Created temporal features including:

* Day
* Month
* Year
* Day of Week
* Weekend Indicator
* Lag Features
* Rolling Mean Features

### 4. Forecasting Models

#### ARIMA

Traditional statistical forecasting model for time series prediction.

#### Prophet

Facebook's forecasting framework designed for trend and seasonality analysis.

#### XGBoost

Machine learning-based forecasting model using engineered time-based features.

### 5. Model Evaluation

The models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

### 6. Visualization

* Actual vs Forecasted Energy Consumption
* Model Performance Comparison

---

## 📈 Results & Findings

### Model Performance

| Model   | MAE        | RMSE       |
| ------- | ---------- | ---------- |
| ARIMA   | 0.3224     | 0.4369     |
| Prophet | 0.8770     | 0.9587     |
| XGBoost | **0.2362** | **0.3265** |

### Key Findings

* XGBoost achieved the best forecasting performance with the lowest MAE and RMSE values.
* ARIMA produced reasonable forecasts but was less accurate than XGBoost.
* Prophet showed the highest prediction error on this dataset.
* Time-based feature engineering significantly improved forecasting accuracy.
* Historical consumption patterns proved highly valuable for short-term energy prediction.

---

## 📊 Visualizations Included

* Daily Energy Consumption Trend
* Energy Consumption Distribution
* ARIMA Forecast Plot
* Prophet Forecast Plot
* XGBoost Forecast Plot
* Model Comparison Bar Chart

---

## 🚀 Future Improvements

* Hyperparameter tuning for XGBoost
* Seasonal ARIMA (SARIMA) implementation
* LSTM and Deep Learning forecasting models
* Real-time energy forecasting dashboard using Streamlit

---

## 📝 Conclusion

This project successfully applied time series forecasting techniques to predict household energy consumption. After preprocessing the data and engineering temporal features, ARIMA, Prophet, and XGBoost models were trained and evaluated. Among all models, XGBoost achieved the highest forecasting accuracy with the lowest prediction error. The project highlights the importance of feature engineering and model comparison in building effective energy forecasting solutions.

---

## 👨‍💻 Author

**Muhammad Sufyan Shahid**

Data Science & Analytics Internship Project

DevelopersHub Corporation
