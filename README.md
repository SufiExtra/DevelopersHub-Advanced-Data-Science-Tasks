# 🚀 DevelopersHub Advanced Data Science Internship Tasks

## 📌 Repository Overview

This repository contains 5 advanced Data Science and Analytics internship tasks completed as part of the **DevelopersHub Corporation Data Science & Analytics Internship**.

The projects cover real-world data science problems including classification, customer segmentation, time series forecasting, business cost optimization, explainable AI, and interactive dashboard development.

Each task is organized in a separate folder with its own code, notebook, visualizations, and documentation.

---

## 🧠 Skills Covered

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Classification Modeling
* Customer Segmentation
* Time Series Forecasting
* Business Cost Optimization
* Explainable AI (SHAP)
* Feature Importance Analysis
* Streamlit Dashboard Development
* Data Visualization
* Business Intelligence
* Data Storytelling

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* CatBoost
* Prophet
* ARIMA
* SHAP
* Streamlit
* Plotly

---

# 📂 Completed Tasks

---

## ✅ [Task 1: Term Deposit Subscription Prediction] (https://github.com/SufiExtra/DevelopersHub-Advanced-Data-Science-Tasks/tree/main/Task%201)

### 📌 Objective

Predict whether a bank customer will subscribe to a term deposit based on customer demographics, financial information, and marketing campaign data.

### 📊 Dataset

**Bank Marketing Dataset**

### 🔍 Approach

* Loaded and explored the dataset
* Performed data cleaning and preprocessing
* Encoded categorical features
* Trained Logistic Regression and Random Forest models
* Evaluated models using Confusion Matrix, F1 Score, and ROC Curve
* Used SHAP to explain model predictions

### 📈 Results

| Model               | Accuracy | F1 Score | ROC-AUC |
| ------------------- | -------: | -------: | ------: |
| Logistic Regression |   89.14% |    0.327 |   0.873 |
| Random Forest       |   90.26% |    0.452 |   0.926 |

### 💡 Key Finding

Random Forest performed better than Logistic Regression and achieved the highest ROC-AUC score. SHAP analysis helped explain important customer factors affecting subscription decisions.

---

## ✅ [Task 2: Customer Segmentation Using Unsupervised Learning] (https://github.com/SufiExtra/DevelopersHub-Advanced-Data-Science-Tasks/tree/main/Task%202)

### 📌 Objective

Cluster customers based on spending habits and propose marketing strategies for each customer segment.

### 📊 Dataset

**Mall Customers Dataset**

### 🔍 Approach

* Conducted Exploratory Data Analysis
* Encoded categorical features
* Applied K-Means Clustering
* Used the Elbow Method and Silhouette Score
* Visualized clusters using PCA
* Created marketing strategies for each customer segment

### 📈 Results

Five customer segments were identified:

* Mature Customers
* High Income, Low Spending Customers
* High Income, High Spending Customers
* Potential Premium Customers
* Young Active Customers

### 💡 Key Finding

Customer segmentation helps businesses identify valuable customer groups and design targeted marketing strategies based on income and spending behavior.

---

## ✅ [Task 3: Energy Consumption Time Series Forecasting] (https://github.com/SufiExtra/DevelopersHub-Advanced-Data-Science-Tasks/tree/main/Task%203)

### 📌 Objective

Forecast short-term household energy consumption using historical time-based patterns.

### 📊 Dataset

**Household Power Consumption Dataset**

### 🔍 Approach

* Parsed Date and Time columns
* Resampled energy data into daily averages
* Engineered time-based features
* Created lag and rolling mean features
* Trained ARIMA, Prophet, and XGBoost models
* Evaluated performance using MAE and RMSE
* Plotted actual vs forecasted energy usage

### 📈 Results

| Model   |    MAE |   RMSE |
| ------- | -----: | -----: |
| ARIMA   | 0.3224 | 0.4369 |
| Prophet | 0.8770 | 0.9587 |
| XGBoost | 0.2362 | 0.3265 |

### 💡 Key Finding

XGBoost achieved the best performance because it effectively used engineered time-based features such as lag values, rolling averages, and weekday/weekend patterns.

---

## ✅ [Task 4: Loan Default Risk with Business Cost Optimization] (https://github.com/SufiExtra/DevelopersHub-Advanced-Data-Science-Tasks/tree/main/Task%204)

### 📌 Objective

Predict the likelihood of loan default and optimize the decision threshold based on cost-benefit analysis.

### 📊 Dataset

**Home Credit Default Risk Dataset**

### 🔍 Approach

* Cleaned and preprocessed the dataset
* Handled missing numerical and categorical values
* Trained Logistic Regression as a baseline model
* Trained CatBoost as an advanced classification model
* Evaluated models using Accuracy, Recall, F1 Score, and ROC-AUC
* Defined business costs for false positives and false negatives
* Optimized the decision threshold to minimize total business cost
* Analyzed feature importance

### 📈 Results

| Model               | Accuracy | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | -----: | -------: | ------: |
| Logistic Regression |   68.29% | 66.70% |    0.253 |   0.742 |
| CatBoost            |   72.63% | 64.47% |    0.275 |   0.751 |
| Optimized CatBoost  |   74.74% | 61.86% |    0.283 |   0.751 |

### 💼 Business Cost Optimization

| Metric             | Value |
| ------------------ | ----: |
| Best Threshold     |  0.52 |
| False Positives    |  2219 |
| False Negatives    |   307 |
| Minimum Total Cost |  5289 |

### 💡 Key Finding

CatBoost performed better than Logistic Regression. Threshold optimization helped reduce business risk by assigning a higher cost to false negatives, since approving a risky customer can create greater financial loss.

---

## ✅ [Task 5: Interactive Business Dashboard in Streamlit] (https://github.com/SufiExtra/DevelopersHub-Advanced-Data-Science-Tasks/tree/main/Task%205)

### 📌 Objective

Develop an interactive dashboard for analyzing sales, profit, customer performance, and segment-wise business performance.

### 📊 Dataset

**Global Superstore Dataset**

### 🔍 Approach

* Cleaned and prepared the dataset
* Built an interactive Streamlit dashboard
* Added filters for Region, Category, and Sub-Category
* Displayed KPI cards for Total Sales, Profit, Orders, and Customers
* Created interactive charts using Plotly
* Designed a clean Power BI-style dashboard layout

### 📊 Dashboard Features

* Total Sales KPI
* Total Profit KPI
* Total Orders KPI
* Total Customers KPI
* Sales by Category
* Profit by Category
* Profit by Region
* Sales by Segment
* Top 5 Customers by Sales
* Sales and Profit Trend Over Time
* Sales vs Profit Analysis

### 💡 Key Finding

The dashboard transforms raw sales data into interactive business insights, helping users analyze performance by region, category, customer segment, and time period.

---

# 🚀 How to Run the Projects

## For Python Scripts

```bash
python "file_name.py"
```

## For Jupyter Notebooks

Open the notebook file using:

```bash
jupyter notebook
```

or use VS Code / JupyterLab.

## For Streamlit Dashboard

```bash
streamlit run "Interactive Business Dashboard in Streamlit.py"
```

---

# 📌 Final Summary

This advanced internship task set helped strengthen my skills in practical data science, machine learning, forecasting, explainable AI, business intelligence, and dashboard development.

The completed projects demonstrate the full data science workflow, including:

* Understanding business problems
* Preparing and analyzing datasets
* Building machine learning models
* Evaluating model performance
* Explaining predictions
* Optimizing business decisions
* Creating interactive dashboards

Overall, these tasks provided valuable hands-on experience in solving real-world data science and analytics problems.

---

# 👨‍💻 Author

**Muhammad Sufyan Shahid**

Data Science & Analytics Internship Project
DevelopersHub Corporation
