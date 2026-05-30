# 📊 Task 1: Term Deposit Subscription Prediction (Bank Marketing)

## 📌 Project Overview

This project aims to predict whether a bank customer will subscribe to a term deposit based on customer demographics, financial information, and previous marketing campaign interactions. The project utilizes Machine Learning classification models and Explainable AI (XAI) techniques to identify key factors influencing customer subscription decisions.

---

## 🎯 Task Objective

The objective of this project is to build a predictive model that determines whether a customer will subscribe to a term deposit after being contacted during a marketing campaign.

Key objectives include:

* Load and explore the Bank Marketing Dataset
* Perform data preprocessing and feature encoding
* Train classification models
* Evaluate model performance using multiple metrics
* Apply Explainable AI techniques (SHAP) to interpret predictions
* Generate business insights for marketing optimization

---

## 📂 Dataset

**Dataset:** Bank Marketing Dataset

The dataset contains customer demographic information, financial attributes, and campaign-related details used to predict term deposit subscriptions.

**Target Variable:**

* `y`

  * Yes = Customer subscribed
  * No = Customer did not subscribe

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SHAP (Explainable AI)

---

## 🔍 Project Approach

### 1. Data Exploration

* Loaded and examined the dataset structure
* Checked for missing values and duplicates
* Performed statistical analysis of features

### 2. Exploratory Data Analysis (EDA)

* Subscription distribution analysis
* Customer age distribution
* Job category analysis
* Education and marital status analysis
* Loan and housing impact analysis
* Correlation analysis

### 3. Data Preprocessing

* Encoded categorical features using Label Encoding
* Prepared features and target variable
* Split data into training and testing sets
* Applied feature scaling for Logistic Regression

### 4. Model Development

Two classification models were implemented:

#### Logistic Regression

A baseline linear classification model.

#### Random Forest Classifier

An ensemble learning model capable of capturing complex relationships in the data.

### 5. Model Evaluation

The models were evaluated using:

* Accuracy Score
* Confusion Matrix
* F1 Score
* ROC Curve
* ROC-AUC Score
* Classification Report

### 6. Explainable AI (XAI)

SHAP (SHapley Additive Explanations) was used to:

* Identify important features
* Explain model predictions
* Interpret customer subscription behavior
* Visualize feature contributions

---

## 📈 Results & Findings

## 📈 Results & Findings

### Model Performance

| Model               | Accuracy | F1 Score | ROC-AUC |
| ------------------- | -------- | -------- | ------- |
| Logistic Regression | 89.14%   | 0.327    | 0.873   |
| Random Forest       | 90.26%   | 0.452    | 0.926   |

### Key Findings

* Random Forest achieved the best overall performance with an accuracy of **90.26%** and ROC-AUC score of **0.926**.
* Logistic Regression provided a strong baseline model with an accuracy of **89.14%**.
* The higher ROC-AUC score of Random Forest indicates better discrimination between customers who subscribe and those who do not.
* SHAP analysis revealed that call duration, campaign interactions, account balance, and previous marketing outcomes were among the most influential features.
* The model can help banks identify customers with a higher probability of subscribing to term deposits, enabling more targeted marketing campaigns.


### Business Impact

* Enables banks to target high-potential customers more effectively.
* Improves marketing campaign efficiency.
* Reduces unnecessary marketing costs.
* Supports data-driven customer engagement strategies.

---

## 📊 Visualizations Included

* Customer Subscription Distribution
* Age Distribution
* Job Type Analysis
* Loan & Housing Analysis
* Correlation Heatmap
* Confusion Matrix
* ROC Curve
* Feature Importance Plot
* SHAP Summary Plot
* SHAP Waterfall Explanations

---

## 🚀 Future Improvements

* Implement XGBoost and CatBoost models
* Perform hyperparameter tuning
* Deploy as a Streamlit web application
* Integrate real-time prediction capabilities

---

## 👨‍💻 Author

**Muhammad Sufyan Shahid**

Data Science & Analytics Internship Project

DevelopersHub Corporation
