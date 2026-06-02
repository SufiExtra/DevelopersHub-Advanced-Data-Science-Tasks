⚠️ If GitHub does not preview the notebook, please open it using nbviewer.

# 💰 Task 4: Loan Default Risk with Business Cost Optimization

## 📌 Project Overview

This project focuses on predicting loan default risk using the Home Credit Default Risk dataset. The goal is not only to predict whether a customer may default, but also to optimize the decision threshold based on business cost.

The project applies binary classification models and cost-based evaluation to support smarter and more risk-aware loan approval decisions.

---

## 🎯 Task Objective

The objective of this task is to predict the likelihood of loan default and optimize the classification threshold using cost-benefit analysis.

Key objectives include:

* Clean and preprocess the dataset
* Train binary classification models
* Evaluate model performance
* Define business costs for false positives and false negatives
* Optimize the decision threshold to minimize total business cost
* Analyze important features affecting loan default risk

---

## 📂 Dataset

**Dataset:** Home Credit Default Risk Dataset

The dataset contains customer, financial, and loan-related information.

### Target Variable

* `TARGET`

  * `0` = No Default
  * `1` = Default

### Dataset Shape

* Rows: 307,511
* Columns: 122

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* CatBoost

---

## 🔍 Project Approach

### 1. Data Exploration

* Loaded and explored the dataset
* Checked dataset shape and column information
* Analyzed missing values
* Examined target class distribution

### 2. Data Preprocessing

* Removed unnecessary ID column
* Handled missing values
* Filled numerical columns using median values
* Filled categorical columns using mode values
* Encoded categorical features for Logistic Regression
* Used CatBoost for handling categorical features effectively

### 3. Model Building

Two classification models were trained:

#### Logistic Regression

Used as a baseline classification model.

#### CatBoost Classifier

Used as the advanced model because it performs well on tabular data and handles categorical features effectively.

### 4. Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix
* Classification Report

---

## 📈 Results & Findings

### Model Performance

| Model               | Accuracy | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | -----: | -------: | ------: |
| Logistic Regression |   68.29% | 66.70% |    0.253 |   0.742 |
| CatBoost            |   72.63% | 64.47% |    0.275 |   0.751 |
| Optimized CatBoost  |   74.74% | 61.86% |    0.283 |   0.751 |

### Key Findings

* The dataset was highly imbalanced, with only around 8% default cases.
* CatBoost performed better than Logistic Regression overall.
* Accuracy alone was not enough due to class imbalance.
* Recall, F1 Score, ROC-AUC, and business cost were more meaningful evaluation metrics.
* The best cost-optimized threshold was **0.52**.

---

## 💼 Business Cost Optimization

A cost-based threshold optimization approach was used.

### Business Assumption

* False Positive: Rejecting a good customer
* False Negative: Approving a customer who defaults

Since false negatives are more financially risky, they were assigned a higher cost.

### Cost Values

* False Positive Cost = 1
* False Negative Cost = 10

### Best Threshold

| Metric             | Value |
| ------------------ | ----: |
| Best Threshold     |  0.52 |
| False Positives    |  2219 |
| False Negatives    |   307 |
| Minimum Total Cost |  5289 |

---

## 🔝 Feature Importance

The most important features identified by CatBoost included:

* EXT_SOURCE_3
* EXT_SOURCE_2
* EXT_SOURCE_1
* DAYS_EMPLOYED
* DAYS_LAST_PHONE_CHANGE
* DAYS_BIRTH
* AMT_CREDIT
* AMT_GOODS_PRICE
* NAME_EDUCATION_TYPE
* AMT_ANNUITY

These features played a major role in predicting loan default risk.

---

## 📊 Visualizations Included

* Target Distribution
* Logistic Regression Confusion Matrix
* CatBoost Confusion Matrix
* ROC Curve Comparison
* Business Cost vs Threshold
* Optimized Threshold Confusion Matrix
* CatBoost Feature Importance Plot

---

## 🚀 Future Improvements

* Train on the full dataset instead of a sample
* Perform hyperparameter tuning for CatBoost
* Add LightGBM or XGBoost for comparison
* Use SHAP for deeper model explainability
* Deploy the model using Streamlit

---

## 📝 Conclusion

This project successfully built a loan default risk prediction model using Logistic Regression and CatBoost. CatBoost achieved better performance and was further improved using business cost-based threshold optimization. The best threshold was **0.52**, which minimized the total business cost. This project demonstrates how machine learning can help financial institutions make more informed and cost-effective loan approval decisions.

---

## 👨‍💻 Author

**Muhammad Sufyan Shahid**

Data Science & Analytics Internship Project
DevelopersHub Corporation
