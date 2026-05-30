# ============================================================
# Task 1: Term Deposit Subscription Prediction
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,roc_curve,roc_auc_score

# ============================================================
# 1. Load Dataset
# ============================================================

df = pd.read_csv("bank-full.csv", sep=";")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ============================================================
# 2. Exploratory Data Analysis
# ============================================================

plt.figure(figsize=(6, 4))
sns.countplot(x="y", data=df)
plt.title("Term Deposit Subscription Distribution")
plt.xlabel("Subscribed")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(12, 5))
sns.countplot(x="job", data=df)
plt.title("Job Type Distribution")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="marital", hue="y", data=df)
plt.title("Marital Status vs Subscription")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="education", hue="y", data=df)
plt.title("Education vs Subscription")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="housing", hue="y", data=df)
plt.title("Housing Loan vs Subscription")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="loan", hue="y", data=df)
plt.title("Personal Loan vs Subscription")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="y", y="balance", data=df)
plt.title("Balance vs Subscription")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="y", y="duration", data=df)
plt.title("Call Duration vs Subscription")
plt.show()

# ============================================================
# 3. Encode Categorical Features
# ============================================================

df_encoded = df.copy()

categorical_cols = df_encoded.select_dtypes(include=["object"]).columns

print("Categorical Columns:")
print(categorical_cols)

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

print(df_encoded.head())
print(df_encoded.dtypes)

# ============================================================
# 4. Correlation Heatmap
# ============================================================

plt.figure(figsize=(14, 8))
sns.heatmap(df_encoded.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ============================================================
# 5. Feature Selection
# ============================================================

X = df_encoded.drop("y", axis=1)
y = df_encoded["y"]

# ============================================================
# 6. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y )

# ============================================================
# 7. Feature Scaling for Logistic Regression
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 8. Logistic Regression Model
# ============================================================

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)
log_prob = log_model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# 9. Random Forest Model
# ============================================================

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# ============================================================
# 10. Model Evaluation
# ============================================================

print("\n================ Logistic Regression Results ================")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("F1 Score:", f1_score(y_test, log_pred))
print("ROC AUC Score:", roc_auc_score(y_test, log_prob))


print("\n================ Random Forest Results ================")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))
print("ROC AUC Score:", roc_auc_score(y_test, rf_prob))




# ============================================================
# 11. Random Forest Confusion Matrix
# ============================================================

plt.figure(figsize=(6, 4))
cm_rf = confusion_matrix(y_test, rf_pred)

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# Logistic Regression Confusion Matrix
# ============================================================

cm_lr = confusion_matrix(y_test, log_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['No','Yes'],
    yticklabels=['No','Yes']
)

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ============================================================
# Comparing Both Confusion Matrix
# ============================================================

fig, ax = plt.subplots(1, 2, figsize=(12,5))

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=ax[0])
ax[0].set_title("Logistic Regression")

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax[1])
ax[1].set_title("Random Forest")

plt.tight_layout()
plt.show()

# ============================================================
# 12. ROC Curve
# ============================================================

log_fpr, log_tpr, _ = roc_curve(y_test, log_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

plt.figure(figsize=(8, 6))
plt.plot(log_fpr, log_tpr, label="Logistic Regression")
plt.plot(rf_fpr, rf_tpr, label="Random Forest")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# ============================================================
# 13. Feature Importance - Random Forest
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance
)

plt.title("Random Forest Feature Importance")
plt.show()

# ============================================================
# 14. SHAP Explainability - Fixed Version
# ============================================================

X_test_sample = X_test.sample(100, random_state=42)

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer(X_test_sample)

print("SHAP values shape:", shap_values.values.shape)
print("X_test_sample shape:", X_test_sample.shape)

# SHAP Summary Plot for class 1 = subscribed yes
shap.summary_plot(
    shap_values[:, :, 1],
    X_test_sample,
    show=True
)

# Explain at least 5 predictions using waterfall plots
for i in range(5):
    print(f"\nExplanation for Prediction {i + 1}")
    print("Actual Value:", y_test.loc[X_test_sample.index[i]])
    print("Predicted Value:", rf_model.predict(X_test_sample.iloc[[i]])[0])
    print("Subscription Probability:", rf_model.predict_proba(X_test_sample.iloc[[i]])[0][1])

    shap.plots.waterfall(
        shap_values[i, :, 1],
        show=True
    )
## Conclusion

# This project successfully developed machine learning models to predict whether a customer will subscribe to a term deposit. 
# Logistic Regression and Random Forest were trained and evaluated using performance metrics such as Accuracy, F1-Score, and ROC-AUC. 
# Among the two models, Random Forest achieved the best performance.
# Additionally, SHAP was used to explain model predictions and identify the most influential factors affecting customer subscription decisions. 
# The results demonstrate how machine learning can help banks improve marketing strategies and target potential customers more effectively.

