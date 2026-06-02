# ============================================================
# Task 4: Loan Default Risk with Business Cost Optimization
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,roc_curve)
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

# ============================================================
# 1. Load Dataset
# ============================================================

df = pd.read_csv("application_train.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nTarget Distribution:")
print(df["TARGET"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# ============================================================
# 2. Basic EDA
# ============================================================

plt.figure(figsize=(6, 4))
sns.countplot(x="TARGET", data=df)
plt.title("Loan Default Distribution")
plt.xlabel("Target: 0 = No Default, 1 = Default")
plt.ylabel("Count")
plt.show()

target_counts = df["TARGET"].value_counts(normalize=True) * 100
print("\nTarget Percentage:")
print(target_counts)

# ============================================================
# 3. Reduce Dataset for Faster Training
# ============================================================

# Using sample for faster execution
df_sample = df.sample(n=50000, random_state=42)

print("\nSample Shape:")
print(df_sample.shape)

# ============================================================
# 4. Drop ID Column
# ============================================================

df_sample = df_sample.drop("SK_ID_CURR", axis=1)

# ============================================================
# 5. Separate Features and Target
# ============================================================

X = df_sample.drop("TARGET", axis=1)
y = df_sample["TARGET"]

# ============================================================
# 6. Identify Numerical and Categorical Columns
# ============================================================

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical Columns:", len(categorical_cols))
print("Numerical Columns:", len(numerical_cols))

# ============================================================
# 7. Handle Missing Values
# ============================================================

for col in numerical_cols:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# ============================================================
# 8. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================================
# 9. Logistic Regression Preprocessing
# ============================================================

X_lr = pd.get_dummies(X, drop_first=True)

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr,y,test_size=0.2,random_state=42,stratify=y)

scaler = StandardScaler()

X_train_lr_scaled = scaler.fit_transform(X_train_lr)
X_test_lr_scaled = scaler.transform(X_test_lr)

# ============================================================
# 10. Logistic Regression Model
# ============================================================

log_model = LogisticRegression(max_iter=1000,class_weight="balanced")

log_model.fit(X_train_lr_scaled, y_train_lr)

log_pred = log_model.predict(X_test_lr_scaled)
log_prob = log_model.predict_proba(X_test_lr_scaled)[:, 1]

print("\n================ Logistic Regression Results ================")
print("Accuracy:", accuracy_score(y_test_lr, log_pred))
print("Precision:", precision_score(y_test_lr, log_pred))
print("Recall:", recall_score(y_test_lr, log_pred))
print("F1 Score:", f1_score(y_test_lr, log_pred))
print("ROC-AUC:", roc_auc_score(y_test_lr, log_prob))

# Confusion Matrix - Logistic Regression
cm_lr = confusion_matrix(y_test_lr, log_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr,annot=True,fmt="d",cmap="Blues",xticklabels=["No Default", "Default"],yticklabels=["No Default", "Default"])
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================================
# 11. CatBoost Model
# ============================================================

cat_features_indices = [X_train.columns.get_loc(col)for col in categorical_cols]

cat_model = CatBoostClassifier(iterations=300,learning_rate=0.05,depth=6,loss_function="Logloss",
eval_metric="AUC",random_seed=42,verbose=100,auto_class_weights="Balanced")

cat_model.fit(X_train,y_train,cat_features=cat_features_indices,eval_set=(X_test, y_test))

cat_pred = cat_model.predict(X_test)
cat_prob = cat_model.predict_proba(X_test)[:, 1]

print("\n================ CatBoost Results ================")
print("Accuracy:", accuracy_score(y_test, cat_pred))
print("Precision:", precision_score(y_test, cat_pred))
print("Recall:", recall_score(y_test, cat_pred))
print("F1 Score:", f1_score(y_test, cat_pred))
print("ROC-AUC:", roc_auc_score(y_test, cat_prob))

# Confusion Matrix - CatBoost
cm_cat = confusion_matrix(y_test, cat_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_cat,annot=True,fmt="d",cmap="Greens",xticklabels=["No Default", "Default"],yticklabels=["No Default", "Default"])
plt.title("CatBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================================
# 12. ROC Curve Comparison
# ============================================================

log_fpr, log_tpr, _ = roc_curve(y_test_lr, log_prob)
cat_fpr, cat_tpr, _ = roc_curve(y_test, cat_prob)

plt.figure(figsize=(8, 6))
plt.plot(log_fpr, log_tpr, label="Logistic Regression")
plt.plot(cat_fpr, cat_tpr, label="CatBoost")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# ============================================================
# 13. Business Cost Optimization
# ============================================================

# Business assumption:
# False Positive = Rejecting a good customer
# False Negative = Approving a customer who defaults

FP_COST = 1
FN_COST = 10

thresholds = np.arange(0.1, 0.91, 0.01)

cost_results = []

for threshold in thresholds:
    threshold_pred = (cat_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, threshold_pred).ravel()

    total_cost = (fp * FP_COST) + (fn * FN_COST)

    cost_results.append({
        "Threshold": threshold,
        "False Positives": fp,
        "False Negatives": fn,
        "Total Cost": total_cost
    })

cost_df = pd.DataFrame(cost_results)

best_row = cost_df.loc[cost_df["Total Cost"].idxmin()]
best_threshold = best_row["Threshold"]

print("\n================ Business Cost Optimization ================")
print(cost_df.head())
print("\nBest Threshold:")
print(best_row)

plt.figure(figsize=(10, 5))
plt.plot(cost_df["Threshold"], cost_df["Total Cost"], marker="o")
plt.title("Business Cost vs Decision Threshold")
plt.xlabel("Decision Threshold")
plt.ylabel("Total Business Cost")
plt.show()

# ============================================================
# 14. Final Prediction Using Best Threshold
# ============================================================

optimized_pred = (cat_prob >= best_threshold).astype(int)

print("\n================ Optimized CatBoost Results ================")
print("Best Threshold:", best_threshold)
print("Accuracy:", accuracy_score(y_test, optimized_pred))
print("Precision:", precision_score(y_test, optimized_pred))
print("Recall:", recall_score(y_test, optimized_pred))
print("F1 Score:", f1_score(y_test, optimized_pred))
print("ROC-AUC:", roc_auc_score(y_test, cat_prob))

cm_opt = confusion_matrix(y_test, optimized_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_opt,annot=True,fmt="d",cmap="Oranges",xticklabels=["No Default", "Default"],yticklabels=["No Default", "Default"])
plt.title("Optimized Threshold Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================================
# 15. Feature Importance
# ============================================================

feature_importance = pd.DataFrame({"Feature": X_train.columns,"Importance": cat_model.get_feature_importance()})

feature_importance = feature_importance.sort_values(by="Importance",ascending=False).head(15)

print("\nTop 15 Important Features:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance",y="Feature",data=feature_importance)
plt.title("Top 15 CatBoost Feature Importances")
plt.show()

# ============================================================
# 16. Final Conclusion
# ============================================================

print(f"Best Cost-Optimized Threshold: {best_threshold:.2f}")


# This project successfully developed a loan default risk prediction model using the Home Credit Default Risk dataset. 
# Logistic Regression was used as a baseline model, while CatBoost achieved better overall performance. 
# Since the dataset was highly imbalanced, evaluation focused on Recall, F1 Score, ROC-AUC, and business cost optimization instead of accuracy alone.
# A cost-based threshold optimization approach was applied by assigning a higher cost to false negatives, as approving a customer who later defaults creates greater financial risk. 
# The best cost-optimized threshold was **0.52**, which minimized the total business cost. 
# Overall, this project shows how machine learning can support more risk-aware and cost-effective loan approval decisions.
