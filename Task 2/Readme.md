⚠️ If GitHub does not preview the notebook, please open it using nbviewer.
# 🛍️ Task 2: Customer Segmentation Using Unsupervised Learning

## 📌 Project Overview

This project focuses on customer segmentation using the Mall Customers Dataset. The objective is to group customers with similar characteristics and spending behaviors using K-Means Clustering. By identifying distinct customer segments, businesses can develop targeted marketing strategies, improve customer engagement, and enhance overall sales performance.

---

## 🎯 Task Objective

The main objective of this project is to cluster customers based on their demographic information and spending habits.

Key objectives include:

* Perform Exploratory Data Analysis (EDA)
* Apply K-Means Clustering for customer segmentation
* Determine the optimal number of clusters
* Visualize customer segments using PCA
* Develop marketing strategies tailored to each customer segment

---

## 📂 Dataset

**Dataset:** Mall Customers Dataset

The dataset contains customer information including:

* CustomerID
* Gender
* Age
* Annual Income (k$)
* Spending Score (1-100)

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* K-Means Clustering
* PCA (Principal Component Analysis)

---

## 🔍 Project Approach

### 1. Data Exploration

* Loaded and explored the dataset
* Checked data types and missing values
* Performed statistical analysis

### 2. Exploratory Data Analysis (EDA)

* Gender Distribution
* Age Distribution
* Annual Income Distribution
* Spending Score Distribution
* Income vs Spending Analysis
* Correlation Heatmap

### 3. Data Preprocessing

* Encoded categorical features
* Selected relevant features
* Standardized numerical data using StandardScaler

### 4. Customer Segmentation

* Applied the Elbow Method to evaluate cluster formation
* Used Silhouette Score to assess clustering quality
* Implemented K-Means Clustering to group customers into segments

### 5. Dimensionality Reduction

* Applied PCA to reduce dimensions
* Visualized customer clusters in a two-dimensional space

### 6. Business Strategy Development

* Analyzed cluster characteristics
* Proposed marketing strategies for each customer segment

---

## 📈 Results & Findings

### Customer Segments Identified

### Cluster 0 – Mature Customers

* Average Age: 56.5 years
* Average Income: 46.1k$
* Average Spending Score: 39.3

### Cluster 1 – High Income, Low Spending Customers

* Average Age: 39.5 years
* Average Income: 85.2k$
* Average Spending Score: 14.1

### Cluster 2 – High Income, High Spending Customers

* Average Age: 28.7 years
* Average Income: 60.9k$
* Average Spending Score: 70.2

### Cluster 3 – Potential Premium Customers

* Average Age: 37.9 years
* Average Income: 82.1k$
* Average Spending Score: 54.4

### Cluster 4 – Young Active Customers

* Average Age: 27.3 years
* Average Income: 38.8k$
* Average Spending Score: 56.2

### Key Findings

* High-income and high-spending customers represent the most valuable customer segment.
* High-income but low-spending customers present opportunities for targeted promotions.
* Younger customers generally exhibit higher spending behavior.
* Customer segmentation enables personalized marketing and improved customer engagement.

---

## 💡 Marketing Strategies

### High Income – High Spending Customers

* Offer VIP memberships
* Premium product recommendations
* Exclusive loyalty rewards

### High Income – Low Spending Customers

* Personalized discounts
* Premium promotional campaigns
* Exclusive shopping incentives

### Young Active Customers

* Social media marketing
* Trend-focused campaigns
* Student and youth discounts

### Mature Customers

* Loyalty programs
* Personalized communication
* Value-based offers

### Potential Premium Customers

* Cross-selling opportunities
* Membership upgrades
* Seasonal promotions

---

## 📊 Visualizations Included

* Gender Distribution
* Age Distribution
* Annual Income Distribution
* Spending Score Distribution
* Income vs Spending Scatter Plot
* Correlation Heatmap
* Elbow Method Plot
* Silhouette Score Analysis
* K-Means Cluster Visualization
* PCA Cluster Visualization
* Cluster Summary Heatmap

---

## 🚀 Future Improvements

* Apply Hierarchical Clustering for comparison
* Use t-SNE for advanced visualization
* Build an interactive Streamlit dashboard
* Perform customer lifetime value analysis

---

## 👨‍💻 Author

**Muhammad Sufyan Shahid**

Data Science & Analytics Internship Project

DevelopersHub Corporation
