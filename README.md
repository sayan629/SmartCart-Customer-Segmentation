<div align="center">

# 🛒 SmartCart Customer Segmentation using Machine Learning

<img src="https://img.icons8.com/color/96/shopping-cart.png" width="80">
<img src="https://img.icons8.com/color/96/artificial-intelligence.png" width="80">
<img src="https://img.icons8.com/color/96/combo-chart.png" width="80">
<img src="https://img.icons8.com/color/96/scatter-plot.png" width="80">

**Unsupervised Learning • Customer Analytics • Business Intelligence**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-Clustering-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Project Objectives](#-project-objectives)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Dataset](#-dataset)
- [Notebook Walkthrough](#-notebook-walkthrough)
- [Technologies Used](#️-technologies-used)
- [Cluster Insights](#-cluster-insights)
- [Conclusion](#-conclusion)

---

## 📖 Project Overview

<p align="center">
<img src="https://img.icons8.com/color/480/customer-insight.png" width="420">
</p>

Customer segmentation is one of the most valuable applications of Machine Learning in retail and e-commerce. Since every customer has unique purchasing habits, shopping frequency, income level, and product preferences, treating all customers the same often leads to ineffective marketing strategies.

The objective of this project is to segment **SmartCart customers** into meaningful groups using **Unsupervised Machine Learning (Clustering)**.

The project follows a complete Machine Learning pipeline:

1. **Data Preprocessing** — handling missing values, dropping unnecessary columns, cleaning the dataset
2. **Feature Engineering** — creating `Age`, `Customer_Tenure_Days`, `Total_Spending`, `Total_Children`, and simplifying `Education` and `Living_With`
3. **Exploratory Data Analysis** — pair plots and correlation heatmaps to understand feature relationships and spot outliers
4. **Outlier Removal** — dropping extreme observations to improve clustering quality
5. **Encoding & Scaling** — One-Hot Encoding for categorical variables, `StandardScaler` for numerical features
6. **Dimensionality Reduction** — Principal Component Analysis (PCA) for visualization and modeling
7. **Optimal K Selection** — combining the **Elbow Method** and **Silhouette Score**, resulting in **K = 4**
8. **Clustering** — comparing **K-Means** and **Agglomerative Hierarchical Clustering**
9. **Cluster Analysis** — profiling each segment by income, spending, and engagement

Based on the visualization results, **Agglomerative Clustering** produces more compact and well-separated customer groups, making it the preferred algorithm for this project.

---

## 🎯 Project Objectives

| # | Objective |
|---|-----------|
| 🧹 | Clean and preprocess customer data |
| ⚙️ | Create meaningful customer features |
| 📊 | Explore customer purchasing behavior |
| 🚫 | Detect and remove outliers |
| 🔄 | Encode categorical variables |
| 📏 | Scale numerical features |
| 📉 | Reduce dimensions using PCA |
| 🎯 | Determine the optimal number of clusters |
| 🤖 | Apply K-Means Clustering |
| 🌳 | Apply Agglomerative Clustering |
| 📈 | Compare clustering performance |
| 💡 | Generate business insights |

---

## 🔄 Machine Learning Workflow

```text
📂 Load Dataset
        │
        ▼
🧹 Data Cleaning
        │
        ▼
⚙️ Feature Engineering
        │
        ▼
📊 Exploratory Data Analysis
        │
        ▼
🚫 Outlier Removal
        │
        ▼
🔄 One-Hot Encoding
        │
        ▼
📏 Feature Scaling
        │
        ▼
📉 PCA
        │
        ▼
🎯 Find Best K
        │
        ▼
🤖 K-Means        🌳 Agglomerative
        │                │
        └───────┬────────┘
                ▼
        📈 Cluster Analysis
                │
                ▼
        💡 Business Insights
```

---

## 🗂 Dataset

The raw dataset contains **2,240 customers** and **22 columns**, including demographic details (`Year_Birth`, `Education`, `Marital_Status`, `Income`), purchase behavior (`MntWines`, `MntFruits`, `MntMeatProducts`, `NumWebPurchases`, `NumStorePurchases`, etc.), and campaign response fields.

| Raw Column | Description |
|---|---|
| `Year_Birth` | Customer's birth year → used to derive `Age` |
| `Income` | Yearly household income |
| `Kidhome` / `Teenhome` | Number of children/teens at home |
| `Dt_Customer` | Date the customer joined |
| `Mnt*` columns | Amount spent per product category |
| `Num*Purchases` | Purchase counts by channel |
| `NumWebVisitsMonth` | Website engagement |

---

## 📓 Notebook Walkthrough

### 1️⃣ Data Preprocessing — Handling Missing Values

Missing `Income` values are imputed with the median, and the dataset is inspected with `isnull().sum()` and `head()`.

<p align="center">
<img src="https://img.icons8.com/fluency/96/data-configuration.png" width="70">
</p>

```python
df["Income"] = df["Income"].fillna(df["Income"].median())
```

### 2️⃣ Feature Engineering

New, business-meaningful features are engineered from the raw columns:

| New Feature | Formula |
|---|---|
| `Age` | `2026 - Year_Birth` |
| `Customer_Tenure_Days` | `max(Dt_Customer) - Dt_Customer` |
| `Total_Spending` | Sum of all `Mnt*` product spending columns |
| `Total_Children` | `Kidhome + Teenhome` |
| `Education` | Simplified to `Undergraduate`, `Graduate`, `PostGraduate` |
| `Living_With` | Simplified to `Alone` or `Partner` |

### 3️⃣ Dropping Redundant Columns

`ID`, `Year_Birth`, `Marital_Status`, `Kidhome`, `Teenhome`, `Dt_Customer`, and the individual `Mnt*` spending columns are dropped once their engineered replacements exist.

### 4️⃣ Exploratory Data Analysis & Outlier Detection

<p align="center">
<img src="https://img.icons8.com/fluency/96/scatter-plot.png" width="70">
<img src="https://img.icons8.com/fluency/96/box-plot.png" width="70">
</p>

A **pair plot** across `Income`, `Recency`, `Response`, `Age`, `Total_Spending`, and `Total_Children` reveals extreme income and age outliers.

```python
df_cleaned = df_cleaned[(df_cleaned["Age"] < 90)]
df_cleaned = df_cleaned[(df_cleaned["Income"] < 600_000)]
```

> **Result:** Data size reduced from **2,240 → 2,236** records after outlier removal.

A correlation **heatmap** (`sns.heatmap`) is then used to check relationships between numerical features before modeling.

### 5️⃣ Feature Encoding & Scaling

Categorical columns (`Education`, `Living_With`) are transformed with `OneHotEncoder`, then all numerical features are standardized with `StandardScaler` so that no single feature dominates the distance-based clustering algorithms.

### 6️⃣ Dimensionality Reduction — PCA

<p align="center">
<img src="https://img.icons8.com/fluency/96/3d-select-tool.png" width="70">
</p>

Both a **2D** and **3D** PCA projection are generated to visualize the customer feature space:

```python
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
# explained_variance_ratio_ ≈ [0.232, 0.114, 0.104]
```

The first three principal components together explain roughly **45%** of the total variance — enough to reveal clear structure in the data.

### 7️⃣ Finding the Optimal Number of Clusters

Two complementary techniques are used and then plotted together on a combined dual-axis chart:

| Method | Result |
|---|---|
| 📉 Elbow Method (`KneeLocator`) | **Best K = 4** |
| 📈 Silhouette Score | Peaks around **K = 4–5**, confirming the elbow result |

<p align="center">
<img src="https://img.icons8.com/fluency/96/statistics.png" width="70">
</p>

### 8️⃣ Clustering — K-Means vs. Agglomerative

Both algorithms are fit with **K = 4** and visualized in 3D PCA space:

| Algorithm | Observation |
|---|---|
| 🤖 **K-Means** | Reasonable separation, but boundaries between clusters overlap more |
| 🌳 **Agglomerative (Ward linkage)** | Tighter, more compact, and better-separated clusters |

> ✅ **Decision:** Agglomerative Clustering is selected for downstream analysis based on visual cluster compactness and separation.

### 9️⃣ Characterization & Business Insights

The final cluster labels are attached back to the feature set, and each segment is profiled using count plots, scatter plots (e.g., **Income vs. Total Spending**), and a `groupby("clusters").mean()` summary table to describe each segment's income level, spending habits, and channel engagement.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| 🐍 Python | Programming Language |
| 🐼 Pandas | Data Manipulation |
| 🔢 NumPy | Numerical Computing |
| 📊 Matplotlib | Visualization |
| 📈 Seaborn | Statistical Plots |
| 🤖 Scikit-Learn | Machine Learning |
| 📉 PCA | Dimensionality Reduction |
| 🌳 Agglomerative Clustering | Hierarchical Clustering |
| 🎯 K-Means | Customer Segmentation |
| 🦾 kneed (`KneeLocator`) | Elbow Point Detection |

---

## 💡 Cluster Insights

<p align="center">
<img src="https://img.icons8.com/fluency/96/conference-call.png" width="70">
</p>

Using the **Income vs. Total Spending** scatter plot (colored by cluster), four distinct customer segments emerge:

| Cluster | Profile |
|---|---|
| 🔴 **0** | Lower income, lower spending — price-sensitive shoppers |
| 🔵 **1** | Higher income, higher spending — premium/loyal customers |
| 🟡 **2** | Lower-to-mid income, low spending — occasional/deal-driven shoppers |
| 🟢 **3** | Mid-to-high income, moderate-to-high spending — steady, engaged customers |

These segments give SmartCart a data-driven foundation for targeted campaigns, loyalty programs, and product recommendations.

---

## 📌 Conclusion

This project demonstrates an end-to-end Machine Learning pipeline for customer segmentation — from raw transactional data to actionable business insight.

Using data preprocessing, feature engineering, PCA, and clustering algorithms, SmartCart customers are grouped into **four meaningful customer segments**.

Among the evaluated algorithms, **Agglomerative Clustering** produces better-defined customer groups than K-Means, making it the preferred model for this dataset.

These customer segments can help the business:

- 🎯 Deliver Personalized Marketing
- 💰 Increase Revenue
- ❤️ Improve Customer Retention
- 📦 Offer Better Product Recommendations
- 📈 Support Data-Driven Decision Making

<div align="center">

### ⭐ SmartCart Clustering Project ⭐

**Machine Learning • Customer Analytics • Business Intelligence**

</div>
