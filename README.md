# ⚡ Power Grid Fault Analysis — (EDA Pipeline and Data Preprocessing)

## 📌 Overview
This project focuses on **fault propagation and stability analysis in electrical power grids** using machine learning.

Building a complete **data foundation pipeline**, including:
- Data loading & inspection  
- Exploratory Data Analysis (EDA)  
- Data cleaning  
- Feature scaling  
- Class imbalance handling (SMOTE)  
- Train/validation/test split  

---

## Dataset
- **Name**: Electrical Grid Stability Simulated Data  
- **Source**: UCI Machine Learning Repository  
- **Target Variable**:
  - `stabf` → Binary classification (`stable` / `unstable`)
  - `stab` → Continuous stability score (dropped to prevent leakage)

---

## Pipeline Steps

### 1. Data Loading & Inspection
- Loads dataset from local path (auto-download supported)
- Displays:
  - Shape
  - Column names
  - Data types
  - Missing values
  - Sample rows

---

### 2. Exploratory Data Analysis
- Class distribution visualization  
- Feature distributions by class  
- Correlation heatmap  
- Boxplots for top features  


---

### 3. Data Cleaning
- Missing values → **Median imputation**
- Outliers → **IQR capping (3×IQR, not removed)**
- Drops `stab` (prevents target leakage)
- Encodes:
  - `stable = 0`
  - `unstable = 1`

---

### 4. Train / Validation / Test Split
- 70% Train  
- 15% Validation  
- 15% Test  
- Uses **stratified sampling** to preserve class balance

---

### 5. Feature Scaling
- **StandardScaler** → for general ML models  
- **MinMaxScaler** → for Isolation Forest / PCA  

---

### 6. Class Imbalance Handling
- Uses **SMOTE (Synthetic Minority Over-sampling Technique)** *(train split only)*

---

## ⚙️ Feature Engineering (Day 2)
Feature engineering expands the raw grid measurements into an enriched feature set for downstream modeling.

### What’s added
- **Temporal enrichment**
  - Rolling mean/std/min/max for `tau*` and `p*` over windows **5 / 10 / 20**
  - Rate-of-change (`diff`) features
  - Lag features (t-1, t-2, t-3) for `g*` and `p*`
- **Graph-derived features (4-node star topology)**
  - Node metrics: degree, betweenness, clustering
  - Edge load ratios + `avg_edge_load` using a simple capacity proxy
- **Feature selection**
  - Random Forest + permutation importance on validation split
  - Keeps features contributing to **95% cumulative importance**

### Run
```bash
python feature_engineering.py
```
