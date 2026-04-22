# ⚡ Power Grid Fault Analysis — (EDA Pipeline and Data Preprocessing)

## 📌 Overview
This project focuses on **fault propagation and stability analysis in electrical power grids** using machine learning.

Building a complete **data foundation pipeline**, including:
- Data loading & inspection  
- Exploratory Data Analysis (EDA)  
- Data cleaning  
- Feature scaling  
- Class imbalance handling (SMOTE vs class_weight)  
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
- Uses **SMOTE (Synthetic Minority Over-sampling Technique)**
- Compares:
  - SMOTE vs `class_weight="balanced"` (Logistic Regression baseline)
- Evaluated using **PR-AUC**

---

### 7. Output Artifacts

#### Saved Data Splits:

- Raw:
  - `X_train_raw.csv`, `X_val_raw.csv`, `X_test_raw.csv`
- Standard Scaled:
  - `X_train_std.csv`, `X_val_std.csv`, `X_test_std.csv`
- MinMax Scaled:
  - `X_train_mm.csv`, `X_val_mm.csv`, `X_test_mm.csv`
- SMOTE:
  - `X_train_smote.csv`
- Labels:
  - `y_train.csv`, `y_val.csv`, `y_test.csv`

---

#### 📊 Saved Visualizations:

- Class distribution  
- Feature histograms  
- Correlation heatmap  
- Boxplots  
- SMOTE comparison  

---

