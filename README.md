# Power Grid Fault Analysis

This project performs fault propagation and stability analysis for electrical grids using a reproducible machine learning pipeline.

## Overview
The repository currently includes:
1. EDA and preprocessing pipeline
2. Feature engineering pipeline
3. Baseline model training and evaluation

## Dataset
- Name: Electrical Grid Stability Simulated Data
- Source: UCI Machine Learning Repository
- Target:
  - `stabf`: binary class label (`stable` / `unstable`)
  - `stab`: continuous stability value (dropped to prevent leakage)

## Project Files
- `eda.py`: data loading, EDA, cleaning, scaling, train/val/test split, SMOTE, and artifact export
- `feature_engineering.py`: temporal + graph feature generation, feature selection, and engineered split export
- `baseline_models.py`: baseline classification models and metric reporting
- `requirements.txt`: Python dependencies

## Setup
Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run Order
Run scripts in this order:

```bash
python eda.py
python feature_engineering.py
python baseline_models.py
```

## Pipeline Details

### 1. EDA and Preprocessing (`eda.py`)
- Loads dataset (auto-download supported if file is missing)
- Performs inspection (shape, columns, dtypes, null checks, sample rows)
- Generates EDA plots:
  - Class distribution
  - Feature distributions by class
  - Correlation heatmap
  - Boxplots by class
- Cleans data:
  - Median imputation (if needed)
  - IQR-based outlier capping (3xIQR)
  - Drops `stab` to avoid target leakage
  - Encodes `stabf` as `stable=0`, `unstable=1`
- Splits data (stratified): 70% train / 15% val / 15% test
- Applies scaling:
  - StandardScaler
  - MinMaxScaler
- Applies SMOTE on train split only
- Saves outputs to `outputs/day1/`

### 2. Feature Engineering (`feature_engineering.py`)
- Repeats cleaning so the script is self-contained
- Adds temporal features:
  - Rolling mean/std/min/max on `tau*` and `p*` (windows 5/10/20)
  - Rate-of-change (`diff`) features
  - Lag features (t-1, t-2, t-3) for `g*` and `p*`
- Adds graph-derived features using a 4-node star topology:
  - Node degree, betweenness, clustering
  - Edge load ratios and `avg_edge_load`
- Performs feature selection:
  - Random Forest + permutation importance on validation split
  - Retains features contributing to 95% cumulative importance
- Saves engineered artifacts to `outputs/day2/`

### 3. Baseline Models (`baseline_models.py`)
Baseline-only setup (no tuning, no CV, no extra resampling):

- Train/test split: 80/20, stratified, `random_state=42`
- Models:
  - `DummyClassifier(strategy="most_frequent")`
  - `LogisticRegression(max_iter=1000)`
  - `RandomForestClassifier(random_state=42)`
- Metrics reported per model:
  - Classification report
  - Confusion matrix
  - ROC-AUC
  - PR-AUC (average precision)

## Outputs
- `outputs/day1/`: EDA plots and preprocessing splits
- `outputs/day2/`: feature engineering plots, selected-feature splits, and importance table

## Notes
- Scripts are deterministic where configured via `random_state=42`.
- Dummy baseline can show undefined precision warnings for the minority class; this is expected when it predicts only the majority class.
