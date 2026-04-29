# Power Grid Fault Analysis

This project performs fault propagation and stability analysis for electrical grids using a reproducible machine learning pipeline.

## Overview
The repository currently includes:
1. EDA and preprocessing pipeline
2. Feature engineering pipeline
3. Baseline model training and evaluation
4. Learning algorithm stack for classification, anomaly detection, and segmentation

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
- `learning_algorithms.py`: PCA, RF, XGBoost, RF+XGBoost ensemble, Isolation Forest,
  K-Means, Stage 5 input export, and cross-validation reports
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
python learning_algorithms.py
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

### 4. Learning Algorithms (`learning_algorithms.py`)
Consumes the selected engineered splits from `outputs/day2/splits/`.

- PCA:
  - Fits on SMOTE training data only
  - Retains components explaining 95% of variance
  - Feeds transformed splits to Isolation Forest and K-Means
- Random Forest:
  - `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_split`,
    and `class_weight="balanced"`
  - Outputs binary probabilities for all original train/val/test samples
- XGBoost:
  - Tunes `scale_pos_weight` against validation PR-AUC
  - Outputs binary probabilities for all original train/val/test samples
- Ensemble:
  - Soft-votes RF and XGBoost probabilities
  - Weights each model by validation PR-AUC
  - Tunes decision threshold on the PR curve with recall favored
- Isolation Forest:
  - Fits on PCA-transformed SMOTE training data
  - Tunes contamination from 0.05 to 0.20 against validation F1
  - Outputs anomaly scores
- K-Means:
  - Uses PCA-transformed train/val/test data
  - Reports elbow and silhouette results
  - Exports named clusters assigned by observed fault rate:
    `normal operation`, `pre-fault stress`, `active fault`
- Evaluation:
  - Reports mean ± std across 5-fold Stratified K-Fold and TimeSeriesSplit
  - Includes PR-AUC, class-wise F1, MCC, ROC-AUC, confusion matrices, and
    calibration curve data

## Outputs
- `outputs/day1/`: EDA plots and preprocessing splits
- `outputs/day2/`: feature engineering plots, selected-feature splits, and importance table
- `outputs/learning_algorithms/stage5_graph_propagation_input.csv`: direct Stage 5 input with
  `[sample_id, RF_prob, XGBoost_prob, ensemble_prob, ensemble_label,
  IF_anomaly_score, kmeans_cluster]`
- `outputs/learning_algorithms/cv_metrics_mean_std.csv`: mean ± std evaluation metrics
- `outputs/learning_algorithms/cv_fold_metrics.csv`: per-fold evaluation metrics
- `outputs/learning_algorithms/artifacts/`: tuning tables, PCA summary, calibration curve data,
  and metadata
- `outputs/learning_algorithms/plots/`: K-Means diagnostics and confusion matrices

## Notes
- Scripts are deterministic where configured via `random_state=42`.
- Dummy baseline can show undefined precision warnings for the minority class; this is expected when it predicts only the majority class.
