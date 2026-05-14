# ⚡ Power Grid Fault Detection

> A machine learning system that monitors a simulated 4-node power grid in real-time and predicts grid instability before it happens, displayed through a modern web dashboard.

![Dashboard](https://img.shields.io/badge/Dashboard-React-blue) ![Backend](https://img.shields.io/badge/Backend-FastAPI-green) ![ML](https://img.shields.io/badge/ML-XGBoost%20%2B%20Random%20Forest-orange)

---

## What It Does

This system predicts whether a power grid will become **unstable (fault)** or remain **stable** in real-time, by analyzing sensor data from a 4-node electrical grid network. It takes raw measurements from each node — reaction time, power output/demand, and grid cooperation — and uses trained machine learning models to calculate the probability of a fault occurring.

## Who It's For

- **Grid operators** monitoring a power network in a control room
- **Researchers** studying decentralized smart grid stability
- **Engineers** who want to detect faults *before* they cascade into blackouts

## How It Works

```
Raw IEEE Dataset → EDA → Feature Engineering → Model Training → Trained Models
                                                                      ↓
                                              Browser ← React ← FastAPI ← Inference Engine
```

1. **Data** — Uses the IEEE Electrical Grid Stability dataset: 10,000 simulated readings from a 4-node star network (1 generator + 3 consumers), each with 12 features
2. **Training** (`learning_algorithms.py`) — Trains an ensemble of ML models (XGBoost, Random Forest, Isolation Forest, PCA + K-Means) combined into a meta-classifier with calibrated fault probability
3. **Inference** (`inference.py`) — Loads trained models, engineers features (graph metrics, interaction terms), scales data, and runs predictions: raw reading → fault probability (0–100%) + risk tier
4. **API** (`api.py`) — FastAPI backend that serves predictions over HTTP
5. **Dashboard** (`frontend/`) — React web app with live-scrolling sensor charts, z-score zones, and per-node warnings

## The 4-Node Grid Topology

```
        Node 1 (Consumer)
            |
            |
Node 2 ---- Node 0 ---- Node 3
(Consumer)  (Generator)  (Consumer)
```

| Node | Role | Sensors | Description |
|------|------|---------|-------------|
| Node 0 | Generator | `tau1`, `p1`, `g1` | Central power producer, `p1` is always positive |
| Node 1 | Consumer 1 | `tau2`, `p2`, `g2` | Draws power, `p2` is always negative |
| Node 2 | Consumer 2 | `tau3`, `p3`, `g3` | Draws power, `p3` is always negative |
| Node 3 | Consumer 3 | `tau4`, `p4`, `g4` | Draws power, `p4` is always negative |

Each node has 3 monitored features:

| Feature | Meaning |
|---------|---------|
| `tau` (τ) | Reaction time — how quickly the node adjusts its price signal |
| `p` | Power — positive = producing energy, negative = consuming energy |
| `g` (γ) | Cooperation coefficient — how willing the node is to cooperate for grid stability (0 to 1) |

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Machine Learning** | Python, XGBoost, scikit-learn (Random Forest, Isolation Forest, PCA, K-Means), NumPy, Pandas |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | React, Vite, Tailwind CSS, Recharts, Framer Motion |

---

## Overview

The repository includes:
1. EDA and preprocessing pipeline
2. Feature engineering pipeline
3. Baseline model training and evaluation
4. Learning algorithm stack for classification, anomaly detection, and segmentation
5. Real-time inference engine and FastAPI backend
6. Interactive React web dashboard for live grid monitoring

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
- `inference.py`: model loading and single/sequence prediction engine
- `api.py`: FastAPI backend serving trained model predictions via REST API
- `run.sh`: one-command launcher for the backend and frontend
- `frontend/`: React web dashboard (Vite + Tailwind CSS + Recharts)
- `requirements.txt`: Python dependencies

## Quick Start
From the project root, run:

```bash
./run.sh
```

This starts both services:

```text
Backend  → http://127.0.0.1:8000
Frontend → http://127.0.0.1:3000
```

Open **http://127.0.0.1:3000** and click **Start** to begin the live simulation.

The launcher will create `.venv`, install Python/frontend dependencies, and rebuild ML artifacts if they are missing.

To use different ports:

```bash
BACKEND_PORT=8001 FRONTEND_PORT=3001 ./run.sh
```

## Manual Pipeline
If you want to manually rebuild the ML pipeline, run scripts in this order:

```bash
python eda.py
python feature_engineering.py
python baseline_models.py
python learning_algorithms.py
```

### Manual Dashboard Startup

After training is complete, start the dashboard:

**Terminal 1 — Backend:**
```bash
uvicorn api:app --port 8000 --host 0.0.0.0
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm install
npm run dev -- --port 3000 --host
```

Open **http://localhost:3000** and click **Start** to begin the live simulation.

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

### 5. Inference Engine (`inference.py`)

Loads all trained models and scalers once (cached via `lru_cache`) and provides a full prediction pipeline from raw 12-feature input to actionable risk output.

#### Public Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `predict_single` | `(raw: dict) → dict` | Predict fault risk for a single reading. Temporal features approximated (no prior context). |
| `predict_sequence` | `(rows: list[dict]) → list[dict]` | Predict for a sequence of readings. The last row benefits from full rolling/lag temporal context. |
| `stable_defaults` | `() → dict` | Returns the median of stable training samples as default input values. |
| `feature_ranges` | `() → dict[str, dict]` | Returns per-feature `{min, max, p25, p75, mean}` from stable training samples. |

**Input keys:** `tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4`

#### Output Fields (per prediction)

| Field | Type | Description |
|-------|------|-------------|
| `RF_prob` | float | Random Forest fault probability (0–1) |
| `XGBoost_prob` | float | XGBoost fault probability (0–1) |
| `ensemble_prob` | float | Weighted ensemble probability |
| `ensemble_label` | int | Binary prediction (1 = fault) using tuned threshold |
| `IF_anomaly_score` | float | Isolation Forest anomaly score (higher = more anomalous) |
| `kmeans_cluster` | str | Cluster label: `normal operation`, `pre-fault stress`, or `active fault` |
| `threshold` | float | Calibrated decision threshold used for `ensemble_label` |
| `z_scores` | dict | Per-feature z-score vs stable training distribution |
| `risk_tier` | str | `Green` (< 0.4), `Amber` (0.4–0.7), or `Red` (> 0.7) |
| `counterfactuals` | list | Per-feature "what if" analysis (single-prediction only) |
| `alert_reason` | list | Top-3 features with the largest z-score deviations |

#### Risk Tier System

```
Ensemble Probability    Tier     Meaning
─────────────────────   ─────    ──────────────────────────────
< 0.4                   Green    Grid operating normally
0.4 – 0.7              Amber    Elevated risk — monitor closely
> 0.7                   Red      High fault probability — act now
```

#### Counterfactual Analysis

For each of the 12 raw features, the engine replaces it with the stable-region mean and re-runs inference. Returns the delta in ensemble probability — showing an operator exactly *which feature to fix* and *how much it would help*:

```json
{
  "feature": "p2",
  "current_value": -3.8,
  "stable_value": -1.2,
  "new_prob": 0.42,
  "delta": -0.31
}
```

#### Scenario Simulation

`run_scenario()` simulates multi-step grid events with optional operator intervention:

```python
run_scenario(
    base_row,                    # Initial stable state (12 features)
    events=[                     # Perturbation events
        {"feature": "p2", "start_step": 5, "end_step": 20,
         "start_val": None, "end_val": "3x"}
    ],
    n_steps=40,                  # Total timesteps
    fix_at_step=25,              # When operator intervenes (optional)
    fix_events=[...]             # Corrective events (optional)
)
```

#### Preset Scenarios

| Scenario | Description |
|----------|-------------|
| Consumer Spike — Node 1 | Node 1 demand ramps 3× over 15 steps (industrial load surge) |
| Consumer Spike — Node 2 | Node 2 demand triples (unexpected large load connection) |
| Generator Sag | Generator output drops to 20% (partial failure / fuel depletion) |
| Response Slowdown — Node 1 | Node 1 reaction time degrades to near-max (controller lag) |
| Cascading Overload | Node 1 spikes first, Node 2 follows 10 steps later (cascading failure) |

---

### 6. FastAPI Backend (`api.py`)

REST API that serves trained model predictions to the frontend. Uses CORS middleware for cross-origin access.

#### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/stats` | Returns stable-region statistics (`mean`, `std`, `p25`, `p75`, `min`, `max` per feature) for computing z-scores in the frontend |
| `GET` | `/api/scenarios` | Returns the preset scenario registry (names, descriptions, event definitions) |
| `GET` | `/api/feed/dataset` | Runs batch inference on the test split, sorts by fault probability (stable → fault), and caches results to `outputs/precomputed_feed.pkl` for fast replay |
| `GET` | `/api/feed/scenario/{name}` | Runs a preset scenario (40 steps) using the most clearly-stable test row as the starting point and returns the full simulation feed |

#### Example Usage

```bash
# Get stable statistics
curl http://localhost:8000/api/stats

# List available scenarios
curl http://localhost:8000/api/scenarios

# Get the full dataset replay feed (cached after first call)
curl http://localhost:8000/api/feed/dataset

# Run a specific scenario
curl http://localhost:8000/api/feed/scenario/Generator%20Sag
```

---

### 7. Frontend Dashboard (`frontend/`)

React SPA (Vite + Tailwind CSS v4) that visualizes live grid predictions from the API.

#### Architecture

| File | Role |
|------|------|
| `App.jsx` | Main app — fetches data, manages playback state, renders the 4-node grid layout |
| `components/NodeTile.jsx` | Per-node tile with 3 mini-charts, status badge, and animated warning bar |

#### Key Features

- **4-Node Grid View** — One tile per grid node (Generator + 3 Consumers), each showing 3 live-scrolling sensor charts:
  - Reaction Time (τ)
  - Power Output / Demand (p)
  - Grid Cooperation (γ)
- **Z-Score Visualization** — All sensor values are displayed as z-scores relative to stable training distribution. Chart zones:
  - **Green band** (±1.5σ): Normal operation
  - **Amber band** (±1.5σ to ±2.5σ): Elevated deviation
  - **Red band** (beyond ±2.5σ): Critical deviation
- **Per-Node Status Badge** — Automatically computed from worst sensor z-score:
  - `Normal` (< 1.5σ) — green
  - `Elevated` (1.5–2.5σ) — amber
  - `Critical` (> 2.5σ) — red
- **Animated Warning Bars** — 10-second sticky warnings with Framer Motion slide-in when any sensor breaches threshold, showing the worst metric and deviation direction
- **Playback Controls** — Start/Stop/Reset with step counter (1 reading/second, 60-step rolling window)
- **Dependencies** — React 19, Recharts, Framer Motion, Lucide React icons, Tailwind CSS v4

---

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
