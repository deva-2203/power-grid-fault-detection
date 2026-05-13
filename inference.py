"""
inference.py — Single-sample and sequence inference for the Power Grid Fault Detection pipeline.

Loads all saved models and scalers from disk once (cached) and exposes:
  - predict_single(raw: dict) -> dict
  - predict_sequence(rows: list[dict]) -> list[dict]   (last row has full temporal context)

Raw input keys: tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DAY2        = ROOT / "outputs" / "day2"
LA          = ROOT / "outputs" / "learning_algorithms"
MODELS_DIR  = LA / "models"
SPLITS_DIR  = DAY2 / "splits"

RAW_FEATURE_COLS = ["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]

FEATURE_LABELS = {
    "tau1": "Node 0 \u2014 Reaction time (\u03c41)",
    "tau2": "Node 1 \u2014 Reaction time (\u03c42)",
    "tau3": "Node 2 \u2014 Reaction time (\u03c43)",
    "tau4": "Node 3 \u2014 Reaction time (\u03c44)",
    "p1":   "Node 0 \u2014 Power output (p1)",
    "p2":   "Node 1 \u2014 Power demand (p2)",
    "p3":   "Node 2 \u2014 Power demand (p3)",
    "p4":   "Node 3 \u2014 Power demand (p4)",
    "g1":   "Node 0 \u2014 Price elasticity (g1)",
    "g2":   "Node 1 \u2014 Price elasticity (g2)",
    "g3":   "Node 2 \u2014 Price elasticity (g3)",
    "g4":   "Node 3 \u2014 Price elasticity (g4)",
}


# ── Lazy loaders (cached after first call) ────────────────────────────────────

@lru_cache(maxsize=1)
def _load_models() -> dict:
    return {
        "rf":       joblib.load(MODELS_DIR / "rf_classifier.joblib"),
        "xgb":      joblib.load(MODELS_DIR / "xgboost_classifier.joblib"),
        "pca":      joblib.load(MODELS_DIR / "pca.joblib"),
        "if":       joblib.load(MODELS_DIR / "isolation_forest.joblib"),
        "kmeans":   joblib.load(MODELS_DIR / "kmeans.joblib"),
        "ensemble": joblib.load(MODELS_DIR / "ensemble_meta.joblib"),
        "scaler":   joblib.load(DAY2 / "standard_scaler.joblib"),
    }


@lru_cache(maxsize=1)
def _load_selected_features() -> list[str]:
    return pd.read_csv(SPLITS_DIR / "selected_features.csv")["feature"].tolist()


@lru_cache(maxsize=1)
def _load_stable_stats() -> pd.DataFrame:
    """Per-feature mean and std computed from the stable (class=0) training samples."""
    X_train = pd.read_csv(SPLITS_DIR / "X_train_eng_raw.csv")
    y_train = pd.read_csv(SPLITS_DIR / "y_train.csv")["stabf"].astype(int)
    stable = X_train[y_train == 0]
    return pd.DataFrame({"mean": stable.mean(), "std": stable.std().clip(lower=1e-6)})


@lru_cache(maxsize=1)
def _load_raw_stable_stats() -> dict:
    """Mean/std/range of stable samples in the raw 12-feature space."""
    stable_raw = pd.read_csv(ROOT / "outputs" / "day1" / "splits" / "X_train_raw.csv")
    y_day1     = pd.read_csv(ROOT / "outputs" / "day1" / "splits" / "y_train.csv")["stabf"].astype(int)
    stable_only = stable_raw[y_day1 == 0]
    return {
        "mean": stable_only[RAW_FEATURE_COLS].mean().to_dict(),
        "std":  stable_only[RAW_FEATURE_COLS].std().clip(lower=1e-6).to_dict(),
        "p25":  stable_only[RAW_FEATURE_COLS].quantile(0.25).to_dict(),
        "p75":  stable_only[RAW_FEATURE_COLS].quantile(0.75).to_dict(),
        "min":  stable_only[RAW_FEATURE_COLS].min().to_dict(),
        "max":  stable_only[RAW_FEATURE_COLS].max().to_dict(),
    }


# ── Feature engineering (mirrors feature_engineering.py exactly) ──────────────

def _build_graph_features(row: pd.Series) -> dict:
    """Compute graph-derived features for a single row using 4-node star topology."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    capacities = {
        (0, 1): abs(row["g1"]) + abs(row["g2"]) + 1e-9,
        (0, 2): abs(row["g1"]) + abs(row["g3"]) + 1e-9,
        (0, 3): abs(row["g1"]) + abs(row["g4"]) + 1e-9,
    }
    flows = {(0, 1): abs(row["p2"]), (0, 2): abs(row["p3"]), (0, 3): abs(row["p4"])}
    for edge, cap in capacities.items():
        load_ratio = min(flows[edge] / cap, 1.0)
        G.add_edge(*edge, weight=load_ratio, capacity=cap, flow=flows[edge])

    deg  = dict(G.degree())
    bet  = nx.betweenness_centrality(G, weight="weight", normalized=True)
    clst = nx.clustering(G, weight="weight")

    feats = {}
    for n in range(4):
        feats[f"node{n}_degree"]      = deg.get(n, 0)
        feats[f"node{n}_betweenness"] = bet.get(n, 0.0)
        feats[f"node{n}_clustering"]  = clst.get(n, 0.0)

    el01 = G[0][1]["weight"] if G.has_edge(0, 1) else 0.0
    el02 = G[0][2]["weight"] if G.has_edge(0, 2) else 0.0
    el03 = G[0][3]["weight"] if G.has_edge(0, 3) else 0.0
    feats["edge_load_0_1"] = el01
    feats["edge_load_0_2"] = el02
    feats["edge_load_0_3"] = el03
    feats["avg_edge_load"] = np.mean([el01, el02, el03])
    return feats


def _engineer_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply temporal + graph feature engineering to a raw DataFrame.
    Works for any number of rows >= 1.
    Single-row behaviour:
      rolling stats = value itself, rate-of-change = 0, lags = same value.
    """
    df = df_raw[RAW_FEATURE_COLS].copy().reset_index(drop=True)

    rolling_features = [c for c in RAW_FEATURE_COLS if c.startswith("tau") or c.startswith("p")]
    lag_features     = ["g1","g2","g3","g4"] + [c for c in RAW_FEATURE_COLS if c.startswith("p")]
    windows          = [5, 10, 20]

    new_cols = {}

    # Rolling statistics
    for col in rolling_features:
        for w in windows:
            new_cols[f"roll_mean_{col}_w{w}"] = df[col].rolling(w, min_periods=1).mean()
            new_cols[f"roll_std_{col}_w{w}"]  = df[col].rolling(w, min_periods=1).std().fillna(0)
            new_cols[f"roll_min_{col}_w{w}"]  = df[col].rolling(w, min_periods=1).min()
            new_cols[f"roll_max_{col}_w{w}"]  = df[col].rolling(w, min_periods=1).max()

    # Rate-of-change
    for col in rolling_features:
        new_cols[f"roc_{col}"] = df[col].diff().fillna(0)

    # Lag features — bfill then fillna(original) handles single-row edge case
    for col in lag_features:
        for lag in [1, 2, 3]:
            lagged = df[col].shift(lag).bfill()
            # If bfill couldn't fill (single row) fall back to the current value
            new_cols[f"lag{lag}_{col}"] = lagged.fillna(df[col])

    # Concat all new temporal columns at once (avoids fragmentation)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Graph features (row-local, no temporal dependency)
    graph_rows = [_build_graph_features(df.iloc[i]) for i in range(len(df))]
    df = pd.concat([df, pd.DataFrame(graph_rows, index=df.index)], axis=1)

    return df



def _run_inference(df_eng: pd.DataFrame) -> list[dict]:
    """
    Given an engineered DataFrame (any number of rows), run all models and
    return a list of result dicts — one per row.
    """
    models    = _load_models()
    sel_feats = _load_selected_features()
    stable    = _load_stable_stats()

    # Select only the features the models were trained on
    available = [f for f in sel_feats if f in df_eng.columns]
    X = df_eng[available].copy()

    # Fill any missing selected features with 0 (shouldn't happen in practice)
    for f in sel_feats:
        if f not in X.columns:
            X[f] = 0.0
    X = X[sel_feats]

    # Scale
    X_scaled = pd.DataFrame(
        models["scaler"].transform(X),
        columns=sel_feats,
    )

    # PCA
    X_pca = models["pca"].transform(X_scaled)

    # Classifier probabilities
    rf_probs  = models["rf"].predict_proba(X_scaled)[:, 1]
    xgb_probs = models["xgb"].predict_proba(X_scaled)[:, 1]

    ens_meta   = models["ensemble"]
    w          = ens_meta["weights"]
    threshold  = float(ens_meta["threshold"])
    ens_probs  = w[0] * rf_probs + w[1] * xgb_probs

    # Anomaly scores (negate score_samples so higher = more anomalous)
    if_scores = -models["if"].score_samples(X_pca)

    # K-Means clusters
    cluster_ids   = models["kmeans"].predict(X_pca)
    cluster_names = ens_meta["cluster_names"]
    # joblib preserves int keys; fall back to str lookup for JSON-loaded metadata
    clusters = [cluster_names.get(int(cid), cluster_names.get(str(cid), "unknown"))
                for cid in cluster_ids]

    # Z-scores vs stable training distribution (only for selected+available features)
    common = [f for f in sel_feats if f in stable.index]
    z_means = stable.loc[common, "mean"]
    z_stds  = stable.loc[common, "std"]

    results = []
    for i in range(len(X)):
        z_row = ((X.iloc[i][common] - z_means) / z_stds).to_dict()
        results.append({
            "RF_prob":         float(rf_probs[i]),
            "XGBoost_prob":    float(xgb_probs[i]),
            "ensemble_prob":   float(ens_probs[i]),
            "ensemble_label":  int(ens_probs[i] >= threshold),
            "IF_anomaly_score": float(if_scores[i]),
            "kmeans_cluster":  clusters[i],
            "threshold":       threshold,
            "z_scores":        z_row,           # dict: feature → z-score vs stable mean
        })
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def predict_single(raw: dict) -> dict:
    """
    Predict fault risk for a single reading of 12 raw features.

    Temporal features are approximated (no prior context):
      rolling stats = the value itself, rate-of-change = 0, lags = same value.

    Parameters
    ----------
    raw : dict with keys tau1-4, p1-4, g1-4

    Returns
    -------
    dict with RF_prob, XGBoost_prob, ensemble_prob, ensemble_label,
         IF_anomaly_score, kmeans_cluster, threshold, z_scores,
         risk_tier, counterfactuals, alert_reason
    """
    df_raw = pd.DataFrame([raw])
    df_eng = _engineer_dataframe(df_raw)
    result = _run_inference(df_eng)[0]

    # Risk tier
    result["risk_tier"] = _tier(result["ensemble_prob"])

    # Override z_scores with raw-feature z-scores (all 12 guaranteed, more interpretable)
    raw_stats = _load_raw_stable_stats()
    raw_z = {
        feat: (raw[feat] - raw_stats["mean"][feat]) / raw_stats["std"][feat]
        for feat in RAW_FEATURE_COLS if feat in raw
    }
    result["z_scores"] = raw_z

    # Counterfactuals — for each raw feature, what happens if it returns to stable median
    result["counterfactuals"] = _compute_counterfactuals(raw, result["ensemble_prob"])

    # Top z-score deviations (for "why is this risky?" panel)
    result["alert_reason"] = _top_deviations(raw_z, n=3)

    return result


def predict_sequence(rows: list[dict]) -> list[dict]:
    """
    Predict for a sequence of readings. The last row benefits from full
    rolling/lag context computed from all preceding rows.

    Returns one result dict per row (same keys as predict_single).
    """
    df_raw = pd.DataFrame(rows)
    df_eng = _engineer_dataframe(df_raw)
    results = _run_inference(df_eng)
    for r in results:
        r["risk_tier"] = _tier(r["ensemble_prob"])
        r["counterfactuals"] = None   # skip for sequence (expensive)
        r["alert_reason"] = _top_deviations(r["z_scores"], n=3)
    return results


def predict_batch(rows: list[dict]) -> list[dict]:
    """
    Predict for a batch of **independent** readings — each row is engineered
    in isolation (no temporal context from neighboring rows).

    Use this for uploaded datasets where rows are independent samples,
    NOT a time series.  Skips counterfactuals for speed.

    Returns one result dict per row.
    """
    # Engineer each row independently (single-row temporal approximation)
    eng_frames = []
    for row in rows:
        df_single = pd.DataFrame([row])
        eng_frames.append(_engineer_dataframe(df_single))

    # Stack all engineered rows and run inference as one batch
    df_all = pd.concat(eng_frames, ignore_index=True)
    results = _run_inference(df_all)

    # Enrich with raw-feature z-scores, risk tiers, and alert reasons
    raw_stats = _load_raw_stable_stats()
    for i, (r, row) in enumerate(zip(results, rows)):
        r["risk_tier"] = _tier(r["ensemble_prob"])

        raw_z = {
            feat: (row[feat] - raw_stats["mean"][feat]) / raw_stats["std"][feat]
            for feat in RAW_FEATURE_COLS if feat in row
        }
        r["z_scores"] = raw_z
        r["alert_reason"] = _top_deviations(raw_z, n=3)
        r["counterfactuals"] = None

    return results


def stable_defaults() -> dict:
    """Return the median of stable training samples as default input values."""
    stats = _load_raw_stable_stats()
    return {k: float(round(v, 4)) for k, v in stats["mean"].items()
            if k in RAW_FEATURE_COLS}


def feature_ranges() -> dict[str, dict]:
    """Return per-feature {min, max, p25, p75, mean} from stable training samples."""
    stats = _load_raw_stable_stats()
    return {
        feat: {
            "min":  float(round(stats["min"][feat], 4)),
            "max":  float(round(stats["max"][feat], 4)),
            "p25":  float(round(stats["p25"][feat], 4)),
            "p75":  float(round(stats["p75"][feat], 4)),
            "mean": float(round(stats["mean"][feat], 4)),
        }
        for feat in RAW_FEATURE_COLS
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _tier(prob: float) -> str:
    if prob < 0.4:
        return "Green"
    if prob <= 0.7:
        return "Amber"
    return "Red"


def _top_deviations(z_scores: dict, n: int = 3) -> list[dict]:
    """Return the n features with the largest absolute z-score deviation."""
    sorted_z = sorted(z_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": k, "z_score": round(v, 2)} for k, v in sorted_z[:n]]


def _compute_counterfactuals(raw: dict, baseline_prob: float) -> list[dict]:
    """
    For each of the 12 raw features, replace it with its stable-region mean
    and re-run inference. Report the delta in ensemble probability.
    Only return features where replacing improves (lowers) the probability.
    """
    stable = _load_raw_stable_stats()
    stable_means = stable["mean"]

    counterfactuals = []
    for feat in RAW_FEATURE_COLS:
        if feat not in stable_means:
            continue
        stable_val = stable_means[feat]
        # Skip if the input is already at the stable mean (no change possible)
        if abs(raw[feat] - stable_val) < 1e-6:
            continue
        modified = {**raw, feat: stable_val}
        df_raw = pd.DataFrame([modified])
        df_eng = _engineer_dataframe(df_raw)
        new_prob = _run_inference(df_eng)[0]["ensemble_prob"]
        delta = new_prob - baseline_prob   # negative = improvement
        counterfactuals.append({
            "feature":       feat,
            "current_value": raw[feat],
            "stable_value":  round(stable_val, 4),
            "new_prob":      round(new_prob, 4),
            "delta":         round(delta, 4),
        })

    # Sort by biggest improvement (most negative delta) first
    counterfactuals.sort(key=lambda x: x["delta"])
    return counterfactuals


# ── Scenario Simulation ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_stable_test_rows() -> list[dict]:
    """Load stable rows from the test split as simulation starting points."""
    X_test = pd.read_csv(ROOT / "outputs" / "day1" / "splits" / "X_test_raw.csv")
    y_test  = pd.read_csv(ROOT / "outputs" / "day1" / "splits" / "y_test.csv")["stabf"].astype(int)
    stable  = X_test[y_test == 0][RAW_FEATURE_COLS].reset_index(drop=True)
    return stable.to_dict("records")


@lru_cache(maxsize=1)
def load_clearly_stable_rows() -> list[dict]:
    """
    Return stable test rows where the model itself also predicts low fault probability (<0.35).
    These are the best starting points for simulation — they'll show a clean Green→Red transition.
    Returns a list of (row_dict, baseline_prob) tuples.
    """
    import warnings
    stable_rows = load_stable_test_rows()
    # Batch inference on all stable rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = predict_sequence(stable_rows)
    screened = [
        {"row": stable_rows[i], "baseline_prob": round(results[i]["ensemble_prob"], 4),
         "index": i, "cluster": results[i]["kmeans_cluster"]}
        for i in range(len(stable_rows))
        if results[i]["ensemble_prob"] < 0.35
    ]
    # Sort by baseline prob ascending (most clearly stable first)
    screened.sort(key=lambda x: x["baseline_prob"])
    return screened


def run_scenario(
    base_row: dict,
    events: list[dict],
    n_steps: int = 40,
    fix_at_step: int | None = None,
    fix_events: list[dict] | None = None,
) -> list[dict]:
    """
    Run a simulation scenario starting from base_row.

    Parameters
    ----------
    base_row   : dict of 12 raw features — the initial stable state
    events     : list of perturbation dicts:
                   {feature, start_step, end_step, start_val, end_val}
    n_steps    : total number of timesteps to simulate
    fix_at_step: if set, the perturbation events are cancelled from this step
                 and fix_events (if any) are applied instead — simulates operator intervention
    fix_events : replacement events applied after fix_at_step

    Returns
    -------
    list[dict] — one per timestep, each with all inference result keys plus:
        "step"      : int — timestep index
        "raw_input" : dict — the 12 raw feature values at this step
    """
    def _apply_events(step: int, evts: list[dict], row: dict) -> dict:
        row = dict(row)
        for ev in evts:
            s, e = ev["start_step"], ev["end_step"]
            if s <= step <= e:
                t = (step - s) / max(e - s, 1)
                row[ev["feature"]] = ev["start_val"] + t * (ev["end_val"] - ev["start_val"])
            elif step > e:
                row[ev["feature"]] = ev["end_val"]
        return row

    rows = []
    for step in range(n_steps):
        if fix_at_step is not None and step >= fix_at_step:
            active_events = fix_events or []
            # Rebase fix events so step 0 = fix_at_step
            rebased = []
            for ev in active_events:
                rebased.append({**ev,
                    "start_step": ev["start_step"] + fix_at_step,
                    "end_step":   ev["end_step"]   + fix_at_step})
            row = _apply_events(step, rebased, base_row)
        else:
            row = _apply_events(step, events, base_row)
        rows.append(row)

    results = predict_sequence(rows)
    for i, (result, row) in enumerate(zip(results, rows)):
        result["step"]      = i
        result["raw_input"] = row
    return results


# Pre-built scenario templates — used by the UI dropdown
PRESET_SCENARIOS = {
    "Consumer Spike — Node 1": {
        "description": "Node 1 power demand ramps up sharply over 15 steps, overwhelming the producer. Simulates a sudden industrial load surge.",
        "events": [{"feature": "p2", "start_step": 5, "end_step": 20, "start_val": None, "end_val": "3x"}],
        "suggested_fix": "Reduce Node 1 load (p2) back toward baseline",
    },
    "Consumer Spike — Node 2": {
        "description": "Node 2 power demand triples over 15 steps. Simulates an unexpected large load connection.",
        "events": [{"feature": "p3", "start_step": 5, "end_step": 20, "start_val": None, "end_val": "3x"}],
        "suggested_fix": "Shed Node 2 load (p3)",
    },
    "Generator Sag": {
        "description": "The producer (Node 0) gradually loses generation capacity — p1 drifts toward 20% of normal. Simulates partial generator failure or fuel depletion.",
        "events": [{"feature": "p1", "start_step": 3, "end_step": 25, "start_val": None, "end_val": "0.2x"}],
        "suggested_fix": "Restore generator output (p1) or shed consumer loads",
    },
    "Response Slowdown — Node 1": {
        "description": "Node 1 response time (tau2) degrades to near-maximum. Simulates controller lag, network delay, or actuator failure.",
        "events": [{"feature": "tau2", "start_step": 5, "end_step": 25, "start_val": None, "end_val": 9.8}],
        "suggested_fix": "Restore normal response time (tau2)",
    },
    "Cascading Overload": {
        "description": "Node 1 spikes first, Node 2 follows 10 steps later. Simulates a cascading failure where one overload triggers another.",
        "events": [
            {"feature": "p2", "start_step": 3,  "end_step": 18, "start_val": None, "end_val": "3x"},
            {"feature": "p3", "start_step": 13, "end_step": 28, "start_val": None, "end_val": "3x"},
        ],
        "suggested_fix": "Shed Node 1 load first (p2), then Node 2 (p3)",
    },
}
