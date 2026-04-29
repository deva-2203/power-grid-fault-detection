from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - exercised by environment setup.
    raise ImportError(
        "learning_algorithms.py requires xgboost. Install dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from exc


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "stabf"
SPLITS_DIR = Path("outputs/day2/splits")
OUTPUT_DIR = Path("outputs/learning_algorithms")
PLOTS_DIR = OUTPUT_DIR / "plots"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_train_smote: pd.DataFrame
    y_train_smote: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def _read_y(path: Path) -> pd.Series:
    return pd.read_csv(path)[TARGET].astype(int)


def load_engineered_splits() -> SplitData:
    required = [
        "X_train_eng_std.csv",
        "X_train_eng_smote.csv",
        "X_val_eng_std.csv",
        "X_test_eng_std.csv",
        "y_train.csv",
        "y_train_smote.csv",
        "y_val.csv",
        "y_test.csv",
    ]
    missing = [name for name in required if not (SPLITS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing engineered split artifacts. Run `python feature_engineering.py` "
            f"first. Missing: {missing}"
        )

    return SplitData(
        X_train=pd.read_csv(SPLITS_DIR / "X_train_eng_std.csv"),
        y_train=_read_y(SPLITS_DIR / "y_train.csv"),
        X_train_smote=pd.read_csv(SPLITS_DIR / "X_train_eng_smote.csv"),
        y_train_smote=_read_y(SPLITS_DIR / "y_train_smote.csv"),
        X_val=pd.read_csv(SPLITS_DIR / "X_val_eng_std.csv"),
        y_val=_read_y(SPLITS_DIR / "y_val.csv"),
        X_test=pd.read_csv(SPLITS_DIR / "X_test_eng_std.csv"),
        y_test=_read_y(SPLITS_DIR / "y_test.csv"),
    )


def fit_pca(data: SplitData) -> tuple[PCA, dict[str, np.ndarray]]:
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_train_smote_pca = pca.fit_transform(data.X_train_smote)
    transformed = {
        "train": pca.transform(data.X_train),
        "train_smote": X_train_smote_pca,
        "val": pca.transform(data.X_val),
        "test": pca.transform(data.X_test),
    }

    pca_summary = pd.DataFrame(
        {
            "component": np.arange(1, pca.n_components_ + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )
    pca_summary.to_csv(ARTIFACTS_DIR / "pca_95_variance_summary.csv", index=False)
    return pca, transformed


def tune_random_forest(data: SplitData) -> RandomForestClassifier:
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 8, 12, 16],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced"],
    }
    search = GridSearchCV(
        estimator=rf,
        param_grid=grid,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(data.X_train_smote, data.y_train_smote)
    pd.DataFrame(search.cv_results_).to_csv(
        ARTIFACTS_DIR / "rf_gridsearch_results.csv", index=False
    )
    best_model = search.best_estimator_
    best_model.set_params(n_jobs=-1)
    return best_model


def tune_xgboost(data: SplitData) -> XGBClassifier:
    original_ratio = (data.y_train == 0).sum() / max((data.y_train == 1).sum(), 1)
    candidates = sorted({0.5, 1.0, round(original_ratio, 3), 2.0, 5.0})
    rows = []
    best_model = None
    best_pr_auc = -np.inf

    for scale_pos_weight in candidates:
        model = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(data.X_train_smote, data.y_train_smote)
        val_prob = model.predict_proba(data.X_val)[:, 1]
        pr_auc = average_precision_score(data.y_val, val_prob)
        rows.append({"scale_pos_weight": scale_pos_weight, "val_pr_auc": pr_auc})
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model = model

    pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).to_csv(
        ARTIFACTS_DIR / "xgboost_scale_pos_weight_results.csv", index=False
    )
    if best_model is None:
        raise RuntimeError("XGBoost tuning failed to produce a fitted model.")
    return best_model


def choose_recall_favoring_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    min_recall: float = 0.90,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    candidates = np.where(recall >= min_recall)[0]
    if len(candidates) == 0:
        return float(thresholds[np.argmax(f1)])
    best_idx = candidates[np.argmax(f1[candidates])]
    return float(thresholds[best_idx])


def validation_weighted_ensemble(
    rf_model: RandomForestClassifier,
    xgb_model: XGBClassifier,
    data: SplitData,
) -> tuple[np.ndarray, float]:
    rf_val = rf_model.predict_proba(data.X_val)[:, 1]
    xgb_val = xgb_model.predict_proba(data.X_val)[:, 1]
    rf_pr_auc = average_precision_score(data.y_val, rf_val)
    xgb_pr_auc = average_precision_score(data.y_val, xgb_val)
    scores = np.array([rf_pr_auc, xgb_pr_auc], dtype=float)
    weights = scores / scores.sum() if scores.sum() > 0 else np.array([0.5, 0.5])
    return weights, choose_recall_favoring_threshold(
        data.y_val, weights[0] * rf_val + weights[1] * xgb_val
    )


def tune_isolation_forest(
    pca_splits: dict[str, np.ndarray],
    y_val: pd.Series,
) -> tuple[IsolationForest, float]:
    rows = []
    best_model = None
    best_f1 = -np.inf
    best_contamination = 0.05

    for contamination in np.round(np.arange(0.05, 0.201, 0.01), 2):
        model = IsolationForest(
            n_estimators=300,
            contamination=float(contamination),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(pca_splits["train_smote"])
        val_pred = (model.predict(pca_splits["val"]) == -1).astype(int)
        f1 = f1_score(y_val, val_pred, zero_division=0)
        rows.append({"contamination": contamination, "val_fault_f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_contamination = float(contamination)
            best_model = model

    pd.DataFrame(rows).sort_values("val_fault_f1", ascending=False).to_csv(
        ARTIFACTS_DIR / "isolation_forest_contamination_results.csv", index=False
    )
    if best_model is None:
        raise RuntimeError("Isolation Forest tuning failed to produce a fitted model.")
    return best_model, best_contamination


def fit_behavioral_kmeans(
    pca_splits: dict[str, np.ndarray],
    labels: pd.Series,
) -> tuple[KMeans, dict[int, str]]:
    X_for_selection = np.vstack(
        [pca_splits["train"], pca_splits["val"], pca_splits["test"]]
    )
    rows = []
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE)
        k_labels = model.fit_predict(X_for_selection)
        inertia = model.inertia_
        silhouette = silhouette_score(X_for_selection, k_labels)
        rows.append({"k": k, "inertia": inertia, "silhouette": silhouette})

    selection = pd.DataFrame(rows)
    selection.to_csv(ARTIFACTS_DIR / "kmeans_elbow_silhouette.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(selection["k"], selection["inertia"], marker="o")
    axes[0].set_title("K-Means Elbow")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(selection["k"], selection["silhouette"], marker="o", color="#D85A30")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kmeans_elbow_silhouette.png", dpi=150)
    plt.close()

    final_k = 3
    model = KMeans(n_clusters=final_k, n_init=30, random_state=RANDOM_STATE)
    cluster_ids = model.fit_predict(X_for_selection)

    fault_rates = (
        pd.DataFrame({"cluster": cluster_ids, "fault": labels.to_numpy()})
        .groupby("cluster")["fault"]
        .mean()
        .sort_values()
    )
    names = ["normal operation", "pre-fault stress", "active fault"]
    name_map = {int(cluster): name for cluster, name in zip(fault_rates.index, names)}
    with open(ARTIFACTS_DIR / "kmeans_cluster_name_map.json", "w", encoding="utf-8") as f:
        json.dump(name_map, f, indent=2)
    return model, name_map


def probabilities_for_all_splits(model, data: SplitData) -> dict[str, np.ndarray]:
    return {
        "train": model.predict_proba(data.X_train)[:, 1],
        "val": model.predict_proba(data.X_val)[:, 1],
        "test": model.predict_proba(data.X_test)[:, 1],
    }


def build_stage5_results(
    data: SplitData,
    pca_splits: dict[str, np.ndarray],
    rf_model: RandomForestClassifier,
    xgb_model: XGBClassifier,
    ensemble_weights: np.ndarray,
    ensemble_threshold: float,
    if_model: IsolationForest,
    kmeans_model: KMeans,
    cluster_names: dict[int, str],
) -> pd.DataFrame:
    rf_probs = probabilities_for_all_splits(rf_model, data)
    xgb_probs = probabilities_for_all_splits(xgb_model, data)

    rows = []
    for split_name in ["train", "val", "test"]:
        ensemble_prob = (
            ensemble_weights[0] * rf_probs[split_name]
            + ensemble_weights[1] * xgb_probs[split_name]
        )
        anomaly_score = -if_model.score_samples(pca_splits[split_name])
        kmeans_labels = kmeans_model.predict(pca_splits[split_name])
        for i in range(len(ensemble_prob)):
            rows.append(
                {
                    "sample_id": f"{split_name}_{i:06d}",
                    "RF_prob": rf_probs[split_name][i],
                    "XGBoost_prob": xgb_probs[split_name][i],
                    "ensemble_prob": ensemble_prob[i],
                    "ensemble_label": int(ensemble_prob[i] >= ensemble_threshold),
                    "IF_anomaly_score": anomaly_score[i],
                    "kmeans_cluster": cluster_names[int(kmeans_labels[i])],
                }
            )

    results = pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "RF_prob",
            "XGBoost_prob",
            "ensemble_prob",
            "ensemble_label",
            "IF_anomaly_score",
            "kmeans_cluster",
        ],
    )
    results.to_csv(OUTPUT_DIR / "stage5_graph_propagation_input.csv", index=False)
    return results


def _classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | list[list[int]]]:
    labels = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, probabilities),
        "f1_stable": f1_score(y_true, labels, pos_label=0, zero_division=0),
        "f1_unstable": f1_score(y_true, labels, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, labels),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, labels).tolist(),
    }


def _summarize_cv(rows: list[dict[str, float | str | list[list[int]]]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric = [
        "pr_auc",
        "f1_stable",
        "f1_unstable",
        "mcc",
        "roc_auc",
        "threshold",
    ]
    summary = (
        df.groupby(["cv_strategy", "model"])[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([part for part in col if part]).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    return summary


def _save_calibration_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    name: str,
    strategy: str,
) -> None:
    frac_pos, mean_pred = calibration_curve(
        y_true, probabilities, n_bins=10, strategy="quantile"
    )
    pd.DataFrame(
        {"mean_predicted_probability": mean_pred, "fraction_of_positives": frac_pos}
    ).to_csv(ARTIFACTS_DIR / f"calibration_{strategy}_{name}.csv", index=False)


def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    rf_factory: Callable[[], RandomForestClassifier],
    xgb_factory: Callable[[], XGBClassifier],
    strategy: str,
) -> tuple[pd.DataFrame, list[dict[str, float | str | list[list[int]]]]]:
    if strategy == "stratified":
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y)
    elif strategy == "timeseries":
        splitter = TimeSeriesSplit(n_splits=5)
        split_iter = splitter.split(X)
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")

    fold_rows = []
    out_of_fold = {"rf": [], "xgb": [], "ensemble": []}
    out_of_fold_y = []

    for fold, (train_idx, val_idx) in enumerate(split_iter, start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

        rf = rf_factory()
        xgb = xgb_factory()
        rf.fit(X_train_bal, y_train_bal)
        xgb.fit(X_train_bal, y_train_bal)

        rf_prob = rf.predict_proba(X_val)[:, 1]
        xgb_prob = xgb.predict_proba(X_val)[:, 1]
        rf_pr_auc = average_precision_score(y_val, rf_prob)
        xgb_pr_auc = average_precision_score(y_val, xgb_prob)
        weights = np.array([rf_pr_auc, xgb_pr_auc], dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.array([0.5, 0.5])
        ensemble_prob = weights[0] * rf_prob + weights[1] * xgb_prob

        model_probs = {
            "rf": rf_prob,
            "xgboost": xgb_prob,
            "ensemble": ensemble_prob,
        }
        for model_name, probabilities in model_probs.items():
            threshold = choose_recall_favoring_threshold(y_val, probabilities)
            metrics = _classification_metrics(y_val.to_numpy(), probabilities, threshold)
            fold_rows.append(
                {
                    "cv_strategy": strategy,
                    "fold": fold,
                    "model": model_name,
                    **metrics,
                }
            )

        out_of_fold["rf"].append(rf_prob)
        out_of_fold["xgb"].append(xgb_prob)
        out_of_fold["ensemble"].append(ensemble_prob)
        out_of_fold_y.append(y_val.to_numpy())

    y_oof = np.concatenate(out_of_fold_y)
    for key, chunks in out_of_fold.items():
        _save_calibration_curve(
            y_oof,
            np.concatenate(chunks),
            name="xgboost" if key == "xgb" else key,
            strategy=strategy,
        )

    return _summarize_cv(fold_rows), fold_rows


def save_confusion_plots(
    fold_rows: list[dict[str, float | str | list[list[int]]]],
) -> None:
    for strategy in sorted({str(row["cv_strategy"]) for row in fold_rows}):
        for model in sorted({str(row["model"]) for row in fold_rows}):
            matrices = [
                np.array(row["confusion_matrix"])
                for row in fold_rows
                if row["cv_strategy"] == strategy and row["model"] == model
            ]
            matrix = np.sum(matrices, axis=0)
            display = ConfusionMatrixDisplay(
                matrix, display_labels=["stable", "unstable"]
            )
            display.plot(cmap="Blues", values_format="d")
            plt.title(f"{strategy} CV confusion matrix - {model}")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"confusion_{strategy}_{model}.png", dpi=150)
            plt.close()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("=" * 72)
    print("LEARNING ALGORITHMS - Classification, Anomaly Detection, Segmentation")
    print("=" * 72)

    data = load_engineered_splits()
    print(f"Loaded engineered splits from {SPLITS_DIR}")
    print(
        f"Train/Val/Test: {len(data.X_train)} / {len(data.X_val)} / {len(data.X_test)} "
        f"| SMOTE train: {len(data.X_train_smote)}"
    )

    pca, pca_splits = fit_pca(data)
    print(
        f"PCA retained {pca.n_components_} components "
        f"({pca.explained_variance_ratio_.sum():.3%} variance)."
    )

    print("Tuning Random Forest with GridSearchCV...")
    rf_model = tune_random_forest(data)
    print(
        "Best RF params: "
        f"n_estimators={rf_model.n_estimators}, "
        f"max_depth={rf_model.max_depth}, "
        f"min_samples_split={rf_model.min_samples_split}, "
        f"class_weight={rf_model.class_weight}"
    )

    print("Tuning XGBoost scale_pos_weight...")
    xgb_model = tune_xgboost(data)
    print(f"Best XGBoost scale_pos_weight: {xgb_model.get_params()['scale_pos_weight']}")

    ensemble_weights, ensemble_threshold = validation_weighted_ensemble(
        rf_model, xgb_model, data
    )
    print(
        "Ensemble weights "
        f"(RF, XGBoost): {ensemble_weights.round(4).tolist()} | "
        f"threshold: {ensemble_threshold:.4f}"
    )

    if_model, best_contamination = tune_isolation_forest(pca_splits, data.y_val)
    print(f"Best Isolation Forest contamination: {best_contamination:.2f}")

    print("Selecting K-Means k with elbow/silhouette and fitting k=3...")
    y_all = pd.concat([data.y_train, data.y_val, data.y_test], ignore_index=True)
    kmeans_model, cluster_names = fit_behavioral_kmeans(pca_splits, y_all)
    print(f"K-Means cluster names: {cluster_names}")

    results = build_stage5_results(
        data=data,
        pca_splits=pca_splits,
        rf_model=rf_model,
        xgb_model=xgb_model,
        ensemble_weights=ensemble_weights,
        ensemble_threshold=ensemble_threshold,
        if_model=if_model,
        kmeans_model=kmeans_model,
        cluster_names=cluster_names,
    )
    print(f"Saved Stage 5 input with {len(results)} rows.")

    X_cv = pd.concat([data.X_train, data.X_val, data.X_test], ignore_index=True)
    y_cv = y_all

    rf_params = rf_model.get_params()
    xgb_params = xgb_model.get_params()

    def rf_factory() -> RandomForestClassifier:
        return RandomForestClassifier(**rf_params)

    def xgb_factory() -> XGBClassifier:
        return XGBClassifier(**xgb_params)

    print("Running 5-fold Stratified K-Fold evaluation...")
    stratified_summary, stratified_rows = evaluate_cv(
        X_cv, y_cv, rf_factory, xgb_factory, strategy="stratified"
    )
    print("Running TimeSeriesSplit evaluation...")
    timeseries_summary, timeseries_rows = evaluate_cv(
        X_cv, y_cv, rf_factory, xgb_factory, strategy="timeseries"
    )

    cv_summary = pd.concat([stratified_summary, timeseries_summary], ignore_index=True)
    cv_rows = stratified_rows + timeseries_rows
    cv_summary.to_csv(OUTPUT_DIR / "cv_metrics_mean_std.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(OUTPUT_DIR / "cv_fold_metrics.csv", index=False)
    save_confusion_plots(cv_rows)

    metadata = {
        "pca_components": int(pca.n_components_),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "rf_best_params": {
            key: value
            for key, value in rf_model.get_params().items()
            if key in {"n_estimators", "max_depth", "min_samples_split", "class_weight"}
        },
        "xgboost_scale_pos_weight": float(xgb_model.get_params()["scale_pos_weight"]),
        "ensemble_weights": {
            "random_forest": float(ensemble_weights[0]),
            "xgboost": float(ensemble_weights[1]),
        },
        "ensemble_threshold": float(ensemble_threshold),
        "isolation_forest_contamination": float(best_contamination),
        "kmeans_clusters": cluster_names,
        "threshold_policy": "PR-curve threshold maximizing F1 with recall >= 0.90 when possible",
        "note": (
            "TimeSeriesSplit uses the row order available in saved split files. "
            "If exact original timestamps are needed, preserve sample indices during "
            "feature_engineering.py export."
        ),
    }
    with open(OUTPUT_DIR / "learning_algorithms_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nEvaluation summary (mean ± std saved to cv_metrics_mean_std.csv):")
    print(cv_summary.to_string(index=False))
    print(f"\nOutputs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
