from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


@dataclass
class ModelResult:
    name: str
    report: str
    confusion: pd.DataFrame
    roc_auc: Optional[float]
    pr_auc: Optional[float]


def _prepare_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.nunique() != 2:
        raise ValueError(
            f"Target column '{target_col}' must be binary, found {y.nunique()} classes."
        )

    return X, y


def _get_positive_scores(model, X_test: pd.DataFrame) -> Optional[pd.Series]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.shape[1] == 2:
            return pd.Series(proba[:, 1], index=X_test.index)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return pd.Series(scores, index=X_test.index)

    return None


def evaluate_model(name: str, model, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> ModelResult:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = _get_positive_scores(model, X_test)

    report = classification_report(y_test, y_pred, digits=4)
    confusion = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )

    roc_auc = roc_auc_score(y_test, y_score) if y_score is not None else None
    pr_auc = average_precision_score(y_test, y_score) if y_score is not None else None

    return ModelResult(
        name=name,
        report=report,
        confusion=confusion,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
    )


def run_baselines(df: pd.DataFrame, target_col: str = "stabf") -> Dict[str, ModelResult]:
    X, y = _prepare_xy(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    models = {
        "DummyClassifier (most_frequent)": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
    }

    results: Dict[str, ModelResult] = {}
    for name, model in models.items():
        results[name] = evaluate_model(name, model, X_train, X_test, y_train, y_test)

    print("=" * 72)
    print("Baseline Model Evaluation (binary classification)")
    print("=" * 72)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print()

    for name, result in results.items():
        print("-" * 72)
        print(name)
        print("-" * 72)
        print("Classification Report:")
        print(result.report)
        print("Confusion Matrix:")
        print(result.confusion)

        if result.roc_auc is not None and result.pr_auc is not None:
            print(f"ROC-AUC: {result.roc_auc:.6f}")
            print(f"PR-AUC : {result.pr_auc:.6f}")
        else:
            print("ROC-AUC: N/A (model does not provide probability or decision scores)")
            print("PR-AUC : N/A (model does not provide probability or decision scores)")
        print()

    return results


if __name__ == "__main__":
    # Replace this loader with your own pipeline output as needed.
    input_df = pd.read_csv("data/Data_for_UCI_named.csv")
    input_df = input_df.drop(columns=["stab"], errors="ignore")
    if input_df["stabf"].dtype == object:
        input_df["stabf"] = (input_df["stabf"] == "unstable").astype(int)

    run_baselines(input_df, target_col="stabf")
