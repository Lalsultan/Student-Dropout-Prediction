"""Leakage-aware baseline evaluation for the student dropout project.

Expected input: data/raw/student dropout.csv
Run from the repository root: python src/evaluate_models.py

This script intentionally excludes Final_Grade because its availability at the
intended prediction time has not been established. Review that assumption before
changing the feature list.
"""
from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "student dropout.csv"
TARGET = "Dropped_Out"
RANDOM_STATE = 42


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Obtain it from an authorized source, "
            "verify its terms, then place it at data/raw/student dropout.csv."
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing.")
    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values; resolve them before evaluation.")

    y = df[TARGET].astype(int)
    # Final_Grade is excluded because prediction-time availability is unverified.
    X = df.drop(columns=[TARGET, "Final_Grade"], errors="ignore")

    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        sparse_threshold=0.0,
    )

    # Keep the test set untouched until model selection is complete.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessing", preprocessing), ("model", estimator)])
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring={"f1": "f1", "recall": "recall", "precision": "precision"},
            n_jobs=-1,
            error_score="raise",
        )
        rows.append(
            {
                "model": name,
                "cv_f1_mean": float(scores["test_f1"].mean()),
                "cv_f1_std": float(scores["test_f1"].std()),
                "cv_recall_mean": float(scores["test_recall"].mean()),
                "cv_precision_mean": float(scores["test_precision"].mean()),
            }
        )
        fitted[name] = pipe

    results = pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False)
    best_name = str(results.iloc[0]["model"])
    best_pipeline = fitted[best_name]
    best_pipeline.fit(X_train, y_train)
    predictions = best_pipeline.predict(X_test)

    holdout = {
        "selected_model_by_training_cv_f1": best_name,
        "holdout_accuracy": float(accuracy_score(y_test, predictions)),
        "holdout_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "holdout_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "holdout_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "holdout_confusion_matrix_labels_0_then_1": confusion_matrix(y_test, predictions).tolist(),
        "holdout_classification_report": classification_report(
            y_test, predictions, target_names=["Not Dropped", "Dropped"], output_dict=True, zero_division=0
        ),
        "feature_policy": "Final_Grade excluded pending prediction-time justification",
        "caveat": "Exploratory academic baseline only; not suitable for individual high-impact decisions.",
    }
    if hasattr(best_pipeline, "predict_proba"):
        probabilities = best_pipeline.predict_proba(X_test)[:, 1]
        holdout["holdout_roc_auc"] = float(roc_auc_score(y_test, probabilities))

    results.to_csv(ROOT / "cv_model_comparison.csv", index=False)
    (ROOT / "holdout_evaluation.json").write_text(json.dumps(holdout, indent=2), encoding="utf-8")
    print("Cross-validation results (training partition only):")
    print(results.to_string(index=False))
    print("\nHeld-out test evaluation (used once after model selection):")
    print(json.dumps(holdout, indent=2))
    print("\nSaved cv_model_comparison.csv and holdout_evaluation.json")


if __name__ == "__main__":
    main()
