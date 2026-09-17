"""STAGE 2 – Model Engineering.

Feature engineering (standardisation inside a sklearn Pipeline fitted on the
training data only), training of two candidate models, evaluation on the test
set, logging of params/metrics/models to MLflow, and packaging of the best
model into models/model.joblib (+ models/metrics.json with the test metrics).
"""
import json
import os                                                              # <-- ДОБАВЬТЕ
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")            
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # allows running both as a module and as a script
    sys.path.insert(0, str(ROOT))

from code.models.build_features import load_processed_data, separate_features_and_target

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

MLFLOW_TRACKING_DIR = ROOT / "mlruns"
MLFLOW_EXPERIMENT = "diabetes-classification"
RANDOM_STATE = 42
CANDIDATES = ("logistic_regression", "random_forest")


def build_model_pipeline(name: str) -> Pipeline:
    """Feature engineering (StandardScaler) + estimator in a single Pipeline."""
    if name == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def main() -> None:
    train_df, test_df = load_processed_data()
    X_train, y_train = separate_features_and_target(train_df)
    X_test, y_test = separate_features_and_target(test_df)
    print(f"[stage2] training on {len(X_train)} rows, evaluating on {len(X_test)} rows")

    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_DIR))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results, fitted = {}, {}
    for name in CANDIDATES:
        pipeline = build_model_pipeline(name)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = metrics

        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                "model": name,
                "feature_engineering": "StandardScaler",
                "n_train_rows": len(X_train),
                "n_test_rows": len(X_test),
                "random_state": RANDOM_STATE,
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                input_example=X_test.head(1),
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

        print(f"[stage2] {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_pipeline = fitted[best_name]

    # Package the model (atomic replace → the running API can hot-reload safely)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(".tmp")
    joblib.dump(best_pipeline, tmp_path)
    tmp_path.replace(MODEL_PATH)
    print(f"[stage2] best model '{best_name}' saved to {MODEL_PATH}")

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_model": best_name,
        "test_metrics": results[best_name],
        "all_candidates": results,
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(f"[stage2] testing metrics written to {METRICS_PATH}")


if __name__ == "__main__":
    main()