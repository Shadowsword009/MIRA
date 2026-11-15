
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULT_DIR = ROOT / "experiments" / "results"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
mlflow.set_experiment("MIRA_ML_Experiments")

X, y = make_classification(
    n_samples=12000,
    n_features=20,
    n_informative=12,
    n_redundant=4,
    n_classes=2,
    weights=[0.75, 0.25],
    flip_y=0.01,
    random_state=42
)

feature_names = [f"feature_{i}" for i in range(20)]

dataset = pd.DataFrame(X, columns=feature_names)
dataset["churn"] = y

dataset.to_csv(RAW_DIR / "baseline.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

baseline_train = pd.DataFrame(X_train, columns=feature_names)
baseline_train["churn"] = y_train
baseline_train.to_csv(PROCESSED_DIR / "baseline_train.csv", index=False)

experiments = {}

experiments["EXP-001"] = {
    "name": "Healthy Baseline",
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
}

experiments["EXP-002"] = {
    "name": "Class Imbalance",
    "X_train": X_train[y_train == 0],
    "X_test": X_test,
    "y_train": y_train[y_train == 0],
    "y_test": y_test
}

shifted_X = X_train.copy()
shifted_X[:, :5] += 3.5

experiments["EXP-003"] = {
    "name": "Feature Distribution Shift",
    "X_train": shifted_X,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
}

small_X, _, small_y, _ = train_test_split(
    X_train,
    y_train,
    train_size=0.08,
    random_state=42,
    stratify=y_train
)

experiments["EXP-004"] = {
    "name": "Limited Training Data",
    "X_train": small_X,
    "X_test": X_test,
    "y_train": small_y,
    "y_test": y_test
}

summary = {}

for experiment_id, experiment in experiments.items():

    training_data = pd.DataFrame(
        experiment["X_train"],
        columns=feature_names
    )
    training_data["churn"] = experiment["y_train"]

    training_data.to_csv(
        PROCESSED_DIR / f"{experiment_id}_train.csv",
        index=False
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ])

    with mlflow.start_run(run_name=experiment_id) as run:

        model.fit(
            experiment["X_train"],
            experiment["y_train"]
        )

        predictions = model.predict(experiment["X_test"])

        metrics = {
            "accuracy": float(accuracy_score(
                experiment["y_test"], predictions
            )),
            "precision": float(precision_score(
                experiment["y_test"], predictions,
                zero_division=0
            )),
            "recall": float(recall_score(
                experiment["y_test"], predictions,
                zero_division=0
            )),
            "f1_score": float(f1_score(
                experiment["y_test"], predictions,
                zero_division=0
            ))
        }

        mlflow.log_params({
            "experiment_id": experiment_id,
            "model": "RandomForestClassifier",
            "training_records": len(experiment["X_train"]),
            "testing_records": len(experiment["X_test"]),
            "n_estimators": 150
        })

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        result = {
            "experiment_id": experiment_id,
            "name": experiment["name"],
            "run_id": run.info.run_id,
            "training_records": len(experiment["X_train"]),
            "testing_records": len(experiment["X_test"]),
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(
                experiment["y_test"], predictions
            ).tolist()
        }

        summary[experiment_id] = result

        print(f"\n{experiment_id}: {experiment['name']}")
        print(f"Training records: {len(experiment['X_train'])}")
        print(f"Metrics: {metrics}")

with open(RESULT_DIR / "experiment_summary.json", "w") as file:
    json.dump(summary, file, indent=4)

print("\nAll experiments completed.")
print(f"Results saved to: {RESULT_DIR}")
print(f"Training datasets saved to: {PROCESSED_DIR}")
