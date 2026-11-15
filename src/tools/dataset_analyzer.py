
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

EXPERIMENT_IDS = [
    "EXP-001",
    "EXP-002",
    "EXP-003",
    "EXP-004"
]


def load_dataset(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)



def analyze_class_distribution(df):
    distribution = df["churn"].value_counts().to_dict()
    total = len(df)

    minimum = min(distribution.values())
    maximum = max(distribution.values())

    return {
        "class_counts": {
            str(key): int(value)
            for key, value in distribution.items()
        },
        "class_percentages": {
            str(key): round(value / total * 100, 2)
            for key, value in distribution.items()
        },
        "imbalance_ratio": (
            round(maximum / minimum, 2)
            if minimum > 0 and len(distribution) > 1
            else None
        ),
        "single_class_detected": len(distribution) < 2
    }



def analyze_missing_values(df):
    missing = df.isnull().sum()

    return {
        column: {
            "missing_count": int(count),
            "missing_percentage": round(
                count / len(df) * 100, 2
            )
        }
        for column, count in missing.items()
        if count > 0
    }


def analyze_feature_distributions(reference_df, target_df):
    features = reference_df.select_dtypes(
        include=np.number
    ).columns.drop("churn", errors="ignore")

    results = {}

    for feature in features:
        reference = reference_df[feature].dropna()
        target = target_df[feature].dropna()

        if len(reference) == 0 or len(target) == 0:
            continue

        statistic, p_value = ks_2samp(reference, target)

        results[feature] = {
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 6),
            "drift_detected": bool(
                statistic >= 0.10 and p_value < 0.05
            )
        }

    return results


def analyze_dataset_size(reference_df, target_df):
    reference_size = len(reference_df)
    target_size = len(target_df)

    return {
        "reference_records": reference_size,
        "target_records": target_size,
        "records_difference": reference_size - target_size,
        "retained_percentage": round(
            target_size / reference_size * 100, 2
        )
    }


def analyze_experiment(experiment_id):
    baseline = load_dataset("baseline_train.csv")
    target = load_dataset(f"{experiment_id}_train.csv")

    feature_results = analyze_feature_distributions(
        baseline,
        target
    )

    drifted_features = [
        feature
        for feature, result in feature_results.items()
        if result["drift_detected"]
    ]

    return {
        "experiment_id": experiment_id,
        "training_records": len(target),
        "class_distribution": analyze_class_distribution(target),
        "missing_values": analyze_missing_values(target),
        "dataset_size": analyze_dataset_size(
            baseline,
            target
        ),
        "feature_drift": feature_results,
        "drifted_features": drifted_features
    }


def analyze_all_experiments():
    return {
        experiment_id: analyze_experiment(experiment_id)
        for experiment_id in EXPERIMENT_IDS
    }



if __name__ == "__main__":
    import json

    results = analyze_all_experiments()

    for experiment_id, result in results.items():
        print(f"\n{experiment_id}")
        print("-" * 40)
        print(f"Training records: {result['training_records']}")
        print(f"Class distribution: {result['class_distribution']}")
        print(f"Dataset retained: {result['dataset_size']['retained_percentage']}%")
        print(f"Missing values: {len(result['missing_values'])}")
        print(f"Drifted features: {result['drifted_features']}")

