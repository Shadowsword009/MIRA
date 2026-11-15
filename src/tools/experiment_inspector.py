
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_FILE = ROOT / "experiments" / "results" / "experiment_summary.json"


def load_experiments():
    with open(RESULT_FILE, "r") as file:
        return json.load(file)


def inspect_experiment(experiment_id):
    experiments = load_experiments()

    if experiment_id not in experiments:
        raise ValueError(f"Experiment {experiment_id} not found")

    experiment = experiments[experiment_id]
    metrics = experiment["metrics"]

    issues = []

    if metrics["f1_score"] < 0.60:
        issues.append("Very low F1 score")

    if metrics["recall"] < 0.50:
        issues.append("Poor recall")

    if metrics["precision"] < 0.50:
        issues.append("Poor precision")

    if metrics["accuracy"] - metrics["recall"] > 0.30:
        issues.append("Possible class imbalance or prediction bias")

    return {
        "experiment_id": experiment_id,
        "name": experiment["name"],
        "metrics": metrics,
        "training_records": experiment["training_records"],
        "testing_records": experiment["testing_records"],
        "confusion_matrix": experiment["confusion_matrix"],
        "detected_issues": issues,
        "status": "SUSPICIOUS" if issues else "HEALTHY"
    }


def compare_experiments(reference_id, target_id):
    experiments = load_experiments()

    if reference_id not in experiments:
        raise ValueError(f"Reference experiment {reference_id} not found")

    if target_id not in experiments:
        raise ValueError(f"Target experiment {target_id} not found")

    reference = experiments[reference_id]
    target = experiments[target_id]

    comparison = {}

    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        reference_value = reference["metrics"][metric]
        target_value = target["metrics"][metric]

        comparison[metric] = {
            "reference": reference_value,
            "target": target_value,
            "difference": round(
                target_value - reference_value, 4
            )
        }

    return {
        "reference_experiment": reference_id,
        "target_experiment": target_id,
        "comparison": comparison
    }


def find_suspicious_experiments():
    experiments = load_experiments()

    results = []

    for experiment_id in experiments:
        result = inspect_experiment(experiment_id)

        if result["status"] == "SUSPICIOUS":
            results.append(result)

    return results


if __name__ == "__main__":
    for experiment in find_suspicious_experiments():
        print(json.dumps(experiment, indent=4))
