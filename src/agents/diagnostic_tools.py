
from src.tools.experiment_inspector import (
    inspect_experiment,
    compare_experiments
)

from src.tools.dataset_analyzer import (
    load_dataset,
    analyze_class_distribution,
    analyze_missing_values,
    analyze_feature_distributions,
    analyze_dataset_size
)


def inspect_performance(experiment_id):
    return inspect_experiment(experiment_id)


def compare_with_baseline(experiment_id):
    if experiment_id == "EXP-001":
        return {
            "message": "The baseline experiment cannot be compared against itself."
        }

    return compare_experiments("EXP-001", experiment_id)


def inspect_class_distribution(experiment_id):
    dataset = load_dataset(f"{experiment_id}_train.csv")
    return analyze_class_distribution(dataset)


def inspect_missing_values(experiment_id):
    dataset = load_dataset(f"{experiment_id}_train.csv")
    return analyze_missing_values(dataset)


def inspect_feature_drift(experiment_id):
    baseline = load_dataset("baseline_train.csv")
    target = load_dataset(f"{experiment_id}_train.csv")

    results = analyze_feature_distributions(baseline, target)

    drifted_features = [
        feature
        for feature, result in results.items()
        if result["drift_detected"]
    ]

    return {
        "drifted_features": drifted_features,
        "feature_results": results
    }


def inspect_dataset_size(experiment_id):
    baseline = load_dataset("baseline_train.csv")
    target = load_dataset(f"{experiment_id}_train.csv")

    return analyze_dataset_size(baseline, target)


DIAGNOSTIC_TOOLS = {
    "inspect_performance": inspect_performance,
    "compare_with_baseline": compare_with_baseline,
    "inspect_class_distribution": inspect_class_distribution,
    "inspect_missing_values": inspect_missing_values,
    "inspect_feature_drift": inspect_feature_drift,
    "inspect_dataset_size": inspect_dataset_size
}


def execute_diagnostic_tool(tool_name, experiment_id):
    if tool_name not in DIAGNOSTIC_TOOLS:
        raise ValueError(f"Unknown diagnostic tool: {tool_name}")

    return DIAGNOSTIC_TOOLS[tool_name](experiment_id)
