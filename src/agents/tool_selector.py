
import json

from src.tools.experiment_inspector import load_experiments
from src.agents.diagnostic_tools import DIAGNOSTIC_TOOLS


def get_experiment_context(experiment_id):
    experiments = load_experiments()

    if experiment_id not in experiments:
        raise ValueError(f"Experiment {experiment_id} not found")

    experiment = experiments[experiment_id]
    baseline = experiments["EXP-001"]

    metrics = experiment["metrics"]
    baseline_metrics = baseline["metrics"]

    metric_changes = {
        metric: round(
            metrics[metric] - baseline_metrics[metric], 4
        )
        for metric in metrics
    }

    retained_percentage = (
        experiment["training_records"]
        / baseline["training_records"]
    ) * 100

    return {
        "experiment_id": experiment_id,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "metric_changes": metric_changes,
        "training_records": experiment["training_records"],
        "baseline_training_records": baseline["training_records"],
        "retained_percentage": round(retained_percentage, 2)
    }


def get_required_tools(context):
    required = ["inspect_performance"]

    experiment_id = context["experiment_id"]
    metrics = context["metrics"]
    changes = context["metric_changes"]
    retained_percentage = context["retained_percentage"]

    if experiment_id != "EXP-001":
        required.append("compare_with_baseline")

    if (
        metrics["recall"] < 0.50
        or metrics["precision"] < 0.50
        or metrics["accuracy"] - metrics["recall"] > 0.30
    ):
        required.append("inspect_class_distribution")

    if retained_percentage < 50:
        required.append("inspect_dataset_size")

    severe_class_problem = (
        metrics["recall"] < 0.50
        or metrics["precision"] < 0.50
    )

    major_data_reduction = retained_percentage < 50

    if (
        changes["f1_score"] < -0.01
        and not severe_class_problem
        and not major_data_reduction
    ):
        required.append("inspect_feature_drift")

    return required


def select_diagnostic_tools(experiment_id):
    context = get_experiment_context(experiment_id)

    required = get_required_tools(context)

    tool_order = list(DIAGNOSTIC_TOOLS)

    selected = sorted(
        required,
        key=tool_order.index
    )

    return {
        "selected_tools": selected,
        "required_tools": required,
        "additional_tools": [],
        "reason": (
            "Diagnostic tools selected using deterministic "
            "rules based on experiment metrics and dataset size."
        )
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id", nargs="?", default="EXP-002")

    args = parser.parse_args()

    result = select_diagnostic_tools(args.experiment_id)

    print(json.dumps(result, indent=4))
