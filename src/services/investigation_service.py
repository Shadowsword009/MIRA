
from src.graph.workflow import mira_graph
from src.tools.experiment_inspector import load_experiments


class InvestigationError(Exception):
    pass


def list_experiments():
    try:
        experiments = load_experiments()
    except Exception as error:
        raise InvestigationError(
            f"Unable to load experiments: {error}"
        ) from error

    return [
        {
            "experiment_id": experiment_id,
            "name": experiment["name"],
            "training_records": experiment["training_records"],
            "testing_records": experiment["testing_records"],
            "metrics": experiment["metrics"]
        }
        for experiment_id, experiment in experiments.items()
    ]


def run_investigation(experiment_id, progress_callback=None):
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("A valid experiment ID is required.")

    try:
        experiments = load_experiments()
    except Exception as error:
        raise InvestigationError(
            f"Unable to load experiment data: {error}"
        ) from error

    if experiment_id not in experiments:
        raise ValueError(
            f"Experiment '{experiment_id}' was not found."
        )

    result = {
        "experiment_id": experiment_id
    }

    try:
        for update in mira_graph.stream(
            {"experiment_id": experiment_id},
            stream_mode="updates"
        ):
            for node_name, node_result in update.items():
                result.update(node_result)

                if progress_callback:
                    progress_callback(node_name)

    except Exception as error:
        raise InvestigationError(
            f"Investigation execution failed: {error}"
        ) from error

    if result.get("error"):
        raise InvestigationError(result["error"])

    report = result.get("report")

    if not report:
        raise InvestigationError(
            "The investigation did not produce a report."
        )

    return {
        "experiment_id": experiment_id,
        "experiment_name": experiments[experiment_id]["name"],
        "status": report["status"],
        "performance": report["performance"],
        "dataset_summary": report["dataset_summary"],
        "findings": report["findings"],
        "recommendations": report["recommendations"],
        "selection": result["selection"],
        "diagnostic_results": result["diagnostic_results"],
        "verification": result["verification"],
        "reasoning": result["reasoning"],
        "formatted_report": result["formatted_report"]
    }
