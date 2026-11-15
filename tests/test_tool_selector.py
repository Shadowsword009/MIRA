
import pytest

from src.agents.tool_selector import get_required_tools


@pytest.mark.parametrize(
    "context, expected",
    [
        (
            {
                "experiment_id": "EXP-001",
                "metrics": {
                    "accuracy": 0.9657,
                    "precision": 0.9644,
                    "recall": 0.8968,
                    "f1_score": 0.9294
                },
                "metric_changes": {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0
                },
                "retained_percentage": 100.0
            },
            ["inspect_performance"]
        ),
        (
            {
                "experiment_id": "EXP-002",
                "metrics": {
                    "accuracy": 0.748,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0
                },
                "metric_changes": {
                    "accuracy": -0.2177,
                    "precision": -0.9644,
                    "recall": -0.8968,
                    "f1_score": -0.9294
                },
                "retained_percentage": 74.79
            },
            [
                "inspect_performance",
                "compare_with_baseline",
                "inspect_class_distribution"
            ]
        ),
        (
            {
                "experiment_id": "EXP-003",
                "metrics": {
                    "accuracy": 0.9557,
                    "precision": 0.9285,
                    "recall": 0.8929,
                    "f1_score": 0.9103
                },
                "metric_changes": {
                    "accuracy": -0.0100,
                    "precision": -0.0360,
                    "recall": -0.0040,
                    "f1_score": -0.0191
                },
                "retained_percentage": 100.0
            },
            [
                "inspect_performance",
                "compare_with_baseline",
                "inspect_feature_drift"
            ]
        ),
        (
            {
                "experiment_id": "EXP-004",
                "metrics": {
                    "accuracy": 0.919,
                    "precision": 0.9605,
                    "recall": 0.7077,
                    "f1_score": 0.8149
                },
                "metric_changes": {
                    "accuracy": -0.0467,
                    "precision": -0.0039,
                    "recall": -0.1892,
                    "f1_score": -0.1145
                },
                "retained_percentage": 8.0
            },
            [
                "inspect_performance",
                "compare_with_baseline",
                "inspect_dataset_size"
            ]
        )
    ]
)
def test_diagnostic_tool_selection(context, expected):
    assert get_required_tools(context) == expected
