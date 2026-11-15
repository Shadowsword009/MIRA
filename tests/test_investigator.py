
import pytest

from src.agents.tool_selector import select_diagnostic_tools
from src.agents.diagnostic_tools import execute_diagnostic_tool
from src.agents.investigator import investigate_experiment


@pytest.mark.parametrize(
    "experiment_id, expected_status, expected_issue",
    [
        ("EXP-001", "HEALTHY", "No major issue identified by selected diagnostics"),
        ("EXP-002", "CRITICAL", "Single-class training data"),
        ("EXP-003", "NEEDS REVIEW", "Feature distribution drift"),
        ("EXP-004", "HIGH RISK", "Insufficient training data")
    ]
)
def test_experiment_investigation(
    experiment_id,
    expected_status,
    expected_issue
):
    selection = select_diagnostic_tools(experiment_id)

    diagnostic_results = {
        tool: execute_diagnostic_tool(tool, experiment_id)
        for tool in selection["selected_tools"]
    }

    report = investigate_experiment(
        experiment_id,
        diagnostic_results
    )

    assert report["status"] == expected_status

    assert any(
        finding["issue"] == expected_issue
        for finding in report["findings"]
    )

    assert report["performance"]["metrics"]

    assert report["findings"]

    assert all(
        finding["evidence"]
        for finding in report["findings"]
    )
