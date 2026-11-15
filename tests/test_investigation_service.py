
import pytest

from src.services.investigation_service import (
    list_experiments,
    run_investigation
)


def test_list_experiments():
    experiments = list_experiments()

    experiment_ids = {
        experiment["experiment_id"]
        for experiment in experiments
    }

    assert experiment_ids == {
        "EXP-001",
        "EXP-002",
        "EXP-003",
        "EXP-004"
    }


@pytest.mark.parametrize(
    "experiment_id",
    [None, "", "   ", "EXP-999"]
)
def test_invalid_experiment_id(experiment_id):
    with pytest.raises(ValueError):
        run_investigation(experiment_id)
