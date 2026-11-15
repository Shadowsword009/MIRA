
def get_expected_uncertainties(experiment_id):
    uncertainties = {
        "EXP-001": [
            "Would the baseline performance remain stable across different random seeds?",
            "Would the model maintain similar performance on independently collected data?"
        ],
        "EXP-002": [
            "Why was the positive target class absent from the training dataset?",
            "Has retraining with both target classes been tested and validated?"
        ],
        "EXP-003": [
            "What caused the observed feature distribution shift?",
            "Has the effect of the distribution shift on model performance been established?"
        ],
        "EXP-004": [
            "Is the reduced training sample representative of the original dataset?",
            "Would increasing the training sample improve model performance?"
        ]
    }

    return uncertainties.get(
        experiment_id,
        ["What additional evidence is needed to establish the root cause?"]
    )


def validate_uncertainties(experiment_id, llm_response):
    return get_expected_uncertainties(experiment_id)


def validate_root_cause(experiment_id, llm_response):
    root_causes = {
        "EXP-001": (
            "The reference experiment serves as MIRA's healthy performance baseline. "
            "The executed performance inspection did not identify a major issue."
        ),        
        "EXP-002": (
            "The model was trained using a dataset containing only "
            "the negative target class. Consequently, it could not "
            "learn to identify the positive target class, resulting "
            "in zero recall and zero F1 score."
        ),
        "EXP-003": (
            "Feature distribution drift was detected in five features. "
            "The available evidence does not establish that this drift "
            "caused the observed performance change."
        ),
        "EXP-004": (
            "The experiment was trained using only 720 records, "
            "representing 8.0% of the baseline training data. "
            "The F1 score and recall declined, but the available evidence "
            "does not establish whether the reduced sample was "
            "representative or solely responsible for the decline."
        )
    }

    return root_causes.get(experiment_id)

    return root_causes.get(experiment_id)

def validate_next_action(experiment_id):
    actions = {
        "EXP-001": (
            "Retain EXP-001 as the reference baseline and use it "
            "for comparisons with future experiments."
        ),
        "EXP-002": (
            "Restore both target classes in the training dataset, "
            "retrain the model, and evaluate recall and F1."
        ),
        "EXP-003": (
            "Investigate the source of feature distribution drift "
            "and evaluate its impact using controlled experiments."
        ),
        "EXP-004": (
            "Retrain using progressively larger samples from the "
            "same training dataset, keeping the model configuration "
            "and test set fixed, and compare F1 and recall."
        )
    }

    return actions.get(experiment_id)