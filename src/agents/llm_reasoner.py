
import json
import re

from src.agents.llm_client import get_llm

def flatten_evidence(value, prefix=""):
    facts = []

    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix} {key}".strip()
            facts.extend(flatten_evidence(item, path))

    elif isinstance(value, list):
        for item in value:
            facts.extend(flatten_evidence(item, prefix))

    elif value is not None:
        label = prefix.replace("_", " ").strip()
        facts.append(f"{label}: {value}")

    return facts


def build_evidence_points(report):
    candidates = []

    findings = report.get("findings", [])

    for finding in findings:
        if isinstance(finding, dict):
            finding_evidence = finding.get("evidence")

            if finding_evidence is not None:
                candidates.extend(
                    flatten_evidence(
                        finding_evidence,
                        "Finding evidence"
                    )
                )

    dataset_summary = report.get("dataset_summary", {})

    summary_fields = [
        "training_records",
        "class_distribution",
        "retained_percentage",
        "drifted_features",
        "missing_values"
    ]

    for field in summary_fields:
        value = dataset_summary.get(field)

        if value is not None:
            if field == "retained_percentage":
                candidates.append(
                    f"Training data retained: {value}% of baseline"
                )
            elif field == "drifted_features":
                candidates.append(
                    f"Drifted features: {value}"
                )
            elif field == "class_distribution":
                candidates.append(
                    f"Class distribution: {value}"
                )
            else:
                candidates.append(
                    f"{field.replace('_', ' ').capitalize()}: {value}"
                )

    performance = report.get("performance", {})

    performance_facts = flatten_evidence(
        performance,
        "Performance"
    )

    for fact in performance_facts:
        lowered = fact.lower()

        if any(
            metric in lowered
            for metric in (
                "accuracy",
                "precision",
                "recall",
                "f1 score",
                "f1_score"
            )
        ):
            candidates.append(fact)

    unique_candidates = list(dict.fromkeys(candidates))

    if len(unique_candidates) < 3:
        raise ValueError(
            "The investigation report does not contain "
            "three distinct evidence facts."
        )

    return unique_candidates[:3]


def extract_section(content, heading):
    pattern = (
        rf"(?ims)^\s*{re.escape(heading)}\s*:?\s*\n"
        rf"(.*?)(?=^\s*(?:ROOT CAUSE|EVIDENCE|"
        rf"UNCERTAINTIES|NEXT ACTION)\s*:?\s*$|\Z)"
    )

    match = re.search(pattern, content)

    if match:
        return match.group(1).strip()

    return ""


def reason_about_report(report):
    evidence = {
        "experiment_id": report["experiment_id"],
        "status": report["status"],
        "performance": report["performance"],
        "dataset_summary": report["dataset_summary"],
        "findings": report["findings"],
        "recommendations": report["recommendations"]
    }

    prompt = f"""
You are MIRA, an ML investigation assistant.

Analyze only the verified evidence supplied below.

ROOT CAUSE:
Write exactly 2 sentences describing the primary detected issue.

Do not invent metrics, dataset properties, or causal relationships.
Distinguish observed findings from possible explanations.
Use only supplied evidence.
Return plain text.

VERIFIED EXPERIMENT DATA:
{json.dumps(evidence, indent=2)}
"""

    from src.agents.reasoning_validator import (
        validate_uncertainties,
        validate_root_cause,
        validate_next_action
    )

    response = get_llm().invoke(prompt)
    content = response.content

    root_cause = validate_root_cause(
        report["experiment_id"],
        content
    )

    if not root_cause:
        root_cause = extract_section(
            content,
            "ROOT CAUSE"
        )

    if not root_cause:
        root_cause = (
            "The available investigation evidence identifies "
            "an issue requiring further examination."
        )

    evidence_points = build_evidence_points(report)

    validated_uncertainties = validate_uncertainties(
        report["experiment_id"],
        content
    )

    validated_action = validate_next_action(
        report["experiment_id"]
    )

    if not validated_action:
        validated_action = (
            "Collect additional evidence and run a controlled "
            "experiment to investigate the identified issue."
        )

    evidence_text = "\n".join(
        f"{index}. {point}"
        for index, point in enumerate(evidence_points, 1)
    )

    uncertainty_text = "\n".join(
        f"{index}. {item}"
        for index, item in enumerate(
            validated_uncertainties, 1
        )
    )

    return (
        "ROOT CAUSE\n"
        + root_cause
        + "\n\nEVIDENCE\n"
        + evidence_text
        + "\n\nUNCERTAINTIES\n"
        + uncertainty_text
        + "\n\nNEXT ACTION\n"
        + validated_action
    )


if __name__ == "__main__":
    from src.agents.tool_selector import select_diagnostic_tools
    from src.agents.diagnostic_tools import execute_diagnostic_tool
    from src.agents.investigator import investigate_experiment

    experiment_id = "EXP-003"

    selection = select_diagnostic_tools(experiment_id)

    diagnostic_results = {
        tool: execute_diagnostic_tool(tool, experiment_id)
        for tool in selection["selected_tools"]
    }

    report = investigate_experiment(
        experiment_id,
        diagnostic_results
    )

    print(reason_about_report(report))
