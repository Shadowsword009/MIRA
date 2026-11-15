
def investigate_experiment(experiment_id, diagnostic_results):
    performance_result = diagnostic_results.get("inspect_performance")
    comparison_result = diagnostic_results.get("compare_with_baseline")

    if not performance_result:
        raise ValueError("Performance inspection is required.")

    metrics = performance_result["metrics"]

    if experiment_id == "EXP-001":
        metric_comparison = {
            metric: {
                "reference": metrics[metric],
                "difference": 0.0
            }
            for metric in metrics
        }

        f1_difference = 0.0
        recall_difference = 0.0

    else:
        if not comparison_result:
            raise ValueError("Baseline comparison is required.")

        metric_comparison = comparison_result["comparison"]

        f1_difference = metric_comparison["f1_score"]["difference"]
        recall_difference = metric_comparison["recall"]["difference"]

    class_distribution = diagnostic_results.get(
        "inspect_class_distribution"
    )

    missing_values = diagnostic_results.get(
        "inspect_missing_values"
    )

    feature_analysis = diagnostic_results.get(
        "inspect_feature_drift"
    )

    dataset_size = diagnostic_results.get(
        "inspect_dataset_size"
    )

    findings = []
    recommendations = []

    if class_distribution:
        if class_distribution["single_class_detected"]:
            findings.append({
                "issue": "Single-class training data",
                "severity": "CRITICAL",
                "confidence": 0.99,
                "evidence": [
                    "Only one target class exists in the training dataset.",
                    f"Class distribution: {class_distribution['class_counts']}",
                    f"Model recall: {metrics['recall']}",
                    f"Model F1 score: {metrics['f1_score']}"
                ],
                "explanation": (
                    "The model was trained without examples of every target "
                    "class. It cannot properly learn to distinguish the missing class."
                )
            })

            recommendations.append(
                "Restore missing target classes in the training dataset."
            )

        elif (
            class_distribution["imbalance_ratio"] is not None
            and class_distribution["imbalance_ratio"] >= 4
        ):
            findings.append({
                "issue": "Severe class imbalance",
                "severity": "HIGH",
                "confidence": 0.90,
                "evidence": [
                    f"Class imbalance ratio: {class_distribution['imbalance_ratio']}",
                    f"Model recall: {metrics['recall']}"
                ],
                "explanation": (
                    "The training dataset contains substantially more "
                    "examples of one target class."
                )
            })

            recommendations.append(
                "Evaluate class weighting, resampling and stratified sampling."
            )

    if feature_analysis:
        drifted_features = feature_analysis["drifted_features"]

        if drifted_features:
            drift_count = len(drifted_features)

            impact_status = (
                "Performance degradation observed."
                if f1_difference < -0.05
                else "Significant performance degradation was not established."
            )

            findings.append({
                "issue": "Feature distribution drift",
                "severity": "HIGH" if f1_difference < -0.05 else "MEDIUM",
                "confidence": 0.95,
                "evidence": [
                    f"{drift_count} features show distribution drift.",
                    f"Drifted features: {drifted_features}",
                    f"F1 score difference from baseline: {f1_difference}",
                    impact_status
                ],
                "explanation": (
                    "The experiment's feature distributions differ from "
                    "the baseline training dataset. This may affect how well "
                    "the model generalizes."
                )
            })

            recommendations.append(
                "Investigate the source of feature drift and evaluate "
                "training on representative data."
            )

    if dataset_size:
        if dataset_size["retained_percentage"] < 20:
            findings.append({
                "issue": "Insufficient training data",
                "severity": "HIGH",
                "confidence": 0.90,
                "evidence": [
                    f"Only {dataset_size['retained_percentage']}% of baseline "
                    "training records were retained.",
                    f"Training records: {dataset_size['target_records']}",
                    f"F1 score difference: {f1_difference}",
                    f"Recall difference: {recall_difference}"
                ],
                "explanation": (
                    "The experiment uses a substantially smaller training "
                    "dataset, which may reduce the model's ability to generalize."
                )
            })

            recommendations.append(
                "Increase the training dataset size and compare performance "
                "across multiple sample sizes."
            )

    if missing_values:
        findings.append({
            "issue": "Missing values detected",
            "severity": "MEDIUM",
            "confidence": 0.95,
            "evidence": [
                f"Columns with missing values: {list(missing_values.keys())}"
            ],
            "explanation": (
                "Missing values can affect preprocessing and model training."
            )
        })

        recommendations.append(
            "Investigate missing-value patterns and apply an appropriate "
            "imputation or removal strategy."
        )

    if not findings:
        findings.append({
            "issue": "No major issue identified by selected diagnostics",
            "severity": "LOW",
            "confidence": 0.90 if experiment_id == "EXP-001" else 0.75,
            "evidence": [
                "The executed diagnostic tools did not identify a major issue.",
                f"F1 score: {metrics['f1_score']}",
                f"Diagnostics executed: {list(diagnostic_results.keys())}"
            ],
            "explanation": (
                "The selected diagnostic checks did not identify a major "
                "issue. Unselected diagnostic areas have not been evaluated."
            )
        })

    critical_count = sum(
        finding["severity"] == "CRITICAL"
        for finding in findings
    )

    high_count = sum(
        finding["severity"] == "HIGH"
        for finding in findings
    )

    if critical_count:
        status = "CRITICAL"
    elif high_count:
        status = "HIGH RISK"
    elif findings[0]["severity"] == "LOW":
        status = "HEALTHY"
    else:
        status = "NEEDS REVIEW"

    return {
        "experiment_id": experiment_id,
        "status": status,
        "performance": {
            "metrics": metrics,
            "baseline_comparison": metric_comparison
        },
        "dataset_summary": {
            "training_records": performance_result["training_records"],
            "class_distribution": class_distribution,
            "retained_percentage": (
                dataset_size["retained_percentage"]
                if dataset_size else None
            ),
            "drifted_features": (
                feature_analysis["drifted_features"]
                if feature_analysis else None
            ),
            "missing_values": missing_values,
            "diagnostics_run": list(diagnostic_results.keys())
        },
        "findings": findings,
        "recommendations": recommendations
    }



def format_report(report):
    line = "=" * 65
    thin_line = "-" * 65

    status_symbols = {
        "CRITICAL": "[!!!]",
        "HIGH RISK": "[HIGH]",
        "NEEDS REVIEW": "[WARN]",
        "HEALTHY": "[ OK ]"
    }

    output = [
        "",
        line,
        "                 MIRA INVESTIGATION REPORT",
        line,
        f"  Experiment      : {report['experiment_id']}",
        f"  Overall Status  : {status_symbols.get(report['status'], '[INFO]')} {report['status']}",
        line,
        "",
        "  PERFORMANCE OVERVIEW",
        thin_line
    ]

    metrics = report["performance"]["metrics"]
    comparison = report["performance"]["baseline_comparison"]

    output.append(
        f"  {'Metric':<15} {'Current':>12} {'Baseline':>12} {'Change':>12}"
    )

    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        current = metrics[metric]
        baseline = comparison[metric]["reference"]
        difference = comparison[metric]["difference"]

        output.append(
            f"  {metric:<15} {current:>11.2%} "
            f"{baseline:>11.2%} {difference:>+11.2%}"
        )

    dataset = report["dataset_summary"]
    class_distribution = dataset["class_distribution"]

    output.extend([
        "",
        "  DATASET INTELLIGENCE",
        thin_line,
        f"  Training Records     : {dataset['training_records']:,}",
        f"  Data Retained        : {dataset['retained_percentage'] if dataset['retained_percentage'] is not None else 'Not inspected'}",
        f"  Class Distribution   : {class_distribution['class_percentages'] if class_distribution else 'Not inspected'}",
        f"  Single Class         : {class_distribution['single_class_detected'] if class_distribution else 'Not inspected'}",
        f"  Missing Value Columns: {len(dataset['missing_values']) if dataset['missing_values'] is not None else 'Not inspected'}",
        f"  Feature Drift        : {len(dataset['drifted_features']) if dataset['drifted_features'] is not None else 'Not inspected'}",
        "",
        "  DIAGNOSTICS EXECUTED",
        thin_line
    ])

    for tool in dataset["diagnostics_run"]:
        output.append(f"  - {tool}")

    if dataset["drifted_features"]:
        output.append(
            f"  Affected Features    : {', '.join(dataset['drifted_features'])}"
        )

    output.extend([
        "",
        "  INVESTIGATION FINDINGS",
        thin_line
    ])

    for index, finding in enumerate(report["findings"], start=1):
        output.extend([
            "",
            f"  {index}. {finding['issue']} [{finding['severity']}]",
            f"     Confidence Score: {finding['confidence']:.0%}",
            f"     {finding['explanation']}",
            "     Supporting Evidence:"
        ])

        for evidence in finding["evidence"]:
            output.append(f"       - {evidence}")

    output.extend([
        "",
        "  RECOMMENDED ACTIONS",
        thin_line
    ])

    for index, recommendation in enumerate(
        report["recommendations"], start=1
    ):
        output.append(f"  {index}. {recommendation}")

    output.extend([
        "",
        line,
        "                  END OF REPORT",
        line,
        ""
    ])

    return "\n".join(output)


if __name__ == "__main__":
    import argparse

    from src.agents.tool_selector import select_diagnostic_tools
    from src.agents.diagnostic_tools import execute_diagnostic_tool

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id", nargs="?", default="EXP-002")

    args = parser.parse_args()

    selection = select_diagnostic_tools(args.experiment_id)

    results = {
        tool: execute_diagnostic_tool(tool, args.experiment_id)
        for tool in selection["selected_tools"]
    }

    report = investigate_experiment(args.experiment_id, results)

    print(format_report(report))
