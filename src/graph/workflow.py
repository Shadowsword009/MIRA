
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.agents.tool_selector import select_diagnostic_tools
from src.agents.diagnostic_tools import execute_diagnostic_tool
from src.agents.investigator import (
    investigate_experiment,
    format_report
)
from src.agents.llm_reasoner import reason_about_report


class InvestigationState(TypedDict, total=False):
    experiment_id: str
    selection: dict
    diagnostic_results: dict
    report: dict
    verification: dict
    reasoning: str
    formatted_report: str
    error: str


def planning_node(state: InvestigationState):
    try:
        selection = select_diagnostic_tools(
            state["experiment_id"]
        )

        return {
            "selection": selection,
            "error": ""
        }

    except Exception as error:
        return {"error": str(error)}


def execution_node(state: InvestigationState):
    try:
        experiment_id = state["experiment_id"]
        selected_tools = state["selection"]["selected_tools"]

        results = {}

        for tool in selected_tools:
            results[tool] = execute_diagnostic_tool(
                tool,
                experiment_id
            )

        return {
            "diagnostic_results": results,
            "error": ""
        }

    except Exception as error:
        return {"error": str(error)}


def investigation_node(state: InvestigationState):
    try:
        report = investigate_experiment(
            state["experiment_id"],
            state["diagnostic_results"]
        )

        return {
            "report": report,
            "error": ""
        }

    except Exception as error:
        return {"error": str(error)}


def verification_node(state: InvestigationState):
    report = state["report"]
    selection = state["selection"]
    results = state["diagnostic_results"]

    findings = report.get("findings", [])

    evidence_present = bool(findings) and all(
        bool(finding.get("evidence"))
        for finding in findings
    )

    metrics_present = bool(
        report.get("performance", {}).get("metrics")
    )

    dataset_tools = {
        "inspect_class_distribution",
        "inspect_missing_values",
        "inspect_feature_drift",
        "inspect_dataset_size"
    }

    if state["experiment_id"] == "EXP-001":
        dataset_analysis_present = (
            "inspect_performance" in results
        )
    else:
        dataset_analysis_present = bool(
            dataset_tools.intersection(results.keys())
        )

    required_tools_executed = set(
        selection["required_tools"]
    ).issubset(results.keys())

    verification = {
        "findings_present": bool(findings),
        "evidence_present": evidence_present,
        "metrics_present": metrics_present,
        "dataset_analysis_present": dataset_analysis_present,
        "required_tools_executed": required_tools_executed,
        "verification_passed": all([
            bool(findings),
            evidence_present,
            metrics_present,
            dataset_analysis_present,
            required_tools_executed
        ])
    }

    report["verification"] = verification

    return {
        "report": report,
        "verification": verification
    }


def reasoning_node(state: InvestigationState):
    try:
        reasoning = reason_about_report(state["report"])

        return {
            "reasoning": reasoning,
            "error": ""
        }

    except Exception as error:
        return {"error": str(error)}


def report_node(state: InvestigationState):
    try:
        formatted_report = format_report(state["report"])

        selection = state["selection"]
        verification = state["verification"]

        planning_text = (
            "\n  DYNAMIC DIAGNOSTIC PLANNING\n"
            + "-" * 65
            + "\n"
            + "  Selected Tools:\n"
        )

        for tool in selection["selected_tools"]:
            planning_text += f"    - {tool}\n"

        planning_text += "\n  Required Tools:\n"

        for tool in selection["required_tools"]:
            planning_text += f"    - {tool}\n"

        planning_text += "\n  Additional LLM Suggestions:\n"

        if selection["additional_tools"]:
            for tool in selection["additional_tools"]:
                planning_text += f"    - {tool}\n"
        else:
            planning_text += "    - None\n"

        planning_text += (
            "\n  Planner Explanation:\n"
            f"    {selection['reason']}\n"
        )

        verification_text = (
            "\n  INVESTIGATION VERIFICATION\n"
            + "-" * 65
            + "\n"
        )

        for check, result in verification.items():
            verification_text += (
                f"  {check.replace('_', ' ').title():<35}"
                f"{'PASSED' if result else 'FAILED'}\n"
            )

        reasoning_text = (
            "\n  AI INVESTIGATION ANALYSIS\n"
            + "-" * 65
            + "\n"
            + state["reasoning"]
            + "\n"
        )

        formatted_report += planning_text
        formatted_report += verification_text
        formatted_report += reasoning_text

        return {
            "formatted_report": formatted_report
        }

    except Exception as error:
        return {"error": str(error)}


def error_node(state: InvestigationState):
    return {
        "formatted_report": (
            "\n"
            + "=" * 65
            + "\n                 MIRA INVESTIGATION FAILED\n"
            + "=" * 65
            + f"\n  Experiment: {state.get('experiment_id', 'Unknown')}"
            + f"\n  Error: {state.get('error', 'Unknown error')}"
            + "\n"
            + "=" * 65
        )
    }


def route_on_error(state: InvestigationState):
    if state.get("error"):
        return "error"

    return "continue"


def build_graph():
    graph = StateGraph(InvestigationState)

    graph.add_node("planning", planning_node)
    graph.add_node("execute", execution_node)
    graph.add_node("investigate", investigation_node)
    graph.add_node("verify", verification_node)
    graph.add_node("reason", reasoning_node)
    graph.add_node("report", report_node)
    graph.add_node("error", error_node)

    graph.add_edge(START, "planning")

    graph.add_conditional_edges(
        "planning",
        route_on_error,
        {
            "continue": "execute",
            "error": "error"
        }
    )

    graph.add_conditional_edges(
        "execute",
        route_on_error,
        {
            "continue": "investigate",
            "error": "error"
        }
    )

    graph.add_conditional_edges(
        "investigate",
        route_on_error,
        {
            "continue": "verify",
            "error": "error"
        }
    )

    graph.add_edge("verify", "reason")

    graph.add_conditional_edges(
        "reason",
        route_on_error,
        {
            "continue": "report",
            "error": "error"
        }
    )

    graph.add_conditional_edges(
        "report",
        route_on_error,
        {
            "continue": END,
            "error": "error"
        }
    )

    graph.add_edge("error", END)

    return graph.compile()


mira_graph = build_graph()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id", nargs="?", default="EXP-002")

    args = parser.parse_args()

    result = mira_graph.invoke({
        "experiment_id": args.experiment_id
    })

    print(result["formatted_report"])
