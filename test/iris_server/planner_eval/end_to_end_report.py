"""Turn end-to-end results into the summary a person reads."""

import eval_metrics
import eval_report
import end_to_end_metrics as stage_metrics
from eval_report import percent


def summary_lines(results: list, classifier_model, planner_model) -> list:
    return (_model_lines(classifier_model, planner_model)
            + eval_report.summary_lines(results, None)
            + _stage_lines(results)
            + _false_action_lines(results)
            + _latency_lines(results))


def _model_lines(classifier_model, planner_model) -> list:
    if not classifier_model:
        return []
    return [f"classifier model: {classifier_model}", f"planner model: {planner_model}", ""]


def _listed(ids: list) -> str:
    return ", ".join(ids) if ids else "none"


def _stage_lines(results) -> list:
    missed = stage_metrics.classifier_missed_ids(results)
    flagged = stage_metrics.classifier_wrongly_flagged_ids(results)
    vetoed = stage_metrics.planner_vetoed_ids(results)
    confirming = stage_metrics.asked_to_confirm_ids(results)
    failed = stage_metrics.planner_failed_ids(results)
    return [
        "classifier stage (end-to-end run):",
        f"  physical cases the classifier missed: "
        f"{percent(len(missed), stage_metrics.physical_case_count(results))}",
        f"    ids: {_listed(missed)}",
        f"  chat cases the classifier wrongly flagged: "
        f"{percent(len(flagged), eval_metrics.chat_case_count(results))}",
        f"    ids: {_listed(flagged)}",
        f"  of those, vetoed by the planner (no steps requested): "
        f"{percent(len(vetoed), len(flagged))}",
        f"    ids: {_listed(vetoed)}",
        f"  turns where the classifier asked 'did you mean?' instead: {len(confirming)}",
        f"    ids: {_listed(confirming)}",
        f"  turns where the planner failed and the classifier's state was kept: {len(failed)}",
        f"    ids: {_listed(failed)}",
        "",
    ]


def _false_action_lines(results) -> list:
    wrong_ids = eval_metrics.false_action_ids(results)
    rate = percent(len(wrong_ids), eval_metrics.chat_case_count(results))
    return [f"END-TO-END false-action rate (chat turned into queue or unsupported): {rate}",
            f"  ids: {_listed(wrong_ids)}", ""]


def _latency_lines(results) -> list:
    lines = ["latency per turn, classifier plus planner (seconds):"]
    for group, seconds in stage_metrics.latency_groups(results).items():
        lines.append(f"  turns {group}: {len(seconds)}")
        if seconds:
            lines += [f"    {name:<8}{value:.3f}"
                      for name, value in eval_metrics.latency_summary(seconds).items()]
    return lines + [""]
