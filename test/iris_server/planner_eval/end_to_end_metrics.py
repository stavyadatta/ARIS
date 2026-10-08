"""Pure scoring of the classifier stage and the planner veto: no model, no I/O.

An end-to-end result is a planner-stage result (see eval_metrics.py) plus the
stage facts from end_to_end_adapter.stage_facts.
"""

from eval_cases import KIND_CHAT, KIND_QUEUE, KIND_UNSUPPORTED

PHYSICAL_KINDS = (KIND_QUEUE, KIND_UNSUPPORTED)


def _ids(results, keep) -> list:
    return [result["id"] for result in results if keep(result)]


def physical_case_count(results) -> int:
    return sum(result["expected"]["kind"] in PHYSICAL_KINDS for result in results)


def classifier_missed_ids(results) -> list:
    """Physical cases the classifier did not flag as physical."""
    return _ids(results, lambda result: result["expected"]["kind"] in PHYSICAL_KINDS
                and not result["classifier_flagged_physical"])


def classifier_wrongly_flagged_ids(results) -> list:
    """Chat cases the classifier flagged as physical."""
    return _ids(results, lambda result: result["expected"]["kind"] == KIND_CHAT
                and result["classifier_flagged_physical"])


def planner_vetoed_ids(results) -> list:
    """Wrongly flagged chat cases the planner then turned back into conversation."""
    return _ids(results, lambda result: result["expected"]["kind"] == KIND_CHAT
                and result["classifier_flagged_physical"] and result["planner_vetoed"])


def asked_to_confirm_ids(results) -> list:
    return _ids(results, lambda result: result["asked_to_confirm"])


def planner_failed_ids(results) -> list:
    return _ids(results, lambda result: result["planner_failed"])


def latency_groups(results) -> dict:
    """Seconds per turn, split by whether the planner was asked."""
    return {
        "without the planner": [r["seconds"] for r in results if not r["planner_asked"]],
        "through the planner": [r["seconds"] for r in results if r["planner_asked"]],
    }
