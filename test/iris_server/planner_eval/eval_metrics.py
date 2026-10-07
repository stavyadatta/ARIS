"""Pure scoring of planner results: no model, no I/O.

A result is a dict with `id`, `category`, `expected` and `got` (each
{"kind", "actions"}), `correct` and `seconds`.
"""

import math
import statistics
from collections import Counter

from eval_cases import KIND_CHAT, KIND_QUEUE, KIND_UNSUPPORTED

ERROR_WRONG_ORDER = "right actions, wrong order"
ERROR_MISSING_STEP = "missing step"
ERROR_EXTRA_STEP = "extra step"
ERROR_OTHER = "other wrong actions"

# Kinds that make the robot move or refuse out loud: for a chat sentence they
# are the dangerous mistake.
KINDS_THAT_ACT = (KIND_QUEUE, KIND_UNSUPPORTED)

SLOW_TAIL_FRACTION = 0.95


def is_correct(expected: dict, got: dict) -> bool:
    """Exact match: the same kind and, for a queue, the same ordered actions."""
    return expected["kind"] == got["kind"] and expected["actions"] == got["actions"]


def correct_count(results) -> int:
    return sum(result["correct"] for result in results)


def accuracy_by_category(results) -> dict:
    """category -> (correct, total), in the order categories first appear."""
    tally = {}
    for result in results:
        correct, total = tally.get(result["category"], (0, 0))
        tally[result["category"]] = (correct + result["correct"], total + 1)
    return tally


def confusion_counts(results) -> Counter:
    """(expected kind, got kind) -> how many cases."""
    return Counter(
        (result["expected"]["kind"], result["got"]["kind"]) for result in results
    )


def classify_queue_error(expected_actions: list, got_actions: list) -> str:
    wanted, produced = Counter(expected_actions), Counter(got_actions)
    if wanted == produced:
        return ERROR_WRONG_ORDER
    if not produced - wanted:
        return ERROR_MISSING_STEP
    if not wanted - produced:
        return ERROR_EXTRA_STEP
    return ERROR_OTHER


def queue_error_split(results) -> Counter:
    """How the wrong queues were wrong, among queue cases answered with a queue."""
    return Counter(
        classify_queue_error(result["expected"]["actions"], result["got"]["actions"])
        for result in results
        if _is_wrong_queue(result)
    )


def _is_wrong_queue(result) -> bool:
    return (
        not result["correct"]
        and result["expected"]["kind"] == KIND_QUEUE
        and result["got"]["kind"] == KIND_QUEUE
    )


def false_action_ids(results) -> list:
    """Ids of chat cases the planner turned into a queue or a refusal."""
    return [
        result["id"] for result in results
        if result["expected"]["kind"] == KIND_CHAT
        and result["got"]["kind"] in KINDS_THAT_ACT
    ]


def chat_case_count(results) -> int:
    return sum(result["expected"]["kind"] == KIND_CHAT for result in results)


def failed_results(results) -> list:
    return [result for result in results if not result["correct"]]


def latency_summary(seconds) -> dict:
    ordered = sorted(seconds)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": _nearest_rank(ordered, SLOW_TAIL_FRACTION),
        "max": ordered[-1],
    }


def _nearest_rank(ordered_values, fraction):
    rank = math.ceil(fraction * len(ordered_values))
    return ordered_values[max(rank, 1) - 1]
