"""Load the labelled planner cases and check their shape.

A mislabelled or malformed case would silently skew every metric, so the
whole file is validated before any model call is paid for.
"""

import json

KIND_QUEUE = "queue"
KIND_UNSUPPORTED = "unsupported"
KIND_CHAT = "chat"
KIND_COULD_NOT_PLAN = "could_not_plan"

# What a case may expect. could_not_plan is only ever something the planner
# produces, never something a person is labelled to deserve.
EXPECTED_KINDS = (KIND_QUEUE, KIND_UNSUPPORTED, KIND_CHAT)

REQUIRED_CASE_FIELDS = ("id", "category", "sentence", "expected")


def load_cases(path) -> list:
    with open(path, encoding="utf-8") as cases_file:
        return json.load(cases_file)


def problems_in_cases(cases, allowed_actions) -> list:
    """Every way `cases` is malformed, as readable lines; empty when sound."""
    if not isinstance(cases, list) or not cases:
        return ["cases must be a non-empty JSON list"]
    problems = []
    seen_ids = set()
    for position, case in enumerate(cases):
        problems += _problems_in_case(case, position, allowed_actions)
        problems += _problems_in_id(case, seen_ids)
    return problems


def _problems_in_id(case, seen_ids) -> list:
    case_id = case.get("id") if isinstance(case, dict) else None
    if case_id in seen_ids:
        return [f"duplicate id {case_id}"]
    seen_ids.add(case_id)
    return []


def _problems_in_case(case, position, allowed_actions) -> list:
    if not isinstance(case, dict):
        return [f"case #{position} is not an object"]
    missing = [field for field in REQUIRED_CASE_FIELDS if field not in case]
    if missing:
        return [f"case #{position} lacks {missing}"]
    label = case["id"]
    problems = []
    if not _is_text(case["sentence"]):
        problems.append(f"{label}: sentence must be non-empty text")
    problems += _problems_in_expected(label, case["expected"], allowed_actions)
    return problems


def _problems_in_expected(label, expected, allowed_actions) -> list:
    if not isinstance(expected, dict):
        return [f"{label}: expected must be an object"]
    kind = expected.get("kind")
    actions = expected.get("actions")
    if kind not in EXPECTED_KINDS:
        return [f"{label}: unknown kind {kind!r}"]
    if not isinstance(actions, list):
        return [f"{label}: actions must be a list"]
    problems = [
        f"{label}: {action!r} is not an allow-listed action"
        for action in actions if action not in allowed_actions
    ]
    if (kind == KIND_QUEUE) != bool(actions):
        problems.append(f"{label}: only queue cases carry actions, and they must")
    return problems


def _is_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())
