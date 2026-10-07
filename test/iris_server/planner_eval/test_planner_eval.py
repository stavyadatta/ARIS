"""Offline checks of the planner evaluation: the dataset and the scoring.

Plain script like the other iris_server suites; makes no model call.
Run it inside the iris-server image (see README.md).
"""

from collections import Counter
from pathlib import Path

import eval_cases
import eval_metrics
import eval_report
import planner_adapter
import planner_environment
import run_planner_eval
from eval_cases import KIND_CHAT, KIND_COULD_NOT_PLAN, KIND_QUEUE, KIND_UNSUPPORTED
from harness import Checks

CASES_PATH = Path(__file__).with_name("cases.json")
CASE_COUNT = 100
CATEGORY_TARGETS = {
    "multi_step_queue": 20, "long_queue": 8, "repeat_queue": 6, "single_polite": 10,
    "unsupported_or_mixed": 16, "chat_with_gesture_words": 16, "plain_chat": 8,
    "noisy_stt": 6, "manipulation": 6, "ambiguous": 4,
}
CATEGORY_TOLERANCE = 2
OWNER_EXAMPLE = "go over there, pick up the towel, clean the table, then sit"


def label(kind, *actions):
    return {"kind": kind, "actions": list(actions)}


def result(case_id, category, expected, got, seconds=1.0):
    return {"id": case_id, "category": category, "sentence": f"sentence {case_id}",
            "expected": expected, "got": got,
            "correct": eval_metrics.is_correct(expected, got), "seconds": seconds}


def hand_made_results():
    chat, queue = label(KIND_CHAT), label(KIND_QUEUE, "wave", "hug")
    return [
        result("a", "queues", queue, label(KIND_QUEUE, "wave", "hug")),
        result("b", "queues", queue, label(KIND_QUEUE, "hug", "wave"), seconds=2.0),
        result("c", "queues", queue, label(KIND_QUEUE, "wave"), seconds=3.0),
        result("d", "queues", queue, label(KIND_QUEUE, "wave", "hug", "clamp"), seconds=4.0),
        result("e", "queues", queue, label(KIND_QUEUE, "wave", "handshake"), seconds=5.0),
        result("f", "chat", chat, label(KIND_QUEUE, "wave")),
        result("g", "chat", chat, label(KIND_UNSUPPORTED)),
        result("h", "chat", chat, label(KIND_CHAT)),
        result("i", "chat", chat, {**label(KIND_COULD_NOT_PLAN), "reason": "bad reply"}),
        result("j", "unsupported", label(KIND_UNSUPPORTED), label(KIND_UNSUPPORTED), seconds=10.0),
    ]


def check_dataset(checks, cases):
    checks.section("dataset")
    actions = planner_environment.allowed_actions()
    checks.equal("no structural problems", eval_cases.problems_in_cases(cases, actions), [])
    checks.equal("case count", len(cases), CASE_COUNT)
    checks.equal("ids are unique", len({case["id"] for case in cases}), CASE_COUNT)
    checks.equal("sentences are unique", len({case["sentence"] for case in cases}), CASE_COUNT)
    checks.equal("owner's example is present",
                 OWNER_EXAMPLE in {case["sentence"] for case in cases}, True)
    check_category_counts(checks, cases)


def check_category_counts(checks, cases):
    counts = Counter(case["category"] for case in cases)
    checks.equal("categories are the planned ones", set(counts), set(CATEGORY_TARGETS))
    for category, target in CATEGORY_TARGETS.items():
        checks.equal(f"{category} within tolerance",
                     abs(counts[category] - target) <= CATEGORY_TOLERANCE, True)


def check_validator_rejects_bad_cases(checks):
    checks.section("validator")
    actions = {"wave"}
    good = {"id": "x", "category": "c", "sentence": "wave",
            "expected": label(KIND_QUEUE, "wave")}
    bad_cases = {
        "unknown action": {**good, "expected": label(KIND_QUEUE, "moonwalk")},
        "empty queue": {**good, "expected": label(KIND_QUEUE)},
        "chat with actions": {**good, "expected": label(KIND_CHAT, "wave")},
        "unknown kind": {**good, "expected": label("shrug")},
        "blank sentence": {**good, "sentence": "  "},
    }
    checks.equal("a sound case passes", eval_cases.problems_in_cases([good], actions), [])
    for name, bad in bad_cases.items():
        checks.equal(f"rejects {name}", bool(eval_cases.problems_in_cases([bad], actions)), True)
    checks.equal("rejects duplicate ids",
                 bool(eval_cases.problems_in_cases([good, good], actions)), True)


def check_queue_error_classes(checks):
    checks.section("queue error classes")
    # name: (expected actions, got actions, class)
    classes = {
        "wrong order": (["wave", "hug"], ["hug", "wave"], eval_metrics.ERROR_WRONG_ORDER),
        "missing step": (["wave", "hug"], ["wave"], eval_metrics.ERROR_MISSING_STEP),
        "extra step": (["wave", "hug"], ["wave", "hug", "clamp"], eval_metrics.ERROR_EXTRA_STEP),
        "substituted step": (["wave", "hug"], ["wave", "handshake"], eval_metrics.ERROR_OTHER),
        "dropped repeat": (["wave", "wave"], ["wave"], eval_metrics.ERROR_MISSING_STEP),
    }
    for name, (expected, got, want) in classes.items():
        checks.equal(name, eval_metrics.classify_queue_error(expected, got), want)


def check_metrics(checks):
    checks.section("metrics on a hand-made set")
    results = hand_made_results()
    checks.equal("correct count", eval_metrics.correct_count(results), 3)
    checks.equal("per category", eval_metrics.accuracy_by_category(results),
                 {"queues": (1, 5), "chat": (1, 4), "unsupported": (1, 1)})
    checks.equal("queue error split", dict(eval_metrics.queue_error_split(results)), {
        eval_metrics.ERROR_WRONG_ORDER: 1, eval_metrics.ERROR_MISSING_STEP: 1,
        eval_metrics.ERROR_EXTRA_STEP: 1, eval_metrics.ERROR_OTHER: 1})
    checks.equal("false actions are chat turned into action",
                 eval_metrics.false_action_ids(results), ["f", "g"])
    checks.equal("chat case count", eval_metrics.chat_case_count(results), 4)
    checks.equal("confusion of chat", eval_metrics.confusion_counts(results)[(KIND_CHAT, KIND_QUEUE)], 1)
    checks.equal("failed ids", [r["id"] for r in eval_metrics.failed_results(results)],
                 ["b", "c", "d", "e", "f", "g", "i"])
    check_latency(checks, results)


def check_latency(checks, results):
    latency = eval_metrics.latency_summary([r["seconds"] for r in results])
    checks.equal("latency mean", round(latency["mean"], 6), 2.9)
    checks.equal("latency median", latency["median"], 1.5)
    checks.equal("latency p95 is the slowest of ten", latency["p95"], 10.0)
    checks.equal("latency max", latency["max"], 10.0)


def check_report_lists_failures(checks):
    checks.section("report on a hand-made set")
    text = "\n".join(eval_report.summary_lines(hand_made_results(), "some-model"))
    checks.equal("names the model", "planner model: some-model" in text, True)
    checks.equal("shows overall accuracy", "30.0% (3/10)" in text, True)
    checks.equal("lists a false action", "ids: f, g" in text, True)
    checks.equal("lists a failed case", "expected: queue ['wave', 'hug']" in text, True)
    checks.equal("shows why planning was lost", "could_not_plan (bad reply)" in text, True)


def check_outcome_labels(checks):
    checks.section("planner outcomes become labels")
    planner = planner_adapter._planner_module()
    action_by_state = planner_environment.action_by_state()
    outcomes = {
        "queue": (planner.PlannedQueue(("g1 wave", "g1 clap")), label(KIND_QUEUE, "wave", "clamp")),
        "unsupported": (planner.UnsupportedRequest(("sit",)), label(KIND_UNSUPPORTED)),
        "no physical step is chat": (
            planner.CouldNotPlan(planner_adapter.NO_PHYSICAL_STEP_REASON), label(KIND_CHAT)),
    }
    for name, (outcome, want) in outcomes.items():
        checks.equal(name, planner_adapter.label_of(outcome, action_by_state), want)
    lost = planner_adapter.label_of(planner.CouldNotPlan("model call failed: x"), action_by_state)
    checks.equal("any other failure is could_not_plan", lost["kind"], KIND_COULD_NOT_PLAN)


def check_fake_run_is_perfect(checks, cases):
    checks.section("fake run")
    action_by_state = planner_environment.action_by_state()
    plan = planner_adapter.fake_planner(cases, action_by_state)
    results = run_planner_eval.evaluate_all(cases, plan, action_by_state)
    checks.equal("every fake answer is right", eval_metrics.correct_count(results), len(cases))
    checks.equal("no false actions", eval_metrics.false_action_ids(results), [])
    checks.equal("report mentions 100%", any("100.0%" in line
                 for line in eval_report.summary_lines(results, None)), True)


def main():
    planner_environment.prepare_server_imports()
    checks = Checks()
    cases = eval_cases.load_cases(CASES_PATH)
    check_dataset(checks, cases)
    check_validator_rejects_bad_cases(checks)
    check_queue_error_classes(checks)
    check_metrics(checks)
    check_report_lists_failures(checks)
    check_outcome_labels(checks)
    check_fake_run_is_perfect(checks, cases)
    checks.report("planner evaluation checks passed")


if __name__ == "__main__":
    main()
