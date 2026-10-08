"""Offline checks of the end-to-end evaluation: its fakes, its metrics and its scrubbing.

Plain script like the other iris_server suites; makes no model call.
Run it inside the iris-server image (see README.md).
"""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import end_to_end_adapter as adapter
import end_to_end_metrics as stage_metrics
import end_to_end_report
import eval_cases
import eval_metrics
import fake_models
import planner_environment
import run_end_to_end_eval
from error_scrubbing import ModelCallFailed, describe_error, exit_with_scrubbed_error, scrubbed
from eval_cases import KIND_CHAT, KIND_COULD_NOT_PLAN, KIND_QUEUE, KIND_UNSUPPORTED
from harness import Checks

CASES_PATH = Path(__file__).with_name("cases.json")
SECRET = "sk-proj-abcd1234SECRETKEY"
CONFIRM_ANSWER = "g1 confirm wave"


class FakeApiError(Exception):
    """Looks like an OpenAI status error: a message that echoes a key, and a status."""

    def __init__(self, status_code):
        super().__init__(f"Incorrect API key provided: {SECRET}")
        self.status_code = status_code


def label(kind, *actions):
    return {"kind": kind, "actions": list(actions)}


def build(cases, ask_classifier, ask_planner):
    planner_environment.prepare_server_imports()
    return adapter.reasoner_with(ask_classifier, ask_planner)


def run(cases, reasoner):
    return run_end_to_end_eval.evaluate_all(cases, reasoner, planner_environment.action_by_state())


def fake_run(cases, flaws=None):
    action_by_state = planner_environment.action_by_state()
    reasoner = run_end_to_end_eval.fake_reasoner(cases, action_by_state, flaws)
    return run(cases, reasoner)


def ids_of(cases, sentences):
    return [case["id"] for case in cases if case["sentence"] in sentences]


def check_error_scrubbing(checks):
    checks.section("errors are scrubbed")
    checks.equal("class and status only", describe_error(FakeApiError(401)), "FakeApiError (HTTP 401)")
    checks.equal("class only when there is no status", describe_error(ValueError(SECRET)), "ValueError")
    checks.equal("a non-numeric status is ignored", describe_error(FakeApiError("401")), "FakeApiError")

    def failing(*arguments):
        raise FakeApiError(429)

    try:
        scrubbed(failing)("anything")
        message, chained = None, None
    except ModelCallFailed as error:
        message, chained = str(error), error.__cause__
    checks.equal("the wrapper keeps no message text", message, "FakeApiError (HTTP 429)")
    checks.equal("and no chained original", chained, None)
    checks.equal("a working call passes through", scrubbed(lambda value: value + 1)(1), 2)

    def stops_with_secret():
        raise FakeApiError(500)

    try:
        exit_with_scrubbed_error(stops_with_secret)
        stopped = None
    except SystemExit as stop:
        stopped = str(stop)
    checks.equal("a stopped run reports class and status only",
                 stopped, "evaluation stopped: FakeApiError (HTTP 500)")


def check_errors_never_reach_results(checks, cases):
    checks.section("a failing model never leaks its message into a result")
    sample = cases[:3]
    action_by_state = planner_environment.action_by_state()
    state_by_action = {action: state for state, action in action_by_state.items()}
    working_planner = fake_models.fake_planner_model(sample)

    def classifier_down(messages):
        raise FakeApiError(401)

    results = run(sample, build(sample, scrubbed(classifier_down), working_planner))
    checks.equal("every turn is could_not_plan with the scrubbed reason",
                 {(r["got"]["kind"], r["got"]["reason"]) for r in results},
                 {(KIND_COULD_NOT_PLAN, "FakeApiError (HTTP 401)")})
    checks.equal("no secret anywhere in the results", SECRET in json.dumps(results), False)
    unscrubbed = run(sample, build(sample, classifier_down, working_planner))
    checks.equal("even an unscrubbed failure is reduced to its class and status",
                 {r["got"]["reason"] for r in unscrubbed}, {"FakeApiError (HTTP 401)"})

    def planner_down(messages, schema):
        raise FakeApiError(503)

    queue_cases = [case for case in cases if case["expected"]["kind"] == KIND_QUEUE][:2]
    classifier = fake_models.fake_classifier(queue_cases, state_by_action)
    results = run(queue_cases, build(queue_cases, classifier, scrubbed(planner_down)))
    checks.equal("a planner failure keeps the classifier's gesture and is counted",
                 (stage_metrics.planner_failed_ids(results), [r["got"]["kind"] for r in results]),
                 ([case["id"] for case in queue_cases], [KIND_QUEUE, KIND_QUEUE]))
    checks.equal("the planner's own reason carries no secret", SECRET in json.dumps(results), False)


def check_decision_labels(checks):
    checks.section("decisions become labels")
    utils = adapter._module("utils")
    turn_decision = adapter._module("reasoner.turn_decision")
    action_by_state = planner_environment.action_by_state()
    decide = turn_decision.TurnDecision
    decisions = {
        "a sequence": (decide(utils.G1_SEQUENCE_STATE, "g1 wave", gestures=("g1 wave", "g1 clap")),
                       label(KIND_QUEUE, "wave", "clamp")),
        "one gesture": (decide("g1 hug", "g1 hug"), label(KIND_QUEUE, "hug")),
        "unsupported": (decide(utils.UNSUPPORTED_ACTION_STATE, "g1 wave"), label(KIND_UNSUPPORTED)),
        "plain conversation": (decide("speak", "g1 wave"), label(KIND_CHAT)),
        "no change": (decide("no change", "no change"), label(KIND_CHAT)),
        "a question": (decide(CONFIRM_ANSWER, CONFIRM_ANSWER), label(KIND_CHAT)),
    }
    for name, (decision, want) in decisions.items():
        checks.equal(name, adapter.label_of_decision(decision, action_by_state), want)


def check_perfect_run(checks, cases):
    checks.section("fake run: perfect models")
    results = fake_run(cases)
    checks.equal("every answer is right", eval_metrics.correct_count(results), len(cases))
    checks.equal("no physical case missed", stage_metrics.classifier_missed_ids(results), [])
    checks.equal("no chat case flagged", stage_metrics.classifier_wrongly_flagged_ids(results), [])
    checks.equal("no false action", eval_metrics.false_action_ids(results), [])
    physical = stage_metrics.physical_case_count(results)
    groups = {name: len(seconds) for name, seconds in stage_metrics.latency_groups(results).items()}
    checks.equal("the planner is asked for exactly the physical turns",
                 groups, {"without the planner": len(cases) - physical, "through the planner": physical})
    checks.equal("the fake classifier answers the first step of a queue",
                 {r["classifier_answer"] for r in results if r["sentence"] == "dance after you wave"},
                 {"g1 wave"})


def check_flawed_run(checks, cases):
    checks.section("fake run: a classifier that misses and over-flags")
    flaws = fake_models.choose_flaws(cases)
    results = fake_run(cases, flaws)
    missed = ids_of(cases, flaws.missed_by_classifier)
    flagged = ids_of(cases, flaws.overflagged_by_classifier)
    fooled = ids_of(cases, flaws.fooled_planner)
    vetoed = [case_id for case_id in flagged if case_id not in fooled]
    checks.equal("the flaw sets have the intended sizes",
                 (len(missed), len(flagged), len(fooled)),
                 (fake_models.CLASSIFIER_MISSES_IN_QUEUE_CASES
                  + fake_models.CLASSIFIER_MISSES_IN_UNSUPPORTED_CASES,
                  fake_models.CLASSIFIER_OVERFLAGS_IN_CHAT_CASES,
                  fake_models.PLANNER_FOOLED_IN_CHAT_CASES))
    checks.equal("missed physical cases", stage_metrics.classifier_missed_ids(results), missed)
    checks.equal("wrongly flagged chat cases",
                 stage_metrics.classifier_wrongly_flagged_ids(results), flagged)
    checks.equal("the planner vetoed the others", stage_metrics.planner_vetoed_ids(results), vetoed)
    checks.equal("end-to-end false actions are the ones the planner believed",
                 eval_metrics.false_action_ids(results), fooled)
    checks.equal("every flaw makes a case wrong, and nothing else does",
                 sorted(r["id"] for r in eval_metrics.failed_results(results)),
                 sorted(missed + fooled))
    asked = sum(len(seconds) for name, seconds in stage_metrics.latency_groups(results).items()
                if name == "through the planner")
    checks.equal("missed cases skip the planner, flagged ones pay for it",
                 asked, stage_metrics.physical_case_count(results) - len(missed) + len(flagged))
    text = "\n".join(end_to_end_report.summary_lines(results, "classifier-x", "planner-y"))
    for line in (
        "classifier model: classifier-x", "planner model: planner-y",
        f"physical cases the classifier missed: 4.2% ({len(missed)}/72)",
        f"chat cases the classifier wrongly flagged: 10.7% ({len(flagged)}/28)",
        f"of those, vetoed by the planner (no steps requested): 66.7% ({len(vetoed)}/{len(flagged)})",
        f"END-TO-END false-action rate (chat turned into queue or unsupported): 3.6% ({len(fooled)}/28)",
        "turns without the planner:", "turns through the planner:",
    ):
        checks.equal(f"the report says {line!r}", line in text, True)


def check_confirmation_counted(checks, cases):
    checks.section("a 'did you mean?' answer is counted and moves nothing")
    chat_case = next(case for case in cases if case["expected"]["kind"] == KIND_CHAT)
    results = run([chat_case], build([chat_case], lambda messages: CONFIRM_ANSWER,
                                     fake_models.fake_planner_model([chat_case])))
    checks.equal("counted", stage_metrics.asked_to_confirm_ids(results), [chat_case["id"]])
    checks.equal("scored as chat", results[0]["got"], label(KIND_CHAT))
    checks.equal("the planner was not asked", results[0]["planner_asked"], False)


def check_report_with_no_physical_cases(checks, cases):
    checks.section("report on a set without physical cases")
    chat_cases = [case for case in cases if case["expected"]["kind"] == KIND_CHAT][:2]
    lines = end_to_end_report.summary_lines(fake_run(chat_cases), None, None)
    checks.equal("does not divide by zero",
                 "  physical cases the classifier missed: n/a (0/0)" in lines, True)
    checks.equal("names no models offline", any(line.startswith("classifier model") for line in lines), False)


def run_main(arguments):
    sys.argv = ["run_end_to_end_eval.py", *arguments]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        run_end_to_end_eval.main()
    return output.getvalue()


def check_runner_options(checks, cases):
    checks.section("runner options")
    small = [cases[0], next(c for c in cases if c["expected"]["kind"] == KIND_CHAT)]
    with tempfile.TemporaryDirectory() as folder:
        cases_path, out_path = Path(folder) / "second_cases.json", Path(folder) / "results.json"
        cases_path.write_text(json.dumps(small), encoding="utf-8")
        printed = run_main(["--fake", "--cases", str(cases_path), "--out", str(out_path)])
        document = json.loads(out_path.read_text(encoding="utf-8"))
    checks.equal("--cases takes another file", [r["id"] for r in document["results"]],
                 [case["id"] for case in small])
    checks.equal("--out is written and named", f"results written to {out_path}" in printed, True)
    checks.equal("the saved file names models and carries stage facts",
                 ("classifier_model" in document, "planner_model" in document,
                  "planner_asked" in document["results"][0]), (True, True, True))
    try:
        run_main(["--fake-mistakes"])
        refused = False
    except SystemExit as stop:
        refused = stop.code != 0
    checks.equal("--fake-mistakes alone is refused", refused, True)


def main():
    planner_environment.prepare_server_imports()
    checks = Checks()
    cases = eval_cases.load_cases(CASES_PATH)
    check_error_scrubbing(checks)
    check_errors_never_reach_results(checks, cases)
    check_decision_labels(checks)
    check_perfect_run(checks, cases)
    check_flawed_run(checks, cases)
    check_confirmation_counted(checks, cases)
    check_report_with_no_physical_cases(checks, cases)
    check_runner_options(checks, cases)
    checks.report("end-to-end evaluation checks passed")


if __name__ == "__main__":
    main()
