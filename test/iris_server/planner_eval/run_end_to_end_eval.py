"""Score the whole path a spoken sentence takes: classifier, then planner.

For each labelled sentence the classifier decides whether it is a physical
request and, if so, the planner builds the queue; the production reasoner makes
both calls and the routing decision (`_Reasoner.decide`). The result is scored
against the same labels as the planner-only stage (run_planner_eval.py), with
extra figures for the classifier stage and the planner's veto.

Live run (real model, real cost): see README.md. `--fake` answers from the
labels instead, to check this runner offline; add `--fake-mistakes` to make the
fake classifier miss and over-flag a few cases so those figures are exercised.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import end_to_end_adapter as adapter
import end_to_end_report
import fake_models
import planner_environment
import run_planner_eval
from error_scrubbing import describe_error, exit_with_scrubbed_error
from eval_cases import KIND_COULD_NOT_PLAN

DEFAULT_CASES_PATH = Path(__file__).with_name("cases.json")
# The documented container mounts the repo read-only, so results go to /tmp
# unless the caller passes a writable --out.
DEFAULT_OUT_PATH = "/tmp/end_to_end_eval_results.json"


def evaluate_case(case: dict, reasoner, action_by_state: dict) -> dict:
    started = time.perf_counter()
    try:
        decision = reasoner.decide(case["sentence"])
    except Exception as error:
        # Only the class name and status: an API error can echo a key.
        lost = {"kind": KIND_COULD_NOT_PLAN, "actions": [], "reason": describe_error(error)}
        result = run_planner_eval.scored_result(case, lost, time.perf_counter() - started)
        return {**result, **adapter.NO_STAGE_FACTS}
    seconds = time.perf_counter() - started
    got = adapter.label_of_decision(decision, action_by_state)
    return {**run_planner_eval.scored_result(case, got, seconds), **adapter.stage_facts(decision)}


def evaluate_all(cases: list, reasoner, action_by_state: dict) -> list:
    with ThreadPoolExecutor(max_workers=run_planner_eval.WORKER_COUNT) as pool:
        return list(pool.map(lambda case: evaluate_case(case, reasoner, action_by_state), cases))


def fake_reasoner(cases: list, action_by_state: dict, flaws):
    """A reasoner whose two models answer from the labels, wrong only on `flaws`."""
    state_by_action = {action: state for state, action in action_by_state.items()}
    return adapter.reasoner_with(
        fake_models.fake_classifier(cases, state_by_action, flaws),
        fake_models.fake_planner_model(cases, flaws),
    )


def write_results(path: str, results: list, classifier_model, planner_model) -> None:
    run_planner_eval.write_json(path, {
        "classifier_model": classifier_model,
        "planner_model": planner_model,
        "results": results,
    })


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true",
                        help="answer from the labels; no model, no network")
    parser.add_argument("--fake-mistakes", action="store_true",
                        help="with --fake: the classifier misses and over-flags a few cases")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="where to write results.json")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="the labelled cases")
    arguments = parser.parse_args()
    if arguments.fake_mistakes and not arguments.fake:
        parser.error("--fake-mistakes only makes sense with --fake")
    return arguments


def main():
    arguments = parse_arguments()
    planner_environment.prepare_server_imports()
    cases = run_planner_eval.load_valid_cases(arguments.cases)
    action_by_state = planner_environment.action_by_state()
    if arguments.fake:
        flaws = fake_models.choose_flaws(cases) if arguments.fake_mistakes else None
        reasoner, classifier_model, planner_model = (
            fake_reasoner(cases, action_by_state, flaws), None, None)
    else:
        reasoner = adapter.real_reasoner()
        classifier_model = planner_environment.classifier_model_name()
        planner_model = planner_environment.planner_model_name()

    results = evaluate_all(cases, reasoner, action_by_state)
    write_results(arguments.out, results, classifier_model, planner_model)
    print("\n".join(end_to_end_report.summary_lines(results, classifier_model, planner_model)))
    print(f"results written to {arguments.out}")


if __name__ == "__main__":
    exit_with_scrubbed_error(main)
