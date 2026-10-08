"""Score the request planner against the 100 labelled sentences in cases.json.

Live run (real model, real cost): see README.md. `--fake` answers every
sentence with its expected label instead, to check this runner offline.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import eval_cases
import eval_metrics
import eval_report
import planner_adapter
import planner_environment
from error_scrubbing import exit_with_scrubbed_error

# Small enough to stay inside the model's rate limit, large enough that 100
# calls of a second or two each finish in about a minute.
WORKER_COUNT = 8

DEFAULT_CASES_PATH = Path(__file__).with_name("cases.json")
# The documented container mounts the repo read-only, so results go to /tmp
# unless the caller passes a writable --out.
DEFAULT_OUT_PATH = "/tmp/planner_eval_results.json"


def scored_result(case: dict, got: dict, seconds: float) -> dict:
    """One case's outcome next to its label; shared with the end-to-end stage."""
    return {
        "id": case["id"],
        "category": case["category"],
        "sentence": case["sentence"],
        "expected": case["expected"],
        "got": got,
        "correct": eval_metrics.is_correct(case["expected"], got),
        "seconds": seconds,
    }


def evaluate_case(case: dict, plan, action_by_state: dict) -> dict:
    started = time.perf_counter()
    outcome = plan(case["sentence"])
    seconds = time.perf_counter() - started
    return scored_result(case, planner_adapter.label_of(outcome, action_by_state), seconds)


def evaluate_all(cases: list, plan, action_by_state: dict) -> list:
    with ThreadPoolExecutor(max_workers=WORKER_COUNT) as pool:
        return list(pool.map(lambda case: evaluate_case(case, plan, action_by_state), cases))


def write_json(path: str, document: dict) -> None:
    with open(path, "w", encoding="utf-8") as out_file:
        json.dump(document, out_file, indent=2, ensure_ascii=False)


def write_results(path: str, results: list, model_name) -> None:
    write_json(path, {"planner_model": model_name, "results": results})


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true",
                        help="answer every sentence with its expected label; no model, no network")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="where to write results.json")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="the labelled cases")
    return parser.parse_args()


def load_valid_cases(path: str) -> list:
    cases = eval_cases.load_cases(path)
    problems = eval_cases.problems_in_cases(cases, planner_environment.allowed_actions())
    if problems:
        raise SystemExit("invalid cases:\n  " + "\n  ".join(problems))
    return cases


def main():
    arguments = parse_arguments()
    planner_environment.prepare_server_imports()
    cases = load_valid_cases(arguments.cases)
    action_by_state = planner_environment.action_by_state()
    if arguments.fake:
        plan, model_name = planner_adapter.fake_planner(cases, action_by_state), None
    else:
        plan, model_name = planner_adapter.real_planner(), planner_environment.planner_model_name()

    results = evaluate_all(cases, plan, action_by_state)
    write_results(arguments.out, results, model_name)
    print("\n".join(eval_report.summary_lines(results, model_name)))
    print(f"results written to {arguments.out}")


if __name__ == "__main__":
    exit_with_scrubbed_error(main)
