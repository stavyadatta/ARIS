"""Turn scored results into the summary a person reads."""

import eval_metrics
from eval_cases import KIND_CHAT, KIND_COULD_NOT_PLAN, KIND_QUEUE, KIND_UNSUPPORTED

PERCENT = 100
KIND_ORDER = (KIND_QUEUE, KIND_UNSUPPORTED, KIND_CHAT, KIND_COULD_NOT_PLAN)


def summary_lines(results: list, model_name) -> list:
    sections = [
        _headline_lines(results, model_name),
        _category_lines(results),
        _confusion_lines(results),
        _queue_error_lines(results),
        _false_action_lines(results),
        _latency_lines(results),
        _failure_lines(results),
    ]
    return [line for section in sections for line in section + [""]]


def _percent(correct: int, total: int) -> str:
    return f"{PERCENT * correct / total:.1f}% ({correct}/{total})"


def _headline_lines(results, model_name) -> list:
    correct = eval_metrics.correct_count(results)
    lines = [f"planner model: {model_name}"] if model_name else []
    return lines + [f"overall exact-match accuracy: {_percent(correct, len(results))}"]


def _category_lines(results) -> list:
    tally = eval_metrics.accuracy_by_category(results)
    return ["accuracy per category:"] + [
        f"  {category:<26}{_percent(correct, total)}"
        for category, (correct, total) in tally.items()
    ]


def _confusion_lines(results) -> list:
    counts = eval_metrics.confusion_counts(results)
    lines = ["confusion (expected -> got):"]
    for expected in KIND_ORDER:
        for got in KIND_ORDER:
            if counts[(expected, got)]:
                lines.append(f"  {expected:<12} -> {got:<15}{counts[(expected, got)]}")
    return lines


def _queue_error_lines(results) -> list:
    split = eval_metrics.queue_error_split(results)
    lines = ["wrong queues (queue cases answered with a queue):"]
    for error in (eval_metrics.ERROR_WRONG_ORDER, eval_metrics.ERROR_MISSING_STEP,
                  eval_metrics.ERROR_EXTRA_STEP, eval_metrics.ERROR_OTHER):
        lines.append(f"  {error:<30}{split[error]}")
    return lines


def _false_action_lines(results) -> list:
    wrong_ids = eval_metrics.false_action_ids(results)
    rate = _percent(len(wrong_ids), eval_metrics.chat_case_count(results))
    listed = ", ".join(wrong_ids) if wrong_ids else "none"
    return [f"FALSE-ACTION rate (chat turned into queue or unsupported): {rate}",
            f"  ids: {listed}"]


def _latency_lines(results) -> list:
    latency = eval_metrics.latency_summary([result["seconds"] for result in results])
    return ["latency per call (seconds):"] + [
        f"  {name:<8}{value:.3f}" for name, value in latency.items()
    ]


def _failure_lines(results) -> list:
    failed = eval_metrics.failed_results(results)
    lines = [f"failed cases: {len(failed)}"]
    for result in failed:
        lines += [
            f"  {result['id']} [{result['category']}] {result['sentence']!r}",
            f"      expected: {_describe(result['expected'])}",
            f"      got:      {_describe(result['got'])}",
        ]
    return lines


def _describe(label: dict) -> str:
    text = f"{label['kind']} {label['actions']}" if label["actions"] else label["kind"]
    reason = label.get("reason")
    return f"{text} ({reason})" if reason else text
