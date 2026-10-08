"""The per-turn classifier's model call, injectable so tests never reach a network.

A ClassifierCall takes the chat messages (system prompt, then what the person
said) and returns the model's answer text.
"""

from typing import Callable

import turn_log
from turn_timing import span

ClassifierCall = Callable[[list], str]

CLASSIFY_SPAN = "reasoner.classify_llm"
CLASSIFY_FALLBACK_SPAN = "reasoner.classify_llm_fallback"


def classifier_call_through(handler, span_name: str) -> ClassifierCall:
    """A ClassifierCall that asks one OpenAI-style handler, timed as `span_name`."""

    def ask(messages: list) -> str:
        with span(span_name):
            response = handler.send_text(messages, stream=False)
        answer = response.choices[0].message.content
        turn_log.step("classify", f"{response.model} -> {answer!r}")
        return answer

    return ask


def with_fallback(primary: ClassifierCall, fallback: ClassifierCall) -> ClassifierCall:
    """Ask `fallback` only when `primary` raises."""

    def ask(messages: list) -> str:
        try:
            return primary(messages)
        except Exception as error:
            print("classifier model failed, trying the fallback", error)
            return fallback(messages)

    return ask


def default_classifier_call(chatgpt, grok) -> ClassifierCall:
    return with_fallback(
        classifier_call_through(chatgpt, CLASSIFY_SPAN),
        classifier_call_through(grok, CLASSIFY_FALLBACK_SPAN),
    )
