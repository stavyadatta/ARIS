import traceback
import random
import re
from typing import Optional

from utils import Neo4j, PersonDetails, message_format
from core_api import Llama, ChatGPT, Grok
from turn_timing import mark, span
from .prompt import action_reasoner_prompt

STATE_SPEAK = "speak"
STATE_BAD_INPUT = "bad input"
NO_CHANGE_RESPONSES = ("no change", "no change.")
# Only "no change" means "whatever state this person is already in stands".
#
# "bad input" must NOT be here. Substituting the previous state sent a garbled
# or silent utterance to whichever API the person was last routed to -- usually
# Speaking, which then answered from conversation history and produced a
# confident reply to something nobody said. It has to reach the bad-input API,
# which yields nothing so the handler speaks a listening fallback instead.
KEEP_CURRENT_STATE_RESPONSES = ("no change",)

G1_STATE_PREFIX = "g1 "
CONFIRM_STATE_PREFIX = "g1 confirm "

# The reasoner LLM classifies intent into a *family* of gestures ("g1
# greeting", "g1 farewell") rather than one specific action, so the family
# can be resolved to one of its concrete states by weighted probability
# instead of always picking the same gesture. Equal weights today; skew
# them (or add more entries) without touching the resolution logic.
_GREETING_GESTURE_WEIGHTS = {"g1 wave": 0.5, "g1 handshake": 0.5}
_FAREWELL_GESTURE_WEIGHTS = {"g1 blow kiss left": 0.5, "g1 blow kiss right": 0.5}
_GESTURE_FAMILY_WEIGHTS = {
    "g1 greeting": _GREETING_GESTURE_WEIGHTS,
    "g1 farewell": _FAREWELL_GESTURE_WEIGHTS,
}

# A gesture is only ever read as a request when one of these appears, so
# narration ("then she waved goodbye") cannot move the robot.
_GESTURE_REQUEST_MARKERS = (
    "please", "can you", "could you", "would you", "will you",
    "give me", "do a", "do an",
)

_CONFIRMATION_REPLIES = frozenset({
    "yes", "yes please", "yeah", "yep", "correct", "do it", "please do",
})

# What Whisper actually returns when someone asks G1 for a wave in a noisy
# room. An explicit list, because a similarity threshold loose enough to
# catch "wait" also catches "have", "gave" and "save" — so ordinary
# sentences ("can you have a look at this") asked the robot to wave.
# Only ever raises a confirmation question; it can never move the robot.
_WAVE_MISHEARINGS = frozenset({
    "wait", "waits", "waive", "waives", "weave", "wade", "wav", "waved",
})


def _weighted_choice(options: dict) -> str:
    """Pick one key from options, weighted by its probability value."""
    return random.choices(list(options.keys()), weights=list(options.values()), k=1)[0]


def _words_in(text: str) -> list:
    return re.findall(r"[a-z]+", text.lower())


def _has_request_marker(text: str) -> bool:
    return any(marker in text for marker in _GESTURE_REQUEST_MARKERS)


class _Reasoner:
    def __init__(self):
        """
            Initializing the reasoner
            :param llama_url: Endpoint for the llama.cpp
        """
        pass

    def to_lowercase(self, input_string):
        """
        Converts all characters in the input string to lowercase.

        Parameters:
            input_string (str): The string to convert.

        Returns:
            str: The input string in lowercase.
        """
        return input_string.lower()

    def _developing_reasoning_prompt(self):
        system_reasoner = action_reasoner_prompt
        system_dict = message_format("system", system_reasoner)
        return [system_dict]

    def _developing_user_prompt(self, text: str):
        user_prompt = message_format("user", text)
        return [user_prompt]

    def _requested_g1_gesture(self, transcription: str) -> Optional[str]:
        """Return one allow-listed G1 gesture for an explicit spoken request.

        Physical intents must not depend on a best-effort LLM classification.
        Whisper supplies the text; this small, auditable gate accepts only the
        three gestures the G1 client will later validate independently.
        """
        text = transcription.lower()
        if not _has_request_marker(text):
            return None
        if "high five" in text or "high-five" in text:
            return "g1 high five"
        if "handshake" in text or "shake my hand" in text or "shake hands" in text:
            return "g1 handshake"
        if "wave" in text:
            return "g1 wave"
        if "dance" in text:
            return "g1 waist drum dance"
        if "dj" in _words_in(text) or "spin disc" in text or "play records" in text:
            return "g1 spin discs"
        if "throw money" in text or "throw cash" in text or "make it rain" in text:
            return "g1 throw money"
        return None

    def _uncertain_g1_gesture(self, transcription: str) -> Optional[str]:
        """Return a confirmation-only gesture candidate, never an action."""
        text = transcription.lower()
        if not _has_request_marker(text):
            return None
        words = _words_in(text)
        if self._sounds_like_wave(words):
            return "g1 wave"
        if "shake" in words:
            return "g1 handshake"
        if "five" in words and any(word in {"hi", "high"} for word in words):
            return "g1 high five"
        return None

    def _sounds_like_wave(self, words: list) -> bool:
        return any(word in _WAVE_MISHEARINGS for word in words)

    def _confirmed_g1_gesture(self, transcription: str) -> bool:
        """Accept only a small explicit confirmation vocabulary."""
        return " ".join(_words_in(transcription)) in _CONFIRMATION_REPLIES

    def _person_record(self, face_id: str) -> PersonDetails:
        person_details = Neo4j.get_person_details(face_id)
        if not person_details:
            Neo4j.create_or_update_person(face_id=face_id)
            person_details = Neo4j.get_person_details(face_id)
        return person_details

    def _route(self, person_details: PersonDetails, state: str,
               user_prompt: list, log_line: str) -> PersonDetails:
        """Commit one chosen state and the utterance that selected it."""
        person_details.set_attribute("state", state)
        person_details.set_latest_usr_message(user_prompt[0])
        print(log_line)
        return person_details

    def _answer_pending_confirmation(self, person_details: PersonDetails,
                                     transcription: str,
                                     user_prompt: list) -> Optional[PersonDetails]:
        """Resolve an outstanding "did you mean X?" question, if any.

        Returns the routed record on confirmation, otherwise None so the
        utterance is classified afresh.
        """
        pending_state = str(person_details.get_attribute("state"))
        if not pending_state.startswith(CONFIRM_STATE_PREFIX):
            return None

        pending_gesture = pending_state.removeprefix(CONFIRM_STATE_PREFIX)
        if self._confirmed_g1_gesture(transcription):
            return self._route(
                person_details,
                G1_STATE_PREFIX + pending_gesture,
                user_prompt,
                f"[g1_action] confirmation={transcription!r} "
                f"route={G1_STATE_PREFIX}{pending_gesture}",
            )

        # Do not allow an abandoned question to trap future ordinary
        # conversation in confirmation mode.
        person_details.set_attribute("state", STATE_SPEAK)
        return None

    def _route_requested_gesture(self, person_details: PersonDetails,
                                 transcription: str,
                                 user_prompt: list) -> Optional[PersonDetails]:
        gesture_state = self._requested_g1_gesture(transcription)
        if gesture_state is None:
            return None
        return self._route(
            person_details,
            gesture_state,
            user_prompt,
            f"[g1_action] transcription={transcription!r} route={gesture_state}",
        )

    def _route_uncertain_gesture(self, person_details: PersonDetails,
                                 transcription: str,
                                 user_prompt: list) -> Optional[PersonDetails]:
        gesture = self._uncertain_g1_gesture(transcription)
        if gesture is None:
            return None
        return self._route(
            person_details,
            CONFIRM_STATE_PREFIX + gesture.removeprefix(G1_STATE_PREFIX),
            user_prompt,
            f"[g1_action] transcription={transcription!r} "
            f"confirmation_needed={gesture}",
        )

    def _routed_by_gates(self, person_details: PersonDetails, transcription: str,
                         user_prompt: list) -> Optional[PersonDetails]:
        """Try each deterministic gate in priority order, cheapest intent first."""
        for route in (self._answer_pending_confirmation,
                      self._route_requested_gesture,
                      self._route_uncertain_gesture):
            routed = route(person_details, transcription, user_prompt)
            if routed is not None:
                return routed
        return None

    def _classify_with_llm(self, total_prompt: list) -> str:
        try:
            with span("reasoner.classify_llm"):
                response = ChatGPT.send_text(total_prompt, stream=False)
            print("The response is ", response)
        except Exception as e:
            print("chatgpt failed ", e)
            with span("reasoner.classify_llm_fallback"):
                response = Grok.send_text(total_prompt, stream=False)
        return response.choices[0].message.content

    def _resolve_gesture_family(self, response_text: str) -> str:
        gesture_family = _GESTURE_FAMILY_WEIGHTS.get(response_text)
        if gesture_family is None:
            return response_text
        resolved_gesture = _weighted_choice(gesture_family)
        print(
            f"[g1_action] llm_category={response_text!r} "
            f"resolved={resolved_gesture}"
        )
        return resolved_gesture

    def _route_llm_state(self, person_details: PersonDetails, response_text: str,
                         user_prompt: list) -> PersonDetails:
        if response_text in KEEP_CURRENT_STATE_RESPONSES:
            response_text = person_details.get_attribute("state")

        if response_text not in NO_CHANGE_RESPONSES:
            person_details.set_attribute("state", response_text)
            print("Person State:", person_details.get_attribute("state"))

        person_details.set_latest_usr_message(user_prompt[0])
        return person_details

    def __call__(self, transcription, face_id: Optional[str], img=None) -> PersonDetails:
        """
            Running the reasoner and deciding on what APIs need to be run
            with the reasoner program
            :param text: using the text to prompt the llm on what to do
            :param face_id: To identify faces for doing an action
            :param img: for the VLM to get more context
        """
        try:
            if face_id is None:
                # Nobody recognised. Iris still listens and still acts; it
                # simply has no one to attribute the turn to, so the turn is
                # not stored. Refusing to answer instead made it tell people
                # standing in front of it that it could not see them.
                mark("guest", True)
                person_details = PersonDetails({"state": STATE_SPEAK})
            else:
                with span("reasoner.person_record"):
                    person_details = self._person_record(face_id)
            user_prompt = self._developing_user_prompt(transcription)

            with span("reasoner.gates"):
                routed = self._routed_by_gates(
                    person_details, transcription, user_prompt
                )
            if routed is not None:
                mark("classifier", "gate")
                return routed

            mark("classifier", "llm")
            total_prompt = self._developing_reasoning_prompt() + user_prompt
            response_text = self._resolve_gesture_family(
                self._classify_with_llm(total_prompt)
            )
            return self._route_llm_state(person_details, response_text, user_prompt)

        except Exception as e:
            print(f"Error in reasoning section: {e}")
            traceback.print_exc()
            raise Exception(e)
