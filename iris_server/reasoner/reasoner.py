import traceback
import re
from typing import Optional

from utils import (
    G1_SEQUENCE_ATTRIBUTE,
    G1_SEQUENCE_STATE,
    UNSUPPORTED_ACTION_STATE,
    UNSUPPORTED_STEPS_ATTRIBUTE,
    Neo4j,
    PersonDetails,
    message_format,
)
from core_api import Llama, ChatGPT, Grok
import turn_log
from turn_timing import mark, span
from .action_planner import model_call_through
from .classifier_call import default_classifier_call
from .prompt import action_reasoner_prompt
from .turn_decision import (
    CONFIRM_STATE_PREFIX,
    G1_STATE_PREFIX,
    STATE_SPEAK,
    TurnDecision,
    decide_turn,
)

STATE_NO_FACE = "no face"
STATE_THANKS = "thanks"
STATE_BAD_INPUT = "speak"
NO_CHANGE_RESPONSES = ("no change", "no change.")
# Only "no change" means "whatever state this person is already in stands".
#
# "bad input" must NOT be here. Substituting the previous state sent a garbled
# or silent utterance to whichever API the person was last routed to -- usually
# Speaking, which then answered from conversation history and produced a
# confident reply to something nobody said. It has to reach the bad-input API,
# which yields nothing so the handler speaks a listening fallback instead.
KEEP_CURRENT_STATE_RESPONSES = ("no change",)

# The whole vocabulary that answers an outstanding "did you mean?" question.
# It decides nothing about what the person wants: the question was asked by the
# classifier, and "yes" only accepts it.
_CONFIRMATION_REPLIES = frozenset({
    "yes", "yes please", "yeah", "yep", "correct", "do it", "please do",
})


def _words_in(text: str) -> list:
    return re.findall(r"[a-z]+", text.lower())


# A turn made only of these is a courtesy shortcut, not a physical gate: it
# saves a model call on a plain thank-you. The robot's own name may be tacked on
# ("thanks Iris"); anything else ("thanks for the dance, now wave") goes to the
# classifier.
_THANKS_PHRASES = frozenset({
    "thank you", "thanks", "thank you very much", "thank you so much",
    "thanks a lot", "thanks so much", "many thanks", "thank you kindly",
    "cheers",
})
_ROBOT_NAME = "iris"


def _is_pure_thanks(transcription: str) -> bool:
    words = re.sub(r"[^a-z\s']", " ", transcription.lower()).split()
    phrase = " ".join(word for word in words if word != _ROBOT_NAME)
    return phrase in _THANKS_PHRASES


class _Reasoner:
    def __init__(self, ask_planner_model=None, ask_classifier_model=None):
        """
            Initializing the reasoner
            :param ask_planner_model: the model call that plans a queue of
                gestures (see action_planner.ModelCall); ChatGPT by default.
            :param ask_classifier_model: the model call that classifies each
                turn (see classifier_call.ClassifierCall); ChatGPT, with Grok
                as the fallback, by default.
                Both are injected so tests never reach a network.
        """
        self._ask_planner_model = ask_planner_model or model_call_through(ChatGPT)
        self._ask_classifier_model = ask_classifier_model or default_classifier_call(ChatGPT, Grok)

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

    def _route_unsupported_request(self, person_details: PersonDetails,
                                   steps: tuple, transcription: str,
                                   user_prompt: list) -> PersonDetails:
        person_details.set_attribute(UNSUPPORTED_STEPS_ATTRIBUTE, list(steps))
        return self._route(
            person_details,
            UNSUPPORTED_ACTION_STATE,
            user_prompt,
            f"[g1_action] transcription={transcription!r} "
            f"route={UNSUPPORTED_ACTION_STATE} steps={list(steps)}",
        )

    def _route_gesture_sequence(self, person_details: PersonDetails,
                                gestures: list, transcription: str,
                                user_prompt: list) -> PersonDetails:
        person_details.set_attribute(G1_SEQUENCE_ATTRIBUTE, gestures)
        return self._route(
            person_details,
            G1_SEQUENCE_STATE,
            user_prompt,
            f"[g1_action] transcription={transcription!r} "
            f"route={G1_SEQUENCE_STATE} gestures={gestures}",
        )

    def _route_thanks(self, person_details: PersonDetails, transcription: str,
                      user_prompt: list) -> Optional[PersonDetails]:
        if not _is_pure_thanks(transcription):
            return None
        return self._route(person_details, STATE_THANKS, user_prompt,
                           f"[thanks] transcription={transcription!r}")

    def _routed_by_gates(self, person_details: PersonDetails, transcription: str,
                         user_prompt: list) -> Optional[PersonDetails]:
        """Answer the two turns that need no model: a reply to an open question
        and a pure thank-you. Neither decides whether something is a request
        for the robot's body; the classifier does that."""
        for route in (self._answer_pending_confirmation, self._route_thanks):
            routed = route(person_details, transcription, user_prompt)
            if routed is not None:
                return routed
        return None

    def decide(self, transcription: str) -> TurnDecision:
        """What the classifier and, for a physical request, the planner make of a sentence.

        Reads and writes no person record, so the evaluation can call it.
        """
        total_prompt = self._developing_reasoning_prompt() + self._developing_user_prompt(transcription)
        classifier_answer = self._ask_classifier_model(total_prompt)
        return decide_turn(classifier_answer, transcription, self._ask_planner_model)

    def _route_decision(self, person_details: PersonDetails, decision: TurnDecision,
                        transcription: str, user_prompt: list) -> PersonDetails:
        if decision.gestures:
            return self._route_gesture_sequence(
                person_details, list(decision.gestures), transcription, user_prompt
            )
        if decision.unsupported_steps:
            return self._route_unsupported_request(
                person_details, decision.unsupported_steps, transcription, user_prompt
            )
        return self._route_llm_state(person_details, decision.state, user_prompt)

    def _route_llm_state(self, person_details: PersonDetails, response_text: str,
                         user_prompt: list) -> PersonDetails:
        if response_text in KEEP_CURRENT_STATE_RESPONSES:
            response_text = person_details.get_attribute("state")

        if response_text not in NO_CHANGE_RESPONSES:
            person_details.set_attribute("state", response_text)
            turn_log.debug(f"person state: {person_details.get_attribute('state')}")

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
        if face_id is None:
            # No recognisable face in frame. This is a *vision* failure, so
            # ask to be seen rather than falling through to "bad input",
            # which speaks an audio retry request and misleads the person.
            mark("classifier", "no_face")
            return PersonDetails({"state": STATE_NO_FACE})
        try:
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
            decision = self.decide(transcription)
            return self._route_decision(person_details, decision, transcription, user_prompt)

        except Exception as e:
            print(f"Error in reasoning section: {e}")
            traceback.print_exc()
            raise Exception(e)
