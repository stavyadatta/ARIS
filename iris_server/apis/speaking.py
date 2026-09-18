from threading import Thread
from typing import Any

from core_api import ChatGPT, Grok, RelationshipChecker, AttributeFinder
from utils import PersonDetails, Neo4j, message_format, ApiObject
from turn_timing import mark, record_first_moment, span
from .api_base import ApiBase

# A backstop, not the main lever -- the prompt is what keeps replies short.
# The G1 client blanks its microphone for 77 ms per character after it starts
# speaking (robot/speak.cpp), clamped at 30 s, so a runaway reply does not just
# cost generation time, it deafens the robot.
#
# Deliberately above a normal two-sentence reply (measured 52-122 characters).
# A cap that binds truncates mid-sentence -- 60 tokens was tried and produced
# replies ending "How's your", which the robot would have spoken aloud. This
# bounds the pathological case only; brevity is the prompt's job.
SPOKEN_REPLY_MAX_TOKENS = 120


class _Speaking(ApiBase):
    def __init__(self) -> None:
        super().__init__()

    def _developing_system_prompt(self, 
                                  person_name, 
                                  person_attributes, 
                                  person_relationships
        ):
        system_prompt = f"""
            Your are playing the role of Iris robot which is a humanoid, as part of this 
            role you are a supposed to have friendly human conversations similar to 
            how people on facebook messenger chat.

            Take care of following before responding
            1) Is my answer relevant to what the user asked
            2) Answer in at most two short sentences. You are speaking out loud to
            someone standing in front of you, not writing a message. If you know
            many things about them, pick the one worth saying and leave the rest
            for when they ask.
            3) Am I polite
            4) Never admit visual errors; visual input is handled by another assistant discreetly.
            6) You may receive dancing context, do not admit you cannot dance, being a 
            robot you can certainly dance. 
            7) Replace your output "as large language model" with "as a robot"
            8) Donot use the following characters: '*', 'bullet points', 'numbered list'

            If you are asked questions like do you remember me, if you have context you should reply
            with yes and their name, then name ONE thing you share, not a list of
            everything you know. Listing everything sounds like a database reciting
            itself, and they can always ask for more.

            for example, treat texts in <> as conditional prompts
            ```

            input: Hey do you remember me
            response: <If name in context> Of course I remember you, <name>! How is <one thing from context> going?

            input: Hey how are you 
            output: I am good, great to see you <name> how are you doing

            input: What did you say before 
            output: <Use latest conversation messages to answer this question>

            input: What do you know about my friendship <or any other relationship>
            output: I know you are friends <or any other relationship> with <people name if details have been provided>
            ```

            Here are some more details about the person 

            name: {person_name}
            person_attributes: {person_attributes}

            Here are the relationships this person has with people: {person_relationships}
        """

        system_dict = message_format("system", system_prompt)
        return [system_dict]
        
    def __call__(self, person_details: PersonDetails) -> Any:
        with span("speaking.context_build"):
            messages, total_prompt = self._build_context(person_details)

        with span("speaking.open_stream"):
            response = self._request_completion(total_prompt)

        llm_response = ""
        with span("speaking.generation"):
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    record_first_moment("speaking_first_token_at_ms")
                    llm_response += content
                    yield ApiObject(content)
        mark("reply_chars", len(llm_response))

        self._record_turn_after_reply(person_details, messages, llm_response)

    def _record_turn_after_reply(self, person_details: PersonDetails,
                                 messages: list, llm_response: str):
        """Hand persistence to a background thread instead of blocking the reply.

        _g1_conversation_chunks holds every speech chunk until this generator
        is exhausted, so anything done here lands in front of the person
        hearing anything at all -- measured at 720 ms of two OpenAI embedding
        round trips plus a Cypher write, entirely after the words were known.

        The person must still hear the reply and answer before the next turn
        reads this back, which is far longer than the write takes. Mirrors the
        daemon-worker pattern in core_api/relationship_checker.
        """
        llm_dict = message_format("assistant", llm_response)
        person_details.set_latest_llm_message(llm_dict)
        person_details.set_relevant_messages(messages + [llm_dict])

        Thread(
            target=self._persist_turn,
            args=(person_details,),
            daemon=True,
        ).start()

    def _persist_turn(self, person_details: PersonDetails):
        try:
            Neo4j.add_message_to_person(person_details)
            RelationshipChecker.adding_text2relationship_checker(person_details)
        except Exception as e:
            # Off the request thread, so an exception here would otherwise be
            # swallowed and the turn would silently vanish from the graph.
            print(f"[speaking] persisting the turn failed: {e}")

    def _build_context(self, person_details: PersonDetails):
        """Gather everything the reply prompt needs. Reads no reasoner output."""
        latest_msg = person_details.get_latest_user_message()
        if person_details.is_guest():
            return [latest_msg], self._guest_prompt() + [latest_msg]

        face_id = person_details.get_attribute("face_id")
        with span("speaking.person_messages"):
            messages = Neo4j.get_person_messages(latest_msg, face_id)

        with span("speaking.relationships"):
            person_relationships = Neo4j.describe_relationships_by_face_id(face_id)
        system_dict = self._developing_system_prompt(
            person_details.get_attribute("name"),
            person_details.get_attribute("attributes"),
            person_relationships
        )

        return messages, system_dict + messages

    def _guest_prompt(self) -> list:
        """Talk to somebody Iris cannot see and will not remember.

        No name, no history and no relationships to draw on, so the persona
        prompt's whole "do you remember me" apparatus would only invite Iris
        to invent things about a stranger.
        """
        return [message_format("system", """
            You are Iris, a friendly humanoid robot, talking with someone you
            cannot see clearly and do not know by name. Be warm and helpful.

            Answer in at most two short sentences. You are speaking out loud,
            so no lists, no emoji. Do not claim to remember them or to know
            anything about them, and do not comment on not being able to see
            them -- just talk with them.
        """)]

    def _request_completion(self, total_prompt: list):
        # response = Llama.send_to_model(total_prompt, stream=True)
        # response = Claude.process_text(messages, system_dict, stream=True)
        try:
            return ChatGPT.send_text(total_prompt, stream=True, model='gpt-4-turbo',
                                     max_tokens=SPOKEN_REPLY_MAX_TOKENS)
        except Exception as e:
            print("chatgpt failed ", e)
            return Grok.send_text(total_prompt, stream=True, grok_model="grok-3")

