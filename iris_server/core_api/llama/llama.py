import os
import json
import numpy as np
from openai import OpenAI
from neo4j import GraphDatabase

class _Llama:
    def __init__(self, llama_url="http://127.0.0.2:8080/v1"):
        """
        Initialize the Llama handler with the llama.cpp server URL.
        :param llama_url: Endpoint of the llama.cpp server.
        """
        self.llama_url = llama_url
        api_key = os.environ["API_KEY"]
        self.client = OpenAI(
            #base_url="http://localhost:8080/v1", 127.0.0.1
            base_url=llama_url,
            api_key = api_key
        )

    def send_to_model(self, messages: list[dict], stream: bool, img=None):
        """
            :param messages: A dictionary of messages for additional context to be 
             provided to the model for benefit
            :param stream: Whether to stream the output or not
            :param img: incase of VLM adding an image for additional context

            :return: Generator of words from llm incase of stream otherwise whole text 
                output
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages= messages,
            temperature=0.7,
            max_tokens=200,
            top_p=0.9,
            stream=stream
        )
        return response

    # def send_to_llama(self, text, face_id):
    #     """
    #     Send transcribed text to the llama.cpp server and stream its response.
    #     :param text: Transcribed text to send.
    #     :return: Generator yielding streamed responses from the llama.cpp server.
    #     """
    #     try:
    #         person_details = self.get_person_details(face_id)
    #         if person_details is None:
    #             self.create_or_update_person(face_id)
    #             person_details = self.get_person_details(face_id)
    #
    #         conversation_history = person_details.get("messages", []) if person_details else []
    #
    #         # Prepare messages for LLM input
    #         messages = [{"role": "system", "content": "You are a helpful assistant (strictly under 20 words)."}]
    #         for message in conversation_history:
    #             role = "user" if message["user"] == "user" else "assistant"
    #             messages.append({"role": role, "content": message["message"]})
    #         messages.append({"role": "user", "content": text})
    #
    #         self.add_message_to_person(face_id, "user", text)
    #
    #         response = self.client.chat.completions.create(
    #             model="gpt-3.5-turbo",
    #             messages= messages,
    #             temperature=0.7,
    #             max_tokens=100,
    #             top_p=0.9,
    #             stream=True
    #         )
    #         llm_response = ""
    #         for chunk in response:
    #             if chunk.choices[0].delta.content is not None:
    #                 content = chunk.choices[0].delta.content
    #                 llm_response += content
    #                 yield content
    #
    #         # Adding llm response to GraphDatabase
    #         self.add_message_to_person(face_id, "llm", llm_response)
    #
    #     except Exception as e:
    #         print(f"Error sending to llama.cpp server: {e}")
    #         yield "Error communicating with llama.cpp server"
    #
